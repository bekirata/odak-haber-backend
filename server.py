from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta, timezone
import feedparser
import httpx
import asyncio
from contextlib import asynccontextmanager
from email.utils import parsedate_to_datetime

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# LLM Integration
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

# Admin credentials (change in production!)
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'odak2024admin')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== BACKGROUND TASK ====================

async def refresh_news_cache():
    """Background task to refresh news cache every 5 minutes"""
    while True:
        try:
            logger.info("Starting news cache refresh...")
            
            # Get all active RSS sources from database
            sources_cursor = db.rss_sources.find({"active": True})
            sources = await sources_cursor.to_list(500)
            
            if not sources:
                # Use default sources if none in DB
                sources = []
                for category, source_list in DEFAULT_RSS_SOURCES.items():
                    for source in source_list:
                        sources.append({
                            "name": source["name"],
                            "url": source["url"],
                            "category": category,
                            "active": True
                        })
            
            all_news = []
            for source in sources:
                try:
                    news_items = await fetch_rss_feed(source["url"], source.get("category", "GÜNDEM"))
                    all_news.extend(news_items)
                except Exception as e:
                    logger.error(f"Error fetching {source['url']}: {e}")
            
            if all_news:
                # Sort by parsed ISO date (proper date sorting)
                all_news.sort(key=lambda x: x.get('pub_date_iso', '2000-01-01'), reverse=True)
                
                # Remove duplicates by link
                seen = set()
                unique_news = []
                for item in all_news:
                    if item['link'] not in seen:
                        seen.add(item['link'])
                        unique_news.append(item)
                
                # Update cache in database
                await db.news_cache.delete_many({})
                if unique_news:
                    await db.news_cache.insert_many(unique_news)
                
                # Update last refresh time
                await db.settings.update_one(
                    {"key": "last_cache_refresh"},
                    {"$set": {"value": datetime.utcnow().isoformat(), "count": len(unique_news)}},
                    upsert=True
                )
                
                logger.info(f"News cache refreshed with {len(unique_news)} items")
                
                # Send push notifications for breaking news
                await send_breaking_news_notifications()
            
        except Exception as e:
            logger.error(f"Error in cache refresh: {e}")
        
        # Wait 5 minutes before next refresh
        await asyncio.sleep(300)

async def generate_global_summaries():
    """Background task to generate AI summaries of top 15 news every 30 minutes"""
    while True:
        try:
            if not EMERGENT_LLM_KEY:
                logger.warning("No EMERGENT_LLM_KEY set, skipping summary generation")
                await asyncio.sleep(10800)
                continue

            news_items = await db.news_cache.find().sort("pub_date_iso", -1).limit(15).to_list(15)
            if not news_items:
                await asyncio.sleep(10800)
                continue

            from openai import AsyncOpenAI
            import json as json_module

            news_text = ""
            for i, item in enumerate(news_items, 1):
                title = item.get('title', '')
                desc = (item.get('description', '') or '')[:150]
                desc = desc.replace('<', '').replace('>', '')
                source = item.get('source', '')
                news_text += f"{i}. [{source}] {title}\n{desc}\n\n"

            ai_client = AsyncOpenAI(
                api_key=EMERGENT_LLM_KEY,
                base_url="https://integrations.emergentagent.com/llm"
            )

            completion = await ai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": """Sen Türkçe haber özetleme uzmanısın. Verilen haberleri JSON formatında özetle.
Her haber için kısa bir özet yaz (1-2 cümle). Sonunda günün genel değerlendirmesini ekle.
SADECE geçerli JSON döndür, başka bir şey yazma.
Format: {"items": [{"index": 1, "title": "başlık", "summary": "özet"}], "overall": "genel değerlendirme"}"""},
                    {"role": "user", "content": f"Bu 15 haberi JSON formatında özetle:\n\n{news_text}"}
                ]
            )

            response = completion.choices[0].message.content

            # Parse the AI response
            summary_items = []
            overall = ""
            try:
                clean = response.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                    clean = clean.rsplit("```", 1)[0]
                parsed = json_module.loads(clean)
                overall = parsed.get("overall", "")
                for idx, ai_item in enumerate(parsed.get("items", [])):
                    if idx < len(news_items):
                        src = news_items[idx]
                        src.pop('_id', None)
                        summary_items.append({
                            "title": src.get("title", ""),
                            "summary": ai_item.get("summary", ""),
                            "source": src.get("source", ""),
                            "category": src.get("category", ""),
                            "image_url": src.get("image_url"),
                            "link": src.get("link", ""),
                            "pub_date": src.get("pub_date", ""),
                        })
            except Exception as parse_err:
                logger.error(f"Error parsing AI summary JSON: {parse_err}")
                overall = response
                for idx, src in enumerate(news_items[:15]):
                    src.pop('_id', None)
                    summary_items.append({
                        "title": src.get("title", ""),
                        "summary": "",
                        "source": src.get("source", ""),
                        "category": src.get("category", ""),
                        "image_url": src.get("image_url"),
                        "link": src.get("link", ""),
                        "pub_date": src.get("pub_date", ""),
                    })

            await db.global_summaries.delete_many({})
            await db.global_summaries.insert_one({
                "id": str(uuid.uuid4()),
                "items": summary_items,
                "overall": overall,
                "news_count": len(summary_items),
                "created_at": datetime.utcnow().isoformat(),
            })
            logger.info(f"Global summaries generated: {len(summary_items)} items")

        except Exception as e:
            logger.error(f"Error generating global summaries: {e}")

        await asyncio.sleep(10800)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ODAK Haber API...")
    
    # Initialize default sources if not exist
    await initialize_default_sources()
    
    # Start background tasks
    news_task = asyncio.create_task(refresh_news_cache())
    summary_task = asyncio.create_task(generate_global_summaries())
    notif_task = asyncio.create_task(notification_scheduler())
    
    yield
    
    # Shutdown
    news_task.cancel()
    summary_task.cancel()
    notif_task.cancel()
    client.close()
    logger.info("ODAK Haber API shutdown complete")

# Create the main app with lifespan
app = FastAPI(lifespan=lifespan)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# ==================== MODELS ====================

class RSSSource(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    url: str
    category: str
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class RSSSourceCreate(BaseModel):
    name: str
    url: str
    category: str
    active: bool = True

class RSSSourceUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    category: Optional[str] = None
    active: Optional[bool] = None

class AdSettings(BaseModel):
    banner_ad_id: str = ""
    interstitial_ad_id: str = ""
    rewarded_ad_id: str = ""
    banner_enabled: bool = False
    interstitial_enabled: bool = False
    rewarded_enabled: bool = False
    interstitial_frequency: int = 5  # Show after every X articles

class AdminLogin(BaseModel):
    password: str

class UserPreferences(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: str
    selected_sources: List[str] = []
    favorites: List[str] = []
    read_later: List[str] = []
    read_news: List[str] = []
    dark_mode: bool = False
    font_size: int = 16
    notifications_enabled: bool = False
    notification_interval: int = 2
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class UserPreferencesUpdate(BaseModel):
    selected_sources: Optional[List[str]] = None
    favorites: Optional[List[str]] = None
    read_later: Optional[List[str]] = None
    read_news: Optional[List[str]] = None
    dark_mode: Optional[bool] = None
    font_size: Optional[int] = None
    notifications_enabled: Optional[bool] = None
    notification_interval: Optional[int] = None

class PushTokenRegister(BaseModel):
    device_id: str
    push_token: str
    platform: str = "unknown"

class NotificationSettings(BaseModel):
    device_id: str
    enabled: bool = True
    interval_hours: int = 2  # Notification check interval in hours
    categories: List[str] = []  # Empty means all categories

class SummaryRequest(BaseModel):
    news_items: List[Dict[str, Any]]

class SummaryResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    summary: str
    news_count: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ContactMessage(BaseModel):
    device_id: str
    name: str
    email: str = ""
    message: str

class ContactReply(BaseModel):
    reply: str

# ==================== DEFAULT RSS SOURCES ====================

DEFAULT_RSS_SOURCES = {
    'GÜNDEM': [
        {'name': 'HÜRRİYET', 'url': 'https://www.hurriyet.com.tr/rss/gundem'},
        {'name': 'SABAH', 'url': 'https://www.sabah.com.tr/rss/gundem.xml'},
        {'name': 'CNN TÜRK', 'url': 'https://www.cnnturk.com/feed/rss/all/news'},
        {'name': 'NTV', 'url': 'https://www.ntv.com.tr/son-dakika.rss'},
    ],
    'SPOR': [
        {'name': 'NTV SPOR', 'url': 'https://www.ntvspor.net/rss'},
        {'name': 'FOTOMAÇ', 'url': 'https://www.fotomac.com.tr/rss/futbol.xml'},
        {'name': 'FANATİK', 'url': 'https://www.fanatik.com.tr/rss/futbol.xml'},
        {'name': 'SPORX', 'url': 'https://www.sporx.com/rss/'},
    ],
    'EKONOMİ': [
        {'name': 'BLOOMBERG HT', 'url': 'https://www.bloomberght.com/rss'},
        {'name': 'BİGPARA', 'url': 'https://bigpara.hurriyet.com.tr/rss/'},
        {'name': 'NTV EKONOMİ', 'url': 'https://www.ntv.com.tr/ekonomi.rss'},
    ],
    'TEKNOLOJİ': [
        {'name': 'WEBTEKNO', 'url': 'https://www.webtekno.com/rss.xml'},
        {'name': 'SHIFTDELETE', 'url': 'https://shiftdelete.net/feed'},
        {'name': 'LOG', 'url': 'https://www.log.com.tr/feed/'},
        {'name': 'DONANIMHaber', 'url': 'https://www.donanimhaber.com/rss/'},
    ],
    'DÜNYA': [
        {'name': 'NTV DÜNYA', 'url': 'https://www.ntv.com.tr/dunya.rss'},
        {'name': 'CNN DÜNYA', 'url': 'https://www.cnnturk.com/feed/rss/dunya/news'},
        {'name': 'HÜRRİYET DÜNYA', 'url': 'https://www.hurriyet.com.tr/rss/dunya'},
    ],
    'MAGAZİN': [
        {'name': 'HÜRRİYET MAGAZİN', 'url': 'https://www.hurriyet.com.tr/rss/magazin'},
        {'name': 'SABAH MAGAZİN', 'url': 'https://www.sabah.com.tr/rss/magazin.xml'},
    ],
}

# ==================== HELPER FUNCTIONS ====================

def parse_rss_date(date_str: str) -> str:
    """Parse RSS date string to ISO format for proper sorting"""
    if not date_str:
        return "2000-01-01T00:00:00"
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        try:
            from dateutil import parser as dateutil_parser
            dt = dateutil_parser.parse(date_str)
            return dt.isoformat()
        except Exception:
            return "2000-01-01T00:00:00"

async def initialize_default_sources():
    """Initialize default RSS sources in database if not exist"""
    count = await db.rss_sources.count_documents({})
    if count == 0:
        logger.info("Initializing default RSS sources...")
        sources_to_insert = []
        for category, source_list in DEFAULT_RSS_SOURCES.items():
            for source in source_list:
                sources_to_insert.append({
                    "id": str(uuid.uuid4()),
                    "name": source["name"],
                    "url": source["url"],
                    "category": category,
                    "active": True,
                    "created_at": datetime.utcnow()
                })
        if sources_to_insert:
            await db.rss_sources.insert_many(sources_to_insert)
            logger.info(f"Inserted {len(sources_to_insert)} default sources")
    
    # Initialize default ad settings
    ad_settings = await db.settings.find_one({"key": "ad_settings"})
    if not ad_settings:
        await db.settings.insert_one({
            "key": "ad_settings",
            "value": AdSettings().dict()
        })

def get_source_name_from_url(url: str, sources: List[dict] = None) -> str:
    if sources:
        for source in sources:
            if source.get('url') == url:
                return source.get('name', 'KAYNAK')
    for category, source_list in DEFAULT_RSS_SOURCES.items():
        for source in source_list:
            if source['url'] == url:
                return source['name']
    return 'KAYNAK'

def get_category_from_url(url: str, sources: List[dict] = None) -> str:
    if sources:
        for source in sources:
            if source.get('url') == url:
                return source.get('category', 'GÜNDEM')
    for category, source_list in DEFAULT_RSS_SOURCES.items():
        for source in source_list:
            if source['url'] == url:
                return category
    return 'GÜNDEM'

async def fetch_rss_feed(url: str, category: str = None) -> List[Dict[str, Any]]:
    """Fetch and parse RSS feed"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            response = await http_client.get(url, follow_redirects=True, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: {response.status_code}")
                return []
            
            feed = feedparser.parse(response.text)
            items = []
            
            # Get source info
            sources = await db.rss_sources.find({"url": url}).to_list(1)
            source_name = sources[0]["name"] if sources else get_source_name_from_url(url)
            source_category = category or (sources[0]["category"] if sources else get_category_from_url(url))
            
            for entry in feed.entries[:20]:
                # Extract image from multiple sources
                image_url = None
                
                # Try media:content
                if hasattr(entry, 'media_content') and entry.media_content:
                    for media in entry.media_content:
                        if media.get('url'):
                            image_url = media.get('url')
                            break
                
                # Try enclosure
                if not image_url and hasattr(entry, 'enclosures') and entry.enclosures:
                    for enc in entry.enclosures:
                        if enc.get('type', '').startswith('image') or enc.get('href', '').endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            image_url = enc.get('href') or enc.get('url')
                            break
                
                # Try media:thumbnail
                if not image_url and hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
                    image_url = entry.media_thumbnail[0].get('url')
                
                # Try to extract from content/description
                if not image_url:
                    content = ''
                    if hasattr(entry, 'content') and entry.content:
                        content = entry.content[0].get('value', '')
                    elif hasattr(entry, 'summary'):
                        content = entry.summary
                    elif hasattr(entry, 'description'):
                        content = entry.description
                    
                    if content:
                        import re
                        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
                        if img_match:
                            image_url = img_match.group(1)
                
                items.append({
                    'id': str(uuid.uuid4()),
                    'title': entry.get('title', ''),
                    'description': entry.get('summary', entry.get('description', '')),
                    'content': entry.get('content', [{}])[0].get('value', '') if hasattr(entry, 'content') and entry.content else entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'pub_date': entry.get('published', entry.get('updated', '')),
                    'pub_date_iso': parse_rss_date(entry.get('published', entry.get('updated', ''))),
                    'image_url': image_url,
                    'source': source_name,
                    'category': source_category,
                    'feed_url': url,
                    'cached_at': datetime.utcnow().isoformat()
                })
            
            return items
    except Exception as e:
        logger.error(f"Error fetching RSS {url}: {e}")
        return []

# ==================== ADMIN AUTH ====================

async def verify_admin(x_admin_token: str = Header(None)):
    """Verify admin token"""
    if not x_admin_token or x_admin_token != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# ==================== API ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "ODAK Haber API", "version": "2.0.0"}

# ==================== NEWS ROUTES (CACHED) ====================

@api_router.get("/sources")
async def get_sources():
    """Get all RSS sources grouped by category"""
    sources = await db.rss_sources.find({"active": True}).to_list(500)
    
    if not sources:
        return DEFAULT_RSS_SOURCES
    
    # Group by category
    grouped = {}
    for source in sources:
        cat = source.get('category', 'GÜNDEM')
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append({
            'name': source['name'],
            'url': source['url'],
            'logo_url': source.get('logo_url', ''),
        })
    
    return grouped

@api_router.get("/news")
async def get_cached_news(category: str = None, limit: int = 100):
    """Get news from cache"""
    query = {}
    if category and category != "TÜMÜ":
        query["category"] = category
    
    news = await db.news_cache.find(query).sort("pub_date_iso", -1).limit(limit).to_list(limit)
    
    # If cache is empty, return empty list (background task will populate it)
    if not news:
        return {"news": [], "count": 0, "cached": True, "message": "Cache is being populated..."}
    
    # Remove MongoDB _id
    for item in news:
        item.pop('_id', None)
    
    return {"news": news, "count": len(news), "cached": True}

@api_router.post("/news/fetch")
async def fetch_news(feed_urls: List[str]):
    """Fetch news - returns from cache if available, otherwise fetches fresh"""
    # First try to get from cache
    if feed_urls:
        cached_news = await db.news_cache.find({"feed_url": {"$in": feed_urls}}).sort("pub_date", -1).limit(100).to_list(100)
        
        if cached_news:
            for item in cached_news:
                item.pop('_id', None)
            return {"news": cached_news, "count": len(cached_news), "cached": True}
    
    # If no cache, fetch fresh (fallback)
    all_news = []
    tasks = [fetch_rss_feed(url) for url in feed_urls[:15]]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, list):
            all_news.extend(result)
    
    all_news.sort(key=lambda x: x.get('pub_date', ''), reverse=True)
    
    seen = set()
    unique_news = []
    for item in all_news:
        if item['link'] not in seen:
            seen.add(item['link'])
            unique_news.append(item)
    
    return {"news": unique_news[:100], "count": len(unique_news), "cached": False}

@api_router.get("/news/refresh")
async def trigger_refresh():
    """Manually trigger cache refresh (admin only in production)"""
    # This is a simplified version - in production, protect with admin auth
    return {"message": "Cache refresh will happen in background", "status": "triggered"}

@api_router.get("/cache/status")
async def get_cache_status():
    """Get cache status"""
    last_refresh = await db.settings.find_one({"key": "last_cache_refresh"})
    news_count = await db.news_cache.count_documents({})
    
    return {
        "news_count": news_count,
        "last_refresh": last_refresh.get("value") if last_refresh else None,
        "status": "healthy" if news_count > 0 else "populating"
    }

# ==================== ADMIN ROUTES ====================

@api_router.post("/admin/login")
async def admin_login(login: AdminLogin):
    """Admin login - returns token if password correct"""
    if login.password == ADMIN_PASSWORD:
        return {"success": True, "token": ADMIN_PASSWORD}
    raise HTTPException(status_code=401, detail="Invalid password")

@api_router.get("/admin/sources")
async def admin_get_sources(authorized: bool = Depends(verify_admin)):
    """Get all RSS sources (admin)"""
    sources = await db.rss_sources.find().to_list(100)
    for source in sources:
        source['_id'] = str(source['_id'])
    return {"sources": sources}

@api_router.post("/admin/sources")
async def admin_add_source(source: RSSSourceCreate, authorized: bool = Depends(verify_admin)):
    """Add new RSS source"""
    # Check if URL already exists
    existing = await db.rss_sources.find_one({"url": source.url})
    if existing:
        raise HTTPException(status_code=400, detail="Source URL already exists")
    
    new_source = RSSSource(**source.dict())
    await db.rss_sources.insert_one(new_source.dict())
    
    return {"success": True, "source": new_source.dict()}

@api_router.put("/admin/sources/{source_id}")
async def admin_update_source(source_id: str, update: RSSSourceUpdate, authorized: bool = Depends(verify_admin)):
    """Update RSS source"""
    update_data = {k: v for k, v in update.dict().items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")
    
    result = await db.rss_sources.update_one(
        {"id": source_id},
        {"$set": update_data}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Source not found")
    
    return {"success": True}

@api_router.delete("/admin/sources/{source_id}")
async def admin_delete_source(source_id: str, authorized: bool = Depends(verify_admin)):
    """Delete RSS source"""
    result = await db.rss_sources.delete_one({"id": source_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Source not found")
    
    return {"success": True}

@api_router.post("/admin/seed-sources")
async def seed_sources(request: Request, authorized: bool = Depends(verify_admin)):
    """Bulk seed RSS sources (admin only) - replaces all existing sources"""
    try:
        sources = await request.json()
        if not isinstance(sources, list):
            raise HTTPException(status_code=400, detail="Expected a JSON array of sources")
        
        await db.rss_sources.delete_many({})
        
        docs = []
        for s in sources:
            docs.append({
                "id": str(uuid.uuid4()),
                "name": s.get("name", ""),
                "url": s.get("url", ""),
                "category": s.get("category", "GÜNDEM"),
                "logo_url": s.get("logo_url", ""),
                "active": True,
                "created_at": datetime.utcnow()
            })
        
        if docs:
            await db.rss_sources.insert_many(docs)
        
        return {"success": True, "count": len(docs)}
    except Exception as e:
        logger.error(f"Error seeding sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin/ads")
async def admin_get_ad_settings(authorized: bool = Depends(verify_admin)):
    """Get AdMob settings"""
    settings = await db.settings.find_one({"key": "ad_settings"})
    if settings:
        return settings.get("value", AdSettings().dict())
    return AdSettings().dict()

@api_router.post("/admin/ads")
async def admin_update_ad_settings(ads: AdSettings, authorized: bool = Depends(verify_admin)):
    """Update AdMob settings"""
    await db.settings.update_one(
        {"key": "ad_settings"},
        {"$set": {"value": ads.dict()}},
        upsert=True
    )
    return {"success": True, "settings": ads.dict()}

@api_router.get("/ads/settings")
async def get_public_ad_settings():
    """Get public ad settings for app"""
    settings = await db.settings.find_one({"key": "ad_settings"})
    if settings:
        return settings.get("value", AdSettings().dict())
    return AdSettings().dict()

@api_router.get("/admin/stats")
async def admin_get_stats(authorized: bool = Depends(verify_admin)):
    """Get admin statistics"""
    news_count = await db.news_cache.count_documents({})
    sources_count = await db.rss_sources.count_documents({})
    active_sources = await db.rss_sources.count_documents({"active": True})
    users_count = await db.preferences.count_documents({})
    messages_count = await db.contact_messages.count_documents({})
    unread_messages = await db.contact_messages.count_documents({"read": False})
    
    last_refresh = await db.settings.find_one({"key": "last_cache_refresh"})
    
    return {
        "news_count": news_count,
        "sources_count": sources_count,
        "active_sources": active_sources,
        "users_count": users_count,
        "messages_count": messages_count,
        "unread_messages": unread_messages,
        "last_cache_refresh": last_refresh.get("value") if last_refresh else None
    }

# ==================== USER PREFERENCES ====================

@api_router.get("/preferences/{device_id}")
async def get_preferences(device_id: str):
    """Get user preferences by device ID"""
    prefs = await db.preferences.find_one({"device_id": device_id})
    if prefs:
        prefs['_id'] = str(prefs['_id'])
        return prefs
    return None

@api_router.post("/preferences/{device_id}")
async def create_or_update_preferences(device_id: str, update: UserPreferencesUpdate):
    """Create or update user preferences"""
    existing = await db.preferences.find_one({"device_id": device_id})
    
    update_data = {k: v for k, v in update.dict().items() if v is not None}
    update_data['updated_at'] = datetime.utcnow()
    
    if existing:
        await db.preferences.update_one(
            {"device_id": device_id},
            {"$set": update_data}
        )
    else:
        prefs = UserPreferences(device_id=device_id, **update_data)
        await db.preferences.insert_one(prefs.dict())
    
    result = await db.preferences.find_one({"device_id": device_id})
    result['_id'] = str(result['_id'])
    return result

@api_router.post("/preferences/{device_id}/favorites/add")
async def add_to_favorites(device_id: str, link: str):
    await db.preferences.update_one(
        {"device_id": device_id},
        {"$addToSet": {"favorites": link}, "$set": {"updated_at": datetime.utcnow()}},
        upsert=True
    )
    return {"success": True}

@api_router.post("/preferences/{device_id}/favorites/remove")
async def remove_from_favorites(device_id: str, link: str):
    await db.preferences.update_one(
        {"device_id": device_id},
        {"$pull": {"favorites": link}, "$set": {"updated_at": datetime.utcnow()}}
    )
    return {"success": True}

@api_router.post("/preferences/{device_id}/readlater/add")
async def add_to_read_later(device_id: str, link: str):
    await db.preferences.update_one(
        {"device_id": device_id},
        {"$addToSet": {"read_later": link}, "$set": {"updated_at": datetime.utcnow()}},
        upsert=True
    )
    return {"success": True}

@api_router.post("/preferences/{device_id}/readlater/remove")
async def remove_from_read_later(device_id: str, link: str):
    await db.preferences.update_one(
        {"device_id": device_id},
        {"$pull": {"read_later": link}, "$set": {"updated_at": datetime.utcnow()}}
    )
    return {"success": True}

# ==================== AI SUMMARY ====================

@api_router.post("/summary/generate")
async def generate_summary(request: SummaryRequest):
    """Generate AI summary of news items"""
    if not EMERGENT_LLM_KEY:
        return {"error": "AI servisi yapılandırılmamış", "summary": None}
    
    if not request.news_items:
        return {"error": "Özetlenecek haber yok", "summary": None}
    
    try:
        from openai import AsyncOpenAI
        
        news_text = ""
        for i, item in enumerate(request.news_items[:10], 1):
            news_text += f"{i}. {item.get('title', '')}\n"
            desc = item.get('description', '')[:200] if item.get('description') else ''
            if desc:
                news_text += f"   {desc}\n"
            news_text += f"   Kaynak: {item.get('source', 'Bilinmiyor')}\n\n"
        
        ai_client = AsyncOpenAI(
            api_key=EMERGENT_LLM_KEY,
            base_url="https://integrations.emergentagent.com/llm"
        )
        
        completion = await ai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": """Sen Türkçe haber özetleme uzmanısın. 
            Sana verilen haberleri okuyup, günün en önemli gelişmelerini kısa ve öz şekilde özetleyeceksin.
            Her haberi bir cümle ile özetle ve sonunda genel bir değerlendirme yap.
            Türkçe yaz ve resmi bir dil kullan."""},
                {"role": "user", "content": f"Aşağıdaki güncel haberleri özetle:\n\n{news_text}\n\nLütfen her haberi kısaca özetle ve sonunda günün genel değerlendirmesini yap."}
            ]
        )
        
        response = completion.choices[0].message.content
        
        summary_obj = SummaryResponse(summary=response, news_count=len(request.news_items[:10]))
        await db.summaries.insert_one(summary_obj.dict())
        
        return {
            "summary": response,
            "news_count": len(request.news_items[:10]),
            "created_at": summary_obj.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return {"error": f"Özet oluşturulurken hata: {str(e)}", "summary": None}

@api_router.get("/summary/latest")
async def get_latest_summary():
    summary = await db.summaries.find_one(sort=[("created_at", -1)])
    if summary:
        summary['_id'] = str(summary['_id'])
        return summary
    return None

@api_router.get("/summary/global")
async def get_global_summary():
    """Get the pre-generated global summary of top 15 news"""
    doc = await db.global_summaries.find_one(sort=[("created_at", -1)])
    if doc:
        doc.pop('_id', None)
        return doc
    return {"items": [], "overall": "", "news_count": 0, "created_at": None}

@api_router.get("/download/backend")
async def download_backend_zip():
    """Download backend files as zip for Railway deployment"""
    import os
    zip_path = os.path.join(os.path.dirname(__file__), "static", "odak-railway.zip")
    if os.path.exists(zip_path):
        return FileResponse(zip_path, filename="odak-railway.zip", media_type="application/zip")
    raise HTTPException(status_code=404, detail="File not found")

@api_router.get("/download-page")
async def download_page():
    """Download page for backend zip"""
    import os
    html_path = os.path.join(os.path.dirname(__file__), "static", "download.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="File not found")

# ==================== CONTACT US (Bize Ulaşın) ====================

@api_router.post("/contact/send")
async def send_contact_message(msg: ContactMessage):
    """User sends a contact message"""
    message_doc = {
        "id": str(uuid.uuid4()),
        "device_id": msg.device_id,
        "name": msg.name,
        "email": msg.email,
        "message": msg.message,
        "admin_reply": None,
        "replied_at": None,
        "created_at": datetime.utcnow().isoformat(),
        "read": False,
    }
    await db.contact_messages.insert_one(message_doc)
    message_doc.pop('_id', None)
    return {"success": True, "message": message_doc}

@api_router.get("/contact/messages/{device_id}")
async def get_user_messages(device_id: str):
    """Get all messages for a specific user/device"""
    messages = await db.contact_messages.find(
        {"device_id": device_id}
    ).sort("created_at", -1).to_list(50)
    for m in messages:
        m.pop('_id', None)
    return {"messages": messages}

@api_router.get("/admin/messages")
async def admin_get_messages(authorized: bool = Depends(verify_admin)):
    """Get all contact messages (admin)"""
    messages = await db.contact_messages.find().sort("created_at", -1).to_list(100)
    for m in messages:
        m.pop('_id', None)
    return {"messages": messages}

@api_router.post("/admin/messages/{message_id}/reply")
async def admin_reply_message(message_id: str, reply: ContactReply, authorized: bool = Depends(verify_admin)):
    """Admin replies to a contact message"""
    result = await db.contact_messages.update_one(
        {"id": message_id},
        {"$set": {
            "admin_reply": reply.reply,
            "replied_at": datetime.utcnow().isoformat(),
            "read": True,
        }}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"success": True}

@api_router.post("/admin/messages/{message_id}/read")
async def admin_mark_read(message_id: str, authorized: bool = Depends(verify_admin)):
    """Mark a message as read"""
    await db.contact_messages.update_one(
        {"id": message_id},
        {"$set": {"read": True}}
    )
    return {"success": True}

# ==================== TEXT TO SPEECH (Sesli Okuma) ====================

@api_router.post("/tts")
async def text_to_speech(request: Request):
    """Convert news text to speech using OpenAI TTS HD - news anchor quality"""
    try:
        body = await request.json()
        text = body.get("text", "")[:4000]
        
        if not text:
            raise HTTPException(status_code=400, detail="Text required")
        
        if not EMERGENT_LLM_KEY:
            raise HTTPException(status_code=500, detail="TTS key not configured")
        
        from openai import AsyncOpenAI
        import base64 as b64
        
        ai_client = AsyncOpenAI(
            api_key=EMERGENT_LLM_KEY,
            base_url="https://integrations.emergentagent.com/llm"
        )
        
        speech_response = await ai_client.audio.speech.create(
            model="tts-1-hd",
            input=text,
            voice="onyx",
            speed=0.95,
            response_format="mp3"
        )
        
        audio_bytes = speech_response.content
        audio_base64 = b64.b64encode(audio_bytes).decode('utf-8')
        
        return {"audio": audio_base64, "format": "mp3"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PUSH NOTIFICATIONS ====================

@api_router.post("/notifications/register")
async def register_push_token(data: PushTokenRegister):
    """Register or update a device's push token"""
    try:
        await db.push_tokens.update_one(
            {"device_id": data.device_id},
            {"$set": {
                "push_token": data.push_token,
                "platform": data.platform,
                "updated_at": datetime.utcnow(),
                "active": True
            }},
            upsert=True
        )
        logger.info(f"Push token registered for device: {data.device_id}")
        return {"success": True, "message": "Token registered successfully"}
    except Exception as e:
        logger.error(f"Error registering push token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/notifications/unregister")
async def unregister_push_token(device_id: str):
    """Unregister a device's push token (disable notifications)"""
    try:
        await db.push_tokens.update_one(
            {"device_id": device_id},
            {"$set": {"active": False, "updated_at": datetime.utcnow()}}
        )
        return {"success": True, "message": "Token unregistered"}
    except Exception as e:
        logger.error(f"Error unregistering push token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/notifications/settings")
async def update_notification_settings(settings: NotificationSettings):
    """Update notification settings for a device"""
    try:
        await db.notification_settings.update_one(
            {"device_id": settings.device_id},
            {"$set": {
                "enabled": settings.enabled,
                "interval_hours": settings.interval_hours,
                "categories": settings.categories,
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )
        return {"success": True}
    except Exception as e:
        logger.error(f"Error updating notification settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/notifications/settings/{device_id}")
async def get_notification_settings(device_id: str):
    """Get notification settings for a device"""
    settings = await db.notification_settings.find_one({"device_id": device_id})
    if settings:
        settings.pop('_id', None)
        return settings
    return {"device_id": device_id, "enabled": False, "interval_hours": 2, "categories": []}

@api_router.get("/notifications/latest/{device_id}")
async def get_latest_notification(device_id: str):
    """Get the latest unseen notification for a device"""
    # Get the last notification sent to this device
    notification = await db.notification_history.find_one(
        {"device_id": device_id},
        sort=[("sent_at", -1)]
    )
    
    if notification:
        notification.pop('_id', None)
        return notification
    
    # If no notification history, get the latest breaking news
    latest_news = await db.news_cache.find_one(
        sort=[("pub_date", -1)]
    )
    if latest_news:
        latest_news.pop('_id', None)
        return {"news": latest_news, "type": "breaking"}
    
    return {"news": None}

@api_router.get("/notifications/history/{device_id}")
async def get_notification_history(device_id: str, limit: int = 20):
    """Get notification history for a device"""
    notifications = await db.notification_history.find(
        {"device_id": device_id}
    ).sort("sent_at", -1).limit(limit).to_list(limit)
    
    for n in notifications:
        n.pop('_id', None)
    
    return {"notifications": notifications, "count": len(notifications)}

@api_router.get("/notifications/breaking")
async def get_breaking_news_for_notification(device_id: str = None):
    """Get the most recent breaking news items - filtered by user's selected sources"""
    query = {}
    
    # If device_id provided, filter by user's selected sources
    if device_id:
        user_prefs = await db.preferences.find_one({"device_id": device_id})
        if user_prefs and user_prefs.get("selected_sources"):
            query["feed_url"] = {"$in": user_prefs["selected_sources"]}
    
    news = await db.news_cache.find(query).sort("pub_date_iso", -1).limit(5).to_list(5)
    
    for item in news:
        item.pop('_id', None)
    
    return {"news": news, "count": len(news)}

@api_router.post("/notifications/send-test")
async def send_test_notification(device_id: str):
    """Send a test notification to a specific device"""
    token_doc = await db.push_tokens.find_one({"device_id": device_id, "active": True})
    
    if not token_doc:
        return {"success": False, "error": "No active push token found for this device"}
    
    push_token = token_doc.get("push_token")
    if not push_token:
        return {"success": False, "error": "Push token is empty"}
    
    # Get latest news for notification content
    latest_news = await db.news_cache.find_one(sort=[("pub_date", -1)])
    
    title = "ODAK Haber - Test Bildirimi"
    body = "Bildirim sistemi çalışıyor!"
    data = {}
    
    if latest_news:
        title = f"📰 {latest_news.get('source', 'ODAK')}"
        body = latest_news.get('title', 'Yeni haber var!')
        data = {
            "newsId": latest_news.get('id', ''),
            "link": latest_news.get('link', ''),
            "type": "breaking_news"
        }
    
    result = await send_expo_push_notification(push_token, title, body, data)
    
    # Save to history
    await db.notification_history.insert_one({
        "id": str(uuid.uuid4()),
        "device_id": device_id,
        "title": title,
        "body": body,
        "data": data,
        "sent_at": datetime.utcnow(),
        "status": "sent" if result.get("success") else "failed",
        "type": "test"
    })
    
    return result

async def send_expo_push_notification(push_token: str, title: str, body: str, data: dict = None):
    """Send push notification via Expo Push API"""
    try:
        message = {
            "to": push_token,
            "sound": "default",
            "title": title,
            "body": body,
            "data": data or {},
            "priority": "high",
            "channelId": "breaking-news"
        }
        
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                "https://exp.host/--/api/v2/push/send",
                json=message,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate",
                    "Content-Type": "application/json"
                }
            )
            
            result = response.json()
            logger.info(f"Push notification sent: {result}")
            
            if response.status_code == 200:
                return {"success": True, "result": result}
            else:
                return {"success": False, "error": result}
    except Exception as e:
        logger.error(f"Error sending push notification: {e}")
        return {"success": False, "error": str(e)}

async def send_breaking_news_notifications():
    """Send push notifications for breaking news to all registered devices"""
    try:
        # Check quiet hours (00:00 - 08:00 Turkish time, UTC+3)
        import pytz
        turkey_tz = pytz.timezone('Europe/Istanbul')
        now_turkey = datetime.now(turkey_tz)
        current_hour = now_turkey.hour
        
        if 0 <= current_hour < 8:
            logger.info(f"Quiet hours active ({current_hour}:00 Turkey time), skipping notifications")
            return

        # Get all active push tokens with their settings
        tokens_cursor = db.push_tokens.find({"active": True})
        tokens = await tokens_cursor.to_list(1000)
        
        if not tokens:
            return
        
        sent_count = 0
        for token_doc in tokens:
            push_token = token_doc.get("push_token")
            device_id = token_doc.get("device_id")
            
            if not push_token:
                continue
            
            # Check device notification settings
            device_settings = await db.notification_settings.find_one({"device_id": device_id})
            if device_settings and not device_settings.get("enabled", True):
                continue
            
            interval_hours = 2
            if device_settings:
                interval_hours = device_settings.get("interval_hours", 2)
            
            # Check interval
            last_device_notif = await db.notification_history.find_one(
                {"device_id": device_id},
                sort=[("sent_at", -1)]
            )
            
            if last_device_notif:
                time_since = datetime.utcnow() - last_device_notif.get("sent_at", datetime.utcnow())
                if time_since.total_seconds() < interval_hours * 3600:
                    continue
            
            # Get user's selected sources
            user_prefs = await db.preferences.find_one({"device_id": device_id})
            selected_sources = user_prefs.get("selected_sources", []) if user_prefs else []
            
            # Find latest news from user's selected sources
            query = {}
            if selected_sources:
                query["feed_url"] = {"$in": selected_sources}
            
            latest = await db.news_cache.find_one(query, sort=[("pub_date_iso", -1)])
            
            if not latest:
                continue
            
            latest.pop('_id', None)
            
            # Skip if we already sent notification for this exact news
            last_sent_news_id = None
            if last_device_notif:
                last_sent_news_id = last_device_notif.get("data", {}).get("newsId")
            if latest.get('id') == last_sent_news_id:
                continue
            
            title = f"📰 {latest.get('source', 'ODAK Haber')}"
            body = latest.get('title', 'Yeni haberler var!')
            data = {
                "newsId": latest.get('id', ''),
                "link": latest.get('link', ''),
                "type": "breaking_news"
            }
            
            result = await send_expo_push_notification(push_token, title, body, data)
            
            # Save to history
            await db.notification_history.insert_one({
                "id": str(uuid.uuid4()),
                "device_id": device_id,
                "title": title,
                "body": body,
                "data": data,
                "sent_at": datetime.utcnow(),
                "status": "sent" if result.get("success") else "failed",
                "type": "breaking_news"
            })
            
            sent_count += 1
        
        if sent_count > 0:
            logger.info(f"Breaking news notifications sent to {sent_count} devices")
        
    except Exception as e:
        logger.error(f"Error sending breaking news notifications: {e}")

async def notification_scheduler():
    """Independent notification scheduler - runs every 5 minutes"""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            await send_breaking_news_notifications()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Notification scheduler error: {e}")
            await asyncio.sleep(60)

# Include the router in the main app
app.include_router(api_router)

# Serve static files (admin panel, privacy policy, terms)
STATIC_DIR = ROOT_DIR / "static"

# Serve admin panel at /api/admin (accessible via ingress)
@app.get("/api/admin-panel")
async def admin_page_api():
    return FileResponse(STATIC_DIR / "admin.html")

@app.get("/api/privacy-page")
async def privacy_page_api():
    return FileResponse(STATIC_DIR / "privacy.html")

@app.get("/api/terms-page")
async def terms_page_api():
    return FileResponse(STATIC_DIR / "terms.html")

# Also serve at root level for direct access
@app.get("/admin")
async def admin_redirect():
    return FileResponse(STATIC_DIR / "admin.html")

@app.get("/privacy")
async def privacy_redirect():
    return FileResponse(STATIC_DIR / "privacy.html")

@app.get("/terms")
async def terms_redirect():
    return FileResponse(STATIC_DIR / "terms.html")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

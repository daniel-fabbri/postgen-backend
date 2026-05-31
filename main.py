from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, PlainTextResponse
from urllib.parse import urlencode
from typing import Optional, List
import os
import json
import subprocess
import tempfile
import requests
from datetime import datetime, timedelta, timezone
from openai import AzureOpenAI
from sqlalchemy.orm import Session
from sqlalchemy import text

from config import (
    BASE_URL, ALLOWED_ORIGINS,
    AZURE_SORA_ENDPOINT, AZURE_SORA_API_KEY,
    INSTAGRAM_APP_ID, INSTAGRAM_APP_SECRET, FRONTEND_URL,
    INSTAGRAM_WEBHOOK_VERIFY_TOKEN,
)
from database import engine, SessionLocal, Base, get_db
from models import (
    UserDB, ChannelDB, PostDB, VideoDB, VideoProjectDB, MediaInsightsDB,
    ReferenceImageDB, AvatarDB, SystemConfigDB,
)
from schemas import (
    Channel, GeneratePostRequest, Post, InsightsOut, DashboardItemOut,
    ChannelDashboardOut, SavedPost, GenerateAvatarRequest, UpdateAvatarRequest,
    TestInstagramRequest, GenerateVideoRequest, SavedVideo, UpdateVideoCaptionRequest,
    VideoProjectOut, CreateVideoProjectRequest, UpdateVideoProjectClipsRequest,
    GenerateProjectClipRequest, AddVideoToProjectRequest, ReferenceImageOut, AvatarInfo,
    UpdatePostRequest, GeneratePostImageRequest,
)
from dependencies import (
    get_current_user, get_or_create_settings, get_azure_client, get_channel_or_404,
)
from services.blob_storage import upload_bytes_to_blob
from services.credits import register_credit_usage
from services.azure_ai import generate_image_bytes, get_reference_context
from routers.auth import router as auth_router, users_router
from routers.admin import router as admin_router
from routers.settings import router as settings_router
from routers.payments import router as payments_router
from routers.credits import router as credits_router


app = FastAPI(title="PostGen API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(admin_router)
app.include_router(settings_router)
app.include_router(payments_router)
app.include_router(credits_router)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        defaults = {"credits_per_real": "1.0", "initial_credits": "0.0"}
        for key, val in defaults.items():
            if not db.query(SystemConfigDB).filter(SystemConfigDB.key == key).first():
                db.add(SystemConfigDB(key=key, value=val))
        db.commit()
    finally:
        db.close()
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE videos ADD COLUMN IF NOT EXISTS caption TEXT DEFAULT ''"))
            conn.execute(text("ALTER TABLE videos ADD COLUMN IF NOT EXISTS is_project_clip BOOLEAN DEFAULT FALSE"))
            conn.execute(text("ALTER TABLE posts ADD COLUMN IF NOT EXISTS prompt TEXT"))
            conn.execute(text("ALTER TABLE video_projects ADD COLUMN IF NOT EXISTS root_video_id VARCHAR(100)"))
            conn.execute(text("ALTER TABLE video_projects ADD COLUMN IF NOT EXISTS clip_urls TEXT DEFAULT '{}'"))
            conn.execute(text("ALTER TABLE video_projects ADD COLUMN IF NOT EXISTS exported_video_id VARCHAR(100)"))
            conn.execute(text("ALTER TABLE channels ADD COLUMN IF NOT EXISTS image_model VARCHAR(20) DEFAULT 'mai'"))
            conn.execute(text("ALTER TABLE posts ADD COLUMN IF NOT EXISTS ig_media_id VARCHAR(100)"))
            conn.execute(text("ALTER TABLE videos ADD COLUMN IF NOT EXISTS ig_media_id VARCHAR(100)"))
            conn.execute(text("ALTER TABLE posts ADD COLUMN IF NOT EXISTS credits_consumed FLOAT DEFAULT 0.0"))
            conn.execute(text("ALTER TABLE videos ADD COLUMN IF NOT EXISTS credits_consumed FLOAT DEFAULT 0.0"))
            conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS credits_balance FLOAT DEFAULT 0.0"))
            conn.execute(text("ALTER TABLE channels ADD COLUMN IF NOT EXISTS auto_reply_enabled BOOLEAN DEFAULT FALSE"))
            conn.execute(text("ALTER TABLE channels ADD COLUMN IF NOT EXISTS auto_reply_prompt TEXT"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS reference_images (
                    id SERIAL PRIMARY KEY,
                    channel_id VARCHAR(50) REFERENCES channels(id) ON DELETE CASCADE,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    blob_url TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS credit_usage (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    channel_id VARCHAR(50) REFERENCES channels(id) ON DELETE CASCADE,
                    resource_type VARCHAR(20) NOT NULL,
                    resource_id VARCHAR(100),
                    operation_type VARCHAR(30) NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    credits_consumed FLOAT DEFAULT 0.0,
                    meta_info TEXT DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS payments (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    mp_payment_id VARCHAR(100) UNIQUE NOT NULL,
                    amount FLOAT NOT NULL,
                    credits_amount FLOAT NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    qr_code TEXT,
                    qr_code_data TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def channel_to_schema(ch: ChannelDB) -> Channel:
    return Channel(
        id=ch.id,
        name=ch.name,
        objective=ch.objective or "",
        text_generation_prompt=ch.text_generation_prompt,
        image_generation_prompt=ch.image_generation_prompt,
        avatar_url=ch.avatar_url,
        suggested_image_url=ch.suggested_image_url,
        created_at=ch.created_at.isoformat() if ch.created_at else None,
        instagram_user_id=ch.instagram_user_id,
        instagram_access_token="***" if ch.instagram_access_token else None,
        image_model=ch.image_model or "mai",
        auto_reply_enabled=ch.auto_reply_enabled or False,
        auto_reply_prompt=ch.auto_reply_prompt,
    )


def _insights_to_schema(ins) -> Optional[InsightsOut]:
    if not ins:
        return None
    return InsightsOut(
        like_count=ins.like_count or 0,
        comments_count=ins.comments_count or 0,
        impressions=ins.impressions,
        reach=ins.reach,
        saved=ins.saved,
        shares=ins.shares,
        video_views=ins.video_views,
        total_interactions=ins.total_interactions or 0,
        engagement_rate=ins.engagement_rate,
        fetched_at=ins.fetched_at.isoformat() if ins.fetched_at else None,
    )


def _insights_ttl(published_at: datetime) -> timedelta:
    now = datetime.now(timezone.utc)
    if published_at:
        pub = published_at.replace(tzinfo=timezone.utc) if not published_at.tzinfo else published_at
        age = now - pub
    else:
        age = timedelta(days=999)
    if age < timedelta(days=1):
        return timedelta(minutes=30)
    elif age < timedelta(days=7):
        return timedelta(hours=2)
    elif age < timedelta(days=30):
        return timedelta(hours=12)
    return timedelta(days=1)


def _insights_stale(ins, published_at: datetime) -> bool:
    if not ins or not ins.fetched_at:
        return True
    ttl = _insights_ttl(published_at)
    now = datetime.now(timezone.utc)
    fetched = ins.fetched_at.replace(tzinfo=timezone.utc) if not ins.fetched_at.tzinfo else ins.fetched_at
    return (now - fetched) > ttl


def _fetch_and_store_insights(
    media_type: str, media_id: str, ig_media_id: str,
    channel_id: str, token: str, db: Session,
):
    result = {}
    ig_media_type = None
    api_base = _ig_api_base(token)
    
    print(f"\n[INSIGHTS DEBUG] Fetching insights for {media_type} {media_id} (IG: {ig_media_id})")
    print(f"[INSIGHTS DEBUG] API Base: {api_base}")

    try:
        url = f"{api_base}/{ig_media_id}"
        params = {"fields": "like_count,comments_count,media_type", "access_token": token}
        print(f"[INSIGHTS DEBUG] Fetching basic data from: {url}")
        print(f"[INSIGHTS DEBUG] Fields: like_count,comments_count,media_type")
        
        resp = requests.get(url, params=params, timeout=15)
        print(f"[INSIGHTS DEBUG] Basic fetch status: {resp.status_code}")
        
        if resp.ok:
            data = resp.json()
            print(f"[INSIGHTS DEBUG] Basic data received: {data}")
            result["like_count"] = data.get("like_count", 0)
            result["comments_count"] = data.get("comments_count", 0)
            ig_media_type = data.get("media_type", "")
        else:
            print(f"[INSIGHTS ERROR] Basic fetch failed {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[INSIGHTS ERROR] Basic fetch exception: {e}")
        import traceback
        traceback.print_exc()

    # Tentar métricas em grupos menores para evitar erros da API
    # Diferentes tipos de mídia suportam diferentes métricas
    # NOTA: A API do Instagram NÃO fornece 'impressions' para posts IMAGE/VIDEO normais
    # Apenas Stories e Reels têm essa métrica disponível
    metrics_to_try = []
    
    if ig_media_type in ("VIDEO", "REELS"):
        # Vídeos e Reels - removido impressions/video_views/plays pois não são suportados
        metrics_to_try = [
            ["reach", "saved"],
            ["shares"]
        ]
    else:
        # IMAGE e CAROUSEL_ALBUM - removido impressions pois não é suportado
        metrics_to_try = [
            ["reach", "saved"],
            ["shares"]
        ]
    
    print(f"[INSIGHTS DEBUG] Media type: {ig_media_type}")
    print(f"[INSIGHTS DEBUG] Will try metric groups: {metrics_to_try}")
    
    # Tentar cada grupo de métricas
    for metric_group in metrics_to_try:
        try:
            ins_url = f"{api_base}/{ig_media_id}/insights"
            ins_params = {"metric": ",".join(metric_group), "period": "lifetime", "access_token": token}
            print(f"[INSIGHTS DEBUG] Trying metrics: {metric_group}")
            
            ins_resp = requests.get(ins_url, params=ins_params, timeout=15)
            print(f"[INSIGHTS DEBUG] Status: {ins_resp.status_code}")
            
            if ins_resp.ok:
                ins_data = ins_resp.json()
                print(f"[INSIGHTS DEBUG] Success! Data: {ins_data}")
                
                for item in ins_data.get("data", []):
                    name = item.get("name", "")
                    val = item.get("value")
                    if val is None:
                        vals = item.get("values", [])
                        val = vals[0].get("value", 0) if vals else 0
                    if val is None:
                        total = item.get("total_value", {})
                        val = total.get("value", 0) if isinstance(total, dict) else 0
                    result[name] = val or 0
                    print(f"[INSIGHTS DEBUG] Metric {name} = {val}")
            else:
                error_text = ins_resp.text
                print(f"[INSIGHTS DEBUG] Group failed {ins_resp.status_code}: {error_text[:200]}")
                # Continue tentando outras métricas
        except Exception as e:
            print(f"[INSIGHTS DEBUG] Exception trying {metric_group}: {e}")
            # Continue tentando outras métricas

    interactions = (result.get("like_count", 0) + result.get("comments_count", 0) + result.get("saved", 0))
    result["total_interactions"] = interactions
    reach = result.get("reach")
    result["engagement_rate"] = round(interactions / reach * 100, 2) if reach else None
    
    print(f"[INSIGHTS DEBUG] Final result: {result}")
    print(f"[INSIGHTS DEBUG] Engagement rate: {result['engagement_rate']}")

    now = datetime.now(timezone.utc)
    ins = db.query(MediaInsightsDB).filter(
        MediaInsightsDB.media_type == media_type,
        MediaInsightsDB.media_id == media_id,
    ).first()

    if ins:
        ins.like_count = result.get("like_count", 0)
        ins.comments_count = result.get("comments_count", 0)
        ins.impressions = result.get("impressions")
        ins.reach = result.get("reach")
        ins.saved = result.get("saved")
        ins.shares = result.get("shares")
        ins.video_views = result.get("video_views")
        ins.total_interactions = result.get("total_interactions", 0)
        ins.engagement_rate = result.get("engagement_rate")
        ins.fetched_at = now
    else:
        ins = MediaInsightsDB(
            media_type=media_type,
            media_id=media_id,
            ig_media_id=ig_media_id,
            channel_id=channel_id,
            like_count=result.get("like_count", 0),
            comments_count=result.get("comments_count", 0),
            impressions=result.get("impressions"),
            reach=result.get("reach"),
            saved=result.get("saved"),
            shares=result.get("shares"),
            video_views=result.get("video_views"),
            total_interactions=result.get("total_interactions", 0),
            engagement_rate=result.get("engagement_rate"),
            fetched_at=now,
        )
        db.add(ins)

    db.commit()
    db.refresh(ins)
    return ins


def post_to_schema(p: PostDB, insights=None) -> SavedPost:
    image_path = p.image_path or ""
    if image_path.startswith("data:"):
        image_path = ""
    return SavedPost(
        id=p.id,
        channel_id=p.channel_id,
        channel_name=p.channel_name,
        text=p.text or "",
        image_path=image_path,
        prompt=getattr(p, "prompt", None),
        ig_media_id=getattr(p, "ig_media_id", None),
        insights=_insights_to_schema(insights),
        credits_consumed=getattr(p, "credits_consumed", 0.0),
        created_at=p.created_at.isoformat() if p.created_at else datetime.now().isoformat(),
        published=p.published or False,
    )


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "PostGen API is running"}


# ---------------------------------------------------------------------------
# Channels endpoints
# ---------------------------------------------------------------------------
@app.get("/api/channels", response_model=List[Channel])
def get_channels(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    channels = db.query(ChannelDB).filter(ChannelDB.user_id == current_user.id).all()
    return [channel_to_schema(ch) for ch in channels]


@app.post("/api/channels", response_model=Channel, status_code=201)
def create_channel(
    data: Channel,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = ChannelDB(
        id=f"ch_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        user_id=current_user.id,
        name=data.name,
        objective=data.objective,
        text_generation_prompt=data.text_generation_prompt,
        image_generation_prompt=data.image_generation_prompt,
        instagram_user_id=data.instagram_user_id,
        instagram_access_token=data.instagram_access_token,
    )
    db.add(ch)
    db.commit()
    db.refresh(ch)

    # Generate avatar if image prompt provided
    if ch.image_generation_prompt:
        try:
            s = get_or_create_settings(current_user, db)
            avatar_prompt = (
                "\n".join(ch.image_generation_prompt.splitlines()[:5])
                + "\n\nFrame: Close-up portrait style, profile picture format."
            )
            image_bytes = generate_image_bytes(avatar_prompt, ch, s, db, width=768, height=768)
            avatar_filename = f"{ch.id}.png"
            avatar_url = upload_bytes_to_blob(image_bytes, f"avatars/{avatar_filename}", "image/png")
            ch.avatar_url = avatar_url
            _register_avatar(avatar_filename, ch.id, db)
            db.commit()
            db.refresh(ch)
        except Exception as e:
            print(f"Error generating avatar: {e}")

    return channel_to_schema(ch)


@app.get("/api/channels/{channel_id}", response_model=Channel)
def get_channel(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return channel_to_schema(get_channel_or_404(channel_id, current_user, db))


@app.put("/api/channels/{channel_id}", response_model=Channel)
def update_channel(
    channel_id: str,
    data: Channel,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)
    ch.name = data.name
    ch.objective = data.objective
    ch.text_generation_prompt = data.text_generation_prompt or None
    ch.image_generation_prompt = data.image_generation_prompt or None
    ch.avatar_url = data.avatar_url
    ch.suggested_image_url = data.suggested_image_url
    ch.instagram_user_id = data.instagram_user_id
    if data.instagram_access_token and data.instagram_access_token != "***":
        ch.instagram_access_token = data.instagram_access_token
    if data.image_model:
        ch.image_model = data.image_model
    if data.auto_reply_enabled is not None:
        ch.auto_reply_enabled = data.auto_reply_enabled
    ch.auto_reply_prompt = data.auto_reply_prompt
    db.commit()
    db.refresh(ch)
    return channel_to_schema(ch)


@app.delete("/api/channels/{channel_id}", status_code=204)
def delete_channel(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)
    db.delete(ch)
    db.commit()


@app.patch("/api/channels/{channel_id}/avatar")
@app.post("/api/channels/{channel_id}/avatar")
def update_channel_avatar(
    channel_id: str,
    data: UpdateAvatarRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)
    ch.avatar_url = data.avatar_url
    filename = data.avatar_url.rstrip("/").split("/")[-1]
    if filename:
        _register_avatar(filename, channel_id, db)
    db.commit()
    db.refresh(ch)
    return {"success": True, "channel": channel_to_schema(ch)}


def _ig_api_base(token: str) -> str:
    if token and token.startswith("IG"):
        return "https://graph.instagram.com/v21.0"
    return "https://graph.facebook.com/v21.0"


@app.get("/api/auth/instagram/authorize")
def instagram_authorize(
    channel_id: str = Query(...),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not INSTAGRAM_APP_ID or not INSTAGRAM_APP_SECRET:
        raise HTTPException(status_code=500, detail="Instagram OAuth não configurado no servidor.")
    get_channel_or_404(channel_id, current_user, db)
    state = jwt.encode(
        {"channel_id": channel_id, "user_id": current_user.id, "exp": datetime.now(timezone.utc) + timedelta(minutes=15)},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )
    params = {
        "client_id": INSTAGRAM_APP_ID,
        "redirect_uri": f"{BASE_URL}/api/auth/instagram/callback",
        "scope": "instagram_business_basic,instagram_business_content_publish,instagram_business_manage_insights",
        "response_type": "code",
        "state": state,
    }
    return {"url": "https://www.instagram.com/oauth/authorize?" + urlencode(params)}


@app.get("/api/auth/instagram/callback")
def instagram_callback(
    code: str = None,
    state: str = None,
    error: str = None,
    db: Session = Depends(get_db),
):
    front = FRONTEND_URL.rstrip("/")
    if error or not code or not state:
        return RedirectResponse(url=f"{front}/channels?ig_error=cancelled")

    try:
        state_data = jwt.decode(state, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        channel_id = state_data["channel_id"]
        user_id = state_data["user_id"]
    except JWTError:
        return RedirectResponse(url=f"{front}/channels?ig_error=invalid_state")

    redirect_uri = f"{BASE_URL}/api/auth/instagram/callback"
    try:
        token_resp = requests.post(
            "https://api.instagram.com/oauth/access_token",
            data={
                "client_id": INSTAGRAM_APP_ID,
                "client_secret": INSTAGRAM_APP_SECRET,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
                "code": code,
            },
            timeout=15,
        )
        token_data = token_resp.json()
        if "access_token" not in token_data:
            err = token_data.get("error_message", "token_exchange_failed")
            return RedirectResponse(url=f"{front}/channels/{channel_id}/edit?ig_error={err}")

        short_token = token_data["access_token"]
        ig_user_id = str(token_data["user_id"])

        ll_resp = requests.get(
            "https://graph.instagram.com/access_token",
            params={
                "grant_type": "ig_exchange_token",
                "client_id": INSTAGRAM_APP_ID,
                "client_secret": INSTAGRAM_APP_SECRET,
                "access_token": short_token,
            },
            timeout=15,
        )
        long_token = ll_resp.json().get("access_token", short_token)

        me_resp = requests.get(
            "https://graph.instagram.com/me",
            params={"fields": "id,username", "access_token": long_token},
            timeout=10,
        )
        username = me_resp.json().get("username", ig_user_id)

        ch = db.query(ChannelDB).filter(
            ChannelDB.id == channel_id,
            ChannelDB.user_id == user_id,
        ).first()
        if not ch:
            return RedirectResponse(url=f"{front}/channels?ig_error=channel_not_found")

        ch.instagram_user_id = ig_user_id
        ch.instagram_access_token = long_token
        db.commit()
        return RedirectResponse(url=f"{front}/channels/{channel_id}/edit?ig_success={username}")

    except requests.RequestException:
        return RedirectResponse(url=f"{front}/channels/{channel_id}/edit?ig_error=network_error")


@app.delete("/api/channels/{channel_id}/instagram")
def instagram_disconnect(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)
    ch.instagram_user_id = None
    ch.instagram_access_token = None
    db.commit()
    return {"success": True}


@app.post("/api/channels/{channel_id}/test-instagram")
def test_instagram_connection(
    channel_id: str,
    data: TestInstagramRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)

    user_id = data.instagram_user_id or ch.instagram_user_id
    token = data.instagram_access_token if (data.instagram_access_token and data.instagram_access_token != "***") else ch.instagram_access_token

    if not user_id or not token:
        raise HTTPException(status_code=400, detail="Preencha o User ID e o Access Token antes de testar.")

    try:
        resp = requests.get(
            f"{_ig_api_base(token)}/{user_id}",
            params={"fields": "id,name,username,followers_count", "access_token": token},
            timeout=10,
        )
        result = resp.json()
        if "error" in result:
            msg = result["error"].get("message", "Erro desconhecido")
            return {"success": False, "error": msg}
        return {
            "success": True,
            "account": {
                "id": result.get("id"),
                "name": result.get("name"),
                "username": result.get("username"),
                "followers_count": result.get("followers_count"),
            },
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Falha ao conectar com o Instagram: {str(e)}")


# ---------------------------------------------------------------------------
# Avatar helpers / endpoints
# ---------------------------------------------------------------------------
def _register_avatar(filename: str, channel_id: str, db: Session):
    existing = db.query(AvatarDB).filter(AvatarDB.filename == filename).first()
    if existing:
        existing.channel_id = channel_id
    else:
        db.add(AvatarDB(filename=filename, channel_id=channel_id))
    db.commit()


@app.get("/api/avatars", response_model=List[AvatarInfo])
def list_avatars(
    channel_id: Optional[str] = None,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if channel_id:
        # Verify channel belongs to user
        get_channel_or_404(channel_id, current_user, db)
        rows = db.query(AvatarDB).filter(AvatarDB.channel_id == channel_id).all()
    else:
        # All avatars for channels owned by this user
        user_channel_ids = [
            ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == current_user.id).all()
        ]
        rows = db.query(AvatarDB).filter(AvatarDB.channel_id.in_(user_channel_ids)).all()

    result = []
    for row in rows:
        blob_url = f"https://postgenstorage.blob.core.windows.net/{AZURE_STORAGE_CONTAINER}/avatars/{row.filename}"
        result.append(AvatarInfo(
            filename=row.filename,
            url=blob_url,
            created_at=row.created_at.isoformat() if row.created_at else None,
        ))
    return sorted(result, key=lambda x: x.created_at or "", reverse=True)


@app.post("/api/avatars/generate")
def generate_avatar(
    data: GenerateAvatarRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = get_or_create_settings(current_user, db)

    ch = None
    channel_prompt = ""
    if data.channel_id:
        ch = db.query(ChannelDB).filter(ChannelDB.id == data.channel_id).first()
        if ch and ch.image_generation_prompt:
            channel_prompt = "\n".join(ch.image_generation_prompt.splitlines()[:5])

    portrait_suffix = "\n\nFrame: Close-up portrait style, profile picture format."
    if channel_prompt and data.prompt:
        full_prompt = f"{channel_prompt}\n\n{data.prompt}{portrait_suffix}"
    else:
        full_prompt = (channel_prompt or data.prompt) + portrait_suffix

    if ch:
        full_prompt += get_reference_context(ch.id, db)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    avatar_filename = f"avatar_{timestamp}.png"

    try:
        image_bytes = generate_image_bytes(full_prompt, ch, s, db, width=768, height=768)
    except HTTPException:
        raise
    except Exception as e:
        if data.prompt and channel_prompt:
            try:
                fallback = data.prompt + portrait_suffix + (get_reference_context(ch.id, db) if ch else "")
                image_bytes = generate_image_bytes(fallback, ch, s, db, width=768, height=768)
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Falha ao gerar avatar: {str(e2)}")
        else:
            raise HTTPException(status_code=500, detail=f"Falha ao gerar avatar: {str(e)}")

    avatar_url = upload_bytes_to_blob(image_bytes, f"avatars/{avatar_filename}", "image/png")
    
    # Track avatar generation credits
    image_model = ch.image_model if ch else "mai"
    register_credit_usage(
        db=db,
        user_id=current_user.id,
        channel_id=data.channel_id,
        resource_type="avatar",
        resource_id=avatar_filename,
        operation_type="image_generation",
        model_name=image_model,
        images_count=1,
        metadata={"prompt_length": len(full_prompt), "size": "768x768"},
    )

    if data.channel_id and ch:
        ch = get_channel_or_404(data.channel_id, current_user, db)
        _register_avatar(avatar_filename, data.channel_id, db)
        ch.avatar_url = avatar_url
        db.commit()

    return {"success": True, "avatar_url": avatar_url, "filename": avatar_filename}


@app.post("/api/avatars/upload")
def upload_avatar(
    file: UploadFile = File(...),
    channel_id: Optional[str] = None,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = file.filename.split(".")[-1] if "." in file.filename else "png"
    avatar_filename = f"avatar_{timestamp}.{ext}"

    data = file.file.read()
    avatar_url = upload_bytes_to_blob(data, f"avatars/{avatar_filename}", file.content_type or "image/png")

    if channel_id:
        ch = get_channel_or_404(channel_id, current_user, db)
        _register_avatar(avatar_filename, channel_id, db)
        ch.avatar_url = avatar_url
        db.commit()

    return {"success": True, "avatar_url": avatar_url, "filename": avatar_filename}


# ---------------------------------------------------------------------------
# Posts endpoints
# ---------------------------------------------------------------------------
@app.get("/api/posts", response_model=List[SavedPost])
def get_posts(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_channel_ids = [
        ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == current_user.id).all()
    ]
    posts = (
        db.query(PostDB)
        .filter(PostDB.channel_id.in_(user_channel_ids))
        .order_by(PostDB.created_at.desc())
        .all()
    )
    post_ids = [p.id for p in posts]
    ins_map = {}
    if post_ids:
        for ins in db.query(MediaInsightsDB).filter(
            MediaInsightsDB.media_type == "post",
            MediaInsightsDB.media_id.in_(post_ids),
        ).all():
            ins_map[ins.media_id] = ins
    return [post_to_schema(p, ins_map.get(p.id)) for p in posts]


@app.get("/api/posts/{post_id}", response_model=SavedPost)
def get_post(
    post_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = _get_post_or_404(post_id, current_user, db)
    return post_to_schema(p)


@app.patch("/api/posts/{post_id}", response_model=SavedPost)
@app.post("/api/posts/{post_id}/save", response_model=SavedPost)
def update_post(
    post_id: str,
    data: UpdatePostRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = _get_post_or_404(post_id, current_user, db)
    if data.text is not None:
        p.text = data.text
    if data.image_path is not None:
        p.image_path = data.image_path
    if data.published is not None:
        p.published = data.published
    db.commit()
    db.refresh(p)
    return post_to_schema(p)


def _get_post_or_404(post_id: str, user: UserDB, db: Session) -> PostDB:
    user_channel_ids = [
        ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == user.id).all()
    ]
    p = db.query(PostDB).filter(
        PostDB.id == post_id,
        PostDB.channel_id.in_(user_channel_ids),
    ).first()
    if not p:
        raise HTTPException(status_code=404, detail="Post não encontrado")
    return p


@app.post("/api/posts/generate", response_model=Post)
def generate_post(
    data: GeneratePostRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(data.channel_id, current_user, db)
    s = get_or_create_settings(current_user, db)
    client = get_azure_client(s)

    # Generate text
    text_prompt = ch.text_generation_prompt or f"""Create an engaging Instagram post for channel "{ch.name}".
Objective: {ch.objective}
Requirements: engaging, authentic, relevant hashtags, 100-200 words.
Return only the post text."""
    if data.additional_prompt:
        text_prompt += f"\n\nAdditional instructions: {data.additional_prompt}"

    text_resp = client.chat.completions.create(
        model=s.azure_openai_deployment_name,
        messages=[
            {"role": "system", "content": "You are a professional social media content creator."},
            {"role": "user", "content": text_prompt},
        ],
        max_tokens=500, temperature=0.7,
    )
    post_text = text_resp.choices[0].message.content.strip()
    
    # Track text generation credits
    text_usage = text_resp.usage
    total_credits = register_credit_usage(
        db=db,
        user_id=current_user.id,
        channel_id=ch.id,
        resource_type="post",
        resource_id=None,  # Will be updated later
        operation_type="text_generation",
        model_name=s.azure_openai_deployment_name,
        input_tokens=text_usage.prompt_tokens,
        output_tokens=text_usage.completion_tokens,
        metadata={"step": "post_text_generation"},
    )

    # Extract main subject for image consistency
    subj_resp = client.chat.completions.create(
        model=s.azure_openai_deployment_name,
        messages=[
            {"role": "system", "content": "You identify the main subject of social media posts."},
            {"role": "user", "content": f"Identify the main subject of this post in 2-5 words max:\n\n{post_text}\n\nReturn only the subject."},
        ],
        max_tokens=20, temperature=0.3,
    )
    main_subject = subj_resp.choices[0].message.content.strip()
    
    # Track subject extraction credits
    subj_usage = subj_resp.usage
    total_credits += register_credit_usage(
        db=db,
        user_id=current_user.id,
        channel_id=ch.id,
        resource_type="post",
        resource_id=None,
        operation_type="text_generation",
        model_name=s.azure_openai_deployment_name,
        input_tokens=subj_usage.prompt_tokens,
        output_tokens=subj_usage.completion_tokens,
        metadata={"step": "subject_extraction"},
    )

    # Generate image
    image_prompt = None
    post_id = f"post_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    blob_url = ""
    image_error = None
    model_ready = s.azure_openai_image_endpoint or (ch.image_model == "gpt-image-2" and GPT_IMAGE_2_API_KEY)
    if not model_ready:
        image_error = "Endpoint de geração de imagem não configurado. Configure em Configurações → Azure OpenAI Image Endpoint."
        print(f"[IMAGE] Skipping image generation: {image_error}")
    else:
        image_prompt = ch.image_generation_prompt or f"Instagram post image for {ch.name}. Theme: {ch.objective}. Main subject: {main_subject}"
        if ch.image_generation_prompt:
            image_prompt += f"\n\nItem específico: {main_subject}"
        if data.additional_prompt:
            image_prompt += f"\n\n{data.additional_prompt}"
        image_prompt += get_reference_context(ch.id, db)
        try:
            img_bytes = generate_image_bytes(image_prompt, ch, s, db)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            blob_url = upload_bytes_to_blob(img_bytes, f"posts/{post_id}_{ts}.png", "image/png")

            # Track image generation credits
            image_model = ch.image_model or "mai"
            total_credits += register_credit_usage(
                db=db,
                user_id=current_user.id,
                channel_id=ch.id,
                resource_type="post",
                resource_id=post_id,
                operation_type="image_generation",
                model_name=image_model,
                images_count=1,
                metadata={"prompt_length": len(image_prompt)},
            )
        except Exception as e:
            image_error = str(e)
            print(f"Image generation failed: {e}")

    p = PostDB(
        id=post_id,
        channel_id=ch.id,
        channel_name=ch.name,
        text=post_text,
        image_path=blob_url,
        prompt=image_prompt,
        published=False,
        credits_consumed=total_credits,
    )
    db.add(p)
    db.commit()
    
    # Update resource_id in credit_usage records
    db.execute(
        text("UPDATE credit_usage SET resource_id = :post_id WHERE resource_id IS NULL AND user_id = :user_id AND channel_id = :channel_id"),
        {"post_id": post_id, "user_id": current_user.id, "channel_id": ch.id}
    )
    db.commit()

    return Post(id=post_id, text=post_text, image_url=blob_url, image_error=image_error)


@app.post("/api/posts/{post_id}/image/upload")
def upload_post_image(
    post_id: str,
    file: UploadFile = File(...),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = _get_post_or_404(post_id, current_user, db)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")

    data = file.file.read()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = (file.content_type or "image/png").split("/")[-1].replace("jpeg", "jpg")
    blob_url = upload_bytes_to_blob(data, f"posts/{post_id}_{ts}.{ext}", file.content_type or "image/png")
    p.image_path = blob_url
    db.commit()
    return {
        "success": True,
        "image_url": blob_url,
        "image_path": blob_url,
    }


@app.post("/api/posts/{post_id}/image/generate")
def generate_post_image(
    post_id: str,
    data: GeneratePostImageRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = _get_post_or_404(post_id, current_user, db)
    s = get_or_create_settings(current_user, db)
    if not s.azure_openai_image_endpoint:
        raise HTTPException(status_code=400, detail="Endpoint de imagem não configurado")

    ch = db.query(ChannelDB).filter(ChannelDB.id == data.channel_id).first()
    channel_prompt = (ch.image_generation_prompt or "") if ch else ""
    if channel_prompt and data.prompt:
        full_prompt = f"{channel_prompt}\n\n{data.prompt}"
    else:
        full_prompt = channel_prompt or data.prompt
    if ch:
        full_prompt += get_reference_context(ch.id, db)

    try:
        img_bytes = generate_image_bytes(full_prompt, ch, s, db)
    except HTTPException:
        raise
    except Exception as e:
        # Content safety fallback: retry with user prompt only
        if data.prompt and channel_prompt:
            try:
                fallback_prompt = data.prompt + (f"\n\n{get_reference_context(ch.id, db)}" if ch else "")
                img_bytes = generate_image_bytes(fallback_prompt, ch, s, db)
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Falha ao gerar imagem: {str(e2)}")
        else:
            raise HTTPException(status_code=500, detail=f"Falha ao gerar imagem: {str(e)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    blob_url = upload_bytes_to_blob(img_bytes, f"posts/{post_id}_{ts}.png", "image/png")
    p.image_path = blob_url
    p.prompt = full_prompt
    
    # Track image generation credits and add to post total
    image_model = ch.image_model if ch else "mai"
    credits = register_credit_usage(
        db=db,
        user_id=current_user.id,
        channel_id=ch.id if ch else None,
        resource_type="post",
        resource_id=post_id,
        operation_type="image_generation",
        model_name=image_model,
        images_count=1,
        metadata={"prompt_length": len(full_prompt), "regenerated": True},
    )
    
    p.credits_consumed = (p.credits_consumed or 0.0) + credits
    db.commit()
    return {
        "success": True,
        "image_url": blob_url,
        "image_path": blob_url,
    }


@app.post("/api/posts/{post_id}/publish")
def publish_post(
    post_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = _get_post_or_404(post_id, current_user, db)
    ch = db.query(ChannelDB).filter(ChannelDB.id == p.channel_id).first()

    if not ch.instagram_user_id or not ch.instagram_access_token:
        raise HTTPException(
            status_code=400,
            detail="Instagram não configurado para este canal.",
        )

    image_url = p.image_path  # blob URL stored directly

    try:
        _api = _ig_api_base(ch.instagram_access_token)
        create_resp = requests.post(
            f"{_api}/{ch.instagram_user_id}/media",
            params={"image_url": image_url, "caption": p.text, "access_token": ch.instagram_access_token},
            timeout=30,
        )
        create_data = create_resp.json()
        if create_resp.status_code != 200 or "id" not in create_data:
            error_msg = create_data.get("error", {}).get("message", create_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao criar container: {error_msg}")

        container_id = create_data["id"]
        for _ in range(15):
            import time; time.sleep(2)
            status_resp = requests.get(
                f"{_api}/{container_id}",
                params={"fields": "status_code", "access_token": ch.instagram_access_token},
                timeout=15,
            )
            sc = status_resp.json().get("status_code", "")
            if sc == "FINISHED":
                break
            if sc == "ERROR":
                raise HTTPException(status_code=502, detail="Erro ao processar mídia no Instagram.")

        pub_resp = requests.post(
            f"{_api}/{ch.instagram_user_id}/media_publish",
            params={"creation_id": container_id, "access_token": ch.instagram_access_token},
            timeout=30,
        )
        pub_data = pub_resp.json()
        if pub_resp.status_code != 200 or "id" not in pub_data:
            error_msg = pub_data.get("error", {}).get("message", pub_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao publicar: {error_msg}")

        p.published = True
        p.ig_media_id = pub_data["id"]
        db.commit()
        return {"success": True, "instagram_post_id": pub_data["id"]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {str(e)}")


# ---------------------------------------------------------------------------
# Videos endpoints
# ---------------------------------------------------------------------------
def video_to_schema(v: VideoDB, video_project_id: str = None, insights=None) -> SavedVideo:
    return SavedVideo(
        id=v.id,
        channel_id=v.channel_id,
        channel_name=v.channel_name,
        prompt=v.prompt or "",
        caption=v.caption or "",
        video_path=v.video_path or "",
        duration_seconds=v.duration_seconds or 4,
        size=v.size or "720x1280",
        credits_consumed=getattr(v, "credits_consumed", 0.0),
        created_at=v.created_at.isoformat() if v.created_at else datetime.now().isoformat(),
        published=v.published or False,
        is_project_clip=v.is_project_clip or False,
        video_project_id=video_project_id,
        ig_media_id=getattr(v, "ig_media_id", None),
        insights=_insights_to_schema(insights),
    )


def _sora_headers():
    return {"Content-Type": "application/json", "Authorization": f"Bearer {AZURE_SORA_API_KEY}"}


@app.get("/api/videos", response_model=List[SavedVideo])
def get_videos(
    channel_id: Optional[str] = None,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_channel_ids = [
        ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == current_user.id).all()
    ]
    q = db.query(VideoDB).filter(
        VideoDB.channel_id.in_(user_channel_ids),
        VideoDB.is_project_clip.is_not(True),
    )
    if channel_id:
        q = q.filter(VideoDB.channel_id == channel_id)
    videos = q.order_by(VideoDB.created_at.desc()).all()

    # Batch-lookup which videos are roots/exports of an existing project
    video_ids = [v.id for v in videos]
    project_map = {}
    if video_ids:
        for vp in db.query(VideoProjectDB.id, VideoProjectDB.root_video_id, VideoProjectDB.exported_video_id).filter(
            (VideoProjectDB.root_video_id.in_(video_ids)) | (VideoProjectDB.exported_video_id.in_(video_ids))
        ).all():
            if vp.root_video_id:
                project_map[vp.root_video_id] = vp.id
            if vp.exported_video_id:
                project_map[vp.exported_video_id] = vp.id

    ins_map = {}
    if video_ids:
        for ins in db.query(MediaInsightsDB).filter(
            MediaInsightsDB.media_type == "video",
            MediaInsightsDB.media_id.in_(video_ids),
        ).all():
            ins_map[ins.media_id] = ins

    return [video_to_schema(v, project_map.get(v.id), ins_map.get(v.id)) for v in videos]


@app.post("/api/videos/generate", response_model=SavedVideo)
def generate_video(
    data: GenerateVideoRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not AZURE_SORA_ENDPOINT or not AZURE_SORA_API_KEY:
        raise HTTPException(status_code=400, detail="Sora não configurado. Defina AZURE_SORA_ENDPOINT e AZURE_SORA_API_KEY.")

    ch = get_channel_or_404(data.channel_id, current_user, db)
    s = get_or_create_settings(current_user, db)

    # Build prompt from channel config
    base_prompt = ch.image_generation_prompt or f"Instagram Reel for channel '{ch.name}'. Theme: {ch.objective}."
    prompt = base_prompt
    if data.additional_prompt:
        prompt += f" {data.additional_prompt}"
    prompt = prompt[:4000]  # Sora max prompt length

    print(f"Sora prompt ({len(prompt)} chars): {prompt[:200]}")
    # Create Sora job (kick off async before generating caption)
    # Endpoint: POST {AZURE_SORA_ENDPOINT}  (e.g. https://postgen-ai.services.ai.azure.com/openai/v1/videos)
    try:
        create_resp = requests.post(
            AZURE_SORA_ENDPOINT,
            headers=_sora_headers(),
            json={
                "prompt": prompt,
                "model": "sora-2",
                "size": data.size,
                "seconds": str(data.seconds),
            },
            timeout=30,
        )
        if not create_resp.ok:
            raise HTTPException(status_code=502, detail=f"Falha ao criar job Sora: {create_resp.status_code} - {create_resp.text}")
        create_resp.raise_for_status()
    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Falha ao criar job Sora: {str(e)}")

    job = create_resp.json()
    print(f"Sora job created: {job}")
    job_id = job.get("id") or job.get("job_id") or job.get("generation_id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"Resposta inesperada do Sora: {job}")

    # Generate Instagram caption text while Sora processes (parallel work)
    caption = ""
    total_credits = 0.0
    try:
        client = get_azure_client(s)
        text_prompt = ch.text_generation_prompt or f"""Crie uma legenda para um Instagram Reel do canal "{ch.name}".
Objetivo do canal: {ch.objective}
Conceito do vídeo: {data.additional_prompt or prompt}
Escreva uma legenda envolvente com emojis e hashtags relevantes, 80-150 palavras.
Retorne apenas o texto da legenda."""
        cap_resp = client.chat.completions.create(
            model=s.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": "Você é um especialista em conteúdo para Instagram."},
                {"role": "user", "content": text_prompt},
            ],
            max_tokens=400, temperature=0.7,
        )
        caption = cap_resp.choices[0].message.content.strip()
        
        # Track caption generation credits
        cap_usage = cap_resp.usage
        total_credits += register_credit_usage(
            db=db,
            user_id=current_user.id,
            channel_id=ch.id,
            resource_type="video",
            resource_id=None,  # Will be updated later
            operation_type="text_generation",
            model_name=s.azure_openai_deployment_name,
            input_tokens=cap_usage.prompt_tokens,
            output_tokens=cap_usage.completion_tokens,
            metadata={"step": "video_caption_generation"},
        )
    except Exception as e:
        print(f"Caption generation failed: {e}")

    # Poll until complete (max 4 minutes)
    # Poll URL: GET {AZURE_SORA_ENDPOINT}/{job_id}
    poll_url = f"{AZURE_SORA_ENDPOINT}/{job_id}"
    import time
    deadline = datetime.now().timestamp() + 240
    completed = False
    while datetime.now().timestamp() < deadline:
        time.sleep(5)
        try:
            poll_resp = requests.get(poll_url, headers=_sora_headers(), timeout=15)
            result = poll_resp.json()
            status = result.get("status", "")
            print(f"Sora poll: status={status} progress={result.get('progress', '?')}")
        except Exception as e:
            print(f"Sora poll error: {e}")
            continue

        if status == "completed":
            completed = True
            break
        if status in ("failed", "error", "cancelled"):
            err_obj = result.get("error") or {}
            err = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
            raise HTTPException(status_code=502, detail=f"Sora falhou: {err or status}")

    if not completed:
        raise HTTPException(status_code=504, detail="Timeout aguardando o Sora. Tente novamente.")

    # Download video from content endpoint and upload to blob
    # Content URL: GET {AZURE_SORA_ENDPOINT}/{job_id}/content
    video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    content_url = f"{AZURE_SORA_ENDPOINT}/{job_id}/content"
    try:
        dl = requests.get(content_url, headers=_sora_headers(), timeout=120, allow_redirects=True)
        dl.raise_for_status()
        video_bytes = dl.content
        blob_url = upload_bytes_to_blob(video_bytes, f"videos/{video_id}.mp4", "video/mp4")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao baixar vídeo Sora: {str(e)}")

    video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Track video generation credits
    total_credits += register_credit_usage(
        db=db,
        user_id=current_user.id,
        channel_id=ch.id,
        resource_type="video",
        resource_id=video_id,
        operation_type="video_generation",
        model_name="sora-2",
        video_seconds=data.seconds,
        metadata={"prompt_length": len(prompt), "size": data.size},
    )

    v = VideoDB(
        id=video_id,
        channel_id=ch.id,
        channel_name=ch.name,
        prompt=prompt,
        caption=caption,
        video_path=blob_url,
        duration_seconds=data.seconds,
        size=data.size,
        published=False,
        credits_consumed=total_credits,
    )
    db.add(v)
    db.commit()
    
    # Update resource_id in credit_usage records
    db.execute(
        text("UPDATE credit_usage SET resource_id = :video_id WHERE resource_id IS NULL AND user_id = :user_id AND channel_id = :channel_id"),
        {"video_id": video_id, "user_id": current_user.id, "channel_id": ch.id}
    )
    db.commit()
    
    db.refresh(v)
    return video_to_schema(v)


@app.delete("/api/videos/{video_id}", status_code=204)
def delete_video(
    video_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_channel_ids = [
        ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == current_user.id).all()
    ]
    v = db.query(VideoDB).filter(
        VideoDB.id == video_id,
        VideoDB.channel_id.in_(user_channel_ids),
    ).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado")
    db.delete(v)
    db.commit()


@app.patch("/api/videos/{video_id}/caption", response_model=SavedVideo)
def update_video_caption(
    video_id: str,
    data: UpdateVideoCaptionRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_channel_ids = [
        ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == current_user.id).all()
    ]
    v = db.query(VideoDB).filter(
        VideoDB.id == video_id,
        VideoDB.channel_id.in_(user_channel_ids),
    ).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado")
    v.caption = data.caption
    db.commit()
    db.refresh(v)
    return video_to_schema(v)


@app.post("/api/videos/{video_id}/publish", response_model=SavedVideo)
def publish_video(
    video_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_channel_ids = [
        ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == current_user.id).all()
    ]
    v = db.query(VideoDB).filter(
        VideoDB.id == video_id,
        VideoDB.channel_id.in_(user_channel_ids),
    ).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado")

    ch = db.query(ChannelDB).filter(ChannelDB.id == v.channel_id).first()
    if not ch.instagram_user_id or not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não configurado para este canal.")

    try:
        _api = _ig_api_base(ch.instagram_access_token)
        # Create Reels container
        create_resp = requests.post(
            f"{_api}/{ch.instagram_user_id}/media",
            params={
                "media_type": "REELS",
                "video_url": v.video_path,
                "caption": v.caption or v.prompt,
                "access_token": ch.instagram_access_token,
            },
            timeout=30,
        )
        create_data = create_resp.json()
        if create_resp.status_code != 200 or "id" not in create_data:
            error_msg = create_data.get("error", {}).get("message", create_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao criar container Reels: {error_msg}")

        # Poll until container is ready (max 2 minutes)
        import time
        container_id = create_data["id"]
        for _ in range(24):
            time.sleep(5)
            status_resp = requests.get(
                f"{_api}/{container_id}",
                params={"fields": "status_code", "access_token": ch.instagram_access_token},
                timeout=15,
            )
            if status_resp.json().get("status_code") == "FINISHED":
                break

        # Publish
        pub_resp = requests.post(
            f"{_api}/{ch.instagram_user_id}/media_publish",
            params={"creation_id": container_id, "access_token": ch.instagram_access_token},
            timeout=30,
        )
        pub_data = pub_resp.json()
        if pub_resp.status_code != 200 or "id" not in pub_data:
            error_msg = pub_data.get("error", {}).get("message", pub_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao publicar Reel: {error_msg}")

        v.published = True
        v.ig_media_id = pub_data["id"]
        db.commit()
        db.refresh(v)
        return video_to_schema(v)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {str(e)}")


# ---------------------------------------------------------------------------
# Video Projects (editor)
# ---------------------------------------------------------------------------

def _video_project_to_schema(vp: VideoProjectDB, db: Session) -> VideoProjectOut:
    try:
        clip_ids = json.loads(vp.clip_ids or "[]")
    except Exception:
        clip_ids = []
    clips = []
    for cid in clip_ids:
        v = db.query(VideoDB).filter(VideoDB.id == cid).first()
        if v:
            clips.append(video_to_schema(v))
    return VideoProjectOut(
        id=vp.id,
        channel_id=vp.channel_id,
        title=vp.title or "",
        clips=clips,
        exported_path=vp.exported_path,
        created_at=vp.created_at.isoformat() if vp.created_at else datetime.now().isoformat(),
        updated_at=vp.updated_at.isoformat() if vp.updated_at else datetime.now().isoformat(),
    )


def _get_project_or_404(project_id: str, user: UserDB, db: Session) -> VideoProjectDB:
    vp = db.query(VideoProjectDB).filter(
        VideoProjectDB.id == project_id,
        VideoProjectDB.user_id == user.id,
    ).first()
    if not vp:
        raise HTTPException(status_code=404, detail="Projeto não encontrado")
    return vp


@app.post("/api/video-projects", response_model=VideoProjectOut)
def create_video_project(
    data: CreateVideoProjectRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(data.channel_id, current_user, db)
    v = db.query(VideoDB).filter(VideoDB.id == data.video_id, VideoDB.channel_id == ch.id).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado")
    project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    vp = VideoProjectDB(
        id=project_id,
        channel_id=ch.id,
        user_id=current_user.id,
        title=f"Projeto {ch.name}",
        clip_ids=json.dumps([data.video_id]),
        clip_urls=json.dumps({data.video_id: v.video_path}),
        root_video_id=data.video_id,
    )
    db.add(vp)
    db.commit()
    db.refresh(vp)
    return _video_project_to_schema(vp, db)


@app.get("/api/video-projects/{project_id}", response_model=VideoProjectOut)
def get_video_project(
    project_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vp = _get_project_or_404(project_id, current_user, db)
    return _video_project_to_schema(vp, db)


@app.put("/api/video-projects/{project_id}/clips", response_model=VideoProjectOut)
def update_video_project_clips(
    project_id: str,
    data: UpdateVideoProjectClipsRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vp = _get_project_or_404(project_id, current_user, db)
    vp.clip_ids = json.dumps(data.clip_ids)
    db.commit()
    db.refresh(vp)
    return _video_project_to_schema(vp, db)


@app.post("/api/video-projects/{project_id}/add-video", response_model=VideoProjectOut)
def add_video_to_project(
    project_id: str,
    data: AddVideoToProjectRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vp = _get_project_or_404(project_id, current_user, db)
    user_channel_ids = [
        ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == current_user.id).all()
    ]
    v = db.query(VideoDB).filter(
        VideoDB.id == data.video_id,
        VideoDB.channel_id.in_(user_channel_ids),
    ).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado")

    try:
        clip_ids = json.loads(vp.clip_ids or "[]")
    except Exception:
        clip_ids = []

    # Heal legacy projects that have root_video_id = NULL
    if not vp.root_video_id and clip_ids:
        vp.root_video_id = clip_ids[0]

    if data.video_id not in clip_ids:
        clip_ids.append(data.video_id)
        vp.clip_ids = json.dumps(clip_ids)

    try:
        clip_urls = json.loads(vp.clip_urls or "{}")
    except Exception:
        clip_urls = {}
    if data.video_id not in clip_urls:
        clip_urls[data.video_id] = v.video_path
        vp.clip_urls = json.dumps(clip_urls)

    v.is_project_clip = True
    db.commit()
    db.refresh(vp)
    return _video_project_to_schema(vp, db)


@app.post("/api/video-projects/{project_id}/generate", response_model=VideoProjectOut)
def generate_project_clip(
    project_id: str,
    data: GenerateProjectClipRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    import time
    vp = _get_project_or_404(project_id, current_user, db)
    ch = get_channel_or_404(vp.channel_id, current_user, db)
    s = get_or_create_settings(current_user, db)

    if not AZURE_SORA_ENDPOINT or not AZURE_SORA_API_KEY:
        raise HTTPException(status_code=400, detail="Sora não configurado.")

    base_prompt = ch.image_generation_prompt or f"Instagram Reel for channel '{ch.name}'. Theme: {ch.objective}."
    prompt = base_prompt
    if data.additional_prompt:
        prompt += f" {data.additional_prompt}"
    prompt = prompt[:4000]

    try:
        create_resp = requests.post(
            AZURE_SORA_ENDPOINT,
            headers=_sora_headers(),
            json={"prompt": prompt, "model": "sora-2", "size": data.size, "seconds": str(data.seconds)},
            timeout=30,
        )
        if not create_resp.ok:
            raise HTTPException(status_code=502, detail=f"Falha ao criar job Sora: {create_resp.status_code} - {create_resp.text}")
    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Falha ao criar job Sora: {str(e)}")

    job = create_resp.json()
    job_id = job.get("id") or job.get("job_id") or job.get("generation_id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"Resposta inesperada do Sora: {job}")

    caption = ""
    try:
        client = get_azure_client(s)
        cap_resp = client.chat.completions.create(
            model=s.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": "Você é um especialista em conteúdo para Instagram."},
                {"role": "user", "content": f"Crie uma legenda curta para um clipe: {data.additional_prompt or prompt[:200]}. Retorne apenas a legenda."},
            ],
            max_tokens=200, temperature=0.7,
        )
        caption = cap_resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Caption generation failed: {e}")

    poll_url = f"{AZURE_SORA_ENDPOINT}/{job_id}"
    deadline = datetime.now().timestamp() + 240
    completed = False
    while datetime.now().timestamp() < deadline:
        time.sleep(5)
        try:
            poll_resp = requests.get(poll_url, headers=_sora_headers(), timeout=15)
            result = poll_resp.json()
            status = result.get("status", "")
        except Exception:
            continue
        if status == "completed":
            completed = True
            break
        if status in ("failed", "error", "cancelled"):
            err_obj = result.get("error") or {}
            err = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
            raise HTTPException(status_code=502, detail=f"Sora falhou: {err or status}")

    if not completed:
        raise HTTPException(status_code=504, detail="Timeout aguardando o Sora.")

    video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    content_url = f"{AZURE_SORA_ENDPOINT}/{job_id}/content"
    try:
        dl = requests.get(content_url, headers=_sora_headers(), timeout=120, allow_redirects=True)
        dl.raise_for_status()
        blob_url = upload_bytes_to_blob(dl.content, f"videos/{video_id}.mp4", "video/mp4")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao baixar vídeo: {str(e)}")

    v = VideoDB(
        id=video_id,
        channel_id=ch.id,
        channel_name=ch.name,
        prompt=prompt,
        caption=caption,
        video_path=blob_url,
        duration_seconds=data.seconds,
        size=data.size,
        published=False,
        is_project_clip=True,
    )
    db.add(v)

    try:
        clip_ids = json.loads(vp.clip_ids or "[]")
    except Exception:
        clip_ids = []
    clip_ids.append(video_id)
    vp.clip_ids = json.dumps(clip_ids)

    try:
        clip_urls = json.loads(vp.clip_urls or "{}")
    except Exception:
        clip_urls = {}
    clip_urls[video_id] = blob_url
    vp.clip_urls = json.dumps(clip_urls)

    db.commit()
    db.refresh(vp)
    return _video_project_to_schema(vp, db)


@app.post("/api/video-projects/{project_id}/save", response_model=VideoProjectOut)
def save_video_project(
    project_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vp = _get_project_or_404(project_id, current_user, db)

    try:
        clip_ids = json.loads(vp.clip_ids or "[]")
    except Exception:
        clip_ids = []

    if not clip_ids:
        raise HTTPException(status_code=400, detail="Projeto sem clipes")

    clips = []
    for cid in clip_ids:
        v = db.query(VideoDB).filter(VideoDB.id == cid).first()
        if v and v.video_path:
            clips.append(v)

    if not clips:
        raise HTTPException(status_code=400, detail="Nenhum clipe válido encontrado")

    try:
        clip_url_map = json.loads(vp.clip_urls or "{}")
    except Exception:
        clip_url_map = {}

    if len(clips) == 1:
        merged_url = clip_url_map.get(clips[0].id) or clips[0].video_path
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_files = []
            for i, v in enumerate(clips):
                clip_path = os.path.join(tmpdir, f"clip_{i}.mp4")
                source_url = clip_url_map.get(v.id) or v.video_path
                try:
                    dl = requests.get(source_url, timeout=120, stream=True)
                    dl.raise_for_status()
                    with open(clip_path, "wb") as f:
                        for chunk in dl.iter_content(chunk_size=65536):
                            f.write(chunk)
                    clip_files.append(clip_path)
                except Exception as e:
                    raise HTTPException(status_code=502, detail=f"Erro ao baixar clipe: {str(e)}")

            concat_list = os.path.join(tmpdir, "concat.txt")
            with open(concat_list, "w") as f:
                for cp in clip_files:
                    f.write(f"file '{cp}'\n")

            output_path = os.path.join(tmpdir, "merged.mp4")
            result = subprocess.run(
                ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", output_path, "-y"],
                capture_output=True, timeout=180,
            )
            if result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Falha ao mesclar vídeos: {result.stderr.decode(errors='replace')}")

            export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            with open(output_path, "rb") as f:
                merged_url = upload_bytes_to_blob(f.read(), f"videos/{export_id}.mp4", "video/mp4")

    # Heal legacy projects that have root_video_id = NULL
    if not vp.root_video_id and clip_ids:
        vp.root_video_id = clip_ids[0]

    if vp.exported_video_id:
        # Re-export: overwrite only the previously compiled entry, never the original clips
        exp_v = db.query(VideoDB).filter(VideoDB.id == vp.exported_video_id).first()
        if exp_v:
            exp_v.video_path = merged_url
            exp_v.duration_seconds = sum(c.duration_seconds or 0 for c in clips)
    else:
        # First export: create a new VideoDB entry for the compiled result
        root_v = db.query(VideoDB).filter(VideoDB.id == vp.root_video_id).first() if vp.root_video_id else None
        export_video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        exp_v = VideoDB(
            id=export_video_id,
            channel_id=vp.channel_id,
            channel_name=root_v.channel_name if root_v else "",
            prompt=root_v.prompt if root_v else "",
            caption=root_v.caption if root_v else "",
            video_path=merged_url,
            duration_seconds=sum(c.duration_seconds or 0 for c in clips),
            size=clips[0].size if clips else "720x1280",
            published=False,
            is_project_clip=False,
        )
        db.add(exp_v)
        vp.exported_video_id = export_video_id

        # Hide the original root clip from the feed — the compiled entry replaces it
        if root_v:
            root_v.is_project_clip = True

    vp.exported_path = merged_url
    db.commit()
    db.refresh(vp)
    return _video_project_to_schema(vp, db)


@app.post("/api/video-projects/{project_id}/export")
def export_video_project(
    project_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vp = _get_project_or_404(project_id, current_user, db)

    try:
        clip_ids = json.loads(vp.clip_ids or "[]")
    except Exception:
        clip_ids = []

    if not clip_ids:
        raise HTTPException(status_code=400, detail="Projeto sem clipes")

    clips = []
    for cid in clip_ids:
        v = db.query(VideoDB).filter(VideoDB.id == cid).first()
        if v and v.video_path:
            clips.append(v)

    if not clips:
        raise HTTPException(status_code=400, detail="Nenhum clipe válido encontrado")

    if len(clips) == 1:
        vp.exported_path = clips[0].video_path
        db.commit()
        return {"exported_url": clips[0].video_path}

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_files = []
        for i, v in enumerate(clips):
            clip_path = os.path.join(tmpdir, f"clip_{i}.mp4")
            try:
                dl = requests.get(v.video_path, timeout=120, stream=True)
                dl.raise_for_status()
                with open(clip_path, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=65536):
                        f.write(chunk)
                clip_files.append(clip_path)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Erro ao baixar clipe {v.id}: {str(e)}")

        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            for cp in clip_files:
                f.write(f"file '{cp}'\n")

        output_path = os.path.join(tmpdir, "merged.mp4")
        result = subprocess.run(
            ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", output_path, "-y"],
            capture_output=True, timeout=180,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Falha ao mesclar vídeos: {result.stderr.decode(errors='replace')}")

        export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        with open(output_path, "rb") as f:
            blob_url = upload_bytes_to_blob(f.read(), f"videos/{export_id}.mp4", "video/mp4")

    vp.exported_path = blob_url
    db.commit()
    return {"exported_url": blob_url}


# ---------------------------------------------------------------------------
# Reference Images endpoints
# ---------------------------------------------------------------------------

@app.get("/api/channels/{channel_id}/references", response_model=List[ReferenceImageOut])
def list_reference_images(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_channel_or_404(channel_id, current_user, db)
    refs = db.query(ReferenceImageDB).filter(
        ReferenceImageDB.channel_id == channel_id,
    ).order_by(ReferenceImageDB.created_at.desc()).all()
    return [ReferenceImageOut(
        id=r.id,
        channel_id=r.channel_id,
        blob_url=r.blob_url,
        description=r.description,
        created_at=r.created_at.isoformat(),
    ) for r in refs]


@app.post("/api/channels/{channel_id}/references/upload", response_model=ReferenceImageOut, status_code=201)
def upload_reference_image(
    channel_id: str,
    file: UploadFile = File(...),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_channel_or_404(channel_id, current_user, db)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")

    data = file.file.read()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = (file.content_type or "image/jpeg").split("/")[-1].replace("jpeg", "jpg")
    blob_url = upload_bytes_to_blob(data, f"references/{channel_id}_{ts}.{ext}", file.content_type or "image/jpeg")

    # Auto-describe via vision model
    description = None
    s = get_or_create_settings(current_user, db)
    if s and s.azure_openai_endpoint and s.azure_openai_api_key:
        try:
            client = get_azure_client(s)
            vision_resp = client.chat.completions.create(
                model=s.azure_openai_deployment_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe the physical appearance of the person in this photo in detail "
                                "(face shape, hair color and style, eye color, skin tone, distinctive features, age range). "
                                "Be specific and concise — this description will be used in AI image generation prompts. "
                                "Answer in English. If there is no person visible, reply with 'no person detected'."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": blob_url}},
                    ],
                }],
                max_tokens=200,
            )
            desc = vision_resp.choices[0].message.content.strip()
            if "no person detected" not in desc.lower():
                description = desc
        except Exception as e:
            print(f"Vision description failed: {e}")

    ref = ReferenceImageDB(
        channel_id=channel_id,
        user_id=current_user.id,
        blob_url=blob_url,
        description=description,
    )
    db.add(ref)
    db.commit()
    db.refresh(ref)
    return ReferenceImageOut(
        id=ref.id,
        channel_id=ref.channel_id,
        blob_url=ref.blob_url,
        description=ref.description,
        created_at=ref.created_at.isoformat(),
    )


@app.delete("/api/channels/{channel_id}/references/{ref_id}", status_code=204)
def delete_reference_image(
    channel_id: str,
    ref_id: int,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_channel_or_404(channel_id, current_user, db)
    ref = db.query(ReferenceImageDB).filter(
        ReferenceImageDB.id == ref_id,
        ReferenceImageDB.channel_id == channel_id,
    ).first()
    if not ref:
        raise HTTPException(status_code=404, detail="Imagem de referência não encontrada")
    db.delete(ref)
    db.commit()


# ---------------------------------------------------------------------------
# Insights endpoints
# ---------------------------------------------------------------------------

def _get_video_or_404(video_id: str, user: UserDB, db: Session) -> VideoDB:
    user_channel_ids = [
        ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == user.id).all()
    ]
    v = db.query(VideoDB).filter(
        VideoDB.id == video_id,
        VideoDB.channel_id.in_(user_channel_ids),
    ).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado")
    return v


@app.get("/api/posts/{post_id}/insights", response_model=InsightsOut)
def get_post_insights(
    post_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = _get_post_or_404(post_id, current_user, db)
    if not p.published or not getattr(p, "ig_media_id", None):
        raise HTTPException(status_code=404, detail="Post não publicado ou sem ID do Instagram")
    ch = db.query(ChannelDB).filter(ChannelDB.id == p.channel_id).first()
    if not ch or not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não conectado")
    ins = db.query(MediaInsightsDB).filter(
        MediaInsightsDB.media_type == "post", MediaInsightsDB.media_id == post_id,
    ).first()
    if _insights_stale(ins, p.created_at):
        ins = _fetch_and_store_insights("post", post_id, p.ig_media_id, p.channel_id, ch.instagram_access_token, db)
    return _insights_to_schema(ins)


@app.post("/api/posts/{post_id}/insights/refresh", response_model=InsightsOut)
def refresh_post_insights(
    post_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = _get_post_or_404(post_id, current_user, db)
    if not p.published or not getattr(p, "ig_media_id", None):
        raise HTTPException(status_code=404, detail="Post não publicado ou sem ID do Instagram")
    ch = db.query(ChannelDB).filter(ChannelDB.id == p.channel_id).first()
    if not ch or not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não conectado")
    ins = _fetch_and_store_insights("post", post_id, p.ig_media_id, p.channel_id, ch.instagram_access_token, db)
    return _insights_to_schema(ins)


@app.get("/api/videos/{video_id}/insights", response_model=InsightsOut)
def get_video_insights(
    video_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    v = _get_video_or_404(video_id, current_user, db)
    if not v.published or not getattr(v, "ig_media_id", None):
        raise HTTPException(status_code=404, detail="Vídeo não publicado ou sem ID do Instagram")
    ch = db.query(ChannelDB).filter(ChannelDB.id == v.channel_id).first()
    if not ch or not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não conectado")
    ins = db.query(MediaInsightsDB).filter(
        MediaInsightsDB.media_type == "video", MediaInsightsDB.media_id == video_id,
    ).first()
    if _insights_stale(ins, v.created_at):
        ins = _fetch_and_store_insights("video", video_id, v.ig_media_id, v.channel_id, ch.instagram_access_token, db)
    return _insights_to_schema(ins)


@app.post("/api/videos/{video_id}/insights/refresh", response_model=InsightsOut)
def refresh_video_insights(
    video_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    v = _get_video_or_404(video_id, current_user, db)
    if not v.published or not getattr(v, "ig_media_id", None):
        raise HTTPException(status_code=404, detail="Vídeo não publicado ou sem ID do Instagram")
    ch = db.query(ChannelDB).filter(ChannelDB.id == v.channel_id).first()
    if not ch or not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não conectado")
    ins = _fetch_and_store_insights("video", video_id, v.ig_media_id, v.channel_id, ch.instagram_access_token, db)
    return _insights_to_schema(ins)


@app.get("/api/channels/{channel_id}/dashboard", response_model=ChannelDashboardOut)
def get_channel_dashboard(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)

    posts = db.query(PostDB).filter(
        PostDB.channel_id == channel_id,
        PostDB.published == True,
        PostDB.ig_media_id.isnot(None),
    ).all()

    videos = db.query(VideoDB).filter(
        VideoDB.channel_id == channel_id,
        VideoDB.published == True,
        VideoDB.ig_media_id.isnot(None),
        VideoDB.is_project_clip.is_not(True),
    ).all()

    post_ids = [p.id for p in posts]
    video_ids = [v.id for v in videos]
    insights_map = {}

    if post_ids:
        for ins in db.query(MediaInsightsDB).filter(
            MediaInsightsDB.media_type == "post",
            MediaInsightsDB.media_id.in_(post_ids),
        ).all():
            insights_map[("post", ins.media_id)] = ins

    if video_ids:
        for ins in db.query(MediaInsightsDB).filter(
            MediaInsightsDB.media_type == "video",
            MediaInsightsDB.media_id.in_(video_ids),
        ).all():
            insights_map[("video", ins.media_id)] = ins

    items = []
    for p in posts:
        ins = insights_map.get(("post", p.id))
        if ins:
            items.append(DashboardItemOut(
                media_type="post", media_id=p.id,
                preview_url=p.image_path or "",
                text_preview=(p.text or "")[:120],
                created_at=p.created_at.isoformat(),
                published=True,
                insights=_insights_to_schema(ins),
            ))
    for v in videos:
        ins = insights_map.get(("video", v.id))
        if ins:
            items.append(DashboardItemOut(
                media_type="video", media_id=v.id,
                preview_url=v.video_path or "",
                text_preview=(v.caption or v.prompt or "")[:120],
                created_at=v.created_at.isoformat(),
                published=True,
                insights=_insights_to_schema(ins),
            ))

    total_reach = sum(i.insights.reach or 0 for i in items)
    total_impressions = sum(i.insights.impressions or 0 for i in items)
    total_interactions = sum(i.insights.total_interactions for i in items)
    total_likes = sum(i.insights.like_count for i in items)
    total_comments = sum(i.insights.comments_count for i in items)
    total_saved = sum(i.insights.saved or 0 for i in items)
    total_shares = sum(i.insights.shares or 0 for i in items)
    rates = [i.insights.engagement_rate for i in items if i.insights.engagement_rate is not None]
    avg_rate = round(sum(rates) / len(rates), 2) if rates else None

    top_by_reach = sorted(items, key=lambda x: x.insights.reach or 0, reverse=True)[:5]
    top_by_engagement = sorted(items, key=lambda x: x.insights.engagement_rate or 0.0, reverse=True)[:5]
    top_by_likes = sorted(items, key=lambda x: x.insights.like_count, reverse=True)[:5]
    top_by_comments = sorted(items, key=lambda x: x.insights.comments_count, reverse=True)[:5]
    top_by_saved = sorted(items, key=lambda x: x.insights.saved or 0, reverse=True)[:5]
    top_by_shares = sorted(items, key=lambda x: x.insights.shares or 0, reverse=True)[:5]

    all_ins = list(insights_map.values())
    last_refreshed = None
    if all_ins:
        valid = [ins.fetched_at for ins in all_ins if ins.fetched_at]
        if valid:
            last_refreshed = max(valid).isoformat()

    return ChannelDashboardOut(
        channel_id=ch.id,
        channel_name=ch.name,
        published_count=len(posts) + len(videos),
        total_reach=total_reach,
        total_impressions=total_impressions,
        total_interactions=total_interactions,
        total_likes=total_likes,
        total_comments=total_comments,
        total_saved=total_saved,
        total_shares=total_shares,
        avg_engagement_rate=avg_rate,
        top_by_reach=top_by_reach,
        top_by_engagement=top_by_engagement,
        top_by_likes=top_by_likes,
        top_by_comments=top_by_comments,
        top_by_saved=top_by_saved,
        top_by_shares=top_by_shares,
        last_refreshed=last_refreshed,
    )


@app.post("/api/channels/{channel_id}/insights/refresh")
def refresh_channel_insights(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)
    if not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não conectado a este canal")

    posts = db.query(PostDB).filter(
        PostDB.channel_id == channel_id,
        PostDB.published == True,
        PostDB.ig_media_id.isnot(None),
    ).all()
    videos = db.query(VideoDB).filter(
        VideoDB.channel_id == channel_id,
        VideoDB.published == True,
        VideoDB.ig_media_id.isnot(None),
        VideoDB.is_project_clip.is_not(True),
    ).all()

    refreshed, errors = 0, 0
    for p in posts:
        try:
            _fetch_and_store_insights("post", p.id, p.ig_media_id, channel_id, ch.instagram_access_token, db)
            refreshed += 1
        except Exception as e:
            print(f"Refresh error post {p.id}: {e}")
            errors += 1
    for v in videos:
        try:
            _fetch_and_store_insights("video", v.id, v.ig_media_id, channel_id, ch.instagram_access_token, db)
            refreshed += 1
        except Exception as e:
            print(f"Refresh error video {v.id}: {e}")
            errors += 1

    return {"refreshed": refreshed, "errors": errors, "total": len(posts) + len(videos)}

# ---------------------------------------------------------------------------
# Instagram Webhooks - Respostas Automáticas
# ---------------------------------------------------------------------------


@app.get("/api/webhooks/instagram")
async def instagram_webhook_verify(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    """
    Endpoint de verificação do webhook do Instagram.
    O Instagram envia uma requisição GET para validar o webhook.
    """
    print(f"[WEBHOOK] Verification request: mode={hub_mode}, token={hub_verify_token}")
    
    if hub_mode == "subscribe" and hub_verify_token == INSTAGRAM_WEBHOOK_VERIFY_TOKEN:
        print(f"[WEBHOOK] Verification successful, returning challenge: {hub_challenge}")
        try:
            return PlainTextResponse(content=hub_challenge)
        except Exception as e:
            print(f"[WEBHOOK] Error returning challenge: {e}")
            return PlainTextResponse(content="200")
    
    print("[WEBHOOK] Verification failed")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


@app.post("/api/webhooks/instagram")
async def instagram_webhook_receive(
    body: dict,
    db: Session = Depends(get_db),
):
    """
    Endpoint para receber notificações de webhooks do Instagram.
    Processa comentários e mensagens para resposta automática.
    """
    print(f"[WEBHOOK] Received event: {json.dumps(body, indent=2)}")
    
    try:
        # Instagram envia o payload na estrutura: body["entry"][0]["changes"]
        if "entry" not in body:
            return {"status": "ignored", "reason": "no entry"}
        
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                field = change.get("field")
                value = change.get("value")
                
                print(f"[WEBHOOK] Processing field={field}, value={json.dumps(value)}")
                
                # Processar comentários
                if field == "comments":
                    await _process_comment_webhook(value, db)
                
                # Processar mensagens (DMs)
                elif field == "messages":
                    await _process_message_webhook(value, db)
        
        return {"status": "processed"}
    
    except Exception as e:
        print(f"[WEBHOOK ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


async def _process_comment_webhook(value: dict, db: Session):
    """Processa webhook de comentário no Instagram"""
    try:
        comment_id = value.get("id")
        text = value.get("text", "")
        from_user = value.get("from", {})
        media_id = value.get("media", {}).get("id")
        parent_id = value.get("parent_id")  # Se tiver parent_id, é uma resposta a outro comentário
        
        if not comment_id or not text:
            print("[WEBHOOK] Comment missing required fields")
            return

        # Ignorar respostas a outros comentários (parent_id presente)
        if parent_id:
            print(f"[WEBHOOK] Ignoring reply comment (parent_id={parent_id}) to prevent loop")
            return

        print(f"[WEBHOOK] New comment: id={comment_id}, text={text[:50]}..., media={media_id}")

        # Buscar canal que tem esse media_id
        channel = db.query(ChannelDB).filter(
            ChannelDB.instagram_access_token.isnot(None),
            ChannelDB.auto_reply_enabled == True,
        ).first()

        if not channel:
            print("[WEBHOOK] No channel with auto_reply enabled")
            return

        # Ignorar comentários feitos pela própria conta do bot (evita loop mesmo sem parent_id)
        from_user_id = str(from_user.get("id", ""))
        if from_user_id and from_user_id == str(channel.instagram_user_id or ""):
            print(f"[WEBHOOK] Ignoring comment from own account ({from_user_id}), prevents loop")
            return
        
        # Buscar informações do post para contexto
        post_context = ""
        post_db = db.query(PostDB).filter(PostDB.ig_media_id == media_id).first()
        if post_db:
            post_context = f"Post: {post_db.text[:200]}"
        
        # Gerar resposta com AI
        reply_text = await _generate_ai_reply(
            channel=channel,
            user_message=text,
            context=post_context,
            message_type="comment"
        )
        
        if not reply_text:
            print("[WEBHOOK] Failed to generate reply")
            return
        
        # Enviar resposta via Instagram API
        success = await _send_instagram_comment_reply(
            comment_id=comment_id,
            reply_text=reply_text,
            access_token=channel.instagram_access_token
        )
        
        if success:
            print(f"[WEBHOOK] Successfully replied to comment {comment_id}")
        else:
            print(f"[WEBHOOK] Failed to send reply to comment {comment_id}")
    
    except Exception as e:
        print(f"[WEBHOOK ERROR] process_comment: {e}")
        import traceback
        traceback.print_exc()


async def _process_message_webhook(value: dict, db: Session):
    """Processa webhook de mensagem (DM) no Instagram"""
    try:
        message_id = value.get("id")
        text = value.get("text", "")
        from_user = value.get("from", {})
        
        if not message_id or not text:
            print("[WEBHOOK] Message missing required fields")
            return
        
        print(f"[WEBHOOK] New message: id={message_id}, text={text[:50]}...")
        
        # Buscar canal com auto_reply ativado
        channel = db.query(ChannelDB).filter(
            ChannelDB.instagram_access_token.isnot(None),
            ChannelDB.auto_reply_enabled == True,
        ).first()
        
        if not channel:
            print("[WEBHOOK] No channel with auto_reply enabled")
            return
        
        # IMPORTANTE: Ignorar mensagens da própria conta para evitar loop infinito
        from_user_id = from_user.get("id")
        if from_user_id and str(from_user_id) == str(channel.instagram_user_id):
            print(f"[WEBHOOK] Ignoring message from own account (prevents loop)")
            return
        
        # Gerar resposta com AI
        reply_text = await _generate_ai_reply(
            channel=channel,
            user_message=text,
            context=f"Canal: {channel.name}. Objetivo: {channel.objective}",
            message_type="dm"
        )
        
        if not reply_text:
            print("[WEBHOOK] Failed to generate reply")
            return
        
        # Enviar resposta via Instagram API
        success = await _send_instagram_message_reply(
            recipient_id=from_user.get("id"),
            reply_text=reply_text,
            instagram_user_id=channel.instagram_user_id,
            access_token=channel.instagram_access_token
        )
        
        if success:
            print(f"[WEBHOOK] Successfully replied to message {message_id}")
        else:
            print(f"[WEBHOOK] Failed to send reply to message {message_id}")
    
    except Exception as e:
        print(f"[WEBHOOK ERROR] process_message: {e}")
        import traceback
        traceback.print_exc()


async def _generate_ai_reply(
    channel: ChannelDB,
    user_message: str,
    context: str,
    message_type: str  # "comment" or "dm"
) -> Optional[str]:
    """Gera uma resposta automática usando AI"""
    try:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        if not endpoint or not api_key:
            print("[WEBHOOK] Azure OpenAI not configured")
            return None
        
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-08-01-preview"
        )
        
        # Usar prompt personalizado se configurado, caso contrário usar o padrão
        if channel.auto_reply_prompt and channel.auto_reply_prompt.strip():
            system_prompt = channel.auto_reply_prompt
        else:
            # Prompt padrão
            system_prompt = f"""Você é um assistente que responde {message_type}s no Instagram para o canal "{channel.name}".

Objetivo do canal: {channel.objective}

Instruções:
- Seja amigável, natural e conversacional
- Responda de forma breve e direta (máximo 2-3 frases)
- Use emojis moderadamente quando apropriado
- Se for um elogio, agradeça com entusiasmo
- Se for uma pergunta, responda de forma útil
- Se for crítica construtiva, agradeça pelo feedback
- Mantenha o tom alinhado com o objetivo do canal
- NÃO use hashtags na resposta
"""
        
        user_prompt = f"""{context}

Mensagem do usuário: "{user_message}"

Gere uma resposta apropriada:"""
        
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        
        reply = response.choices[0].message.content.strip()
        print(f"[WEBHOOK] Generated reply: {reply}")
        return reply
    
    except Exception as e:
        print(f"[WEBHOOK ERROR] generate_ai_reply: {e}")
        import traceback
        traceback.print_exc()
        return None


async def _send_instagram_comment_reply(
    comment_id: str,
    reply_text: str,
    access_token: str
) -> bool:
    """Envia resposta a um comentário no Instagram via API"""
    try:
        url = f"https://graph.instagram.com/v21.0/{comment_id}/replies"
        payload = {
            "message": reply_text,
            "access_token": access_token
        }
        
        response = requests.post(url, json=payload)
        
        if response.ok:
            print(f"[WEBHOOK] Comment reply sent successfully: {response.json()}")
            return True
        else:
            print(f"[WEBHOOK ERROR] Failed to send comment reply: {response.status_code} {response.text}")
            return False
    
    except Exception as e:
        print(f"[WEBHOOK ERROR] send_instagram_comment_reply: {e}")
        return False


async def _send_instagram_message_reply(
    recipient_id: str,
    reply_text: str,
    instagram_user_id: str,
    access_token: str
) -> bool:
    """Envia resposta a uma mensagem (DM) no Instagram via API"""
    try:
        url = f"https://graph.instagram.com/v21.0/{instagram_user_id}/messages"
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": reply_text},
            "access_token": access_token
        }
        
        response = requests.post(url, json=payload)
        
        if response.ok:
            print(f"[WEBHOOK] Message reply sent successfully: {response.json()}")
            return True
        else:
            print(f"[WEBHOOK ERROR] Failed to send message reply: {response.status_code} {response.text}")
            return False
    
    except Exception as e:
        print(f"[WEBHOOK ERROR] send_instagram_message_reply: {e}")
        return False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

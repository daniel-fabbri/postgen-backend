from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from config import ALLOWED_ORIGINS
from database import engine, SessionLocal, Base
from models import SystemConfigDB

from routers.auth import router as auth_router, users_router
from routers.admin import router as admin_router
from routers.settings import router as settings_router
from routers.payments import router as payments_router
from routers.credits import router as credits_router
from routers.channels import router as channels_router
from routers.posts import router as posts_router
from routers.videos import router as videos_router
from routers.insights import router as insights_router
from routers.webhooks import router as webhooks_router

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
app.include_router(channels_router)
app.include_router(posts_router)
app.include_router(videos_router)
app.include_router(insights_router)
app.include_router(webhooks_router)


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
            conn.execute(text("ALTER TABLE channels ADD COLUMN IF NOT EXISTS lora_training_id VARCHAR(100)"))
            conn.execute(text("ALTER TABLE channels ADD COLUMN IF NOT EXISTS lora_status VARCHAR(20)"))
            conn.execute(text("ALTER TABLE channels ADD COLUMN IF NOT EXISTS lora_version VARCHAR(200)"))
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


@app.get("/")
def root():
    return {"message": "PostGen API is running"}

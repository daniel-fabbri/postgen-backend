from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from urllib.parse import urlencode
from pydantic import BaseModel
from typing import Optional, List
import os
import base64
import json
import shutil
import subprocess
import tempfile
import requests
from datetime import datetime, timedelta, timezone
from openai import AzureOpenAI
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, String, Text, Boolean, Integer, Float,
    DateTime, ForeignKey, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.sql import func
from passlib.context import CryptContext
from jose import JWTError, jwt
from azure.storage.blob import BlobServiceClient, ContentSettings
import mercadopago

load_dotenv()

app = FastAPI(title="PostGen API")
# Config
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8004").rstrip("/")
_code_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getenv("STORAGE_BASE", _code_dir)
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

DATABASE_URL = os.getenv("DATABASE_URL", "")
JWT_SECRET = os.getenv("JWT_SECRET", "changeme-insecure-default-secret-32chars!!")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "images")

AZURE_SORA_ENDPOINT = os.getenv("AZURE_SORA_ENDPOINT", "").rstrip("/")
AZURE_SORA_API_KEY = os.getenv("AZURE_SORA_API_KEY", "")

GPT_IMAGE_2_ENDPOINT = os.getenv("GPT_IMAGE_2_ENDPOINT", "https://postgen-ai.openai.azure.com").rstrip("/")
GPT_IMAGE_2_API_KEY = os.getenv("GPT_IMAGE_2_API_KEY", "")

INSTAGRAM_APP_ID = os.getenv("INSTAGRAM_APP_ID", "")
INSTAGRAM_APP_SECRET = os.getenv("INSTAGRAM_APP_SECRET", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

MERCADOPAGO_ACCESS_TOKEN = os.getenv("MERCADOPAGO_ACCESS_TOKEN", "")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    credits_balance = Column(Float, default=0.0)  # Saldo de créditos disponíveis
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    channels = relationship("ChannelDB", back_populates="user", cascade="all, delete-orphan")
    settings = relationship("SettingsDB", back_populates="user", uselist=False, cascade="all, delete-orphan")
    payments = relationship("PaymentDB", back_populates="user", cascade="all, delete-orphan")


class ChannelDB(Base):
    __tablename__ = "channels"
    id = Column(String(50), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    objective = Column(Text, default="")
    text_generation_prompt = Column(Text, nullable=True)
    image_generation_prompt = Column(Text, nullable=True)
    avatar_url = Column(Text, nullable=True)
    suggested_image_url = Column(Text, nullable=True)
    instagram_user_id = Column(String(255), nullable=True)
    instagram_access_token = Column(Text, nullable=True)
    image_model = Column(String(20), default="mai")   # "mai" | "gpt-image-2"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("UserDB", back_populates="channels")
    posts = relationship("PostDB", back_populates="channel", cascade="all, delete-orphan")
    videos = relationship("VideoDB", back_populates="channel", cascade="all, delete-orphan")
    avatars = relationship("AvatarDB", back_populates="channel")
    reference_images = relationship("ReferenceImageDB", back_populates="channel", cascade="all, delete-orphan")


class PostDB(Base):
    __tablename__ = "posts"
    id = Column(String(100), primary_key=True)
    channel_id = Column(String(50), ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    channel_name = Column(String(255), nullable=False)
    text = Column(Text, default="")
    image_path = Column(Text, default="")
    prompt = Column(Text, nullable=True)
    ig_media_id = Column(String(100), nullable=True)
    published = Column(Boolean, default=False)
    credits_consumed = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    channel = relationship("ChannelDB", back_populates="posts")


class VideoDB(Base):
    __tablename__ = "videos"
    id = Column(String(100), primary_key=True)
    channel_id = Column(String(50), ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    channel_name = Column(String(255), nullable=False)
    prompt = Column(Text, default="")
    caption = Column(Text, default="")
    video_path = Column(Text, default="")
    duration_seconds = Column(Integer, default=4)
    size = Column(String(20), default="720x1280")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    published = Column(Boolean, default=False)
    is_project_clip = Column(Boolean, default=False)
    ig_media_id = Column(String(100), nullable=True)
    credits_consumed = Column(Float, default=0.0)
    channel = relationship("ChannelDB", back_populates="videos")


class VideoProjectDB(Base):
    __tablename__ = "video_projects"
    id = Column(String(100), primary_key=True)
    channel_id = Column(String(50), ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), default="")
    clip_ids = Column(Text, default="[]")
    clip_urls = Column(Text, default="{}")  # {video_id: original_url} — never overwritten by exports
    root_video_id = Column(String(100), nullable=True)
    exported_video_id = Column(String(100), nullable=True)  # the compiled result in the feed
    exported_path = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class MediaInsightsDB(Base):
    __tablename__ = "media_insights"
    id = Column(Integer, primary_key=True, autoincrement=True)
    media_type = Column(String(10), nullable=False)   # "post" | "video"
    media_id = Column(String(100), nullable=False, index=True)
    ig_media_id = Column(String(100), nullable=False)
    channel_id = Column(String(50), ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    like_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    impressions = Column(Integer, nullable=True)
    reach = Column(Integer, nullable=True)
    saved = Column(Integer, nullable=True)
    shares = Column(Integer, nullable=True)
    video_views = Column(Integer, nullable=True)
    total_interactions = Column(Integer, default=0)
    engagement_rate = Column(Float, nullable=True)   # (interactions / reach) * 100
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ReferenceImageDB(Base):
    __tablename__ = "reference_images"
    id = Column(Integer, primary_key=True, autoincrement=True)
    channel_id = Column(String(50), ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    blob_url = Column(Text, nullable=False)
    description = Column(Text, nullable=True)   # Auto-extracted via vision model
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    channel = relationship("ChannelDB", back_populates="reference_images")


class AvatarDB(Base):
    __tablename__ = "avatars"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), unique=True, index=True, nullable=False)
    channel_id = Column(String(50), ForeignKey("channels.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    channel = relationship("ChannelDB", back_populates="avatars")


class SettingsDB(Base):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    azure_openai_endpoint = Column(Text, default="")
    azure_openai_api_key = Column(Text, default="")
    azure_openai_deployment_name = Column(Text, default="gpt-4")
    azure_openai_image_deployment = Column(Text, default="dall-e-3")
    azure_openai_image_endpoint = Column(Text, default="")
    azure_openai_api_version = Column(Text, default="2024-02-01")
    public_base_url = Column(Text, default="http://localhost:8004")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user = relationship("UserDB", back_populates="settings")


class CreditUsageDB(Base):
    __tablename__ = "credit_usage"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    channel_id = Column(String(50), ForeignKey("channels.id", ondelete="CASCADE"), nullable=True)
    resource_type = Column(String(20), nullable=False)  # "post" | "video" | "avatar" | "image"
    resource_id = Column(String(100), nullable=True)  # ID do post/vídeo/avatar gerado
    operation_type = Column(String(30), nullable=False)  # "text_generation" | "image_generation" | "video_generation" | "tts"
    model_name = Column(String(100), nullable=False)  # Nome do modelo usado
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    credits_consumed = Column(Float, default=0.0)  # Créditos consumidos (calculado)
    meta_info = Column(Text, default="{}")  # JSON com informações adicionais
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PaymentDB(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    mp_payment_id = Column(String(100), unique=True, nullable=False)  # ID do Mercado Pago
    amount = Column(Float, nullable=False)  # Valor em R$
    credits_amount = Column(Float, nullable=False)  # Quantidade de créditos (1 R$ = 1 crédito)
    status = Column(String(20), default="pending")  # pending | approved | rejected | cancelled
    qr_code = Column(Text, nullable=True)  # QRCode base64
    qr_code_data = Column(Text, nullable=True)  # Código PIX copia-cola
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user = relationship("UserDB", back_populates="payments")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="PostGen API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Blob storage helpers
# ---------------------------------------------------------------------------
def _blob_client():
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise HTTPException(status_code=500, detail="Azure Storage não configurado")
    return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)


def upload_bytes_to_blob(data: bytes, blob_name: str, content_type: str = "image/png") -> str:
    client = _blob_client()
    container = client.get_container_client(AZURE_STORAGE_CONTAINER)
    blob = container.get_blob_client(blob_name)
    blob.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type=content_type))
    return blob.url


def upload_file_to_blob(file_path: str, blob_name: str, content_type: str = "image/png") -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    return upload_bytes_to_blob(data, blob_name, content_type)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
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
# Pydantic models
# ---------------------------------------------------------------------------
class UserRegister(BaseModel):
    email: str
    password: str
    name: str


class UserLogin(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    credits_balance: float
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    name: str
    email: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class Settings(BaseModel):
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment_name: str = "gpt-4"
    azure_openai_image_deployment: str = "dall-e-3"
    azure_openai_image_endpoint: str = ""
    azure_openai_api_version: str = "2024-02-01"
    public_base_url: str = "http://localhost:8004"

    class Config:
        from_attributes = True


class Channel(BaseModel):
    id: Optional[str] = None
    name: str
    objective: str
    text_generation_prompt: Optional[str] = None
    image_generation_prompt: Optional[str] = None
    avatar_url: Optional[str] = None
    suggested_image_url: Optional[str] = None
    created_at: Optional[str] = None
    instagram_user_id: Optional[str] = None
    instagram_access_token: Optional[str] = None
    image_model: Optional[str] = "mai"

    class Config:
        from_attributes = True


class GeneratePostRequest(BaseModel):
    channel_id: str
    additional_prompt: Optional[str] = None


class Post(BaseModel):
    id: str = ""
    text: str
    image_url: str


class InsightsOut(BaseModel):
    like_count: int = 0
    comments_count: int = 0
    impressions: Optional[int] = None
    reach: Optional[int] = None
    saved: Optional[int] = None
    shares: Optional[int] = None
    video_views: Optional[int] = None
    total_interactions: int = 0
    engagement_rate: Optional[float] = None
    fetched_at: Optional[str] = None


class DashboardItemOut(BaseModel):
    media_type: str
    media_id: str
    preview_url: str
    text_preview: str
    created_at: str
    published: bool
    insights: InsightsOut


class ChannelDashboardOut(BaseModel):
    channel_id: str
    channel_name: str
    published_count: int
    total_reach: int
    total_impressions: int
    total_interactions: int
    total_likes: int
    total_comments: int
    avg_engagement_rate: Optional[float]
    top_by_reach: List[DashboardItemOut]
    top_by_engagement: List[DashboardItemOut]
    top_by_likes: List[DashboardItemOut]
    top_by_comments: List[DashboardItemOut]
    last_refreshed: Optional[str]


class SavedPost(BaseModel):
    id: str
    channel_id: str
    channel_name: str
    text: str
    image_path: str
    prompt: Optional[str] = None
    ig_media_id: Optional[str] = None
    insights: Optional[InsightsOut] = None
    credits_consumed: float = 0.0
    created_at: str
    published: bool = False

    class Config:
        from_attributes = True


class GenerateAvatarRequest(BaseModel):
    prompt: str
    channel_id: Optional[str] = None


class UpdateAvatarRequest(BaseModel):
    avatar_url: str


class TestInstagramRequest(BaseModel):
    instagram_user_id: Optional[str] = None
    instagram_access_token: Optional[str] = None


class GenerateVideoRequest(BaseModel):
    channel_id: str
    additional_prompt: Optional[str] = None
    seconds: int = 4
    size: str = "720x1280"


class SavedVideo(BaseModel):
    id: str
    channel_id: str
    channel_name: str
    prompt: str
    caption: str = ""
    video_path: str
    duration_seconds: int
    size: str
    credits_consumed: float = 0.0
    created_at: str
    published: bool = False
    is_project_clip: bool = False
    video_project_id: Optional[str] = None
    ig_media_id: Optional[str] = None
    insights: Optional[InsightsOut] = None

    class Config:
        from_attributes = True


class UpdateVideoCaptionRequest(BaseModel):
    caption: str


class VideoProjectOut(BaseModel):
    id: str
    channel_id: str
    title: str
    clips: List[SavedVideo]
    exported_path: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class CreateVideoProjectRequest(BaseModel):
    channel_id: str
    video_id: str


class UpdateVideoProjectClipsRequest(BaseModel):
    clip_ids: List[str]


class GenerateProjectClipRequest(BaseModel):
    additional_prompt: Optional[str] = None
    seconds: int = 4
    size: str = "720x1280"


class AddVideoToProjectRequest(BaseModel):
    video_id: str


class ReferenceImageOut(BaseModel):
    id: int
    channel_id: str
    blob_url: str
    description: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


class AvatarInfo(BaseModel):
    filename: str
    url: str
    created_at: Optional[str] = None


class UpdatePostRequest(BaseModel):
    text: Optional[str] = None
    image_path: Optional[str] = None
    published: Optional[bool] = None


class GeneratePostImageRequest(BaseModel):
    prompt: str
    channel_id: str


class CreditUsageOut(BaseModel):
    id: int
    user_id: int
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    resource_type: str
    resource_id: Optional[str] = None
    operation_type: str
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    credits_consumed: float
    metadata: dict
    created_at: str

    class Config:
        from_attributes = True


class PaymentCreate(BaseModel):
    amount: float  # Valor em R$ (1 R$ = 1 crédito)


class PaymentOut(BaseModel):
    id: int
    user_id: int
    mp_payment_id: str
    amount: float
    credits_amount: float
    status: str
    qr_code: Optional[str] = None
    qr_code_data: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Credit tracking utilities
# ---------------------------------------------------------------------------

# Tabela de custos aproximados por 1000 tokens (em créditos)
# Baseado nos preços da Azure OpenAI e outros serviços
CREDIT_COSTS = {
    # Modelos de texto
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-35-turbo": {"input": 0.5, "output": 1.5},
    
    # Modelos de imagem (custo por imagem)
    "dall-e-3": {"per_image": 40.0},
    "dall-e-2": {"per_image": 20.0},
    "mai": {"per_image": 30.0},  # MAI Image 2e
    "gpt-image-2": {"per_image": 35.0},
    
    # Modelos de vídeo (custo por segundo)
    "sora-2": {"per_second": 50.0},
    
    # TTS (custo por 1000 caracteres)
    "tts": {"per_1k_chars": 15.0},
}


def calculate_credits(
    operation_type: str,
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    images_count: int = 0,
    video_seconds: int = 0,
    text_length: int = 0,
) -> float:
    """
    Calcula créditos consumidos baseado na operação e modelo usado.
    """
    credits = 0.0
    
    # Normalizar nome do modelo
    model_key = model_name.lower()
    for key in CREDIT_COSTS.keys():
        if key in model_key:
            model_key = key
            break
    
    if model_key not in CREDIT_COSTS:
        model_key = "gpt-4o-mini"  # fallback
    
    costs = CREDIT_COSTS[model_key]
    
    if operation_type == "text_generation":
        credits = (input_tokens / 1000.0 * costs.get("input", 0)) + \
                  (output_tokens / 1000.0 * costs.get("output", 0))
    
    elif operation_type == "image_generation":
        credits = images_count * costs.get("per_image", 30.0)
    
    elif operation_type == "video_generation":
        credits = video_seconds * costs.get("per_second", 50.0)
    
    elif operation_type == "tts":
        credits = (text_length / 1000.0) * costs.get("per_1k_chars", 15.0)
    
    return round(credits, 4)


def register_credit_usage(
    db: Session,
    user_id: int,
    channel_id: Optional[str],
    resource_type: str,
    resource_id: Optional[str],
    operation_type: str,
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    images_count: int = 0,
    video_seconds: int = 0,
    text_length: int = 0,
    metadata: dict = None,
) -> float:
    """
    Registra uso de créditos no banco de dados e retorna o valor consumido.
    """
    total_tokens = input_tokens + output_tokens
    
    credits = calculate_credits(
        operation_type=operation_type,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        images_count=images_count,
        video_seconds=video_seconds,
        text_length=text_length,
    )
    
    usage = CreditUsageDB(
        user_id=user_id,
        channel_id=channel_id,
        resource_type=resource_type,
        resource_id=resource_id,
        operation_type=operation_type,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        credits_consumed=credits,
        meta_info=json.dumps(metadata or {}),
    )
    
    db.add(usage)
    db.commit()
    
    return credits


# ---------------------------------------------------------------------------
# Auth utilities
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    return jwt.encode({"sub": email, "exp": expire}, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> UserDB:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        email: str = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Token inválido")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido ou expirado")
    user = db.query(UserDB).filter(UserDB.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Usuário não encontrado")
    return user


# ---------------------------------------------------------------------------
# Settings helpers (per user, seeded from env on first access)
# ---------------------------------------------------------------------------
def _env_defaults() -> dict:
    return {
        "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "azure_openai_api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
        "azure_openai_deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
        "azure_openai_image_deployment": os.getenv("AZURE_OPENAI_IMAGE_DEPLOYMENT", "dall-e-3"),
        "azure_openai_image_endpoint": os.getenv("AZURE_OPENAI_IMAGE_ENDPOINT", ""),
        "azure_openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        "public_base_url": os.getenv("PUBLIC_BASE_URL", "http://localhost:8004"),
    }


def get_or_create_settings(user: UserDB, db: Session) -> SettingsDB:
    s = db.query(SettingsDB).filter(SettingsDB.user_id == user.id).first()
    if not s:
        defaults = _env_defaults()
        s = SettingsDB(user_id=user.id, **defaults)
        db.add(s)
        db.commit()
        db.refresh(s)
    return s


# ---------------------------------------------------------------------------
# Azure OpenAI client helper
# ---------------------------------------------------------------------------
def get_azure_client(s: SettingsDB) -> AzureOpenAI:
    if not s.azure_openai_endpoint or not s.azure_openai_api_key:
        raise HTTPException(status_code=400, detail="Azure OpenAI não configurado")
    return AzureOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,
        api_version=s.azure_openai_api_version,
    )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def save_image_from_base64(base64_data: str, post_id: str) -> str:
    """Upload base64 image to blob storage, return blob URL."""
    if base64_data.startswith("data:image"):
        base64_data = base64_data.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    blob_name = f"posts/{post_id}_{ts}.png"
    return upload_bytes_to_blob(image_bytes, blob_name, "image/png")


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
    )


def _generate_image_bytes(
    prompt: str, ch: Optional["ChannelDB"], s: "SettingsDB", db: Session,
    width: int = 1024, height: int = 1024,
) -> bytes:
    """Route image generation to the correct model for the channel. Falls back to MAI if ch is None."""
    model = (ch.image_model or "mai") if ch else "mai"

    if model == "gpt-image-2":
        if not GPT_IMAGE_2_API_KEY:
            raise HTTPException(status_code=400, detail="GPT_IMAGE_2_API_KEY não configurado no servidor")
        from openai import AzureOpenAI as _AzOAI
        import io as _io
        img_client = _AzOAI(
            azure_endpoint=GPT_IMAGE_2_ENDPOINT,
            api_key=GPT_IMAGE_2_API_KEY,
            api_version="2025-04-01-preview",
        )
        # Use images.edit with reference if available, else plain generate
        refs = db.query(ReferenceImageDB).filter(
            ReferenceImageDB.channel_id == ch.id,
        ).order_by(ReferenceImageDB.created_at.desc()).limit(1).all()

        if refs:
            try:
                ref_bytes = requests.get(refs[0].blob_url, timeout=20).content
                size_str = f"{width}x{height}" if width == height else "1024x1024"
                result = img_client.images.edit(
                    model="gpt-image-2",
                    image=("reference.jpg", _io.BytesIO(ref_bytes), "image/jpeg"),
                    prompt=prompt,
                    n=1,
                    size=size_str,
                )
                return base64.b64decode(result.data[0].b64_json)
            except Exception as e:
                print(f"gpt-image-2 edit failed, falling back to generate: {e}")

        result = img_client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        # Azure AI Foundry returns b64_json by default (no response_format param needed)
        return base64.b64decode(result.data[0].b64_json)

    else:  # MAI
        if not s.azure_openai_image_endpoint:
            raise HTTPException(status_code=400, detail="Endpoint de imagem não configurado")
        resp = requests.post(
            s.azure_openai_image_endpoint,
            headers={"Content-Type": "application/json", "api-key": s.azure_openai_api_key},
            json={"prompt": prompt, "width": width, "height": height, "model": s.azure_openai_image_deployment},
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        if not result.get("data") or "b64_json" not in result["data"][0]:
            raise HTTPException(status_code=500, detail="Sem dados de imagem na resposta")
        return base64.b64decode(result["data"][0]["b64_json"])


def _get_reference_context(channel_id: str, db: Session) -> str:
    """Return a visual reference description string to inject into image prompts."""
    refs = db.query(ReferenceImageDB).filter(
        ReferenceImageDB.channel_id == channel_id,
        ReferenceImageDB.description.isnot(None),
    ).order_by(ReferenceImageDB.created_at.desc()).limit(3).all()
    if not refs:
        return ""
    descriptions = [r.description for r in refs if r.description]
    if not descriptions:
        return ""
    return "\n\nVisual reference for the person/character in this image: " + " ".join(descriptions)


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

    try:
        resp = requests.get(
            f"{api_base}/{ig_media_id}",
            params={"fields": "like_count,comments_count,media_type", "access_token": token},
            timeout=15,
        )
        if resp.ok:
            data = resp.json()
            result["like_count"] = data.get("like_count", 0)
            result["comments_count"] = data.get("comments_count", 0)
            ig_media_type = data.get("media_type", "")
        else:
            print(f"Insights basic fetch failed {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"Insights basic fetch error: {e}")

    metrics = ["impressions", "reach", "saved", "shares"]
    if ig_media_type in ("VIDEO", "REELS"):
        metrics += ["video_views", "plays"]
    try:
        ins_resp = requests.get(
            f"{api_base}/{ig_media_id}/insights",
            params={"metric": ",".join(metrics), "period": "lifetime", "access_token": token},
            timeout=15,
        )
        if ins_resp.ok:
            for item in ins_resp.json().get("data", []):
                name = item.get("name", "")
                val = item.get("value")
                if val is None:
                    vals = item.get("values", [])
                    val = vals[0].get("value", 0) if vals else 0
                if val is None:
                    total = item.get("total_value", {})
                    val = total.get("value", 0) if isinstance(total, dict) else 0
                result[name] = val or 0
        else:
            print(f"Insights advanced fetch failed {ins_resp.status_code}: {ins_resp.text[:200]}")
    except Exception as e:
        print(f"Insights advanced fetch skipped: {e}")

    interactions = (result.get("like_count", 0) + result.get("comments_count", 0) + result.get("saved", 0))
    result["total_interactions"] = interactions
    reach = result.get("reach")
    result["engagement_rate"] = round(interactions / reach * 100, 2) if reach else None

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


def get_channel_or_404(channel_id: str, user: UserDB, db: Session) -> ChannelDB:
    ch = db.query(ChannelDB).filter(
        ChannelDB.id == channel_id,
        ChannelDB.user_id == user.id,
    ).first()
    if not ch:
        raise HTTPException(status_code=404, detail="Canal não encontrado")
    return ch


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "PostGen API is running"}


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
@app.post("/api/auth/register", response_model=TokenOut, status_code=201)
def register(data: UserRegister, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.email == data.email).first():
        raise HTTPException(status_code=409, detail="E-mail já cadastrado")
    user = UserDB(
        email=data.email,
        password_hash=hash_password(data.password),
        name=data.name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.email)
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@app.post("/api/auth/login", response_model=TokenOut)
def login(data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == data.email).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="E-mail ou senha incorretos")
    token = create_access_token(user.email)
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@app.get("/api/auth/me", response_model=UserOut)
def me(current_user: UserDB = Depends(get_current_user)):
    return UserOut.model_validate(current_user)


# ---------------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------------
@app.put("/api/users/profile", response_model=UserOut)
def update_profile(
    data: UserUpdate,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Atualiza o perfil do usuário atual"""
    # Verificar se o email já está em uso por outro usuário
    if data.email != current_user.email:
        existing = db.query(UserDB).filter(
            UserDB.email == data.email,
            UserDB.id != current_user.id
        ).first()
        if existing:
            raise HTTPException(status_code=409, detail="E-mail já está em uso")
    
    current_user.name = data.name
    current_user.email = data.email
    db.commit()
    db.refresh(current_user)
    return UserOut.model_validate(current_user)


@app.get("/api/admin/users", response_model=list[UserOut])
def list_all_users(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Lista todos os usuários (apenas para admin)"""
    # Verificar se é admin
    if current_user.email != "daniel.fabbri@avanade.com":
        raise HTTPException(status_code=403, detail="Acesso negado")
    
    users = db.query(UserDB).order_by(UserDB.created_at.desc()).all()
    return [UserOut.model_validate(u) for u in users]


# ---------------------------------------------------------------------------
# Payment endpoints
# ---------------------------------------------------------------------------
@app.post("/api/payments/create")
def create_payment(
    payment_data: PaymentCreate,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cria uma cobrança PIX no Mercado Pago"""
    if not MERCADOPAGO_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="Mercado Pago não configurado")
    
    try:
        # Inicializar SDK do Mercado Pago
        sdk = mercadopago.SDK(MERCADOPAGO_ACCESS_TOKEN)
        
        # Criar dados do pagamento PIX
        payment_request = {
            "transaction_amount": float(payment_data.amount),
            "description": f"Compra de {payment_data.amount} créditos PostGen",
            "payment_method_id": "pix",
            "payer": {
                "email": current_user.email,
                "first_name": current_user.name.split()[0] if current_user.name else "Cliente",
            },
            "notification_url": f"{BASE_URL}/api/payments/webhook",
        }
        
        # Criar pagamento
        payment_response = sdk.payment().create(payment_request)
        payment = payment_response["response"]
        
        if payment_response["status"] not in [200, 201]:
            raise HTTPException(status_code=500, detail="Erro ao criar pagamento")
        
        # Extrair informações do PIX
        mp_payment_id = str(payment["id"])
        qr_code_base64 = payment.get("point_of_interaction", {}).get("transaction_data", {}).get("qr_code_base64", "")
        qr_code_data = payment.get("point_of_interaction", {}).get("transaction_data", {}).get("qr_code", "")
        
        # Salvar no banco
        db_payment = PaymentDB(
            user_id=current_user.id,
            mp_payment_id=mp_payment_id,
            amount=payment_data.amount,
            credits_amount=payment_data.amount,  # 1 R$ = 1 crédito
            status="pending",
            qr_code=qr_code_base64,
            qr_code_data=qr_code_data,
        )
        db.add(db_payment)
        db.commit()
        db.refresh(db_payment)
        
        return {
            "payment_id": db_payment.id,
            "mp_payment_id": mp_payment_id,
            "amount": payment_data.amount,
            "credits_amount": payment_data.amount,
            "status": "pending",
            "qr_code": qr_code_base64,
            "qr_code_data": qr_code_data,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao criar pagamento: {str(e)}")


@app.post("/api/payments/webhook")
async def payment_webhook(
    request: dict,
    db: Session = Depends(get_db),
):
    """Webhook para receber notificações do Mercado Pago"""
    try:
        # Verificar se é notificação de pagamento
        if request.get("type") != "payment":
            return {"status": "ignored"}
        
        # Obter ID do pagamento
        mp_payment_id = str(request.get("data", {}).get("id", ""))
        if not mp_payment_id:
            return {"status": "error", "message": "Payment ID not found"}
        
        # Buscar pagamento no banco
        payment = db.query(PaymentDB).filter(PaymentDB.mp_payment_id == mp_payment_id).first()
        if not payment:
            return {"status": "error", "message": "Payment not found in database"}
        
        # Consultar status atualizado no Mercado Pago
        sdk = mercadopago.SDK(MERCADOPAGO_ACCESS_TOKEN)
        payment_info = sdk.payment().get(mp_payment_id)
        
        if payment_info["status"] != 200:
            return {"status": "error", "message": "Failed to get payment info"}
        
        mp_status = payment_info["response"].get("status", "")
        
        # Atualizar status no banco
        payment.status = mp_status
        payment.updated_at = func.now()
        
        # Se aprovado, adicionar créditos ao usuário
        if mp_status == "approved" and payment.status != "approved":
            user = db.query(UserDB).filter(UserDB.id == payment.user_id).first()
            if user:
                user.credits_balance += payment.credits_amount
        
        db.commit()
        
        return {"status": "success", "payment_status": mp_status}
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/payments/{payment_id}")
def get_payment_status(
    payment_id: int,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Consulta o status de um pagamento"""
    payment = db.query(PaymentDB).filter(
        PaymentDB.id == payment_id,
        PaymentDB.user_id == current_user.id
    ).first()
    
    if not payment:
        raise HTTPException(status_code=404, detail="Pagamento não encontrado")
    
    # Consultar status atualizado no Mercado Pago
    try:
        sdk = mercadopago.SDK(MERCADOPAGO_ACCESS_TOKEN)
        payment_info = sdk.payment().get(payment.mp_payment_id)
        
        if payment_info["status"] == 200:
            mp_status = payment_info["response"].get("status", payment.status)
            
            # Atualizar status se mudou
            if mp_status != payment.status:
                old_status = payment.status
                payment.status = mp_status
                payment.updated_at = func.now()
                
                # Se aprovado agora, adicionar créditos
                if mp_status == "approved" and old_status != "approved":
                    current_user.credits_balance += payment.credits_amount
                
                db.commit()
                db.refresh(payment)
    except Exception:
        pass  # Se falhar, retorna o status atual do banco
    
    return PaymentOut.model_validate(payment)


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------
@app.get("/api/settings", response_model=Settings)
def get_settings(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = get_or_create_settings(current_user, db)
    result = Settings.model_validate(s)
    result.azure_openai_api_key = "***" if s.azure_openai_api_key else ""
    return result


@app.put("/api/settings")
def update_settings(
    data: Settings,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = get_or_create_settings(current_user, db)
    s.azure_openai_endpoint = data.azure_openai_endpoint
    s.azure_openai_deployment_name = data.azure_openai_deployment_name
    s.azure_openai_image_deployment = data.azure_openai_image_deployment
    s.azure_openai_image_endpoint = data.azure_openai_image_endpoint
    s.azure_openai_api_version = data.azure_openai_api_version
    s.public_base_url = data.public_base_url
    if data.azure_openai_api_key and data.azure_openai_api_key != "***":
        s.azure_openai_api_key = data.azure_openai_api_key
    db.commit()
    return {"message": "Configurações salvas"}


@app.get("/api/test-azure")
def test_azure(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = get_or_create_settings(current_user, db)
    if not s.azure_openai_endpoint or not s.azure_openai_api_key:
        return {"success": False, "error": "Azure OpenAI não configurado"}
    try:
        client = get_azure_client(s)
        resp = client.chat.completions.create(
            model=s.azure_openai_deployment_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
        )
        return {"success": True, "test_response": resp.choices[0].message.content}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
            image_bytes = _generate_image_bytes(avatar_prompt, ch, s, db, width=768, height=768)
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
    ch.text_generation_prompt = data.text_generation_prompt
    ch.image_generation_prompt = data.image_generation_prompt
    ch.avatar_url = data.avatar_url
    ch.suggested_image_url = data.suggested_image_url
    ch.instagram_user_id = data.instagram_user_id
    if data.instagram_access_token and data.instagram_access_token != "***":
        ch.instagram_access_token = data.instagram_access_token
    if data.image_model:
        ch.image_model = data.image_model
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
        full_prompt += _get_reference_context(ch.id, db)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    avatar_filename = f"avatar_{timestamp}.png"

    try:
        image_bytes = _generate_image_bytes(full_prompt, ch, s, db, width=768, height=768)
    except HTTPException:
        raise
    except Exception as e:
        if data.prompt and channel_prompt:
            try:
                fallback = data.prompt + portrait_suffix + (_get_reference_context(ch.id, db) if ch else "")
                image_bytes = _generate_image_bytes(fallback, ch, s, db, width=768, height=768)
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
    model_ready = s.azure_openai_image_endpoint or (ch.image_model == "gpt-image-2" and GPT_IMAGE_2_API_KEY)
    if model_ready:
        image_prompt = ch.image_generation_prompt or f"Instagram post image for {ch.name}. Theme: {ch.objective}. Main subject: {main_subject}"
        if ch.image_generation_prompt:
            image_prompt += f"\n\nItem específico: {main_subject}"
        if data.additional_prompt:
            image_prompt += f"\n\n{data.additional_prompt}"
        image_prompt += _get_reference_context(ch.id, db)
        try:
            img_bytes = _generate_image_bytes(image_prompt, ch, s, db)
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

    return Post(id=post_id, text=post_text, image_url=blob_url)


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
        full_prompt += _get_reference_context(ch.id, db)

    try:
        img_bytes = _generate_image_bytes(full_prompt, ch, s, db)
    except HTTPException:
        raise
    except Exception as e:
        # Content safety fallback: retry with user prompt only
        if data.prompt and channel_prompt:
            try:
                fallback_prompt = data.prompt + (f"\n\n{_get_reference_context(ch.id, db)}" if ch else "")
                img_bytes = _generate_image_bytes(fallback_prompt, ch, s, db)
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
    s = db.query(SettingsDB).filter(SettingsDB.user_id == current_user.id).first()
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
    rates = [i.insights.engagement_rate for i in items if i.insights.engagement_rate is not None]
    avg_rate = round(sum(rates) / len(rates), 2) if rates else None

    top_by_reach = sorted(items, key=lambda x: x.insights.reach or 0, reverse=True)[:5]
    top_by_engagement = sorted(items, key=lambda x: x.insights.engagement_rate or 0.0, reverse=True)[:5]
    top_by_likes = sorted(items, key=lambda x: x.insights.like_count, reverse=True)[:5]
    top_by_comments = sorted(items, key=lambda x: x.insights.comments_count, reverse=True)[:5]

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
        avg_engagement_rate=avg_rate,
        top_by_reach=top_by_reach,
        top_by_engagement=top_by_engagement,
        top_by_likes=top_by_likes,
        top_by_comments=top_by_comments,
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
# Credits endpoints
# ---------------------------------------------------------------------------

@app.get("/api/credits/summary")
def get_credits_summary(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retorna um resumo do consumo de créditos do usuário.
    """
    # Total geral
    total_result = db.execute(
        text("SELECT SUM(credits_consumed) as total FROM credit_usage WHERE user_id = :user_id"),
        {"user_id": current_user.id}
    ).fetchone()
    total_credits = float(total_result[0] or 0.0)
    
    # Por tipo de operação
    by_operation = db.execute(
        text("""
            SELECT operation_type, SUM(credits_consumed) as total
            FROM credit_usage
            WHERE user_id = :user_id
            GROUP BY operation_type
            ORDER BY total DESC
        """),
        {"user_id": current_user.id}
    ).fetchall()
    
    # Por canal
    by_channel = db.execute(
        text("""
            SELECT c.name as channel_name, cu.channel_id, SUM(cu.credits_consumed) as total
            FROM credit_usage cu
            LEFT JOIN channels c ON cu.channel_id = c.id
            WHERE cu.user_id = :user_id AND cu.channel_id IS NOT NULL
            GROUP BY cu.channel_id, c.name
            ORDER BY total DESC
        """),
        {"user_id": current_user.id}
    ).fetchall()
    
    # Por tipo de recurso
    by_resource = db.execute(
        text("""
            SELECT resource_type, SUM(credits_consumed) as total
            FROM credit_usage
            WHERE user_id = :user_id
            GROUP BY resource_type
            ORDER BY total DESC
        """),
        {"user_id": current_user.id}
    ).fetchall()
    
    # Últimos 30 dias
    last_30_days = db.execute(
        text("""
            SELECT DATE(created_at) as date, SUM(credits_consumed) as total
            FROM credit_usage
            WHERE user_id = :user_id AND created_at >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """),
        {"user_id": current_user.id}
    ).fetchall()
    
    return {
        "total_credits": total_credits,
        "by_operation": [{"operation_type": row[0], "credits": float(row[1] or 0.0)} for row in by_operation],
        "by_channel": [{"channel_id": row[1], "channel_name": row[0], "credits": float(row[2] or 0.0)} for row in by_channel],
        "by_resource": [{"resource_type": row[0], "credits": float(row[1] or 0.0)} for row in by_resource],
        "last_30_days": [{"date": str(row[0]), "credits": float(row[1] or 0.0)} for row in last_30_days],
    }


@app.get("/api/credits/log", response_model=List[CreditUsageOut])
def get_credits_log(
    channel_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retorna o log detalhado de consumo de créditos.
    """
    query = db.query(CreditUsageDB).filter(CreditUsageDB.user_id == current_user.id)
    
    if channel_id:
        query = query.filter(CreditUsageDB.channel_id == channel_id)
    
    if resource_type:
        query = query.filter(CreditUsageDB.resource_type == resource_type)
    
    usage_records = query.order_by(CreditUsageDB.created_at.desc()).limit(limit).all()
    
    # Get channel names
    channel_ids = list(set([r.channel_id for r in usage_records if r.channel_id]))
    channel_names = {}
    if channel_ids:
        channels = db.query(ChannelDB.id, ChannelDB.name).filter(ChannelDB.id.in_(channel_ids)).all()
        channel_names = {ch.id: ch.name for ch in channels}
    
    result = []
    for record in usage_records:
        result.append(CreditUsageOut(
            id=record.id,
            user_id=record.user_id,
            channel_id=record.channel_id,
            channel_name=channel_names.get(record.channel_id),
            resource_type=record.resource_type,
            resource_id=record.resource_id,
            operation_type=record.operation_type,
            model_name=record.model_name,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            total_tokens=record.total_tokens,
            credits_consumed=record.credits_consumed,
            metadata=json.loads(record.meta_info or "{}"),
            created_at=record.created_at.isoformat(),
        ))
    
    return result


@app.get("/api/credits/channel/{channel_id}")
def get_channel_credits(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retorna estatísticas de créditos para um canal específico.
    """
    ch = get_channel_or_404(channel_id, current_user, db)
    
    # Total do canal
    total_result = db.execute(
        text("SELECT SUM(credits_consumed) as total FROM credit_usage WHERE channel_id = :channel_id"),
        {"channel_id": channel_id}
    ).fetchone()
    total_credits = float(total_result[0] or 0.0)
    
    # Por tipo de operação
    by_operation = db.execute(
        text("""
            SELECT operation_type, SUM(credits_consumed) as total, COUNT(*) as count
            FROM credit_usage
            WHERE channel_id = :channel_id
            GROUP BY operation_type
            ORDER BY total DESC
        """),
        {"channel_id": channel_id}
    ).fetchall()
    
    # Últimos recursos criados com créditos
    recent_resources = db.execute(
        text("""
            SELECT resource_type, resource_id, SUM(credits_consumed) as total, MAX(created_at) as created_at
            FROM credit_usage
            WHERE channel_id = :channel_id AND resource_id IS NOT NULL
            GROUP BY resource_type, resource_id
            ORDER BY created_at DESC
            LIMIT 20
        """),
        {"channel_id": channel_id}
    ).fetchall()
    
    return {
        "channel_id": channel_id,
        "channel_name": ch.name,
        "total_credits": total_credits,
        "by_operation": [
            {
                "operation_type": row[0],
                "credits": float(row[1] or 0.0),
                "count": int(row[2])
            } for row in by_operation
        ],
        "recent_resources": [
            {
                "resource_type": row[0],
                "resource_id": row[1],
                "credits": float(row[2] or 0.0),
                "created_at": row[3].isoformat() if row[3] else None
            } for row in recent_resources
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

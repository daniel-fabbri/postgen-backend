from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import os
import base64
import shutil
import requests
from datetime import datetime, timedelta, timezone
from openai import AzureOpenAI
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, String, Text, Boolean, Integer,
    DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.sql import func
from passlib.context import CryptContext
from jose import JWTError, jwt
from azure.storage.blob import BlobServiceClient, ContentSettings

load_dotenv()

# ---------------------------------------------------------------------------
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
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    channels = relationship("ChannelDB", back_populates="user", cascade="all, delete-orphan")
    settings = relationship("SettingsDB", back_populates="user", uselist=False, cascade="all, delete-orphan")


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
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("UserDB", back_populates="channels")
    posts = relationship("PostDB", back_populates="channel", cascade="all, delete-orphan")
    videos = relationship("VideoDB", back_populates="channel", cascade="all, delete-orphan")
    avatars = relationship("AvatarDB", back_populates="channel")


class PostDB(Base):
    __tablename__ = "posts"
    id = Column(String(100), primary_key=True)
    channel_id = Column(String(50), ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    channel_name = Column(String(255), nullable=False)
    text = Column(Text, default="")
    image_path = Column(Text, default="")
    published = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    channel = relationship("ChannelDB", back_populates="posts")


class VideoDB(Base):
    __tablename__ = "videos"
    id = Column(String(100), primary_key=True)
    channel_id = Column(String(50), ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    channel_name = Column(String(255), nullable=False)
    prompt = Column(Text, default="")
    video_path = Column(Text, default="")
    duration_seconds = Column(Integer, default=4)
    size = Column(String(20), default="720x1280")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    published = Column(Boolean, default=False)
    channel = relationship("ChannelDB", back_populates="videos")


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
    created_at: datetime

    class Config:
        from_attributes = True


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

    class Config:
        from_attributes = True


class GeneratePostRequest(BaseModel):
    channel_id: str
    additional_prompt: Optional[str] = None


class Post(BaseModel):
    id: str = ""
    text: str
    image_url: str


class SavedPost(BaseModel):
    id: str
    channel_id: str
    channel_name: str
    text: str
    image_path: str
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
    video_path: str
    duration_seconds: int
    size: str
    created_at: str
    published: bool = False

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
    blob_name = f"posts/{post_id}.png"
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
    )


def post_to_schema(p: PostDB) -> SavedPost:
    return SavedPost(
        id=p.id,
        channel_id=p.channel_id,
        channel_name=p.channel_name,
        text=p.text or "",
        image_path=p.image_path or "",
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
            if s.azure_openai_image_endpoint:
                avatar_prompt = (
                    "\n".join(ch.image_generation_prompt.splitlines()[:5])
                    + "\n\nFrame: Close-up portrait style, profile picture format."
                )
                resp = requests.post(
                    s.azure_openai_image_endpoint,
                    headers={"Content-Type": "application/json", "api-key": s.azure_openai_api_key},
                    json={"prompt": avatar_prompt, "width": 768, "height": 768, "model": s.azure_openai_image_deployment},
                )
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("data") and "b64_json" in result["data"][0]:
                        avatar_filename = f"{ch.id}.png"
                        image_bytes = base64.b64decode(result["data"][0]["b64_json"])
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
            f"https://graph.facebook.com/v21.0/{user_id}",
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
    if not s.azure_openai_image_endpoint:
        raise HTTPException(status_code=400, detail="Endpoint de imagem não configurado")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    avatar_filename = f"avatar_{timestamp}.png"

    resp = requests.post(
        s.azure_openai_image_endpoint,
        headers={"Content-Type": "application/json", "api-key": s.azure_openai_api_key},
        json={
            "prompt": data.prompt + "\n\nFrame: Close-up portrait style, profile picture format.",
            "width": 768, "height": 768, "model": s.azure_openai_image_deployment,
        },
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Falha ao gerar avatar: {resp.text}")

    result = resp.json()
    if not result.get("data") or "b64_json" not in result["data"][0]:
        raise HTTPException(status_code=500, detail="Sem dados de imagem na resposta")

    image_bytes = base64.b64decode(result["data"][0]["b64_json"])
    avatar_url = upload_bytes_to_blob(image_bytes, f"avatars/{avatar_filename}", "image/png")

    if data.channel_id:
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
    return [post_to_schema(p) for p in posts]


@app.get("/api/posts/{post_id}", response_model=SavedPost)
def get_post(
    post_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = _get_post_or_404(post_id, current_user, db)
    return post_to_schema(p)


@app.patch("/api/posts/{post_id}", response_model=SavedPost)
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

    # Generate image
    image_url = f"https://placehold.co/1024x1024/4F46E5/FFFFFF?text=PostGen"
    if s.azure_openai_image_endpoint:
        image_prompt = ch.image_generation_prompt or f"Instagram post image for {ch.name}. Theme: {ch.objective}. Main subject: {main_subject}"
        if ch.image_generation_prompt:
            image_prompt += f"\n\nItem específico: {main_subject}"
        if data.additional_prompt:
            image_prompt += f"\n\n{data.additional_prompt}"
        try:
            img_resp = requests.post(
                s.azure_openai_image_endpoint,
                headers={"Content-Type": "application/json", "api-key": s.azure_openai_api_key},
                json={"prompt": image_prompt, "width": 1024, "height": 1024, "model": s.azure_openai_image_deployment},
                timeout=60,
            )
            img_resp.raise_for_status()
            img_result = img_resp.json()
            if img_result.get("data") and "b64_json" in img_result["data"][0]:
                image_url = f"data:image/png;base64,{img_result['data'][0]['b64_json']}"
        except Exception as e:
            print(f"Image generation failed: {e}")

    # Save post — upload image to blob, store blob URL
    post_id = f"post_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    blob_url = ""
    if image_url.startswith("data:image"):
        try:
            blob_url = save_image_from_base64(image_url, post_id)
        except Exception as e:
            print(f"Error uploading image to blob: {e}")
            blob_url = image_url  # fallback: keep base64
    else:
        blob_url = image_url

    p = PostDB(
        id=post_id,
        channel_id=ch.id,
        channel_name=ch.name,
        text=post_text,
        image_path=blob_url,
        published=False,
    )
    db.add(p)
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
    blob_url = upload_bytes_to_blob(data, f"posts/{post_id}.png", file.content_type or "image/png")
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

    resp = requests.post(
        s.azure_openai_image_endpoint,
        headers={"Content-Type": "application/json", "api-key": s.azure_openai_api_key},
        json={"prompt": data.prompt, "width": 1024, "height": 1024, "model": s.azure_openai_image_deployment},
        timeout=60,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Falha ao gerar imagem: {resp.text}")

    result = resp.json()
    if not result.get("data") or "b64_json" not in result["data"][0]:
        raise HTTPException(status_code=500, detail="Sem dados de imagem na resposta")

    blob_url = save_image_from_base64(result["data"][0]["b64_json"], post_id)
    p.image_path = blob_url
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
        create_resp = requests.post(
            f"https://graph.facebook.com/v21.0/{ch.instagram_user_id}/media",
            params={"image_url": image_url, "caption": p.text, "access_token": ch.instagram_access_token},
            timeout=30,
        )
        create_data = create_resp.json()
        if create_resp.status_code != 200 or "id" not in create_data:
            error_msg = create_data.get("error", {}).get("message", create_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao criar container: {error_msg}")

        pub_resp = requests.post(
            f"https://graph.facebook.com/v21.0/{ch.instagram_user_id}/media_publish",
            params={"creation_id": create_data["id"], "access_token": ch.instagram_access_token},
            timeout=30,
        )
        pub_data = pub_resp.json()
        if pub_resp.status_code != 200 or "id" not in pub_data:
            error_msg = pub_data.get("error", {}).get("message", pub_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao publicar: {error_msg}")

        p.published = True
        db.commit()
        return {"success": True, "instagram_post_id": pub_data["id"]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {str(e)}")


# ---------------------------------------------------------------------------
# Videos endpoints
# ---------------------------------------------------------------------------
def video_to_schema(v: VideoDB) -> SavedVideo:
    return SavedVideo(
        id=v.id,
        channel_id=v.channel_id,
        channel_name=v.channel_name,
        prompt=v.prompt or "",
        video_path=v.video_path or "",
        duration_seconds=v.duration_seconds or 4,
        size=v.size or "720x1280",
        created_at=v.created_at.isoformat() if v.created_at else datetime.now().isoformat(),
        published=v.published or False,
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
    q = db.query(VideoDB).filter(VideoDB.channel_id.in_(user_channel_ids))
    if channel_id:
        q = q.filter(VideoDB.channel_id == channel_id)
    return [video_to_schema(v) for v in q.order_by(VideoDB.created_at.desc()).all()]


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

    # Create Sora job
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
        create_resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Falha ao criar job Sora: {str(e)}")

    job = create_resp.json()
    print(f"Sora job created: {job}")
    job_id = job.get("id") or job.get("job_id") or job.get("generation_id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"Resposta inesperada do Sora: {job}")

    # Poll until complete (max 3 minutes)
    poll_url = f"{AZURE_SORA_ENDPOINT}/{job_id}"
    deadline = datetime.now().timestamp() + 180
    video_url = None
    while datetime.now().timestamp() < deadline:
        import time
        time.sleep(5)
        try:
            poll_resp = requests.get(poll_url, headers=_sora_headers(), timeout=15)
            result = poll_resp.json()
            print(f"Sora poll: {result.get('status', result)}")
        except Exception as e:
            print(f"Sora poll error: {e}")
            continue

        status = result.get("status", "")
        if status in ("succeeded", "completed", "done"):
            # Extract video URL from various possible response shapes
            gens = result.get("generations") or result.get("outputs") or []
            if gens:
                video_url = gens[0].get("url") or gens[0].get("video_url")
            if not video_url:
                video_url = result.get("url") or result.get("video_url") or result.get("result", {}).get("url")
            break
        if status in ("failed", "error", "cancelled"):
            err = result.get("error", {}).get("message") or result.get("message") or "Job falhou"
            raise HTTPException(status_code=502, detail=f"Sora falhou: {err}")

    if not video_url:
        raise HTTPException(status_code=504, detail="Timeout aguardando o Sora. Tente novamente.")

    # Download video and upload to blob
    video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    try:
        dl = requests.get(video_url, timeout=60, stream=True)
        dl.raise_for_status()
        video_bytes = dl.content
        blob_url = upload_bytes_to_blob(video_bytes, f"videos/{video_id}.mp4", "video/mp4")
    except Exception as e:
        # If download fails (expired URL), store the original URL temporarily
        print(f"Video download failed: {e}. Storing original URL.")
        blob_url = video_url

    v = VideoDB(
        id=video_id,
        channel_id=ch.id,
        channel_name=ch.name,
        prompt=prompt,
        video_path=blob_url,
        duration_seconds=data.seconds,
        size=data.size,
        published=False,
    )
    db.add(v)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

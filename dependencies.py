import os
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from openai import AzureOpenAI
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_DAYS, ADMIN_EMAIL
from database import get_db
from models import UserDB, ChannelDB, SettingsDB

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRE_DAYS)
    return jwt.encode({"sub": email, "exp": expire}, JWT_SECRET, algorithm=JWT_ALGORITHM)


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


def _require_admin(current_user: UserDB = Depends(get_current_user)):
    if current_user.email != ADMIN_EMAIL:
        raise HTTPException(status_code=403, detail="Acesso restrito a administradores")
    return current_user


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


def get_azure_client(s: SettingsDB) -> AzureOpenAI:
    if not s.azure_openai_endpoint or not s.azure_openai_api_key:
        raise HTTPException(status_code=400, detail="Azure OpenAI não configurado")
    return AzureOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,
        api_version=s.azure_openai_api_version,
    )


def get_channel_or_404(channel_id: str, user: UserDB, db: Session) -> ChannelDB:
    ch = db.query(ChannelDB).filter(
        ChannelDB.id == channel_id,
        ChannelDB.user_id == user.id,
    ).first()
    if not ch:
        raise HTTPException(status_code=404, detail="Canal não encontrado")
    return ch

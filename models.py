from sqlalchemy import (
    Column, String, Text, Boolean, Integer, Float,
    DateTime, ForeignKey,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    credits_balance = Column(Float, default=0.0)
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
    image_model = Column(String(20), default="mai")
    auto_reply_enabled = Column(Boolean, default=False)
    auto_reply_prompt = Column(Text, nullable=True)
    lora_training_id = Column(String(100), nullable=True)
    lora_status = Column(String(20), nullable=True)
    lora_version = Column(String(200), nullable=True)
    elevenlabs_voice_id = Column(String(100), nullable=True)
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
    clip_urls = Column(Text, default="{}")
    root_video_id = Column(String(100), nullable=True)
    exported_video_id = Column(String(100), nullable=True)
    exported_path = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class MediaInsightsDB(Base):
    __tablename__ = "media_insights"
    id = Column(Integer, primary_key=True, autoincrement=True)
    media_type = Column(String(10), nullable=False)
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
    engagement_rate = Column(Float, nullable=True)
    fetched_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ReferenceImageDB(Base):
    __tablename__ = "reference_images"
    id = Column(Integer, primary_key=True, autoincrement=True)
    channel_id = Column(String(50), ForeignKey("channels.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    blob_url = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
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
    resource_type = Column(String(20), nullable=False)
    resource_id = Column(String(100), nullable=True)
    operation_type = Column(String(30), nullable=False)
    model_name = Column(String(100), nullable=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    credits_consumed = Column(Float, default=0.0)
    meta_info = Column(Text, default="{}")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class VideoJobDB(Base):
    """Job assíncrono para geração de vídeo com personagem (evita timeout 240s do Container Apps)."""
    __tablename__ = "video_jobs"
    id = Column(String(50), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    channel_id = Column(String(50), nullable=True)
    status = Column(String(20), default="processing")  # processing / completed / failed
    video_id = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SystemConfigDB(Base):
    __tablename__ = "system_config"
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PaymentDB(Base):
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    mp_payment_id = Column(String(100), unique=True, nullable=False)
    amount = Column(Float, nullable=False)
    credits_amount = Column(Float, nullable=False)
    status = Column(String(20), default="pending")
    qr_code = Column(Text, nullable=True)
    qr_code_data = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user = relationship("UserDB", back_populates="payments")

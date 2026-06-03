from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


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
    auto_reply_enabled: Optional[bool] = False
    auto_reply_prompt: Optional[str] = None
    lora_status: Optional[str] = None
    lora_version: Optional[str] = None
    elevenlabs_voice_id: Optional[str] = None

    class Config:
        from_attributes = True


class GeneratePostRequest(BaseModel):
    channel_id: str
    additional_prompt: Optional[str] = None


class Post(BaseModel):
    id: str = ""
    text: str
    image_url: str
    image_error: Optional[str] = None


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
    total_saved: int
    total_shares: int
    avg_engagement_rate: Optional[float]
    top_by_reach: List[DashboardItemOut]
    top_by_engagement: List[DashboardItemOut]
    top_by_likes: List[DashboardItemOut]
    top_by_comments: List[DashboardItemOut]
    top_by_saved: List[DashboardItemOut]
    top_by_shares: List[DashboardItemOut]
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
    voice_script: Optional[str] = None


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


class VideoJobOut(BaseModel):
    job_id: str
    status: str  # processing / completed / failed
    video: Optional[SavedVideo] = None
    error: Optional[str] = None


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
    amount: float


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

"""Pure conversion functions: ORM models → Pydantic schemas."""
import json
from datetime import datetime
from typing import Optional

from models import ChannelDB, PostDB, VideoDB, VideoProjectDB
from schemas import Channel, InsightsOut, SavedPost, SavedVideo, VideoProjectOut


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
        lora_status=ch.lora_status,
        lora_version=ch.lora_version,
        elevenlabs_voice_id=ch.elevenlabs_voice_id,
    )


def insights_to_schema(ins) -> Optional[InsightsOut]:
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
        insights=insights_to_schema(insights),
        credits_consumed=getattr(p, "credits_consumed", 0.0),
        created_at=p.created_at.isoformat() if p.created_at else datetime.now().isoformat(),
        published=p.published or False,
    )


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
        insights=insights_to_schema(insights),
    )


def video_project_to_schema(vp: VideoProjectDB, db) -> VideoProjectOut:
    from models import VideoDB as _VideoDB
    try:
        clip_ids = json.loads(vp.clip_ids or "[]")
    except Exception:
        clip_ids = []
    clips = []
    for cid in clip_ids:
        v = db.query(_VideoDB).filter(_VideoDB.id == cid).first()
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

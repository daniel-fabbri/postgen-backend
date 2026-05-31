from typing import List

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from database import get_db
from dependencies import get_current_user, get_channel_or_404, get_post_or_404, get_video_or_404
from models import UserDB, ChannelDB, PostDB, VideoDB, MediaInsightsDB
from schemas import InsightsOut, DashboardItemOut, ChannelDashboardOut
from services.converters import insights_to_schema
from services.instagram import fetch_and_store_insights, insights_stale

router = APIRouter(tags=["insights"])


@router.get("/api/posts/{post_id}/insights", response_model=InsightsOut)
def get_post_insights(
    post_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = get_post_or_404(post_id, current_user, db)
    if not p.published or not getattr(p, "ig_media_id", None):
        raise HTTPException(status_code=404, detail="Post não publicado ou sem ID do Instagram")
    ch = db.query(ChannelDB).filter(ChannelDB.id == p.channel_id).first()
    if not ch or not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não conectado")
    ins = db.query(MediaInsightsDB).filter(
        MediaInsightsDB.media_type == "post", MediaInsightsDB.media_id == post_id,
    ).first()
    if insights_stale(ins, p.created_at):
        ins = fetch_and_store_insights("post", post_id, p.ig_media_id, p.channel_id, ch.instagram_access_token, db)
    return insights_to_schema(ins)


@router.post("/api/posts/{post_id}/insights/refresh", response_model=InsightsOut)
def refresh_post_insights(
    post_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = get_post_or_404(post_id, current_user, db)
    if not p.published or not getattr(p, "ig_media_id", None):
        raise HTTPException(status_code=404, detail="Post não publicado ou sem ID do Instagram")
    ch = db.query(ChannelDB).filter(ChannelDB.id == p.channel_id).first()
    if not ch or not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não conectado")
    return insights_to_schema(
        fetch_and_store_insights("post", post_id, p.ig_media_id, p.channel_id, ch.instagram_access_token, db)
    )


@router.get("/api/videos/{video_id}/insights", response_model=InsightsOut)
def get_video_insights(
    video_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    v = get_video_or_404(video_id, current_user, db)
    if not v.published or not getattr(v, "ig_media_id", None):
        raise HTTPException(status_code=404, detail="Vídeo não publicado ou sem ID do Instagram")
    ch = db.query(ChannelDB).filter(ChannelDB.id == v.channel_id).first()
    if not ch or not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não conectado")
    ins = db.query(MediaInsightsDB).filter(
        MediaInsightsDB.media_type == "video", MediaInsightsDB.media_id == video_id,
    ).first()
    if insights_stale(ins, v.created_at):
        ins = fetch_and_store_insights("video", video_id, v.ig_media_id, v.channel_id, ch.instagram_access_token, db)
    return insights_to_schema(ins)


@router.post("/api/videos/{video_id}/insights/refresh", response_model=InsightsOut)
def refresh_video_insights(
    video_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    v = get_video_or_404(video_id, current_user, db)
    if not v.published or not getattr(v, "ig_media_id", None):
        raise HTTPException(status_code=404, detail="Vídeo não publicado ou sem ID do Instagram")
    ch = db.query(ChannelDB).filter(ChannelDB.id == v.channel_id).first()
    if not ch or not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não conectado")
    return insights_to_schema(
        fetch_and_store_insights("video", video_id, v.ig_media_id, v.channel_id, ch.instagram_access_token, db)
    )


@router.get("/api/channels/{channel_id}/dashboard", response_model=ChannelDashboardOut)
def get_channel_dashboard(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)

    posts = db.query(PostDB).filter(
        PostDB.channel_id == channel_id, PostDB.published == True, PostDB.ig_media_id.isnot(None),
    ).all()
    videos = db.query(VideoDB).filter(
        VideoDB.channel_id == channel_id, VideoDB.published == True,
        VideoDB.ig_media_id.isnot(None), VideoDB.is_project_clip.is_not(True),
    ).all()

    insights_map = {}
    post_ids = [p.id for p in posts]
    video_ids = [v.id for v in videos]
    if post_ids:
        for ins in db.query(MediaInsightsDB).filter(
            MediaInsightsDB.media_type == "post", MediaInsightsDB.media_id.in_(post_ids),
        ).all():
            insights_map[("post", ins.media_id)] = ins
    if video_ids:
        for ins in db.query(MediaInsightsDB).filter(
            MediaInsightsDB.media_type == "video", MediaInsightsDB.media_id.in_(video_ids),
        ).all():
            insights_map[("video", ins.media_id)] = ins

    items = []
    for p in posts:
        ins = insights_map.get(("post", p.id))
        if ins:
            items.append(DashboardItemOut(
                media_type="post", media_id=p.id,
                preview_url=p.image_path or "", text_preview=(p.text or "")[:120],
                created_at=p.created_at.isoformat(), published=True,
                insights=insights_to_schema(ins),
            ))
    for v in videos:
        ins = insights_map.get(("video", v.id))
        if ins:
            items.append(DashboardItemOut(
                media_type="video", media_id=v.id,
                preview_url=v.video_path or "", text_preview=(v.caption or v.prompt or "")[:120],
                created_at=v.created_at.isoformat(), published=True,
                insights=insights_to_schema(ins),
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

    all_ins = list(insights_map.values())
    last_refreshed = None
    if all_ins:
        valid = [ins.fetched_at for ins in all_ins if ins.fetched_at]
        if valid:
            last_refreshed = max(valid).isoformat()

    return ChannelDashboardOut(
        channel_id=ch.id, channel_name=ch.name,
        published_count=len(posts) + len(videos),
        total_reach=total_reach, total_impressions=total_impressions,
        total_interactions=total_interactions, total_likes=total_likes,
        total_comments=total_comments, total_saved=total_saved, total_shares=total_shares,
        avg_engagement_rate=avg_rate,
        top_by_reach=sorted(items, key=lambda x: x.insights.reach or 0, reverse=True)[:5],
        top_by_engagement=sorted(items, key=lambda x: x.insights.engagement_rate or 0.0, reverse=True)[:5],
        top_by_likes=sorted(items, key=lambda x: x.insights.like_count, reverse=True)[:5],
        top_by_comments=sorted(items, key=lambda x: x.insights.comments_count, reverse=True)[:5],
        top_by_saved=sorted(items, key=lambda x: x.insights.saved or 0, reverse=True)[:5],
        top_by_shares=sorted(items, key=lambda x: x.insights.shares or 0, reverse=True)[:5],
        last_refreshed=last_refreshed,
    )


@router.post("/api/channels/{channel_id}/insights/refresh")
def refresh_channel_insights(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)
    if not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não conectado a este canal")

    posts = db.query(PostDB).filter(
        PostDB.channel_id == channel_id, PostDB.published == True, PostDB.ig_media_id.isnot(None),
    ).all()
    videos = db.query(VideoDB).filter(
        VideoDB.channel_id == channel_id, VideoDB.published == True,
        VideoDB.ig_media_id.isnot(None), VideoDB.is_project_clip.is_not(True),
    ).all()

    refreshed, errors = 0, 0
    for p in posts:
        try:
            fetch_and_store_insights("post", p.id, p.ig_media_id, channel_id, ch.instagram_access_token, db)
            refreshed += 1
        except Exception as e:
            print(f"Refresh error post {p.id}: {e}")
            errors += 1
    for v in videos:
        try:
            fetch_and_store_insights("video", v.id, v.ig_media_id, channel_id, ch.instagram_access_token, db)
            refreshed += 1
        except Exception as e:
            print(f"Refresh error video {v.id}: {e}")
            errors += 1

    return {"refreshed": refreshed, "errors": errors, "total": len(posts) + len(videos)}

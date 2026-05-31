"""Instagram Graph API helpers: insights fetching, TTL, base URL."""
import traceback
from datetime import datetime, timedelta, timezone

import requests
from sqlalchemy.orm import Session

from models import MediaInsightsDB


def ig_api_base(token: str) -> str:
    if token and token.startswith("IG"):
        return "https://graph.instagram.com/v21.0"
    return "https://graph.facebook.com/v21.0"


def insights_ttl(published_at: datetime) -> timedelta:
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


def insights_stale(ins, published_at: datetime) -> bool:
    if not ins or not ins.fetched_at:
        return True
    ttl = insights_ttl(published_at)
    now = datetime.now(timezone.utc)
    fetched = ins.fetched_at.replace(tzinfo=timezone.utc) if not ins.fetched_at.tzinfo else ins.fetched_at
    return (now - fetched) > ttl


def fetch_and_store_insights(
    media_type: str, media_id: str, ig_media_id: str,
    channel_id: str, token: str, db: Session,
):
    result = {}
    ig_media_type = None
    api_base = ig_api_base(token)

    print(f"\n[INSIGHTS DEBUG] Fetching insights for {media_type} {media_id} (IG: {ig_media_id})")
    print(f"[INSIGHTS DEBUG] API Base: {api_base}")

    try:
        url = f"{api_base}/{ig_media_id}"
        params = {"fields": "like_count,comments_count,media_type", "access_token": token}
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
        traceback.print_exc()

    metrics_to_try = [["reach", "saved"], ["shares"]]

    for metric_group in metrics_to_try:
        try:
            ins_url = f"{api_base}/{ig_media_id}/insights"
            ins_params = {"metric": ",".join(metric_group), "period": "lifetime", "access_token": token}
            ins_resp = requests.get(ins_url, params=ins_params, timeout=15)
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
                print(f"[INSIGHTS DEBUG] Group {metric_group} failed {ins_resp.status_code}: {ins_resp.text[:200]}")
        except Exception as e:
            print(f"[INSIGHTS DEBUG] Exception trying {metric_group}: {e}")

    interactions = result.get("like_count", 0) + result.get("comments_count", 0) + result.get("saved", 0)
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

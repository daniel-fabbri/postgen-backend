import json
from typing import Optional, List

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_db
from dependencies import get_current_user, get_channel_or_404
from models import UserDB, CreditUsageDB, ChannelDB
from schemas import CreditUsageOut

router = APIRouter(prefix="/api/credits", tags=["credits"])


@router.get("/summary")
def get_credits_summary(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    total_result = db.execute(
        text("SELECT SUM(credits_consumed) as total FROM credit_usage WHERE user_id = :user_id"),
        {"user_id": current_user.id},
    ).fetchone()
    total_credits = float(total_result[0] or 0.0)

    by_operation = db.execute(
        text("""
            SELECT operation_type, SUM(credits_consumed) as total
            FROM credit_usage
            WHERE user_id = :user_id
            GROUP BY operation_type
            ORDER BY total DESC
        """),
        {"user_id": current_user.id},
    ).fetchall()

    by_channel = db.execute(
        text("""
            SELECT c.name as channel_name, cu.channel_id, SUM(cu.credits_consumed) as total
            FROM credit_usage cu
            LEFT JOIN channels c ON cu.channel_id = c.id
            WHERE cu.user_id = :user_id AND cu.channel_id IS NOT NULL
            GROUP BY cu.channel_id, c.name
            ORDER BY total DESC
        """),
        {"user_id": current_user.id},
    ).fetchall()

    by_resource = db.execute(
        text("""
            SELECT resource_type, SUM(credits_consumed) as total
            FROM credit_usage
            WHERE user_id = :user_id
            GROUP BY resource_type
            ORDER BY total DESC
        """),
        {"user_id": current_user.id},
    ).fetchall()

    last_30_days = db.execute(
        text("""
            SELECT DATE(created_at) as date, SUM(credits_consumed) as total
            FROM credit_usage
            WHERE user_id = :user_id AND created_at >= NOW() - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """),
        {"user_id": current_user.id},
    ).fetchall()

    return {
        "total_credits": total_credits,
        "credits_balance": current_user.credits_balance,
        "by_operation": [{"operation_type": row[0], "credits": float(row[1] or 0.0)} for row in by_operation],
        "by_channel": [{"channel_id": row[1], "channel_name": row[0], "credits": float(row[2] or 0.0)} for row in by_channel],
        "by_resource": [{"resource_type": row[0], "credits": float(row[1] or 0.0)} for row in by_resource],
        "last_30_days": [{"date": str(row[0]), "credits": float(row[1] or 0.0)} for row in last_30_days],
    }


@router.get("/log", response_model=List[CreditUsageOut])
def get_credits_log(
    channel_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(CreditUsageDB).filter(CreditUsageDB.user_id == current_user.id)
    if channel_id:
        query = query.filter(CreditUsageDB.channel_id == channel_id)
    if resource_type:
        query = query.filter(CreditUsageDB.resource_type == resource_type)
    usage_records = query.order_by(CreditUsageDB.created_at.desc()).limit(limit).all()

    channel_ids = list(set([r.channel_id for r in usage_records if r.channel_id]))
    channel_names = {}
    if channel_ids:
        channels = db.query(ChannelDB.id, ChannelDB.name).filter(ChannelDB.id.in_(channel_ids)).all()
        channel_names = {ch.id: ch.name for ch in channels}

    return [
        CreditUsageOut(
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
        )
        for record in usage_records
    ]


@router.get("/channel/{channel_id}")
def get_channel_credits(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)

    total_result = db.execute(
        text("SELECT SUM(credits_consumed) as total FROM credit_usage WHERE channel_id = :channel_id"),
        {"channel_id": channel_id},
    ).fetchone()
    total_credits = float(total_result[0] or 0.0)

    by_operation = db.execute(
        text("""
            SELECT operation_type, SUM(credits_consumed) as total, COUNT(*) as count
            FROM credit_usage
            WHERE channel_id = :channel_id
            GROUP BY operation_type
            ORDER BY total DESC
        """),
        {"channel_id": channel_id},
    ).fetchall()

    recent_resources = db.execute(
        text("""
            SELECT resource_type, resource_id, SUM(credits_consumed) as total, MAX(created_at) as created_at
            FROM credit_usage
            WHERE channel_id = :channel_id AND resource_id IS NOT NULL
            GROUP BY resource_type, resource_id
            ORDER BY created_at DESC
            LIMIT 20
        """),
        {"channel_id": channel_id},
    ).fetchall()

    return {
        "channel_id": channel_id,
        "channel_name": ch.name,
        "total_credits": total_credits,
        "by_operation": [
            {"operation_type": row[0], "credits": float(row[1] or 0.0), "count": int(row[2])}
            for row in by_operation
        ],
        "recent_resources": [
            {
                "resource_type": row[0],
                "resource_id": row[1],
                "credits": float(row[2] or 0.0),
                "created_at": row[3].isoformat() if row[3] else None,
            }
            for row in recent_resources
        ],
    }

import json
from typing import Optional

from sqlalchemy.orm import Session

from models import CreditUsageDB

CREDIT_COSTS = {
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-35-turbo": {"input": 0.5, "output": 1.5},
    "dall-e-3": {"per_image": 40.0},
    "dall-e-2": {"per_image": 20.0},
    "mai": {"per_image": 30.0},
    "gpt-image-2": {"per_image": 35.0},
    "sora-2": {"per_second": 50.0},
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
    credits = 0.0
    model_key = model_name.lower()
    for key in CREDIT_COSTS.keys():
        if key in model_key:
            model_key = key
            break
    if model_key not in CREDIT_COSTS:
        model_key = "gpt-4o-mini"
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

import base64
import requests
from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from config import GPT_IMAGE_2_ENDPOINT, GPT_IMAGE_2_API_KEY
from models import ReferenceImageDB, SettingsDB, ChannelDB
from services.blob_storage import upload_bytes_to_blob


def generate_image_bytes(
    prompt: str,
    ch: Optional[ChannelDB],
    s: SettingsDB,
    db: Session,
    width: int = 1024,
    height: int = 1024,
) -> bytes:
    """Gera imagem via gpt-image-2 no Azure OpenAI."""
    if not GPT_IMAGE_2_API_KEY:
        raise HTTPException(status_code=400, detail="GPT_IMAGE_2_API_KEY não configurado no servidor")

    print(f"[GEN_IMAGE] modelo=gpt-image-2 prompt={len(prompt)}chars")

    try:
        from openai import AzureOpenAI as _AzOAI
        client = _AzOAI(
            azure_endpoint=GPT_IMAGE_2_ENDPOINT,
            api_key=GPT_IMAGE_2_API_KEY,
            api_version="2025-04-01-preview",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao criar cliente Azure OpenAI: {e}")

    result = client.images.generate(
        model="gpt-image-2",
        prompt=prompt,
        n=1,
        size="1024x1024",
    )
    img_bytes = base64.b64decode(result.data[0].b64_json)
    print(f"[GEN_IMAGE] ✓ {len(img_bytes)} bytes")
    return img_bytes


def get_reference_context(channel_id: str, db: Session) -> str:
    """Retorna descrição textual das referências visuais para enriquecer o prompt."""
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


def save_image_from_base64(base64_data: str, post_id: str) -> str:
    if base64_data.startswith("data:image"):
        base64_data = base64_data.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    blob_name = f"posts/{post_id}_{ts}.png"
    return upload_bytes_to_blob(image_bytes, blob_name, "image/png")

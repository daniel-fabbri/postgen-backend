import base64
import io
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
    """Roteia geração de imagem para o modelo correto do canal (MAI ou GPT-Image-2)."""
    model = (ch.image_model or "mai") if ch else "mai"

    if model == "gpt-image-2":
        if not GPT_IMAGE_2_API_KEY:
            raise HTTPException(status_code=400, detail="GPT_IMAGE_2_API_KEY não configurado no servidor")
        from openai import AzureOpenAI as _AzOAI
        img_client = _AzOAI(
            azure_endpoint=GPT_IMAGE_2_ENDPOINT,
            api_key=GPT_IMAGE_2_API_KEY,
            api_version="2025-04-01-preview",
        )
        refs = db.query(ReferenceImageDB).filter(
            ReferenceImageDB.channel_id == ch.id,
        ).order_by(ReferenceImageDB.created_at.desc()).limit(1).all()

        if refs:
            try:
                ref_bytes = requests.get(refs[0].blob_url, timeout=20).content
                size_str = f"{width}x{height}" if width == height else "1024x1024"
                result = img_client.images.edit(
                    model="gpt-image-2",
                    image=("reference.jpg", io.BytesIO(ref_bytes), "image/jpeg"),
                    prompt=prompt,
                    n=1,
                    size=size_str,
                )
                return base64.b64decode(result.data[0].b64_json)
            except Exception as e:
                print(f"gpt-image-2 edit failed, falling back to generate: {e}")

        result = img_client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        return base64.b64decode(result.data[0].b64_json)

    else:  # MAI
        if not s.azure_openai_image_endpoint:
            raise HTTPException(status_code=400, detail="Endpoint de imagem não configurado")
        resp = requests.post(
            s.azure_openai_image_endpoint,
            headers={"Content-Type": "application/json", "api-key": s.azure_openai_api_key},
            json={"prompt": prompt, "width": width, "height": height, "model": s.azure_openai_image_deployment},
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        if not result.get("data") or "b64_json" not in result["data"][0]:
            raise HTTPException(status_code=500, detail="Sem dados de imagem na resposta")
        return base64.b64decode(result["data"][0]["b64_json"])


def get_reference_context(channel_id: str, db: Session) -> str:
    """Retorna descrição visual de imagens de referência para injetar no prompt."""
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
    """Decodifica imagem base64 e faz upload para blob storage."""
    if base64_data.startswith("data:image"):
        base64_data = base64_data.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    blob_name = f"posts/{post_id}_{ts}.png"
    return upload_bytes_to_blob(image_bytes, blob_name, "image/png")

import base64
import requests
from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from config import AZURE_FOUNDRY_ENDPOINT, AZURE_FOUNDRY_API_KEY
from models import ReferenceImageDB, SettingsDB, ChannelDB
from services.blob_storage import upload_bytes_to_blob

_IMAGE_MODEL = "gpt-image-1"


def generate_image_bytes(
    prompt: str,
    ch: Optional[ChannelDB],
    s: SettingsDB,
    db: Session,
    width: int = 1024,
    height: int = 1024,
) -> bytes:
    """Gera imagem via gpt-image-1 no Azure AI Foundry."""
    if not AZURE_FOUNDRY_ENDPOINT or not AZURE_FOUNDRY_API_KEY:
        raise HTTPException(status_code=400, detail="Azure AI Foundry não configurado (AZURE_FOUNDRY_ENDPOINT / AZURE_FOUNDRY_API_KEY)")

    print(f"[GEN_IMAGE] modelo={_IMAGE_MODEL} size={width}x{height} prompt={len(prompt)}chars")

    payload = {
        "model": _IMAGE_MODEL,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
    }

    resp = requests.post(
        AZURE_FOUNDRY_ENDPOINT,
        headers={"Content-Type": "application/json", "api-key": AZURE_FOUNDRY_API_KEY},
        json=payload,
        timeout=120,
    )

    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=f"Erro na geração de imagem: {resp.text[:400]}")

    result = resp.json()
    if not result.get("data"):
        raise HTTPException(status_code=500, detail="Sem dados de imagem na resposta")

    item = result["data"][0]
    if "b64_json" in item:
        return base64.b64decode(item["b64_json"])
    if "url" in item:
        img_resp = requests.get(item["url"], timeout=60)
        img_resp.raise_for_status()
        return img_resp.content

    raise HTTPException(status_code=500, detail=f"Formato inesperado na resposta: {list(item.keys())}")


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

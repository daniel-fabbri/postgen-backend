import base64
import io
import requests
from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from config import (
    GPT_IMAGE_2_ENDPOINT, GPT_IMAGE_2_API_KEY,
    AZURE_FOUNDRY_ENDPOINT, AZURE_FOUNDRY_API_KEY,
)
from models import ReferenceImageDB, SettingsDB, ChannelDB
from services.blob_storage import upload_bytes_to_blob

# Model IDs as used in Azure AI Foundry requests
_FLUX_KONTEXT_MODEL = "FLUX.1-Kontext-pro"
_FLUX_2_PRO_MODEL = "FLUX.2-pro"


def _call_foundry_images(payload: dict, timeout: int = 120) -> bytes:
    """POST to Azure AI Foundry image generation endpoint and return image bytes."""
    if not AZURE_FOUNDRY_ENDPOINT or not AZURE_FOUNDRY_API_KEY:
        raise HTTPException(status_code=400, detail="Azure AI Foundry não configurado no servidor (AZURE_FOUNDRY_ENDPOINT / AZURE_FOUNDRY_API_KEY)")

    resp = requests.post(
        AZURE_FOUNDRY_ENDPOINT,
        headers={"Content-Type": "application/json", "api-key": AZURE_FOUNDRY_API_KEY},
        json=payload,
        timeout=timeout,
    )
    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=f"Azure Foundry error: {resp.text[:400]}")

    result = resp.json()
    if not result.get("data"):
        raise HTTPException(status_code=500, detail="Sem dados de imagem na resposta do Foundry")

    item = result["data"][0]
    if "b64_json" in item:
        return base64.b64decode(item["b64_json"])
    if "url" in item:
        img_resp = requests.get(item["url"], timeout=60)
        img_resp.raise_for_status()
        return img_resp.content

    raise HTTPException(status_code=500, detail=f"Formato inesperado na resposta: {list(item.keys())}")


def generate_image_bytes(
    prompt: str,
    ch: Optional[ChannelDB],
    s: SettingsDB,
    db: Session,
    width: int = 1024,
    height: int = 1024,
) -> bytes:
    """Roteia geração de imagem para o modelo correto do canal."""
    model = (ch.image_model or "mai") if ch else "mai"
    print(f"[GEN_IMAGE] modelo={model} size={width}x{height} prompt={len(prompt)}chars")

    # ------------------------------------------------------------------
    # FLUX.1-Kontext-pro — image-to-image com rosto fixo de referência
    # ------------------------------------------------------------------
    if model == "flux-kontext":
        refs = db.query(ReferenceImageDB).filter(
            ReferenceImageDB.channel_id == ch.id,
        ).order_by(ReferenceImageDB.created_at.desc()).limit(1).all()

        if refs:
            print(f"[GEN_IMAGE] Kontext: image-to-image com referência {refs[0].id} url={refs[0].blob_url[:60]}")
            payload = {
                "model": _FLUX_KONTEXT_MODEL,
                "prompt": prompt,
                "n": 1,
                "image": refs[0].blob_url,  # URL direta — confirmado que a API aceita
            }
        else:
            print(f"[GEN_IMAGE] Kontext: sem referência, usando text-to-image")
            payload = {
                "model": _FLUX_KONTEXT_MODEL,
                "prompt": prompt,
                "n": 1,
            }

        return _call_foundry_images(payload)

    # ------------------------------------------------------------------
    # FLUX.2-pro — text-to-image de alta qualidade
    # ------------------------------------------------------------------
    elif model == "flux-2-pro":
        print(f"[GEN_IMAGE] FLUX.2-pro text-to-image")
        payload = {
            "model": _FLUX_2_PRO_MODEL,
            "prompt": prompt,
            "n": 1,
        }
        return _call_foundry_images(payload)

    # ------------------------------------------------------------------
    # GPT-Image-2
    # ------------------------------------------------------------------
    elif model == "gpt-image-2":
        print(f"[GEN_IMAGE] Usando GPT-Image-2")
        if not GPT_IMAGE_2_API_KEY:
            raise HTTPException(status_code=400, detail="GPT_IMAGE_2_API_KEY não configurado no servidor")

        try:
            from openai import AzureOpenAI as _AzOAI
            img_client = _AzOAI(
                azure_endpoint=GPT_IMAGE_2_ENDPOINT,
                api_key=GPT_IMAGE_2_API_KEY,
                api_version="2025-04-01-preview",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro ao criar cliente GPT-Image-2: {e}")

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
                print(f"[GEN_IMAGE] gpt-image-2 edit falhou, usando generate: {str(e)[:200]}")

        result = img_client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        return base64.b64decode(result.data[0].b64_json)

    # ------------------------------------------------------------------
    # MAI / DALL-E (padrão)
    # ------------------------------------------------------------------
    else:
        print(f"[GEN_IMAGE] Usando MAI/DALL-E endpoint")
        if not s.azure_openai_image_endpoint:
            raise HTTPException(status_code=400, detail="Endpoint de imagem não configurado")

        size_str = f"{width}x{height}"
        payload = {
            "prompt": prompt,
            "n": 1,
            "size": size_str,
            "model": s.azure_openai_image_deployment,
        }
        resp = requests.post(
            s.azure_openai_image_endpoint,
            headers={"Content-Type": "application/json", "api-key": s.azure_openai_api_key},
            json=payload,
            timeout=60,
        )
        if not resp.ok:
            resp.raise_for_status()

        result = resp.json()
        if not result.get("data"):
            raise HTTPException(status_code=500, detail="Sem dados de imagem na resposta")

        item = result["data"][0]
        if "b64_json" in item:
            return base64.b64decode(item["b64_json"])
        if "url" in item:
            img_resp = requests.get(item["url"], timeout=30)
            img_resp.raise_for_status()
            return img_resp.content

        raise HTTPException(status_code=500, detail=f"Formato de resposta inesperado: {list(item.keys())}")


def get_reference_context(channel_id: str, db: Session) -> str:
    """Retorna descrição visual de imagens de referência para injetar no prompt (apenas para modelos sem suporte a imagem)."""
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

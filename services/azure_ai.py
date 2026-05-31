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
    print(f"[GEN_IMAGE] Iniciando geração com modelo: {model}")
    print(f"[GEN_IMAGE] Dimensões: {width}x{height}")
    print(f"[GEN_IMAGE] Prompt length: {len(prompt)} chars")

    if model == "gpt-image-2":
        print(f"[GEN_IMAGE] Usando GPT-Image-2")
        if not GPT_IMAGE_2_API_KEY:
            print(f"[GEN_IMAGE] ✗ GPT_IMAGE_2_API_KEY não configurado")
            raise HTTPException(status_code=400, detail="GPT_IMAGE_2_API_KEY não configurado no servidor")
        
        try:
            from openai import AzureOpenAI as _AzOAI
            print(f"[GEN_IMAGE] Criando cliente Azure OpenAI para GPT-Image-2")
            img_client = _AzOAI(
                azure_endpoint=GPT_IMAGE_2_ENDPOINT,
                api_key=GPT_IMAGE_2_API_KEY,
                api_version="2025-04-01-preview",
            )
            print(f"[GEN_IMAGE] ✓ Cliente criado")
        except Exception as e:
            print(f"[GEN_IMAGE] ✗ Erro ao criar cliente: {str(e)}")
            raise
        
        try:
            refs = db.query(ReferenceImageDB).filter(
                ReferenceImageDB.channel_id == ch.id,
            ).order_by(ReferenceImageDB.created_at.desc()).limit(1).all()
            print(f"[GEN_IMAGE] Imagens de referência encontradas: {len(refs)}")
        except Exception as e:
            print(f"[GEN_IMAGE] ✗ Erro ao buscar referências: {str(e)}")
            raise

        if refs:
            try:
                print(f"[GEN_IMAGE] Tentando gerar com edit usando referência...")
                ref_bytes = requests.get(refs[0].blob_url, timeout=20).content
                size_str = f"{width}x{height}" if width == height else "1024x1024"
                result = img_client.images.edit(
                    model="gpt-image-2",
                    image=("reference.jpg", io.BytesIO(ref_bytes), "image/jpeg"),
                    prompt=prompt,
                    n=1,
                    size=size_str,
                )
                img_bytes = base64.b64decode(result.data[0].b64_json)
                print(f"[GEN_IMAGE] ✓ Imagem gerada com edit: {len(img_bytes)} bytes")
                return img_bytes
            except Exception as e:
                print(f"[GEN_IMAGE] gpt-image-2 edit falhou, usando generate: {str(e)[:200]}")

        try:
            print(f"[GEN_IMAGE] Gerando com images.generate...")
            result = img_client.images.generate(
                model="gpt-image-2",
                prompt=prompt,
                n=1,
                size="1024x1024",
            )
            img_bytes = base64.b64decode(result.data[0].b64_json)
            print(f"[GEN_IMAGE] ✓ Imagem gerada: {len(img_bytes)} bytes")
            return img_bytes
        except Exception as e:
            print(f"[GEN_IMAGE] ✗ Erro no generate: {str(e)}")
            raise

    else:  # MAI / DALL-E compatible endpoint
        print(f"[GEN_IMAGE] Usando MAI/DALL-E endpoint")
        if not s.azure_openai_image_endpoint:
            print(f"[GEN_IMAGE] ✗ Endpoint não configurado")
            raise HTTPException(status_code=400, detail="Endpoint de imagem não configurado")
        
        size_str = f"{width}x{height}"
        endpoint = s.azure_openai_image_endpoint
        deployment = s.azure_openai_image_deployment
        print(f"[GEN_IMAGE] Endpoint: {endpoint[:60]}...")
        print(f"[GEN_IMAGE] Deployment: {deployment}")
        print(f"[GEN_IMAGE] Size: {size_str}")
        
        payload = {"prompt": prompt, "n": 1, "size": size_str, "model": deployment}
        print(f"[GEN_IMAGE] Payload keys: {list(payload.keys())}")
        
        try:
            print(f"[GEN_IMAGE] Fazendo requisição POST...")
            resp = requests.post(
                endpoint,
                headers={"Content-Type": "application/json", "api-key": s.azure_openai_api_key},
                json=payload,
                timeout=60,
            )
            print(f"[GEN_IMAGE] Response status: {resp.status_code}")
            
            if not resp.ok:
                error_text = resp.text[:500]
                print(f"[GEN_IMAGE] ✗ MAI error {resp.status_code}: {error_text}")
                resp.raise_for_status()
            
            result = resp.json()
            print(f"[GEN_IMAGE] ✓ Response recebida, keys: {list(result.keys())}")
            
            if not result.get("data"):
                print(f"[GEN_IMAGE] ✗ Sem campo 'data' na resposta")
                raise HTTPException(status_code=500, detail="Sem dados de imagem na resposta")
            
            item = result["data"][0]
            print(f"[GEN_IMAGE] Item keys: {list(item.keys())}")
            
            if "b64_json" in item:
                print(f"[GEN_IMAGE] Decodificando b64_json...")
                img_bytes = base64.b64decode(item["b64_json"])
                print(f"[GEN_IMAGE] ✓ Imagem decodificada: {len(img_bytes)} bytes")
                return img_bytes
            
            if "url" in item:
                img_url = item["url"]
                print(f"[GEN_IMAGE] Baixando de URL: {img_url[:80]}...")
                img_resp = requests.get(img_url, timeout=30)
                img_resp.raise_for_status()
                img_bytes = img_resp.content
                print(f"[GEN_IMAGE] ✓ Imagem baixada: {len(img_bytes)} bytes")
                return img_bytes
            
            print(f"[GEN_IMAGE] ✗ Formato de resposta inesperado: {list(item.keys())}")
            raise HTTPException(status_code=500, detail=f"Formato de resposta inesperado: {list(item.keys())}")
        except requests.exceptions.RequestException as e:
            print(f"[GEN_IMAGE] ✗ Erro de requisição: {str(e)[:300]}")
            raise
        except Exception as e:
            print(f"[GEN_IMAGE] ✗ Erro inesperado: {str(e)[:300]}")
            raise


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

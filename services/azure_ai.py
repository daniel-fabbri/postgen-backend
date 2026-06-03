import base64
import io
import json
import requests
from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from config import (
    GPT_IMAGE_2_ENDPOINT, GPT_IMAGE_2_API_KEY,
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
)
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
        size=f"{width}x{height}",
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


def _detect_faces_opencv(image_url: str) -> list:
    """Baixa a imagem e detecta faces com OpenCV Haar Cascade com filtros anti-falso-positivo."""
    import cv2
    import numpy as np

    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()

    img_array = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    # minNeighbors=6 é mais restritivo — elimina falsos positivos em janelas/padrões arquitetônicos
    min_face_px = max(60, int(min(w, h) * 0.06))
    detections = cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=6, minSize=(min_face_px, min_face_px)
    )

    faces = []
    for (x, y, fw, fh) in (detections if len(detections) > 0 else []):
        rx, ry = float(x) / w, float(y) / h
        rw, rh = float(fw) / w, float(fh) / h

        # Filtro 1: aspect ratio — rostos são ~quadrados (0.65–1.5); janelas podem ser muito retangulares
        aspect = rw / rh if rh > 0 else 0
        if not (0.65 <= aspect <= 1.5):
            continue

        # Filtro 2: posição vertical — rejeitar detecções muito no topo (y < 15% = background buildings)
        # mas só se o centro vertical da detecção estiver acima de 25% da imagem
        center_y = ry + rh / 2
        if center_y < 0.20:
            continue

        faces.append({"x": rx, "y": ry, "w": rw, "h": rh})

    print(f"[VISION] OpenCV detectou {len(faces)} face(s) após filtros (de {len(detections) if len(detections) > 0 else 0} raw)")
    return faces


def identify_main_person_bbox(image_url: str, person_description: str = "") -> Optional[dict]:
    """
    Detecta faces com OpenCV e retorna a bounding box da face mais proeminente
    (maior área × mais central = sujeito principal do post).
    Retorna {"x": f, "y": f, "w": f, "h": f} como frações 0-1, ou None.
    """
    try:
        faces = _detect_faces_opencv(image_url)
        print(f"[VISION] OpenCV detectou {len(faces)} face(s)")

        if not faces:
            return None

        if len(faces) == 1:
            print(f"[VISION] Face única: {faces[0]}")
            return faces[0]

        # Múltiplas faces: maior área com peso para centralidade (sujeito principal)
        best, best_score = None, -1
        for face in faces:
            area = face["w"] * face["h"]
            face_cx = face["x"] + face["w"] / 2
            face_cy = face["y"] + face["h"] / 2
            dist = ((face_cx - 0.5) ** 2 + (face_cy - 0.5) ** 2) ** 0.5
            centrality = 1 - dist / (0.5 * 2 ** 0.5)
            score = area * (0.7 + 0.3 * centrality)
            if score > best_score:
                best_score = score
                best = face

        print(f"[VISION] Face principal: {best}")
        return best

    except Exception as e:
        print(f"[VISION] detect_faces_opencv falhou: {e}")
        return None


def create_inpaint_mask(bbox: dict, image_w: int = 1024, image_h: int = 1024) -> bytes:
    """
    Cria uma máscara PNG para inpainting: branco onde a cabeça está (região a alterar),
    preto no resto (preservar cena).
    A região cobre a cabeça completa incluindo cabelo, com bordas suavizadas.
    """
    from PIL import Image, ImageDraw, ImageFilter

    x = int(bbox["x"] * image_w)
    y = int(bbox["y"] * image_h)
    w = int(bbox["w"] * image_w)
    h = int(bbox["h"] * image_h)

    # Centro deslocado para cima para incluir o cabelo (25% da altura do bbox acima do centro)
    cx = x + w // 2
    cy = y + h // 2 - int(h * 0.25)

    # Semi-eixos generosos: cobre a cabeça inteira incluindo cabelo e orelhas
    # rx = 90% da largura do bbox (face + orelhas), ry = 110% (inclui cabelo acima + pescoço abaixo)
    rx = int(w * 0.90)
    ry = int(h * 1.10)

    x1 = max(0, cx - rx)
    y1 = max(0, cy - ry)
    x2 = min(image_w, cx + rx)
    y2 = min(image_h, cy + ry)

    mask = Image.new("L", (image_w, image_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([x1, y1, x2, y2], fill=255)
    # Blur generoso para borda suave e sem artefato de círculo visível
    mask = mask.filter(ImageFilter.GaussianBlur(radius=28))

    mask_rgb = Image.new("RGB", (image_w, image_h), (0, 0, 0))
    mask_rgb.paste((255, 255, 255), mask=mask)

    buf = io.BytesIO()
    mask_rgb.save(buf, format="PNG")
    print(f"[VISION] Máscara: centro=({cx},{cy}) rx={rx} ry={ry} bbox=({x1},{y1})→({x2},{y2})")
    return buf.getvalue()


def save_image_from_base64(base64_data: str, post_id: str) -> str:
    if base64_data.startswith("data:image"):
        base64_data = base64_data.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    blob_name = f"posts/{post_id}_{ts}.png"
    return upload_bytes_to_blob(image_bytes, blob_name, "image/png")

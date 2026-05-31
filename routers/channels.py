import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from urllib.parse import urlencode

from config import (
    BASE_URL, FRONTEND_URL, INSTAGRAM_APP_ID, INSTAGRAM_APP_SECRET,
    JWT_SECRET, JWT_ALGORITHM, AZURE_STORAGE_CONTAINER,
)
from database import get_db
from dependencies import get_current_user, get_or_create_settings, get_channel_or_404
from models import UserDB, ChannelDB, AvatarDB, ReferenceImageDB
from schemas import (
    Channel, UpdateAvatarRequest, TestInstagramRequest, AvatarInfo,
    ReferenceImageOut, GenerateAvatarRequest,
)
from services.azure_ai import generate_image_bytes, get_reference_context
from services.blob_storage import upload_bytes_to_blob
from services.converters import channel_to_schema
from services.credits import register_credit_usage

router = APIRouter(tags=["channels"])


# ---------------------------------------------------------------------------
# Channels CRUD
# ---------------------------------------------------------------------------

@router.get("/api/channels", response_model=List[Channel])
def get_channels(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    channels = db.query(ChannelDB).filter(ChannelDB.user_id == current_user.id).all()
    return [channel_to_schema(ch) for ch in channels]


@router.post("/api/channels", response_model=Channel, status_code=201)
def create_channel(
    data: Channel,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = ChannelDB(
        id=f"ch_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        user_id=current_user.id,
        name=data.name,
        objective=data.objective,
        text_generation_prompt=data.text_generation_prompt,
        image_generation_prompt=data.image_generation_prompt,
        instagram_user_id=data.instagram_user_id,
        instagram_access_token=data.instagram_access_token,
    )
    db.add(ch)
    db.commit()
    db.refresh(ch)

    if ch.image_generation_prompt:
        try:
            s = get_or_create_settings(current_user, db)
            avatar_prompt = (
                "\n".join(ch.image_generation_prompt.splitlines()[:5])
                + "\n\nFrame: Close-up portrait style, profile picture format."
            )
            image_bytes = generate_image_bytes(avatar_prompt, ch, s, db, width=768, height=768)
            avatar_filename = f"{ch.id}.png"
            avatar_url = upload_bytes_to_blob(image_bytes, f"avatars/{avatar_filename}", "image/png")
            ch.avatar_url = avatar_url
            _register_avatar(avatar_filename, ch.id, db)
            db.commit()
            db.refresh(ch)
        except Exception as e:
            print(f"Error generating avatar: {e}")

    return channel_to_schema(ch)


@router.get("/api/channels/{channel_id}", response_model=Channel)
def get_channel(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return channel_to_schema(get_channel_or_404(channel_id, current_user, db))


@router.put("/api/channels/{channel_id}", response_model=Channel)
def update_channel(
    channel_id: str,
    data: Channel,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)
    ch.name = data.name
    ch.objective = data.objective
    ch.text_generation_prompt = data.text_generation_prompt or None
    ch.image_generation_prompt = data.image_generation_prompt or None
    ch.avatar_url = data.avatar_url
    ch.suggested_image_url = data.suggested_image_url
    ch.instagram_user_id = data.instagram_user_id
    if data.instagram_access_token and data.instagram_access_token != "***":
        ch.instagram_access_token = data.instagram_access_token
    if data.image_model:
        ch.image_model = data.image_model
    if data.auto_reply_enabled is not None:
        ch.auto_reply_enabled = data.auto_reply_enabled
    ch.auto_reply_prompt = data.auto_reply_prompt
    db.commit()
    db.refresh(ch)
    return channel_to_schema(ch)


@router.delete("/api/channels/{channel_id}", status_code=204)
def delete_channel(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)
    db.delete(ch)
    db.commit()


@router.patch("/api/channels/{channel_id}/avatar")
@router.post("/api/channels/{channel_id}/avatar")
def update_channel_avatar(
    channel_id: str,
    data: UpdateAvatarRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)
    ch.avatar_url = data.avatar_url
    filename = data.avatar_url.rstrip("/").split("/")[-1]
    if filename:
        _register_avatar(filename, channel_id, db)
    db.commit()
    db.refresh(ch)
    return {"success": True, "channel": channel_to_schema(ch)}


# ---------------------------------------------------------------------------
# Instagram OAuth
# ---------------------------------------------------------------------------

@router.get("/api/auth/instagram/authorize")
def instagram_authorize(
    channel_id: str = Query(...),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not INSTAGRAM_APP_ID or not INSTAGRAM_APP_SECRET:
        raise HTTPException(status_code=500, detail="Instagram OAuth não configurado no servidor.")
    get_channel_or_404(channel_id, current_user, db)
    state = jwt.encode(
        {"channel_id": channel_id, "user_id": current_user.id,
         "exp": datetime.now(timezone.utc) + timedelta(minutes=15)},
        JWT_SECRET, algorithm=JWT_ALGORITHM,
    )
    params = {
        "client_id": INSTAGRAM_APP_ID,
        "redirect_uri": f"{BASE_URL}/api/auth/instagram/callback",
        "scope": "instagram_business_basic,instagram_business_content_publish,instagram_business_manage_insights",
        "response_type": "code",
        "state": state,
    }
    return {"url": "https://www.instagram.com/oauth/authorize?" + urlencode(params)}


@router.get("/api/auth/instagram/callback")
def instagram_callback(
    code: str = None,
    state: str = None,
    error: str = None,
    db: Session = Depends(get_db),
):
    front = FRONTEND_URL.rstrip("/")
    if error or not code or not state:
        return RedirectResponse(url=f"{front}/channels?ig_error=cancelled")
    try:
        state_data = jwt.decode(state, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        channel_id = state_data["channel_id"]
        user_id = state_data["user_id"]
    except JWTError:
        return RedirectResponse(url=f"{front}/channels?ig_error=invalid_state")

    redirect_uri = f"{BASE_URL}/api/auth/instagram/callback"
    try:
        token_resp = requests.post(
            "https://api.instagram.com/oauth/access_token",
            data={
                "client_id": INSTAGRAM_APP_ID,
                "client_secret": INSTAGRAM_APP_SECRET,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
                "code": code,
            },
            timeout=15,
        )
        token_data = token_resp.json()
        if "access_token" not in token_data:
            err = token_data.get("error_message", "token_exchange_failed")
            return RedirectResponse(url=f"{front}/channels/{channel_id}/edit?ig_error={err}")

        short_token = token_data["access_token"]
        ig_user_id = str(token_data["user_id"])

        ll_resp = requests.get(
            "https://graph.instagram.com/access_token",
            params={
                "grant_type": "ig_exchange_token",
                "client_id": INSTAGRAM_APP_ID,
                "client_secret": INSTAGRAM_APP_SECRET,
                "access_token": short_token,
            },
            timeout=15,
        )
        long_token = ll_resp.json().get("access_token", short_token)

        me_resp = requests.get(
            "https://graph.instagram.com/me",
            params={"fields": "id,username", "access_token": long_token},
            timeout=10,
        )
        username = me_resp.json().get("username", ig_user_id)

        ch = db.query(ChannelDB).filter(
            ChannelDB.id == channel_id,
            ChannelDB.user_id == user_id,
        ).first()
        if not ch:
            return RedirectResponse(url=f"{front}/channels?ig_error=channel_not_found")

        ch.instagram_user_id = ig_user_id
        ch.instagram_access_token = long_token
        db.commit()
        return RedirectResponse(url=f"{front}/channels/{channel_id}/edit?ig_success={username}")

    except requests.RequestException:
        return RedirectResponse(url=f"{front}/channels/{channel_id}/edit?ig_error=network_error")


@router.delete("/api/channels/{channel_id}/instagram")
def instagram_disconnect(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(channel_id, current_user, db)
    ch.instagram_user_id = None
    ch.instagram_access_token = None
    db.commit()
    return {"success": True}


@router.post("/api/channels/{channel_id}/test-instagram")
def test_instagram_connection(
    channel_id: str,
    data: TestInstagramRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from services.instagram import ig_api_base
    ch = get_channel_or_404(channel_id, current_user, db)
    user_id = data.instagram_user_id or ch.instagram_user_id
    token = data.instagram_access_token if (data.instagram_access_token and data.instagram_access_token != "***") else ch.instagram_access_token
    if not user_id or not token:
        raise HTTPException(status_code=400, detail="Preencha o User ID e o Access Token antes de testar.")
    try:
        resp = requests.get(
            f"{ig_api_base(token)}/{user_id}",
            params={"fields": "id,name,username,followers_count", "access_token": token},
            timeout=10,
        )
        result = resp.json()
        if "error" in result:
            return {"success": False, "error": result["error"].get("message", "Erro desconhecido")}
        return {
            "success": True,
            "account": {
                "id": result.get("id"),
                "name": result.get("name"),
                "username": result.get("username"),
                "followers_count": result.get("followers_count"),
            },
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Falha ao conectar com o Instagram: {str(e)}")


# ---------------------------------------------------------------------------
# Avatars
# ---------------------------------------------------------------------------

def _register_avatar(filename: str, channel_id: str, db: Session):
    existing = db.query(AvatarDB).filter(AvatarDB.filename == filename).first()
    if existing:
        existing.channel_id = channel_id
    else:
        db.add(AvatarDB(filename=filename, channel_id=channel_id))
    db.commit()


@router.get("/api/avatars", response_model=List[AvatarInfo])
def list_avatars(
    channel_id: Optional[str] = None,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if channel_id:
        get_channel_or_404(channel_id, current_user, db)
        rows = db.query(AvatarDB).filter(AvatarDB.channel_id == channel_id).all()
    else:
        user_channel_ids = [
            ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == current_user.id).all()
        ]
        rows = db.query(AvatarDB).filter(AvatarDB.channel_id.in_(user_channel_ids)).all()

    result = []
    for row in rows:
        blob_url = f"https://postgenstorage.blob.core.windows.net/{AZURE_STORAGE_CONTAINER}/avatars/{row.filename}"
        result.append(AvatarInfo(
            filename=row.filename,
            url=blob_url,
            created_at=row.created_at.isoformat() if row.created_at else None,
        ))
    return sorted(result, key=lambda x: x.created_at or "", reverse=True)


@router.post("/api/avatars/generate")
def generate_avatar(
    data: GenerateAvatarRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = get_or_create_settings(current_user, db)
    ch = None
    channel_prompt = ""
    if data.channel_id:
        ch = db.query(ChannelDB).filter(ChannelDB.id == data.channel_id).first()
        if ch and ch.image_generation_prompt:
            channel_prompt = "\n".join(ch.image_generation_prompt.splitlines()[:5])

    portrait_suffix = "\n\nFrame: Close-up portrait style, profile picture format."
    if channel_prompt and data.prompt:
        full_prompt = f"{channel_prompt}\n\n{data.prompt}{portrait_suffix}"
    else:
        full_prompt = (channel_prompt or data.prompt) + portrait_suffix

    if ch:
        full_prompt += get_reference_context(ch.id, db)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    avatar_filename = f"avatar_{timestamp}.png"

    try:
        image_bytes = generate_image_bytes(full_prompt, ch, s, db, width=768, height=768)
    except HTTPException:
        raise
    except Exception as e:
        if data.prompt and channel_prompt:
            try:
                fallback = data.prompt + portrait_suffix + (get_reference_context(ch.id, db) if ch else "")
                image_bytes = generate_image_bytes(fallback, ch, s, db, width=768, height=768)
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Falha ao gerar avatar: {str(e2)}")
        else:
            raise HTTPException(status_code=500, detail=f"Falha ao gerar avatar: {str(e)}")

    avatar_url = upload_bytes_to_blob(image_bytes, f"avatars/{avatar_filename}", "image/png")
    image_model = ch.image_model if ch else "mai"
    register_credit_usage(
        db=db, user_id=current_user.id, channel_id=data.channel_id,
        resource_type="avatar", resource_id=avatar_filename,
        operation_type="image_generation", model_name=image_model,
        images_count=1, metadata={"prompt_length": len(full_prompt), "size": "768x768"},
    )

    if data.channel_id and ch:
        ch = get_channel_or_404(data.channel_id, current_user, db)
        _register_avatar(avatar_filename, data.channel_id, db)
        ch.avatar_url = avatar_url
        db.commit()

    return {"success": True, "avatar_url": avatar_url, "filename": avatar_filename}


@router.post("/api/avatars/upload")
def upload_avatar(
    file: UploadFile = File(...),
    channel_id: Optional[str] = None,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = file.filename.split(".")[-1] if "." in file.filename else "png"
    avatar_filename = f"avatar_{timestamp}.{ext}"
    data = file.file.read()
    avatar_url = upload_bytes_to_blob(data, f"avatars/{avatar_filename}", file.content_type or "image/png")
    if channel_id:
        ch = get_channel_or_404(channel_id, current_user, db)
        _register_avatar(avatar_filename, channel_id, db)
        ch.avatar_url = avatar_url
        db.commit()
    return {"success": True, "avatar_url": avatar_url, "filename": avatar_filename}


# ---------------------------------------------------------------------------
# Reference Images
# ---------------------------------------------------------------------------

@router.get("/api/channels/{channel_id}/references", response_model=List[ReferenceImageOut])
def list_reference_images(
    channel_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_channel_or_404(channel_id, current_user, db)
    refs = db.query(ReferenceImageDB).filter(
        ReferenceImageDB.channel_id == channel_id,
    ).order_by(ReferenceImageDB.created_at.desc()).all()
    return [ReferenceImageOut(
        id=r.id, channel_id=r.channel_id, blob_url=r.blob_url,
        description=r.description, created_at=r.created_at.isoformat(),
    ) for r in refs]


@router.post("/api/channels/{channel_id}/references/upload", response_model=ReferenceImageOut, status_code=201)
def upload_reference_image(
    channel_id: str,
    file: UploadFile = File(...),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_channel_or_404(channel_id, current_user, db)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")

    data = file.file.read()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = (file.content_type or "image/jpeg").split("/")[-1].replace("jpeg", "jpg")
    blob_url = upload_bytes_to_blob(data, f"references/{channel_id}_{ts}.{ext}", file.content_type or "image/jpeg")

    description = None
    s = get_or_create_settings(current_user, db)
    if s and s.azure_openai_endpoint and s.azure_openai_api_key:
        from dependencies import get_azure_client
        try:
            client = get_azure_client(s)
            vision_resp = client.chat.completions.create(
                model=s.azure_openai_deployment_name,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": (
                        "Describe the physical appearance of the person in this photo in detail "
                        "(face shape, hair color and style, eye color, skin tone, distinctive features, age range). "
                        "Be specific and concise — this description will be used in AI image generation prompts. "
                        "Answer in English. If there is no person visible, reply with 'no person detected'."
                    )},
                    {"type": "image_url", "image_url": {"url": blob_url}},
                ]}],
                max_tokens=200,
            )
            desc = vision_resp.choices[0].message.content.strip()
            if "no person detected" not in desc.lower():
                description = desc
        except Exception as e:
            print(f"Vision description failed: {e}")

    ref = ReferenceImageDB(
        channel_id=channel_id, user_id=current_user.id,
        blob_url=blob_url, description=description,
    )
    db.add(ref)
    db.commit()
    db.refresh(ref)
    return ReferenceImageOut(
        id=ref.id, channel_id=ref.channel_id, blob_url=ref.blob_url,
        description=ref.description, created_at=ref.created_at.isoformat(),
    )


@router.delete("/api/channels/{channel_id}/references/{ref_id}", status_code=204)
def delete_reference_image(
    channel_id: str,
    ref_id: int,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    get_channel_or_404(channel_id, current_user, db)
    ref = db.query(ReferenceImageDB).filter(
        ReferenceImageDB.id == ref_id,
        ReferenceImageDB.channel_id == channel_id,
    ).first()
    if not ref:
        raise HTTPException(status_code=404, detail="Imagem de referência não encontrada")
    db.delete(ref)
    db.commit()

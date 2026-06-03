import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Optional, List

import requests
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import AZURE_SORA_ENDPOINT, AZURE_SORA_API_KEY, REPLICATE_API_KEY
from database import get_db
from dependencies import (
    get_current_user, get_or_create_settings, get_azure_client,
    get_channel_or_404, get_video_or_404,
)
from models import UserDB, ChannelDB, VideoDB, VideoProjectDB
from schemas import (
    GenerateVideoRequest, SavedVideo, UpdateVideoCaptionRequest,
    VideoProjectOut, CreateVideoProjectRequest, UpdateVideoProjectClipsRequest,
    GenerateProjectClipRequest, AddVideoToProjectRequest,
)
from services.blob_storage import upload_bytes_to_blob
from services.replicate_ai import generate_with_lora, generate_video_from_image
from services.converters import video_to_schema, video_project_to_schema
from services.credits import register_credit_usage
from services.instagram import ig_api_base

router = APIRouter(tags=["videos"])


def _sora_headers():
    return {"Content-Type": "application/json", "Authorization": f"Bearer {AZURE_SORA_API_KEY}"}


def _get_project_or_404(project_id: str, user: UserDB, db: Session) -> VideoProjectDB:
    vp = db.query(VideoProjectDB).filter(
        VideoProjectDB.id == project_id,
        VideoProjectDB.user_id == user.id,
    ).first()
    if not vp:
        raise HTTPException(status_code=404, detail="Projeto não encontrado")
    return vp


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

@router.get("/api/videos", response_model=List[SavedVideo])
def get_videos(
    channel_id: Optional[str] = None,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from models import MediaInsightsDB, VideoProjectDB as _VP
    user_channel_ids = [
        ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == current_user.id).all()
    ]
    q = db.query(VideoDB).filter(
        VideoDB.channel_id.in_(user_channel_ids),
        VideoDB.is_project_clip.is_not(True),
    )
    if channel_id:
        q = q.filter(VideoDB.channel_id == channel_id)
    videos = q.order_by(VideoDB.created_at.desc()).all()

    video_ids = [v.id for v in videos]
    project_map = {}
    if video_ids:
        for vp in db.query(_VP.id, _VP.root_video_id, _VP.exported_video_id).filter(
            (_VP.root_video_id.in_(video_ids)) | (_VP.exported_video_id.in_(video_ids))
        ).all():
            if vp.root_video_id:
                project_map[vp.root_video_id] = vp.id
            if vp.exported_video_id:
                project_map[vp.exported_video_id] = vp.id

    ins_map = {}
    if video_ids:
        for ins in db.query(MediaInsightsDB).filter(
            MediaInsightsDB.media_type == "video",
            MediaInsightsDB.media_id.in_(video_ids),
        ).all():
            ins_map[ins.media_id] = ins

    return [video_to_schema(v, project_map.get(v.id), ins_map.get(v.id)) for v in videos]


@router.post("/api/videos/generate", response_model=SavedVideo)
def generate_video(
    data: GenerateVideoRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not AZURE_SORA_ENDPOINT or not AZURE_SORA_API_KEY:
        raise HTTPException(status_code=400, detail="Sora não configurado. Defina AZURE_SORA_ENDPOINT e AZURE_SORA_API_KEY.")

    ch = get_channel_or_404(data.channel_id, current_user, db)
    s = get_or_create_settings(current_user, db)

    base_prompt = ch.image_generation_prompt or f"Instagram Reel for channel '{ch.name}'. Theme: {ch.objective}."
    prompt = base_prompt
    if data.additional_prompt:
        prompt += f" {data.additional_prompt}"
    prompt = prompt[:4000]

    print(f"Sora prompt ({len(prompt)} chars): {prompt[:200]}")
    try:
        create_resp = requests.post(
            AZURE_SORA_ENDPOINT,
            headers=_sora_headers(),
            json={"prompt": prompt, "model": "sora-2", "size": data.size, "seconds": str(data.seconds)},
            timeout=30,
        )
        if not create_resp.ok:
            raise HTTPException(status_code=502, detail=f"Falha ao criar job Sora: {create_resp.status_code} - {create_resp.text}")
        create_resp.raise_for_status()
    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Falha ao criar job Sora: {str(e)}")

    job = create_resp.json()
    job_id = job.get("id") or job.get("job_id") or job.get("generation_id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"Resposta inesperada do Sora: {job}")

    caption = ""
    total_credits = 0.0
    try:
        client = get_azure_client(s)
        text_prompt = ch.text_generation_prompt or f"""Crie uma legenda para um Instagram Reel do canal "{ch.name}".
Objetivo do canal: {ch.objective}
Conceito do vídeo: {data.additional_prompt or prompt}
Escreva uma legenda envolvente com emojis e hashtags relevantes, 80-150 palavras.
Retorne apenas o texto da legenda."""
        cap_resp = client.chat.completions.create(
            model=s.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": "Você é um especialista em conteúdo para Instagram."},
                {"role": "user", "content": text_prompt},
            ],
            max_tokens=400, temperature=0.7,
        )
        caption = cap_resp.choices[0].message.content.strip()
        cap_usage = cap_resp.usage
        total_credits += register_credit_usage(
            db=db, user_id=current_user.id, channel_id=ch.id,
            resource_type="video", resource_id=None,
            operation_type="text_generation", model_name=s.azure_openai_deployment_name,
            input_tokens=cap_usage.prompt_tokens, output_tokens=cap_usage.completion_tokens,
            metadata={"step": "video_caption_generation"},
        )
    except Exception as e:
        print(f"Caption generation failed: {e}")

    poll_url = f"{AZURE_SORA_ENDPOINT}/{job_id}"
    deadline = datetime.now().timestamp() + 240
    completed = False
    while datetime.now().timestamp() < deadline:
        time.sleep(5)
        try:
            poll_resp = requests.get(poll_url, headers=_sora_headers(), timeout=15)
            result = poll_resp.json()
            status = result.get("status", "")
            print(f"Sora poll: status={status} progress={result.get('progress', '?')}")
        except Exception as e:
            print(f"Sora poll error: {e}")
            continue
        if status == "completed":
            completed = True
            break
        if status in ("failed", "error", "cancelled"):
            err_obj = result.get("error") or {}
            err = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
            raise HTTPException(status_code=502, detail=f"Sora falhou: {err or status}")

    if not completed:
        raise HTTPException(status_code=504, detail="Timeout aguardando o Sora. Tente novamente.")

    video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    content_url = f"{AZURE_SORA_ENDPOINT}/{job_id}/content"
    try:
        dl = requests.get(content_url, headers=_sora_headers(), timeout=120, allow_redirects=True)
        dl.raise_for_status()
        blob_url = upload_bytes_to_blob(dl.content, f"videos/{video_id}.mp4", "video/mp4")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao baixar vídeo Sora: {str(e)}")

    video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    total_credits += register_credit_usage(
        db=db, user_id=current_user.id, channel_id=ch.id,
        resource_type="video", resource_id=video_id,
        operation_type="video_generation", model_name="sora-2",
        video_seconds=data.seconds, metadata={"prompt_length": len(prompt), "size": data.size},
    )

    v = VideoDB(
        id=video_id, channel_id=ch.id, channel_name=ch.name,
        prompt=prompt, caption=caption, video_path=blob_url,
        duration_seconds=data.seconds, size=data.size,
        published=False, credits_consumed=total_credits,
    )
    db.add(v)
    db.commit()
    db.execute(
        text("UPDATE credit_usage SET resource_id = :video_id WHERE resource_id IS NULL AND user_id = :user_id AND channel_id = :channel_id"),
        {"video_id": video_id, "user_id": current_user.id, "channel_id": ch.id},
    )
    db.commit()
    db.refresh(v)
    return video_to_schema(v)


_VIDEO_WITH_CHARACTER_DURATION = 6  # minimax/video-01 gera ~6s fixos


@router.post("/api/videos/generate-with-character", response_model=SavedVideo)
def generate_video_with_character(
    data: GenerateVideoRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Gera vídeo com personagem do canal via LoRA + minimax/video-01.
    Requer LoRA treinado (ch.lora_status == "succeeded").
    Fluxo:
      1. Gera frame portrait (9:16) via LoRA com trigger word TOK
      2. Anima o frame com minimax/video-01
      3. Gera legenda via GPT-4o
    """
    ch = get_channel_or_404(data.channel_id, current_user, db)
    s = get_or_create_settings(current_user, db)

    if ch.lora_status != "succeeded" or not ch.lora_version:
        raise HTTPException(
            status_code=400,
            detail="Canal sem personagem treinado. Treine o LoRA primeiro nas configurações do canal.",
        )
    if not REPLICATE_API_KEY:
        raise HTTPException(status_code=400, detail="REPLICATE_API_KEY não configurado no servidor.")

    # 1. Prompt para o frame de referência (inclui trigger word TOK do LoRA)
    base_prompt = ch.image_generation_prompt or f"portrait of TOK, {ch.objective}"
    lora_frame_prompt = f"TOK, {base_prompt}"
    if data.additional_prompt:
        lora_frame_prompt += f", {data.additional_prompt}"
    lora_frame_prompt = lora_frame_prompt[:2000]

    print(f"[VIDEO-WITH-CHARACTER] Step 1 – frame LoRA | channel={ch.id}")

    # 2. Gerar frame portrait via LoRA
    lora_image_bytes = generate_with_lora(
        prompt=lora_frame_prompt,
        lora_ref=ch.lora_version,
        api_key=REPLICATE_API_KEY,
        aspect_ratio="9:16",
    )
    frame_id = f"charframe_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    frame_url = upload_bytes_to_blob(lora_image_bytes, f"temp/{frame_id}.png", "image/png")
    print(f"[VIDEO-WITH-CHARACTER] Frame gerado: {frame_url[:60]}")

    # 3. Prompt de movimento para o vídeo
    motion_prompt = ch.image_generation_prompt or f"Cinematic motion, {ch.objective}."
    if data.additional_prompt:
        motion_prompt += f" {data.additional_prompt}"
    motion_prompt = motion_prompt[:2000]

    print(f"[VIDEO-WITH-CHARACTER] Step 2 – minimax/video-01 | prompt={motion_prompt[:80]}")

    # 4. Animar o frame
    video_bytes = generate_video_from_image(
        image_url=frame_url,
        prompt=motion_prompt,
        api_key=REPLICATE_API_KEY,
    )

    # 5. Legenda via GPT-4o
    caption = ""
    total_credits = 0.0
    try:
        client = get_azure_client(s)
        text_prompt = ch.text_generation_prompt or f"""Crie uma legenda para um Instagram Reel do canal "{ch.name}".
Objetivo do canal: {ch.objective}
Conceito do vídeo: {data.additional_prompt or motion_prompt}
Escreva uma legenda envolvente com emojis e hashtags relevantes, 80-150 palavras.
Retorne apenas o texto da legenda."""
        cap_resp = client.chat.completions.create(
            model=s.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": "Você é um especialista em conteúdo para Instagram."},
                {"role": "user", "content": text_prompt},
            ],
            max_tokens=400, temperature=0.7,
        )
        caption = cap_resp.choices[0].message.content.strip()
        cap_usage = cap_resp.usage
        total_credits += register_credit_usage(
            db=db, user_id=current_user.id, channel_id=ch.id,
            resource_type="video", resource_id=None,
            operation_type="text_generation", model_name=s.azure_openai_deployment_name,
            input_tokens=cap_usage.prompt_tokens, output_tokens=cap_usage.completion_tokens,
            metadata={"step": "video_caption_generation"},
        )
    except Exception as e:
        print(f"[VIDEO-WITH-CHARACTER] Caption generation failed: {e}")

    # 6. Upload do vídeo
    video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    blob_url = upload_bytes_to_blob(video_bytes, f"videos/{video_id}.mp4", "video/mp4")

    # 7. Registrar créditos (LoRA frame + minimax video como operação única)
    total_credits += register_credit_usage(
        db=db, user_id=current_user.id, channel_id=ch.id,
        resource_type="video", resource_id=video_id,
        operation_type="video_generation", model_name="minimax/video-01",
        video_seconds=_VIDEO_WITH_CHARACTER_DURATION,
        metadata={"frame_url": frame_url, "size": "720x1280", "lora_version": str(ch.lora_version)[:16]},
    )

    # 8. Persistir
    v = VideoDB(
        id=video_id, channel_id=ch.id, channel_name=ch.name,
        prompt=lora_frame_prompt, caption=caption, video_path=blob_url,
        duration_seconds=_VIDEO_WITH_CHARACTER_DURATION, size="720x1280",
        published=False, credits_consumed=total_credits,
    )
    db.add(v)
    db.commit()
    db.execute(
        text("UPDATE credit_usage SET resource_id = :video_id WHERE resource_id IS NULL AND user_id = :user_id AND channel_id = :channel_id"),
        {"video_id": video_id, "user_id": current_user.id, "channel_id": ch.id},
    )
    db.commit()
    db.refresh(v)
    return video_to_schema(v)


@router.delete("/api/videos/{video_id}", status_code=204)
def delete_video(
    video_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    v = get_video_or_404(video_id, current_user, db)
    db.delete(v)
    db.commit()


@router.patch("/api/videos/{video_id}/caption", response_model=SavedVideo)
def update_video_caption(
    video_id: str,
    data: UpdateVideoCaptionRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    v = get_video_or_404(video_id, current_user, db)
    v.caption = data.caption
    db.commit()
    db.refresh(v)
    return video_to_schema(v)


@router.post("/api/videos/{video_id}/publish", response_model=SavedVideo)
def publish_video(
    video_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    v = get_video_or_404(video_id, current_user, db)
    ch = db.query(ChannelDB).filter(ChannelDB.id == v.channel_id).first()
    if not ch.instagram_user_id or not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não configurado para este canal.")

    try:
        _api = ig_api_base(ch.instagram_access_token)
        create_resp = requests.post(
            f"{_api}/{ch.instagram_user_id}/media",
            params={
                "media_type": "REELS", "video_url": v.video_path,
                "caption": v.caption or v.prompt, "access_token": ch.instagram_access_token,
            },
            timeout=30,
        )
        create_data = create_resp.json()
        if create_resp.status_code != 200 or "id" not in create_data:
            error_msg = create_data.get("error", {}).get("message", create_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao criar container Reels: {error_msg}")

        container_id = create_data["id"]
        for _ in range(24):
            time.sleep(5)
            status_resp = requests.get(
                f"{_api}/{container_id}",
                params={"fields": "status_code", "access_token": ch.instagram_access_token},
                timeout=15,
            )
            if status_resp.json().get("status_code") == "FINISHED":
                break

        pub_resp = requests.post(
            f"{_api}/{ch.instagram_user_id}/media_publish",
            params={"creation_id": container_id, "access_token": ch.instagram_access_token},
            timeout=30,
        )
        pub_data = pub_resp.json()
        if pub_resp.status_code != 200 or "id" not in pub_data:
            error_msg = pub_data.get("error", {}).get("message", pub_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao publicar Reel: {error_msg}")

        v.published = True
        v.ig_media_id = pub_data["id"]
        db.commit()
        db.refresh(v)
        return video_to_schema(v)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {str(e)}")


# ---------------------------------------------------------------------------
# Video Projects
# ---------------------------------------------------------------------------

@router.post("/api/video-projects", response_model=VideoProjectOut)
def create_video_project(
    data: CreateVideoProjectRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(data.channel_id, current_user, db)
    v = db.query(VideoDB).filter(VideoDB.id == data.video_id, VideoDB.channel_id == ch.id).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vídeo não encontrado")
    project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    vp = VideoProjectDB(
        id=project_id, channel_id=ch.id, user_id=current_user.id,
        title=f"Projeto {ch.name}",
        clip_ids=json.dumps([data.video_id]),
        clip_urls=json.dumps({data.video_id: v.video_path}),
        root_video_id=data.video_id,
    )
    db.add(vp)
    db.commit()
    db.refresh(vp)
    return video_project_to_schema(vp, db)


@router.get("/api/video-projects/{project_id}", response_model=VideoProjectOut)
def get_video_project(
    project_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return video_project_to_schema(_get_project_or_404(project_id, current_user, db), db)


@router.put("/api/video-projects/{project_id}/clips", response_model=VideoProjectOut)
def update_video_project_clips(
    project_id: str,
    data: UpdateVideoProjectClipsRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vp = _get_project_or_404(project_id, current_user, db)
    vp.clip_ids = json.dumps(data.clip_ids)
    db.commit()
    db.refresh(vp)
    return video_project_to_schema(vp, db)


@router.post("/api/video-projects/{project_id}/add-video", response_model=VideoProjectOut)
def add_video_to_project(
    project_id: str,
    data: AddVideoToProjectRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vp = _get_project_or_404(project_id, current_user, db)
    v = get_video_or_404(data.video_id, current_user, db)

    try:
        clip_ids = json.loads(vp.clip_ids or "[]")
    except Exception:
        clip_ids = []
    if not vp.root_video_id and clip_ids:
        vp.root_video_id = clip_ids[0]
    if data.video_id not in clip_ids:
        clip_ids.append(data.video_id)
        vp.clip_ids = json.dumps(clip_ids)

    try:
        clip_urls = json.loads(vp.clip_urls or "{}")
    except Exception:
        clip_urls = {}
    if data.video_id not in clip_urls:
        clip_urls[data.video_id] = v.video_path
        vp.clip_urls = json.dumps(clip_urls)

    v.is_project_clip = True
    db.commit()
    db.refresh(vp)
    return video_project_to_schema(vp, db)


@router.post("/api/video-projects/{project_id}/generate", response_model=VideoProjectOut)
def generate_project_clip(
    project_id: str,
    data: GenerateProjectClipRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vp = _get_project_or_404(project_id, current_user, db)
    ch = get_channel_or_404(vp.channel_id, current_user, db)
    s = get_or_create_settings(current_user, db)

    if not AZURE_SORA_ENDPOINT or not AZURE_SORA_API_KEY:
        raise HTTPException(status_code=400, detail="Sora não configurado.")

    base_prompt = ch.image_generation_prompt or f"Instagram Reel for channel '{ch.name}'. Theme: {ch.objective}."
    prompt = (base_prompt + (f" {data.additional_prompt}" if data.additional_prompt else ""))[:4000]

    try:
        create_resp = requests.post(
            AZURE_SORA_ENDPOINT,
            headers=_sora_headers(),
            json={"prompt": prompt, "model": "sora-2", "size": data.size, "seconds": str(data.seconds)},
            timeout=30,
        )
        if not create_resp.ok:
            raise HTTPException(status_code=502, detail=f"Falha ao criar job Sora: {create_resp.status_code} - {create_resp.text}")
    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Falha ao criar job Sora: {str(e)}")

    job = create_resp.json()
    job_id = job.get("id") or job.get("job_id") or job.get("generation_id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"Resposta inesperada do Sora: {job}")

    caption = ""
    try:
        client = get_azure_client(s)
        cap_resp = client.chat.completions.create(
            model=s.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": "Você é um especialista em conteúdo para Instagram."},
                {"role": "user", "content": f"Crie uma legenda curta para um clipe: {data.additional_prompt or prompt[:200]}. Retorne apenas a legenda."},
            ],
            max_tokens=200, temperature=0.7,
        )
        caption = cap_resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Caption generation failed: {e}")

    poll_url = f"{AZURE_SORA_ENDPOINT}/{job_id}"
    deadline = datetime.now().timestamp() + 240
    completed = False
    while datetime.now().timestamp() < deadline:
        time.sleep(5)
        try:
            result = requests.get(poll_url, headers=_sora_headers(), timeout=15).json()
            status = result.get("status", "")
        except Exception:
            continue
        if status == "completed":
            completed = True
            break
        if status in ("failed", "error", "cancelled"):
            err_obj = result.get("error") or {}
            err = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
            raise HTTPException(status_code=502, detail=f"Sora falhou: {err or status}")

    if not completed:
        raise HTTPException(status_code=504, detail="Timeout aguardando o Sora.")

    video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    try:
        dl = requests.get(f"{AZURE_SORA_ENDPOINT}/{job_id}/content", headers=_sora_headers(), timeout=120, allow_redirects=True)
        dl.raise_for_status()
        blob_url = upload_bytes_to_blob(dl.content, f"videos/{video_id}.mp4", "video/mp4")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao baixar vídeo: {str(e)}")

    v = VideoDB(
        id=video_id, channel_id=ch.id, channel_name=ch.name,
        prompt=prompt, caption=caption, video_path=blob_url,
        duration_seconds=data.seconds, size=data.size,
        published=False, is_project_clip=True,
    )
    db.add(v)

    try:
        clip_ids = json.loads(vp.clip_ids or "[]")
    except Exception:
        clip_ids = []
    clip_ids.append(video_id)
    vp.clip_ids = json.dumps(clip_ids)

    try:
        clip_urls = json.loads(vp.clip_urls or "{}")
    except Exception:
        clip_urls = {}
    clip_urls[video_id] = blob_url
    vp.clip_urls = json.dumps(clip_urls)

    db.commit()
    db.refresh(vp)
    return video_project_to_schema(vp, db)


@router.post("/api/video-projects/{project_id}/save", response_model=VideoProjectOut)
def save_video_project(
    project_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vp = _get_project_or_404(project_id, current_user, db)
    try:
        clip_ids = json.loads(vp.clip_ids or "[]")
    except Exception:
        clip_ids = []
    if not clip_ids:
        raise HTTPException(status_code=400, detail="Projeto sem clipes")

    clips = [v for cid in clip_ids if (v := db.query(VideoDB).filter(VideoDB.id == cid).first()) and v.video_path]
    if not clips:
        raise HTTPException(status_code=400, detail="Nenhum clipe válido encontrado")

    try:
        clip_url_map = json.loads(vp.clip_urls or "{}")
    except Exception:
        clip_url_map = {}

    if len(clips) == 1:
        merged_url = clip_url_map.get(clips[0].id) or clips[0].video_path
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_files = []
            for i, v in enumerate(clips):
                clip_path = os.path.join(tmpdir, f"clip_{i}.mp4")
                source_url = clip_url_map.get(v.id) or v.video_path
                try:
                    dl = requests.get(source_url, timeout=120, stream=True)
                    dl.raise_for_status()
                    with open(clip_path, "wb") as f:
                        for chunk in dl.iter_content(chunk_size=65536):
                            f.write(chunk)
                    clip_files.append(clip_path)
                except Exception as e:
                    raise HTTPException(status_code=502, detail=f"Erro ao baixar clipe: {str(e)}")

            concat_list = os.path.join(tmpdir, "concat.txt")
            with open(concat_list, "w") as f:
                for cp in clip_files:
                    f.write(f"file '{cp}'\n")

            output_path = os.path.join(tmpdir, "merged.mp4")
            result = subprocess.run(
                ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", output_path, "-y"],
                capture_output=True, timeout=180,
            )
            if result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Falha ao mesclar vídeos: {result.stderr.decode(errors='replace')}")

            export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            with open(output_path, "rb") as f:
                merged_url = upload_bytes_to_blob(f.read(), f"videos/{export_id}.mp4", "video/mp4")

    if not vp.root_video_id and clip_ids:
        vp.root_video_id = clip_ids[0]

    if vp.exported_video_id:
        exp_v = db.query(VideoDB).filter(VideoDB.id == vp.exported_video_id).first()
        if exp_v:
            exp_v.video_path = merged_url
            exp_v.duration_seconds = sum(c.duration_seconds or 0 for c in clips)
    else:
        root_v = db.query(VideoDB).filter(VideoDB.id == vp.root_video_id).first() if vp.root_video_id else None
        export_video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        exp_v = VideoDB(
            id=export_video_id, channel_id=vp.channel_id,
            channel_name=root_v.channel_name if root_v else "",
            prompt=root_v.prompt if root_v else "",
            caption=root_v.caption if root_v else "",
            video_path=merged_url,
            duration_seconds=sum(c.duration_seconds or 0 for c in clips),
            size=clips[0].size if clips else "720x1280",
            published=False, is_project_clip=False,
        )
        db.add(exp_v)
        vp.exported_video_id = export_video_id
        if root_v:
            root_v.is_project_clip = True

    vp.exported_path = merged_url
    db.commit()
    db.refresh(vp)
    return video_project_to_schema(vp, db)


@router.post("/api/video-projects/{project_id}/export")
def export_video_project(
    project_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    vp = _get_project_or_404(project_id, current_user, db)
    try:
        clip_ids = json.loads(vp.clip_ids or "[]")
    except Exception:
        clip_ids = []
    if not clip_ids:
        raise HTTPException(status_code=400, detail="Projeto sem clipes")

    clips = [v for cid in clip_ids if (v := db.query(VideoDB).filter(VideoDB.id == cid).first()) and v.video_path]
    if not clips:
        raise HTTPException(status_code=400, detail="Nenhum clipe válido encontrado")

    if len(clips) == 1:
        vp.exported_path = clips[0].video_path
        db.commit()
        return {"exported_url": clips[0].video_path}

    with tempfile.TemporaryDirectory() as tmpdir:
        clip_files = []
        for i, v in enumerate(clips):
            clip_path = os.path.join(tmpdir, f"clip_{i}.mp4")
            try:
                dl = requests.get(v.video_path, timeout=120, stream=True)
                dl.raise_for_status()
                with open(clip_path, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=65536):
                        f.write(chunk)
                clip_files.append(clip_path)
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Erro ao baixar clipe {v.id}: {str(e)}")

        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            for cp in clip_files:
                f.write(f"file '{cp}'\n")

        output_path = os.path.join(tmpdir, "merged.mp4")
        result = subprocess.run(
            ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", output_path, "-y"],
            capture_output=True, timeout=180,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Falha ao mesclar vídeos: {result.stderr.decode(errors='replace')}")

        export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        with open(output_path, "rb") as f:
            blob_url = upload_bytes_to_blob(f.read(), f"videos/{export_id}.mp4", "video/mp4")

    vp.exported_path = blob_url
    db.commit()
    return {"exported_url": blob_url}

import json
import os
import subprocess
import tempfile
import threading
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
from models import UserDB, ChannelDB, VideoDB, VideoProjectDB, VideoJobDB
from schemas import (
    GenerateVideoRequest, SavedVideo, UpdateVideoCaptionRequest,
    VideoProjectOut, CreateVideoProjectRequest, UpdateVideoProjectClipsRequest,
    GenerateProjectClipRequest, AddVideoToProjectRequest, VideoJobOut,
)
from services.azure_ai import generate_image_bytes, identify_main_person_bbox, create_inpaint_mask
from services.blob_storage import upload_bytes_to_blob
from services.replicate_ai import generate_with_lora, generate_video_from_image, generate_talking_head
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


def _run_character_video_job(job_id: str, user_id: int, channel_id: str, additional_prompt: str, image_size: str = "1024x1792", voice_script: str = ""):
    """
    Roda o pipeline GPT-Image-2 → LoRA inpainting → MiniMax em background thread.
    Usa sua própria sessão de DB — não reutiliza a sessão do request (não é thread-safe).
    """
    from database import SessionLocal as _SL
    db = _SL()
    try:
        ch = db.query(ChannelDB).filter(ChannelDB.id == channel_id).first()
        user = db.query(UserDB).filter(UserDB.id == user_id).first()
        s = get_or_create_settings(user, db)

        frame_id = f"charframe_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        base_prompt = ch.image_generation_prompt or f"Portrait photo of a person, {ch.objective}"

        # Step 1: GPT-Image-2 — cena rica com pessoa
        scene_prompt = (
            f"{base_prompt}, portrait photo, person facing camera, upper body visible"
            + (f", {additional_prompt}" if additional_prompt else "")
        )[:4000]
        print(f"[CHARACTER JOB {job_id}] Step 1 – GPT-Image-2 | {scene_prompt[:60]}")
        img_w, img_h = (int(x) for x in image_size.split("x"))
        scene_bytes = generate_image_bytes(scene_prompt, ch, s, db, width=img_w, height=img_h)
        scene_url = upload_bytes_to_blob(scene_bytes, f"temp/{frame_id}_scene.png", "image/png")

        total_credits = register_credit_usage(
            db=db, user_id=user_id, channel_id=channel_id,
            resource_type="video", resource_id=None,
            operation_type="image_generation", model_name="gpt-image-2",
            images_count=1, metadata={"step": "scene_generation"},
        )

        # Step 2: LoRA img2img — aplica identidade do personagem na cena inteira (sem máscara)
        # Inpainting com elipse criava seam visível; img2img preserva o background via prompt_strength
        lora_prompt = (
            f"TOK, {base_prompt}"
            + (f", {additional_prompt}" if additional_prompt else "")
        )[:2000]
        print(f"[CHARACTER JOB {job_id}] Step 2 – LoRA img2img")
        lora_image_bytes = generate_with_lora(
            prompt=lora_prompt,
            lora_ref=ch.lora_version,
            api_key=REPLICATE_API_KEY,
            base_image_url=scene_url,
            mask_url=None,
        )
        char_scene_url = upload_bytes_to_blob(lora_image_bytes, f"temp/{frame_id}_char.png", "image/png")

        total_credits += register_credit_usage(
            db=db, user_id=user_id, channel_id=channel_id,
            resource_type="video", resource_id=None,
            operation_type="image_generation", model_name="replicate/flux-lora",
            images_count=1, metadata={"step": "lora_img2img"},
        )

        # Step 3: MiniMax — anima a cena (corpo se move)
        motion_prompt = (
            f"{base_prompt}" + (f", {additional_prompt}" if additional_prompt else "")
        )[:2000]
        print(f"[CHARACTER JOB {job_id}] Step 3 – minimax/video-01")
        video_bytes = generate_video_from_image(
            image_url=char_scene_url,
            prompt=motion_prompt,
            api_key=REPLICATE_API_KEY,
        )

        # Step 4: Áudio (TTS) + SadTalker sobre o vídeo MiniMax — lipsync com still_mode=True
        from config import ELEVENLABS_API_KEY
        if ch.elevenlabs_voice_id and voice_script and ELEVENLABS_API_KEY:
            from services.elevenlabs import generate_tts, mix_audio_into_video
            tts_bytes = None
            try:
                print(f"[CHARACTER JOB {job_id}] Step 4a – ElevenLabs TTS ({len(voice_script)} chars)")
                tts_bytes = generate_tts(voice_script, ch.elevenlabs_voice_id, ELEVENLABS_API_KEY)
                print(f"[CHARACTER JOB {job_id}] Step 4a – TTS ok: {len(tts_bytes)} bytes")
            except Exception as e:
                print(f"[CHARACTER JOB {job_id}] Step 4a – TTS FALHOU: {e}")

            if tts_bytes:
                audio_id = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                audio_url = upload_bytes_to_blob(tts_bytes, f"temp/{audio_id}.mp3", "audio/mpeg")
                video_tmp_id = f"video_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                video_tmp_url = upload_bytes_to_blob(video_bytes, f"temp/{video_tmp_id}.mp4", "video/mp4")
                try:
                    print(f"[CHARACTER JOB {job_id}] Step 4b – SadTalker lipsync")
                    video_bytes = generate_talking_head(video_tmp_url, audio_url, REPLICATE_API_KEY)
                    print(f"[CHARACTER JOB {job_id}] Step 4b – SadTalker ok: {len(video_bytes)} bytes")
                except Exception as e:
                    print(f"[CHARACTER JOB {job_id}] Step 4b – SadTalker FALHOU ({e}), fallback ffmpeg mix")
                    try:
                        video_bytes = mix_audio_into_video(video_bytes, tts_bytes)
                        print(f"[CHARACTER JOB {job_id}] Step 4b – ffmpeg mix ok")
                    except Exception as e2:
                        print(f"[CHARACTER JOB {job_id}] Step 4b – ffmpeg FALHOU: {e2}")

        # Step 5: Legenda
        caption = ""
        try:
            client = get_azure_client(s)
            conceito = additional_prompt or voice_script or base_prompt
            text_prompt = ch.text_generation_prompt or (
                f'Crie uma legenda para um Instagram Reel do canal "{ch.name}".\n'
                f"Objetivo: {ch.objective}\nConceito: {conceito}\n"
                "Escreva uma legenda envolvente com emojis e hashtags, 80-150 palavras. Retorne só o texto."
            )
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
                db=db, user_id=user_id, channel_id=channel_id,
                resource_type="video", resource_id=None,
                operation_type="text_generation", model_name=s.azure_openai_deployment_name,
                input_tokens=cap_usage.prompt_tokens, output_tokens=cap_usage.completion_tokens,
                metadata={"step": "caption_generation"},
            )
        except Exception as e:
            print(f"[CHARACTER JOB {job_id}] Caption failed: {e}")

        # Upload vídeo
        video_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        blob_url = upload_bytes_to_blob(video_bytes, f"videos/{video_id}.mp4", "video/mp4")

        total_credits += register_credit_usage(
            db=db, user_id=user_id, channel_id=channel_id,
            resource_type="video", resource_id=video_id,
            operation_type="video_generation", model_name="minimax/video-01",
            video_seconds=_VIDEO_WITH_CHARACTER_DURATION,
            metadata={"scene_url": char_scene_url[:60], "size": image_size},
        )

        v = VideoDB(
            id=video_id, channel_id=channel_id, channel_name=ch.name,
            prompt=lora_prompt, caption=caption, video_path=blob_url,
            duration_seconds=_VIDEO_WITH_CHARACTER_DURATION, size=image_size,
            published=False, credits_consumed=total_credits,
        )
        db.add(v)

        job = db.query(VideoJobDB).filter(VideoJobDB.id == job_id).first()
        if job:
            job.status = "completed"
            job.video_id = video_id

        db.commit()
        print(f"[CHARACTER JOB {job_id}] ✓ Concluído: video_id={video_id}")

    except Exception as e:
        import traceback
        print(f"[CHARACTER JOB {job_id}] ✗ Falhou: {e}")
        traceback.print_exc()
        try:
            job = db.query(VideoJobDB).filter(VideoJobDB.id == job_id).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)[:500]
                db.commit()
        except Exception as e2:
            print(f"[CHARACTER JOB {job_id}] ✗ Erro ao salvar falha: {e2}")
    finally:
        db.close()


@router.post("/api/videos/generate-with-character", response_model=VideoJobOut)
def generate_video_with_character(
    data: GenerateVideoRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Inicia geração de vídeo com personagem de forma assíncrona.
    Retorna job_id imediatamente — use GET /api/videos/character-job/{job_id} para polling.
    Pipeline rodando em background: GPT-Image-2 → LoRA inpainting → MiniMax → legenda.
    """
    ch = get_channel_or_404(data.channel_id, current_user, db)

    if ch.lora_status != "succeeded" or not ch.lora_version:
        raise HTTPException(
            status_code=400,
            detail="Canal sem personagem treinado. Treine o LoRA primeiro nas configurações do canal.",
        )
    if not REPLICATE_API_KEY:
        raise HTTPException(status_code=400, detail="REPLICATE_API_KEY não configurado no servidor.")

    job_id = f"charjob_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    job = VideoJobDB(id=job_id, user_id=current_user.id, channel_id=ch.id, status="processing")
    db.add(job)
    db.commit()

    thread = threading.Thread(
        target=_run_character_video_job,
        args=(job_id, current_user.id, ch.id, data.additional_prompt or "", data.size or "1024x1792", data.voice_script or ""),
        daemon=True,
    )
    thread.start()
    print(f"[CHARACTER JOB {job_id}] Iniciado em background thread")

    return VideoJobOut(job_id=job_id, status="processing")


@router.get("/api/videos/character-job/{job_id}", response_model=VideoJobOut)
def get_character_job(
    job_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Polling endpoint — retorna status do job e o vídeo quando concluído."""
    job = db.query(VideoJobDB).filter(
        VideoJobDB.id == job_id,
        VideoJobDB.user_id == current_user.id,
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    video = None
    if job.status == "completed" and job.video_id:
        v = db.query(VideoDB).filter(VideoDB.id == job.video_id).first()
        if v:
            video = video_to_schema(v)

    return VideoJobOut(
        job_id=job_id,
        status=job.status,
        video=video,
        error=job.error_message,
    )


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

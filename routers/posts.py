import time
from datetime import datetime
from typing import List

import requests
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import GPT_IMAGE_2_API_KEY
from database import get_db
from dependencies import (
    get_current_user, get_or_create_settings, get_azure_client,
    get_channel_or_404, get_post_or_404,
)
from models import UserDB, ChannelDB, PostDB, MediaInsightsDB
from schemas import (
    GeneratePostRequest, Post, SavedPost, UpdatePostRequest, GeneratePostImageRequest,
)
from services.azure_ai import generate_image_bytes, get_reference_context
from services.blob_storage import upload_bytes_to_blob
from services.converters import post_to_schema, insights_to_schema
from services.credits import register_credit_usage
from services.instagram import ig_api_base

router = APIRouter(tags=["posts"])


@router.get("/api/posts", response_model=List[SavedPost])
def get_posts(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_channel_ids = [
        ch.id for ch in db.query(ChannelDB.id).filter(ChannelDB.user_id == current_user.id).all()
    ]
    posts = (
        db.query(PostDB)
        .filter(PostDB.channel_id.in_(user_channel_ids))
        .order_by(PostDB.created_at.desc())
        .all()
    )
    post_ids = [p.id for p in posts]
    ins_map = {}
    if post_ids:
        for ins in db.query(MediaInsightsDB).filter(
            MediaInsightsDB.media_type == "post",
            MediaInsightsDB.media_id.in_(post_ids),
        ).all():
            ins_map[ins.media_id] = ins
    return [post_to_schema(p, ins_map.get(p.id)) for p in posts]


@router.get("/api/posts/{post_id}", response_model=SavedPost)
def get_post(
    post_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return post_to_schema(get_post_or_404(post_id, current_user, db))


@router.patch("/api/posts/{post_id}", response_model=SavedPost)
@router.post("/api/posts/{post_id}/save", response_model=SavedPost)
def update_post(
    post_id: str,
    data: UpdatePostRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = get_post_or_404(post_id, current_user, db)
    if data.text is not None:
        p.text = data.text
    if data.image_path is not None:
        p.image_path = data.image_path
    if data.published is not None:
        p.published = data.published
    db.commit()
    db.refresh(p)
    return post_to_schema(p)


@router.post("/api/posts/generate", response_model=Post)
def generate_post(
    data: GeneratePostRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ch = get_channel_or_404(data.channel_id, current_user, db)
    s = get_or_create_settings(current_user, db)
    client = get_azure_client(s)

    text_prompt = ch.text_generation_prompt or f"""Create an engaging Instagram post for channel "{ch.name}".
Objective: {ch.objective}
Requirements: engaging, authentic, relevant hashtags, 100-200 words.
Return only the post text."""
    if data.additional_prompt:
        text_prompt += f"\n\nAdditional instructions: {data.additional_prompt}"

    text_resp = client.chat.completions.create(
        model=s.azure_openai_deployment_name,
        messages=[
            {"role": "system", "content": "You are a professional social media content creator."},
            {"role": "user", "content": text_prompt},
        ],
        max_tokens=500, temperature=0.7,
    )
    post_text = text_resp.choices[0].message.content.strip()
    text_usage = text_resp.usage
    total_credits = register_credit_usage(
        db=db, user_id=current_user.id, channel_id=ch.id,
        resource_type="post", resource_id=None,
        operation_type="text_generation", model_name=s.azure_openai_deployment_name,
        input_tokens=text_usage.prompt_tokens, output_tokens=text_usage.completion_tokens,
        metadata={"step": "post_text_generation"},
    )

    subj_resp = client.chat.completions.create(
        model=s.azure_openai_deployment_name,
        messages=[
            {"role": "system", "content": "You identify the main subject of social media posts."},
            {"role": "user", "content": f"Identify the main subject of this post in 2-5 words max:\n\n{post_text}\n\nReturn only the subject."},
        ],
        max_tokens=20, temperature=0.3,
    )
    main_subject = subj_resp.choices[0].message.content.strip()
    subj_usage = subj_resp.usage
    total_credits += register_credit_usage(
        db=db, user_id=current_user.id, channel_id=ch.id,
        resource_type="post", resource_id=None,
        operation_type="text_generation", model_name=s.azure_openai_deployment_name,
        input_tokens=subj_usage.prompt_tokens, output_tokens=subj_usage.completion_tokens,
        metadata={"step": "subject_extraction"},
    )

    image_prompt = None
    post_id = f"post_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    blob_url = ""
    image_error = None
    model_ready = s.azure_openai_image_endpoint or (ch.image_model == "gpt-image-2" and GPT_IMAGE_2_API_KEY)
    if not model_ready:
        image_error = "Endpoint de geração de imagem não configurado. Configure em Configurações → Azure OpenAI Image Endpoint."
        print(f"[IMAGE] Skipping image generation: {image_error}")
    else:
        # Combina prompts do canal: visual (image_prompt) + contexto de personagens (text_prompt) + assunto do post
        base_image = ch.image_generation_prompt or f"Instagram post image for {ch.name}. Theme: {ch.objective}."
        base_text = ch.text_generation_prompt or ""
        parts = [p for p in [base_image, base_text] if p.strip()]
        image_prompt = "\n\n".join(parts)
        image_prompt += f"\n\nTema específico desta imagem: {main_subject}"
        if data.additional_prompt:
            image_prompt += f"\n\n{data.additional_prompt}"
        image_prompt += get_reference_context(ch.id, db)
        print(f"[IMAGE] Prompt ({len(image_prompt)} chars): {image_prompt[:300]}")
        try:
            img_bytes = generate_image_bytes(image_prompt, ch, s, db)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            blob_url = upload_bytes_to_blob(img_bytes, f"posts/{post_id}_{ts}.png", "image/png")
            image_model = ch.image_model or "mai"
            total_credits += register_credit_usage(
                db=db, user_id=current_user.id, channel_id=ch.id,
                resource_type="post", resource_id=post_id,
                operation_type="image_generation", model_name=image_model,
                images_count=1, metadata={"prompt_length": len(image_prompt)},
            )
        except Exception as e:
            image_error = str(e)
            print(f"Image generation failed: {e}")

    p = PostDB(
        id=post_id, channel_id=ch.id, channel_name=ch.name,
        text=post_text, image_path=blob_url, prompt=image_prompt,
        published=False, credits_consumed=total_credits,
    )
    db.add(p)
    db.commit()
    db.execute(
        text("UPDATE credit_usage SET resource_id = :post_id WHERE resource_id IS NULL AND user_id = :user_id AND channel_id = :channel_id"),
        {"post_id": post_id, "user_id": current_user.id, "channel_id": ch.id},
    )
    db.commit()
    return Post(id=post_id, text=post_text, image_url=blob_url, image_error=image_error)


@router.post("/api/posts/{post_id}/image/upload")
def upload_post_image(
    post_id: str,
    file: UploadFile = File(...),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = get_post_or_404(post_id, current_user, db)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
    data = file.file.read()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = (file.content_type or "image/png").split("/")[-1].replace("jpeg", "jpg")
    blob_url = upload_bytes_to_blob(data, f"posts/{post_id}_{ts}.{ext}", file.content_type or "image/png")
    p.image_path = blob_url
    db.commit()
    return {"success": True, "image_url": blob_url, "image_path": blob_url}


@router.post("/api/posts/{post_id}/image/generate")
def generate_post_image(
    post_id: str,
    data: GeneratePostImageRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = get_post_or_404(post_id, current_user, db)
    s = get_or_create_settings(current_user, db)
    if not s.azure_openai_image_endpoint:
        raise HTTPException(status_code=400, detail="Endpoint de imagem não configurado")

    ch = db.query(ChannelDB).filter(ChannelDB.id == data.channel_id).first()
    # Combina: prompt de imagem do canal + contexto de texto (quem são os personagens) + prompt do usuário
    image_prompt = (ch.image_generation_prompt or "") if ch else ""
    text_context = (ch.text_generation_prompt or "") if ch else ""
    parts = [p for p in [image_prompt, text_context, data.prompt] if p.strip()]
    full_prompt = "\n\n".join(parts)
    if ch:
        full_prompt += get_reference_context(ch.id, db)

    try:
        img_bytes = generate_image_bytes(full_prompt, ch, s, db)
    except HTTPException:
        raise
    except Exception as e:
        # Fallback: tenta só com o prompt do usuário + referências visuais
        if data.prompt:
            try:
                fallback_prompt = data.prompt + (f"\n\n{get_reference_context(ch.id, db)}" if ch else "")
                img_bytes = generate_image_bytes(fallback_prompt, ch, s, db)
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Falha ao gerar imagem: {str(e2)}")
        else:
            raise HTTPException(status_code=500, detail=f"Falha ao gerar imagem: {str(e)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    blob_url = upload_bytes_to_blob(img_bytes, f"posts/{post_id}_{ts}.png", "image/png")
    p.image_path = blob_url
    p.prompt = full_prompt
    image_model = ch.image_model if ch else "mai"
    credits = register_credit_usage(
        db=db, user_id=current_user.id, channel_id=ch.id if ch else None,
        resource_type="post", resource_id=post_id,
        operation_type="image_generation", model_name=image_model,
        images_count=1, metadata={"prompt_length": len(full_prompt), "regenerated": True},
    )
    p.credits_consumed = (p.credits_consumed or 0.0) + credits
    db.commit()
    return {"success": True, "image_url": blob_url, "image_path": blob_url}


@router.post("/api/posts/{post_id}/publish")
def publish_post(
    post_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    p = get_post_or_404(post_id, current_user, db)
    ch = db.query(ChannelDB).filter(ChannelDB.id == p.channel_id).first()
    if not ch.instagram_user_id or not ch.instagram_access_token:
        raise HTTPException(status_code=400, detail="Instagram não configurado para este canal.")

    try:
        _api = ig_api_base(ch.instagram_access_token)
        create_resp = requests.post(
            f"{_api}/{ch.instagram_user_id}/media",
            params={"image_url": p.image_path, "caption": p.text, "access_token": ch.instagram_access_token},
            timeout=30,
        )
        create_data = create_resp.json()
        if create_resp.status_code != 200 or "id" not in create_data:
            error_msg = create_data.get("error", {}).get("message", create_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao criar container: {error_msg}")

        container_id = create_data["id"]
        for _ in range(15):
            time.sleep(2)
            status_resp = requests.get(
                f"{_api}/{container_id}",
                params={"fields": "status_code", "access_token": ch.instagram_access_token},
                timeout=15,
            )
            sc = status_resp.json().get("status_code", "")
            if sc == "FINISHED":
                break
            if sc == "ERROR":
                raise HTTPException(status_code=502, detail="Erro ao processar mídia no Instagram.")

        pub_resp = requests.post(
            f"{_api}/{ch.instagram_user_id}/media_publish",
            params={"creation_id": container_id, "access_token": ch.instagram_access_token},
            timeout=30,
        )
        pub_data = pub_resp.json()
        if pub_resp.status_code != 200 or "id" not in pub_data:
            error_msg = pub_data.get("error", {}).get("message", pub_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao publicar: {error_msg}")

        p.published = True
        p.ig_media_id = pub_data["id"]
        db.commit()
        return {"success": True, "instagram_post_id": pub_data["id"]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {str(e)}")

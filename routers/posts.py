import time
from datetime import datetime
from typing import List

import requests
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import GPT_IMAGE_2_API_KEY, AZURE_FOUNDRY_API_KEY, REPLICATE_API_KEY
from database import get_db
from dependencies import (
    get_current_user, get_or_create_settings, get_azure_client,
    get_channel_or_404, get_post_or_404,
)
from models import UserDB, ChannelDB, PostDB, MediaInsightsDB, ReferenceImageDB
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
    model_ready = bool(GPT_IMAGE_2_API_KEY)
    if not model_ready:
        image_error = "GPT_IMAGE_2_API_KEY não configurado no servidor."
        print(f"[IMAGE] Skipping image generation: {image_error}")
    else:
        base_image = ch.image_generation_prompt or f"Instagram post image for {ch.name}. Theme: {ch.objective}."
        base_text = ch.text_generation_prompt or ""
        parts = [p for p in [base_image, base_text] if p.strip()]
        image_prompt = "\n\n".join(parts)
        image_prompt += f"\n\nTema específico desta imagem: {main_subject}"
        if data.additional_prompt:
            image_prompt += f"\n\n{data.additional_prompt}"
        image_prompt += get_reference_context(ch.id, db)
        print(f"[IMAGE] Prompt ({len(image_prompt)} chars): {image_prompt[:300]}")

        def _try_generate(prompt: str) -> bytes:
            return generate_image_bytes(prompt, ch, s, db)

        # Tenta com prompt completo; se bloqueado por content safety, tenta versão simplificada
        prompts_to_try = [
            image_prompt,
            f"Artistic illustration: {main_subject}. Style: natural, candid photography.",
        ]
        img_bytes = None
        for attempt_prompt in prompts_to_try:
            try:
                img_bytes = _try_generate(attempt_prompt)
                image_prompt = attempt_prompt
                break
            except Exception as e:
                err_str = str(e)
                print(f"[IMAGE] Attempt failed: {err_str[:200]}")
                if "content_safety" not in err_str.lower() and "responsibleai" not in err_str.lower():
                    image_error = err_str
                    break
                image_error = "Prompt bloqueado pelo filtro de segurança do modelo. Considere revisar o prompt de imagem do canal."

        if img_bytes:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            blob_url = upload_bytes_to_blob(img_bytes, f"posts/{post_id}_{ts}.png", "image/png")
            image_model = "gpt-image-2"
            total_credits += register_credit_usage(
                db=db, user_id=current_user.id, channel_id=ch.id,
                resource_type="post", resource_id=post_id,
                operation_type="image_generation", model_name=image_model,
                images_count=1, metadata={"prompt_length": len(image_prompt)},
            )
            image_error = None

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
    print(f"\n{'='*80}")
    print(f"[IMAGE REGEN] Iniciando regeneração de imagem para post_id={post_id}")
    print(f"[IMAGE REGEN] User: {current_user.email}, Channel ID: {data.channel_id}")
    print(f"[IMAGE REGEN] Prompt do usuário: {data.prompt[:100] if data.prompt else '(vazio)'}...")
    
    try:
        p = get_post_or_404(post_id, current_user, db)
        print(f"[IMAGE REGEN] ✓ Post encontrado")
    except Exception as e:
        print(f"[IMAGE REGEN] ✗ ERRO ao buscar post: {str(e)}")
        raise
    
    try:
        s = get_or_create_settings(current_user, db)
        print(f"[IMAGE REGEN] ✓ Settings carregadas")
        print(f"[IMAGE REGEN] Image Endpoint: {s.azure_openai_image_endpoint[:50] if s.azure_openai_image_endpoint else 'NÃO CONFIGURADO'}...")
    except Exception as e:
        print(f"[IMAGE REGEN] ✗ ERRO ao carregar settings: {str(e)}")
        raise
    
    try:
        ch = db.query(ChannelDB).filter(ChannelDB.id == data.channel_id).first()
        print(f"[IMAGE REGEN] ✓ Canal encontrado: {ch.name if ch else 'None'}")
    except Exception as e:
        print(f"[IMAGE REGEN] ✗ ERRO ao buscar canal: {str(e)}")
        raise

    if not GPT_IMAGE_2_API_KEY:
        raise HTTPException(status_code=400, detail="GPT_IMAGE_2_API_KEY não configurado no servidor")

    try:
        image_prompt = (ch.image_generation_prompt or "") if ch else ""
        text_context = (ch.text_generation_prompt or "") if ch else ""
        parts = [p for p in [image_prompt, text_context, data.prompt] if p.strip()]
        full_prompt = "\n\n".join(parts)
        if ch:
            full_prompt += get_reference_context(ch.id, db)
        print(f"[IMAGE REGEN] ✓ Prompt construído ({len(full_prompt)} caracteres)")
        print(f"[IMAGE REGEN] Prompt preview: {full_prompt[:200]}...")
    except Exception as e:
        print(f"[IMAGE REGEN] ✗ ERRO ao construir prompt: {str(e)}")
        raise

    img_bytes = None
    prompts_to_try = [full_prompt]
    # Fallbacks progressivos caso o prompt completo seja bloqueado por content safety
    if data.prompt and full_prompt != data.prompt:
        prompts_to_try.append(data.prompt)
    prompts_to_try.append(f"Artistic photo illustration: {data.prompt or 'scene'}")
    
    print(f"[IMAGE REGEN] Tentará {len(prompts_to_try)} variações de prompt")

    last_error = None
    for i, attempt_prompt in enumerate(prompts_to_try, 1):
        print(f"[IMAGE REGEN] Tentativa {i}/{len(prompts_to_try)}: prompt de {len(attempt_prompt)} chars")
        try:
            print(f"[IMAGE REGEN]   Chamando generate_image_bytes...")
            img_bytes = generate_image_bytes(attempt_prompt, ch, s, db)
            full_prompt = attempt_prompt
            last_error = None
            print(f"[IMAGE REGEN] ✓ Imagem gerada com sucesso! ({len(img_bytes)} bytes)")
            break
        except HTTPException as he:
            print(f"[IMAGE REGEN] ✗ HTTPException capturada: status={he.status_code}, detail={he.detail}")
            raise
        except Exception as e:
            last_error = str(e)
            print(f"[IMAGE REGEN] ✗ Tentativa {i} falhou: {last_error[:300]}")
            if "content_safety" not in last_error.lower() and "responsibleai" not in last_error.lower():
                print(f"[IMAGE REGEN]   Erro não é de content safety - abortando tentativas")
                break
            print(f"[IMAGE REGEN]   Erro de content safety - tentando fallback...")

    if not img_bytes:
        error_msg = f"Falha ao gerar imagem após {len(prompts_to_try)} tentativas: {last_error}"
        print(f"[IMAGE REGEN] ✗ FALHA FINAL: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        blob_name = f"posts/{post_id}_{ts}.png"
        print(f"[IMAGE REGEN] Fazendo upload para blob: {blob_name}")
        blob_url = upload_bytes_to_blob(img_bytes, blob_name, "image/png")
        print(f"[IMAGE REGEN] ✓ Upload concluído: {blob_url[:80]}...")
    except Exception as e:
        print(f"[IMAGE REGEN] ✗ ERRO no upload do blob: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao fazer upload da imagem: {str(e)}")
    
    try:
        p.image_path = blob_url
        p.prompt = full_prompt
        image_model = "gpt-image-2"
        credits = register_credit_usage(
            db=db, user_id=current_user.id, channel_id=ch.id if ch else None,
            resource_type="post", resource_id=post_id,
            operation_type="image_generation", model_name=image_model,
            images_count=1, metadata={"prompt_length": len(full_prompt), "regenerated": True},
        )
        p.credits_consumed = (p.credits_consumed or 0.0) + credits
        db.commit()
        print(f"[IMAGE REGEN] ✓ Post atualizado no banco, créditos: {credits}")
    except Exception as e:
        print(f"[IMAGE REGEN] ✗ ERRO ao atualizar banco: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao salvar no banco de dados: {str(e)}")
    
    print(f"[IMAGE REGEN] ✓✓✓ SUCESSO TOTAL! Retornando URL")
    print(f"{'='*80}\n")
    return {"success": True, "image_url": blob_url, "image_path": blob_url}


@router.post("/api/posts/{post_id}/image/face-apply")
def face_apply_post_image(
    post_id: str,
    data: GeneratePostImageRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Aplica o rosto da imagem de referência do canal na imagem atual do post (face swap)."""
    if not REPLICATE_API_KEY:
        raise HTTPException(status_code=400, detail="REPLICATE_API_KEY não configurado no servidor")

    p = get_post_or_404(post_id, current_user, db)
    if not p.image_path:
        raise HTTPException(status_code=400, detail="Post não tem imagem para aplicar o rosto")

    ch = db.query(ChannelDB).filter(ChannelDB.id == data.channel_id).first()
    if not ch:
        raise HTTPException(status_code=404, detail="Canal não encontrado")

    refs = db.query(ReferenceImageDB).filter(
        ReferenceImageDB.channel_id == ch.id,
    ).order_by(ReferenceImageDB.created_at.desc()).limit(1).all()
    if not refs:
        raise HTTPException(status_code=400, detail="Canal sem imagens de referência. Adicione fotos do rosto na aba Referências.")

    target_url = p.image_path if p.image_path.startswith("http") else None
    if not target_url:
        raise HTTPException(status_code=400, detail="URL da imagem do post não é pública")

    print(f"[FACE_APPLY] post={post_id} target={target_url[:60]} swap={refs[0].blob_url[:60]}")

    from services.replicate_ai import apply_face_swap
    img_bytes = apply_face_swap(
        target_image_url=target_url,
        swap_image_url=refs[0].blob_url,
        api_key=REPLICATE_API_KEY,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    blob_url = upload_bytes_to_blob(img_bytes, f"posts/{post_id}_{ts}_face.png", "image/png")
    p.image_path = blob_url
    register_credit_usage(
        db=db, user_id=current_user.id, channel_id=ch.id,
        resource_type="post", resource_id=post_id,
        operation_type="face_apply", model_name="replicate/faceswap",
        images_count=1, metadata={"type": "face_swap"},
    )
    db.commit()
    print(f"[FACE_APPLY] ✓ Concluído: {blob_url[:60]}")
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

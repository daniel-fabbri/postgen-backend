import time
import requests
from fastapi import HTTPException

_REPLICATE_API = "https://api.replicate.com/v1"
# Hash fixo da versão — obtido via GET /v1/models/fofr/consistent-character
_CONSISTENT_CHARACTER_VERSION = "9c77a3c2f884193fcee4d89645f02a0b9def9434f9e03cb98460456b831c8772"

# Timeout total de polling: 60 tentativas × 3s = 3 minutos
_POLL_INTERVAL = 3
_POLL_MAX = 60


def generate_consistent_character(
    prompt: str,
    subject_url: str,
    api_key: str,
) -> bytes:
    """
    Chama fofr/consistent-character no Replicate.
    subject_url: URL pública da foto de referência do personagem.
    prompt: descrição da cena (sem descrever o rosto — ele vem do subject).
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="REPLICATE_API_KEY não configurado no servidor")

    print(f"[REPLICATE] Iniciando consistent-character | subject={subject_url[:60]} | prompt={prompt[:80]}")

    create_resp = requests.post(
        f"{_REPLICATE_API}/predictions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Prefer": "wait",  # espera até 60s na própria resposta antes de polling
        },
        json={
            "version": _CONSISTENT_CHARACTER_VERSION,
            "input": {
                "prompt": prompt,
                "subject": subject_url,
                "number_of_outputs": 1,
                "output_format": "png",
                "output_quality": 95,
                "disable_safety_checker": False,
            }
        },
        timeout=90,
    )

    if not create_resp.ok:
        raise HTTPException(
            status_code=create_resp.status_code,
            detail=f"Replicate error: {create_resp.text[:400]}",
        )

    prediction = create_resp.json()
    status = prediction.get("status")
    prediction_id = prediction.get("id")
    print(f"[REPLICATE] Prediction {prediction_id} status={status}")

    # Se o header Prefer:wait resolveu na hora, já temos o resultado
    if status == "succeeded":
        return _download_output(prediction)

    if status in ("failed", "canceled"):
        raise HTTPException(status_code=500, detail=f"Replicate falhou: {prediction.get('error')}")

    # Polling
    for attempt in range(_POLL_MAX):
        time.sleep(_POLL_INTERVAL)
        poll = requests.get(
            f"{_REPLICATE_API}/predictions/{prediction_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if not poll.ok:
            continue
        result = poll.json()
        status = result.get("status")
        print(f"[REPLICATE] Poll {attempt+1}/{_POLL_MAX} status={status}")

        if status == "succeeded":
            return _download_output(result)

        if status in ("failed", "canceled"):
            raise HTTPException(status_code=500, detail=f"Replicate falhou: {result.get('error')}")

    raise HTTPException(status_code=504, detail="Replicate prediction não concluiu em 3 minutos")


_FACESWAP_VERSION = "9a4298548422074c3f57258c5d544497314ae4112df80d116f0d2109e843d20d"


def apply_face_swap(
    target_image_url: str,
    swap_image_url: str,
    api_key: str,
) -> bytes:
    """
    Aplica o rosto de swap_image_url na cena de target_image_url.
    target_image_url: imagem do post (cena gerada)
    swap_image_url:   foto de referência do rosto
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="REPLICATE_API_KEY não configurado no servidor")

    print(f"[REPLICATE] face-swap | target={target_image_url[:60]} | swap={swap_image_url[:60]}")

    create_resp = requests.post(
        f"{_REPLICATE_API}/predictions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        },
        json={
            "version": _FACESWAP_VERSION,
            "input": {
                "target_image": target_image_url,
                "swap_image": swap_image_url,
            },
        },
        timeout=90,
    )

    if not create_resp.ok:
        raise HTTPException(
            status_code=create_resp.status_code,
            detail=f"Replicate face-swap error: {create_resp.text[:400]}",
        )

    prediction = create_resp.json()
    status = prediction.get("status")
    prediction_id = prediction.get("id")
    print(f"[REPLICATE] face-swap prediction {prediction_id} status={status}")

    if status == "succeeded":
        return _download_output(prediction)
    if status in ("failed", "canceled"):
        raise HTTPException(status_code=500, detail=f"Replicate face-swap falhou: {prediction.get('error')}")

    for attempt in range(_POLL_MAX):
        time.sleep(_POLL_INTERVAL)
        poll = requests.get(
            f"{_REPLICATE_API}/predictions/{prediction_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if not poll.ok:
            continue
        result = poll.json()
        status = result.get("status")
        print(f"[REPLICATE] face-swap poll {attempt+1} status={status}")
        if status == "succeeded":
            return _download_output(result)
        if status in ("failed", "canceled"):
            raise HTTPException(status_code=500, detail=f"Replicate face-swap falhou: {result.get('error')}")

    raise HTTPException(status_code=504, detail="Replicate face-swap não concluiu em 3 minutos")


def generate_flux_pulid(prompt: str, face_image_url: str, api_key: str) -> bytes:
    """
    Gera imagem via fofr/flux-pulid: identidade preservada desde a geração (não é face swap).
    face_image_url: URL pública da foto de referência do rosto.
    prompt: descrição da cena.
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="REPLICATE_API_KEY não configurado no servidor")

    print(f"[REPLICATE] flux-pulid | face={face_image_url[:60]} | prompt={prompt[:80]}")

    create_resp = requests.post(
        f"{_REPLICATE_API}/predictions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        },
        json={
            "version": "8baa7ef2255075b46f4d91cd238c21d31181b3e6a864463f967960bb0112525b",
            "input": {
                "prompt": prompt,
                "main_face_image": face_image_url,
                "num_outputs": 1,
                "num_steps": 20,
                "output_format": "png",
                "output_quality": 95,
                "guidance_scale": 4,
                "id_weight": 1,
                "start_step": 4,
            }
        },
        timeout=90,
    )

    if not create_resp.ok:
        raise HTTPException(
            status_code=create_resp.status_code,
            detail=f"Replicate flux-pulid error: {create_resp.text[:400]}",
        )

    prediction = create_resp.json()
    status = prediction.get("status")
    prediction_id = prediction.get("id")
    print(f"[REPLICATE] flux-pulid prediction {prediction_id} status={status}")

    if status == "succeeded":
        return _download_output(prediction)
    if status in ("failed", "canceled"):
        raise HTTPException(status_code=500, detail=f"Replicate flux-pulid falhou: {prediction.get('error')}")

    for attempt in range(_POLL_MAX):
        time.sleep(_POLL_INTERVAL)
        poll = requests.get(
            f"{_REPLICATE_API}/predictions/{prediction_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if not poll.ok:
            continue
        result = poll.json()
        status = result.get("status")
        print(f"[REPLICATE] flux-pulid poll {attempt+1} status={status}")
        if status == "succeeded":
            return _download_output(result)
        if status in ("failed", "canceled"):
            raise HTTPException(status_code=500, detail=f"Replicate flux-pulid falhou: {result.get('error')}")

    raise HTTPException(status_code=504, detail="Replicate flux-pulid não concluiu em 3 minutos")


_LORA_TRAINER_VERSION = "26dce37af90b9d997eeb970d92e47de3064d46c300504ae376c75bef6a9022d2"
_REPLICATE_ACCOUNT = "daniel-fabbri"


def _sanitize_model_name(channel_id: str) -> str:
    import re
    name = re.sub(r"[^a-z0-9-]", "-", channel_id.lower())[:40].strip("-")
    return f"postgen-ch-{name}"


def start_lora_training(channel_id: str, images_zip_url: str, api_key: str) -> str:
    """
    Inicia o fine-tuning de LoRA com as imagens de referência do canal.
    Retorna o training_id do Replicate (não aguarda conclusão).
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="REPLICATE_API_KEY não configurado")

    model_name = _sanitize_model_name(channel_id)
    destination = f"{_REPLICATE_ACCOUNT}/{model_name}"

    # Garante que o modelo de destino existe
    create_model_resp = requests.post(
        f"{_REPLICATE_API}/models",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "owner": _REPLICATE_ACCOUNT,
            "name": model_name,
            "visibility": "private",
            "hardware": "gpu-l40s",
            "description": f"PostGen LoRA para canal {channel_id}",
        },
        timeout=30,
    )
    # 409 = já existe — tudo bem. Qualquer outro erro é bloqueante.
    if not create_model_resp.ok and create_model_resp.status_code != 409:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao criar modelo de destino no Replicate: {create_model_resp.text[:300]}",
        )
    print(f"[LORA] Modelo destino: {destination} (status={create_model_resp.status_code})")

    print(f"[LORA] Iniciando training | destination={destination} | zip={images_zip_url[:60]}")

    train_resp = requests.post(
        f"{_REPLICATE_API}/models/ostris/flux-dev-lora-trainer/versions/{_LORA_TRAINER_VERSION}/trainings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "destination": destination,
            "input": {
                "input_images": images_zip_url,
                "steps": 1000,
                "lora_rank": 16,
                "trigger_word": "TOK",
                "autocaption": True,
                "learning_rate": 0.0004,
            },
        },
        timeout=30,
    )

    if not train_resp.ok:
        raise HTTPException(
            status_code=train_resp.status_code,
            detail=f"Erro ao iniciar treinamento: {train_resp.text[:400]}",
        )

    training = train_resp.json()
    training_id = training.get("id")
    print(f"[LORA] Training iniciado: {training_id}")
    return training_id


def get_lora_training_status(training_id: str, api_key: str) -> dict:
    """
    Consulta o status de um training no Replicate.
    Retorna dict com keys: status ('starting'|'processing'|'succeeded'|'failed'|'canceled'), version (str|None), error (str|None)
    """
    resp = requests.get(
        f"{_REPLICATE_API}/trainings/{training_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    if not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=f"Erro ao consultar training: {resp.text[:200]}")

    data = resp.json()
    status = data.get("status", "unknown")
    version = None
    if status == "succeeded":
        output = data.get("output") or {}
        version = output.get("version") or data.get("output", {}).get("weights")
    return {"status": status, "version": version, "error": data.get("error")}


def _resolve_lora_version(lora_ref: str, api_key: str) -> str:
    """
    Garante que temos um version ID (hex 64 chars).
    Se receber um model path (contém '/'), busca o latest_version.id.
    """
    if "/" not in lora_ref:
        return lora_ref  # já é um version ID
    mr = requests.get(
        f"{_REPLICATE_API}/models/{lora_ref}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    if not mr.ok:
        raise HTTPException(status_code=500, detail=f"Modelo LoRA não encontrado: {lora_ref}")
    version_id = mr.json().get("latest_version", {}).get("id")
    if not version_id:
        raise HTTPException(status_code=500, detail="Modelo LoRA treinado não tem versão disponível")
    return version_id


def generate_with_lora(
    prompt: str,
    lora_ref: str,
    api_key: str,
    base_image_url: str = None,
    mask_url: str = None,
    aspect_ratio: str = "1:1",
) -> bytes:
    """
    Gera imagem usando LoRA treinado pessoalmente. O trigger word TOK já deve estar no prompt.
    lora_ref: version ID (64 hex chars) ou model path 'owner/model'.
    base_image_url: usa img2img preservando a cena original.
    mask_url: máscara para inpainting — branco = alterar (rosto do Daniel), preto = preservar.
              Quando fornecida junto com base_image_url, apenas a área mascarada é alterada.
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="REPLICATE_API_KEY não configurado")

    version_id = _resolve_lora_version(lora_ref, api_key)
    mode = "inpaint" if (base_image_url and mask_url) else ("img2img" if base_image_url else "txt2img")
    print(f"[LORA] Gerando com LoRA [{mode}] | version={version_id[:16]}... | prompt={prompt[:80]}")

    lora_input = {
        "prompt": prompt,
        "num_outputs": 1,
        "output_format": "png",
        "output_quality": 95,
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
    }

    if base_image_url and mask_url:
        # Inpainting: só altera a área mascarada (rosto do Daniel), resto preservado
        lora_input["image"] = base_image_url
        lora_input["mask"] = mask_url
        lora_input["prompt_strength"] = 0.85  # Alta influência do LoRA na área mascarada
    elif base_image_url:
        # img2img sem máscara: preserva cena em geral
        lora_input["image"] = base_image_url
        lora_input["prompt_strength"] = 0.50
    else:
        lora_input["aspect_ratio"] = aspect_ratio

    create_resp = requests.post(
        f"{_REPLICATE_API}/predictions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        },
        json={
            "version": version_id,
            "input": lora_input,
        },
        timeout=90,
    )

    if not create_resp.ok:
        raise HTTPException(
            status_code=create_resp.status_code,
            detail=f"Replicate LoRA generate error: {create_resp.text[:400]}",
        )

    prediction = create_resp.json()
    status = prediction.get("status")
    prediction_id = prediction.get("id")
    print(f"[LORA] Prediction {prediction_id} status={status}")

    if status == "succeeded":
        return _download_output(prediction)
    if status in ("failed", "canceled"):
        raise HTTPException(status_code=500, detail=f"LoRA generate falhou: {prediction.get('error')}")

    for attempt in range(_POLL_MAX):
        time.sleep(_POLL_INTERVAL)
        poll = requests.get(
            f"{_REPLICATE_API}/predictions/{prediction_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if not poll.ok:
            continue
        result = poll.json()
        status = result.get("status")
        print(f"[LORA] Poll {attempt+1} status={status}")
        if status == "succeeded":
            return _download_output(result)
        if status in ("failed", "canceled"):
            raise HTTPException(status_code=500, detail=f"LoRA generate falhou: {result.get('error')}")

    raise HTTPException(status_code=504, detail="LoRA generate não concluiu em 3 minutos")


def generate_video_from_image(
    image_url: str,
    prompt: str,
    api_key: str,
) -> bytes:
    """
    Gera vídeo a partir de frame de referência usando minimax/video-01 no Replicate.
    image_url: URL pública do frame gerado pelo LoRA.
    prompt: descrição de movimento e cena.
    Retorna bytes do vídeo MP4.
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="REPLICATE_API_KEY não configurado")

    print(f"[REPLICATE] minimax/video-01 | image={image_url[:60]} | prompt={prompt[:80]}")

    create_resp = requests.post(
        f"{_REPLICATE_API}/models/minimax/video-01/predictions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "input": {
                "prompt": prompt,
                "first_frame_image": image_url,
                "prompt_optimizer": False,
            }
        },
        timeout=30,
    )

    if not create_resp.ok:
        raise HTTPException(
            status_code=create_resp.status_code,
            detail=f"Replicate minimax error: {create_resp.text[:400]}",
        )

    prediction = create_resp.json()
    status = prediction.get("status")
    prediction_id = prediction.get("id")
    print(f"[REPLICATE] minimax prediction {prediction_id} status={status}")

    if status == "succeeded":
        return _download_video_output(prediction)
    if status in ("failed", "canceled"):
        raise HTTPException(status_code=500, detail=f"minimax/video-01 falhou: {prediction.get('error')}")

    # Poll com timeout estendido — vídeo pode levar até 10 min
    for attempt in range(200):  # 200 × 3s = 10 minutos
        time.sleep(_POLL_INTERVAL)
        poll = requests.get(
            f"{_REPLICATE_API}/predictions/{prediction_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if not poll.ok:
            continue
        result = poll.json()
        status = result.get("status")
        print(f"[REPLICATE] minimax poll {attempt+1}/200 status={status}")
        if status == "succeeded":
            return _download_video_output(result)
        if status in ("failed", "canceled"):
            raise HTTPException(status_code=500, detail=f"minimax/video-01 falhou: {result.get('error')}")

    raise HTTPException(status_code=504, detail="minimax/video-01 não concluiu em 10 minutos")


def _download_video_output(prediction: dict) -> bytes:
    output = prediction.get("output")
    if not output:
        raise HTTPException(status_code=500, detail="Replicate não retornou vídeo")
    url = output[0] if isinstance(output, list) else output
    print(f"[REPLICATE] Baixando vídeo: {str(url)[:80]}")
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    print(f"[REPLICATE] ✓ Vídeo baixado: {len(resp.content)} bytes")
    return resp.content


def _download_output(prediction: dict) -> bytes:
    output = prediction.get("output") or []
    if not output:
        raise HTTPException(status_code=500, detail="Replicate não retornou imagem")
    url = output[0] if isinstance(output, list) else output
    print(f"[REPLICATE] Baixando resultado: {str(url)[:80]}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    print(f"[REPLICATE] ✓ Imagem baixada: {len(resp.content)} bytes")
    return resp.content


def generate_lipsync(video_url: str, audio_url: str, api_key: str) -> bytes:
    """
    Aplica lipsync no vídeo usando zsxkib/latentsync no Replicate.
    video_url: URL pública do vídeo MP4 gerado pelo MiniMax.
    audio_url: URL pública do áudio MP3 com a voz clonada.
    Retorna bytes do vídeo MP4 com lipsync + áudio embutido.
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="REPLICATE_API_KEY não configurado")

    print(f"[REPLICATE] latentsync | video={video_url[:60]} | audio={audio_url[:60]}")

    create_resp = requests.post(
        f"{_REPLICATE_API}/models/zsxkib/latentsync/predictions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "input": {
                "video_path": video_url,
                "audio_path": audio_url,
                "guidance_scale": 1.5,
                "inference_steps": 20,
            }
        },
        timeout=30,
    )

    if not create_resp.ok:
        raise HTTPException(
            status_code=create_resp.status_code,
            detail=f"Replicate latentsync error: {create_resp.text[:400]}",
        )

    prediction = create_resp.json()
    status = prediction.get("status")
    prediction_id = prediction.get("id")
    print(f"[REPLICATE] latentsync prediction {prediction_id} status={status}")

    if status == "succeeded":
        return _download_video_output(prediction)
    if status in ("failed", "canceled"):
        raise HTTPException(status_code=500, detail=f"latentsync falhou: {prediction.get('error')}")

    for attempt in range(200):  # 200 × 3s = 10 minutos
        time.sleep(_POLL_INTERVAL)
        poll = requests.get(
            f"{_REPLICATE_API}/predictions/{prediction_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if not poll.ok:
            continue
        result = poll.json()
        status = result.get("status")
        print(f"[REPLICATE] latentsync poll {attempt+1}/200 status={status}")
        if status == "succeeded":
            return _download_video_output(result)
        if status in ("failed", "canceled"):
            raise HTTPException(status_code=500, detail=f"latentsync falhou: {result.get('error')}")

    raise HTTPException(status_code=504, detail="latentsync não concluiu em 10 minutos")

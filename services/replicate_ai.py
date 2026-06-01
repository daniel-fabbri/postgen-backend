import time
import requests
from fastapi import HTTPException

_REPLICATE_API = "https://api.replicate.com/v1"
_CONSISTENT_CHARACTER_MODEL = "fofr/consistent-character"

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
        f"{_REPLICATE_API}/models/{_CONSISTENT_CHARACTER_MODEL}/predictions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Prefer": "wait",  # espera até 60s na própria resposta antes de polling
        },
        json={
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

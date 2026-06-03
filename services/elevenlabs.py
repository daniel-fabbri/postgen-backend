import os
import subprocess
import tempfile
import requests
from fastapi import HTTPException

_BASE = "https://api.elevenlabs.io/v1"


def create_voice_clone(name: str, audio_data: bytes, filename: str, api_key: str) -> str:
    """Cria clone de voz no ElevenLabs a partir de amostra de áudio. Retorna voice_id."""
    resp = requests.post(
        f"{_BASE}/voices/add",
        headers={"xi-api-key": api_key},
        data={"name": name, "description": "PostGen voice clone"},
        files={"files": (filename, audio_data, "audio/mpeg")},
        timeout=120,
    )
    if not resp.ok:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"ElevenLabs voice clone error: {resp.text[:400]}",
        )
    return resp.json()["voice_id"]


def generate_tts(text: str, voice_id: str, api_key: str) -> bytes:
    """Gera áudio TTS com a voz clonada. Retorna bytes MP3."""
    resp = requests.post(
        f"{_BASE}/text-to-speech/{voice_id}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=60,
    )
    if not resp.ok:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"ElevenLabs TTS error: {resp.text[:400]}",
        )
    return resp.content


def mix_audio_into_video(video_bytes: bytes, audio_bytes: bytes) -> bytes:
    """Combina áudio com vídeo usando ffmpeg. A faixa de áudio é sobreposta no vídeo."""
    vf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    af = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    out_path = vf.name + "_mixed.mp4"

    try:
        vf.write(video_bytes)
        vf.close()
        af.write(audio_bytes)
        af.close()

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", vf.name,
                "-i", af.name,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                out_path,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )

        with open(out_path, "rb") as f:
            return f.read()

    finally:
        for path in [vf.name, af.name, out_path]:
            try:
                os.unlink(path)
            except OSError:
                pass

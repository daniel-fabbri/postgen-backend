import json
import os
import traceback
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import PlainTextResponse
from openai import AzureOpenAI
from sqlalchemy.orm import Session

from config import INSTAGRAM_WEBHOOK_VERIFY_TOKEN
from database import get_db
from models import ChannelDB, PostDB

router = APIRouter(tags=["webhooks"])


@router.get("/api/webhooks/instagram")
async def instagram_webhook_verify(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    print(f"[WEBHOOK] Verification request: mode={hub_mode}, token={hub_verify_token}")
    if hub_mode == "subscribe" and hub_verify_token == INSTAGRAM_WEBHOOK_VERIFY_TOKEN:
        print(f"[WEBHOOK] Verification successful, returning challenge: {hub_challenge}")
        try:
            return PlainTextResponse(content=hub_challenge)
        except Exception as e:
            print(f"[WEBHOOK] Error returning challenge: {e}")
            return PlainTextResponse(content="200")
    print("[WEBHOOK] Verification failed")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


@router.post("/api/webhooks/instagram")
async def instagram_webhook_receive(
    body: dict,
    db: Session = Depends(get_db),
):
    print(f"[WEBHOOK] Received event: {json.dumps(body, indent=2)}")
    try:
        if "entry" not in body:
            return {"status": "ignored", "reason": "no entry"}
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                field = change.get("field")
                value = change.get("value")
                print(f"[WEBHOOK] Processing field={field}, value={json.dumps(value)}")
                if field == "comments":
                    await _process_comment_webhook(value, db)
                elif field == "messages":
                    await _process_message_webhook(value, db)
        return {"status": "processed"}
    except Exception as e:
        print(f"[WEBHOOK ERROR] {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


async def _process_comment_webhook(value: dict, db: Session):
    try:
        comment_id = value.get("id")
        text = value.get("text", "")
        from_user = value.get("from", {})
        media_id = value.get("media", {}).get("id")
        parent_id = value.get("parent_id")

        if not comment_id or not text:
            print("[WEBHOOK] Comment missing required fields")
            return
        if parent_id:
            print(f"[WEBHOOK] Ignoring reply comment (parent_id={parent_id}) to prevent loop")
            return

        print(f"[WEBHOOK] New comment: id={comment_id}, text={text[:50]}..., media={media_id}")

        channel = db.query(ChannelDB).filter(
            ChannelDB.instagram_access_token.isnot(None),
            ChannelDB.auto_reply_enabled == True,
        ).first()
        if not channel:
            print("[WEBHOOK] No channel with auto_reply enabled")
            return

        from_user_id = str(from_user.get("id", ""))
        if from_user_id and from_user_id == str(channel.instagram_user_id or ""):
            print(f"[WEBHOOK] Ignoring comment from own account ({from_user_id}), prevents loop")
            return

        post_context = ""
        post_db = db.query(PostDB).filter(PostDB.ig_media_id == media_id).first()
        if post_db:
            post_context = f"Post: {post_db.text[:200]}"

        reply_text = await _generate_ai_reply(channel=channel, user_message=text, context=post_context, message_type="comment")
        if not reply_text:
            print("[WEBHOOK] Failed to generate reply")
            return

        success = await _send_instagram_comment_reply(comment_id=comment_id, reply_text=reply_text, access_token=channel.instagram_access_token)
        if success:
            print(f"[WEBHOOK] Successfully replied to comment {comment_id}")
        else:
            print(f"[WEBHOOK] Failed to send reply to comment {comment_id}")

    except Exception as e:
        print(f"[WEBHOOK ERROR] process_comment: {e}")
        traceback.print_exc()


async def _process_message_webhook(value: dict, db: Session):
    try:
        message_id = value.get("id")
        text = value.get("text", "")
        from_user = value.get("from", {})

        if not message_id or not text:
            print("[WEBHOOK] Message missing required fields")
            return

        print(f"[WEBHOOK] New message: id={message_id}, text={text[:50]}...")

        channel = db.query(ChannelDB).filter(
            ChannelDB.instagram_access_token.isnot(None),
            ChannelDB.auto_reply_enabled == True,
        ).first()
        if not channel:
            print("[WEBHOOK] No channel with auto_reply enabled")
            return

        from_user_id = from_user.get("id")
        if from_user_id and str(from_user_id) == str(channel.instagram_user_id):
            print("[WEBHOOK] Ignoring message from own account (prevents loop)")
            return

        reply_text = await _generate_ai_reply(
            channel=channel, user_message=text,
            context=f"Canal: {channel.name}. Objetivo: {channel.objective}", message_type="dm",
        )
        if not reply_text:
            print("[WEBHOOK] Failed to generate reply")
            return

        success = await _send_instagram_message_reply(
            recipient_id=from_user.get("id"), reply_text=reply_text,
            instagram_user_id=channel.instagram_user_id, access_token=channel.instagram_access_token,
        )
        if success:
            print(f"[WEBHOOK] Successfully replied to message {message_id}")
        else:
            print(f"[WEBHOOK] Failed to send reply to message {message_id}")

    except Exception as e:
        print(f"[WEBHOOK ERROR] process_message: {e}")
        traceback.print_exc()


async def _generate_ai_reply(channel: ChannelDB, user_message: str, context: str, message_type: str) -> Optional[str]:
    try:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        if not endpoint or not api_key:
            print("[WEBHOOK] Azure OpenAI not configured")
            return None

        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2024-08-01-preview")

        if channel.auto_reply_prompt and channel.auto_reply_prompt.strip():
            system_prompt = channel.auto_reply_prompt
        else:
            system_prompt = f"""Você é um assistente que responde {message_type}s no Instagram para o canal "{channel.name}".

Objetivo do canal: {channel.objective}

Instruções:
- Seja amigável, natural e conversacional
- Responda de forma breve e direta (máximo 2-3 frases)
- Use emojis moderadamente quando apropriado
- Se for um elogio, agradeça com entusiasmo
- Se for uma pergunta, responda de forma útil
- Se for crítica construtiva, agradeça pelo feedback
- Mantenha o tom alinhado com o objetivo do canal
- NÃO use hashtags na resposta
"""
        user_prompt = f"""{context}

Mensagem do usuário: "{user_message}"

Gere uma resposta apropriada:"""

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=150, temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        print(f"[WEBHOOK] Generated reply: {reply}")
        return reply

    except Exception as e:
        print(f"[WEBHOOK ERROR] generate_ai_reply: {e}")
        traceback.print_exc()
        return None


async def _send_instagram_comment_reply(comment_id: str, reply_text: str, access_token: str) -> bool:
    try:
        response = requests.post(
            f"https://graph.instagram.com/v21.0/{comment_id}/replies",
            json={"message": reply_text, "access_token": access_token},
        )
        if response.ok:
            print(f"[WEBHOOK] Comment reply sent successfully: {response.json()}")
            return True
        print(f"[WEBHOOK ERROR] Failed to send comment reply: {response.status_code} {response.text}")
        return False
    except Exception as e:
        print(f"[WEBHOOK ERROR] send_instagram_comment_reply: {e}")
        return False


async def _send_instagram_message_reply(recipient_id: str, reply_text: str, instagram_user_id: str, access_token: str) -> bool:
    try:
        response = requests.post(
            f"https://graph.instagram.com/v21.0/{instagram_user_id}/messages",
            json={"recipient": {"id": recipient_id}, "message": {"text": reply_text}, "access_token": access_token},
        )
        if response.ok:
            print(f"[WEBHOOK] Message reply sent successfully: {response.json()}")
            return True
        print(f"[WEBHOOK ERROR] Failed to send message reply: {response.status_code} {response.text}")
        return False
    except Exception as e:
        print(f"[WEBHOOK ERROR] send_instagram_message_reply: {e}")
        return False

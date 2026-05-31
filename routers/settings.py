from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from dependencies import get_current_user, get_or_create_settings, get_azure_client
from models import UserDB
from schemas import Settings

router = APIRouter(prefix="/api", tags=["settings"])


@router.get("/settings", response_model=Settings)
def get_settings(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = get_or_create_settings(current_user, db)
    result = Settings.model_validate(s)
    result.azure_openai_api_key = "***" if s.azure_openai_api_key else ""
    return result


@router.put("/settings")
def update_settings(
    data: Settings,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = get_or_create_settings(current_user, db)
    s.azure_openai_endpoint = data.azure_openai_endpoint
    s.azure_openai_deployment_name = data.azure_openai_deployment_name
    s.azure_openai_image_deployment = data.azure_openai_image_deployment
    s.azure_openai_image_endpoint = data.azure_openai_image_endpoint
    s.azure_openai_api_version = data.azure_openai_api_version
    s.public_base_url = data.public_base_url
    if data.azure_openai_api_key and data.azure_openai_api_key != "***":
        s.azure_openai_api_key = data.azure_openai_api_key
    db.commit()
    return {"message": "Configurações salvas"}


@router.get("/test-azure")
def test_azure(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = get_or_create_settings(current_user, db)
    if not s.azure_openai_endpoint or not s.azure_openai_api_key:
        return {"success": False, "error": "Azure OpenAI não configurado"}
    try:
        client = get_azure_client(s)
        resp = client.chat.completions.create(
            model=s.azure_openai_deployment_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
        )
        return {"success": True, "test_response": resp.choices[0].message.content}
    except Exception as e:
        return {"success": False, "error": str(e)}

import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8004").rstrip("/")
_code_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getenv("STORAGE_BASE", _code_dir)
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

DATABASE_URL = os.getenv("DATABASE_URL", "")
JWT_SECRET = os.getenv("JWT_SECRET", "changeme-insecure-default-secret-32chars!!")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "images")

AZURE_SORA_ENDPOINT = os.getenv("AZURE_SORA_ENDPOINT", "").rstrip("/")
AZURE_SORA_API_KEY = os.getenv("AZURE_SORA_API_KEY", "")

GPT_IMAGE_2_ENDPOINT = os.getenv("GPT_IMAGE_2_ENDPOINT", "https://postgen-ai.openai.azure.com").rstrip("/")
GPT_IMAGE_2_API_KEY = os.getenv("GPT_IMAGE_2_API_KEY", "")

# Azure OpenAI — GPT-4o / visão (stocks-foundry-ai)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# Azure AI Foundry — FLUX.1-Kontext-pro e FLUX.2-pro
AZURE_FOUNDRY_ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT", "").rstrip("/")
AZURE_FOUNDRY_API_KEY = os.getenv("AZURE_FOUNDRY_API_KEY", "")

# Replicate — fofr/consistent-character (geração com rosto consistente)
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY", "")

INSTAGRAM_APP_ID = os.getenv("INSTAGRAM_APP_ID", "")
INSTAGRAM_APP_SECRET = os.getenv("INSTAGRAM_APP_SECRET", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

MERCADOPAGO_ACCESS_TOKEN = os.getenv("MERCADOPAGO_ACCESS_TOKEN", "")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

ADMIN_EMAIL = "daniel.fabbri@avanade.com"
INSTAGRAM_WEBHOOK_VERIFY_TOKEN = os.getenv("INSTAGRAM_WEBHOOK_VERIFY_TOKEN", "postgen_webhook_secret_2026")

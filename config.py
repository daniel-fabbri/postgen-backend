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

INSTAGRAM_APP_ID = os.getenv("INSTAGRAM_APP_ID", "")
INSTAGRAM_APP_SECRET = os.getenv("INSTAGRAM_APP_SECRET", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

MERCADOPAGO_ACCESS_TOKEN = os.getenv("MERCADOPAGO_ACCESS_TOKEN", "")

ADMIN_EMAIL = "daniel.fabbri@avanade.com"
INSTAGRAM_WEBHOOK_VERIFY_TOKEN = os.getenv("INSTAGRAM_WEBHOOK_VERIFY_TOKEN", "postgen_webhook_secret_2026")

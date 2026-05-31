from fastapi import HTTPException
from azure.storage.blob import BlobServiceClient, ContentSettings

from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER


def _blob_client() -> BlobServiceClient:
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise HTTPException(status_code=500, detail="Azure Storage não configurado")
    return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)


def upload_bytes_to_blob(data: bytes, blob_name: str, content_type: str = "image/png") -> str:
    client = _blob_client()
    container = client.get_container_client(AZURE_STORAGE_CONTAINER)
    blob = container.get_blob_client(blob_name)
    blob.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type=content_type))
    return blob.url


def upload_file_to_blob(file_path: str, blob_name: str, content_type: str = "image/png") -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    return upload_bytes_to_blob(data, blob_name, content_type)

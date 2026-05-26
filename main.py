from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import json
import os
from datetime import datetime
from openai import AzureOpenAI
import requests
import base64
import shutil
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

app = FastAPI(title="PostGen API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
CHANNELS_FILE = os.path.join(BASE_DIR, "channels.json")
POSTS_DIR = os.path.join(BASE_DIR, "posts")
IMAGES_DIR = os.path.join(POSTS_DIR, "images")
AVATARS_DIR = os.path.join(BASE_DIR, "avatars")
AVATARS_METADATA_FILE = os.path.join(AVATARS_DIR, "_metadata.json")

os.makedirs(POSTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(AVATARS_DIR, exist_ok=True)

app.mount("/posts/images", StaticFiles(directory=IMAGES_DIR), name="post_images")
app.mount("/avatars", StaticFiles(directory=AVATARS_DIR), name="avatars")

# Models
class Settings(BaseModel):
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment_name: str = "gpt-4"
    azure_openai_image_deployment: str = "dall-e-3"
    azure_openai_image_endpoint: str = ""
    azure_openai_api_version: str = "2024-02-01"
    public_base_url: str = "http://localhost:8004"

class Channel(BaseModel):
    id: Optional[str] = None
    name: str
    objective: str
    text_generation_prompt: Optional[str] = None
    image_generation_prompt: Optional[str] = None
    avatar_url: Optional[str] = None
    suggested_image_url: Optional[str] = None
    created_at: Optional[str] = None
    instagram_user_id: Optional[str] = None
    instagram_access_token: Optional[str] = None

class GeneratePostRequest(BaseModel):
    channel_id: str
    additional_prompt: Optional[str] = None

class Post(BaseModel):
    text: str
    image_url: str

class SavedPost(BaseModel):
    id: str
    channel_id: str
    channel_name: str
    text: str
    image_path: str
    created_at: str
    published: bool = False


class GenerateAvatarRequest(BaseModel):
    prompt: str
    channel_id: Optional[str] = None

class UpdateAvatarRequest(BaseModel):
    avatar_url: str

class AvatarInfo(BaseModel):
    filename: str
    url: str
    created_at: Optional[str] = None

class UpdatePostRequest(BaseModel):
    text: Optional[str] = None
    image_path: Optional[str] = None
    published: Optional[bool] = None

class GeneratePostImageRequest(BaseModel):
    prompt: str
    channel_id: str

# Helper functions
def load_settings() -> Settings:
    data: dict = {}

    # Base: load from settings.json if it exists
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            file_data = json.load(f)
            known = Settings.model_fields.keys()
            data = {k: v for k, v in file_data.items() if k in known}

    # Env vars override the file (non-empty values only)
    _env_map = {
        "azure_openai_endpoint": "AZURE_OPENAI_ENDPOINT",
        "azure_openai_api_key": "AZURE_OPENAI_API_KEY",
        "azure_openai_deployment_name": "AZURE_OPENAI_DEPLOYMENT_NAME",
        "azure_openai_image_deployment": "AZURE_OPENAI_IMAGE_DEPLOYMENT",
        "azure_openai_image_endpoint": "AZURE_OPENAI_IMAGE_ENDPOINT",
        "azure_openai_api_version": "AZURE_OPENAI_API_VERSION",
        "public_base_url": "PUBLIC_BASE_URL",
    }
    for field, env_var in _env_map.items():
        val = os.getenv(env_var)
        if val:
            data[field] = val

    return Settings(**data) if data else Settings()

def save_settings(settings: Settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings.model_dump(), f, indent=2)

def load_channels() -> List[Channel]:
    if os.path.exists(CHANNELS_FILE):
        with open(CHANNELS_FILE, 'r') as f:
            data = json.load(f)
            return [Channel(**ch) for ch in data]
    return []

def save_channels(channels: List[Channel]):
    with open(CHANNELS_FILE, 'w') as f:
        json.dump([ch.model_dump() for ch in channels], f, indent=2)

def load_posts() -> List[SavedPost]:
    """Load all saved posts from JSON files"""
    posts = []
    if os.path.exists(POSTS_DIR):
        for filename in os.listdir(POSTS_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(POSTS_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    posts.append(SavedPost(**data))
    # Sort by created_at descending (most recent first)
    posts.sort(key=lambda x: x.created_at, reverse=True)
    return posts

def save_post(post: SavedPost):
    """Save a post to JSON file"""
    filepath = os.path.join(POSTS_DIR, f"{post.id}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(post.model_dump(), f, indent=2, ensure_ascii=False)

def load_avatars_metadata() -> dict:
    if os.path.exists(AVATARS_METADATA_FILE):
        with open(AVATARS_METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_avatars_metadata(metadata: dict):
    with open(AVATARS_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def register_avatar_channel(filename: str, channel_id: str):
    metadata = load_avatars_metadata()
    metadata[filename] = channel_id
    save_avatars_metadata(metadata)

def save_image_from_base64(base64_data: str, post_id: str) -> str:
    """Save base64 image to file and return the relative path"""
    # Remove data URL prefix if present
    if base64_data.startswith('data:image'):
        base64_data = base64_data.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(base64_data)
    
    # Save to file
    image_filename = f"{post_id}.png"
    image_path = os.path.join(IMAGES_DIR, image_filename)
    
    with open(image_path, 'wb') as f:
        f.write(image_bytes)
    
    # Return relative path
    return f"images/{image_filename}"

def get_azure_openai_client():
    settings = load_settings()
    if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
        raise HTTPException(status_code=400, detail="Azure OpenAI not configured")
    
    print(f"[DEBUG] Creating client with:")
    print(f"  Endpoint: {settings.azure_openai_endpoint}")
    print(f"  API Version: {settings.azure_openai_api_version}")
    print(f"  Text Deployment: {settings.azure_openai_deployment_name}")
    
    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version
    )

# Routes
@app.get("/")
async def root():
    return {"message": "PostGen API is running"}

@app.get("/api/settings", response_model=Settings)
async def get_settings():
    settings = load_settings()
    settings.azure_openai_api_key = "***" if settings.azure_openai_api_key else ""
    return settings

@app.put("/api/settings")
async def update_settings(settings: Settings):
    # Load existing settings to preserve masked values
    existing = load_settings()
    
    # Only update if not masked
    if settings.azure_openai_api_key and settings.azure_openai_api_key != "***":
        existing.azure_openai_api_key = settings.azure_openai_api_key

    # Update other fields
    existing.azure_openai_endpoint = settings.azure_openai_endpoint
    existing.azure_openai_deployment_name = settings.azure_openai_deployment_name
    existing.azure_openai_image_deployment = settings.azure_openai_image_deployment
    existing.azure_openai_image_endpoint = settings.azure_openai_image_endpoint
    existing.azure_openai_api_version = settings.azure_openai_api_version
    existing.public_base_url = settings.public_base_url
    
    save_settings(existing)
    return {"message": "Settings updated successfully"}

@app.get("/api/test-azure")
async def test_azure_connection():
    """Test Azure OpenAI connection and return deployment info"""
    try:
        settings = load_settings()
        if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
            return {"success": False, "error": "Azure OpenAI not configured"}
        
        client = get_azure_openai_client()
        
        # Try a simple completion
        try:
            response = client.chat.completions.create(
                model=settings.azure_openai_deployment_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return {
                "success": True,
                "endpoint": settings.azure_openai_endpoint,
                "api_version": settings.azure_openai_api_version,
                "deployment": settings.azure_openai_deployment_name,
                "test_response": response.choices[0].message.content
            }
        except Exception as e:
            return {
                "success": False,
                "endpoint": settings.azure_openai_endpoint,
                "api_version": settings.azure_openai_api_version,
                "deployment": settings.azure_openai_deployment_name,
                "error": str(e),
                "error_type": type(e).__name__
            }
    except Exception as e:
        return {"success": False, "error": str(e)}

def mask_channel(ch: Channel) -> Channel:
    masked = ch.model_copy()
    if masked.instagram_access_token:
        masked.instagram_access_token = "***"
    return masked

@app.get("/api/channels", response_model=List[Channel])
async def get_channels():
    return [mask_channel(ch) for ch in load_channels()]

@app.post("/api/channels", response_model=Channel)
async def create_channel(channel: Channel):
    channels = load_channels()
    
    # Generate ID and timestamp
    channel.id = f"ch_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    channel.created_at = datetime.now().isoformat()
    
    # Generate avatar image using custom image prompt if provided
    if channel.image_generation_prompt:
        try:
            settings = load_settings()
            
            # Use custom image endpoint for avatar generation
            if settings.azure_openai_image_endpoint:
                print(f"Generating avatar for channel: {channel.name}")
                
                # Simplify the prompt for avatar generation
                avatar_prompt = channel.image_generation_prompt.split('\n')[0:5]  # Take first 5 lines
                avatar_prompt = '\n'.join(avatar_prompt) + "\n\nFrame: Close-up portrait style, profile picture format."
                
                payload = {
                    "prompt": avatar_prompt,
                    "width": 768,
                    "height": 768,
                    "model": settings.azure_openai_image_deployment
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "api-key": settings.azure_openai_api_key
                }
                
                response = requests.post(
                    settings.azure_openai_image_endpoint,
                    headers=headers,
                    json=payload
                )
                
                print(f"Avatar generation response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    if 'data' in result and len(result['data']) > 0:
                        image_data = result['data'][0]
                        
                        if 'b64_json' in image_data:
                            # Save avatar to file
                            avatar_filename = f"{channel.id}.png"
                            avatar_path = os.path.join(AVATARS_DIR, avatar_filename)
                            
                            image_bytes = base64.b64decode(image_data['b64_json'])
                            with open(avatar_path, 'wb') as f:
                                f.write(image_bytes)
                            
                            channel.avatar_url = f"{BASE_URL}/avatars/{avatar_filename}"
                            register_avatar_channel(avatar_filename, channel.id)
                            print(f"Avatar saved: {channel.avatar_url}")
                else:
                    print(f"Avatar generation failed: {response.text}")
        except Exception as e:
            print(f"Error generating avatar: {e}")
    
    channels.append(channel)
    save_channels(channels)
    return channel

@app.get("/api/channels/{channel_id}", response_model=Channel)
async def get_channel(channel_id: str):
    channels = load_channels()
    channel = next((ch for ch in channels if ch.id == channel_id), None)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    return mask_channel(channel)

@app.put("/api/channels/{channel_id}", response_model=Channel)
async def update_channel(channel_id: str, updated_data: Channel):
    channels = load_channels()
    channel_index = next((i for i, ch in enumerate(channels) if ch.id == channel_id), None)

    if channel_index is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    existing_channel = channels[channel_index]
    updated_data.id = existing_channel.id
    updated_data.created_at = existing_channel.created_at
    # Preserve masked token
    if updated_data.instagram_access_token == "***":
        updated_data.instagram_access_token = existing_channel.instagram_access_token

    channels[channel_index] = updated_data
    save_channels(channels)
    return mask_channel(updated_data)

@app.delete("/api/channels/{channel_id}")
async def delete_channel(channel_id: str):
    channels = load_channels()
    channels = [ch for ch in channels if ch.id != channel_id]
    save_channels(channels)
    return {"message": "Channel deleted successfully"}

# Avatar endpoints
@app.get("/api/avatars", response_model=List[AvatarInfo])
async def list_avatars(channel_id: Optional[str] = None):
    """List avatars, optionally filtered by channel"""
    metadata = load_avatars_metadata()

    # Current avatar filename for the requested channel (fallback for pre-metadata avatars)
    current_avatar_filename = None
    if channel_id:
        channels = load_channels()
        channel = next((ch for ch in channels if ch.id == channel_id), None)
        if channel and channel.avatar_url:
            current_avatar_filename = channel.avatar_url.rstrip('/').split('/')[-1]

    avatars = []
    if os.path.exists(AVATARS_DIR):
        for filename in os.listdir(AVATARS_DIR):
            if filename.startswith('_') or not filename.endswith(('.png', '.jpg', '.jpeg')):
                continue
            if channel_id is not None:
                in_metadata = metadata.get(filename) == channel_id
                is_current = filename == current_avatar_filename
                is_creation_avatar = filename == f"{channel_id}.png"
                if not in_metadata and not is_current and not is_creation_avatar:
                    continue
                # Register missing associations on-the-fly so future calls use metadata
                if not in_metadata and (is_current or is_creation_avatar):
                    register_avatar_channel(filename, channel_id)
            file_path = os.path.join(AVATARS_DIR, filename)
            created_at = datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
            avatars.append(AvatarInfo(
                filename=filename,
                url=f"{BASE_URL}/avatars/{filename}",
                created_at=created_at
            ))
    return sorted(avatars, key=lambda x: x.created_at, reverse=True)

@app.post("/api/avatars/generate")
async def generate_avatar(request: GenerateAvatarRequest):
    """Generate a new avatar using AI"""
    try:
        settings = load_settings()
        
        if not settings.azure_openai_image_endpoint:
            raise HTTPException(status_code=400, detail="Azure OpenAI image endpoint not configured")
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        avatar_filename = f"avatar_{timestamp}.png"
        avatar_path = os.path.join(AVATARS_DIR, avatar_filename)
        
        # Prepare prompt for avatar generation
        avatar_prompt = request.prompt + "\n\nFrame: Close-up portrait style, profile picture format."
        
        payload = {
            "prompt": avatar_prompt,
            "width": 768,
            "height": 768,
            "model": settings.azure_openai_image_deployment
        }
        
        headers = {
            "Content-Type": "application/json",
            "api-key": settings.azure_openai_api_key
        }
        
        response = requests.post(
            settings.azure_openai_image_endpoint,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to generate avatar: {response.text}")
        
        result = response.json()
        if 'data' not in result or len(result['data']) == 0:
            raise HTTPException(status_code=500, detail="No image data in response")
        
        image_data = result['data'][0]
        if 'b64_json' not in image_data:
            raise HTTPException(status_code=500, detail="No base64 image data in response")
        
        # Save avatar
        image_bytes = base64.b64decode(image_data['b64_json'])
        with open(avatar_path, 'wb') as f:
            f.write(image_bytes)
        
        avatar_url = f"{BASE_URL}/avatars/{avatar_filename}"

        # If channel_id is provided, update the channel's avatar and save metadata
        if request.channel_id:
            register_avatar_channel(avatar_filename, request.channel_id)
            channels = load_channels()
            channel_index = next((i for i, ch in enumerate(channels) if ch.id == request.channel_id), None)
            if channel_index is not None:
                channels[channel_index].avatar_url = avatar_url
                save_channels(channels)

        return {
            "success": True,
            "avatar_url": avatar_url,
            "filename": avatar_filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/avatars/upload")
async def upload_avatar(file: UploadFile = File(...), channel_id: Optional[str] = None):
    """Upload an avatar image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'png'
        avatar_filename = f"avatar_{timestamp}.{file_extension}"
        avatar_path = os.path.join(AVATARS_DIR, avatar_filename)
        
        # Save uploaded file
        with open(avatar_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        avatar_url = f"{BASE_URL}/avatars/{avatar_filename}"

        # If channel_id is provided, update the channel's avatar and save metadata
        if channel_id:
            register_avatar_channel(avatar_filename, channel_id)
            channels = load_channels()
            channel_index = next((i for i, ch in enumerate(channels) if ch.id == channel_id), None)
            if channel_index is not None:
                channels[channel_index].avatar_url = avatar_url
                save_channels(channels)

        return {
            "success": True,
            "avatar_url": avatar_url,
            "filename": avatar_filename
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/channels/{channel_id}/avatar")
async def update_channel_avatar(channel_id: str, request: UpdateAvatarRequest):
    """Update channel avatar URL"""
    channels = load_channels()
    channel_index = next((i for i, ch in enumerate(channels) if ch.id == channel_id), None)

    if channel_index is None:
        raise HTTPException(status_code=404, detail="Channel not found")

    channels[channel_index].avatar_url = request.avatar_url
    save_channels(channels)

    # Register the avatar-channel association so it appears in filtered gallery
    filename = request.avatar_url.rstrip('/').split('/')[-1]
    if filename:
        register_avatar_channel(filename, channel_id)
    
    return {
        "success": True,
        "message": "Avatar updated successfully",
        "channel": channels[channel_index]
    }

@app.post("/api/posts/generate", response_model=Post)
async def generate_post(request: GeneratePostRequest):
    channels = load_channels()
    channel = next((ch for ch in channels if ch.id == request.channel_id), None)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    try:
        client = get_azure_openai_client()
        settings = load_settings()
        
        print(f"Generating post for channel: {channel.name}")
        print(f"Using endpoint: {settings.azure_openai_endpoint}")
        print(f"Text deployment: {settings.azure_openai_deployment_name}")
        print(f"Image deployment: {settings.azure_openai_image_deployment}")
        
        # Generate post text
        if channel.text_generation_prompt:
            # Use custom prompt from channel
            text_prompt = channel.text_generation_prompt
        else:
            # Use default prompt
            text_prompt = f"""Create an engaging social media post for Instagram.
        
Channel: {channel.name}
Objective: {channel.objective}

The post should be:
- Engaging and authentic
- Include relevant hashtags
- Optimized for Instagram
- Between 100-200 words

Generate only the post text, nothing else."""
        
        # Add additional context if provided
        if request.additional_prompt:
            text_prompt += f"\n\nAdditional context/instructions: {request.additional_prompt}"

        print("Generating text...")
        text_response = client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": "You are a professional social media content creator."},
                {"role": "user", "content": text_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        post_text = text_response.choices[0].message.content.strip()
        print(f"Text generated successfully: {len(post_text)} characters")
        
        # Extract main subject from text (e.g., doce name, car model, etc.) for image consistency
        print("Extracting main subject from text...")
        subject_response = client.chat.completions.create(
            model=settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies the main subject of a social media post."},
                {"role": "user", "content": f"""Read this social media post and identify the MAIN subject (e.g., dish name, car model, product, person, etc.) in 2-5 words maximum. Be specific and concise.

Post text:
{post_text}

Return ONLY the subject name, nothing else. Examples:
- "Cocada"
- "Fusca 1975"
- "Pudim de leite"
- "Chevrolet Opala"
"""}
            ],
            max_tokens=20,
            temperature=0.3
        )
        
        main_subject = subject_response.choices[0].message.content.strip()
        print(f"Main subject identified: {main_subject}")
        
        # Generate post image using custom API endpoint
        if channel.image_generation_prompt:
            # Use custom prompt from channel
            image_prompt = channel.image_generation_prompt
            # Add the specific subject to ensure image consistency
            image_prompt += f"\n\nIMPORTANTE: O doce/item específico deve ser: {main_subject}"
        else:
            # Use default prompt
            image_prompt = f"""Create a visually appealing Instagram post image for: {channel.name}.
Theme: {channel.objective}
Style: Modern, professional, eye-catching, suitable for social media.
Main subject: {main_subject}"""
        
        # Add additional context if provided
        if request.additional_prompt:
            image_prompt += f"\n\nAdditional context/instructions: {request.additional_prompt}"

        print("Generating image...")
        try:
            # Use custom Azure AI Services REST API for MAI/FLUX models
            if settings.azure_openai_image_endpoint:
                headers = {
                    "Content-Type": "application/json",
                    "api-key": settings.azure_openai_api_key
                }
                payload = {
                    "prompt": image_prompt,
                    "width": 1024,
                    "height": 1024,
                    "model": settings.azure_openai_image_deployment
                }
                
                print(f"Using custom image endpoint: {settings.azure_openai_image_endpoint}")
                print(f"Model: {settings.azure_openai_image_deployment}")
                print(f"Payload: {payload}")
                
                response = requests.post(
                    settings.azure_openai_image_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                # Log response before raising error
                print(f"Response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"Error response body: {response.text}")
                
                response.raise_for_status()
                
                result = response.json()
                
                # Handle base64 image response
                if 'data' in result and len(result['data']) > 0:
                    if 'b64_json' in result['data'][0]:
                        # Convert base64 to data URL
                        b64_image = result['data'][0]['b64_json']
                        image_url = f"data:image/png;base64,{b64_image}"
                        print(f"Image generated successfully (base64)")
                    elif 'url' in result['data'][0]:
                        image_url = result['data'][0]['url']
                        print(f"Image generated successfully (URL)")
                    else:
                        raise Exception("No image data in response")
                else:
                    raise Exception("Invalid response format")
            else:
                # Fallback to standard OpenAI SDK (for DALL-E)
                image_response = client.images.generate(
                    model=settings.azure_openai_image_deployment,
                    prompt=image_prompt,
                    n=1,
                    size="1024x1024"
                )
                image_url = image_response.data[0].url
                print(f"Image generated successfully")
        except Exception as img_error:
            print(f"Error generating image: {str(img_error)}")
            # If image generation fails, use a placeholder
            image_url = "https://via.placeholder.com/1024x1024/4F46E5/FFFFFF?text=PostGen"
            print("Using placeholder image")
        
        # Save the generated post
        post_id = f"post_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Save image to disk if it's base64
        image_path = ""
        if image_url.startswith('data:image'):
            try:
                image_path = save_image_from_base64(image_url, post_id)
                print(f"Image saved to: {image_path}")
            except Exception as save_error:
                print(f"Error saving image: {str(save_error)}")
                image_path = "placeholder"
        else:
            # For URL images, we could download them, but for now just store the URL
            image_path = image_url
        
        # Create and save post metadata
        saved_post = SavedPost(
            id=post_id,
            channel_id=request.channel_id,
            channel_name=channel.name,
            text=post_text,
            image_path=image_path,
            created_at=datetime.now().isoformat(),
            published=False
        )
        save_post(saved_post)
        print(f"Post saved: {post_id}")
        
        return Post(text=post_text, image_url=image_url)
        
    except Exception as e:
        print(f"Error in generate_post: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating post: {str(e)}")


@app.get("/api/posts", response_model=List[SavedPost])
async def get_posts():
    """Get all saved posts"""
    return load_posts()

@app.get("/api/posts/{post_id}", response_model=SavedPost)
async def get_post(post_id: str):
    """Get a specific post by ID"""
    posts = load_posts()
    post = next((p for p in posts if p.id == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post

@app.patch("/api/posts/{post_id}", response_model=SavedPost)
async def update_post(post_id: str, request: UpdatePostRequest):
    """Update post text, image_path, and/or published status"""
    posts = load_posts()
    post = next((p for p in posts if p.id == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if request.text is not None:
        post.text = request.text
    if request.image_path is not None:
        post.image_path = request.image_path
    if request.published is not None:
        post.published = request.published
    save_post(post)
    return post

@app.post("/api/posts/{post_id}/image/upload")
async def upload_post_image(post_id: str, file: UploadFile = File(...)):
    """Replace a post's image with an uploaded file"""
    posts = load_posts()
    post = next((p for p in posts if p.id == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_filename = f"{post_id}.png"
    image_path_full = os.path.join(IMAGES_DIR, image_filename)
    with open(image_path_full, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    post.image_path = f"images/{image_filename}"
    save_post(post)
    return {
        "success": True,
        "image_url": f"{BASE_URL}/posts/images/{image_filename}",
        "image_path": post.image_path
    }

@app.post("/api/posts/{post_id}/image/generate")
async def generate_post_image(post_id: str, request: GeneratePostImageRequest):
    """Generate a new image for an existing post"""
    posts = load_posts()
    post = next((p for p in posts if p.id == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    settings = load_settings()
    if not settings.azure_openai_image_endpoint:
        raise HTTPException(status_code=400, detail="Image endpoint not configured")

    headers = {"Content-Type": "application/json", "api-key": settings.azure_openai_api_key}
    payload = {
        "prompt": request.prompt,
        "width": 1024,
        "height": 1024,
        "model": settings.azure_openai_image_deployment
    }

    response = requests.post(
        settings.azure_openai_image_endpoint,
        headers=headers,
        json=payload,
        timeout=60
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {response.text}")

    result = response.json()
    if 'data' not in result or not result['data'] or 'b64_json' not in result['data'][0]:
        raise HTTPException(status_code=500, detail="No image data in response")

    image_path = save_image_from_base64(result['data'][0]['b64_json'], post_id)
    post.image_path = image_path
    save_post(post)

    image_filename = f"{post_id}.png"
    return {
        "success": True,
        "image_url": f"{BASE_URL}/posts/images/{image_filename}",
        "image_path": image_path
    }

@app.post("/api/posts/{post_id}/publish")
async def publish_post_instagram(post_id: str):
    """Publish a post to Instagram via Graph API"""
    posts = load_posts()
    post = next((p for p in posts if p.id == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    channels = load_channels()
    channel = next((ch for ch in channels if ch.id == post.channel_id), None)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    if not channel.instagram_user_id or not channel.instagram_access_token:
        raise HTTPException(
            status_code=400,
            detail="Instagram não configurado para este canal. Adicione o Instagram User ID e o Access Token nas configurações do canal."
        )

    settings = load_settings()
    public_base_url = settings.public_base_url.rstrip('/')
    image_url = f"{public_base_url}/posts/{post.image_path}"

    try:
        # Step 1: Create media container
        create_resp = requests.post(
            f"https://graph.facebook.com/v21.0/{channel.instagram_user_id}/media",
            params={
                "image_url": image_url,
                "caption": post.text,
                "access_token": channel.instagram_access_token,
            },
            timeout=30
        )
        create_data = create_resp.json()
        if create_resp.status_code != 200 or "id" not in create_data:
            error_msg = create_data.get("error", {}).get("message", create_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao criar container de mídia: {error_msg}")

        creation_id = create_data["id"]

        # Step 2: Publish the container
        pub_resp = requests.post(
            f"https://graph.facebook.com/v21.0/{channel.instagram_user_id}/media_publish",
            params={
                "creation_id": creation_id,
                "access_token": channel.instagram_access_token,
            },
            timeout=30
        )
        pub_data = pub_resp.json()
        if pub_resp.status_code != 200 or "id" not in pub_data:
            error_msg = pub_data.get("error", {}).get("message", pub_resp.text)
            raise HTTPException(status_code=502, detail=f"Erro ao publicar: {error_msg}")

        # Mark post as published
        post.published = True
        save_post(post)

        return {
            "success": True,
            "instagram_post_id": pub_data["id"],
            "post": post,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro inesperado: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

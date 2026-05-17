import asyncio
import logging
import time
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from f5_tts.api import F5TTS

project_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tts_server.log'),
        logging.StreamHandler()  # This will show logs in console too
    ]
)
logger = logging.getLogger(__name__)

# Idle timeout: seconds of inactivity before unloading the model (0 = disabled)
MODEL_IDLE_TIMEOUT: int = int(os.environ.get("MODEL_IDLE_TIMEOUT", "0"))

# Model state – None when unloaded
_model_lock = asyncio.Lock()
_f5tts: F5TTS | None = None
_active_inferences: int = 0
_last_request_time: float = time.time()


def _load_model_sync() -> F5TTS:
    logger.info("Loading F5-TTS model...")
    model = F5TTS(model="F5TTS_v1_Base")
    logger.info("F5-TTS model loaded")
    return model


def _release_model_memory():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    except Exception:
        pass


async def _ensure_model_loaded_and_pin() -> F5TTS:
    """Return the loaded model, incrementing the active-inference counter atomically.

    Caller MUST decrement _active_inferences in a finally block.
    """
    global _f5tts, _active_inferences
    async with _model_lock:
        if _f5tts is None:
            loop = asyncio.get_running_loop()
            _f5tts = await loop.run_in_executor(None, _load_model_sync)
        _active_inferences += 1
        return _f5tts


async def _idle_monitor_task() -> None:
    """Unload the model after MODEL_IDLE_TIMEOUT seconds of inactivity."""
    global _f5tts
    check_interval = min(60, max(5, MODEL_IDLE_TIMEOUT // 2))
    logger.info(
        f"Idle monitor started (timeout={MODEL_IDLE_TIMEOUT}s, "
        f"check_interval={check_interval}s)"
    )
    while True:
        await asyncio.sleep(check_interval)
        # Fast path: skip lock acquisition when clearly not eligible
        if _f5tts is None or _active_inferences > 0:
            continue
        if time.time() - _last_request_time < MODEL_IDLE_TIMEOUT:
            continue
        async with _model_lock:
            if (
                _f5tts is not None
                and _active_inferences == 0
                and time.time() - _last_request_time >= MODEL_IDLE_TIMEOUT
            ):
                logger.info(
                    f"Model idle for >{MODEL_IDLE_TIMEOUT}s – unloading to free memory"
                )
                _f5tts = None
                _release_model_memory()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    logger.info("F5-TTS Server starting up...")
    logger.info(f"Project root: {project_root}")

    if MODEL_IDLE_TIMEOUT > 0:
        logger.info(f"Model idle auto-unload enabled: timeout={MODEL_IDLE_TIMEOUT}s")
        asyncio.create_task(_idle_monitor_task())
    else:
        logger.info("Model idle auto-unload disabled (MODEL_IDLE_TIMEOUT=0)")

    logger.info("Server ready to accept TTS requests (model loads on first request)")

class TTSRequest(BaseModel):
    gen_text: str
    speed: float = 1.0
    nfe_steps: int = 32
    crossfade_duration: float = 0.15
    remove_silence: bool = False
    randomize_seed: bool = True
    seed: int | None = None
    ref_audio: str = "default/basic_ref_en.wav"
    ref_text: str = ""

@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open("static/tts/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index file not found")
    except UnicodeDecodeError:
        raise HTTPException(status_code=500, detail="Error reading index file")

@app.get("/conversation", response_class=HTMLResponse)
async def read_conversation():
    try:
        with open("static/conversation/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Conversation interface not found")
    except UnicodeDecodeError:
        raise HTTPException(status_code=500, detail="Error reading conversation interface")

@app.get("/ref-audios/")
async def list_reference_audios():
    """List available reference audio files from both default and custom folders"""
    ref_audios_path = os.path.join(project_root, "ref_audios")

    if not os.path.exists(ref_audios_path):
        return {"files": [], "default": "default/basic_ref_en.wav", "ref_texts": {}}

    try:
        # Get all audio files from both default and custom directories
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        files = []
        ref_texts = {}

        # Known reference texts for common files
        known_ref_texts = {
            "default/basic_ref_en.wav": "Some call me nature, others call me mother nature.",
            "default/basic_ref_zh.wav": "对，这就是我，万人敬仰的太乙真人。"
        }

        # Scan both default and custom folders
        folders_to_scan = ["default", "custom"]

        for folder in folders_to_scan:
            folder_path = os.path.join(ref_audios_path, folder)
            if not os.path.exists(folder_path):
                continue

            for filename in os.listdir(folder_path):
                if any(filename.lower().endswith(ext) for ext in audio_extensions):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        # Store with folder prefix for identification
                        file_key = f"{folder}/{filename}"
                        files.append(file_key)

                        # Try to find corresponding .txt file first
                        base_name = os.path.splitext(filename)[0]
                        txt_file_path = os.path.join(folder_path, f"{base_name}.txt")

                        if os.path.isfile(txt_file_path):
                            try:
                                with open(txt_file_path, 'r', encoding='utf-8') as f:
                                    ref_texts[file_key] = f.read().strip()
                                    logger.info(f"Loaded reference text from {folder}/{base_name}.txt")
                            except Exception as e:
                                logger.error(f"Error reading {txt_file_path}: {e}")
                                ref_texts[file_key] = known_ref_texts.get(file_key, "")
                        else:
                            # Fall back to known reference texts
                            ref_texts[file_key] = known_ref_texts.get(file_key, "")

        files.sort()  # Sort alphabetically

        return {
            "files": files,
            "default": "default/basic_ref_en.wav" if "default/basic_ref_en.wav" in files else (files[0] if files else None),
            "ref_texts": ref_texts
        }
    except Exception as e:
        logger.error(f"Error listing reference audio files: {e}")
        return {"files": [], "default": "default/basic_ref_en.wav", "ref_texts": {}}

@app.post("/upload-ref-audio/")
async def upload_reference_audio(file: UploadFile = File(...)):
    """Upload a reference audio file"""

    # Validate file type - support multiple formats as per F5-TTS
    allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/flac', 'audio/x-m4a', 'audio/ogg']
    allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']

    if file.content_type not in allowed_types:
        # Also check file extension as backup
        file_extension = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Invalid file type. Only WAV, MP3, FLAC, M4A, and OGG files are allowed.")

    # Validate file size (50MB limit)
    max_size = 50 * 1024 * 1024  # 50MB
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(status_code=400, detail="File size must be less than 50MB.")

    # Sanitize filename
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
    if not safe_filename or safe_filename.startswith('.'):
        safe_filename = f"uploaded_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file.filename.split('.')[-1].lower()}"

    # Check if file already exists and create unique name if needed
    ref_audios_path = os.path.join(project_root, "ref_audios")
    custom_folder_path = os.path.join(ref_audios_path, "custom")
    os.makedirs(custom_folder_path, exist_ok=True)

    final_filename = safe_filename
    counter = 1
    while os.path.exists(os.path.join(custom_folder_path, final_filename)):
        name, ext = os.path.splitext(safe_filename)
        final_filename = f"{name}_{counter}{ext}"
        counter += 1

    file_path = os.path.join(custom_folder_path, final_filename)

    try:
        # Write file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        logger.info(f"Reference audio uploaded successfully: custom/{final_filename}")

        return {
            "message": "File uploaded successfully",
            "filename": f"custom/{final_filename}",  # Return with folder prefix
            "size": len(file_content)
        }

    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        # Clean up partial file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

@app.post("/upload-text-file/")
async def upload_text_file(file: UploadFile = File(...)):
    """Upload a text file to the ref_audios/custom folder"""

    # Validate file type - only .txt files allowed
    if not file.filename.lower().endswith('.txt'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only TXT files are allowed.")

    # Validate file size (10MB limit for text files)
    max_size = 10 * 1024 * 1024  # 10MB
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(status_code=400, detail="File size must be less than 10MB.")

    # Validate that it's actually text content
    try:
        text_content = file_content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must contain valid UTF-8 text.")

    # Sanitize filename
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
    if not safe_filename or safe_filename.startswith('.'):
        safe_filename = f"uploaded_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # Ensure .txt extension
    if not safe_filename.lower().endswith('.txt'):
        safe_filename += '.txt'

    # Check if file already exists and create unique name if needed
    ref_audios_path = os.path.join(project_root, "ref_audios")
    custom_folder_path = os.path.join(ref_audios_path, "custom")
    os.makedirs(custom_folder_path, exist_ok=True)

    final_filename = safe_filename
    counter = 1
    while os.path.exists(os.path.join(custom_folder_path, final_filename)):
        name, ext = os.path.splitext(safe_filename)
        final_filename = f"{name}_{counter}{ext}"
        counter += 1

    file_path = os.path.join(custom_folder_path, final_filename)

    try:
        # Write text file to disk
        with open(file_path, "w", encoding="utf-8") as buffer:
            buffer.write(text_content)

        logger.info(f"Text file uploaded successfully: custom/{final_filename}")

        return {
            "message": "Text file uploaded successfully",
            "filename": final_filename,
            "content": text_content,
            "size": len(file_content)
        }

    except Exception as e:
        logger.error(f"Error saving uploaded text file: {e}")
        # Clean up partial file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Failed to save uploaded text file")

@app.delete("/delete-ref-audio/{file_path:path}")
async def delete_reference_audio(file_path: str):
    """Delete a reference audio file (only allows deleting custom files)"""

    # Security check - only allow deleting files from the custom folder
    if not file_path.startswith("custom/"):
        raise HTTPException(status_code=403, detail="Only custom reference audio files can be deleted")

    # Build the actual file path
    actual_file_path = os.path.join(project_root, "ref_audios", file_path)

    # Security check - ensure the file is within the ref_audios directory
    ref_audios_path = os.path.join(project_root, "ref_audios")
    if not os.path.abspath(actual_file_path).startswith(os.path.abspath(ref_audios_path)):
        raise HTTPException(status_code=403, detail="Access denied")

    # Check if file exists
    if not os.path.isfile(actual_file_path):
        raise HTTPException(status_code=404, detail="Reference audio file not found")

    # Extract just the filename for extension checking
    filename = os.path.basename(file_path)
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    if not any(filename.lower().endswith(ext) for ext in audio_extensions):
        raise HTTPException(status_code=400, detail="Invalid audio file format")

    try:
        # Delete the audio file
        os.remove(actual_file_path)
        logger.info(f"Deleted reference audio file: {file_path}")

        # Also try to delete the corresponding .txt file if it exists
        base_name = os.path.splitext(filename)[0]
        txt_file_path = os.path.join(os.path.dirname(actual_file_path), f"{base_name}.txt")
        if os.path.isfile(txt_file_path):
            os.remove(txt_file_path)
            logger.info(f"Deleted corresponding text file: custom/{base_name}.txt")

        return {
            "message": "Reference audio file deleted successfully",
            "filename": file_path
        }

    except Exception as e:
        logger.error(f"Error deleting reference audio file: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete reference audio file")

@app.get("/ref-audios/{file_path:path}")
async def serve_reference_audio(file_path: str):
    """Serve reference audio files from default or custom folders"""
    # Handle both "folder/filename" and just "filename" formats
    if "/" not in file_path:
        # Legacy format - assume it's in the root ref_audios directory (backward compatibility)
        actual_file_path = os.path.join(project_root, "ref_audios", file_path)
    else:
        # New format with folder prefix
        actual_file_path = os.path.join(project_root, "ref_audios", file_path)

    # Security check - ensure the file is within the ref_audios directory
    ref_audios_path = os.path.join(project_root, "ref_audios")
    if not os.path.abspath(actual_file_path).startswith(os.path.abspath(ref_audios_path)):
        raise HTTPException(status_code=403, detail="Access denied")

    # Check if file exists and is a valid audio file
    if not os.path.isfile(actual_file_path):
        raise HTTPException(status_code=404, detail="Reference audio file not found")

    # Extract just the filename for extension checking
    filename = os.path.basename(file_path)
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    if not any(filename.lower().endswith(ext) for ext in audio_extensions):
        raise HTTPException(status_code=400, detail="Invalid audio file format")

    return FileResponse(actual_file_path, media_type="audio/wav", filename=filename)

@app.post("/tts/")
async def text_to_speech(request: TTSRequest):
    global _active_inferences, _last_request_time

    start_time = time.time()
    _last_request_time = start_time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    # Log the incoming request
    logger.info(f"TTS Request received - ID: {timestamp}")
    logger.info(f"Input text: '{request.gen_text[:100]}{'...' if len(request.gen_text) > 100 else ''}'")
    logger.info(f"Text length: {len(request.gen_text)} characters")
    logger.info(f"Speed setting: {request.speed}x")
    logger.info(f"NFE steps: {request.nfe_steps}")
    logger.info(f"Cross-fade duration: {request.crossfade_duration}s")
    logger.info(f"Remove silence: {request.remove_silence}")
    logger.info(f"Randomize seed: {request.randomize_seed}")
    logger.info(f"Seed: {request.seed if not request.randomize_seed else 'random'}")
    logger.info(f"Reference audio: {request.ref_audio}")
    logger.info(f"Reference text: '{request.ref_text[:50]}{'...' if len(request.ref_text) > 50 else ''}'" if request.ref_text else "Reference text: (auto-transcribe)")

    # Handle folder-aware ref audio paths
    ref_audio_path = os.path.join(project_root, "ref_audios", request.ref_audio)
    output_filename = f"{timestamp}.wav"
    output_path = os.path.join("output", output_filename)

    logger.info(f"Using reference audio: {ref_audio_path}")
    logger.info(f"Output file: {output_path}")

    # Reference text priority: user textarea > .txt file > F5-TTS auto-transcription
    # (the API's infer() handles auto-transcription when ref_text is empty)
    processed_ref_text = request.ref_text.strip()

    if processed_ref_text:
        logger.info("Using user-provided reference text from textarea")
    else:
        # Try to find corresponding .txt file
        if "/" in request.ref_audio:
            folder_path, audio_filename = request.ref_audio.split("/", 1)
            base_name = os.path.splitext(audio_filename)[0]
            txt_file_path = os.path.join(project_root, "ref_audios", folder_path, f"{base_name}.txt")
        else:
            base_name = os.path.splitext(request.ref_audio)[0]
            txt_file_path = os.path.join(project_root, "ref_audios", f"{base_name}.txt")

        if os.path.isfile(txt_file_path):
            try:
                with open(txt_file_path, 'r', encoding='utf-8') as f:
                    processed_ref_text = f.read().strip()
                logger.info(f"Using reference text from {base_name}.txt file")
            except Exception as e:
                logger.warning(f"Error reading {txt_file_path}: {e}")
                processed_ref_text = ""

        if not processed_ref_text:
            logger.info("No reference text found, API will auto-transcribe")

    # Determine seed: None lets the API pick a random one
    if request.randomize_seed:
        seed = None
        logger.info("Using random seed (API-generated)")
    else:
        if request.seed is not None and 0 <= request.seed <= 2**31 - 1:
            seed = request.seed
        else:
            logger.warning(f"Invalid seed {request.seed}, falling back to random")
            seed = None
        logger.info(f"Using seed: {seed}")

    # Load model if needed; increment active-inference counter atomically under the lock
    # so the idle monitor cannot unload between the load check and inference start.
    model = await _ensure_model_loaded_and_pin()

    logger.info("Starting TTS generation...")

    try:
        loop = asyncio.get_running_loop()
        wav, sr, spec = await loop.run_in_executor(
            None,
            lambda: model.infer(
                ref_file=ref_audio_path,
                ref_text=processed_ref_text,
                gen_text=request.gen_text,
                nfe_step=request.nfe_steps,
                cross_fade_duration=request.crossfade_duration,
                speed=request.speed,
                remove_silence=request.remove_silence,
                file_wave=output_path,
                seed=seed,
                show_info=logger.info,
            ),
        )
        used_seed = model.seed
        logger.info("TTS generation completed successfully")
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error during TTS generation: {e}")
    finally:
        _active_inferences -= 1
        _last_request_time = time.time()

    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        generation_time = time.time() - start_time

        logger.info(f"Audio file generated successfully")
        logger.info(f"File size: {file_size} bytes ({file_size/1024:.1f} KB)")
        logger.info(f"Generation time: {generation_time:.2f} seconds")
        logger.info(f"Request {timestamp} completed successfully")

        def iter_file():
            with open(output_path, "rb") as file_like:
                yield from file_like

        headers = {
            "X-Generation-Time": str(generation_time),
            "X-File-Size": str(file_size),
            "X-Used-Seed": str(used_seed),
        }

        return StreamingResponse(iter_file(), media_type="audio/wav", headers=headers)
    else:
        logger.error(f"Generated audio file not found at {output_path}")
        logger.error(f"Request {timestamp} failed - file not found")
        raise HTTPException(status_code=404, detail="Generated audio file not found. The TTS inference may have failed silently.")

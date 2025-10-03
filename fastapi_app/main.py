import os
import json
import time
import hashlib
from collections import OrderedDict
import sys
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
    # Ensure UTF-8 output on Windows to avoid UnicodeEncodeError when model prints special chars
os.environ.setdefault('PYTHONUTF8', '1')
try:
    if sys.stdout and sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr and sys.stderr.encoding != 'utf-8':
        import io
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

class SafeWriter(io.TextIOBase):
    def __init__(self, buffer):
        self._buffer = buffer
    def write(self, s):
        try:
            if s is None:
                return 0
            if not isinstance(s, str):
                s = str(s)
            data = s.encode('utf-8', errors='replace')
            try:
                self._buffer.write(data)
                try:
                    self._buffer.flush()
                except Exception:
                    pass
            except Exception:
                pass
            return len(s)
        except Exception:
            return 0
    def flush(self):
        try:
            self._buffer.flush()
        except Exception:
            pass

try:
    if hasattr(sys, '__stdout__') and getattr(sys.__stdout__, 'buffer', None):
        sys.stdout = SafeWriter(sys.__stdout__.buffer)
    if hasattr(sys, '__stderr__') and getattr(sys.__stderr__, 'buffer', None):
        sys.stderr = SafeWriter(sys.__stderr__.buffer)
except Exception:
    pass

try:
    import builtins as _builtins
    _real_print = _builtins.print
    def _safe_print(*args, **kwargs):
        try:
            sep = kwargs.get('sep', ' ')
            end = kwargs.get('end', '\n')
            file = kwargs.get('file', sys.stdout)
            text = sep.join(str(a) for a in args) + end
            try:
                if hasattr(file, 'write'):
                    file.write(text)
                else:
                    sys.stdout.write(text)
            except Exception:
                try:
                    _real_print(text, file=sys.__stderr__)
                except Exception:
                    pass
        except Exception:
            try:
                _real_print('<<print wrapper error>>', file=sys.__stderr__)
            except Exception:
                pass
    _builtins.print = _safe_print
except Exception:
    pass
except Exception:
    pass
# Monkeypatch builtins.print to be UTF-8-safe
try:
    import builtins as _builtins
    _real_print = _builtins.print
    def _safe_print(*args, **kwargs):
        try:
            sep = kwargs.get('sep', ' ')
            end = kwargs.get('end', '\n')
            file = kwargs.get('file', sys.stdout)
            text = sep.join(str(a) for a in args) + end
            try:
                if hasattr(file, 'buffer'):
                    file.buffer.write(text.encode('utf-8', errors='replace'))
                    file.flush()
                else:
                    file.write(text)
                    file.flush()
            except Exception:
                try:
                    _real_print(text, file=sys.__stderr__)
                except Exception:
                    pass
        except Exception:
            try:
                _real_print('<<print wrapper error>>', file=sys.__stderr__)
            except Exception:
                pass
    _builtins.print = _safe_print
except Exception:
    pass
from gradio_client import Client, handle_file
import asyncio as _asyncio
import tempfile
import sys
os.environ.setdefault('PYTHONUTF8', '1')
try:
    if sys.stdout and sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr and sys.stderr.encoding != 'utf-8':
        import io
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

# Standalone IndexTTS2 support (bypass Gradio)
USE_STANDALONE_TTS = os.getenv("USE_STANDALONE_TTS", "0") == "1"
index_tts_instance = None
index_tts_lock = _asyncio.Lock()


async def get_index_tts_instance():
    """Lazily initialize IndexTTS2 for main API when running in standalone mode.

    Uses real-time GPU VRAM query (nvidia-smi) to pick a GPU that has
    enough free memory. Honors INDEXTTS_CUDA_DEVICE and INDEXTTS_REQUIRED_VRAM_MB.
    Can auto-enable fp16 if INDEXTTS_ALLOW_AUTO_FP16=1 and no single GPU meets the
    fp32 requirement but one meets half of it.
    """
    global index_tts_instance
    async with index_tts_lock:
        if index_tts_instance is not None:
            return index_tts_instance

        from indextts.infer_v2 import IndexTTS2
        import torch as _torch
        import subprocess

        os.environ.setdefault('TORCH_CUDA_ALLOC_CONF', os.getenv('TORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64'))

        # Enforce CUDA-only
        if not _torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This service requires a CUDA-enabled PyTorch. Install CUDA-enabled torch and restart the service.")

        def _query_gpu_memory():
            try:
                out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL, universal_newlines=True)
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                res = []
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        idx = int(parts[0]); total = int(parts[1]); used = int(parts[2]); free = total - used
                        res.append((idx, free, total))
                return res
            except Exception:
                try:
                    infos = []
                    cnt = _torch.cuda.device_count()
                    for i in range(cnt):
                        prop = _torch.cuda.get_device_properties(i)
                        total = int(prop.total_memory / (1024*1024))
                        infos.append((i, total, total))
                    return infos
                except Exception:
                    return []

        def _choose_cuda_device():
            env_val = os.getenv('INDEXTTS_CUDA_DEVICE')
            dev_count = 0
            try:
                dev_count = _torch.cuda.device_count()
            except Exception:
                dev_count = 0

            # Respect explicit env var
            if env_val:
                try:
                    if isinstance(env_val, str) and env_val.startswith('cuda:'):
                        idx = int(env_val.split(':', 1)[1])
                    else:
                        idx = int(env_val)
                    if idx < 0 or (dev_count and idx >= dev_count):
                        print(f">> INDEXTTS_CUDA_DEVICE={env_val} out of range, falling back to auto selection")
                    else:
                        return (f"cuda:{idx}", False)
                except Exception:
                    print(f">> Failed to parse INDEXTTS_CUDA_DEVICE='{env_val}', falling back to automatic selection")

            required_vram = int(os.getenv('INDEXTTS_REQUIRED_VRAM_MB', '10000'))
            allow_auto_fp16 = os.getenv('INDEXTTS_ALLOW_AUTO_FP16', '1').strip() in ('1', 'true', 'True')

            gpu_infos = _query_gpu_memory()
            if gpu_infos:
                # pick GPU with sufficient free memory
                for idx, free, total in sorted(gpu_infos, key=lambda x: -x[1]):
                    if free >= required_vram:
                        return (f"cuda:{idx}", False)

                if allow_auto_fp16:
                    req2 = max(1024, required_vram // 2)
                    for idx, free, total in sorted(gpu_infos, key=lambda x: -x[1]):
                        if free >= req2:
                            print(f">> Not enough VRAM for fp32; enabling fp16 and selecting cuda:{idx}")
                            return (f"cuda:{idx}", True)

                best = max(gpu_infos, key=lambda x: x[1])
                print(f">> No GPU has required VRAM ({required_vram}MB); selecting gpu {best[0]} with free {best[1]}MB")
                return (f"cuda:{best[0]}", False)

            if dev_count <= 1:
                return ("cuda:0", False)

            try:
                if sys.stdin and sys.stdin.isatty():
                    print(f">> Detected {dev_count} CUDA devices:")
                    for i in range(dev_count):
                        try:
                            name = _torch.cuda.get_device_name(i)
                        except Exception:
                            name = f"cuda:{i}"
                        print(f"   [{i}] {name}")
                    sel_raw = input(f"Select device index to use for IndexTTS2 [0-{dev_count-1}] (default 0): ")
                    try:
                        sel = int(sel_raw) if sel_raw.strip() != '' else 0
                    except Exception:
                        sel = 0
                    if sel < 0 or sel >= dev_count:
                        print(f">> Selection {sel} out of range, using 0")
                        sel = 0
                    return (f"cuda:{sel}", False)
            except Exception:
                pass

            return ("cuda:0", False)

        device, auto_fp16 = _choose_cuda_device()

        # Optionally enable fp16 via explicit env var or due to automatic selection
        use_fp16_env = os.getenv('INDEXTTS_USE_FP16', '0')
        use_fp16 = str(use_fp16_env).strip() in ('1', 'true', 'True') or bool(auto_fp16)
        if use_fp16:
            print('>> INDEXTTS_USE_FP16 enabled for main API')

        # Apply per-process memory fraction if requested
        mem_frac = os.getenv('INDEXTTS_CUDA_MEM_FRACTION')
        if mem_frac:
            try:
                frac = float(mem_frac)
                if 0.0 < frac <= 1.0:
                    try:
                        dev_idx = int(device.split(':')[-1]) if isinstance(device, str) and 'cuda' in device else 0
                        _torch.cuda.set_per_process_memory_fraction(frac, dev_idx)
                        print(f">> Set per-process CUDA memory fraction to {frac} on {device}")
                    except Exception as e:
                        print(f">> Failed to set per-process memory fraction: {e}")
            except Exception:
                print(f">> Invalid INDEXTTS_CUDA_MEM_FRACTION='{mem_frac}', must be float in (0,1]")

        print(f">> Initializing IndexTTS2 for main API (device={device or 'cpu'}) ...")
        tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=use_fp16, use_cuda_kernel=False, device=device)
        index_tts_instance = tts
        print(">> IndexTTS2 initialized for main API")
        return index_tts_instance
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import BaseModel
from typing import Literal
import uvicorn
import asyncio
import traceback
import httpx # import httpx for sending HTTP requests
from fastapi.middleware.cors import CORSMiddleware

# 1. Import the WebSocket manager
from .websocket_manager import router as websocket_router, manager as websocket_manager

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global gradio_client, MODEL_PROMPT_MAP
    print("🚀 Initializing service...")

    # Load model reference audios
    print("🔍 Loading model reference audios...")
    MODEL_PROMPT_MAP = load_model_prompt_map()
    print("✅ Model reference audios loaded.")

    # Print API and WebSocket addresses
    host = os.getenv("UVICORN_HOST", "127.0.0.1")
    port = int(os.getenv("UVICORN_PORT", "8010"))
    print(f"\n🎉 Service initialized")
    print(f"🔗 API docs (Swagger UI): http://{host}:{port}/docs")
    print(f"🔌 WebSocket endpoint: ws://{host}:{port}/ws\n")

    # Try to connect to the Gradio API. Increase retries and use backoff to allow Gradio time to become ready.
    # Ensure local connections bypass any environment HTTP proxy which can cause connection failures
    os.environ.setdefault('NO_PROXY', '127.0.0.1,localhost')
    os.environ.setdefault('no_proxy', '127.0.0.1,localhost')

    print(f"🔧 DEBUG GRADIO_URL={GRADIO_URL} NO_PROXY={os.environ.get('NO_PROXY')} no_proxy={os.environ.get('no_proxy')}")

    # Start a non-blocking background task that will try to connect to Gradio
    # Only start this when not using the standalone IndexTTS2 mode.
    reconnect_task = None
    if not USE_STANDALONE_TTS:
        async def _gradio_reconnect_loop():
            global gradio_client
            while True:
                if gradio_client is None:
                    try:
                        gradio_client = Client(GRADIO_URL)
                        print(f"✅ Gradio client connected (background): {GRADIO_URL}")
                        try:
                            asyncio.create_task(send_startup_request())
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"ℹ️ Gradio client not ready: {e}")
                await asyncio.sleep(5)

        reconnect_task = asyncio.create_task(_gradio_reconnect_loop())
    
    # Start background monitoring task
    monitor_task = asyncio.create_task(monitor_inactivity())
    
    yield
    
    # Shutdown
    print("🔌 Shutting down service...")
    monitor_task.cancel()
    try:
        reconnect_task.cancel()
    except Exception:
        pass
    try:
        await monitor_task
    except asyncio.CancelledError:
        print("✅ Background monitor task cancelled successfully.")

    # Cleanup temporary files created by the service (safe, repo-local directory)
    try:
        CLEANUP_DIR = os.getenv('SERVICE_TMP_DIR', os.path.join(os.getcwd(), '.tmp'))
        if os.path.isdir(CLEANUP_DIR):
            import shutil
            shutil.rmtree(CLEANUP_DIR)
            print(f"🧹 Removed temporary directory: {CLEANUP_DIR}")
        else:
            print(f"ℹ️ No temporary directory to remove at: {CLEANUP_DIR}")
    except Exception as e:
        print(f"⚠️ Failed to cleanup temporary files: {e}")


app = FastAPI(lifespan=lifespan)

# --- 新增：用于记录被拒绝的 WebSocket 连接的中间件 ---
ALLOWED_ORIGINS = {"http://localhost:19100", "http://hdcotd--8010.ap-shanghai.cloudstudio.work"}

@app.middleware("http")
async def log_denied_websocket_connections(request: Request, call_next):
    if request.scope["type"] == "websocket":
        origin = request.headers.get("origin")
        if origin not in ALLOWED_ORIGINS:
            print(f"[CORS] ❌ Denied WebSocket connection attempt from unauthorized origin <{origin}>.", flush=True)
    
    response = await call_next(request)
    return response
# --- 中间件结束 ---

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:19100", "http://hdcotd--8010.ap-shanghai.cloudstudio.work"],  # 允许来自指定源的请求
    allow_credentials=True, # 允许携带 cookie
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 请求头
)

# 2. 将 WebSocket 路由集成到主应用中
app.include_router(websocket_router)


@app.get('/v1/voxta/provider')
def voxta_provider_main():
    """Return indextts_voxta_provider.json for Voxta discovery (main API)."""
    try:
        here = os.getcwd()
        provider_path = os.path.join(here, 'indextts_voxta_provider.json')
        if not os.path.exists(provider_path):
            return JSONResponse(status_code=404, content={"error": "provider file not found"})
        with open(provider_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Allow overriding Gradio port with GRADIO_PORT env var for environments where 7860 is blocked
gradio_port = os.getenv('GRADIO_PORT')
if gradio_port:
    GRADIO_URL = f"http://127.0.0.1:{gradio_port}/"
else:
    GRADIO_URL = os.getenv("GRADIO_URL", "http://127.0.0.1:7860/")

gradio_client = None

# --- In-memory cache configuration ---
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "100")) # 最大缓存条目数
# 使用 OrderedDict 实现 LRU 缓存
# 键是请求的哈希，值是音频内容的bytes
in_memory_cache = OrderedDict()

MODEL_PROMPT_MAP = {}
def load_model_prompt_map():
    """
    Dynamically load all .wav and .m4a files under the model_wav directory and
    generate MODEL_PROMPT_MAP. Keys are filenames (without extension), values
    are the relative file paths.
    """
    model_wav_dir = "model_wav"
    supported_extensions = (".wav", ".m4a")  # 支持的文件扩展名
    if not os.path.isdir(model_wav_dir):
        print(f"⚠️ Warning: '{model_wav_dir}' directory not found; no model reference audios will be loaded.")
        return {}

    prompt_map = {}
    for filename in os.listdir(model_wav_dir):
        if filename.lower().endswith(supported_extensions):
            model_name = os.path.splitext(filename)[0]
            prompt_map[model_name] = os.path.join(model_wav_dir, filename)
            print(f"  - Found model: '{model_name}' -> '{prompt_map[model_name]}'")
    
    if not prompt_map:
        print(f"⚠️ Warning: no supported audio files found in '{model_wav_dir}' ({', '.join(supported_extensions)}).")
        
    return prompt_map

DEFAULT_PROMPT_AUDIO_PATH = "model_wav/default_prompt.wav"

# Serve examples directory as static files so sample audio and metadata are reachable via HTTP
EXAMPLES_DIR = os.path.join(os.getcwd(), "examples")
if os.path.isdir(EXAMPLES_DIR):
    try:
        app.mount("/examples", StaticFiles(directory=EXAMPLES_DIR), name="examples")
        print(f"✅ Mounted examples folder at /examples (serving from {EXAMPLES_DIR})")
    except Exception as e:
        print(f"⚠️ Failed to mount examples directory: {e}")
else:
    print(f"ℹ️ Examples directory not found at {EXAMPLES_DIR}; /examples endpoint will be unavailable.")

class SpeechRequest(BaseModel):
    model: str
    # Optional Voxta-style voice id (examples/<voice>.wav). If provided, this will be
    # preferred for zero-shot speaker prompt resolution over the model default.
    voice: str = ""
    input: str
    # ISO language code, e.g. 'en', 'zh', 'ja'. Default to 'en'.
    language: str = "en"
    emo_control_method: Literal['Same as the voice reference', 'Use emotion reference audio', 'Use emotion vector', 'Use text description to control emotion'] = "Same as the voice reference"
    emo_weight: float = 0.8
    # Optional named emotion label (e.g. 'happy', 'sad') that Voxta can send.
    emotion: str = ""
    vec1: float = 0
    vec2: float = 0
    vec3: float = 0
    vec4: float = 0
    vec5: float = 0
    vec6: float = 0
    vec7: float = 0
    vec8: float = 0
    emo_text: str = ""
    emo_random: bool = False
    max_text_tokens_per_sentence: int = 120
    do_sample: bool = True
    top_p: float = 0.8
    top_k: int = 30
    temperature: float = 0.8
    length_penalty: float = 0.0
    num_beams: int = 3
    repetition_penalty: float = 10.0
    max_mel_tokens: int = 1500


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_gradio_with_retry(client, *args, **kwargs):
    return client.predict(*args, **kwargs)

@app.post('/v1/audio/speech')
async def create_speech(speech_request: SpeechRequest):
    try:
        # LRU cache key (include selected voice to avoid returning cached audio for a different speaker)
        selected_voice = getattr(speech_request, 'voice', '') if hasattr(speech_request, 'voice') else ''
        cache_key_content = f"{speech_request.model}:{speech_request.input}:{selected_voice}"
        cache_key = hashlib.md5(cache_key_content.encode()).hexdigest()

        if cache_key in in_memory_cache:
            in_memory_cache.move_to_end(cache_key)
            return Response(content=in_memory_cache[cache_key], media_type="audio/wav")

        # If configured to use standalone IndexTTS2, call it directly and bypass Gradio
        if USE_STANDALONE_TTS:
            # Resolve speaker prompt preference: Voxta `voice` -> examples/<voice>.wav -> examples/<model>.wav -> model_wav/<model>.wav
            spk_prompt = None
            voice = getattr(speech_request, 'voice', None) if hasattr(speech_request, 'voice') else None
            if voice:
                candidate = os.path.join(EXAMPLES_DIR, f"{voice}.wav")
                if os.path.exists(candidate):
                    spk_prompt = candidate

            if not spk_prompt:
                ex_candidate = os.path.join(EXAMPLES_DIR, f"{speech_request.model}.wav")
                if os.path.exists(ex_candidate):
                    spk_prompt = ex_candidate
                else:
                    prompt_file_path = MODEL_PROMPT_MAP.get(speech_request.model)
                    if prompt_file_path:
                        spk_prompt = os.path.join(os.getcwd(), prompt_file_path)
                    else:
                        spk_prompt = os.path.join(os.getcwd(), DEFAULT_PROMPT_AUDIO_PATH)

            # Initialize IndexTTS2 instance and call infer
            tts = await get_index_tts_instance()

            # Map emo control
            emo_choice = speech_request.emo_control_method
            emo_audio_prompt = None
            emo_vector = None
            use_emo_text = False
            emo_text_value = speech_request.emo_text if speech_request.emo_text else (speech_request.emotion or None)

            if emo_choice == 'Same as the voice reference':
                emo_audio_prompt = None
            elif emo_choice == 'Use emotion reference audio':
                emo_audio_prompt = spk_prompt
            elif emo_choice == 'Use emotion vector':
                emo_vector = [speech_request.vec1, speech_request.vec2, speech_request.vec3, speech_request.vec4,
                              speech_request.vec5, speech_request.vec6, speech_request.vec7, speech_request.vec8]
            elif emo_choice == 'Use text description to control emotion':
                use_emo_text = True

            tmpfd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="indextts_")
            os.close(tmpfd)
            try:
                tts.infer(spk_audio_prompt=spk_prompt,
                          text=speech_request.input,
                          output_path=tmp_path,
                          emo_audio_prompt=emo_audio_prompt,
                          emo_alpha=speech_request.emo_weight,
                          emo_vector=emo_vector,
                          use_emo_text=use_emo_text,
                          emo_text=emo_text_value,
                          use_random=speech_request.emo_random,
                          verbose=False,
                          max_text_tokens_per_segment=speech_request.max_text_tokens_per_sentence,
                          do_sample=bool(speech_request.do_sample),
                          top_p=float(speech_request.top_p),
                          top_k=int(speech_request.top_k) if int(speech_request.top_k) > 0 else None,
                          temperature=float(speech_request.temperature),
                          length_penalty=float(speech_request.length_penalty),
                          num_beams=int(speech_request.num_beams),
                          repetition_penalty=float(speech_request.repetition_penalty),
                          max_mel_tokens=int(speech_request.max_mel_tokens))
            except Exception as e:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                raise HTTPException(status_code=500, detail=f"TTS engine failed: {e}")

            if not os.path.exists(tmp_path):
                raise HTTPException(status_code=500, detail="TTS engine did not produce output file")

            with open(tmp_path, 'rb') as fh:
                audio_content = fh.read()

            if len(in_memory_cache) >= MAX_CACHE_SIZE:
                in_memory_cache.popitem(last=False)
            in_memory_cache[cache_key] = audio_content

            try:
                os.remove(tmp_path)
            except Exception:
                pass

            return Response(content=audio_content, media_type="audio/wav")

        # Fallback to Gradio proxy behavior (existing code)
        if not gradio_client:
            # Look for examples/<model>.wav first, then model_wav/default_prompt.wav
            fallback_paths = []
            ex_path = os.path.join(os.getcwd(), 'examples', f"{speech_request.model}.wav")
            if os.path.exists(ex_path):
                fallback_paths.append(ex_path)
            default_path = os.path.join(os.getcwd(), DEFAULT_PROMPT_AUDIO_PATH)
            if os.path.exists(default_path):
                fallback_paths.append(default_path)

            if fallback_paths:
                # Return the first available fallback audio file
                fp = fallback_paths[0]
                try:
                    with open(fp, 'rb') as fh:
                        data = fh.read()
                    print(f"⚠️ Gradio not connected — returning local fallback audio for model '{speech_request.model}' from {fp}")
                    return Response(content=data, media_type='audio/wav')
                except Exception as e:
                    print(f"⚠️ Failed to read fallback audio {fp}: {e}")
            # No fallback available; return a clear 503 with guidance
            raise HTTPException(status_code=503, detail=("Gradio backend is not connected and no local fallback audio found. "
                                                         "Start the web UI or provide a sample at examples/<model>.wav or model_wav/default_prompt.wav."))

        # --- Existing Gradio proxy generation continues below ---
    except HTTPException:
        raise  # re-raise handled HTTPException
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"💥 Unexpected error while processing request: {e}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")

@app.get('/health')
def health_check():
    cache_info = {
        "cache_type": "in_memory",
        "current_entries": len(in_memory_cache),
        "max_entries": MAX_CACHE_SIZE
    }
    
    if gradio_client:
        return {"status": "ok", "gradio_connected": True, "cache_info": cache_info, "message": "Service healthy; Gradio client is connected."}
    else:
        return {"status": "degraded", "gradio_connected": False, "cache_info": cache_info, "message": "Gradio client not connected; some features may be limited."}


@app.get('/v1/voices')
def list_voices(request: Request, full: bool = False):
    """
    List voices. Default returns a Voxta-friendly bare array of objects like
    {"label": "<name>", "parameters": {"voice": "<id>"}}
    For backward compatibility, callers may request `?full=true` to receive
    the richer OpenAI-style wrapper: {"voices": [...]} with extra metadata.
    """
    examples_dir = EXAMPLES_DIR
    if not os.path.isdir(examples_dir):
        return {"voices": []} if full else []

    supported_audio_exts = {'.wav', '.mp3', '.m4a', '.ogg'}
    simple_list = []
    rich_list = []
    for fname in sorted(os.listdir(examples_dir)):
        fpath = os.path.join(examples_dir, fname)
        if os.path.isfile(fpath):
            name, ext = os.path.splitext(fname)
            if ext.lower() in supported_audio_exts:
                # simple entry for Voxta
                simple_list.append({"label": name, "parameters": {"voice": name}})

                # build rich metadata when requested
                public_url = os.getenv('PUBLIC_URL')
                if public_url:
                    base = public_url.rstrip('/')
                else:
                    base = str(request.base_url).rstrip('/')
                sample_url = f"{base}/examples/{fname}"
                meta = {"id": name, "name": name, "language": "und", "sample_url": sample_url}
                meta_path = os.path.join(examples_dir, f"{name}.json")
                if os.path.exists(meta_path):
                    try:
                        import json
                        with open(meta_path, 'r', encoding='utf-8') as fh:
                            j = json.load(fh)
                            meta.update({k: v for k, v in j.items() if k not in meta})
                    except Exception as e:
                        print(f"⚠️ Failed to load metadata for {name}: {e}")
                rich_list.append({
                    "id": name,
                    "name": meta.get('name', name),
                    "language": meta.get('language', 'und'),
                    "sample_url": sample_url,
                    "tts": {k: v for k, v in meta.items() if k not in {'id', 'name', 'language', 'sample_url'}}
                })

    return {"voices": rich_list} if full else simple_list


@app.get('/v1/voxta/voices')
def list_voxta_voices(request: Request):
    """Return voices formatted to match the simple Voxta `VoicesFormat` example.
    Each entry will be an object like: {"label": "<name>", "parameters": {"voice": "<id>"}}
    """
    res = []
    examples_dir = EXAMPLES_DIR
    if not os.path.isdir(examples_dir):
        return JSONResponse(status_code=200, content={"voices": []})

    supported_audio_exts = {'.wav', '.mp3', '.m4a', '.ogg'}
    for fname in sorted(os.listdir(examples_dir)):
        fpath = os.path.join(examples_dir, fname)
        if os.path.isfile(fpath):
            name, ext = os.path.splitext(fname)
            if ext.lower() in supported_audio_exts:
                entry = {
                    "label": name,
                    "voice": name,
                    "parameters": {"voice": name}
                }
                res.append(entry)

    # Voxta expects a bare JSON array; return the list directly to avoid parsing errors
    return res


@app.get('/get_predefined_voices')
def get_predefined_voices(request: Request):
    """Compatibility endpoint for Voxta legacy expectations.
    Returns a bare array of {label, parameters:{voice}} entries (no wrapper object).
    Accepts optional `Authorization` header and ignores it for now.
    """
    res = []
    examples_dir = EXAMPLES_DIR
    if not os.path.isdir(examples_dir):
        return []

    supported_audio_exts = {'.wav', '.mp3', '.m4a', '.ogg'}
    for fname in sorted(os.listdir(examples_dir)):
        fpath = os.path.join(examples_dir, fname)
        if os.path.isfile(fpath):
            name, ext = os.path.splitext(fname)
            if ext.lower() in supported_audio_exts:
                entry = {
                    "label": name,
                    "voice": name,
                    "parameters": {"voice": name}
                }
                res.append(entry)

    return res

# --- 新增：服务活动监控 ---
INACTIVITY_TIMEOUT = 1800  # 30分钟的秒数
# 4. 移除不再需要的 NOTIFICATION_URL
# NOTIFICATION_URL = "http://127.0.0.1:8082/notify"
last_activity_time = time.time()

@app.middleware("http")
async def update_activity_timestamp(request: Request, call_next):
    """中间件，在每个请求处理后更新活动时间戳。"""
    global last_activity_time
    last_activity_time = time.time()
    response = await call_next(request)
    return response

async def monitor_inactivity():
    """后台任务，监控并处理服务长时间无活动的情况。"""
    global last_activity_time
    while True:
        try:
            await asyncio.sleep(60)  # check every 60 seconds
            idle_time = time.time() - last_activity_time

            if idle_time > INACTIVITY_TIMEOUT:
                print(f"🚨 Service idle for more than {INACTIVITY_TIMEOUT} seconds; sending notification...")
                # Broadcast a notification via the websocket manager
                try:
                    await websocket_manager.broadcast("stop edge")
                    print(f"✅ Notification sent via WebSocket manager.")
                except Exception:
                    print("⚠️ Failed to broadcast stop message to websocket clients:")
                    traceback.print_exc()
                # Reset timer to avoid immediate repeats
                last_activity_time = time.time()
            else:
                try:
                    active_count = len(websocket_manager.active_connections)
                except Exception:
                    active_count = 0
                print(f"Info: service idle time {idle_time:.2f}s (active connections: {active_count})", flush=True)
        except asyncio.CancelledError:
            # Allow task cancellation to propagate cleanly on shutdown
            raise
        except Exception:
            print("⚠️ Unexpected error in monitor_inactivity:")
            traceback.print_exc()

# --- 新增部分：自动发送请求 ---

async def send_startup_request():
    """
    在服务启动后发送一个测试请求。
    """
    await asyncio.sleep(2) # 等待5秒，确保服务完全启动并监听
    print("\n🌟 服务启动成功！尝试发送一个自动请求...")
    
    # 获取当前运行的Uvicorn地址和端口
    # 注意：在实际部署中，可能需要根据环境变量或配置来确定HOST和PORT
    host = os.getenv("UVICORN_HOST", "127.0.0.1")
    port = os.getenv("UVICORN_PORT", "8010")
    
    request_url = f"http://{host}:{port}/v1/audio/speech"
    headers = {"Content-Type": "application/json"}
    
    # 动态构造请求体
    # 检查 MODEL_PROMPT_MAP 是否有已加载的模型
    if MODEL_PROMPT_MAP:
        # 使用列表中的第一个模型
        first_model_name = next(iter(MODEL_PROMPT_MAP))
    else:
        # 如果没有找到任何模型，则使用一个默认的备用名称
        first_model_name = "default_model"
        print("⚠️ 警告：未在 model_wav 目录中找到任何模型，将使用默认模型名进行启动测试。")

    payload = {
        "model": first_model_name, 
        "input": "您好，欢迎使用。"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            print(f"发送第一次请求: {payload['input']}")
            response = await client.post(request_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ 第一次自动请求发送成功！状态码: {response.status_code}, 响应内容类型: {response.headers.get('Content-Type')}")
                # 等待一小段时间，再次发送相同的请求，验证缓存
                await asyncio.sleep(1) 
                print(f"发送第二次请求 (期望命中缓存): {payload['input']}")
                response_cached = await client.post(request_url, headers=headers, json=payload, timeout=30)
                if response_cached.status_code == 200:
                    print(f"✅ 第二次自动请求发送成功！(期望命中缓存) 状态码: {response_cached.status_code}")
                else:
                    print(f"❌ 第二次自动请求失败！状态码: {response_cached.status_code}, 响应体: {response_cached.text}")
            else:
                print(f"❌ 第一次自动请求失败！状态码: {response.status_code}, 响应体: {response.text}")
    except httpx.RequestError as e:
        print(f"💥 自动请求发送过程中发生网络错误: {e}")
    except Exception as e:
        print(f"🚨 自动请求处理过程中发生未知错误: {e}")
    print("--- 自动请求任务完成 ---")

if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8010, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
    
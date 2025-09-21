import os
import time
import tempfile
import hashlib
from collections import OrderedDict
from typing import Literal

from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel

import asyncio

# Lazy import heavy TTS backend
tts_instance = None
tts_lock = asyncio.Lock()

MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "100"))
in_memory_cache = OrderedDict()

EXAMPLES_DIR = os.path.join(os.getcwd(), "examples")
DEFAULT_PROMPT_AUDIO_PATH = "model_wav/default_prompt.wav"


def load_model_prompt_map():
    model_wav_dir = "model_wav"
    supported_extensions = (".wav", ".m4a")
    if not os.path.isdir(model_wav_dir):
        return {}
    prompt_map = {}
    for filename in os.listdir(model_wav_dir):
        if filename.lower().endswith(supported_extensions):
            model_name = os.path.splitext(filename)[0]
            prompt_map[model_name] = os.path.join(model_wav_dir, filename)
    return prompt_map


class SpeechRequest(BaseModel):
    model: str
    # Optional Voxta-style voice id (examples/<voice>.wav)
    voice: str = ""
    input: str
    language: str = "en"
    emo_control_method: Literal[
        'Same as the voice reference',
        'Use emotion reference audio',
        'Use emotion vector',
        'Use text description to control emotion'
    ] = 'Same as the voice reference'
    emo_weight: float = 0.8
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


app = FastAPI(title="IndexTTS Standalone API")


async def get_tts_instance():
    """Lazily initialize the IndexTTS2 instance. By default require CUDA unless
    environment variable STANDALONE_ALLOW_CPU=1 is set for testing."""
    global tts_instance
    async with tts_lock:
        if tts_instance is not None:
            return tts_instance
        # Import heavy module inside function to avoid startup cost when not used
        from indextts.infer_v2 import IndexTTS2
        import torch

        allow_cpu = os.getenv("STANDALONE_ALLOW_CPU", "0") == "1"
        if not torch.cuda.is_available() and not allow_cpu:
            raise RuntimeError("CUDA not available. Torch without CUDA is unacceptable for standalone API. Set STANDALONE_ALLOW_CPU=1 to override for testing.")

        # Use CUDA if available; otherwise fall back to CPU when allowed
        device = None
        if torch.cuda.is_available():
            device = "cuda:0"

        print(f">> Initializing IndexTTS2 (device={device or 'cpu'}) ...")
        tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_cuda_kernel=False, device=device)
        print(">> IndexTTS2 initialized")
        tts_instance_local = tts
        tts_instance = tts_instance_local
        return tts_instance


def resolve_prompt_path(model_name: str):
    MODEL_PROMPT_MAP = load_model_prompt_map()
    prompt_file_path = MODEL_PROMPT_MAP.get(model_name)
    if not prompt_file_path:
        prompt_file_path = DEFAULT_PROMPT_AUDIO_PATH
    full_prompt_path = os.path.join(os.getcwd(), prompt_file_path)
    if not os.path.exists(full_prompt_path):
        raise FileNotFoundError(full_prompt_path)
    return full_prompt_path


@app.post('/v1/audio/speech')
async def create_speech(speech_request: SpeechRequest):
    try:
        # LRU cache key
        cache_key_content = f"{speech_request.model}:{speech_request.input}"
        cache_key = hashlib.md5(cache_key_content.encode()).hexdigest()
        if cache_key in in_memory_cache:
            in_memory_cache.move_to_end(cache_key)
            return Response(content=in_memory_cache[cache_key], media_type="audio/wav")

        # resolve speaker prompt audio (prefer Voxta `voice` -> examples/<voice>.wav)
        full_prompt_path = None
        if getattr(speech_request, 'voice', None):
            candidate = os.path.join(EXAMPLES_DIR, f"{speech_request.voice}.wav")
            if os.path.exists(candidate):
                full_prompt_path = candidate

        if not full_prompt_path:
            # try examples/<model>.wav (backward compat), then model_wav/default_prompt.wav
            ex_candidate = os.path.join(EXAMPLES_DIR, f"{speech_request.model}.wav")
            if os.path.exists(ex_candidate):
                full_prompt_path = ex_candidate
            else:
                try:
                    full_prompt_path = resolve_prompt_path(speech_request.model)
                except FileNotFoundError:
                    raise HTTPException(status_code=400, detail=f"Reference audio for model '{speech_request.model}' not found.")

        # ensure tts instance
        tts = await get_tts_instance()

        # Map emo control
        emo_choice = speech_request.emo_control_method
        emo_audio_prompt = None
        emo_vector = None
        use_emo_text = False
        emo_text_value = speech_request.emo_text if speech_request.emo_text else (speech_request.emotion or None)

        if emo_choice == 'Same as the voice reference':
            emo_audio_prompt = None
        elif emo_choice == 'Use emotion reference audio':
            emo_audio_prompt = full_prompt_path
        elif emo_choice == 'Use emotion vector':
            emo_vector = [speech_request.vec1, speech_request.vec2, speech_request.vec3, speech_request.vec4,
                          speech_request.vec5, speech_request.vec6, speech_request.vec7, speech_request.vec8]
        elif emo_choice == 'Use text description to control emotion':
            use_emo_text = True

        # Build generation kwargs
        gen_kwargs = {
            "do_sample": bool(speech_request.do_sample),
            "top_p": float(speech_request.top_p),
            "top_k": int(speech_request.top_k) if int(speech_request.top_k) > 0 else None,
            "temperature": float(speech_request.temperature),
            "length_penalty": float(speech_request.length_penalty),
            "num_beams": int(speech_request.num_beams),
            "repetition_penalty": float(speech_request.repetition_penalty),
            "max_mel_tokens": int(speech_request.max_mel_tokens),
            "max_text_tokens_per_segment": int(speech_request.max_text_tokens_per_sentence),
        }

        # Create a temporary output path
        tmpfd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="indextts_")
        os.close(tmpfd)

        # Call the TTS backend
        try:
            tts.infer(spk_audio_prompt=full_prompt_path,
                      text=speech_request.input,
                      output_path=tmp_path,
                      emo_audio_prompt=emo_audio_prompt,
                      emo_alpha=speech_request.emo_weight,
                      emo_vector=emo_vector,
                      use_emo_text=use_emo_text,
                      emo_text=emo_text_value,
                      use_random=speech_request.emo_random,
                      verbose=False,
                      **gen_kwargs)
        except Exception as e:
            # Clean up tmp file
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=f"TTS engine failed: {e}")

        if not os.path.exists(tmp_path):
            raise HTTPException(status_code=500, detail="TTS engine did not produce output file")

        with open(tmp_path, 'rb') as fh:
            data = fh.read()

        # cache
        if len(in_memory_cache) >= MAX_CACHE_SIZE:
            in_memory_cache.popitem(last=False)
        in_memory_cache[cache_key] = data

        # cleanup
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return Response(content=data, media_type='audio/wav')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
def health_check():
    return {
        "status": "ok",
        "model_loaded": tts_instance is not None,
        "cache": {"current_entries": len(in_memory_cache), "max_entries": MAX_CACHE_SIZE}
    }


@app.get('/v1/voices')
def list_voices(request: Request, full: bool = False):
    if not os.path.isdir(EXAMPLES_DIR):
        return {"voices": []} if full else []
    supported_audio_exts = {'.wav', '.mp3', '.m4a', '.ogg'}
    simple_list = []
    rich_list = []
    for fname in sorted(os.listdir(EXAMPLES_DIR)):
        fpath = os.path.join(EXAMPLES_DIR, fname)
        if os.path.isfile(fpath):
            name, ext = os.path.splitext(fname)
            if ext.lower() in supported_audio_exts:
                simple_list.append({"label": name, "parameters": {"voice": name}})
                public_url = os.getenv('PUBLIC_URL')
                if public_url:
                    base = public_url.rstrip('/')
                else:
                    base = str(request.base_url).rstrip('/')
                sample_url = f"{base}/examples/{fname}"
                rich_list.append({"id": name, "name": name, "language": "und", "sample_url": sample_url, "tts": {}})
    return {"voices": rich_list} if full else simple_list


@app.get('/v1/voxta/voices')
def list_voxta_voices():
    res = []
    if not os.path.isdir(EXAMPLES_DIR):
        return res
    supported_audio_exts = {'.wav', '.mp3', '.m4a', '.ogg'}
    for fname in sorted(os.listdir(EXAMPLES_DIR)):
        fpath = os.path.join(EXAMPLES_DIR, fname)
        if os.path.isfile(fpath):
            name, ext = os.path.splitext(fname)
            if ext.lower() in supported_audio_exts:
                entry = {"label": name, "parameters": {"voice": name}}
                res.append(entry)
    return res

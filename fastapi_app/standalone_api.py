import os
import sys
import io
import time
import tempfile
import hashlib
from collections import OrderedDict
from typing import Literal

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
import logging
import traceback
from pydantic import BaseModel

import asyncio
import subprocess
import json
import random
import base64
import urllib.request
import re

# Lazy import heavy TTS backend
tts_instance = None
tts_lock = asyncio.Lock()

MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "100"))
in_memory_cache = OrderedDict()

EXAMPLES_DIR = os.path.join(os.getcwd(), "examples")
DEFAULT_PROMPT_AUDIO_PATH = "model_wav/default_prompt.wav"

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
    # Best-effort: if rewrapping fails, continue; PYTHONUTF8 will still help.
    pass
# Install a SafeWriter that encodes all text to UTF-8 and writes to the original binary buffer.
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
                # swallow write errors
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
    # Replace sys.stdout/stderr with SafeWriter backed by the original buffers
    if hasattr(sys, 'stdout') and hasattr(sys, '__stdout__') and getattr(sys.__stdout__, 'buffer', None):
        sys.stdout = SafeWriter(sys.__stdout__.buffer)
    if hasattr(sys, 'stderr') and hasattr(sys, '__stderr__') and getattr(sys.__stderr__, 'buffer', None):
        sys.stderr = SafeWriter(sys.__stderr__.buffer)
except Exception:
    pass

# Replace builtins.print with a wrapper that writes via the safe writers to avoid encoding errors
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
                    # fallback to writing to sys.stdout
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
# Replace builtins.print with a UTF-8-safe wrapper so third-party modules that call print
# cannot trigger a UnicodeEncodeError on Windows consoles that default to cp1252.
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
                # Prefer writing to the underlying buffer with utf-8 encoding
                if hasattr(file, 'buffer'):
                    file.buffer.write(text.encode('utf-8', errors='replace'))
                    file.flush()
                else:
                    file.write(text)
                    file.flush()
            except Exception:
                # Fallback to real_print to stderr
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


# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("indextts_standalone")

app = FastAPI(title="IndexTTS Standalone API")


@app.on_event('startup')
async def _maybe_preload_tts():
    """Preload the IndexTTS2 model during startup so the server is ready to serve requests.

    Set env INDEXTTS_PRELOAD=0 to skip preloading (useful for low-memory or test environments).
    """
    preload = os.getenv('INDEXTTS_PRELOAD', '1')
    if preload.strip() in ('0', 'false', 'False'):
        print('>> INDEXTTS_PRELOAD disabled; skipping model preload')
        return
    try:
        print('>> Preloading IndexTTS2 model during startup...')
        await get_tts_instance()
        print('>> IndexTTS2 preloaded successfully')
    except Exception as e:
        # Log but don't stop the server from starting; the endpoint will raise the same error on first call
        print(f'>> Failed to preload IndexTTS2 during startup: {e}')


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log full traceback to stderr and return a 500 JSON response without crashing
    tb = traceback.format_exc()
    logger.error("Unhandled exception during request %s: %s", request.url, tb)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


async def get_tts_instance():
    """Lazily initialize the IndexTTS2 instance. By default require CUDA unless
    environment variable STANDALONE_ALLOW_CPU=1 is set for testing."""
    global tts_instance
    async with tts_lock:
        if tts_instance is not None:
            return tts_instance
        # Import heavy module inside function to avoid startup cost when not used
        from indextts.infer_v2 import IndexTTS2
        # Allow tuning of CUDA allocator and reduced-precision to lower VRAM usage.
        # User-facing env vars:
        #  - INDEXTTS_USE_FP16=1 to enable fp16 model weights where supported
        #  - INDEXTTS_CUDA_MEM_FRACTION=<0-1> to cap per-process GPU memory fraction
        #  - TORCH_CUDA_ALLOC_CONF can be supplied externally; we set a reasonable default if not present.
        os.environ.setdefault('TORCH_CUDA_ALLOC_CONF', os.getenv('TORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64'))
        import torch
        from indextts.utils.device import select_device
        # Enforce CUDA-only: do not allow CPU fallback
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This standalone API requires a CUDA-enabled PyTorch. Install CUDA-enabled torch and restart the service.")

        print('>> Selecting GPU device (unified selector)...')
        # Force interactive menu if available for API startup by setting INDEXTTS_INTERACTIVE_SELECT=1 (default already enabled)
        try:
            device, auto_fp16 = select_device(require_cuda=True)
        except Exception as sel_err:
            # Provide clearer guidance and abort initialization
            raise RuntimeError(f"GPU device selection failed: {sel_err}") from sel_err

        # Optionally enable fp16 via explicit env var or due to automatic selection
        use_fp16_env = os.getenv('INDEXTTS_USE_FP16', '0')
        use_fp16 = str(use_fp16_env).strip() in ('1', 'true', 'True') or bool(auto_fp16)
        if use_fp16:
            print('>> INDEXTTS_USE_FP16 enabled: IndexTTS2 will attempt to use fp16 where supported')

        if not device:
            raise RuntimeError("Device selection failed (no CUDA device resolved). Set INDEXTTS_FORCE_DEVICE.")
        print(f">> Initializing IndexTTS2 (device={device}) ...")
        tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=use_fp16, use_cuda_kernel=False, use_deepspeed=False, device=device)
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


# Robust voice discovery helper used by create_speech and debug_resolve
def discover_voice_from_payload(raw_body):
    """Scan a raw JSON payload and return a best-effort voice id string.

    This helper tolerates:
    - `voice` being a string or an object like {"voice": "voice_12"} or {"value":"voice_12"}
    - `parameters` being a stringified JSON
    - display labels with parentheses or extra text (e.g. "Sam (voice_12)")
    - searching all string values in the payload for matches against example filenames
    """
    if not isinstance(raw_body, dict):
        return ''

    # normalize parameters field if it's a string
    params = raw_body.get('parameters') or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            params = {}

    # direct checks
    def _extract_from(val):
        if not val:
            return ''
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, dict):
            for k in ('voice', 'value', 'id', 'name'):
                if k in val and isinstance(val[k], str):
                    return val[k].strip()
        return ''

    candidates_try = []
    # top-level voice
    candidates_try.append(_extract_from(raw_body.get('voice')))
    # parameters.voice
    candidates_try.append(_extract_from(params.get('voice') if isinstance(params, dict) else None))
    # common aliases
    for alias in ('speaker', 'speaker_id', 'spk', 'speaker_voice', 'actor', 'character'):
        candidates_try.append(_extract_from(raw_body.get(alias)))

    # model field may contain voice id
    model_val = raw_body.get('model') or ''
    if isinstance(model_val, str) and model_val.strip() and '{{' not in model_val:
        candidates_try.append(model_val.strip())

    # Flatten all string values in the payload to search for embedded voice ids
    def _collect_strings(o):
        res = []
        if isinstance(o, str):
            res.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                res.extend(_collect_strings(v))
        elif isinstance(o, list):
            for v in o:
                res.extend(_collect_strings(v))
        return res

    all_strings = _collect_strings(raw_body)
    candidates_try.extend(all_strings[:50])

    # Normalize and compare against examples filenames
    def _norm(s):
        if not isinstance(s, str):
            return ''
        s2 = s.strip().lower()
        # extract parenthesized token if present (common label format 'Name (voice_12)')
        m = re.search(r"\(([^)]+)\)", s2)
        if m:
            s2 = m.group(1)
        s2 = s2.replace(' ', '_').replace('-', '_')
        s2 = re.sub(r'[^a-z0-9_]', '', s2)
        return s2

    try:
        example_files = [os.path.splitext(f)[0] for f in os.listdir(EXAMPLES_DIR) if os.path.isfile(os.path.join(EXAMPLES_DIR, f))]
    except Exception:
        example_files = []

    norm_examples = {_norm(e): e for e in example_files}

    for cand in candidates_try:
        if not cand:
            continue
        nc = _norm(cand)
        if not nc:
            continue
        # direct normalized filename match
        if nc in norm_examples:
            return norm_examples[nc]
        # match patterns like voice_12 or digits
        m = re.match(r'voice_?(\d+)$', nc)
        if m:
            v = f"voice_{m.group(1)}"
            if v in norm_examples:
                return norm_examples[v]
            if m.group(1) in norm_examples:
                return norm_examples[m.group(1)]

    # nothing matched
    return ''


@app.post('/v1/audio/speech')
async def create_speech(request: Request, speech_request: SpeechRequest):
    # Single entrypoint that accepts either the structured SpeechRequest or Voxta-style JSON.
    # Returns WAV audio bytes.
    raw_body = None
    try:
        raw_body = await request.json()
    except Exception:
        raw_body = None

    # Resolve selected voice using robust discovery helper
    selected_voice = ''
    try:
        # If SpeechRequest.voice is set, prefer it
        if getattr(speech_request, 'voice', None):
            selected_voice = str(speech_request.voice).strip()
        else:
            selected_voice = discover_voice_from_payload(raw_body) if isinstance(raw_body, dict) else ''
    except Exception:
        selected_voice = ''

    # Extract Voxta parameters override if present
    raw_params = {}
    if isinstance(raw_body, dict):
        try:
            raw_params = raw_body.get('parameters') or {}
        except Exception:
            raw_params = {}

    # Local generation settings (defaults from pydantic model)
    do_sample = bool(speech_request.do_sample)
    temperature = float(speech_request.temperature)
    top_p = float(speech_request.top_p)
    top_k = int(speech_request.top_k)
    num_beams = int(speech_request.num_beams)
    repetition_penalty = float(speech_request.repetition_penalty)
    max_mel_tokens = int(speech_request.max_mel_tokens)
    max_text_tokens_per_sentence = int(speech_request.max_text_tokens_per_sentence)
    emo_weight = float(speech_request.emo_weight)
    emo_text_value = speech_request.emo_text if speech_request.emo_text else (speech_request.emotion or None)
    format_requested = None
    generation_seed = None

    # Apply overrides from raw_params
    if isinstance(raw_params, dict):
        try:
            if raw_params.get('do_sample') is not None:
                do_sample = bool(raw_params.get('do_sample'))
            if raw_params.get('temperature') is not None:
                temperature = float(raw_params.get('temperature'))
            if raw_params.get('top_p') is not None:
                top_p = float(raw_params.get('top_p'))
            if raw_params.get('top_k') is not None:
                top_k = int(raw_params.get('top_k'))
            if raw_params.get('num_beams') is not None:
                num_beams = int(raw_params.get('num_beams'))
            if raw_params.get('repetition_penalty') is not None:
                repetition_penalty = float(raw_params.get('repetition_penalty'))
            if raw_params.get('max_mel_tokens') is not None:
                max_mel_tokens = int(raw_params.get('max_mel_tokens'))
            if raw_params.get('max_text_tokens_per_sentence') is not None:
                max_text_tokens_per_sentence = int(raw_params.get('max_text_tokens_per_sentence'))
            if raw_params.get('emo_weight') is not None:
                emo_weight = float(raw_params.get('emo_weight'))
            if raw_params.get('emo_text') is not None:
                emo_text_value = raw_params.get('emo_text')
            if raw_params.get('format') is not None:
                format_requested = str(raw_params.get('format')).lower()
            if raw_params.get('generation_seed') is not None:
                try:
                    generation_seed = int(raw_params.get('generation_seed'))
                except Exception:
                    generation_seed = None
        except Exception:
            pass

    # Resolve speaker prompt audio (priority: inline/URL/data-uri => examples/<voice>.wav => examples/<model>.wav => model_wav)
    full_prompt_path = None
    temp_downloads = []

    def _is_data_uri(s):
        return isinstance(s, str) and s.strip().startswith('data:')

    def _is_url(s):
        return isinstance(s, str) and re.match(r'^https?://', s.strip())

    def _download_url_to_temp(url):
        try:
            tmpfd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(url.split('?')[0])[1] or '.wav', prefix='indextts_dl_')
            os.close(tmpfd)
            urllib.request.urlretrieve(url, tmp_path)
            temp_downloads.append(tmp_path)
            return tmp_path
        except Exception as e:
            print(f">> Failed to download audio from {url}: {e}")
            return None

    def _decode_data_uri_to_temp(data_uri):
        try:
            header, b64 = data_uri.split(',', 1)
            m = re.search(r'data:(?P<type>[^;]+)?(;base64)?', header)
            ext = '.wav'
            if m and m.group('type'):
                t = m.group('type')
                if '/' in t:
                    ext = '.' + t.split('/')[-1]
            raw = base64.b64decode(b64)
            tmpfd, tmp_path = tempfile.mkstemp(suffix=ext, prefix='indextts_data_')
            with os.fdopen(tmpfd, 'wb') as out:
                out.write(raw)
            temp_downloads.append(tmp_path)
            return tmp_path
        except Exception as e:
            print(f">> Failed to decode data URI audio: {e}")
            return None

    # Check for inline speaker audio provided by Voxta
    spk_audio_src = None
    if isinstance(raw_body, dict):
        spk_audio_src = raw_body.get('spk_audio') or raw_body.get('speaker_audio') or (raw_body.get('parameters') or {}).get('spk_audio')

    if spk_audio_src:
        if _is_data_uri(spk_audio_src):
            p = _decode_data_uri_to_temp(spk_audio_src)
            if p:
                full_prompt_path = p
        elif _is_url(spk_audio_src):
            p = _download_url_to_temp(spk_audio_src)
            if p:
                full_prompt_path = p
        else:
            candidate_local = os.path.join(os.getcwd(), spk_audio_src) if not os.path.isabs(spk_audio_src) else spk_audio_src
            if os.path.exists(candidate_local):
                full_prompt_path = candidate_local

    # If no inline prompt, try selected_voice candidates. Perform tolerant matching against filenames
    if not full_prompt_path and selected_voice:
        # Helper to normalize names for fuzzy compare
        def _norm(s):
            if not isinstance(s, str):
                return ''
            s2 = s.strip().lower()
            s2 = s2.replace(' ', '_').replace('-', '_')
            s2 = re.sub(r'[^a-z0-9_]', '', s2)
            return s2

        wanted = _norm(selected_voice)
        # Collect candidate filenames from examples dir (no extensions)
        try:
            example_files = [os.path.splitext(f)[0] for f in os.listdir(EXAMPLES_DIR) if os.path.isfile(os.path.join(EXAMPLES_DIR, f))]
        except Exception:
            example_files = []

        # Direct exact/normalized matches first
        tried = []
        # try exact filename as provided
        tried.append(selected_voice)
        # common forms
        if selected_voice.isdigit():
            tried.append(f"voice_{selected_voice}")
        if selected_voice.lower().startswith('voice_'):
            tried.append(selected_voice.lower())
            tried.append(selected_voice.lower().replace('voice_', ''))
        # normalized variant
        tried.append(wanted)
        # also try removing a leading 'voice_' prefix
        if wanted.startswith('voice') and wanted.startswith('voice_') is False:
            # nothing
            pass

        # create candidate file paths to check (preserve order)
        candidates = []
        for t in tried:
            if not t:
                continue
            candidates.append(os.path.join(EXAMPLES_DIR, f"{t}.wav"))
            candidates.append(os.path.join(EXAMPLES_DIR, f"voice_{t}.wav"))

        # Fallback: fuzzy match against available example filenames using normalized compare
        if not any(os.path.exists(c) for c in candidates):
            for ef in example_files:
                if _norm(ef) == wanted:
                    candidates.insert(0, os.path.join(EXAMPLES_DIR, f"{ef}.wav"))
                    break

        # Finally check candidates in order
        for candidate in candidates:
            try:
                print(f">> Checking candidate speaker prompt: {candidate}")
            except Exception:
                pass
            if os.path.exists(candidate):
                full_prompt_path = candidate
                break

    # model-based fallback
    if not full_prompt_path:
        ex_candidate = os.path.join(EXAMPLES_DIR, f"{speech_request.model}.wav")
        if os.path.exists(ex_candidate):
            full_prompt_path = ex_candidate
        else:
            try:
                full_prompt_path = resolve_prompt_path(speech_request.model)
            except FileNotFoundError:
                try:
                    print(f">> Failed to resolve prompt for model='{speech_request.model}', selected_voice='{selected_voice}', raw_body_keys={list(raw_body.keys()) if isinstance(raw_body, dict) else raw_body}")
                except Exception:
                    pass
                raise HTTPException(status_code=400, detail=f"Reference audio for model '{speech_request.model}' not found.")

    # Compute prompt checksum and cache key
    prompt_checksum = None
    try:
        if full_prompt_path and os.path.exists(full_prompt_path):
            with open(full_prompt_path, 'rb') as pf:
                prompt_checksum = hashlib.md5(pf.read()).hexdigest()
    except Exception:
        prompt_checksum = None

    cache_key_content = f"{speech_request.model}:{speech_request.input}:{selected_voice}:{prompt_checksum or ''}:ds={int(do_sample)}:t={temperature}:nb={num_beams}:seed={generation_seed or ''}"
    cache_key = hashlib.md5(cache_key_content.encode()).hexdigest()
    if cache_key in in_memory_cache:
        in_memory_cache.move_to_end(cache_key)
        return Response(content=in_memory_cache[cache_key], media_type='audio/wav')

    # Ensure TTS instance
    tts = await get_tts_instance()

    # Map emotion control (allow override from parameters)
    emo_choice = speech_request.emo_control_method
    try:
        if isinstance(raw_params, dict) and raw_params.get('emo_control_method'):
            emo_choice = raw_params.get('emo_control_method')
    except Exception:
        pass

    emo_audio_prompt = None
    emo_vector = None
    use_emo_text = False
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
        'do_sample': do_sample,
        'top_p': top_p,
        'top_k': top_k if int(top_k) > 0 else None,
        'temperature': temperature,
        'length_penalty': float(speech_request.length_penalty),
        'num_beams': num_beams,
        'repetition_penalty': repetition_penalty,
        'max_mel_tokens': max_mel_tokens,
        'max_text_tokens_per_segment': max_text_tokens_per_sentence,
    }

    # Apply generation seed if provided
    if generation_seed is not None:
        try:
            import numpy as _np
            import torch as _torch
            random.seed(generation_seed)
            _np.random.seed(generation_seed)
            _torch.manual_seed(generation_seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(generation_seed)
            print(f">> Applied generation_seed={generation_seed} to random/numpy/torch")
        except Exception as e:
            print(f">> Failed to apply generation_seed: {e}")

    # Deterministic enforcement when requested
    try:
        if not do_sample or float(temperature) == 0.0:
            gen_kwargs['do_sample'] = False
            gen_kwargs['temperature'] = 0.0
            gen_kwargs['top_p'] = None
            gen_kwargs['top_k'] = None
            gen_kwargs['num_beams'] = max(int(num_beams or 1), 3)
            print(f">> Deterministic generation enforced: num_beams={gen_kwargs['num_beams']} temperature=0.0 do_sample=False")
    except Exception:
        pass

    # Prepare temp output
    tmpfd, tmp_path = tempfile.mkstemp(suffix='.wav', prefix='indextts_')
    os.close(tmpfd)

    try:
        try:
            md_info = prompt_checksum if prompt_checksum else 'none'
            print(f">> Selected voice: '{selected_voice}' | resolved prompt: {full_prompt_path} md5={md_info}")
            # print available example files for debugging
            try:
                ex_list = sorted([f for f in os.listdir(EXAMPLES_DIR)]) if os.path.isdir(EXAMPLES_DIR) else []
                print(f">> examples_dir files: {ex_list}")
            except Exception:
                pass
        except Exception:
            print(f">> Request spk_audio_prompt: {full_prompt_path}")
        print(f">> Emotion settings: emo_audio_prompt={emo_audio_prompt}, emo_alpha={emo_weight}, use_emo_text={use_emo_text}, emo_text={emo_text_value}, emo_vector={emo_vector}, use_random={speech_request.emo_random}")

        if format_requested and format_requested != 'wav':
            print(f">> Voxta requested format='{format_requested}', but only 'wav' is supported. Ignoring and returning wav.")

        try:
            tts.infer(spk_audio_prompt=full_prompt_path,
                      text=speech_request.input,
                      output_path=tmp_path,
                      emo_audio_prompt=emo_audio_prompt,
                      emo_alpha=emo_weight,
                      emo_vector=emo_vector,
                      use_emo_text=use_emo_text,
                      emo_text=emo_text_value,
                      use_random=speech_request.emo_random,
                      verbose=True,
                      **gen_kwargs)
        except RuntimeError as rt_err:
            import torch as _torch
            err_msg = str(rt_err)
            if 'CUDA out of memory' in err_msg:
                import os as _os
                if _os.getenv('INDEXTTS_OOM_RETRY', '1') != '1':
                    raise HTTPException(status_code=500, detail='CUDA OOM (retry disabled)')
                print('>> OOM detected during inference. Attempting automatic failover...')
                try:
                    # Identify current device index
                    current_dev = None
                    try:
                        current_dev = tts.device if hasattr(tts, 'device') else None
                    except Exception:
                        pass
                    current_idx = None
                    if current_dev and isinstance(current_dev, str) and current_dev.startswith('cuda:'):
                        try:
                            current_idx = int(current_dev.split(':')[1])
                        except Exception:
                            current_idx = None
                    # Enumerate other GPUs and pick one with most free memory that isn't current
                    alt_device = None
                    free_list = []
                    try:
                        import subprocess as _sub
                        out = _sub.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used','--format=csv,noheader,nounits'], universal_newlines=True, stderr=_sub.DEVNULL)
                        for line in out.strip().splitlines():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                idx = int(parts[0]); total = int(parts[1]); used = int(parts[2]); free = total - used
                                if current_idx is not None and idx == current_idx:
                                    continue
                                free_list.append((idx, free, total))
                    except Exception:
                        pass
                    min_free_req = int(_os.getenv('INDEXTTS_OOM_MIN_FREE_MB', '512'))
                    if free_list:
                        free_list.sort(key=lambda x: -x[1])
                        cand_idx, cand_free, cand_total = free_list[0]
                        # Require at least configured headroom
                        if cand_free > min_free_req and _os.getenv('INDEXTTS_OOM_ALT_GPU', '1') == '1':
                            alt_device = f'cuda:{cand_idx}'
                            print(f">> Failover candidate: {alt_device} free={cand_free}MB total={cand_total}MB")
                    if alt_device is None:
                        print('>> No alternative GPU with sufficient free memory found; trying in-place fp16 + reduced tokens.')
                        # Try in-place strategy: enable fp16 and shrink max_mel_tokens
                        try:
                            reduce_same = float(_os.getenv('INDEXTTS_OOM_REDUCE_FACTOR_SAME', '0.6'))
                            reduce_same = 0.6 if reduce_same <= 0 or reduce_same >= 1 else reduce_same
                            gen_kwargs['max_mel_tokens'] = max(256, int(gen_kwargs['max_mel_tokens'] * reduce_same))
                            print(f">> Retrying on same device with fp16={True} max_mel_tokens={gen_kwargs['max_mel_tokens']}")
                            # Re-init model in fp16 on same device if not already
                            if hasattr(tts, 'use_fp16') and not tts.use_fp16:
                                # Recreate model quickly (lazy simple approach)
                                from indextts.infer_v2 import IndexTTS2 as _IndexTTS2
                                new_tts = _IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=True, use_cuda_kernel=False, use_deepspeed=False, device=current_dev)
                                tts_instance_local = new_tts
                                globals()['tts_instance'] = new_tts
                                tts = new_tts
                            tts.infer(spk_audio_prompt=full_prompt_path,
                                      text=speech_request.input,
                                      output_path=tmp_path,
                                      emo_audio_prompt=emo_audio_prompt,
                                      emo_alpha=emo_weight,
                                      emo_vector=emo_vector,
                                      use_emo_text=use_emo_text,
                                      emo_text=emo_text_value,
                                      use_random=speech_request.emo_random,
                                      verbose=True,
                                      **gen_kwargs)
                        except Exception as inner_e:
                            raise HTTPException(status_code=500, detail=f"CUDA OOM and fallback failed: {inner_e}")
                    else:
                        # Recreate model on alt_device with fp16 enforced and reduced tokens
                        try:
                            from indextts.infer_v2 import IndexTTS2 as _IndexTTS2
                            reduce_alt = float(_os.getenv('INDEXTTS_OOM_REDUCE_FACTOR_ALT', '0.7'))
                            reduce_alt = 0.7 if reduce_alt <= 0 or reduce_alt >= 1 else reduce_alt
                            reduced_tokens = max(256, int(gen_kwargs['max_mel_tokens'] * reduce_alt))
                            print(f">> Reinitializing model on {alt_device} with fp16=1 max_mel_tokens={reduced_tokens} (reduce factor {reduce_alt})")
                            new_tts = _IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=True, use_cuda_kernel=False, use_deepspeed=False, device=alt_device)
                            globals()['tts_instance'] = new_tts
                            tts = new_tts
                            gen_kwargs['max_mel_tokens'] = reduced_tokens
                            tts.infer(spk_audio_prompt=full_prompt_path,
                                      text=speech_request.input,
                                      output_path=tmp_path,
                                      emo_audio_prompt=emo_audio_prompt,
                                      emo_alpha=emo_weight,
                                      emo_vector=emo_vector,
                                      use_emo_text=use_emo_text,
                                      emo_text=emo_text_value,
                                      use_random=speech_request.emo_random,
                                      verbose=True,
                                      **gen_kwargs)
                        except Exception as reinit_err:
                            raise HTTPException(status_code=500, detail=f"CUDA OOM fallback (alt GPU) failed: {reinit_err}")
                except HTTPException:
                    raise
                except Exception as fallback_error:
                    raise HTTPException(status_code=500, detail=f"CUDA OOM and no successful fallback: {fallback_error}")
            else:
                raise

        if not os.path.exists(tmp_path):
            raise HTTPException(status_code=500, detail='TTS engine did not produce output file')

        with open(tmp_path, 'rb') as fh:
            data = fh.read()

        # cache
        if len(in_memory_cache) >= MAX_CACHE_SIZE:
            in_memory_cache.popitem(last=False)
        in_memory_cache[cache_key] = data

        return Response(content=data, media_type='audio/wav')
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error('TTS infer failed. input=%s, tb=%s', speech_request.input, tb)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # cleanup
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        for p in temp_downloads:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


@app.post('/v1/debug/resolve')
async def debug_resolve(request: Request):
    """Debug endpoint: echo how a raw Voxta payload would resolve the selected voice
    and which candidate prompt files exist. This is lightweight and does NOT load the model.
    """
    try:
        try:
            raw_body = await request.json()
        except Exception:
            raw_body = None

        if not isinstance(raw_body, dict):
            return JSONResponse(status_code=400, content={"error": "expected JSON object"})

        # Resolve voice using same heuristics as create_speech (but without SpeechRequest)
        selected_voice = discover_voice_from_payload(raw_body)

        candidates = []
        existing = []
        if selected_voice:
            candidates.append(os.path.join(EXAMPLES_DIR, f"{selected_voice}.wav"))
            try:
                if selected_voice.isdigit():
                    candidates.append(os.path.join(EXAMPLES_DIR, f"voice_{selected_voice}.wav"))
                elif selected_voice.lower().startswith('voice_'):
                    candidates.append(os.path.join(EXAMPLES_DIR, f"{selected_voice.lower().replace('voice_','')}.wav"))
            except Exception:
                pass

        # Also include model-based fallback candidate if provided
        if raw_body.get('model'):
            candidates.append(os.path.join(EXAMPLES_DIR, f"{raw_body.get('model')}.wav"))

        for c in candidates:
            try:
                if os.path.exists(c):
                    existing.append(c)
            except Exception:
                pass

        return {
            "selected_voice": selected_voice,
            "raw_keys": list(raw_body.keys()),
            "candidates_checked": candidates,
            "existing_candidates": existing
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post('/v1/debug/resolve_verbose')
async def debug_resolve_verbose(request: Request):
    """Return detailed trace for how a payload would be resolved to a prompt file.

    Useful to debug mismatches between UI-selected character and repository filenames.
    """
    try:
        try:
            raw_body = await request.json()
        except Exception:
            raw_body = None

        if not isinstance(raw_body, dict):
            return JSONResponse(status_code=400, content={"error": "expected JSON object"})

        trace = {}
        trace['incoming_keys'] = list(raw_body.keys())
        trace['discovered_voice_candidate'] = discover_voice_from_payload(raw_body)
        # show normalized candidates attempted by the matching logic
        try:
            sel = trace['discovered_voice_candidate']
            norm = lambda s: re.sub(r'[^a-z0-9_]', '', s.strip().lower().replace(' ', '_').replace('-', '_')) if s else ''
            trace['normalized'] = norm(sel)
            try:
                trace['examples_files'] = sorted(os.listdir(EXAMPLES_DIR)) if os.path.isdir(EXAMPLES_DIR) else []
            except Exception:
                trace['examples_files'] = []
        except Exception:
            pass

        return trace
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


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
                simple_list.append({"label": name, "voice": name, "parameters": {"voice": name}})
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
                entry = {"label": name, "voice": name, "parameters": {"voice": name}}
                res.append(entry)
    return res


@app.get('/v1/voxta/provider')
def voxta_provider():
    """Return the Voxta provider JSON descriptor from repository file `indextts_voxta_provider.json`.

    This allows Voxta to discover provider metadata dynamically.
    """
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

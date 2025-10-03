FastAPI / Standalone API Helpers (Windows Friendly)
==================================================

This directory contains:

1. A production‑style standalone FastAPI app (`standalone_api.py`) that exposes
	 Voxta‑compatible TTS endpoints backed directly by IndexTTS2 (no Gradio layer).
2. A legacy / combined app (`main.py`) that can still host the original web UI integration.
3. PowerShell helper scripts for quick local setup on Windows.

Cheat Sheet (TL;DR)
-------------------

Startup (inside venv):

```powershell
python -m uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011 --log-level info
```

Critical env vars (set before launch):

| Var | Example | Purpose |
|-----|---------|---------|
| INDEXTTS_CUDA_DEVICE | 0 | Force GPU index |
| INDEXTTS_USE_FP16 | 1 | Enable fp16 (lower VRAM) |
| INDEXTTS_REQUIRED_VRAM_MB | 10000 | Desired free VRAM threshold |
| INDEXTTS_ALLOW_AUTO_FP16 | 1 | Auto fallback to fp16 if low VRAM |
| INDEXTTS_CUDA_MEM_FRACTION | 0.8 | Cap process VRAM usage |
| INDEXTTS_PRELOAD | 1 | Preload model at startup |

Essential endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Liveness check |
| GET | /v1/voxta/voices | List available example voices |
| GET | /v1/voxta/provider | Provider descriptor JSON |
| POST | /v1/debug/resolve | Voice/prompt resolution (lightweight) |
| POST | /v1/debug/resolve_verbose | Detailed resolution trace |
| POST | /v1/audio/speech | Synthesize voice (WAV) |

Minimal synthesis example (deterministic, voice_12):

```powershell
$body = '{"parameters":{"voice":"voice_12","generation_seed":123,"do_sample":false,"temperature":0},"input":"Test line","language":"en"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8011/v1/audio/speech -Body $body -ContentType 'application/json' -OutFile out.wav
```

Check voice mapping quickly:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8011/v1/debug/resolve_verbose -Body '{"character":"Sam (voice_12)","input":"Hi"}' -ContentType 'application/json'
```

Common fixes:

| Symptom | One-liner |
|---------|-----------|
| Port busy | `netstat -aon | findstr :8011` then `taskkill /PID <pid> /F` |
| CUDA not available | Reinstall CUDA-enabled torch wheel (see root README) |
| Voice not cloning | Use `/v1/debug/resolve_verbose` to confirm voice file match |
| OOM / alloc fail | Set `INDEXTTS_USE_FP16=1` and/or lower `max_mel_tokens` |
| Non‑deterministic | `do_sample=false`, `temperature=0`, add `generation_seed` |

All voice prompts come from the existing `examples/voice_*.wav` files only (no extra assets added).

PowerShell Helper Scripts
-------------------------

Run from repository root (or `fastapi_app/`):

* `start_api.ps1` – Create/activate `.venv`, install `fastapi_app/requirements.txt` (if needed), start the API (`uvicorn fastapi_app.main:app`).
* `start_webui.ps1` – Same environment bootstrap, then launches `webui.py` demo.
* `start_all.ps1` – Runs both API and web UI as background jobs.

Example:

```powershell
cd fastapi_app
./start_api.ps1
# or
./start_webui.ps1
```

Standalone TTS API (`standalone_api.py`)
---------------------------------------

Set the environment variable `USE_STANDALONE_TTS=1` (optional – only needed if you have logic elsewhere switching modes) and run:

```powershell
python -m uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011 --log-level info
```

### Already inside an activated venv?

If you've already activated `.venv` (or another virtual environment) you can run directly, without any helper scripts:

Windows PowerShell:

```powershell
python -m uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011 --log-level info
```

Unix / WSL / Linux / macOS (bash/zsh):

```bash
uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011 --log-level info
```

Deterministic test run (disable sampling + fixed seed):

```powershell
$env:INDEXTTS_USE_FP16='1'
python -m uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011 --log-level warning
```

Or (bash):

```bash
INDEXTTS_USE_FP16=1 uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011 --log-level warning
```

Key Endpoints
-------------

* `GET /health` – Liveness check.
* `POST /v1/audio/speech` – Main synthesis endpoint (Voxta style). Returns raw WAV bytes.
* `POST /v1/debug/resolve` – Lightweight voice & prompt resolution (no model inference).
* `POST /v1/debug/resolve_verbose` – Extended trace of how a payload maps to a prompt file.
* `GET /v1/voxta/voices` – Enumerate available example voices.
* `GET /v1/voxta/provider` – Returns the provider descriptor (`indextts_voxta_provider.json`).

Voxta Integration Details
-------------------------

This API is intentionally Voxta-friendly. Below are the integration points and mappings you can rely on when configuring a Voxta TTS provider. A more expanded narrative guide (with tables) is also available at: `docs/VOXTA_INTEGRATION.md`.

### Provider Discovery Flow

1. Voxta hits `GET /v1/voxta/provider` to obtain a JSON descriptor (pulled directly from `indextts_voxta_provider.json`).
2. It then calls `GET /v1/voxta/voices` to list available voice IDs (derived from existing `examples/voice_*.wav` files).
3. During chat, Voxta sends synthesis requests to `POST /v1/audio/speech` with a payload resembling an OpenAI-ish schema. The server performs voice & parameter normalization so that small client differences do not break cloning.

### Field / Parameter Mapping

| Voxta / OpenAI-style Field | Accepted Here | Internal Mapping / Notes |
|----------------------------|---------------|---------------------------|
| `model` | Optional | Ignored if empty; if it matches a voice pattern it can act as a fallback voice id. |
| `voice` | Yes | Primary voice selector (examples/voice_X.wav). |
| `parameters.voice` | Yes | Alternative voice location if top-level `voice` absent. |
| `speaker` / `speaker_id` / `spk` / `actor` / `character` | Yes | All inspected for a voice id; parentheses variant supported (e.g. `Sam (voice_12)`). |
| `input` | Yes | Text to synthesize (`text`). |
| `language` | Yes | Currently informational; English assumed if unspecified. |
| `emotion` | Placeholder | Reserved; primary emotion control is via `emo_*` fields below. |
| `emo_control_method` | Yes | Selects which emotion pathway to use. |
| `emo_text` | Yes | Guides emotion when using text-description mode. |
| `emo_weight` | Yes | Mapped to internal `emo_alpha`. |
| `emo_random` | Yes | Enables internal randomization (reduces deterministic cloning fidelity). |
| `vec1..vec8` | Yes | Emotion vector (when using emotion vector mode). |
| `spk_audio` | Yes (extended) | Accepts: local relative path, HTTP/HTTPS URL, or data URI base64 (downloaded/decoded to a temp wav). Overrides discovered/static prompt file. |
| `generation_seed` (inside `parameters`) | Yes | Seeds `random`, `numpy`, and `torch` for reproducibility. Part of cache key. |
| `do_sample` | Yes | If false, sampling disabled; combined with `temperature=0` enforces deterministic path. |
| `temperature` | Yes | Standard nucleus sampling temperature. `0` + `do_sample=false` => deterministic. |
| `top_p`, `top_k`, `num_beams`, `repetition_penalty` | Yes | Passed through to model generation args. |
| `max_mel_tokens` | Yes | Upper bound for generated mel tokens (duration control). |
| Unsupported / ignored | e.g. `response_format` | Silently ignored (safe forward compatibility). |

### Voice Resolution Heuristics

The selection algorithm gathers candidates from all probable voice-bearing fields, normalizes them (lowercase, underscore, digit extraction), and matches against existing example filenames. Parenthetical IDs (e.g. `Name (voice_07)`) are extracted; digits alone (`"7"`) also map to `voice_07` if the file exists.

### Emotion Modes Recap

1. Same as reference (baseline cloning)
2. Emotion reference audio (future extended usage; supply `emo_audio_prompt` in future releases)
3. Emotion vector (`vec1..vec8`)
4. Text description (`emo_text` + `emo_weight`)

### Determinism & Caching for Voxta

Deterministic settings (`do_sample=false`, `temperature=0`, plus optional `generation_seed`) guarantee stable reuse from the in-memory LRU cache (also keyed by the MD5 of the selected prompt audio and the full normalized parameter set). This helps Voxta avoid regenerating identical lines when re-sending context windows.

### Error Responses

| Scenario | HTTP Code | JSON Structure |
|----------|-----------|----------------|
| Missing or unreadable prompt audio | 400 | `{ "error": "prompt audio not found ..." }` |
| Unsupported `format` request | 400 | `{ "error": "unsupported format=<value>" }` |
| Generic failure / internal exception | 500 | `{ "detail": "Internal server error" }` (trace logged server-side) |

### Recommended Voxta Configuration Hints

* Use the `voice` setting that exactly matches one of: `voice_01` … `voice_12` (or a label that contains `(voice_XX)`).
* For consistent voice identity across sessions, set sampling off during baseline tuning. Introduce sampling later only if expressive variability outweighs cloning fidelity.
* When providing custom reference audio via `spk_audio`, ensure a short (1–8s) clean exemplar; longer clips slow prompt processing without improving timbre match proportionally.

### Known Limitations

* `language` is not yet driving multilingual phonemization (English assumed by default in this build).
* `emotion` top-level field is currently a placeholder; use `emo_text` or vector inputs instead.
* Streaming synthesis (chunked audio) not yet implemented; responses are returned once full WAV bytes are available.

Future enhancements will consider: streaming chunks for lower latency, explicit multilingual front-end selection, and direct support for separate `emo_audio` payloads in Voxta-style requests.

Voxta / Payload Parameters
--------------------------

The `/v1/audio/speech` endpoint accepts a superset of OpenAI / Voxta style JSON. Core fields:

```jsonc
{
	"model": "",                  // optional; can be blank if voice field used
	"voice": "voice_12",           // OR parameters.voice / character / speaker variants
	"input": "Some text to speak",
	"language": "en",
	"parameters": {
		"voice": "voice_12",       // alternative location
		"do_sample": false,
		"temperature": 0,
		"top_p": 0.8,
		"top_k": 30,
		"num_beams": 3,
		"repetition_penalty": 10.0,
		"max_mel_tokens": 1500,
		"generation_seed": 12345
	},
	"emo_control_method": "Use text description to control emotion", // optional
	"emo_text": "You scared me to death! Are you a ghost?",          // optional
	"emo_weight": 0.6,                                                // maps to emo_alpha
	"emo_random": false
}
```

Voice Resolution Logic
----------------------

The server employs a robust discovery helper:

1. Direct fields: `voice`, `parameters.voice`, `speaker`, `character`, `actor`, etc.
2. Model field (if it looks like a voice id and not a template).
3. Nested objects (e.g. `{"voice": {"value": "voice_12"}}`).
4. Stringified JSON inside `parameters`.
5. Any parenthesized id inside labels (e.g. `Sam (voice_12)`).
6. Fuzzy normalization (strip punctuation, unify `_`, digits) against filenames in `examples/`.

You can verify how your client payload is interpreted by calling:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8011/v1/debug/resolve_verbose -Body '{"parameters":{"voice":"voice_12"},"input":"Hi"}' -ContentType 'application/json'
```

Emotion Control
---------------

Supported modes:

* Same as the voice reference (default)
* Use emotion reference audio (`emo_audio_prompt` – supply separately; future extension)
* Use emotion vector (provide an 8‑float vector via `vec1..vec8` – internally mapped)
* Use text description to control emotion (`emo_text` + `emo_weight` / `emo_alpha`)

Determinism & Caching
---------------------

* Set `generation_seed` (or `do_sample=false` / `temperature=0`) for deterministic paths.
* Internal LRU cache keyed on voice + input + parameters (including seed & prompt MD5).

GPU / Memory Selection
----------------------

Environment variables:

* `INDEXTTS_CUDA_DEVICE` – Force a CUDA device index (e.g. `0` or `cuda:1`).
* `INDEXTTS_REQUIRED_VRAM_MB` – Target VRAM threshold for selecting a GPU (default 10000).
* `INDEXTTS_ALLOW_AUTO_FP16=1` – If no GPU has required VRAM, retry with half requirement and enable fp16.
* `INDEXTTS_USE_FP16=1` – Explicitly force fp16.
* `INDEXTTS_CUDA_MEM_FRACTION=0.8` – Set per‑process CUDA memory fraction.
* `INDEXTTS_PRELOAD=0` – Skip model preload on startup.

### GPU Requirements (Summary)

| Tier | VRAM (Approx) | Suggested Mode | Key Env Vars |
|------|---------------|----------------|--------------|
| Minimum | 6–8 GB | fp16, shorter text | `INDEXTTS_USE_FP16=1`, lower `max_mel_tokens` |
| Recommended | 8–12 GB | fp16 standard | `INDEXTTS_USE_FP16=1` (or auto) |
| High | 12–16 GB | fp16/fp32 mix | Adjust `INDEXTTS_REQUIRED_VRAM_MB` |
| Premium | 16 GB+ | Full features | (optionally disable fp16) |

CPU fallback is intentionally disabled; install a CUDA-enabled PyTorch wheel. For the full detailed table, troubleshooting matrix, and rationale see the root **GPU Requirements** section in `README.md`.

Quick device pin:

```powershell
$env:INDEXTTS_CUDA_DEVICE='0'; $env:INDEXTTS_USE_FP16='1'
python -m uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011
```

If you encounter OOM during initialization: enable fp16, reduce `max_mel_tokens`, or set `INDEXTTS_CUDA_MEM_FRACTION` (e.g. `0.8`) to limit allocator growth.

Diagnostics
-----------

Foreground run (see full logs):

```powershell
python -m uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011 --log-level debug
```

If the port is stuck (Windows lingering TIME_WAIT sockets), terminate existing processes bound to 8011:

```powershell
netstat -aon | findstr :8011
taskkill /PID <pid> /F
```

Common Issues
-------------

* Empty `server_8011.log`: The background process likely failed before writing (check foreground run for bind errors or missing CUDA).
* `CUDA not available` exception: Ensure you installed a CUDA‑enabled PyTorch wheel (see project root README section on GPU setup).
* Voice not cloning: Call `/v1/debug/resolve_verbose` with the exact payload your client sends; verify `discovered_voice_candidate` matches a file under `examples/` (e.g. `voice_12.wav`).

Full Environment Bootstrap (Pinned)
----------------------------------

For a fully pinned environment (including `numba`) use the repository helper:

```powershell
./scripts/setup_venv_py311.ps1
# or
./scripts/setup_venv_py311.ps1 -PythonCmd 'C:\\Program Files\\Python311\\python.exe'
```

That script:

1. Locates (or uses provided) Python 3.11
2. Creates `.venv`
3. Installs root `requirements.txt`
4. Writes `requirements-lock.txt`
5. Runs a smoke test

License / Attribution
---------------------
See the repository root `LICENSE` / `LICENSE_ZH.txt` and model license references.

Changelog (Recent Internal Enhancements)
---------------------------------------

* 2025-08-08: Added CORS config for specific localhost origins; improved startup and WS logging.
* 2025-09-**: Added robust voice discovery, deterministic generation seed, `/v1/debug/resolve_verbose` endpoint, improved GPU / fp16 auto‑selection, UTF‑8 safe logging on Windows.

Feedback & Contributions
------------------------
Issues and PRs welcome. For feature requests around additional Voxta compatibility or new emotion control modes, open an issue with sample payloads.

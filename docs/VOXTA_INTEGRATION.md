# Voxta Integration Guide

This document expands on the brief summary in the root `README.md`, detailing how the standalone
FastAPI app (`fastapi_app/standalone_api.py`) interoperates with Voxta.

## 1. Discovery Flow

1. `GET /v1/voxta/provider` – Returns `indextts_voxta_provider.json` (static descriptor).
2. `GET /v1/voxta/voices` – Lists available voice IDs derived from existing `examples/voice_*.wav` files.
3. `POST /v1/audio/speech` – Main synthesis endpoint; accepts OpenAI / Voxta style JSON payloads.
4. (Optional) `POST /v1/debug/resolve` or `/v1/debug/resolve_verbose` – Inspect how a payload maps to a prompt file without running inference.

No new sample audio files are added beyond the original repository voices (e.g. `voice_01.wav` … `voice_12.wav`).

## 2. Field Mapping

| Input Field / Location | Accepted | Internal Handling / Notes |
|------------------------|----------|---------------------------|
| `model` (top-level) | Optional | If matches voice-like pattern, can act as fallback voice id; otherwise ignored for cloning. |
| `voice` (top-level) | Yes | Primary voice selection. |
| `parameters.voice` | Yes | Alternate voice location if top-level `voice` absent. |
| `speaker` / `speaker_id` / `spk` / `actor` / `character` | Yes | All scanned for a voice id; parentheses variant extracted. |
| `input` | Yes | Text to synthesize. |
| `language` | Yes | Currently informational; English assumed. |
| `emotion` | Placeholder | Not primary; use `emo_*` fields. |
| `emo_control_method` | Yes | Selects pathway (same ref / vector / text). |
| `emo_text` | Yes | Guides emotion if text-description mode selected. |
| `emo_weight` | Yes | Maps to internal `emo_alpha`. |
| `emo_random` | Yes | Enables randomization (reduces deterministic fidelity). |
| `vec1..vec8` | Yes | Emotion vector components (when using vector mode). |
| `spk_audio` | Yes | Local path, HTTP(S) URL, or data URI base64; overrides static example prompt for this request. |
| `parameters.generation_seed` | Yes | Seeds `random`, `numpy`, `torch` for reproducibility; part of cache key. |
| `parameters.do_sample` | Yes | If `false` plus `temperature=0` => deterministic path. |
| `parameters.temperature` | Yes | Sampling temperature. |
| `parameters.top_p`, `top_k`, `num_beams`, `repetition_penalty` | Yes | Forwarded to model generation args. |
| `parameters.max_mel_tokens` | Yes | Duration / length control upper bound. |
| Unsupported extras | Ignored | Safe forward compatibility. |

## 3. Voice Resolution Heuristics

Resolution logic:
1. Collect all candidate strings from known fields.
2. Parse stringified `parameters` if necessary.
3. Extract parenthetical tokens (e.g. `Sam (voice_07)` → `voice_07`).
4. Normalize (lowercase, replace spaces & hyphens with `_`, strip non-alphanumerics except `_`).
5. Match against existing example filenames (with and without `voice_` prefix, or bare digits -> `voice_##`).

You can confirm the result using:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8011/v1/debug/resolve_verbose -Body '{"character":"Sam (voice_07)","input":"Hi"}' -ContentType 'application/json'
```

## 4. Emotion Control Modes

| Mode String (`emo_control_method`) | Required Fields | Notes |
|-----------------------------------|-----------------|-------|
| `Same as the voice reference` | none | Baseline cloning; emotion from reference prompt. |
| `Use emotion vector` | vec1..vec8 | Eight floats define emotion blend. |
| `Use text description to control emotion` | emo_text, emo_weight | Text→vector mapping plus scalar weight. |
| `Use emotion reference audio` | (future) emo_audio_prompt | Planned extension. |

`emo_random=true` introduces stochastic variation; keep `false` for strict cloning.

## 5. Determinism & Caching

The in-memory LRU cache key includes:
- Normalized voice id / resolved prompt md5
- Input text
- Core generation parameters (`do_sample`, `temperature`, `top_p`, etc.)
- `generation_seed` (if provided)

Force determinism by setting:
```jsonc
"parameters": {
  "do_sample": false,
  "temperature": 0,
  "generation_seed": 12345
}
```

## 6. `spk_audio` Handling

Accepted forms:
- Relative file path (`examples/voice_05.wav`)
- Absolute path (sanity-checked)
- HTTP/HTTPS URL (downloaded to a temp file; cleaned after request)
- Data URI: `data:audio/wav;base64,<...>` decoded to a temp file

If `spk_audio` is present it overrides the static example voice file for that single inference.

## 7. Error Semantics

| Scenario | Code | JSON |
|----------|------|------|
| Prompt not found | 400 | `{ "error": "prompt audio not found ..." }` |
| Unsupported output format | 400 | `{ "error": "unsupported format=<val>" }` |
| Internal exception | 500 | `{ "detail": "Internal server error" }` (trace logged) |

## 8. Recommended Voxta Settings

| Goal | Suggested Settings |
|------|--------------------|
| Maximum cloning fidelity | `do_sample=false`, `temperature=0`, set a `generation_seed` |
| Lower VRAM usage | Set env `INDEXTTS_USE_FP16=1` before starting server |
| Stable retries | Include `generation_seed` so cache hits re-use audio |
| Force specific GPU | Set `INDEXTTS_CUDA_DEVICE` |

## 9. Known Limitations

- Streaming audio (chunked responses) not yet implemented.
- Language selection currently ignores non-English values.
- Top-level `emotion` field placeholder – rely on vector or text modes.
- No persistent cache across restarts (in-memory only).

## 10. Future Considerations

Planned or potential enhancements:
- Streaming synthesis endpoint.
- Explicit multilingual front-end / phoneme control.
- Separate `emo_audio` upload field in Voxta-style payload.
- External (disk or Redis) cache backend.

## 11. Quick Diagnostic Commands

List voices:
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8011/v1/voxta/voices -Method Get
```

Verbose resolve:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8011/v1/debug/resolve_verbose -Body '{"voice":"voice_12","input":"Test"}' -ContentType 'application/json'
```

Deterministic synthesis:
```powershell
$body = '{"parameters":{"voice":"voice_12","generation_seed":42,"do_sample":false,"temperature":0},"input":"Sample line","language":"en"}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8011/v1/audio/speech -Body $body -ContentType 'application/json' -OutFile sample.wav
```

---
All integration features operate solely on the existing bundled example prompt files; no additional voice assets are committed.

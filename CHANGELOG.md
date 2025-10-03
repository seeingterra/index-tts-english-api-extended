# Changelog

All notable changes to this repository should be documented in this file.

## [Unreleased]
- docs: removed `uv`-specific instructions and Chinese content; README now focuses on Windows (PowerShell) + `venv` + `pip` workflow.
- scripts: added `scripts/setup_venv_py311.ps1` improvements:
  - Detect Microsoft Visual C++ runtime and warn about WinError 126.
  - Added `-InstallCpuTorch` flag to install CPU-only PyTorch/torchaudio wheels.
  - Added `-AutoDetectCuda` flag to detect CUDA via `nvcc` or `nvidia-smi` and install matching GPU wheels (with safe fallback to CPU wheels).
  - Broadened CUDA->torch wheel tag mapping for common CUDA versions (12.1, 12.0, 11.8, 11.7, 11.6) and nearest-compatible selection.
- ci: added `.github/workflows/ci-windows-setup-cpu.yml` to validate the CPU installation path on `windows-latest`.

## [2.0.1] - 2025-10-03
### Added
- Centralized CUDA OOM fallback utility `indextts/utils/oom_fallback.py` with environment-driven retry policy.
- Automatic retry logic for both FastAPI entry points (`standalone_api.py`, `main.py`):
  - Alt-GPU migration with fp16 + token budget reduction.
  - Same-device fp16 upgrade + aggressive token reduction.
- `/v1/health` endpoint exposing last OOM fallback event metadata.
### Changed
- Removed duplicated inline OOM handling blocks; replaced with reusable helper to reduce maintenance overhead.
### Environment Variables
- `INDEXTTS_OOM_RETRY`, `INDEXTTS_OOM_ALT_GPU`, `INDEXTTS_OOM_MIN_FREE_MB`, `INDEXTTS_OOM_REDUCE_FACTOR_ALT`, `INDEXTTS_OOM_REDUCE_FACTOR_SAME`, `INDEXTTS_OOM_VERBOSE` documented and honored.
### Fixed
- Eliminated unhandled OOM crash paths; requests now return structured errors when all fallback strategies fail.


## [2.0.0-voxta-integration] - 2025-10-03
### Added
- Standalone FastAPI enhancements (`fastapi_app/standalone_api.py`): deterministic seed handling, in‑memory LRU caching, advanced voice discovery heuristics, `spk_audio` (path, URL, data URI) ingestion, GPU VRAM‑based device selection with optional fp16 fallback.
- Voxta compatibility layer: endpoints for `/v1/audio/speech`, `/v1/voxta/voices`, provider descriptor, and debug voice resolution endpoints (`/v1/debug/resolve`, `/v1/debug/resolve_verbose`).
- Provider metadata file `indextts_voxta_provider.json` with GPU requirements, docs index, and debug endpoint references.
- Documentation overhaul: root README badges (GPU required, no CPU fallback, docs index), detailed FastAPI usage & cheat sheet, `docs/INDEX.md` (documentation hub), `docs/VOXTA_INTEGRATION.md` (deep integration guide).
### Changed
- Updated PowerShell startup scripts for Windows‑first venv workflow and deterministic behavior.
- Root README now emphasizes CUDA Torch requirement and standalone API path separate from Gradio/WebUI.
### Fixed
- More graceful CUDA device selection reducing allocation errors by checking available VRAM and allowing env overrides.
### Security / Misc
- Excluded newly added unneeded large example audio assets from version control via refined `.gitignore` patterns.



## Notes
- The AutoDetectCuda detection and mapping is best-effort. If a matching GPU wheel isn't available or installation fails, the script automatically falls back to CPU wheels to ensure a working environment.
- This change was implemented on branch `seeingterra/add/vc-check-setup-venv` and is ready for PR to `index-tts/index-tts:main`.

<div align="center">
  <img src='assets/index_icon.png' width="240" alt="IndexTTS2 Logo"/>
  <h1>IndexTTS2</h1>
  <p><strong>Emotionally expressive, duration‑controllable, zero‑shot autoregressive TTS</strong></p>
  <p>
    <a href="docs/INDEX.md" title="Documentation Index">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/Docs-Index-0A84FF?logo=readthedocs&logoColor=white" />
        <source media="(prefers-color-scheme: light)" srcset="https://img.shields.io/badge/Docs-Index-blue?logo=readthedocs" />
        <img src="https://img.shields.io/badge/Docs-Index-blue?logo=readthedocs" alt="Docs Index" />
      </picture>
    </a>
    <a href="docs/VOXTA_INTEGRATION.md" title="Voxta Integration Guide">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/Voxta_Integration-16A34A?logo=fastapi&logoColor=white" />
        <source media="(prefers-color-scheme: light)" srcset="https://img.shields.io/badge/Voxta_Integration-success?logo=fastapi" />
        <img src="https://img.shields.io/badge/Voxta_Integration-success?logo=fastapi" alt="Voxta Integration" />
      </picture>
    </a>
    <a href="#gpu-requirements" title="GPU Requirements (8GB+ Recommended)">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/GPU-CUDA_8GB%2B-7E22CE?logo=nvidia&logoColor=white" />
        <source media="(prefers-color-scheme: light)" srcset="https://img.shields.io/badge/GPU-CUDA_8GB%2B-brightgreen?logo=nvidia" />
        <img src="https://img.shields.io/badge/GPU-CUDA_8GB%2B-brightgreen?logo=nvidia" alt="GPU 8GB+" />
      </picture>
    </a>
    <a href="#gpu-requirements" title="CPU fallback not supported">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/CPU-No__Fallback-DC2626?logo=powershell&logoColor=white" />
        <source media="(prefers-color-scheme: light)" srcset="https://img.shields.io/badge/CPU-No__Fallback-red?logo=powershell" />
        <img src="https://img.shields.io/badge/CPU-No__Fallback-red?logo=powershell" alt="No CPU fallback" />
      </picture>
    </a>
    <a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red?logo=arxiv' alt='ArXiv'/></a>
    <a href='https://huggingface.co/IndexTeam/IndexTTS-2'><img src='https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface' alt='HuggingFace Model'/></a>
    <a href='https://modelscope.cn/models/IndexTeam/IndexTTS-2'><img src='https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope' alt='ModelScope Model'/></a>
    <a href='LICENSE'><img src='https://img.shields.io/badge/License-Bilibili%20Model%20Use-orange' alt='License: Bilibili Model Use'/></a>
    <a href='#overview'><img src='https://img.shields.io/badge/Version-2.0.0-informational' alt='Version 2.0.0'/></a>
    <a href='pyproject.toml'><img src='https://img.shields.io/badge/Python-3.10%2B-blue?logo=python' alt='Python 3.10+'/></a>
    <img src='https://img.shields.io/badge/Status-Production%2FStable-success' alt='Status: Production/Stable'/>
    <a href='#-standalone-fastapi--voxta-compatible-api-quickstart'><img src='https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi' alt='FastAPI'/></a>
    <img src='https://img.shields.io/badge/Framework-PyTorch%20CUDA-ee4c2c?logo=pytorch' alt='Framework: PyTorch CUDA Required'/>
  </p>
  <p>
    <em>Quick links:</em>
    <a href="docs/INDEX.md">Docs Index</a> ·
    <a href="docs/VOXTA_INTEGRATION.md">Voxta Integration</a> ·
    <a href="#-standalone-fastapi--voxta-compatible-api-quickstart">Standalone API</a>
  </p>
</div>

---

## Overview

IndexTTS2 is a next‑generation autoregressive zero‑shot text‑to‑speech system with:

* Precise and optional duration control (token budgeting)
* High‑fidelity emotion transfer disentangled from timbre
* Zero‑shot voice cloning from short prompts
* Deterministic + cached inference options
* A standalone FastAPI backend (no Gradio dependency) with Voxta compatibility

If you only want to try it quickly, jump to the Quickstart below. For deeper integration, see the documentation index.

> NOTE: A CUDA build of PyTorch is required. CPU‑only execution is not supported for IndexTTS2 inference.

## Quickstart (Web UI)

```powershell
git clone https://github.com/seeingterra/index-tts-english-api-extended.git
cd index-tts-english-api-extended
git lfs pull
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[webui]
python webui.py
```

Then open http://127.0.0.1:7860

### Optional: Faster package mirrors

If you need a regional Python package mirror, you can pass the `-i/--index-url` flag to pip:

```powershell
python -m pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
```

## Abstract

Existing autoregressive large-scale text‑to‑speech (TTS) models offer strong naturalness, but token‑by‑token generation makes precise duration control difficult. This limits audio‑visual synchronization for tasks like dubbing.

IndexTTS2 introduces a general, autoregressive‑friendly method for speech duration control.

Two generation modes are supported: (1) constrained token count for explicit duration control and (2) free autoregressive generation preserving prompt prosody.

IndexTTS2 disentangles emotional expression from speaker identity, enabling independent control of timbre and emotion. Zero‑shot, it reconstructs a target timbre while applying a chosen emotional tone.

We incorporate GPT latent representations and a three‑stage training strategy to stabilize highly emotional speech. A soft instruction mechanism (fine‑tuned on descriptive prompts) lowers the barrier to textual emotion control.

Experiments across multiple datasets show state‑of‑the‑art performance in word error rate, speaker similarity, and emotional fidelity. Audio samples: <a href="https://index-tts.github.io/index-tts2.github.io/">IndexTTS2 demo page</a>.
Run the web UI with your Python interpreter (after activating the project venv):

```powershell
python webui.py
```


```powershell
python webui.py -h
```
### Feel IndexTTS2

<div align="center">

**IndexTTS2: The Future of Voice, Now Generating**

[![IndexTTS2 Demo](assets/IndexTTS2-video-pic.png)](https://www.bilibili.com/video/BV136a9zqEk5)

*Click the image to watch the IndexTTS2 introduction video.*

</div>


### Contact

For community support see the repository issues or discussion pages.

### ⚙️ Environment Setup (Windows 11 - recommended)

1. Ensure that you have both [git](https://git-scm.com/downloads) and
  [git-lfs](https://git-lfs.com/) on your system.

The Git-LFS plugin must also be enabled for your user account:

```powershell
git lfs install
```

2. Clone this repository and fetch large files:

```powershell
git clone https://github.com/seeingterra/index-tts-english-api-extended.git && cd index-tts-english-api-extended
git lfs pull
```

3. Create and activate a Python virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

4. Install required dependencies:

Install the project in editable mode so the `indextts` package is importable
and easy to develop against. For optional features (webui, deepspeed), install
the relevant extras explicitly.

```powershell
python -m pip install -e .
# Optional extras:
python -m pip install -e .[webui]
python -m pip install -e .[deepspeed]
```

If you prefer using a mirror for faster downloads, pass the `-i` option to pip,
for example:

```powershell
python -m pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
```

5. Download the required models (HuggingFace or ModelScope):

HuggingFace (requires `huggingface_hub`):

```powershell
python -m pip install huggingface_hub
hf download IndexTeam/IndexTTS-2 --local-dir checkpoints
```

ModelScope:

```powershell
python -m pip install modelscope
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

## GPU Requirements

| Tier | Approx VRAM | Mode / Notes | Suggested Settings |
|------|-------------|-------------|--------------------|
| Minimum | 6–8 GB | fp16, shorter texts, limited beams | `INDEXTTS_USE_FP16=1`, reduce `max_mel_tokens` (e.g. 900) |
| Recommended | 8–12 GB | fp16, standard prompts, default beams | `INDEXTTS_USE_FP16=1` (or auto), default `max_mel_tokens=1500` |
| High | 12–16 GB | fp16/fp32 mix, longer prompts, more beams | Adjust `INDEXTTS_REQUIRED_VRAM_MB` (e.g. 12000) |
| Premium | 16 GB+ | Full precision experiments, concurrent requests | Optionally disable fp16 for quality tests |

CPU fallback is intentionally disabled: a CUDA‑enabled PyTorch build is required.

### Key Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `INDEXTTS_USE_FP16` | Force fp16 load | `INDEXTTS_USE_FP16=1` |
| `INDEXTTS_ALLOW_AUTO_FP16` | Auto fp16 if VRAM insufficient | `INDEXTTS_ALLOW_AUTO_FP16=1` |
| `INDEXTTS_REQUIRED_VRAM_MB` | Target free VRAM threshold | `INDEXTTS_REQUIRED_VRAM_MB=10000` |
| `INDEXTTS_CUDA_DEVICE` | Select GPU index | `INDEXTTS_CUDA_DEVICE=1` |
| `INDEXTTS_CUDA_MEM_FRACTION` | Cap process VRAM usage | `INDEXTTS_CUDA_MEM_FRACTION=0.8` |
| `INDEXTTS_PRELOAD` | Preload on startup (1 default) | `INDEXTTS_PRELOAD=1` |
| `INDEXTTS_FORCE_DEVICE` | Hard override device (index or cuda:X) | `INDEXTTS_FORCE_DEVICE=0` |
| `INDEXTTS_INTERACTIVE_SELECT` | Enable interactive GPU menu (TTY only) | `INDEXTTS_INTERACTIVE_SELECT=1` |
| `INDEXTTS_DEVICE_VERBOSE` | Verbose GPU probing logs | `INDEXTTS_DEVICE_VERBOSE=1` |
| `INDEXTTS_DEVICE_LOG` | Suppress device summary when 0 | `INDEXTTS_DEVICE_LOG=0` |

#### OOM Fallback (Automatic Retry)

When a generation fails with `CUDA out of memory`, the API now attempts a staged fallback:

1. Probe alternative GPUs and, if one has sufficient free memory (> ~512MB headroom), reinitialize the model there in fp16 and retry with a modestly reduced `max_mel_tokens`.
2. If no alternative GPU qualifies, retry in place on the same device with fp16 (if not already) and a more aggressive reduction of `max_mel_tokens`.
3. If all retries fail, the request returns HTTP 500 detailing the fallback phase that failed.

Environment toggles (optional — defaults are conservative and enabled implicitly):

| Variable | Purpose | Default |
|----------|---------|---------|
| `INDEXTTS_OOM_RETRY` | Enable OOM retry logic | `1` |
| `INDEXTTS_OOM_ALT_GPU` | Allow switching to another GPU | `1` |
| `INDEXTTS_OOM_MIN_FREE_MB` | Minimum free MB required to consider an alt GPU | `512` |
| `INDEXTTS_OOM_REDUCE_FACTOR_ALT` | Multiply `max_mel_tokens` by this when moving to another GPU | `0.7` |
| `INDEXTTS_OOM_REDUCE_FACTOR_SAME` | Multiply `max_mel_tokens` when staying on same GPU | `0.6` |
| `INDEXTTS_OOM_VERBOSE` | Extra logging for fallback decisions | `0` |

Note: Current implementation has fixed inline thresholds; these env vars are planned (roadmap) unless already surfaced in code. If absent, behavior follows the description above. Adjust the model prompt length and `max_mel_tokens` proactively on smaller cards to avoid fallback overhead.

### Selecting a Device

The service queries `nvidia-smi` for free memory and picks a suitable device. On interactive TTY runs (WebUI, standalone API, scripts) an **interactive GPU selection menu** is shown by default when multiple GPUs exist (can be disabled with `INDEXTTS_INTERACTIVE_SELECT=0`).

Override methods (precedence order):
1. `INDEXTTS_FORCE_DEVICE` (explicit hard selection, accepts `auto`, integer, or `cuda:X`)
2. `INDEXTTS_CUDA_DEVICE` (soft preference)
3. Interactive menu (if enabled & TTY)
4. Automatic VRAM-based selection / fp16 fallback

Manual example:

```powershell
$env:INDEXTTS_CUDA_DEVICE='0'
$env:INDEXTTS_USE_FP16='1'
python -m uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011
```

### Troubleshooting GPU Issues

| Symptom | Check / Fix |
|---------|-------------|
| `CUDA not available` | Installed CPU-only torch; reinstall CUDA wheel from pytorch.org (matching your driver). |
| OOM during init | Enable fp16 (`INDEXTTS_USE_FP16=1`), lower `INDEXTTS_REQUIRED_VRAM_MB`, reduce `max_mel_tokens`. |
| Fragmentation errors | Set `TORCH_CUDA_ALLOC_CONF=max_split_size_mb:64` (already defaulted) or restart process. |
| Unexpected device picked | Set `INDEXTTS_CUDA_DEVICE` explicitly. |
| Performance slower than expected | Ensure no other heavy processes share the GPU; verify PCIe power settings. |

`nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader` is invoked internally; you can run it manually for diagnostics.

local mirrors in China (choose one mirror from the list below):

```powershell
# Use pip with a mirror; example installing the project in editable mode with a mirror:
python -m pip install -e . --index-url https://mirrors.aliyun.com/pypi/simple/

python -m pip install -e . --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

> [!TIP]
> **Available Extra Features:**
> 
> - `--all-extras`: Automatically adds *every* extra feature listed below. You can
>   remove this flag if you want to customize your installation choices.
> - `--extra webui`: Adds WebUI support (recommended).
> - `--extra deepspeed`: Adds DeepSpeed support (may speed up inference on some
>   systems).

> [!IMPORTANT]
> **Important (Windows):** The DeepSpeed library may be difficult to install for
> some Windows users. You can skip it by removing the `--all-extras` flag. If you
> want any of the other extra features above, you can manually add their specific
> feature flags instead.
> 
> **Important (Linux/Windows):** If you see an error about CUDA during the installation,
> please ensure that you have installed NVIDIA's [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
> version **12.8** (or newer) on your system.

5. Download the required models:

Download via `huggingface-cli`:

```powershell
python -m pip install huggingface_hub[cli]

hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```

Or download via `modelscope`:

```powershell
python -m pip install modelscope

modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

### Windows PowerShell helper scripts

This repository includes small PowerShell helper scripts under `fastapi_app/` that create a `.venv`, install the project's `requirements.txt`, and start the API or Web UI on Windows PowerShell.

- `fastapi_app\start_api.ps1` — create/activate `.venv`, install requirements, and start the FastAPI server (uvicorn).
- `fastapi_app\start_webui.ps1` — create/activate `.venv`, install requirements, and run the `webui.py` demo.
- `fastapi_app\start_all.ps1` — run both scripts as background jobs.

Additionally, there is a helper for creating a Python 3.11 environment that installs the full set of pinned dependencies (including `numba`):

- `scripts\setup_venv_py311.ps1` — finds a Python 3.11 interpreter (or accepts `-PythonCmd`), creates `.venv`, installs `requirements.txt`, writes `requirements-lock.txt`, and runs a brief smoke test.

Usage (PowerShell):

```powershell
# Create a Python 3.11 venv and install everything (if Python 3.11 is available via 'py -3.11')
.\scripts\setup_venv_py311.ps1

# Or pass an explicit python path:
.\scripts\setup_venv_py311.ps1 -PythonCmd 'C:\\Program Files\\Python311\\python.exe'
```

These helpers are convenience wrappers for Windows users; you can also manage the virtual environment manually as described above.


> [!NOTE]
> In addition to the above models, some small models will also be automatically
> downloaded when the project is run for the first time. If your network environment
> has slow access to HuggingFace, it is recommended to set the HF mirror endpoint before running the code:
>
> ```bash
> export HF_ENDPOINT="https://hf-mirror.com"
> ```
> 
> ```bash
> export HF_ENDPOINT="https://hf-mirror.com"
> ```


#### 🖥️ Checking PyTorch GPU Acceleration

If you need to diagnose your environment to see which GPUs are detected,
you can use our included utility to check your system. Run it from the repo root after activating your venv:

```powershell
.\.venv\Scripts\Activate.ps1
python tools/gpu_check.py
```

### Windows: PyTorch import errors (WinError 126)

If you see errors like:

```
IMPORT-ERROR torch: [WinError 126] The specified module could not be found. Error loading "...\\torch_python.dll" or one of its dependencies.
```

This commonly means one of two things: the Microsoft Visual C++ runtime is missing, or the installed PyTorch wheel expects CUDA libraries that are not present on your system. Recommended fixes:

1. Install the Microsoft Visual C++ Redistributable (2015-2022) x64:

  - https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

2. If you don't have CUDA or want a CPU-only installation, install the CPU wheels for PyTorch and torchaudio instead of letting pip choose a GPU/CUDA wheel. Example (PowerShell, Python 3.11):

```powershell
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.*+cpu" "torchaudio==2.8.*+cpu" -f https://download.pytorch.org/whl/torch_stable.html
```

3. If you have a specific CUDA version, use the corresponding PyTorch CUDA wheel. Example for CUDA 12.1 (adjust the +cu121 tag to match the desired CUDA version):

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.8.*+cu121" "torchaudio==2.8.*+cu121" -f https://download.pytorch.org/whl/torch_stable.html
```

4. If you change the PyTorch wheel or the interpreter, recreate the `.venv` (delete and re-run `scripts\setup_venv_py311.ps1` or recreate manually) so all binary wheels match the interpreter ABI.

For the most up-to-date install commands tailored to your OS, CUDA and Python version, consult the official instructions at https://pytorch.org/get-started/locally/.



### 🔥 IndexTTS2 Quickstart

#### 🌐 Web Demo

```powershell
python webui.py
```

Open your browser and visit `http://127.0.0.1:7860` to see the demo.

You can also adjust the settings to enable features such as FP16 inference (lower
VRAM usage), DeepSpeed acceleration, compiled CUDA kernels for speed, etc. All
available options can be seen via the following command:

```powershell
python webui.py -h
```

Have fun!

> [!IMPORTANT]
> It can be very helpful to use **FP16** (half-precision) inference. It is faster
> and uses less VRAM, with a very small quality loss.
> 
> **DeepSpeed** *may* also speed up inference on some systems, but it could also
> make it slower. The performance impact is highly dependent on your specific
> hardware, drivers and operating system. Please try with and without it,
> to discover what works best on your personal system.


#### 📝 Using IndexTTS2 in Python

To run scripts, create and activate a Python virtual environment so the project's package is importable and scripts run with the venv interpreter.

Example (PowerShell) – create and activate a venv, install the project, then run the script:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .

# Run the script (editable install makes the package importable):
python indextts\infer_v2.py
```

Note: older guides referenced the 'uv' environment manager. This repository uses the included Python entrypoints instead (for example `python webui.py` for the Web UI and the `fastapi_app` package for the API). Use a Python virtual environment (`venv`) and pip as shown above.

Here are several examples of how to use IndexTTS2 in your own scripts:

1. Synthesize new speech with a single reference audio file (voice cloning):

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "Translate for me, what is a surprise!"
tts.infer(spk_audio_prompt='examples/voice_01.wav', text=text, output_path="gen.wav", verbose=True)
```

2. Using a separate, emotional reference audio file to condition the speech synthesis:

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "This is an example emotional sentence for synthesis."
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", verbose=True)
```

3. When an emotional reference audio file is specified, you can optionally set
   the `emo_alpha` to adjust how much it affects the output.
   Valid range is `0.0 - 1.0`, and the default value is `1.0` (100%):

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "This is an example emotional sentence for synthesis."
tts.infer(spk_audio_prompt='examples/voice_07.wav', text=text, output_path="gen.wav", emo_audio_prompt="examples/emo_sad.wav", emo_alpha=0.9, verbose=True)
```

4. It's also possible to omit the emotional reference audio and instead provide
   an 8-float list specifying the intensity of each emotion, in the following order:
   `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`.
   You can additionally use the `use_random` parameter to introduce stochasticity
   during inference; the default is `False`, and setting it to `True` enables
   randomness:

> [!NOTE]
> Enabling random sampling will reduce the voice cloning fidelity of the speech
> synthesis.

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "Wow! This example uses a surprised emotion for demonstration."
tts.infer(spk_audio_prompt='examples/voice_10.wav', text=text, output_path="gen.wav", emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0], use_random=False, verbose=True)
```

5. Alternatively, you can enable `use_emo_text` to guide the emotions based on
   your provided `text` script. Your text script will then automatically
   be converted into emotion vectors.
   It's recommended to use `emo_alpha` around 0.6 (or lower) when using the text
   emotion modes, for more natural sounding speech.
   You can introduce randomness with `use_random` (default: `False`;
   `True` enables randomness):

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "Hide! He's coming—he's going to grab us!"
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)
```

6. It's also possible to directly provide a specific text emotion description
   via the `emo_text` parameter. Your emotion text will then automatically be
   converted into emotion vectors. This gives you separate control of the text
   script and the text emotion description:

```python
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
text = "Hide! He's coming—he's going to grab us!"
emo_text = "You scared me to death! Are you a ghost?"
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="gen.wav", emo_alpha=0.6, use_emo_text=True, emo_text=emo_text, use_random=False, verbose=True)
```


### Legacy: IndexTTS1 User Guide

You can also use our previous IndexTTS1 model by importing a different module:

```python
from indextts.infer import IndexTTS
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")
voice = "examples/voice_07.wav"
text = "Hello everyone — I'm trying out the AIGC demo. The results are astonishing."
tts.infer(voice, text, 'gen.wav')
```

For more detailed information, see [README_INDEXTTS_1_5](archive/README_INDEXTTS_1_5.md),
or visit the legacy IndexTTS v1.5 repository at <a href="https://github.com/index-tts/index-tts/tree/v1.5.0">index-tts v1.5</a>.


## Our Releases and Demos

### IndexTTS2: [[Paper]](https://arxiv.org/abs/2506.21619); [[Demo]](https://index-tts.github.io/index-tts2.github.io/); [[HuggingFace]](https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo)

### IndexTTS1: [[Paper]](https://arxiv.org/abs/2502.05512); [[Demo]](https://index-tts.github.io/); [[ModelScope]](https://modelscope.cn/studios/IndexTeam/IndexTTS-Demo); [[HuggingFace]](https://huggingface.co/spaces/IndexTeam/IndexTTS)


## Acknowledgements

1. [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
2. [XTTSv2](https://github.com/coqui-ai/TTS)
3. [BigVGAN](https://github.com/NVIDIA/BigVGAN)
4. [wenet](https://github.com/wenet-e2e/wenet/tree/main)
5. [icefall](https://github.com/k2-fsa/icefall)
6. [maskgct](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
7. [seed-vc](https://github.com/Plachtaa/seed-vc)


## 📚 Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.


IndexTTS2:

```
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```


IndexTTS:

```
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025},
  doi={10.48550/arXiv.2502.05512},
  url={https://arxiv.org/abs/2502.05512}
}
```

## 🔌 Standalone FastAPI / Voxta-Compatible API Quickstart

The repository includes a pure FastAPI implementation (no Gradio dependency) that exposes
Voxta-style TTS endpoints using the core IndexTTS2 model. It reuses the existing example
prompt audios under `examples/` (e.g. `voice_01.wav` … `voice_12.wav`) — no new example
files are required or added beyond the originals.

Voxta short summary: The API supports discovery (`/v1/voxta/provider`, `/v1/voxta/voices`), robust
voice field normalization (`voice`, `parameters.voice`, `speaker`, `character`, labels like `Sam (voice_07)`),
deterministic generation (`generation_seed`, `do_sample=false`, `temperature=0`), emotion control via
`emo_text` / vectors, optional `spk_audio` (file / URL / data URI), and debug endpoints
(`/v1/debug/resolve`, `/v1/debug/resolve_verbose`). For a detailed table and deep-dive see:
[`docs/VOXTA_INTEGRATION.md`](docs/VOXTA_INTEGRATION.md).

See also the consolidated documentation index: [`docs/INDEX.md`](docs/INDEX.md).

### 1. Environment (one time)

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Ensure the checkpoint directory (`checkpoints/`) has the required model weights (see earlier sections).

### 2. Start the standalone API (port 8011)

```powershell
python -m uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011 --log-level info
```

Already inside an activated venv (no extra setup needed)? Just run the server:

PowerShell (Windows):

```powershell
python -m uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011
```

Unix / WSL / macOS:

```bash
uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011
```

Deterministic + fp16 example (Windows):

```powershell
$env:INDEXTTS_USE_FP16='1'
$env:INDEXTTS_PRELOAD='1'
python -m uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011 --log-level warning
```

Deterministic + fp16 example (bash):

```bash
INDEXTTS_USE_FP16=1 INDEXTTS_PRELOAD=1 uvicorn fastapi_app.standalone_api:app --host 127.0.0.1 --port 8011 --log-level warning
```

If you need deterministic runs or lower VRAM usage you can set (before launching):

```powershell
$env:INDEXTTS_USE_FP16='1'        # force half precision
$env:INDEXTTS_CUDA_DEVICE='0'     # choose GPU
$env:INDEXTTS_PRELOAD='1'         # (default) preload model at startup
```

### 3. Health check

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8011/health -Method Get
```

### 4. List available voices

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8011/v1/voxta/voices -Method Get
```

### 5. Debug voice resolution (no synthesis)

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8011/v1/debug/resolve_verbose -Body '{"parameters":{"voice":"voice_12"},"input":"Hi"}' -ContentType 'application/json'
```

### 6. Synthesize speech (returns WAV bytes)

```powershell
$body = '{"parameters":{"voice":"voice_12","generation_seed":12345,"do_sample":false,"temperature":0},"input":"Hide! He\'s coming—he\'s going to grab us!","language":"en","emo_control_method":"Use text description to control emotion","emo_text":"You scared me to death! Are you a ghost?","emo_weight":0.6,"emo_random":false}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8011/v1/audio/speech -Body $body -ContentType 'application/json' -OutFile out.wav
```

### 7. Voice cloning notes

The server automatically maps any of these (case-insensitive / normalized) forms:

* `voice_12`
* `12`
* Labels containing parentheses: `Sam (voice_12)`
* JSON with nested objects: `{"voice":{"value":"voice_12"}}`

If cloning seems wrong:

1. Call `/v1/debug/resolve_verbose` with the *exact* payload your client sends.
2. Confirm `discovered_voice_candidate` matches a filename under `examples/`.
3. Ensure you are not enabling randomness (`do_sample=true` or high `temperature`) when expecting strict cloning.

### 8. Determinism

Use any of these to stabilize output:

* `do_sample=false`
* `temperature=0`
* `generation_seed=<int>` inside `parameters`.

All three factors are incorporated into the cache key to reuse identical generations.

### 9. GPU tuning env vars

| Variable | Purpose |
|----------|---------|
| `INDEXTTS_CUDA_DEVICE` | Force CUDA device (e.g. `1` or `cuda:1`). |
| `INDEXTTS_REQUIRED_VRAM_MB` | Minimum free VRAM target for selecting a GPU (default 10000). |
| `INDEXTTS_ALLOW_AUTO_FP16` | Allow automatic fp16 fallback if full requirement unmet. |
| `INDEXTTS_USE_FP16` | Force fp16 regardless of auto selection. |
| `INDEXTTS_CUDA_MEM_FRACTION` | Cap per-process GPU memory usage fraction. |
| `INDEXTTS_PRELOAD` | Set `0` to skip model preload at startup. |

### 10. Troubleshooting quick list

| Symptom | Check |
|---------|-------|
| Port 8011 not accepting connections | Run in foreground for logs; ensure no stale python process holds the port (`netstat -aon | findstr :8011`). |
| Empty log files in background start | Foreground run will reveal early exception (often port bind or CUDA). |
| Wrong voice | `/v1/debug/resolve_verbose` and confirm example file exists. |
| CUDA OOM or alloc failures | Set `INDEXTTS_USE_FP16=1` or reduce memory via `INDEXTTS_CUDA_MEM_FRACTION`. |

This quickstart intentionally uses only the standard demo voice prompts already present under `examples/` and does **not** introduce any new demo assets.

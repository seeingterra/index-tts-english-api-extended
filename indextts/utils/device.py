"""Unified GPU device selection logic for IndexTTS scripts and services.

This centralizes the automatic CUDA device choice used by:
 - Web UI (webui.py)
 - Legacy inference scripts (infer.py / infer_v2.py)
 - FastAPI services (standalone_api, main)

Environment variables supported:
  INDEXTTS_CUDA_DEVICE         Explicit device index or cuda:X string
  INDEXTTS_REQUIRED_VRAM_MB    Minimum free VRAM target (default 10000)
  INDEXTTS_ALLOW_AUTO_FP16     Allow automatic fp16 fallback if VRAM low (default 1)
  INDEXTTS_USE_FP16            Force fp16 regardless of VRAM logic
  INDEXTTS_CUDA_MEM_FRACTION   Per-process memory fraction cap (0<frac<=1)
  TORCH_CUDA_ALLOC_CONF        Torch allocator tuning (default sets max_split_size_mb:64)

Returns (device:str, use_fp16:bool) where device is a CUDA device string or 'cpu'.
CPU fallback is only returned if CUDA is entirely unavailable; higher layers may forbid it.
"""
from __future__ import annotations

import os, sys, subprocess, re
from typing import Tuple

import torch

def _query_gpu_memory():
    try:
        out = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits'
        ], stderr=subprocess.DEVNULL, universal_newlines=True)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        res = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                idx = int(parts[0]); total = int(parts[1]); used = int(parts[2]); free = total - used
                res.append((idx, free, total))
        return res
    except Exception:
        # Fallback: approximate via torch
        infos = []
        try:
            cnt = torch.cuda.device_count()
            for i in range(cnt):
                prop = torch.cuda.get_device_properties(i)
                total = int(prop.total_memory / (1024*1024))
                infos.append((i, total, total))
        except Exception:
            pass
        return infos

def select_device(require_cuda: bool = True) -> Tuple[str, bool]:
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("CUDA not available. Install a CUDA-enabled PyTorch build.")
        return ("cpu", False)

    env_val = os.getenv('INDEXTTS_CUDA_DEVICE')
    dev_count = torch.cuda.device_count()
    if env_val:
        try:
            if env_val.startswith('cuda:'):
                idx = int(env_val.split(':',1)[1])
            else:
                idx = int(env_val)
            if 0 <= idx < dev_count:
                device = f"cuda:{idx}"
            else:
                print(f">> INDEXTTS_CUDA_DEVICE={env_val} out of range; falling back to auto selection")
                device = None
        except Exception:
            print(f">> Could not parse INDEXTTS_CUDA_DEVICE='{env_val}'; auto-selecting")
            device = None
    else:
        device = None

    required_vram = int(os.getenv('INDEXTTS_REQUIRED_VRAM_MB', '10000'))
    allow_auto_fp16 = os.getenv('INDEXTTS_ALLOW_AUTO_FP16', '1').strip().lower() in ('1','true','yes')
    gpu_infos = _query_gpu_memory()

    auto_fp16 = False
    selected_info = None  # (idx, free, total)
    if device is None:
        if gpu_infos:
            # Try full precision first
            for idx, free, total in sorted(gpu_infos, key=lambda x: -x[1]):
                if free >= required_vram:
                    device = f"cuda:{idx}"; selected_info = (idx, free, total)
                    break
            # Try fp16 fallback
            if device is None and allow_auto_fp16:
                half_req = max(1024, required_vram // 2)
                for idx, free, total in sorted(gpu_infos, key=lambda x: -x[1]):
                    if free >= half_req:
                        print(f">> Not enough VRAM for fp32 (need {required_vram}MB); enabling fp16 on cuda:{idx}")
                        device = f"cuda:{idx}"; auto_fp16 = True; selected_info = (idx, free, total)
                        break
            # Still none: pick GPU with most free
            if device is None:
                best = max(gpu_infos, key=lambda x: x[1])
                print(f">> No GPU meets VRAM target ({required_vram}MB); selecting cuda:{best[0]} (free {best[1]}MB)")
                device = f"cuda:{best[0]}"; selected_info = best
        else:
            # Interactive multi-device fallback
            if dev_count <= 1:
                device = "cuda:0"; selected_info = (0, 0, 0)
            else:
                if sys.stdin and sys.stdin.isatty():
                    try:
                        print(f">> Detected {dev_count} CUDA devices:")
                        for i in range(dev_count):
                            try:
                                name = torch.cuda.get_device_name(i)
                            except Exception:
                                name = f"cuda:{i}"
                            print(f"   [{i}] {name}")
                        sel_raw = input(f"Select device [0-{dev_count-1}] (default 0): ")
                        sel = int(sel_raw) if sel_raw.strip() else 0
                        if sel < 0 or sel >= dev_count:
                            print(f">> Selection {sel} out of range; using 0")
                            sel = 0
                        device = f"cuda:{sel}"; selected_info = (sel, 0, 0)
                    except Exception:
                        device = "cuda:0"; selected_info = (0, 0, 0)
                else:
                    device = "cuda:0"; selected_info = (0, 0, 0)

    # Determine fp16 usage
    force_fp16 = os.getenv('INDEXTTS_USE_FP16', '0').strip().lower() in ('1','true','yes')
    use_fp16 = force_fp16 or auto_fp16
    if use_fp16:
        print(f">> FP16 enabled ({'forced' if force_fp16 else 'auto'}).")

    # Always print a concise selection summary unless explicitly disabled.
    if os.getenv('INDEXTTS_DEVICE_LOG', '1').strip().lower() not in ('0','false','no'):
        if selected_info and selected_info[1] and selected_info[2]:
            idx, free, total = selected_info
            print(f">> Device selection summary: {device} free={free}MB total={total}MB fp16={use_fp16}")
        else:
            # Fallback if memory stats not available
            try:
                name = torch.cuda.get_device_name(int(device.split(':')[-1]))
            except Exception:
                name = device
            print(f">> Device selection summary: {device} ({name}) fp16={use_fp16}")

    # Allocator tuning + optional memory cap
    os.environ.setdefault('TORCH_CUDA_ALLOC_CONF', os.getenv('TORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64'))
    mem_frac = os.getenv('INDEXTTS_CUDA_MEM_FRACTION')
    if mem_frac:
        try:
            frac = float(mem_frac)
            if 0.0 < frac <= 1.0:
                try:
                    torch.cuda.set_per_process_memory_fraction(frac, int(device.split(':')[-1]))
                    print(f">> Set per-process CUDA memory fraction to {frac} on {device}")
                except Exception as e:
                    print(f">> Failed to set per-process memory fraction: {e}")
            else:
                print(f">> INDEXTTS_CUDA_MEM_FRACTION out of range; ignored")
        except Exception:
            print(f">> Invalid INDEXTTS_CUDA_MEM_FRACTION value; ignored")

    return device, use_fp16

__all__ = ["select_device"]

import os
import time
import subprocess
import logging
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger("indextts.oom")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass
class OOMEvent:
    occurred: bool = False
    fallback_strategy: Optional[str] = None  # 'alt_device' | 'same_device' | 'disabled' | 'failed'
    original_device: Optional[str] = None
    alt_device: Optional[str] = None
    reduce_factor: Optional[float] = None
    max_mel_tokens_new: Optional[int] = None
    error: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self):
        return asdict(self)

_LAST_EVENT = OOMEvent()

# Metrics counters (simple in-memory; reset only on process restart)
_OOM_TOTAL = 0
_OOM_ALT_DEVICE = 0
_OOM_SAME_DEVICE = 0
_OOM_DISABLED = 0
_OOM_FAILED = 0

def get_oom_metrics() -> dict:
    return {
        'oom_total': _OOM_TOTAL,
        'oom_alt_device': _OOM_ALT_DEVICE,
        'oom_same_device': _OOM_SAME_DEVICE,
        'oom_disabled': _OOM_DISABLED,
        'oom_failed': _OOM_FAILED,
    }

def render_prometheus_metrics() -> str:
    m = get_oom_metrics()
    lines = [
        '# HELP indextts_oom_total Total CUDA OOM events encountered',
        '# TYPE indextts_oom_total counter',
        f'indextts_oom_total {m["oom_total"]}',
        '# HELP indextts_oom_alt_device Total OOM fallbacks that moved to an alternate GPU',
        '# TYPE indextts_oom_alt_device counter',
        f'indextts_oom_alt_device {m["oom_alt_device"]}',
        '# HELP indextts_oom_same_device Total OOM fallbacks that retried on same GPU',
        '# TYPE indextts_oom_same_device counter',
        f'indextts_oom_same_device {m["oom_same_device"]}',
        '# HELP indextts_oom_disabled OOM events where retry was disabled',
        '# TYPE indextts_oom_disabled counter',
        f'indextts_oom_disabled {m["oom_disabled"]}',
        '# HELP indextts_oom_failed OOM events where all fallback strategies failed',
        '# TYPE indextts_oom_failed counter',
        f'indextts_oom_failed {m["oom_failed"]}',
    ]
    return '\n'.join(lines) + '\n'

def get_last_oom_event() -> OOMEvent:
    return _LAST_EVENT

def _enumerate_alt_devices(skip_idx: Optional[int]) -> list:
    results = []
    try:
        out = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits'
        ], universal_newlines=True, stderr=subprocess.DEVNULL)
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                idx = int(parts[0]); total = int(parts[1]); used = int(parts[2]); free = total - used
                if skip_idx is not None and idx == skip_idx:
                    continue
                results.append((idx, free, total))
    except Exception:
        pass
    return results

def run_tts_with_oom_retry(
    tts,
    call_kwargs: Dict,
    reinit_fn: Callable[[str, bool], object],
    *,
    verbose: bool = False,
) -> Tuple[object, Dict]:
    """
    Attempt tts.infer(**call_kwargs); on CUDA OOM optionally retry on another or same device.

    Parameters:
      tts: current TTS model instance (must provide .infer())
      call_kwargs: arguments to pass into infer
      reinit_fn: function(device:str, fp16:bool) -> new tts instance (used when moving / upgrading precision)
      verbose: extra logging

    Returns:
      (tts_instance (possibly new), possibly modified call_kwargs)

    Environment variables honored:
      INDEXTTS_OOM_RETRY, INDEXTTS_OOM_ALT_GPU, INDEXTTS_OOM_MIN_FREE_MB,
      INDEXTTS_OOM_REDUCE_FACTOR_ALT, INDEXTTS_OOM_REDUCE_FACTOR_SAME, INDEXTTS_OOM_VERBOSE
    """
    global _LAST_EVENT
    global _OOM_TOTAL, _OOM_ALT_DEVICE, _OOM_SAME_DEVICE, _OOM_DISABLED, _OOM_FAILED
    _LAST_EVENT = OOMEvent(occurred=False)

    try:
        tts.infer(**call_kwargs)
        return tts, call_kwargs
    except RuntimeError as e:
        if 'CUDA out of memory' not in str(e):
            raise
        if os.getenv('INDEXTTS_OOM_RETRY', '1') != '1':
            _LAST_EVENT = OOMEvent(occurred=True, fallback_strategy='disabled', original_device=getattr(tts, 'device', None), error='retry disabled', timestamp=time.time())
            _OOM_TOTAL += 1; _OOM_DISABLED += 1
            raise
        # Begin fallback
        original_device = getattr(tts, 'device', None)
        original_idx = None
        if isinstance(original_device, str) and original_device.startswith('cuda:'):
            try:
                original_idx = int(original_device.split(':')[1])
            except Exception:
                original_idx = None

        min_free = int(os.getenv('INDEXTTS_OOM_MIN_FREE_MB', '512'))
        allow_alt = os.getenv('INDEXTTS_OOM_ALT_GPU', '1') == '1'
        reduce_same = float(os.getenv('INDEXTTS_OOM_REDUCE_FACTOR_SAME', '0.6'))
        if reduce_same <= 0 or reduce_same >= 1:
            reduce_same = 0.6
        reduce_alt = float(os.getenv('INDEXTTS_OOM_REDUCE_FACTOR_ALT', '0.7'))
        if reduce_alt <= 0 or reduce_alt >= 1:
            reduce_alt = 0.7
        verbose_env = os.getenv('INDEXTTS_OOM_VERBOSE', '0') == '1'
        if verbose_env:
            verbose = True

        alt_device = None
        if allow_alt:
            alt_list = _enumerate_alt_devices(original_idx)
            if alt_list:
                alt_list.sort(key=lambda x: -x[1])
                cand_idx, cand_free, cand_total = alt_list[0]
                if cand_free > min_free:
                    alt_device = f'cuda:{cand_idx}'
                    if verbose:
                        logger.info(f"OOM fallback considering alt device {alt_device} free={cand_free}MB total={cand_total}MB")

        # Adjust tokens
        max_tokens = call_kwargs.get('max_mel_tokens')
        if alt_device is not None:
            if max_tokens is not None:
                call_kwargs['max_mel_tokens'] = max(256, int(max_tokens * reduce_alt))
            try:
                if verbose:
                    logger.warning(f"Reinitializing model on {alt_device} (reduce_factor={reduce_alt})")
                new_tts = reinit_fn(alt_device, True)
                new_tts.infer(**call_kwargs)
                _LAST_EVENT = OOMEvent(occurred=True, fallback_strategy='alt_device', original_device=original_device,
                                       alt_device=alt_device, reduce_factor=reduce_alt,
                                       max_mel_tokens_new=call_kwargs.get('max_mel_tokens'), timestamp=time.time())
                _OOM_TOTAL += 1; _OOM_ALT_DEVICE += 1
                return new_tts, call_kwargs
            except Exception as alt_err:
                if verbose:
                    logger.error(f"Alt device fallback failed: {alt_err}")
                # Fall through to same-device attempt
        # Same device attempt
        if max_tokens is not None:
            call_kwargs['max_mel_tokens'] = max(256, int(max_tokens * reduce_same))
        try:
            if verbose:
                logger.warning(f"Retrying on same device {original_device} with reduce_factor={reduce_same}")
            # Reinit only if fp16 upgrade needed
            need_fp16_upgrade = hasattr(tts, 'use_fp16') and not getattr(tts, 'use_fp16')
            if need_fp16_upgrade:
                tts = reinit_fn(original_device, True)
            tts.infer(**call_kwargs)
            _LAST_EVENT = OOMEvent(occurred=True, fallback_strategy='same_device', original_device=original_device,
                                   reduce_factor=reduce_same, max_mel_tokens_new=call_kwargs.get('max_mel_tokens'), timestamp=time.time())
            _OOM_TOTAL += 1; _OOM_SAME_DEVICE += 1
            return tts, call_kwargs
        except Exception as same_err:
            _LAST_EVENT = OOMEvent(occurred=True, fallback_strategy='failed', original_device=original_device,
                                   error=str(same_err), timestamp=time.time())
            _OOM_TOTAL += 1; _OOM_FAILED += 1
            raise

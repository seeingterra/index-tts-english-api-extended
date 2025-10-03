import pytest
from indextts.utils.oom_fallback import run_tts_with_oom_retry, get_last_oom_event, get_oom_metrics

class FakeTTS:
    def __init__(self, device='cuda:0', use_fp16=False, fail_mode='first_alt_then_success'):
        self.device = device
        self.use_fp16 = use_fp16
        self._calls = 0
        self.fail_mode = fail_mode
    def infer(self, **kwargs):
        self._calls += 1
        # First call always OOM
        if self._calls == 1:
            raise RuntimeError('CUDA out of memory. X GiB allocated')
        if self.fail_mode == 'always_oom':
            raise RuntimeError('CUDA out of memory again')
        # Succeeds on retry
        return None

def reinit(device: str, fp16: bool):
    return FakeTTS(device=device, use_fp16=fp16, fail_mode='first_alt_then_success')

def test_same_device_retry(monkeypatch):
    monkeypatch.setenv('INDEXTTS_OOM_ALT_GPU', '0')  # Force same device path
    tts = FakeTTS(device='cuda:0', use_fp16=False)
    kwargs = {'max_mel_tokens': 1000}
    new_tts, updated = run_tts_with_oom_retry(tts, kwargs, reinit, verbose=True)
    evt = get_last_oom_event().to_dict()
    metrics = get_oom_metrics()
    assert evt['occurred'] is True
    assert evt['fallback_strategy'] in ('same_device',)  # Should be same_device
    assert metrics['oom_total'] >= 1
    assert updated['max_mel_tokens'] <= 1000


def test_disabled_retry(monkeypatch):
    monkeypatch.setenv('INDEXTTS_OOM_RETRY', '0')
    tts = FakeTTS(device='cuda:0', use_fp16=False)
    with pytest.raises(RuntimeError):
        run_tts_with_oom_retry(tts, {'max_mel_tokens': 800}, reinit, verbose=False)
    evt = get_last_oom_event().to_dict()
    assert evt['fallback_strategy'] == 'disabled'


def test_failed_retry(monkeypatch):
    monkeypatch.setenv('INDEXTTS_OOM_RETRY', '1')
    monkeypatch.setenv('INDEXTTS_OOM_ALT_GPU', '0')
    # Force always failing second call
    tts = FakeTTS(device='cuda:0', use_fp16=False, fail_mode='always_oom')
    with pytest.raises(RuntimeError):
        run_tts_with_oom_retry(tts, {'max_mel_tokens': 900}, reinit, verbose=False)
    evt = get_last_oom_event().to_dict()
    assert evt['fallback_strategy'] == 'failed'

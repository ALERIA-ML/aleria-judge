"""The vision engine's cache key (no GPU: the CUDA queries are stubbed)."""
import json

import pytest

from aleria_judge import vision_tune


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'gemma4'}))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'x.safetensors'}}))
    gpu = {'name': 'NVIDIA H100 PCIe', 'cc': (9, 0)}
    monkeypatch.setattr(vision_tune.torch.cuda, 'get_device_name', lambda device=None: gpu['name'])
    monkeypatch.setattr(vision_tune.torch.cuda, 'get_device_capability', lambda device=None: gpu['cc'])
    monkeypatch.setattr(vision_tune, '_driver', lambda: '580.126.09')
    return tmp_path, gpu


def test_the_key_is_stable_and_changes_with_what_invalidates_an_engine(checkpoint):
    path, gpu = checkpoint
    key, parts = vision_tune.engine_key(path, 5040)
    assert vision_tune.engine_key(path, 5040)[0] == key
    assert parts['max_patches'] == 5040 and parts['backend'] == 'TorchTensorRTAotBackend'
    assert vision_tune.engine_key(path, 10080)[0] != key  # another image budget
    gpu['name'], gpu['cc'] = 'NVIDIA RTX PRO 6000 Blackwell Server Edition', (12, 0)
    other_gpu = vision_tune.engine_key(path, 5040)[0]
    assert other_gpu != key
    (path / 'config.json').write_text(json.dumps({'model_type': 'gemma4', 'changed': True}))
    assert vision_tune.engine_key(path, 5040)[0] != other_gpu  # another checkpoint


def test_tuning_samples_stay_inside_the_patch_range():
    for max_patches in (630, 5040, 10080):
        samples = vision_tune._samples(max_patches, 'cpu')
        assert len(samples) == 2
        for s in samples:
            n = s['hidden'].shape[0]
            assert 9 <= n <= max_patches and s['positions'].shape == (n, 2) and s['hidden'].shape[1] == vision_tune.HIDDEN

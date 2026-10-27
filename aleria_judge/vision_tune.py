"""The vision tower's encoder layers (`lean_core.encoder_layers`) tuned with NVIDIA AITune: one Torch-TensorRT engine
over every patch count of the image budget (9 to soft tokens x 9), built on the first start and kept in the cache
directory (the compose file's per-checkpoint cache volume), then loaded on later starts.

The key covers what makes an engine invalid elsewhere: the GPU and its driver, the TensorRT / Torch-TensorRT / torch /
AITune versions, the patch range and the checkpoint (its config and weight index). A first start builds into a
temporary folder and renames it into place, so an interrupted build is never loaded. Any failure stops the service
with the error: serve without the engine with JUDGE_VISION_ENGINE=eager.

AITune choices: one backend (Torch-TensorRT ahead of time, BF16), so the engine is always the TensorRT one (no silent
eager selection); the eager-baseline comparison disabled; the transformers / diffusers integrations off (they patch
those libraries on import).
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch

NAME = 'vision_encoder'
HIDDEN = 1152  # Gemma 4 vision tower width


def _versions():
    from importlib.metadata import version
    out = {}
    for package in ('aitune', 'torch', 'torch_tensorrt', 'tensorrt_cu13'):
        try:
            out[package] = version(package)
        except Exception:
            out[package] = None
    return out


def _driver():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlSystemGetDriverVersion()
    except Exception:
        return None


def engine_key(model_dir, max_patches, device='cuda'):
    """(key, the parts it hashes) for this checkpoint, patch range, GPU and software."""
    model_dir = Path(model_dir)
    checkpoint = hashlib.sha256()
    for name in ('config.json', 'model.safetensors.index.json'):
        checkpoint.update((model_dir / name).read_bytes())
    parts = {'gpu': torch.cuda.get_device_name(device), 'capability': list(torch.cuda.get_device_capability(device)),
             'driver': _driver(), 'versions': _versions(), 'max_patches': max_patches,
             'checkpoint': checkpoint.hexdigest()[:16], 'backend': 'TorchTensorRTAotBackend'}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16], parts


def _samples(max_patches, device):
    """Two inputs of real grid positions inside the patch range: the shapes AITune records (values do not matter)."""
    out = []
    for n in sorted({max(9, int(0.965 * max_patches)), max(9, max_patches // 2)}):
        side = int(n ** 0.5)
        gx, gy = torch.meshgrid(torch.arange(side, device=device), torch.arange(n // side, device=device), indexing='xy')
        pos = torch.stack((gx, gy), -1).reshape(-1, 2)
        out.append({'hidden': torch.randn(len(pos), HIDDEN, device=device, dtype=torch.bfloat16), 'positions': pos})
    return out


def tuned_layers(layers, model_dir, max_patches, device='cuda', cache_dir=None):
    """`layers` as the AITune-tuned module, loaded from the cache or built and saved there; and what was done."""
    os.environ.setdefault('AITUNE_TRANSFORMERS_INTEGRATION', '0')
    os.environ.setdefault('AITUNE_DIFFUSERS_INTEGRATION', '0')
    import aitune.torch as ait
    from aitune.torch.backend import TorchTensorRTAotBackend
    root = Path(cache_dir or Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'aleria-judge' / 'vision-engine')
    key, parts = engine_key(model_dir, max_patches, device)
    folder = root / key
    info = {'engine': 'aitune', 'key': key, 'path': str(folder), 'max_patches': max_patches}
    t0 = time.perf_counter()
    if (folder / f'{NAME}.ait').exists():
        module = ait.load(ait.Module(layers, name=NAME), f'{NAME}.ait', storage=ait.LocalTorchStorage(base_folder=str(folder)))
        info['load_s'] = round(time.perf_counter() - t0, 1)
        return module, info
    tmp = root / f'{key}.tmp{os.getpid()}'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    strategy = ait.OneBackendStrategy(backend=TorchTensorRTAotBackend()).enable_performance_validation(False)
    n = ait.DynamicDim('patches', min=9, opt=max(9, int(0.965 * max_patches)), max=max_patches)
    module = ait.Module(layers, name=NAME, strategy=strategy, dynamic_shapes={'hidden': (1, n, HIDDEN), 'positions': (1, n, 2)})
    # The backend saves its engine and reloads that file before tuning returns, so this process serves what later
    # starts load from the cache.
    ait.tune(lambda hidden, positions: module(hidden, positions), _samples(max_patches, device), batch_sizes=[1],
             device=device, ignore_failing_modules=False)
    if module.state.name != 'TUNED':
        raise RuntimeError(f'AITune left the vision encoder {module.state.name.lower()}, not tuned')
    ait.save(module, f'{NAME}.ait', storage=ait.LocalTorchStorage(base_folder=str(tmp)))
    (tmp / 'key.json').write_text(json.dumps(parts, indent=1))
    if folder.exists():  # another process finished the same build first
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        tmp.replace(folder)
    info['built_s'] = round(time.perf_counter() - t0, 1)
    return module, info

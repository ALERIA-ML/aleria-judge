"""Service settings, read from JUDGE_* environment variables, and the checkpoint check.

The decision method is fixed per effort in `method.py`. The capacity settings change speed and memory, not the
method; other values than the evaluated run's are allowed, logged at startup and reported in /health.
"""
from dataclasses import dataclass, fields
import hashlib
import os
from pathlib import Path

from .checkpoint import CHECKPOINTS, say
from .images import SOFT_TOKENS
from .method import profile

# The settings of the evaluated Decision Index run; they are also the defaults, except context_length (CONTEXT_LENGTH
# below). cache_fraction 0.7: at 1,120 soft
# tokens per image (the largest budget) the vision encoder needs the headroom (0.8 ran out of memory in the encoder on an H100 80 GB);
# text only (vision false) can use 0.8.
EVALUATED = {'checkpoint': 'bf16', 'default_effort': 'high', 'vision': True, 'image_soft_tokens': SOFT_TOKENS,
             'context_length': 65536, 'branch_slots': 32, 'prefill_tokens': 2048, 'cache_fraction': 0.7}
# Tokens per branch when context_length is 0, by engine. Lean: 131,072, run on an H100 80 GB with every long request of
# the index (up to ~97k tokens with the question list) and synthetic 127k-token states answered. TensorRT-LLM keeps the
# evaluated 65,536: its KV cache must hold a whole branch at startup, and the NVFP4 path has not run longer branches.
CONTEXT_LENGTH = {'lean': 131072, 'trt': 65536}
NVFP4_BACKENDS = ('cutlass', 'cublaslt', 'cuda_core', 'cutedsl', 'marlin')
TRUE, FALSE = ('1', 'true', 'yes'), ('0', 'false', 'no')
# Up to 1.1.0 the service was MORJ and its settings MORJ_*; those names still work, with a warning, when the
# JUDGE_* name is not set.
PREFIX, OLD_PREFIX = 'JUDGE_', 'MORJ_'


def env(name, environ=os.environ):
    """The JUDGE_<name> variable as (variable, value), else MORJ_<name>; value None when neither is set."""
    for variable in (PREFIX + name, OLD_PREFIX + name):
        if environ.get(variable) is not None:
            if variable.startswith(OLD_PREFIX):
                say(f'{variable} is deprecated; use {PREFIX + name}')
            return variable, environ[variable]
    return PREFIX + name, None


@dataclass(frozen=True)
class Settings:
    # Engine: 'lean' (one tree-prefill forward per pass, BF16) or 'trt' (TensorRT-LLM)
    engine: str = 'lean'
    # torch CPU threads: torch's default (one per physical core) stalls the CPU side of image requests on large hosts
    cpu_threads: int = 8
    # Model
    checkpoint: str = 'bf16'  # 'bf16' (google/gemma-4-31B-it) or 'nvfp4' (nvidia/Gemma-4-31B-IT-NVFP4, Blackwell)
    model_path: str = ''      # empty: a checkpoint mounted at /model, else the pinned revision from the HF cache
    tokenizer_path: str = ''  # empty: the overlay of canonical tokenizer files is built under ~/.cache at startup
    model_name: str = ''      # empty: the name the checkpoint was evaluated under (gemma-4-31b-trt / -nvfp4-trt)
    context_length: int = 0  # tokens per branch; 0: the engine's default (CONTEXT_LENGTH)
    # Method: the effort profile of requests that name none. The official kit names none.
    default_effort: str = 'high'
    # TensorRT-LLM capacity
    branch_slots: int = 32
    prefill_tokens: int = 2048
    cache_fraction: float = 0.7
    # Images (the `images` field): loads Gemma 4's vision tower.
    vision: bool = True
    image_soft_tokens: int = SOFT_TOKENS  # Gemma 4 vision tokens per image: 70, 140, 280 (checkpoint default), 560 (default) or 1120
    vision_encoder_batch: int = 2     # images per vision-tower call (bounds encoder memory; same embeddings)
    # TensorRT-LLM's host KV tier, in GB of pinned host RAM: 0 (default, no tier) or at least 0.5. Never TensorRT-LLM's
    # auto sizing: under a container's default RLIMIT_MEMLOCK (64 KB) it builds a one-block tier on which the V2 KV
    # manager can stop making progress (docs/tensorrt-llm-patch.md). This service never used a tier in testing (no suspensions).
    host_kv_cache_gb: float = 0
    # NVFP4 GEMM backends TensorRT-LLM may pick from, comma-separated; empty keeps its default.
    # Blackwell (SM100/SM120) needs nothing here. 'marlin' runs the weights on Hopper/Ada with BF16
    # activations: a functional check only, not the evaluated numerics or speed.
    nvfp4_backends: str = ''
    # Admission and deadlines
    # requests in inference at once; 0: the engine's default, 1 for lean (its passes run one at a time, so more would
    # only keep more K/V in memory; queued requests still decode images and build prompts meanwhile) and 4 for trt
    max_requests: int = 0
    max_queue: int = 16
    queue_timeout: float = 30
    request_timeout: float = 300  # per request, inference only; the longest index requests take ~10-20 s
    drain_timeout: float = 10
    retry_after: int = 1
    max_body_bytes: int = 64 * 1024 * 1024
    # HTTP
    host: str = '127.0.0.1'
    port: int = 30104
    keep_alive: float = 75
    api_key: str = ''
    access_log: bool = True
    trace: bool = True  # POST /v1/trace: answers plus every branch's prompt, cache reuse, timing and readout

    def __post_init__(self):
        if self.context_length == 0:  # resolved here, so every reader sees the effective limit
            object.__setattr__(self, 'context_length', CONTEXT_LENGTH.get(self.engine, CONTEXT_LENGTH['trt']))
        backends = [b for b in self.nvfp4_backends.split(',') if b]
        try:
            profile(self.default_effort)
            effort_ok = True
        except ValueError:
            effort_ok = False
        checks = {
            f'checkpoint must be one of {", ".join(CHECKPOINTS)}': self.checkpoint in CHECKPOINTS,
            "engine must be 'lean' or 'trt'": self.engine in ('trt', 'lean'),
            'cpu_threads must be positive': self.cpu_threads >= 1,
            'the lean engine serves the bf16 checkpoint': self.engine == 'trt' or self.checkpoint == 'bf16',
            'default_effort must be low, medium or high': effort_ok,
            'capacity settings must be positive': min(self.branch_slots, self.prefill_tokens, self.max_body_bytes) >= 1
                                                  and self.max_requests >= 0 and self.max_queue >= 0,
            'timeouts must be positive': min(self.queue_timeout, self.request_timeout, self.keep_alive) > 0,
            'context_length must be at least 2': self.context_length >= 2,
            'host_kv_cache_gb must be 0 (no host KV tier) or at least 0.5': self.host_kv_cache_gb == 0 or self.host_kv_cache_gb >= 0.5,
            'image_soft_tokens must be one of 70, 140, 280, 560, 1120': self.image_soft_tokens in (70, 140, 280, 560, 1120),
            'vision_encoder_batch must be at least 1': self.vision_encoder_batch >= 1,
            'cache_fraction must be in (0, 1)': 0 < self.cache_fraction < 1,
            'drain_timeout >= 0 and retry_after >= 1': self.drain_timeout >= 0 and self.retry_after >= 1,
            f'nvfp4_backends must be a comma-separated subset of {",".join(NVFP4_BACKENDS)}':
                all(b in NVFP4_BACKENDS for b in backends),
        }
        failed = [message for message, ok in checks.items() if not ok]
        if failed:
            raise ValueError('Invalid settings: ' + '; '.join(failed))

    @property
    def active_requests(self):
        """Requests in inference at once: max_requests, or the engine's default when it is 0."""
        return self.max_requests or (1 if self.engine == 'lean' else 4)

    @property
    def spec(self):
        return CHECKPOINTS[self.checkpoint]

    @property
    def served_name(self):
        return self.model_name or self.spec.model_name

    @classmethod
    def from_env(cls, environ=os.environ):
        values = {}
        for field in fields(cls):
            variable, raw = env(field.name.upper(), environ)
            if raw is None:
                continue
            if field.type is bool:
                if raw.strip().lower() not in TRUE + FALSE:
                    raise ValueError(f'{variable} must be a boolean, got {raw!r}')
                values[field.name] = raw.strip().lower() in TRUE
            else:
                values[field.name] = field.type(raw)
        return cls(**values)


def differences(settings):
    """Settings that differ from the evaluated run's, as {name: [value, evaluated value]}; empty when none do. A longer
    context_length is not a difference: it only admits longer requests, and requests within the evaluated length get the
    same answers (checked at 131,072 on the lean engine: 200 index requests, every probability identical)."""
    return {k: [getattr(settings, k), v] for k, v in EVALUATED.items()
            if getattr(settings, k) != v and not (k == 'context_length' and getattr(settings, k) > v)}


def verify_checkpoint(settings):
    """The checkpoint files must match the pinned revision."""
    expected = settings.spec.identity()
    for name, digest in expected['sha256'].items():
        if hashlib.sha256((Path(settings.model_path) / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'{name} differs from the pinned {expected["model"]} checkpoint')

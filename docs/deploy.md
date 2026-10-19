# Running the service

One process owns one GPU. It serves the playground at `/`, `POST /v1/systemone` and
`POST /v1/trace` on a single port, bound to loopback by default.

## Requirements

- Linux with one NVIDIA GPU. `CHECKPOINT=bf16` (default, text and images): at least 80 GB; BF16 weights take
  57.2 GiB; validated on H100 PCIe 80 GB and RTX PRO 6000. `CHECKPOINT=nvfp4` (text only, `VISION=false`): a
  Blackwell GPU (FP4 tensor cores, SM100 or SM120), target RTX PRO 6000 96 GB; the checkpoint is about 33 GB
  with an FP8 KV cache.
- Docker with the NVIDIA Container Toolkit, and a driver that runs NVIDIA's
  `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc29` image (CUDA 13).
- Free disk: 35 GB for the image, 62 GB (bf16) or 33 GB (nvfp4) for the checkpoint, a few GB of caches.
- Host RAM: none pinned by default. A host KV tier (`HOST_KV_CACHE_GB` > 0) pins that much; count it in the host's
  or the pod's memory limit.

## Start

```bash
docker compose up -d
docker compose logs -f
```

That is the whole setup. On the first start the log shows, in order:

1. The image build. The only step is checking that the TensorRT-LLM sources match the ones our
   patches were validated against.
2. `downloading google/gemma-4-31B-it at revision 842da37 …` (or `nvidia/Gemma-4-31B-IT-NVFP4 at
   revision 4135a98 …`): the checkpoint goes into
   `~/.cache/huggingface` in the standard Hugging Face cache layout. An interrupted download resumes,
   and a copy that other tools already put in that cache is reused.
3. `serving … default effort high`: the tokenizer overlay with Google's canonical chat template
   is assembled and checked. Then `trtllm_patches_installed` and `kv_pool` events: 3–4 minutes for
   bf16 on a first start, longer for nvfp4 (GEMM autotuning).
4. `ready. Playground: http://localhost:30104/`.

Keep the server detached (`-d`) and follow its log with `docker compose logs -f`, which you can stop and
start again at any time. The server writes one log line per request to stdout, and an attached
`docker compose up` whose terminal stops reading (a sleeping laptop, a dropped SSH session) blocks those
writes: requests then wait until the terminal reads again. Stop it with `docker compose down`.

The service fails fast instead of starting degraded. It stops when no GPU is visible, the GPU has
no FP4 tensor cores (unless `NVFP4_BACKENDS=marlin`), the checkpoint or tokenizer files differ from
the pinned ones, or the KV cache cannot hold one branch of `CONTEXT_LENGTH` tokens.

## Settings

Every setting has a default. Pass overrides in the shell, for example
`GPU=3 PORT=8080 docker compose up -d`, or put them in a `.env` file next to `docker-compose.yml`.

| Variable | Default | |
| --- | --- | --- |
| `GPU` | `0` | Physical GPU ordinal (as listed by `nvidia-smi`) |
| `PORT` | `30104` | Port of the playground and the API |
| `HOST` | `127.0.0.1` | Listening address; `0.0.0.0` accepts remote clients |
| `API_KEY` | empty | Bearer key for `/v1/*`; set it before using `HOST=0.0.0.0` |
| `MODEL_DIR` | — | Serve a checkpoint already on disk (a directory with `config.json` and the safetensors shards) |
| `HF_CACHE` | `~/.cache/huggingface` | Hugging Face cache that receives the download |
| `CHECKPOINT` | `bf16` | `bf16` (google/gemma-4-31B-it) or `nvfp4` (nvidia/Gemma-4-31B-IT-NVFP4, Blackwell) |
| `DEFAULT_EFFORT` | `high` | Effort profile of requests that name none; the official kit names none |
| `CACHE_DIR` | `~/.cache/aleria-judge` | Compilation caches (one subfolder per checkpoint), the tokenizer overlay and temporary files |
| `HF_TOKEN` | empty | Not needed for these Apache-2.0 checkpoints; a saved `hf auth login` token is also used |
| `VISION` | `true` | Load Gemma 4's vision tower and accept `images`; `false` serves text only (required for `nvfp4`) |
| `IMAGE_SOFT_TOKENS` | `560` | Vision tokens per image: 70, 140, 280 (the checkpoint's own default), 560 (the Decision Index vision board's budget for Gemma 4) or 1120 (native). TensorRT-LLM reads the budget only from the checkpoint's `processor_config.json`, so startup builds a symlinked copy of the checkpoint with that file edited (`checkpoint/image560` in the cache directory; no weights are copied) and refuses to start if the engine's budget differs |
| `CONTEXT_LENGTH` | `0` | Tokens per branch; `0` is the engine's default: 131,072 for `lean`, 65,536 for `trt`. A longer question is refused with 422, never truncated. Lower it to need less KV cache |
| `CACHE_FRACTION` | `0.7` | Share of free GPU memory for the KV cache. 0.7 leaves the vision encoder its headroom at 1,120 tokens per image; text only (`VISION=false`) can use 0.8 |
| `BRANCH_SLOTS` | `32` | Branches TensorRT-LLM runs at once |
| `PREFILL_TOKENS` | `2048` | Prefill chunk size in tokens |
| `NVFP4_BACKENDS` | empty | `nvfp4` only: GEMM backends (comma-separated); empty keeps TensorRT-LLM's Blackwell defaults |
| `HOST_KV_CACHE_GB` | `0` | TensorRT-LLM's host KV tier in GB of pinned host RAM, where the V2 KV manager moves suspended and evicted pages. `0` (default): no tier. This service generates one token per branch and admits only what fits the GPU pool, so nothing was ever suspended and a tier stayed empty under every load tested (sequential, 8 and 32 concurrent clients, 8 x ~46k-token requests). `/metrics` counts TensorRT-LLM suspensions (`judge_kv_suspensions`), and without a tier each one is logged as a `kv_suspended` warning; set it (about the GPU KV quota: ~11 GiB on an H100, ~25 GiB on an RTX PRO 6000) if they ever appear. Values between 0 and 0.5 are refused, and TensorRT-LLM's own auto sizing is never used: under the default container `RLIMIT_MEMLOCK` (64 KB) it makes a one-block tier on which TensorRT-LLM 1.3.0rc29 can stop making progress on long multi-branch requests, and the stall guard then stops the service ([tensorrt-llm-patch.md](tensorrt-llm-patch.md#host-kv-tier)) |

The defaults are the settings of the evaluated run, except `CONTEXT_LENGTH` on the lean engine: the run used 65,536, and 1.1.1 admits up to 131,072 (requests within 65,536 get the same answers; a longer limit is not reported as a difference). `CONTEXT_LENGTH`, `CACHE_FRACTION`, `BRANCH_SLOTS` and
`PREFILL_TOKENS` change speed and memory, not the method; when any setting differs from the evaluated run
(these, `CHECKPOINT`, `VISION`, `IMAGE_SOFT_TOKENS` or `DEFAULT_EFFORT`), startup logs the difference and
`/health` lists it under `differs_from_evaluated`. Use the defaults for Decision Index runs.

Admission limits do not affect answers and can be tuned with `JUDGE_MAX_REQUESTS` (lean 1, trt 4), `JUDGE_MAX_QUEUE`
(16), `JUDGE_QUEUE_TIMEOUT` (30 s) and `JUDGE_REQUEST_TIMEOUT` (300 s); pass them through the `environment`
section of `docker-compose.yml`. A request is answered (or refused with 504) within
`queue_timeout + request_timeout + 10` = 340 s, so give clients a timeout above that (the runbook uses 400 s).

With `MODEL_DIR`, the directory must hold the same revision; startup compares the hashes of its
config, weight index, quantization config and tokenizer with `aleria_judge/provenance/identity-<checkpoint>.json`.

## Use it

On the GPU machine, open <http://localhost:30104/>. From a laptop, tunnel the port and open the
same URL:

```bash
ssh -N -L 30104:127.0.0.1:30104 gpu-server
```

In the playground, pick a preset or edit the state and questions (attach images by dropping or pasting
them), then press **Run on the engine** (Cmd/Ctrl+Enter). Click a branch to see its prompt. **Save trace** downloads
the full record, and **Open a saved trace…** shows it later, even from `playground.html` opened
straight from disk without a server. Two traces recorded with the BF16 checkpoint on an H100 are in
[`traces/`](traces/). The request format, effort selection and status codes are in [api.md](api.md).

The API takes the same JSON:

```bash
curl -fsS http://127.0.0.1:30104/v1/systemone -H 'content-type: application/json' -d '{"state": "The invoice was paid twice.", "questions": {"refund": {"type": "noul", "instructions": "Should we refund the duplicate payment?"}, "team": {"type": "choice", "criteria": {"billing": null, "engineering": null, "sales": null}}}}'
```

`/health` returns 503 until the engine has answered a readiness request, then reports the method and
settings.

## Tests

Offline tests need no GPU:

```bash
pip install -e '.[test]'
```

```bash
python -m pytest
```

With the tokenizer overlay the service builds (or a `google/gemma-4-31B-it` directory, whose
tokenizer files are identical), the prompt tests also check the real token layout:

```bash
JUDGE_TOKENIZER=~/.cache/aleria-judge/bf16/aleria-judge/tokenizer python -m pytest tests/test_prompt.py
```

## Operations

- **Monitoring.** `/metrics` exposes request counts by status, end-to-end and per-stage latency,
  branches, gated questions, prompt and cached tokens, queue depth, KV budget and TensorRT-LLM KV
  suspensions.
  Every request writes one JSON log line with its ID, stage timings and token counts.
- **Overload.** Above 4 running and 16 waiting requests, the service answers 529 with
  `Retry-After` instead of growing queues. Concurrency adds latency rather than throughput,
  because prefill saturates the GPU.
- **Failures.** If native inference fails, the service stops admitting requests, lets in-flight
  ones finish or fail with 503, then exits; compose restarts it with a fresh CUDA context.
- **Shutdown** waits up to `queue_timeout + request_timeout + 10` seconds for admitted requests.
- **Files.** The container runs as root, so files it creates in the cache directories belong to root.

## Functional check without a Blackwell GPU

TensorRT-LLM can run the NVFP4 weights on Hopper or Ada through Marlin kernels with BF16
activations. It checks loading, the FP8 KV cache, prompts and the API before renting a
Blackwell card; its decisions and timings are not the release's:

```bash
CHECKPOINT=nvfp4 NVFP4_BACKENDS=marlin docker compose up -d
```

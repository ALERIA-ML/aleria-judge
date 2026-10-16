<p align="center">
  <a href="https://aleria.ai">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/aleria-logo-dark.png">
      <img alt="Aleria" src="docs/assets/aleria-logo-light.png" width="280">
    </picture>
  </a>
</p>

# Aleria Judge

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Decision Index](https://img.shields.io/badge/Decision_Index_Text_(public)-60.23-2ea44f)](https://huggingface.co/spaces/multimodalart/jev-decision-index)
[![Vision](https://img.shields.io/badge/Decision_Index_Vision_(public)-73.38-2ea44f)](https://huggingface.co/spaces/multimodalart/jev-decision-index)

Zero-shot structured decisions from stock **Gemma 4 31B** on one GPU. Give Aleria Judge a state (a ticket, a document, a conversation, any JSON, up to eight images) and typed questions: pick one of up to 255 options, yes or no, a score on an ordinal scale. It answers each with probabilities.

There is no fine-tuning and no generated text. Judge reads the model's next-token distribution over the option labels, for several rotations of the option order, and combines them. It runs on its own prefill-only engine (TensorRT-LLM remains available) behind the Decision Index `POST /v1/systemone` API and ships a playground that shows what the engine did for any request.

Aleria Judge is built by [Aleria AI](https://aleria.ai). Aleria builds sovereign AI systems that run entirely inside your environment, on-premise or air-gapped, with no external API calls. Systems like that need reliable typed decisions, and we would rather ship them as an open method on a stock model than as opaque weights behind someone else's endpoint: everything here runs on your own GPU.

## Results

Aleria Judge 1.2.1 adds one change to 1.1.0: every prompt first lists all of the request's questions, so each question is answered knowing what else is asked ([release 1.2.1](docs/method.md#release-121)). Every default (BF16, effort `high`, 560 soft tokens per image), one request at a time through the HTTP API, official [Decision Index](https://github.com/apolinario/decision-index) kit (edition 0.3), on H100 PCIe 80 GB with one server per GPU:

| | 1.2.1 | 1.1.0, same GPUs | Latency per request, 1.2.1: median / p80 / mean | 1.1.0 |
| --- | --- | --- | --- | --- |
| Decision Index 0.3, public part | **63.43** | 60.20 | 104 / 387 / 319 ms | 95 / 357 / 279 ms |
| Decision Index Vision 0.3, public part | **74.06** | 73.54 | 245 / 310 / 296 ms | 234 / 300 / 288 ms |

Both text runs answered all 140,178 scored requests and the vision runs all 12,210. 1.2.1 has not been run on an RTX PRO 6000 yet. There, 1.1.0 scored:

| 1.1.0, one RTX PRO 6000 Blackwell Server Edition | Score | Requests | Latency per request: median / p80 / mean |
| --- | --- | --- | --- |
| Decision Index 0.3.2, public part | **60.23** | 140,620 of 140,620 answered | 97 / 393 / 312 ms |
| Decision Index Vision 0.3.2, public part | **73.38** | 12,210 of 12,210 answered | 245 / 299 / 295 ms |

Edition 0.3.2 is the board's label; the kit calls it 0.3 (panel `decision-index-0.3`). The suites were rebuilt with the kit and checked against its pinned rows: text corpus SHA-256 `9c83366c…`, the vision suite's 12,210 rows SHA-256 `f5f1cfb5…`, the same rows per benchmark as the board's vision file. These are our runs, not maintainer-reviewed board entries; the board measures its latency gate (median, mean and p80 at most 1,000 ms on one RTX PRO 6000) itself.

## Quickstart

You need Linux with one NVIDIA GPU with 80 GB or more (BF16; validated on H100 PCIe 80 GB and RTX PRO 6000 96 GB), Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), and about 100 GB of disk for the TensorRT-LLM image and the checkpoint.

```bash
docker compose up -d
docker compose logs -f
```

Keep the server detached and follow its log with `logs -f`, which you can stop and start again at any time. An attached `docker compose up` whose terminal stops reading (a sleeping laptop, a dropped SSH session) blocks the server's log writes, and requests wait until the terminal reads again.

The first start builds the image on NVIDIA's TensorRT-LLM 1.3.0rc29 release image, downloads [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) at a pinned revision into `~/.cache/huggingface`, loads it with Gemma 4's vision tower (a few minutes). When the log says `ready`, open <http://localhost:30104/> for the playground, or call the API:

```bash
curl -fsS http://127.0.0.1:30104/v1/systemone -H 'content-type: application/json' -d '{"state": "The invoice was paid twice.", "questions": {"refund": {"type": "noul", "instructions": "Should we refund the duplicate payment?"}, "team": {"type": "choice", "criteria": {"billing": null, "engineering": null, "sales": null}}}}'
```

Images go in `images` as base64 data URLs (PNG, JPEG, WebP or GIF; up to eight, 5 MB each; remote URLs are never fetched). The body is piped in, as an encoded image is too long for a command-line argument:

```bash
printf '{"state": "A customer sent this photo with a return request.", "images": ["data:image/jpeg;base64,%s"], "questions": {"damaged": {"type": "noul", "instructions": "Is the item visibly damaged?"}, "category": {"type": "choice", "criteria": {"electronics": null, "clothing": null, "furniture": null, "other": null}}}}' "$(base64 < photo.jpg | tr -d '\n')" | curl -fsS http://127.0.0.1:30104/v1/systemone -H 'content-type: application/json' --data-binary @-
```

On a Blackwell GPU, NVIDIA's 4-bit checkpoint (`nvidia/Gemma-4-31B-IT-NVFP4`, 33 GB) runs on the TensorRT-LLM engine, scores about 1 point lower and serves text only:

```bash
ENGINE=trt CHECKPOINT=nvfp4 VISION=false docker compose up -d
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENGINE` | `lean` | `lean`: each pass of a request is one forward over its prompts laid out as a token tree (BF16); `trt`: TensorRT-LLM, as in 1.0.1 (needed for `nvfp4`) |
| `CHECKPOINT` | `bf16` | `bf16` or `nvfp4` (with `ENGINE=trt`) |
| `DEFAULT_EFFORT` | `high` | Effort of requests that name none: `low`, `medium` or `high` |
| `GPU` | `0` | Physical GPU ordinal |
| `PORT` | `30104` | Playground and API port |
| `HOST` | `127.0.0.1` | `0.0.0.0` accepts remote clients; set `API_KEY` too |
| `API_KEY` | empty | Bearer key for `/v1/*`; empty disables authentication |
| `MODEL_DIR` | unset | Serve a checkpoint already on disk instead of downloading it |
| `VISION` | `true` | Accept `images` (BF16 only); `false` serves text only |
| `CPU_THREADS` | `8` | torch CPU threads; torch's default of one per physical core slows image requests on large hosts |
| `HOST_KV_CACHE_GB` | `0` | TensorRT-LLM's host KV tier: `0` (none) or at least 0.5; never left to TensorRT-LLM ([why](docs/tensorrt-llm-patch.md#host-kv-tier)) |

All settings, checks and operations: [docs/deploy.md](docs/deploy.md).

## How it works

Each question becomes a few prompts (*branches*) in the state-first layout of [Mapika/decider](https://github.com/Mapika/decider): the list of all the request's questions, the state, one question, its options with single-token labels, and `Answer: (`. The list and the state are the same in every branch of a request, so the engine computes them once. The engine prefills each branch once and reads the softmax over the label tokens. For a four-option question:

```
           labels (A)(B)(C)(D)
order 1     A   B   C   D       options as given
order 2     C   D   A   B       rotated by n/2, mapped back and averaged with order 1
order 3     B   C   D   A       ┐ only when orders 1 and 2 pick different options
order 4     D   A   B   C       ┘ (rotations by n/4 and 3n/4; about 10% of requests)
```

1. **Two orders.** Averaging the original and the half-rotated order cancels much of the model's position bias.
2. **Nine-digit rating.** Every two-option question also asks, in both orders, how likely option (A) is on a 1–9 scale, and the expected digit becomes a probability. Gemma is near-certain on almost every yes/no answer; the rating spreads that probability, which matters for Brier scores and for ranking by p(yes).
3. **Gated orders.** When the two orders disagree on a question with three or more options, the rotations by n/4 and 3n/4 are scored too (with three options, the one rotation left) and all orders are averaged.
4. **Option-count temperature.** Questions with three or more options soften each order with `T(n) = max(0.05, 6.9 − 0.7 ln n)` before averaging, fitted on training data ([disclosures](docs/method.md#disclosures)).

The default engine lays out all branches of a request as one token tree: the system turn, images and state they share appear once, each branch's own tokens hang off it, and one forward pass computes the tree, each token attending only to its own branch's path. It reads out only the label tokens, encodes each image once per request, and keeps the tree's K/V so a gated second pass computes only the new orders. On the full index on one RTX PRO 6000 that cuts the median latency from 173 to 97 ms (mean 403 to 312 ms) against TensorRT-LLM, and on H100 runs the two engines give the same decision on 99.60% of the questions ([release 1.1.0](docs/method.md#release-110)). Formulas, per-benchmark gains, both engines and how the method was chosen: [docs/method.md](docs/method.md). Order averaging helps fine-tuned models too: it added 0.98 points to Cloudflare's clef on Decision Index 0.2.1 ([method.md](docs/method.md#order-averaging-on-a-fine-tuned-model)).

## Effort

Each request can trade accuracy for latency with a header (`X-Decision-Effort: low`) or a field (`"options": {"effort": "low"}`); clients that send neither, including the Decision Index kit, get `high`.

| `effort` | What each question gets | Branches / question | Decision Index 0.3 |
| --- | --- | --- | --- |
| `low` | one order | 1.0 | 62.45 |
| `medium` | two orders; rating of every two-option question in both orders; temperature on the rest | 3.34 | no full run |
| `high` (default) | all four views above | 3.43 | 63.43 |

The index is the full public part with 1.2.1, BF16 on H100 PCIe 80 GB (`low`: 70 / 228 / 201 ms median / p80 / mean per request; `high`: 104 / 387 / 319 ms); `medium` has no full run. Every response reports the effort that ran in `usage.effort`. Request and response format, status codes and the playground: [docs/api.md](docs/api.md).

## Repository layout

```
aleria_judge/  the service; the method itself is aleria_judge/method.py
tests/         offline tests, no GPU needed
docs/          method, API, deployment, evaluation runbook, TensorRT-LLM patch note
```

| Document | Contents |
| --- | --- |
| [docs/method.md](docs/method.md) | The four views, what each adds, effort profiles, the engine, disclosures and limitations |
| [docs/api.md](docs/api.md) | Endpoints, request and response format, effort, status codes, playground |
| [docs/deploy.md](docs/deploy.md) | Requirements, settings, tests, operations |
| [docs/evaluation.md](docs/evaluation.md) | Reproducing a Decision Index number and submitting a run |
| [docs/tensorrt-llm-patch.md](docs/tensorrt-llm-patch.md) | The TensorRT-LLM issues Judge works around (host KV tier, stall guard, vision encoder memory, image budget) |

## Tests

The offline tests need no GPU:

```bash
uv run --extra test python -m pytest
```

The lean engine's tests in `tests/test_lean.py` also import torch and Triton, which the `test` extra does not install; where they are missing (on macOS, for example), skip that file with `--ignore=tests/test_lean.py`.

With the Gemma tokenizer files (a `google/gemma-4-31B-it` download, or the overlay the service builds in `~/.cache/aleria-judge/bf16/aleria-judge/tokenizer`), the prompt tests also check the real token layout:

```bash
JUDGE_TOKENIZER=/path/to/gemma-4-31B-it uv run --extra test python -m pytest tests/test_prompt.py
```

## Limitations

- Images need the BF16 checkpoint (NVFP4 serves text only); at most eight per request, 562 prompt tokens each at the default 560 soft tokens.
- One process serves one GPU and one checkpoint. The default engine runs one request's pass at a time (about 3 requests/s on the index mix on one RTX PRO 6000); concurrency adds latency, not throughput: add GPUs for throughput.
- The default engine is specific to Gemma 4 31B (dense, BF16). On long documents it gains less (the shared prefill dominates) and, on one 8k-token benchmark (LongReason), trails 1.0.1 by 0.6 points.
- NVFP4 and `ENGINE=trt` depend on a TensorRT-LLM release candidate plus patches; upgrading means revalidating them.
- Wide menus are costly: two orders double the option prefill, and gating can double it again.
- Probabilities are only partly calibrated: a stated 0.9 is not guaranteed to be right 90% of the time on your task. If you act on `confidence` or a probability threshold, tune it on your own labelled data.

## Acknowledgements

The prompt layout comes from [Mapika/decider](https://github.com/Mapika/decider) (vendored under Apache-2.0 in `aleria_judge/vendor/`), and the rating readout and mapping from [simple-jev](https://github.com/featherless-ai/simple-jev) v1. Order averaging and rating are well-known debiasing and calibration ideas; Judge's contribution is their combination, the disagreement gate and the serving engine. Scores come from the [Decision Index](https://github.com/apolinario/decision-index) kit. NVIDIA supported the index runs with $1,000 of credits on [brev.nvidia.com](https://brev.nvidia.com/). Thank you.

## License

Apache-2.0 ([LICENSE](LICENSE)); third-party notices in [NOTICE](NOTICE). Gemma 4 is released by Google under Apache-2.0.

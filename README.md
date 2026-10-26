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
[![Decision Index](https://img.shields.io/badge/Decision_Index_Text_(public)-63.43-2ea44f)](https://huggingface.co/spaces/multimodalart/jev-decision-index)
[![Vision](https://img.shields.io/badge/Decision_Index_Vision_(public)-74.06-2ea44f)](https://huggingface.co/spaces/multimodalart/jev-decision-index)

Zero-shot structured decisions from stock **Gemma 4 31B** on one GPU. Give Aleria Judge a state (a ticket, a document, a conversation, any JSON, up to eight images) and typed questions: pick one of up to 255 options, yes or no, a score on an ordinal scale. It answers each with probabilities.

There is no fine-tuning and no generated text. Judge reads the model's next-token distribution over the option labels, for several rotations of the option order, and combines them. It runs on its own prefill-only engine (TensorRT-LLM remains available) behind the Decision Index `POST /v1/systemone` API and ships a playground that shows what the engine did for any request.

Aleria Judge is built by [Aleria AI](https://aleria.ai). Aleria builds sovereign AI systems that run entirely inside your environment, on-premise or air-gapped, with no external API calls. Systems like that need reliable typed decisions, and we would rather ship them as an open method on a stock model than as opaque weights behind someone else's endpoint: everything here runs on your own GPU.

## Results

Aleria Judge 1.1.2 makes effort `xhigh` the default: every prompt reads the request's input twice, and two-option questions are no longer rated ([`xhigh`](docs/method.md#xhigh-the-input-read-twice)). 1.1.1 added the list of all the request's questions to every prompt ([release 1.1.1](docs/method.md#release-111)). Every run uses every default (BF16, the release's default effort, 560 soft tokens per image), one request at a time through the HTTP API, the official [Decision Index](https://github.com/apolinario/decision-index) kit (edition 0.3) on the public part of each index, and one server per GPU. Each cell gives the score, then latency per request as median / p80 / mean:

| Aleria Judge | GPU | Decision Index 0.3 | Decision Index Vision 0.3 |
| --- | --- | --- | --- |
| 1.1.2 | H100 PCIe 80 GB | **64.73** · 111 / 494 / 369 ms | not run |
| 1.1.1 | H100 PCIe 80 GB | 63.43 · 104 / 387 / 319 ms | **74.06** · 245 / 310 / 296 ms |
| 1.1.1 | RTX PRO 6000 Blackwell Server Edition | 63.46 · 119 / 403 / 320 ms | not run |
| 1.1.0 | H100 PCIe 80 GB | 60.20 · 95 / 357 / 279 ms | 73.54 · 234 / 300 / 288 ms |
| 1.1.0 | RTX PRO 6000 Blackwell Server Edition | 60.23 · 97 / 393 / 312 ms | 73.38 · 245 / 299 / 295 ms |

Every run answered all 140,178 scored text requests and all 12,210 vision requests.

The board labels edition 0.3 as 0.3.2 (panel `decision-index-0.3`). The suites were checked against the kit's pinned rows: the text rows' SHA-256 `b2b56d6f…` (uncompressed), the vision suite's 12,210 rows SHA-256 `f5f1cfb5…`, the same rows per benchmark as the board's vision file. These are our runs, not maintainer-reviewed board entries; the board measures its latency gate (median, mean and p80 at most 1,000 ms on one RTX PRO 6000) itself.

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
| `DEFAULT_EFFORT` | `xhigh` | Effort of requests that name none: `low`, `medium`, `high` or `xhigh` |
| `GPU` | `0` | Physical GPU ordinal |
| `PORT` | `30104` | Playground and API port |
| `HOST` | `127.0.0.1` | `0.0.0.0` accepts remote clients; set `API_KEY` too |
| `API_KEY` | empty | Bearer key for `/v1/*`; empty disables authentication |
| `MODEL_DIR` | unset | Serve a checkpoint already on disk instead of downloading it |
| `VISION` | `true` | Accept `images` (BF16 only); `false` serves text only |
| `CONTEXT_LENGTH` | `0` | Tokens per branch; `0`: 131,072 on `lean`, 65,536 on `trt`. A longer question is refused with 422, never truncated |
| `CPU_THREADS` | `8` | torch CPU threads; torch's default of one per physical core slows image requests on large hosts |
| `HOST_KV_CACHE_GB` | `0` | TensorRT-LLM's host KV tier: `0` (none) or at least 0.5; never left to TensorRT-LLM ([why](docs/tensorrt-llm-patch.md#host-kv-tier)) |

All settings, checks and operations: [docs/deploy.md](docs/deploy.md).

## How it works

Each question becomes a few prompts (*branches*) in the state-first layout of [Mapika/decider](https://github.com/Mapika/decider): the list of all the request's questions, the state, one question, its options with single-token labels, and `Answer: (`. The list and the state are the same in every branch of a request, so the engine computes them once, prefills each branch's own tokens once and reads the softmax over the label tokens. For a four-option question:

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

The default engine lays out all branches of a request as one token tree: the system turn, images, question list and state they share appear once, each branch's own tokens hang off it, and one forward pass computes the tree, each token attending only to its own branch's path. It reads out only the label tokens, encodes each image once per request, and keeps the tree's K/V so a gated second pass computes only the new orders. On the full index on one RTX PRO 6000 that cuts the median latency from 173 to 97 ms (mean 403 to 312 ms) against TensorRT-LLM, and on H100 runs the two engines give the same decision on 99.60% of the questions ([release 1.1.0](docs/method.md#release-110)). Formulas, per-benchmark gains, both engines and how the method was chosen: [docs/method.md](docs/method.md). Order averaging helps fine-tuned models too: it added 0.98 points to Cloudflare's clef on Decision Index 0.2.1 ([method.md](docs/method.md#order-averaging-on-a-fine-tuned-model)).

## Effort

Each request can trade accuracy for latency with a header (`X-Decision-Effort: low`) or a field (`"options": {"effort": "low"}`); clients that send neither, including the Decision Index kit, get `xhigh`.

| `effort` | What each question gets | Branches / question | Decision Index 0.3 |
| --- | --- | --- | --- |
| `low` | one order | 1.0 | 62.45 |
| `medium` | two orders; rating of every two-option question in both orders; temperature on the rest | 3.34 | no full run |
| `high` | all four views above | 3.43 | 63.43 |
| `xhigh` (default) | `high` without the rating, with the request's input written twice in the shared prefix (once when twice would not fit the context) | 2.08 | 64.72 |

The index is the full public part with 1.1.1, BF16 on H100 PCIe 80 GB (`low`: 70 / 228 / 201 ms median / p80 / mean per request; `high`: 104 / 387 / 319 ms); `medium` has no full run. `xhigh` is the same full public part on the same GPU type, with held-out accuracy unchanged against `high`; on a 2,000-request uniform sample of the index it ran at 106 / 452 / 345 ms. Every response reports the effort that ran in `usage.effort`. Request and response format, status codes and the playground: [docs/api.md](docs/api.md).

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
- Every branch carries the whole question list, so requests with many or long questions cost more than in 1.1.0 (ToolRet and BRIGHT about twice the median latency). A branch holds at most 131,072 tokens on the default engine (65,536 with `ENGINE=trt`); a longer question is refused with 422, never truncated.
- Probabilities are only partly calibrated: a stated 0.9 is not guaranteed to be right 90% of the time on your task. If you act on `confidence` or a probability threshold, tune it on your own labelled data.

## Acknowledgements

The prompt layout comes from [Mapika/decider](https://github.com/Mapika/decider) (vendored under Apache-2.0 in `aleria_judge/vendor/`), the question list from [wcole3/jev-brain](https://github.com/wcole3/jev-brain), and the rating readout and mapping from [simple-jev](https://github.com/featherless-ai/simple-jev) v1. Order averaging and rating are well-known debiasing and calibration ideas; Judge's contribution is their combination, the disagreement gate and the serving engine. Scores come from the [Decision Index](https://github.com/apolinario/decision-index) kit. NVIDIA supported the index runs with $1,000 of credits on [brev.nvidia.com](https://brev.nvidia.com/). Thank you.

## License

Apache-2.0 ([LICENSE](LICENSE)); third-party notices in [NOTICE](NOTICE). Gemma 4 is released by Google under Apache-2.0.

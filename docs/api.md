# API

Aleria Judge speaks the `POST /v1/systemone` decision API of the [Decision Index](https://github.com/apolinario/decision-index):
one request carries a `state` (any text or JSON) and one or more typed `questions`, and every question gets
an answer with probabilities. The kit's `http` engine works unchanged.

## Endpoints

| Endpoint | |
| --- | --- |
| `POST /v1/systemone` | Decisions: `state` plus typed `questions` |
| `POST /v1/trace` | The same input; the same answers plus the engine's record (see [Playground](#playground)). Off with `JUDGE_TRACE=false` |
| `GET /` | The playground |
| `GET /health` | 503 while starting (until the engine has answered its own readiness request) or draining; then the method, checkpoint, modalities, default effort, settings and `differs_from_evaluated` (settings that differ from the evaluated run; empty by default) |
| `GET /live` | Liveness |
| `GET /metrics` | Prometheus metrics (`judge_*`), including TensorRT-LLM KV suspensions (`judge_kv_suspensions`, expected 0) |
| `GET /v1/models` | The served model in vLLM's OpenAI list format: `id`, `root` (the checkpoint) and `max_model_len` (tokens per branch) |

With `API_KEY` set, `/v1/*` needs `Authorization: Bearer <key>`.

## Request

```json
{
  "state": {
    "customer": "Dana Ortiz",
    "plan": "enterprise",
    "messages": [
      {"from": "customer", "text": "Our nightly CSV export has failed three nights in a row with \"timeout after 300s\". Finance needs the file by 9am for month-end close."},
      {"from": "agent", "text": "Thanks, I can see the job retried twice. Can you confirm the export size?"},
      {"from": "customer", "text": "About 4 GB. It worked last month. If this is not fixed today we will have to escalate to our account manager."}
    ]
  },
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should own this ticket?",
      "criteria": {
        "billing": "Invoices, payments, refunds",
        "bug": "A product defect or regression",
        "howto": "Usage question with a documented answer",
        "infra": "Capacity, timeouts or performance of hosted jobs",
        "sales": "Plan changes or commercial terms"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is the ticket?",
      "criteria": ["Can wait a week", "This week", "Today", "Within the hour"]
    },
    "escalate": {
      "type": "noul",
      "instructions": "Should a human manager be looped in now?"
    }
  }
}
```

| Question type | `criteria` | Answer |
| --- | --- | --- |
| `choice` | 1–255 named options, each with an optional description | `choice`, `probabilities` over the option names, `confidence = (p_best − 1/n) / (1 − 1/n)` |
| `score` | 2–10 ordered levels | `score` (the expected level), `probabilities` per level, a `legend` and a distance-based `confidence` |
| `noul` | optional descriptions of `true` and `false` | `noul`: P(true) |

`instructions` is optional text for the question. Inputs are never truncated and options are never dropped:
a question whose prompt would exceed the branch limit (`CONTEXT_LENGTH`: 131,072 tokens by default, 65,536 with `ENGINE=trt`) is rejected with 422.

### Images

An optional `images` list carries base64 data URLs (PNG, JPEG, WebP or GIF; at most 8 per request, 5 MB
each); remote URLs are never fetched. The images come before the state in every prompt, are encoded once per
request and their key-value blocks are reused by every branch. Each image costs 562 prompt tokens (560
vision tokens by default; `IMAGE_SOFT_TOKENS`). With two or more images each is introduced as `Image k:`, so questions
can refer to images by number; a single image is a bare placeholder. A malformed image is a 422, and so is any
image when the service runs text only (`VISION=false`, required for the NVFP4 checkpoint).

```json
{"state": "Which chart shows the larger drop?", "images": ["data:image/png;base64,iVBORw0...", "data:image/png;base64,iVBORw0..."],
 "questions": {"chart": {"type": "choice", "criteria": {"first": "Image 1", "second": "Image 2"}}}}
```

## Response

The answers to the request above, as recorded on one H100 (the full record is
[`traces/support-ticket.json`](traces/support-ticket.json)):

```json
{
  "answers": {
    "route": {
      "type": "choice",
      "choice": "infra",
      "probabilities": {"billing": 0.0096, "bug": 0.0172, "howto": 0.0137, "infra": 0.9484, "sales": 0.0111},
      "confidence": 0.9355
    },
    "urgency": {
      "type": "score",
      "score": 1.9594,
      "probabilities": {"0": 0.0325, "1": 0.0381, "2": 0.867, "3": 0.0625},
      "legend": {"0": "Can wait a week", "1": "This week", "2": "Today", "3": "Within the hour"},
      "confidence": 0.8345
    },
    "escalate": {"type": "noul", "noul": 0.9465}
  },
  "usage": {"input_tokens": 2468, "output_tokens": 0, "cached_tokens": 1866, "image_tokens": 0, "effort": "high",
            "branches": 8, "images": 0}
}
```

Probabilities are rounded here. `output_tokens` is always 0: nothing is generated. `input_tokens` sums the
prompts of every branch as TensorRT-LLM counts them, each image as its budget plus two tokens (562 at the default 560). `usage.effort` names the
profile that ran, `usage.branches` the number of prompts scored and `usage.images` the images read. `usage.cached_tokens`
is the part of `input_tokens` served from the K/V cache; `usage.image_tokens` the image soft tokens in one
prompt (images times the per-image budget, `IMAGE_SOFT_TOKENS`), also returned at the top level as `image_tokens`
when the request has images.

## Effort

Each request can choose how many views of each question the engine scores:

| `effort` | What each question gets | Branches / question |
| --- | --- | --- |
| `low` | one candidate order, T = 1 | 1.0 |
| `medium` | two orders; nine-digit rating of every two-option question in both orders; option-count temperature on questions with three or more options | 3.34 |
| `high` (default) | `medium` + two more orders where the first two disagree (three or more options) | 3.43 |

Branch counts are averages over the Decision Index. Choose the effort with a header or a field; a request
with neither gets the service default (`DEFAULT_EFFORT`, `high` unless changed):

```bash
curl -fsS http://127.0.0.1:30104/v1/systemone -H 'content-type: application/json' -H 'X-Decision-Effort: low' -d '{"state": "The invoice was paid twice.", "questions": {"refund": {"type": "noul", "instructions": "Should we refund the duplicate payment?"}}}'
```

```bash
curl -fsS http://127.0.0.1:30104/v1/systemone -H 'content-type: application/json' -d '{"state": "The invoice was paid twice.", "options": {"effort": "medium"}, "questions": {"team": {"type": "choice", "criteria": {"billing": null, "engineering": null, "sales": null}}}}'
```

- Other names, and a header and field that disagree, are rejected with 422 before queueing.
- `options` is Judge's extension of the request; the Decision Index kit omits it and gets the default.

Index score and latency of each effort: [method.md](method.md#effort-profiles).

## Status codes

| Code | Meaning |
| --- | --- |
| 200 | Answered |
| 401 | Missing or invalid API key |
| 413 | Body over 64 MiB |
| 422 | Invalid input, an unknown effort, or a question branch over the branch limit (131,072 tokens by default) |
| 499 | Client disconnected |
| 503 | Not ready yet, or draining after a native failure (`Retry-After`) |
| 504 | Deadline exceeded |
| 529 | Queue full (`Retry-After`) |

A request is answered, or refused with 504, within `queue_timeout + request_timeout + 10` = 340 s by
default; give clients a timeout above that.

## Playground

`GET /` serves a playground that runs your request through `POST /v1/trace`. Use the same JSON as
`/v1/systemone`; you get the same answers plus the engine's record:

- **Images.** **Add images…**, drag and drop or paste attach up to eight; with two or more they are shown
  as `Image 1`, `Image 2`… as the prompt introduces them. Other formats the browser can decode, and files over
  the limits, are re-encoded in the browser. The **Two charts** preset draws two charts to try it.
- **Branches, prefix reuse and timeline.** Every branch's token bar is split into template, images, state,
  question and tail, with the part TensorRT-LLM served from cache hatched over it. A timeline shows the seed
  running alone, the concurrent branches and the gated second pass. Click a branch to see the prompt it adds
  after the shared state: options in that order, or the rating instruction.
- **Per question.** Each order's distribution mapped back to the original options (raw and softened), the
  gate verdict, the nine-bin histograms with E[rating] → p(A), and how the letters and the rating combine
  into the final answer.

**Save trace** downloads the full record and **Open a saved trace…** shows it later, even from
`aleria_judge/playground.html` opened straight from disk without a server. [`traces/`](traces/) holds two traces
recorded on an H100, one of them with a gated second pass.

# Method

Aleria Judge answers a structured question without generating text. Each question becomes a few prompts
(*branches*); the engine prefills each branch once, reads the next-token distribution over the option
labels, and combines the branches into one answer. The default effort, `xhigh`, reads the input twice and
combines two to four orders of the same question. This page describes them, what each adds, how the engine runs
them, and how the method was chosen.

## A branch

All prompts use the state-first chat layout of [Mapika/decider](https://github.com/Mapika/decider),
vendored under Apache-2.0 in `aleria_judge/vendor/`. A prompt opens with a system turn describing the replies it reads, then lists every question of the request (`Questions that will be asked about the context:` and one `- <question>` line each, in request order; since 1.1.1), then contains the state (text as is; JSON indented by two spaces, exactly as sent), one question, its options
with single-token labels (`(A)`, `(B)`, …, then two-letter labels up to 255 options) and `Answer: (`. Each
option is shown as `name: description` (its name alone without a description), except that a name repeating one
of the question's labels exactly (`A`, or `(A)` as shown) is dropped when the option has a description: rotated,
`(A) B: no` would contradict its label. A description that is an exact copy of its name is shown once (`Neutral`,
not `Neutral: Neutral`); if a shortening makes two options look alike, both are shown in full. The
service reads the logits of the label tokens at that position and applies a softmax. That is one prefill
per branch and no decoding.

## The four views of `high`

1. **Two candidate orders.** Each question is scored with its options as given and rotated by n // 2. Each
   order's distribution is mapped back to the original options and the two are averaged. This cancels much
   of the model's position bias.
2. **Nine-bin rating.** A two-option question gets one more branch: the same question, then *"Rate how
   likely option (A) is the correct answer on a scale from 1 to 9 … Return only the digit."* and
   `Rating: `. The expected digit E maps to `p(A) = clamp(0.01 + (E/10 − 0.1) · 0.98/0.8, 0.01, 0.99)`,
   following [simple-jev](https://github.com/featherless-ai/simple-jev) v1. Every two-option question (Noul,
   a two-option Choice or a two-level Score) is rated in both orders, and `p = (p_forward + 1 − p_swapped) / 2`.
   The letter average and the rating are averaged. At T = 1, Gemma puts more than 0.99 on over 95% of its yes/no answers, yet only
   87% of those are right. The rating spreads that probability, which matters for Brier scores and for
   ranking by p(yes).
3. **Gated orders.** If a question has three or more options and its two orders pick different options, a
   second pass scores the rotations by n // 4 and 3n // 4, and all orders are averaged (with three options
   the rotation by n // 4 is the original order, so only one is added). On a 10,000-request balanced
   sample of the index, the gate fired on 9.7% of requests (18.8% of those with a question of three or more
   options).
4. **Option-count temperature.** Questions with three or more options soften each order's distribution with
   `T(n) = max(0.05, 6.9 − 0.7 ln n)` before averaging. This never changes an order's argmax, but it
   changes how disagreeing orders combine and how confident the answer is. Rated questions keep their
   letters at T = 1: tempering them too (T(2) = 6.4) left the rating to order candidates by p(yes) and cost
   about 1.2 nDCG@10 on BRIGHT and ToolRet (full Decision Index 0.3: 59.10 against 59.16). See
   [Disclosures](#disclosures) for how the curve was fitted.

The response follows the shared API ([api.md](api.md)): Choice returns `choice`, `probabilities` over the
original keys and `confidence = (p_best − 1/n) / (1 − 1/n)`. Noul returns P(true). Score returns the
expected level, the distribution, a legend and a distance-based confidence.

## What each view adds

Full Decision Index 0.2.1 runs with the official reproduction kit and scorer, one sequential HTTP client
per server, every request answered. Each row adds one change to the row above it. In these runs truth
questions (Noul, or a Choice whose options are exactly yes/no) were rated in one order only; the last row,
on Decision Index 0.3, rates them in both orders like every other two-option question.

| Configuration | Index | Δ | p50 ms | p95 ms |
| --- | --- | --- | --- | --- |
| One candidate order (Decider prompt, T = 1) | 56.98 | | 89 | 610 |
| + second order, rotated by n/2 | 57.68 | +0.70 | 180 | 1,008 |
| + nine-bin rating on truth questions, option-count temperature elsewhere | 58.37 | +0.69 | 144 | 847 |
| + rating in both orders on other two-option questions | 58.42 | +0.05 | 151 | 951 |
| **+ gated orders** | **58.68** | **+0.26** | **178** | **1,249** |
| *Decision Index 0.3:* the same (one RTX PRO 6000) | 59.09 | | 171 | 1,396 |
| **+ every two-option question rated in both orders (H100)** | **59.16** | **+0.07** | **163** | **1,115** |

The 0.2.1 runs were on one H100 PCIe 80 GB per server; the last one was split into two halves on two H100s.
The last run was split into six shards on six H100 PCIe 80 GB, one sequential client each; its latency is
not the RTX PRO 6000 figure. 99.36% of its decisions equal the previous row's. Latencies come from successive
serving revisions, so compare them only roughly. All of these runs used the TensorRT-LLM engine; release 1.1.0
changes only the engine ([below](#release-110)).

### Release 1.0.1

Release 1.0.1 changes the prompt ([A branch](#a-branch)). Full Decision Index 0.3 runs, each sharded over eight
H100 PCIe 80 GB; each row adds one change to the row above it:

| Configuration | Index | Δ |
| --- | --- | --- |
| Release 1.0.0 | 59.15 | |
| + an option name that repeats one of the question's labels is dropped when the option has a description | 59.34 | +0.19 |
| + a description equal to its name is shown once; JSON states are indented by two spaces, exactly as sent, without Decider's `_index` tags | 59.52 | +0.18 |
| **+ a system turn that states the replies the template reads (release 1.0.1)** | **60.25** | **+0.73** |
| *Release 1.0.1 on one RTX PRO 6000, one sequential client* | 60.24 | |

On 14,038 requests from 11 public datasets outside the index, 1.0.1's macro accuracy is 85.79% against 85.15%
for 1.0.0 (+0.64 points, 95% CI +0.35 to +0.94; Brier 0.232 → 0.223), so the gain is not specific to the index.

### Release 1.1.0

Release 1.1.0 keeps 1.0.1's prompts and method and replaces the engine ([Engine](#engine)). Full Decision Index
0.3 runs, one sequential client per GPU:

| Run | Engine | Hardware | Index | p50 / p95 / mean ms |
| --- | --- | --- | --- | --- |
| 1.0.1 | TensorRT-LLM | one RTX PRO 6000 | 60.24 | 173 / 871 / 403 |
| **1.1.0** | **lean** | **one RTX PRO 6000** | **60.23** | **97 / 759 / 312** |
| 1.0.1 | TensorRT-LLM | H100 PCIe 80 GB, eight shards | 60.25 | 165 / 933 / 413 |
| 1.1.0 | lean | H100 PCIe 80 GB, seven shards | 60.20 | 95 / 798 / 306 |

On the H100 runs the two engines give the same decision on 99.60% of the index's 323,259 questions; two full
lean runs on different GPUs, in different row orders, gave the same decision on all of them. Public vision suite
(12,210 rows, 560 soft tokens per image): 73.38 on one RTX PRO 6000 and 73.54 on an H100 with the lean engine,
73.71 for 1.0.1 on an H100.

Outside the index, macro accuracy on public datasets that are not index or vision-board sources (one sequential
client per H100; paired 95% CI of 1.1.0 − 1.0.1):

| Set | 1.0.1 | 1.1.0 | Δ points (95% CI) |
| --- | --- | --- | --- |
| The 14,038 requests above, 11 text datasets | 85.79 | 85.77 | −0.02 (−0.07 to +0.02) |
| 16,979 requests, 20 more text datasets | 81.36 | 81.32 | −0.04 (−0.13 to +0.04) |
| 6,868 image requests, 5 datasets | 87.91 | 88.24 | +0.32 (−0.04 to +0.71) |
| 2,000 SEED-Bench image questions | 81.50 | 81.35 | −0.15 (−0.50 to +0.15) |
| LongReason-8k, 794 long-context questions | 79.72 | 79.09 | −0.63 (−1.26 to −0.13) |

The 20 text datasets are RTE, MRPC, QQP, CoLA and SST-2 (GLUE), WiC (SuperGLUE), PAWS, SST-5, SciTail, TweetEval
irony, offensive and hate, DBpedia-14, Yahoo Answers, LEDGAR, CaseHOLD, QASC, MedQA-USMLE, AQuA-RAT and
StrategyQA; the image datasets AI2D, ScienceQA (image questions), MMBench (dev), NaturalBench and VQA-RAD
(closed questions). Brier, NLL and ECE agree within 0.006 on every set. Long contexts are the one place the lean
engine trails 1.0.1; the cause is not known yet.

Where the gain over one order comes from, in native benchmark metrics:

| Benchmark | One order | Four views | Main contributor |
| --- | --- | --- | --- |
| ForecastBench (Brier ↓) | 0.262 | 0.187 | rating: a spread probability instead of 0/1 |
| BRIGHT (nDCG@10) | 0.436 | 0.472 | rating: ranking by p(yes) is no longer tied |
| ToolRet (nDCG@10) | 0.629 | 0.651 | rating |
| GSM8K | 0.825 | 0.867 | orders and gated orders |
| MMLU-Pro | 0.695 | 0.712 | orders and gated orders |
| iSarcasmEval (F1) | 0.511 | 0.547 | orders and rating |
| BPoMP | 0.903 | 0.978 | second order |
| HLE | 0.154 | 0.128 | regression (501 questions) |

### Release 1.1.1

Release 1.1.1 keeps 1.1.0's method and engine and adds one thing to every prompt: the list of all the request's
questions, after the images and before `Context:`, one `- <question>` line each, in request order (a question is
shown exactly as its own branch shows it). Every chunk and pass of a request lists all its questions, so all its
branches share one prefix, computed once. The layout is the `header` layout of
[wcole3/jev-brain](https://github.com/wcole3/jev-brain), whose Decision Index entry (stock Gemma 4 31B, one option
order) reached 61.69. Full Decision Index 0.3 runs, BF16, lean engine, one sequential client per H100 PCIe 80 GB:

| Run | Index | Knowledge | Language | Retrieval | Tools | Arts | p50 / p80 / p95 / mean ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1.0 | 60.20 | 48.35 | 62.04 | 63.56 | 77.47 | 47.79 | 95 / 357 / 769 / 279 |
| **1.1.1** | **63.43** | **50.91** | **67.98** | **66.14** | **78.87** | **50.42** | **104 / 387 / 799 / 319** |
| 1.1.1, effort `low` (one order) | 62.45 | 49.87 | 67.55 | 65.85 | 78.30 | 45.95 | 70 / 228 / 570 / 201 |

Latency is over the 140,178 scored requests. Decisions differ from 1.1.0 on 7.5% of the questions. The largest
moves in skill points: ACOS +21.0 (each review asks 58-64 aspect-sentiment yes/no questions; alone, the model said
yes about three times too often), CLadder +12.9, POP909-CL +12.0, VAST +11.6, iSarcasmEval +9.8, PhishNChips +8.1
(each email is asked five phishing-signal questions beside the verdict), ANLI +6.3; NLI4CT −3.2 and cfcolor −1.0
are the largest losses. Two orders still add about one point with the list (`low` 62.45 against `high` 63.43; BPoMP
−13.6 with one order). The public vision suite gives 74.06 (1.1.0: 73.54, same GPU type).

Outside the index (one sequential client per H100; paired 95% CI of 1.1.1 − 1.1.0, points). The multi-question sets
were built for this check, with the question wording fixed before any result: MultiRC (SuperGLUE validation; one
request per paragraph, one yes/no question per candidate answer, 4,848 questions) and UNFAIR-ToS (LexGLUE test; one
request per Terms of Service sentence, a yes/no question per unfair-term category, 1,607 requests):

| Set | 1.1.0 | 1.1.1 | Δ points (95% CI) |
| --- | --- | --- | --- |
| MultiRC, F1 over candidate answers | 87.29 | 89.12 | +1.83 (+0.96 to +2.72) |
| UNFAIR-ToS, F1 over categories (yes answers: 854 → 562; 186 true) | 32.88 | 45.99 | +13.10 (+10.93 to +15.32) |
| 14,038 single-question requests, 11 text datasets (accuracy) | 85.77 | 86.18 | +0.41 (+0.07 to +0.73) |
| 16,979 single-question requests, 20 more text datasets | 81.32 | 81.28 | −0.04 (−0.49 to +0.39) |
| 6,868 image requests, 5 datasets | 88.24 | 88.25 | +0.01 (−0.59 to +0.60) |
| 2,000 SEED-Bench image questions | 81.35 | 81.35 | +0.00 (−0.50 to +0.50) |
| LongReason-8k, 794 long-context questions | 79.09 | 79.97 | +0.88 (−1.13 to +2.90) |

With a single question the list only repeats it before the state, and results are flat; the gain comes from
requests that ask several questions about one state. Cost: the index's prompts compute 9% more tokens. Requests
whose questions carry long texts pay most, because every question's text is in the shared prefix: BRIGHT and
ToolRet (32 candidate documents each) take 6.3 and 4.1 s at the median (1.1.0: 2.8 and 2.1 s). The list also counts
toward the branch limit (`CONTEXT_LENGTH`; the evaluated runs used 65,536): the longest scored public branch is 54,048 tokens, and
the 380 long ToolRet and BRIGHT chunks that the 0.3 scorer excludes reach about 97,000 with the list. At a limit of
131,072 on an H100 80 GB, all 442 excluded requests were answered (median 39 s, at most 89 s), and synthetic
requests with a 127,000-token state or 96,000-112,000 tokens of question text were answered correctly; requests
within 65,536 tokens get identical answers at the higher limit (200 requests). At 262,144, a 160,000-token state
works and 192,000 tokens or more return 413 on that card. 1.1.1 therefore defaults to 131,072 tokens per branch on the
lean engine; TensorRT-LLM keeps 65,536, as evaluated.

### Order averaging on a fine-tuned model

Position bias is not only a stock-model problem. On [Cloudflare/clef](https://huggingface.co/Cloudflare/clef)
(a Qwen3.8-27B fine-tune that reads answers from a joint schema head), averaging the original option order
with its rotation by n/2 raised the full Decision Index 0.2.1 from 61.96 (Cloudflare's server, one order) to
62.94; gated orders added 0.07 more (63.01). Only the orders transfer: clef has its own answer head, so the
rating and temperature do not apply. The two-order runs used our TensorRT-LLM port of clef, whose answers
matched clef's PyTorch code with two orders on 812 of 812 questions, so the gain is the second order's. These
were internal runs; the port is not part of this repository.

## Effort profiles

The same service runs cheaper subsets of the views when a request asks for them
([api.md](api.md#effort)):

| `effort` | Views per question | Branches / question | Decision Index 0.3 |
| --- | --- | --- | --- |
| `low` | one order, T = 1 | 1.0 | no full run |
| `medium` | two orders; rating of every two-option question in both orders; temperature on the rest | 3.34 | no full run |
| `high` | the four views above | 3.43 | 60.23 |
| `xhigh` (default) | `high` without the rating, input read twice (below) | 2.08 | 64.72 |

Branch counts are averages over the Decision Index 0.3 questions. The extra views cost prefill only. On the lean
engine every branch of a pass is computed in one forward over a token tree that holds the shared template, images
and state once; on the TensorRT-LLM engine the branches reuse the KV cache of that prefix, and three branches per
question cost about 2.3 times one at the median (one question, BF16 on one H100 PCIe 80 GB: one order 59 ms, the
four views 135 ms with truth questions rated in one order). The lean engine runs one request's pass at a time,
about 3 requests/s on the index mix on one RTX PRO 6000 (mean 312 ms); the TensorRT-LLM engine sustains about 1.25
requests/s on one H100 at any concurrency. Add GPUs for throughput.

### `xhigh`: the input read twice

The model reads left to right, so a state token never sees the state text after it. `xhigh` writes the state a
second time in the shared prefix, after `Read the same input again before answering:`: the question list, the
state, that line and the state again, then every branch as at `high`. Every token of the second copy has seen the
whole first copy. The copy sits in the prefix the request's branches share, so it costs one extra read of the state
per request, not per branch. A request whose branches would not fit the context limit that way reads the state once;
nothing is truncated. `xhigh` also drops the nine-bin rating, which was neutral on top of the question list (+0.05 on
the index without it) and saves two branches per two-option question.

Measured on the full public part of Decision Index 0.3 (every scored request; BF16 on H100 PCIe 80 GB; question list
and system turn as in 1.1.1): 64.72 against 63.43 for 1.1.1 at `high` (63.46 for the submitted 1.1.1 run). Largest
gains: Home appliance +15.9, CRUXEval +11.1, HoVer +9.2, GSM8K +5.5 and BBH +5.5 skill points; largest losses:
ToolRet −2.5, iSarcasmEval −2.5, FinEntity −2.3 and ACOS −1.7, mostly requests with many questions about a short
input. On held-out sets outside the index the accuracy is unchanged (between −0.22 and +0.50 points on each half,
none significant). Repeating the input follows shisa-ai/shisa-de-2 and the FlexDM entry on the Decision Index board,
which repeat the whole user turn in every question's prompt; here the copy is shared.

## Engine

Two engines serve the same method behind the same server and API: the lean engine (the default,
`JUDGE_ENGINE=lean`) and TensorRT-LLM (`JUDGE_ENGINE=trt`, needed for the NVFP4 checkpoint).

### Lean engine (default)

A prefill-only Gemma 4 31B engine in PyTorch ([`aleria_judge/lean_core.py`](../aleria_judge/lean_core.py),
[`aleria_judge/lean_engine.py`](../aleria_judge/lean_engine.py)): cuBLAS GEMMs on fused weights, FlashInfer attention, norms and
activation, and a few small Triton kernels, BF16 on one GPU. Its numerics follow transformers' `modeling_gemma4`.

- **One forward per pass.** A request's branches are laid out as a token tree: the tokens they share (template,
  images, state, and for ratings the question block) appear once, each branch's own tokens hang off it, and every
  token's position is its depth in its own branch. In each layer every token writes its K/V to its slot, then one
  paged-prefill attention call (pages of one token) lets each token attend to its own branch's path only. One
  forward computes every branch; each gets what its prompt would get alone, up to rounding.
- **Template head and kept K/V.** The template head every prompt starts with is computed at startup and copied
  into each request. The gated orders a request may need are laid into its tree up front and its K/V is kept (up
  to 10 GB, otherwise only what the gated pass will read), so a gated second pass computes only the new orders'
  tokens.
- **Readout.** The last layer runs its queries, attention and MLP only for the branch-final tokens, and the readout
  multiplies those rows by the label rows of the tied LM head only, with Gemma's final logit softcap.
- **Images.** Gemma 4's vision tower (from transformers) encodes each image once per request, at most two per
  tower call; the soft tokens sit in the shared prefix, so every branch and both passes reuse them. Image blocks
  attend bidirectionally in sliding-window layers, as TensorRT-LLM's Gemma 4 does.
- **Any request size.** Up to 16,384 tokens run in one forward; larger trees run in 8,192-token steps, halved until
  the K/V pools and activations fit. A pass that still does not fit drops other requests' kept K/V and retries
  without the gated orders; only a pass too large on its own is the request's HTTP 413, and the service keeps
  serving.
- **Scheduling.** One pass runs at a time on one worker thread (`JUDGE_MAX_REQUESTS` 1; more in inference would only
  keep more K/V in memory); queued requests decode images and build prompts meanwhile. A pass still running 60 s
  past the request timeout stops the service with a stack dump instead of hanging. torch's CPU threads are capped
  at 8 (`JUDGE_CPU_THREADS`): one per physical core stalled image requests on large hosts.
- **Isolation and repeatability.** Each request has its own tree and K/V, dropped when it ends. Answering a
  request set in order, reversed and with 16 clients at once, with new K/V pools filled with NaN, gave identical
  answers; two full index runs on different GPUs gave identical decisions.

### TensorRT-LLM engine (`ENGINE=trt`)

The TensorRT-LLM 1.3.0rc29 PyTorch backend through its `LLM` API, the same layer `trtllm-serve` wraps, on one GPU.

- **Shared prefix.** All branches of a request begin with the same template and state. The first branch
  (the *seed*) runs alone and fills the KV cache; the remaining branches then run concurrently in up to 32
  branch slots and reuse that prefix through TensorRT-LLM's block reuse. KV admission counts a request's
  shared prefix once. When the branches share fewer than 256 prompt tokens, waiting for a seed buys
  nothing, so they all run in one wave. A random `cache_salt` per request, shared by its passes, keeps
  requests from ever sharing KV state while the gated pass reuses the first pass's template, images and state.
- **Prefill only.** Each branch requests one token and reads its generation logits, with FlashInfer
  attention and 2,048-token chunked prefill. No CUDA graphs: on text, breakable prefill graphs were a net
  latency loss (mean −10% and p80 −21% without them on H100, 99.88% of answers equal), and TensorRT-LLM
  refuses them for multimodal models. At startup the service sends itself one readiness request; `/health`
  turns ready after it is answered.
- **Images.** Gemma 4's vision tower encodes a request's images (560 vision tokens each by default) once; they sit
  before the state, so every branch reuses their KV blocks. The tower encodes at most two images per call:
  TensorRT-LLM otherwise encodes every image of an engine step at once, and eight images ran an H100 out of
  memory. Each image is encoded on its own either way, so the embeddings are the same. With two or more
  images each is introduced as `Image k:`; without labels the model lost the middle images of eight (BLINK,
  927 rows: 66.0 against 64.7 skill, multi-image rows 49.8 against 46.9).
- **KV pools.** Gemma 4 mixes full-attention and sliding-window layers. The pools are sized for a
  4,096-token typical sequence, so long many-option requests (several ~7k-token branches) prefill side by
  side. The cache holds at least one branch of the full context length; startup refuses less.
- **Admission.** Branches are admitted first-come, first-served against a budget of resident context
  tokens, so long branches queue instead of forcing KV preemption. Above 4 running and 16 waiting requests
  the service answers 529 with `Retry-After`. Per-request deadlines cancel native work. A native failure
  drains in-flight requests and exits for the supervisor to restart.
- **Host KV tier.** TensorRT-LLM's host (CPU) KV tier is always sized explicitly; the default is none.
  Left to TensorRT-LLM, a container's default memlock limit gives a one-block tier on which the KV manager
  can stop making progress on a long multi-branch request ([tensorrt-llm-patch.md](tensorrt-llm-patch.md)).
- **Patches.** Two process-local patches live in `aleria_judge/patches/`: one stops an engine that makes no progress,
  and one encodes images in slices of two. They install only when the patched TensorRT-LLM files match the
  pinned SHA-256 values ([tensorrt-llm-patch.md](tensorrt-llm-patch.md)).

BF16 weights take 57.2 GiB on the GPU for the text tower, plus the vision tower when images are on. The
evaluated run used a 65,536-token context, 32 branch slots, 2,048-token prefill chunks and a 0.7 KV cache
fraction (headroom for the vision encoder); these are the defaults, except that the lean engine admits 131,072 tokens
per branch since 1.1.1. They change speed and memory, not the
method: other values are allowed ([deploy.md](deploy.md#settings)), logged at startup and reported in
`/health`. In earlier runs, 4 against 32 branch slots gave the same decisions on 99.91% of answers.

## Disclosures

- **Evaluated code.** The 1.1.1 runs (Decision Index 0.3, the public vision suite and the held-out sets on H100
  PCIe 80 GB) were served by 1.1.0's code before the rename plus this release's question list; before release this
  code was checked against it: identical token branches (both passes, 25,711 branches) on 3,000 seeded index
  requests, 396 of them with several questions. The 1.1.0 runs (Decision Index 0.3 and the public vision suite on one RTX PRO 6000) were
  served by this release's code without the question list; apart from the list, only names have changed since (the
  package, settings and metrics), not the method or the engines. The 1.0.x Decision Index 0.3 run was served by the pre-release source of this code. Before
  release this code was checked against it: identical token branches, gate decisions and answers on fixed
  synthetic requests, and the same decisions on 300 random requests of the edition (760 of 760 questions).
- **Temperature fit data.** The option-count temperature curve was fitted by negative log-likelihood on 512
  examples from the training splits of BANKING77, CLINC150, WANLI and ARC-Challenge. BANKING77, CLINC150 and
  ARC are also sources of index benchmarks; the index uses their test splits. Examples equal to an index item
  were meant to be excluded, but the check compared whole strings and missed utterances that the index embeds
  in a longer instruction: one fit example duplicates a CLINC150 test item up to punctuation, and 7 more
  differ from one by a word or two. Refitting without the duplicate gives the same constants; without every
  near match the fit moves one grid step (6.8 − 0.675 ln n).
- **Method selection.** The index was also our development benchmark. The nine-bin rating was frozen as the
  primary change before a full-suite run that applied it to every two-option question (58.17). Restricting
  the single rating to truth questions was decided after that run showed losses on other pairs (58.35 on the
  same stored outputs). Rating the other pairs in both orders and the disagreement gate were then accepted
  by a pre-registered rule on a fixed 10,000-request balanced sample (at least +0.20 index, at most 1.5×
  median latency) and confirmed on the full suite. The rule on option names was later replaced by one rule,
  every two-option question rated in both orders: 59.16 on a full 0.3 run against 59.09, a difference within
  noise, with one rule fewer. Applying the temperature to rated letters as well was tested
  on the full suite and rejected (59.10). The rating text, its mapping and all constants were never tuned on
  index labels. Temperature stays at 1 everywhere except the documented curve. The serving defaults were
  also selected on index runs. The prompt changes of 1.0.1 were measured the same way, on index rows and then
  full runs, before release. The system turn's wording was not tuned: one pre-declared paraphrase gives the
  same index (60.25). The held-out datasets in [Release 1.0.1](#release-101) show the gain outside the index.
  The question list of 1.1.1 was found by comparing our public index run with jev-brain's, row by row, and adopted
  after one full public run; its held-out multi-question sets were chosen and worded before their results. A second
  change, a system turn that mentions the list, was tested in the same night on the full index (63.20) and not
  adopted.
- **Prior work.** The prompt layout comes from Mapika/decider, the question list from wcole3/jev-brain, and the
  rating readout and mapping from simple-jev v1. Order averaging and rating are well-known debiasing and calibration ideas. Our contribution
  is their combination, the disagreement gate, and the TensorRT-LLM serving path.

## Limitations

- Images need the BF16 checkpoint; NVFP4 serves text only. At most 8 images per request.
- BF16 is validated on H100 PCIe 80 GB and RTX PRO 6000 Blackwell Server Edition; NVFP4 on RTX PRO 6000.
  1.1.1 has run the full Decision Index sharded over seven H100 PCIe 80 GB; 1.1.0 also ran it on one RTX PRO 6000.
  Other GPUs are untested.
- Depends on a TensorRT-LLM release candidate plus patches; upgrading means revalidating them.
- Wide menus are costly: two orders double the option prefill, and gating can double it again.
- Long question lists are costly too: every question's text is in each branch's prefix and counts toward the
  branch limit (131,072 tokens by default on the lean engine, 65,536 on TensorRT-LLM; see [Release 1.1.1](#release-111)).
  Such requests take seconds to minutes, and the lean engine runs one request at a time, so they delay the queue.
- Probabilities are only partly calibrated: the rating and the temperature curve reduce Gemma's
  overconfidence, but a stated 0.9 is not guaranteed to be right 90% of the time on your task. If you act on
  `confidence` or a probability threshold, tune that threshold on your own labelled data.

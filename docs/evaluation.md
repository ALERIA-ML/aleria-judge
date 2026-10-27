# Evaluation on the Decision Index

Every Decision Index number for this service is produced **through the production endpoint**: the
service from [deploy.md](deploy.md) runs as deployed, and the official
[reproduction kit](https://github.com/apolinario/decision-index) at commit `9eb2dbe2` (edition 0.3) sends
each request's `state` and `questions` to `POST /v1/systemone` exactly as stored, one request at a time,
with its HTTP engine and scorer unchanged. Nothing calls the engine in-process. The kit's own HTTP timeout
(600 s) is kept: the service answers or returns 504 within its 340 s request bound.

Run the steps in order on the card you evaluate and stop at the first failure. The board measures
latency on one **RTX PRO 6000 Blackwell Server Edition** 96 GB and adds a model only if its median,
mean and 80th-percentile request latency are each at most 1,000 ms. This runbook is for edition 0.3, the
board edition since 2026-10-06.

| Edition 0.3 | |
| --- | --- |
| Public suite | 140,178 scoreable requests (140,620 sent), 43 benchmarks, 37 in the index |
| Board ranking | Full score: 20% public index, 50% private tests of the same skills, 30% private new-domain tasks (the private parts are run by the maintainers only) |
| Latency gate | median, mean and p80 each at most 1,000 ms on one RTX PRO 6000 |
| Kit commit | [`9eb2dbe`](https://github.com/apolinario/decision-index/tree/9eb2dbe2a358004c8782c66e40a83ac07b953fec) |

What you need: a Linux machine with one NVIDIA GPU and Docker (see [deploy.md](deploy.md#requirements)), a
CPU-only Python 3.11+ environment for the kit, and time: a full run takes about 17 hours (BF16) or 16 hours
(NVFP4) on one RTX PRO 6000 and about 18 hours on one H100. Split the suite across GPUs to finish sooner
([step 3](#3-full-run)).

**Effort.** The kit sends requests without `options`, so every request gets the service default, `xhigh`:
the evaluated method. `/health` shows `default_effort`, and every result row records `usage.effort`.

## 0. Install the kit and get the 0.3 suite

A CPU-only Python 3.11+ environment next to the service (on hosts with an older Python, `uv venv -p 3.12`):

```bash
pip install "decision-index[rebuild] @ https://github.com/apolinario/decision-index/archive/9eb2dbe2a358004c8782c66e40a83ac07b953fec.tar.gz"
```

Edition 0.3 is the 0.2.1 row files plus `gsm8k-rows.jsonl.gz` (2,638 rebuilt GSM8K requests). Either
rebuild everything (about 7 GB of downloads; `HF_TOKEN` with accepted
[HLE access](https://huggingface.co/datasets/cais/hle)):

```bash
python -m decision_index suite rebuild --edition 0.3 --work work
```

or import row files you already have (`gsm8k-rows.jsonl.gz` is built from the pinned GSM8K test file
and the 0.2.1 rows by the kit's `decision_index.suite.build.gsm8k_v3`):

```bash
python -m decision_index suite import --edition 0.3 --dir suite-0.3 --rows selected-rows.jsonl.gz --added-rows added-rows.jsonl.gz --gsm8k-rows gsm8k-rows.jsonl.gz
```

A pip-installed kit has no `hub/` folder, so `suite import` copies only the row files. Add the exclusion
list and the 0.3 manifest from the kit's repository, then verify the folder (`"match": true`):

```bash
curl -fsSL https://raw.githubusercontent.com/apolinario/decision-index/9eb2dbe2a358004c8782c66e40a83ac07b953fec/hub/excluded-questions.json -o suite-0.3/excluded-questions.json && curl -fsSL https://raw.githubusercontent.com/apolinario/decision-index/9eb2dbe2a358004c8782c66e40a83ac07b953fec/hub/0.3/manifest.json -o suite-0.3/manifest.json
```

```bash
python -m decision_index suite verify --edition 0.3 --dir suite-0.3
```

`suite import` refuses files whose hashes differ from the published ones. Expected uncompressed hashes:
`b2b56d6f…0792d5` (base), `7429f3c9…4ce89` (added), `75933489…9f4738a` (GSM8K); exclusions
`331df32d…155e81`. A full 0.3 run has 140,620 requests, 140,178 scoreable. The commands below read
`./suite-0.3` (pass `--suite-dir` otherwise).

## 1. Start the service and record the card

```bash
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm --format=csv
```

Workstation and Max-Q editions of the RTX PRO 6000 have other power limits and clocks; record what you
ran on. Start the service (see [deploy.md](deploy.md) for `CHECKPOINT`, `GPU`, `PORT`, `MODEL_DIR` and the
caches) and wait for `/health`:

```bash
docker compose up -d --build
```

```bash
until curl -fsS http://127.0.0.1:30104/health; do sleep 10; done
```

The examples assume `http://127.0.0.1:30104` and the served model name in `$MODEL`
(`gemma-4-31b-trt` for `CHECKPOINT=bf16`, `gemma-4-31b-nvfp4-trt` for `nvfp4`; `/health` shows it).
The kit does not create missing output folders for row files, so create `runs/` first:

```bash
export MODEL=$(curl -fsS http://127.0.0.1:30104/health | python3 -c "import json,sys; print(json.load(sys.stdin)['model'])") && mkdir -p runs
```

If authentication is on, export the key as `DECISION_INDEX_API_KEY`: the kit's HTTP engine sends it as a bearer token.

## 2. Sample run

Before the long run, answer a balanced 200-row sample; every row must come back `ok`, and the scores give a
first look at answers and latency:

```bash
python -m decision_index suite sample --edition 0.3 --dir suite-0.3 --n 200 --seed 20261002 --out runs/sample-rows.jsonl
```

```bash
python -m decision_index run --edition 0.3 --suite-dir suite-0.3 --engine http --option base_url=http://127.0.0.1:30104 --option model=$MODEL --rows runs/sample-rows.jsonl --out runs/sample --compact
```

On a new GPU, answer the same rows on a validated deployment too (BF16 on an H100 PCIe or RTX PRO 6000, the
same code and the default effort) and compare the two `results.jsonl` files question by question. With
`CHECKPOINT=bf16`, decisions should match almost everywhere (kernel differences move near-ties); with
`nvfp4` the weights change, so they will not. Scattered near-ties are expected; a whole benchmark or question
type flipping points to a serving fault (template, KV cache, kernels).

## 3. Full run

The complete 0.3 suite, sequentially, through the same endpoint:

```bash
python -m decision_index pipeline --edition 0.3 --suite-dir suite-0.3 --engine http --option base_url=http://127.0.0.1:30104 --option model=$MODEL --out runs/full --compact
```

`pipeline` runs, scores and writes `scores.json` with `complete: true` when every request has an answer.
Rerunning the same command resumes: answered rows are kept and errors are retried. At about 0.44 s per
request this is about 17 hours on one RTX PRO 6000. Copy `runs/full` off the machine regularly; cloud
instances lose their disk when deleted.

To finish sooner, split the suite into group-disjoint row files and run one service plus one sequential
runner per GPU with `run --rows`, then concatenate the `results.jsonl` files and score them together:

```bash
python -m decision_index score --edition 0.3 --suite-dir suite-0.3 --results runs/merged/results.jsonl --engine http --out runs/merged
```

## 4. Submit

Follow the kit's [instructions](https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard).
The run must be complete (`scores.json` says `"complete": true`) and untouched: the maintainers re-score
`results.jsonl`. Upload the run's files unchanged to `runs/<name>/` of a Hugging Face dataset:

```bash
hf upload <you>/<dataset> runs/full runs/<name> --repo-type dataset --include results.jsonl --include scores.json --include index.json --include benchmark-summary.json --include environment.json --include status.json
```

The maintainers must be able to open it: a private dataset gives them a 404. `results.jsonl` contains
benchmark text, including gated HLE items, so check the source datasets' terms before choosing public or
gated access.

Then open a pull request to the kit adding a row to `submissions/README.md`: the model name, the model repo
(`google/gemma-4-31B-it` @ `842da37`; NVIDIA's NVFP4 quantization of it serves text only), the results link
(`runs/<name>/scores.json`), the kit commit and engine (`http` → this repository at its commit), the
hardware, and the settings (`docker compose up -d`: `CHECKPOINT=bf16`, default effort `xhigh`). After review, the
maintainers start this service themselves, run the private parts of the Full score on it, compare answers on
their own sample and measure latency on one RTX PRO 6000 with their own stratified sample, sent one request
at a time after untimed warm-up rows. The run's own latency (`latency_ms` in `scores.json`) is only a guide:
keep the GPU otherwise idle during it.

Declare the capacity limits: text, plus up to 8 images per request (BF16; PNG, JPEG, WebP or GIF data URLs,
5 MB each, 1,122 prompt tokens each); up to 131,072 tokens per branch (65,536 with `ENGINE=trt`); 1–255 Choice options and 2–10 Score
levels; inputs never truncated and options never filtered.

Disclose (see [method.md](method.md#disclosures)): an inference technique over a stock checkpoint; the
option-count temperature was fitted on the training splits of WANLI, ARC-Challenge, BANKING77 and CLINC150
(three index datasets, whose test splits the index uses; one fit example duplicates an index item up to
punctuation, and refitting without it gives the same constants); the method and the serving defaults were
selected on Decision Index runs (development subsets and full runs).

## Reading the scores

`scores.json` holds the headline `decision_index` (the 0.3 public index, 0–100), area skills and each
benchmark's native metric and chance-corrected skill, plus the run's latency percentiles (`latency_ms`).
Compare runs only on the same edition and the same complete set of requests.

## Cheaper checks

- **Without a GPU.** `JUDGE_TOKENIZER=<google/gemma-4-31B-it directory> python -m pytest tests/test_prompt.py`
  checks the real token layout of the prompts ([deploy.md](deploy.md#tests)).
- **On a sample.** The 200-row sample run of [step 2](#2-sample-run) takes a few minutes.

## What to expect

- **Same code, same GPU type:** the same decisions; the C math library can change the last bit of a
  probability across machines, never a decision.
- **Another GPU:** the method scored 58.68 on two H100s and 58.67 on one RTX PRO 6000 (0.2.1, timeouts
  included). On 2,001 sampled 0.3 requests, an H100 gave the same decisions as an RTX PRO 6000 on 99.81% of
  questions.
- **NVFP4 against BF16:** 1.01 lower, mostly Knowledge, Language and Tools.
- **Stalls:** if the service stops answering, keep the client timeout and rerun `pipeline` to retry errors.
  `docker kill -s USR1 <container>` appends every thread's stack to `$TMPDIR/aleria-judge-stacks.log` in the
  container.

"""GPU-free stand-ins for the prompt builder and TensorRT-LLM engine."""
import asyncio
from contextlib import asynccontextmanager
import time
from types import SimpleNamespace

import httpx

from aleria_judge.images import image_tokens
from aleria_judge.method import HIGH, Branch, gated_orders, keys, orders, rating_kinds

PAYLOAD = {'state': 'A fact', 'questions': {'q': {'type': 'choice', 'criteria': {'a': None, 'b': None}}}}
DIGITS = tuple(range(100, 109))
IMAGE, IMAGE_TOKENS = 7, image_tokens()  # a placeholder token, and what the engine expands it to (1,122)


class Builder:
    """Branches shaped like the real builder's: orders, ratings, gated orders; at most 128 questions per call."""
    label_names = [f'L{i}' for i in range(255)]

    def __init__(self):
        self.calls = []
        self.twice = []  # read_twice of every call, in order
        self.tokenizer = SimpleNamespace(decode=lambda ids: ' '.join(map(str, ids)))
        self.chat = SimpleNamespace(head=[1])
        self.rating = SimpleNamespace(answer_tail=[9])

    def prepare(self, state, questions, profile=HIGH, images=0, listed=None, read_twice=False):
        self.images = images
        self.listed = listed
        return self._build(state, questions, False, profile, images, read_twice)

    def extra(self, state, questions, images=0, listed=None, read_twice=False):
        self.listed = listed
        return self._build(state, questions, True, images=images, read_twice=read_twice)

    def _build(self, state, questions, gated, profile=HIGH, images=0, read_twice=False):
        assert 1 <= len(questions) <= 128
        self.calls.append(len(questions))
        self.twice.append(read_twice)
        # 'too long' never fits; 'fits once' fits only with the state written once
        if state == 'too long' or (state == 'fits once' and read_twice):
            raise ValueError('Prompt exceeds context limit; no truncation')
        branches = []
        for key, spec in questions.items():
            n = len(keys(spec))
            for order in gated_orders(n) if gated else orders(n, profile.orders):
                branches.append(Branch(key, [1] + [IMAGE] * images + [2, 3], tuple(range(n)), order, 'answer',
                                       (1, images, 1, 1, 0)))
            for kind in [] if gated else rating_kinds(n, profile):
                branches.append(Branch(key, [1] + [IMAGE] * images + [2, 4, 5], DIGITS, tuple(range(9)), kind,
                                       (1, images, 1, 2, 0)))
        return branches


class Engine:
    """Answers put `peak` on the first shown label (uniform without it); ratings say 9. `pick` overrides per branch."""

    def __init__(self, delay=0, peak=None, pick=None):
        self.calls = 0
        self.delay, self.peak, self.pick, self.fail_on, self.cancelled = delay, peak, pick, None, 0
        self.passes = []
        self.salts = []

    async def score_request(self, branches, images=(), salt=None):
        self.images = list(images)
        self.salts.append(salt)
        started = self.calls > 0  # the startup readiness probe is never delayed
        try:
            await asyncio.sleep(self.delay if started else 0)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.fail_on is not None and any(b.question == self.fail_on for b in branches):
            raise RuntimeError('Native failure')
        self.calls += 1
        self.passes.append(len(branches))
        rows = []
        for b in branches:
            n = len(b.label_ids)
            if b.kind != 'answer':
                rows.append([0.0] * 8 + [1.0])
                continue
            top = self.pick(b) if self.pick else 0
            if n == 1:
                rows.append([1.0])
            elif self.peak or self.pick:
                peak = self.peak or .9
                rows.append([peak if i == top else (1 - peak) / (n - 1) for i in range(n)])
            else:
                rows.append([1 / n] * n)
        now = time.perf_counter()
        return rows, {'prompt_tokens': [len(b.input_ids) + (IMAGE_TOKENS - 1) * len(images) for b in branches],
                      'cached_tokens': [0] + [2] * (len(branches) - 1), 'timings': [(now, now, now)] * len(branches)}


@asynccontextmanager
async def serving(app):
    """Run the app's lifespan and yield an in-process async client."""
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            yield client


async def until(condition, timeout=1.0):
    for _ in range(int(timeout / .001)):
        if condition():
            return
        await asyncio.sleep(.001)
    raise AssertionError('condition not reached')

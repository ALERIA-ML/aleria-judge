"""The Judge engine on the lean tree-prefill forward (`lean_core`): the same `score_request` contract as `engine.Engine`.

A request's branches form one token tree computed in one forward (several for very long requests). The server passes
the gated orders of every question up front (`future`), so the tree reserves their slots; when the gate fires, the
second pass computes only their new tokens against the K/V kept from the first pass. `release(salt)` drops a request's
K/V when the server is done with it. The forward runs on one worker thread; requests are served one pass at a time.
Images: encoded once per request by Gemma 4's vision tower (`lean_core.VisionEncoder`), their soft tokens placed in
the shared prefix, so every branch and both passes reuse them.
"""
import asyncio
import gc
from collections import OrderedDict
import concurrent.futures
import json
import threading
import time

import torch

from .lean_core import LeanGemma4, VisionEncoder

KEEP = 8  # requests whose K/V may be held at once (released ones go first)
ENDED = 4096  # released salts remembered


def key(b):
    return b.question, b.kind, tuple(b.order)


class LeanEngine:
    wants_future = True
    needs_image_warmup = True

    def __init__(self, model, *, shared_prefix=(), retain_budget_gb=10.0, vision=False, image_soft_tokens=None,
                 vision_batch=2, vision_engine='eager', precision='bf16'):
        self.worker = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix='lean')
        self.model = self.worker.submit(LeanGemma4, model, retain_budget_gb=retain_budget_gb, precision=precision).result()
        self.vision, self.image_soft_tokens = None, image_soft_tokens
        if vision:
            # images run through the tower one at a time, on their real patches only (`vision_batch` is TensorRT-LLM's)
            self.vision = self.worker.submit(VisionEncoder, model, batch=vision_batch, soft_tokens=image_soft_tokens,
                                             engine=vision_engine).result()
            print(json.dumps({'event': 'vision_engine', **self.vision.engine_info}), flush=True)
            self.model.free_bytes = torch.cuda.mem_get_info()[0] - (3 << 30)  # the tower and its buffers
        if shared_prefix:
            self.worker.submit(self.model.set_shared_prefix, list(shared_prefix)).result()
        self.requests = OrderedDict()  # salt -> (lean request, {branch key: sequence index})
        self.lock = threading.Lock()  # requests: the worker thread and release() on the event loop
        self.ended = OrderedDict()  # recently released salts: a pass still running for them must not keep K/V
        print(json.dumps({'event': 'lean_engine', 'precision': precision, 'shared_prefix_tokens': len(shared_prefix),
                          'free_gib': round(self.model.free_bytes / 2**30, 1)}), flush=True)

    async def score_request(self, branches, images=(), salt=None, future=()):
        if images and self.vision is None:
            raise ValueError('This deployment is text-only')
        if not branches:
            raise ValueError('At least one scoring branch is required')
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.worker, self._score, list(branches), salt, list(future), list(images))

    def release(self, salt):
        """Drop a request's kept K/V now (not behind queued passes); a pass still running keeps its own reference."""
        with self.lock:
            self.requests.pop(salt, None)
            self.ended[salt] = None
            while len(self.ended) > ENDED:
                self.ended.popitem(last=False)

    def _kept_bytes(self, exclude):
        with self.lock:
            reqs = [r for s, (r, _) in self.requests.items() if s != exclude]
        return sum(t.numel() * t.element_size() for r in reqs if r.pools for kv in r.pools for t in kv)

    def _evict_others(self, salt):
        with self.lock:
            for s in [s for s in self.requests if s != salt]:
                del self.requests[s]

    def busy_seconds(self):
        """How long the pass on the worker has been running; None when it is idle."""
        since = getattr(self, 'busy_since', None)
        return None if since is None else time.perf_counter() - since

    def _score(self, branches, salt, future, images=()):
        self.busy_since = time.perf_counter()
        try:
            return self._score_pass(branches, salt, future, images)
        finally:
            self.busy_since = None

    def _score_pass(self, branches, salt, future, images=()):
        queued = time.perf_counter()
        features = None
        for attempt in (1, 2):
            # Attempt 1 plans against the memory that other requests' kept K/V leaves and lays this request's possible
            # gated orders into its tree, so their K/V is kept. If that does not fit, attempt 2 drops every kept K/V
            # (other requests recompute their next pass) and computes this pass alone; only a pass that is too large on
            # its own is the request's 413.
            available = self.model.free_bytes - self._kept_bytes(exclude=salt)
            failure = None
            try:
                with self.lock:
                    entry = self.requests.get(salt) if salt and attempt == 1 else None
                if entry is not None and all(key(b) in entry[1] for b in branches):
                    req, index = entry
                    seqs = [index[key(b)] for b in branches]
                else:
                    extra = [b for b in future if key(b) not in {key(x) for x in branches}] if attempt == 1 else []
                    items = branches + extra
                    if images and features is None:
                        features = self.vision.encode(images, self.image_soft_tokens)
                    req = self.model.prepare([b.input_ids for b in items], [list(b.label_ids) for b in items],
                                             retain=bool(extra), images=features, available=available)
                    index = {key(b): i for i, b in enumerate(items)}
                    seqs = list(range(len(branches)))
                    if salt and (req.retain or req.pinned):
                        with self.lock:  # not for a request that already ended (cancelled while this pass waited)
                            if salt not in self.ended:
                                self.requests[salt] = (req, index)
                            while len(self.requests) > KEEP:
                                self.requests.popitem(last=False)
                started = time.perf_counter()
                probs = self.model.run(req, seqs, available=available)
                break
            except (torch.OutOfMemoryError, MemoryError) as error:
                failure = str(error)
            # Outside the except block: no live exception holds the failed pass's frames (and their GPU tensors), so
            # they are freed now, not whenever the cycle collector runs; a later request would otherwise find the GPU
            # still full.
            req = entry = None
            self._evict_others(salt)
            with self.lock:
                self.requests.pop(salt, None)
            gc.collect()
            torch.cuda.empty_cache()
            if attempt == 2:
                features = None
                raise MemoryError(failure)
        finished = time.perf_counter()
        # Tokens computed for each branch in this pass: its path's computed nodes not already counted for an earlier
        # branch (the shared prefix belongs to the first); the rest of its prompt was reused.
        tree, computed, seen = req.tree, set(self.model.last_need), set()
        cached = []
        for b, s in zip(branches, seqs):
            own = [n for n in tree.paths[s] if n in computed and n not in seen]
            seen.update(own)
            cached.append(len(tree.seqs[s]) - sum(tree.nodes[n][2] - tree.nodes[n][1] for n in own))
        # prompt tokens with each image placeholder expanded, as TensorRT-LLM counts them
        return probs, {'prompt_tokens': [len(tree.seqs[s]) for s in seqs], 'cached_tokens': cached,
                       'timings': [(queued, started, finished)] * len(branches)}

    def close(self):
        self.worker.shutdown(wait=True)

    def shutdown(self, timeout=30):
        self.close()

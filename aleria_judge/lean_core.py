"""Lean prefill-only Gemma 4 (text) engine over packed token trees, BF16, one GPU.

A request's prompts (branches) share prefixes: the template head, the state, and for ratings the question block.
`Tree` lays out the distinct tokens once; every token has a slot in the request's K/V pool. A pass computes the tree
nodes its sequences need that are not cached yet, all in one forward: per layer, K/V of every computed token are written
to their slots first, then one batched paged-prefill attention call (page size 1) lets each node attend to its
ancestors' slots, cached or fresh. With `retain`, the pool lives for the whole request, so a later pass (Judge's gated
orders) computes only its new tokens; otherwise each layer's pool is a temporary. A shared prefix registered once at
startup (`set_shared_prefix`, Judge's 73-token template head) is never recomputed: its K/V are copied into the slots
of any request whose sequences start with it.

Readout: final norm and the tied LM head on branch-final tokens only, label rows only, Gemma's final logit softcap,
softmax over the labels (as Judge's `answer_probabilities`).

Numerics follow transformers' modeling_gemma4 (RMSNorm in fp32 with plain weights, attention scaling 1.0, V = K
projection on full-attention layers with an unscaled V norm, proportional partial RoPE on full-attention layers,
per-layer scalar on the residual stream, embedding scale rounded to BF16).
"""
import json
import os
import time

import numpy as np
import torch
import triton
import triton.language as tl


# Test only (JUDGE_LEAN_POISON=1): fill every new K/V pool with NaN instead of leaving freed memory in it, so a read of
# a slot no pass wrote (which could hold another request's K/V) shows up as NaN probabilities.
POISON = os.environ.get('JUDGE_LEAN_POISON') == '1'
# LeanGemma4(precision=...) sets the projection weights of every layer (q/k/v/o, gate/up/down): 'bf16' (the evaluated
# precision) or 'fp8': quantized at load from the BF16 checkpoint with one float32 scale per output channel (channel
# amax / 448), activations quantized per token at run time. Per-token scales keep a token's result independent of the
# other tokens in the forward (other branches, passes, steps). Embeddings, the LM head rows, norms, attention (BF16 K/V)
# and the vision tower stay BF16.
FP8_MAX = 448.0


class Tree:
    """Radix tree of token sequences. Nodes are ordered by start depth, so parents precede children.
    `offsets[i]:offsets[i+1]` are node i's slots."""

    def __init__(self, seqs):
        self.seqs = [np.asarray(s, dtype=np.int64) for s in seqs]
        self.nodes = []   # [sequence index owning the tokens, start depth, end depth]
        self.paths = []   # per sequence: node ids from the root
        for b, seq in enumerate(self.seqs):
            best, depth = -1, 0
            for j in range(b):
                other = self.seqs[j]
                n = min(len(seq), len(other))
                if n <= depth:
                    continue
                diff = np.flatnonzero(seq[:n] != other[:n])
                common = n if len(diff) == 0 else int(diff[0])
                if common > depth:
                    best, depth = j, common
            path = []
            if best >= 0:
                for nid in list(self.paths[best]):
                    _, start, end = self.nodes[nid]
                    if end <= depth:
                        path.append(nid)
                        continue
                    if start < depth:  # split the node at the branch point
                        self.nodes.append([self.nodes[nid][0], depth, end])
                        new = len(self.nodes) - 1
                        self.nodes[nid][2] = depth
                        for p in self.paths:
                            if nid in p:
                                p.insert(p.index(nid) + 1, new)
                        path.append(nid)
                    break
            if depth < len(seq):
                self.nodes.append([b, depth, len(seq)])
                path.append(len(self.nodes) - 1)
            self.paths.append(path)
        order = sorted(range(len(self.nodes)), key=lambda i: (self.nodes[i][1], i))
        rank = {nid: r for r, nid in enumerate(order)}
        self.nodes = [self.nodes[i] for i in order]
        self.paths = [[rank[n] for n in p] for p in self.paths]
        self.offsets = np.concatenate([[0], np.cumsum([e - s for _, s, e in self.nodes])]).astype(np.int64)
        self.num_tokens = int(self.offsets[-1])
        self.prompt_tokens = sum(len(s) for s in self.seqs)
        self.node_path = {}
        for p in self.paths:
            for i, nid in enumerate(p):
                self.node_path.setdefault(nid, p[:i + 1])

    def slots(self, nid):
        return np.arange(self.offsets[nid], self.offsets[nid + 1])

    def readout_slot(self, b):
        last = len(self.seqs[b]) - 1
        for nid in self.paths[b]:
            _, s, e = self.nodes[nid]
            if s <= last < e:
                return nid, int(self.offsets[nid] + last - s)
        raise ValueError('empty sequence')

    def pass_nodes(self, seqs, cached):
        """Nodes a pass over `seqs` computes: those on their paths not in `cached`, plus each readout token's node
        (hidden states are not kept, so a cached readout token is computed again; its K/V come out the same)."""
        need = {n for b in seqs for n in self.paths[b]} - set(cached)
        need |= {self.readout_slot(b)[0] for b in seqs}
        return sorted(need)

    def pinned_for(self, done, future, window):
        """Offsets a later pass over sequences `future` will read, given the nodes `done` once this pass ends: on each
        future path, every computed token before its first uncomputed node (or before its readout node, which is always
        recomputed) for full-attention layers ('g'), and only the last `window` of them for sliding layers ('s')."""
        keep = {'g': [], 's': []}
        for f in future:
            path = self.paths[f]
            readout = self.readout_slot(f)[0]
            j = next((i for i, n in enumerate(path) if n not in done or n == readout), len(path))
            if j == 0:
                continue
            before = np.concatenate([self.slots(n) for n in path[:j]])
            d0 = self.nodes[path[j]][1] if j < len(path) else self.nodes[path[-1]][2]
            keep['g'].append(before)
            keep['s'].append(before[max(0, d0 - window + 1):])  # offsets along a path are listed by depth
        return {k: np.unique(np.concatenate(v)) if v else np.zeros(0, np.int64) for k, v in keep.items()}

    def plan_pass(self, compute, readout_seqs, window, single_max, step_tokens, identity, prealloc=(), initial=None,
                  pinned=None, bidir=frozenset()):
        """Plan one pass over the nodes `compute` (ascending ids, so parents come first).

        Up to `single_max` tokens run as one forward; more are cut into steps of `step_tokens` tokens (a node may span
        steps; a segment may depend on segments of its own step, whose K/V are written before attention). Each segment
        attends over its path: every earlier position for full-attention layers ('g'), the last `window` positions for
        sliding layers ('s'). Slots: with `identity`, a token's global offset (a pool sized to the tree, kept across
        passes); otherwise a liveness allocator per kind hands out slots and frees each token's slot after the last step
        that reads it. Resident tokens start with slots: `initial` (per kind, offset -> slot or -1, from an earlier
        pass) or else `prealloc` offsets (the shared head, copied in, never computed). `pinned` offsets (per kind) are
        never freed, so a later pass can read them. `bidir` nodes are image blocks: never split across steps and placed
        first in their step (`n_bidir` segments), because their queries attend without the causal mask in sliding
        layers.
        Returns host arrays per step, pool capacities and slot tables per kind, and the step and row of every readout."""
        total = sum(self.nodes[n][2] - self.nodes[n][1] for n in compute)
        budget = total if total <= single_max else step_tokens
        if total > single_max:  # depth-first, so a finished subtree's K/V can be freed before its siblings run
            todo, children = set(compute), {}
            for n in compute:
                parent = self.node_path[n][-2] if len(self.node_path[n]) > 1 else -1
                children.setdefault(parent if parent in todo else -1, []).append(n)
            order, stack = [], sorted(children.get(-1, []), reverse=True)
            while stack:
                n = stack.pop()
                order.append(n)
                stack.extend(sorted(children.get(n, []), reverse=True))
            compute = order
        steps, cur, room = [], [], budget
        for n in compute:
            a, end = self.nodes[n][1], self.nodes[n][2]
            if n in bidir and end - a > room and cur:  # an image block is never split across steps
                steps.append(cur)
                cur, room = [], budget
            while a < end:
                take = end - a if n in bidir else min(end - a, room)
                cur.append((n, a, a + take))
                a, room = a + take, room - take
                if room <= 0:
                    steps.append(cur)
                    cur, room = [], budget
        if cur:
            steps.append(cur)
        # image blocks first in each step: their queries attend without the causal mask in sliding layers
        steps = [[s for s in segs if s[0] in bidir] + [s for s in segs if s[0] not in bidir] for segs in steps]
        want = {}
        for i, b in enumerate(readout_seqs):
            want.setdefault(self.readout_slot(b)[1], []).append(i)
        readout = [None] * len(readout_seqs)
        plans, last = [], {'g': np.full(self.num_tokens, -1), 's': np.full(self.num_tokens, -1)}
        for k, segs in enumerate(steps):
            own, kv, pos, toks = [], {'g': [], 's': []}, [], []
            for n, a, b in segs:
                bn, sn, _ = self.nodes[n]
                mine = self.offsets[n] + np.arange(a - sn, b - sn)
                path = np.concatenate([self.slots(p) for p in self.node_path[n][:-1]] +
                                      [self.offsets[n] + np.arange(0, b - sn)])
                kv['g'].append(path)
                kv['s'].append(path[max(0, a - window + 1):])
                for kind in 'gs':
                    last[kind][kv[kind][-1]] = k
                own.append(mine)
                pos.append(np.arange(a, b))
                toks.append(self.seqs[bn][a:b])
            own = np.concatenate(own)
            for r in np.flatnonzero(np.isin(own, list(want))):
                for i in want[int(own[r])]:
                    readout[i] = (k, int(r))
            plans.append(dict(segs=segs, own=own, kv=kv, positions=np.concatenate(pos), tokens=np.concatenate(toks),
                              n_bidir=sum(1 for s in segs if s[0] in bidir),
                              qo_indptr=np.concatenate([[0], np.cumsum([b - a for _, a, b in segs])])))
        if any(r is None for r in readout):
            raise ValueError('a readout token is not computed in this pass')
        slot_of, cap = {}, {}
        for kind in 'gs':
            if identity:
                slot_of[kind], cap[kind] = np.arange(self.num_tokens), self.num_tokens
                continue
            if initial is not None:  # tokens already resident (pinned by an earlier pass, or the copied head)
                table = initial[kind].copy()
            else:
                table = np.full(self.num_tokens, -1)
                table[np.asarray(prealloc, dtype=np.int64)] = np.arange(len(prealloc))
            if pinned is not None:  # kept after this pass for a later one
                last[kind][pinned[kind]] = len(plans)
            resident = table >= 0
            top = int(table.max()) + 1 if resident.any() else 0
            used = np.zeros(top, dtype=bool)
            used[table[resident]] = True
            free = list(np.flatnonzero(~used)[::-1])  # pop() hands out the lowest free slot
            for k, pl in enumerate(plans):
                own = pl['own']
                take = min(len(free), len(own))
                got = [free.pop() for _ in range(take)]
                table[own] = np.concatenate([np.asarray(got, dtype=np.int64), top + np.arange(len(own) - take)])
                top += len(own) - take
                resident[own] = True
                if len(plans) > 1:  # free every slot no later step reads
                    done = np.flatnonzero(resident & (last[kind] <= k))
                    resident[done] = False
                    free.extend(sorted(table[done].tolist(), reverse=True))
                    free.sort(reverse=True)
            slot_of[kind], cap[kind] = table, top
        for pl in plans:
            pl['slots'] = {kind: slot_of[kind][pl['own']] for kind in 'gs'}
            for kind in 'gs':
                idx = [slot_of[kind][x] for x in pl['kv'][kind]]
                assert all((i >= 0).all() for i in idx)
                pl['kv_indptr_' + kind] = np.concatenate([[0], np.cumsum([len(i) for i in idx])])
                pl['kv_indices_' + kind] = np.concatenate(idx)
            del pl['kv']
        return dict(steps=plans, cap=cap, slot_of=slot_of, readout=readout)


@triton.jit
def _qkv_post(qkv, q_out, k_pool, v_pool, slots, qw, kw, cos, sin, stride_tok, HQ: tl.constexpr, HK: tl.constexpr,
              HAS_V: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, BH: tl.constexpr, eps):
    """BH heads of one token: RMSNorm (q, k with weights; v without), rotate-half RoPE on q and k; q to the packed
    output, k and v to the token's pool slot."""
    tok = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    valid = h < HQ + 2 * HK
    is_q = h < HQ
    is_k = (h >= HQ) & (h < HQ + HK)
    is_v = (h >= HQ + HK) & valid
    hq = tl.where(is_q, h, 0)
    hk = tl.where(is_k, h - HQ, 0)
    hv = tl.where(is_v, h - HQ - HK, 0)
    col = tl.where(is_q, h * D, tl.where(is_k, HQ * D + hk * D, (HQ + HAS_V * HK) * D + hv * D))
    offs = tl.arange(0, HALF)
    src = qkv + tok * stride_tok + col[:, None] + offs[None, :]
    m = valid[:, None]
    x1 = tl.load(src, mask=m, other=0.).to(tl.float32)
    x2 = tl.load(src + HALF, mask=m, other=0.).to(tl.float32)
    rms = 1.0 / tl.sqrt((tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)) / D + eps)
    q1 = tl.load(qw + offs).to(tl.float32)[None, :]
    q2 = tl.load(qw + HALF + offs).to(tl.float32)[None, :]
    k1 = tl.load(kw + offs).to(tl.float32)[None, :]
    k2 = tl.load(kw + HALF + offs).to(tl.float32)[None, :]
    w1 = tl.where(is_q[:, None], q1, tl.where(is_v[:, None], 1.0, k1))
    w2 = tl.where(is_q[:, None], q2, tl.where(is_v[:, None], 1.0, k2))
    # transformers rounds the normed vector to BF16 before RoPE
    y1 = (x1 * rms[:, None] * w1).to(tl.bfloat16).to(tl.float32)
    y2 = (x2 * rms[:, None] * w2).to(tl.bfloat16).to(tl.float32)
    c = tl.where(is_v[:, None], 1.0, tl.load(cos + tok * HALF + offs)[None, :])
    s = tl.where(is_v[:, None], 0.0, tl.load(sin + tok * HALF + offs)[None, :])
    o1 = (y1 * c - y2 * s).to(tl.bfloat16)
    o2 = (y2 * c + y1 * s).to(tl.bfloat16)
    slot = tl.load(slots + tok).to(tl.int64)
    qd = (q_out + (tok * HQ + hq) * D)[:, None] + offs[None, :]
    kd = (k_pool + (slot * HK + hk) * D)[:, None] + offs[None, :]
    vd = (v_pool + (slot * HK + hv) * D)[:, None] + offs[None, :]
    tl.store(qd, o1, mask=is_q[:, None])
    tl.store(qd + HALF, o2, mask=is_q[:, None])
    tl.store(kd, o1, mask=is_k[:, None])
    tl.store(kd + HALF, o2, mask=is_k[:, None])
    tl.store(vd, o1, mask=is_v[:, None])
    tl.store(vd + HALF, o2, mask=is_v[:, None])


@triton.jit
def _add_scale(x, d, scale, n, BLOCK: tl.constexpr):
    """x = (x + d) * scale with transformers' BF16 rounding after the add."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = (tl.load(x + i, mask=m).to(tl.float32) + tl.load(d + i, mask=m).to(tl.float32)).to(tl.bfloat16)
    s = tl.load(scale).to(tl.float32)
    tl.store(x + i, (a.to(tl.float32) * s).to(tl.bfloat16), mask=m)


@triton.jit
def _attn_epilogue(o, x, h_out, w_post, w_pre, H, eps, BLOCK: tl.constexpr):
    """Per row: o' = rmsnorm(o) * w_post; x = x + o'; h = rmsnorm(x) * w_pre (BF16 rounding as transformers)."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < H
    ov = tl.load(o + row * H + offs, mask=m, other=0.).to(tl.float32)
    on = (ov / tl.sqrt(tl.sum(ov * ov, 0) / H + eps) * tl.load(w_post + offs, mask=m, other=0.).to(tl.float32))
    xn = (tl.load(x + row * H + offs, mask=m, other=0.).to(tl.float32) + on.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(x + row * H + offs, xn, mask=m)
    xf = xn.to(tl.float32)
    hv = xf / tl.sqrt(tl.sum(xf * xf, 0) / H + eps) * tl.load(w_pre + offs, mask=m, other=0.).to(tl.float32)
    tl.store(h_out + row * H + offs, hv.to(tl.bfloat16), mask=m)


@triton.jit
def _mlp_epilogue(d, x, h_out, w_post, scale, w_next, H, eps, HAS_NEXT: tl.constexpr, BLOCK: tl.constexpr):
    """Per row: d' = rmsnorm(d) * w_post; x = (x + d') * layer_scalar; h = rmsnorm(x) * w_next (the next layer's input
    norm, when there is one). BF16 rounding after each step, as transformers."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < H
    dv = tl.load(d + row * H + offs, mask=m, other=0.).to(tl.float32)
    dn = (dv / tl.sqrt(tl.sum(dv * dv, 0) / H + eps) * tl.load(w_post + offs, mask=m, other=0.).to(tl.float32))
    xa = (tl.load(x + row * H + offs, mask=m, other=0.).to(tl.float32) + dn.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    xs = (xa.to(tl.float32) * tl.load(scale).to(tl.float32)).to(tl.bfloat16)
    tl.store(x + row * H + offs, xs, mask=m)
    if HAS_NEXT:
        xf = xs.to(tl.float32)
        hv = xf / tl.sqrt(tl.sum(xf * xf, 0) / H + eps) * tl.load(w_next + offs, mask=m, other=0.).to(tl.float32)
        tl.store(h_out + row * H + offs, hv.to(tl.bfloat16), mask=m)


@triton.jit
def _quant_rows(x, q, s, K, stride, BLOCK: tl.constexpr):
    """Per row: scale = max |x| / 448 (at least 1e-12), q = x / scale as float8_e4m3fn."""
    row = tl.program_id(0).to(tl.int64)
    acc = tl.zeros((BLOCK,), tl.float32)
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        acc = tl.maximum(acc, tl.abs(tl.load(x + row * stride + offs, mask=offs < K, other=0.).to(tl.float32)))
    scale = tl.maximum(tl.max(acc, 0) / 448.0, 1e-12)
    tl.store(s + row, scale)
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        v = tl.load(x + row * stride + offs, mask=offs < K, other=0.).to(tl.float32) / scale
        tl.store(q + row * K + offs, tl.minimum(tl.maximum(v, -448.0), 448.0).to(tl.float8e4nv), mask=offs < K)


class Fp8:
    """A linear weight [out, in] as float8_e4m3fn with one float32 scale per output channel."""

    def __init__(self, q, scale):
        self.q, self.scale, self.shape = q, scale, q.shape

    @classmethod
    def quantize(cls, w):
        """As quantization_methods/fp8_per_channel.quantize_weight: per row of a fused weight, which equals quantizing
        each of its parts (q, k, v; gate, up) on its own."""
        wf = w.float()
        scale = (wf.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
        q = (wf / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return cls(q.contiguous(), scale.contiguous())

    def __getitem__(self, rows):
        """A slice of the output channels."""
        return Fp8(self.q[rows], self.scale[rows])


def mm(x, w):
    """x @ w.T in BF16; for an Fp8 weight, x quantized per row and one row-wise scaled FP8 GEMM with BF16 output."""
    if not isinstance(w, Fp8):
        return x @ w.T
    n, k = x.shape
    xq = torch.empty(n, k, dtype=torch.float8_e4m3fn, device=x.device)
    sx = torch.empty(n, 1, dtype=torch.float32, device=x.device)
    if n:
        _quant_rows[(n,)](x, xq, sx, k, x.stride(0), 1024, num_warps=4)
    return torch._scaled_mm(xq, w.q.t(), scale_a=sx, scale_b=w.scale.view(1, -1), out_dtype=torch.bfloat16)


class Request:
    """One request's tree, label IDs, computed nodes and (with `retain`) its K/V pools."""

    def __init__(self, tree, label_ids, retain, head_nodes, pinned=False):
        self.tree, self.label_ids, self.retain = tree, label_ids, retain
        self.pinned = pinned           # too large to keep whole: keep only what later passes read (`Tree.pinned_for`)
        self.head_nodes = head_nodes   # nodes whose slots hold the shared prefix (copied, never computed)
        self.computed = set(head_nodes)
        self.read = set()              # sequences already scored
        self.tables = None             # pinned mode: per kind, offset -> slot of the tokens kept for later passes
        self.pools = None              # per layer (K, V) when retained or pinned
        self.n_real = len(tree.seqs)   # scored sequences come first; the rest only shape the tree (splits, head)
        self.features = None           # image soft-token embeddings, concatenated in placeholder order
        self.bidir = frozenset()       # nodes that are image blocks


class LeanGemma4:
    def __init__(self, model_dir, device='cuda', workspace_mb=512, retain_budget_gb=10.0, single_max=16384,
                 step_tokens=8192, precision='bf16'):
        import flashinfer
        from safetensors import safe_open
        self.fi = flashinfer
        top = json.load(open(f'{model_dir}/config.json'))
        cfg = top['text_config']
        self.image_id, self.boi_id, self.eoi_id = top.get('image_token_id'), top.get('boi_token_id'), top.get('eoi_token_id')
        self.cfg = cfg
        self.hidden = cfg['hidden_size']
        self.eps = cfg['rms_norm_eps']
        self.softcap = cfg['final_logit_softcapping']
        self.types = cfg['layer_types']
        self.window = cfg['sliding_window']
        self.hq = cfg['num_attention_heads']
        self.geom = {'sliding_attention': (cfg['num_key_value_heads'], cfg['head_dim']),
                     'full_attention': (cfg['num_global_key_value_heads'], cfg['global_head_dim'])}
        assert cfg['attention_k_eq_v'] and not cfg['enable_moe_block'] and not cfg['hidden_size_per_layer_input']
        assert cfg['num_kv_shared_layers'] == 0 and cfg['tie_word_embeddings']
        assert precision in ('bf16', 'fp8'), f"precision must be 'bf16' or 'fp8', got {precision!r}"
        self.precision = precision
        self.device = device
        index = json.load(open(f'{model_dir}/model.safetensors.index.json'))['weight_map']
        handles = {f: safe_open(f'{model_dir}/{f}', 'pt', device=device) for f in sorted(set(index.values()))}
        get = lambda k: handles[index[k]].get_tensor(k)
        P = 'model.language_model.'
        self.embed = get(P + 'embed_tokens.weight')
        self.norm = get(P + 'norm.weight')
        self.embed_scale = torch.tensor(self.hidden ** 0.5, dtype=torch.bfloat16, device=device)
        self.layers = []
        for i, kind in enumerate(self.types):
            p = f'{P}layers.{i}.'
            a = p + 'self_attn.'
            parts = [get(a + 'q_proj.weight'), get(a + 'k_proj.weight')]
            if kind == 'sliding_attention':
                parts.append(get(a + 'v_proj.weight'))
            qkv = torch.cat(parts, 0)
            del parts
            gu = torch.cat([get(p + 'mlp.gate_proj.weight'), get(p + 'mlp.up_proj.weight')], 0)
            o, down = get(a + 'o_proj.weight'), get(p + 'mlp.down_proj.weight')
            if self.precision == 'fp8':
                qkv, o, gu, down = [Fp8.quantize(w) for w in (qkv, o, gu, down)]
            self.layers.append(dict(
                kind=kind, qkv=qkv, o=o, gu=gu, down=down,
                q_norm=get(a + 'q_norm.weight'), k_norm=get(a + 'k_norm.weight'),
                in_ln=get(p + 'input_layernorm.weight'), post_attn=get(p + 'post_attention_layernorm.weight'),
                pre_ff=get(p + 'pre_feedforward_layernorm.weight'), post_ff=get(p + 'post_feedforward_layernorm.weight'),
                scalar=get(p + 'layer_scalar').to(torch.bfloat16)))
            torch.cuda.empty_cache()
        del handles
        # RoPE inverse frequencies (transformers: default for sliding, proportional partial for full attention).
        rp = cfg['rope_parameters']
        d = cfg['head_dim']
        self.inv_freq = {'sliding_attention': 1.0 / (rp['sliding_attention']['rope_theta'] ** (
            torch.arange(0, d, 2, dtype=torch.int64, device=device).float() / d))}
        gd = cfg['global_head_dim']
        angles = int(rp['full_attention'].get('partial_rotary_factor', 1.0) * gd // 2)
        rot = 1.0 / (rp['full_attention']['rope_theta'] ** (
            torch.arange(0, 2 * angles, 2, dtype=torch.int64, device=device).float() / gd))
        self.inv_freq['full_attention'] = torch.cat([rot, torch.zeros(gd // 2 - angles, device=device)])
        self.workspace = {k: torch.empty(workspace_mb << 20, dtype=torch.uint8, device=device) for k in self.geom}
        self.workspace['bidir'] = torch.empty(workspace_mb << 20, dtype=torch.uint8, device=device)
        self.wrappers = {k: flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace[k], 'NHD') for k in self.workspace}
        # The last layer runs queries only for readout tokens (every token still writes its K/V for them). Its
        # workspace is as large as the others: FlashInfer's split-KV buffers grow with the GPU's SM count, and 64 MB
        # (enough on an H100 PCIe) overflowed on an RTX PRO 6000 (batch_prefill_tmp_v needed 98.6 MB).
        self.last_ws = torch.empty(workspace_mb << 20, dtype=torch.uint8, device=device)
        self.last_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.last_ws, 'NHD')
        self.prune_last = True
        self.H_BLOCK = triton.next_power_of_2(self.hidden)
        # K/V bytes per token over all layers (a retained pool's cost)
        self.kv_bytes = sum(2 * 2 * self.geom[k][0] * self.geom[k][1] for k in self.types)
        self.retain_budget = retain_budget_gb * 2**30
        self.shared, self.shared_kv = None, None
        self.single_max, self.step_tokens = single_max, step_tokens
        self.timing = {}
        # Memory left for K/V pools and activations, with a reserve for the allocator and library workspaces.
        self.free_bytes = torch.cuda.mem_get_info()[0] - (3 << 30)
        # Activation bytes per token of a forward step (BF16): x, normed x, qkv, q, attention out, o, gate-up, act, down.
        hd = max(d for _, d in self.geom.values())
        self.act_bytes = 2 * (4 * self.hidden + max(self.layers[0]['qkv'].shape[0], self.layers[5]['qkv'].shape[0])
                              + 2 * self.hq * hd + 3 * self.layers[0]['gu'].shape[0] // 2)
        if self.precision == 'fp8':  # the FP8 copy of the widest GEMM input (the down projection's) and its scale
            self.act_bytes += self.layers[0]['down'].shape[1] + 4

    # ---- requests -------------------------------------------------------------------------------------------------
    def set_shared_prefix(self, tokens):
        """Compute and keep the K/V of a prefix many requests start with (Judge's template head)."""
        self.shared, self.shared_kv = None, None
        req = self.prepare([list(tokens)], [[0]], retain=True)
        self.run(req, [0])
        self.shared = np.asarray(tokens, dtype=np.int64)
        slots = torch.as_tensor(np.concatenate([req.tree.slots(n) for n in req.tree.paths[0]]), device=self.device)
        self.shared_kv = [(k[slots].clone(), v[slots].clone()) for k, v in req.pools]

    def prepare(self, seqs, label_ids, retain=False, images=None, available=None):
        """A request over `seqs` (every prompt it may score). With `retain`, its K/V persist across `run` calls if they
        fit the budget; `req.retain` says whether they do. `images`: one feature tensor (soft tokens x hidden, from
        `VisionEncoder.encode`) per image placeholder, in order; every sequence holds the same placeholders in its
        shared prefix. Each placeholder becomes BOI + soft tokens + EOI; the soft tokens' embeddings are the features."""
        seqs = [list(s) for s in seqs]
        splits, blobs = [], []
        if images:
            seqs = [self._expand(s, images) for s in seqs]
            first = np.asarray(seqs[0])
            starts = np.flatnonzero((first == self.image_id) & (np.r_[True, first[:-1] != self.image_id]))
            blobs = [(int(s), int(s) + len(f)) for s, f in zip(starts, images)]
            end = blobs[-1][1]
            if any(len(s) <= end or s[:end] != seqs[0][:end] for s in seqs):
                raise ValueError('every branch must hold the same images in its shared prefix')
            for s, e in blobs:  # node boundaries at every image block's start and end
                splits += [seqs[0][:s], seqs[0][:e]]
        shared = self.shared is not None and len(seqs) and all(
            len(s) > len(self.shared) and np.array_equal(np.asarray(s[:len(self.shared)]), self.shared) for s in seqs)
        tree = Tree(seqs + splits + ([self.shared] if shared else []))
        head = tree.paths[-1] if shared else []
        # `available`: the bytes this request may use (the engine's free memory less other requests' kept K/V)
        avail = self.free_bytes if available is None else available
        fits = tree.num_tokens * self.kv_bytes <= min(
            self.retain_budget, avail - min(tree.num_tokens, self.single_max) * self.act_bytes)
        req = Request(tree, list(label_ids) + [[0]] * len(splits) + ([[0]] if shared else []), retain and fits, head,
                      pinned=retain and not fits)
        req.n_real = len(seqs)
        if images:
            req.features = torch.cat([f.to(self.device, torch.bfloat16) for f in images])
            req.blob_start = np.asarray([s for s, _ in blobs])
            req.blob_end = np.asarray([e for _, e in blobs])
            req.blob_off = np.concatenate([[0], np.cumsum([e - s for s, e in blobs])])[:-1]
            req.bidir = frozenset(n for n in tree.paths[0] if any(
                s <= tree.nodes[n][1] and tree.nodes[n][2] <= e for s, e in blobs))
        if req.retain:
            req.pools = [self._pool(kind, tree.num_tokens) for kind in self.types]
            if shared:
                head_slots = np.concatenate([tree.slots(n) for n in head])
                for (k, v), (sk, sv) in zip(req.pools, self.shared_kv):
                    self._fill_head(k, v, head_slots, sk, sv)
        return req

    def _expand(self, seq, images):
        """Each image placeholder (in order) becomes BOI + one image token per soft token + EOI."""
        out, k = [], 0
        for t in seq:
            if t == self.image_id:
                if k >= len(images):
                    raise ValueError('more image placeholders than images')
                out += [self.boi_id] + [self.image_id] * len(images[k]) + [self.eoi_id]
                k += 1
            else:
                out.append(t)
        if k != len(images):
            raise ValueError(f'{len(images)} images but {k} placeholders')
        return out

    def _pool(self, kind, n):
        hk, d = self.geom[kind]
        if POISON:  # test only: a slot that is read before it is written turns the answer into NaN
            return (torch.full((n, 1, hk, d), float('nan'), dtype=torch.bfloat16, device=self.device),
                    torch.full((n, 1, hk, d), float('nan'), dtype=torch.bfloat16, device=self.device))
        return (torch.empty(n, 1, hk, d, dtype=torch.bfloat16, device=self.device),
                torch.empty(n, 1, hk, d, dtype=torch.bfloat16, device=self.device))

    @torch.inference_mode()
    def run(self, req, seqs, available=None):
        """Label probabilities for sequences `seqs` of `req`, computing only the nodes not cached yet."""
        tree = req.tree
        t0 = time.perf_counter()
        cached = req.computed if req.retain or req.pinned else set(req.head_nodes)
        need = tree.pass_nodes(seqs, cached)
        self.last_need = need
        head = np.concatenate([tree.slots(n) for n in req.head_nodes]) if req.head_nodes else np.zeros(0, np.int64)
        kinds = {'sliding_attention': 's', 'full_attention': 'g'}
        per_token = {kk: sum(4 * self.geom[k][0] * self.geom[k][1] for k in self.types if kinds[k] == kk) for kk in 'gs'}
        pinned = None
        if req.pinned:
            future = [b for b in range(req.n_real) if b not in req.read and b not in set(seqs)]
            pinned = tree.pinned_for(cached | set(need), future, self.window)
        step, single = self.step_tokens, self.single_max
        while True:
            plan = tree.plan_pass(need, seqs, self.window, single, step, req.retain,
                                  prealloc=() if req.retain else head, initial=req.tables, pinned=pinned,
                                  bidir=req.bidir)
            multi = len(plan['steps']) > 1
            persistent = req.pinned or (multi and not req.retain)
            largest = max(len(st['tokens']) for st in plan['steps'])
            pools = sum(plan['cap'][kk] * per_token[kk] for kk in 'gs') if persistent else 0
            if pools + largest * self.act_bytes <= (self.free_bytes if available is None else available):
                break
            if step <= 512:
                raise MemoryError(f'request needs {pools / 2**30:.1f} GiB of K/V pools; '
                                  f'{(self.free_bytes if available is None else available) / 2**30:.1f} GiB available')
            if not multi and single:
                single = 0  # pools for every layer at once: cut the pass into steps so slots are freed as it goes
            else:
                # smaller steps: fewer live sliding-window slots and smaller activations. Also when the pass is one step
                # that does not fit even as steps: setting `single` again would change nothing and loop forever, while
                # halving reaches the MemoryError above (the engine then frees memory and retries)
                step //= 2
        if req.retain:
            pools = req.pools
        elif req.pinned:  # pools kept for the request; grown (old slots copied) when a pass needs more
            pools = self._grow(req, plan['cap'], kinds)
            if req.tables is None and len(head):
                for i, (k, v) in enumerate(pools):
                    self._fill_head(k, v, plan['slot_of'][kinds[self.types[i]]][head], *self.shared_kv[i])
        elif multi:  # pools for the pass, every layer
            pools = [self._pool(kind, plan['cap'][kinds[kind]]) for kind in self.types]
            if len(head):
                for i, (k, v) in enumerate(pools):
                    self._fill_head(k, v, plan['slot_of'][kinds[self.types[i]]][head], *self.shared_kv[i])
        else:
            pools = None  # one forward: each layer's pool is a temporary
        labels = [req.label_ids[b] for b in seqs]
        union = sorted({i for ids in labels for i in ids})
        col = {tok: j for j, tok in enumerate(union)}
        union_rows = torch.as_tensor(union, device=self.device)
        cols = [torch.as_tensor([col[i] for i in ids], device=self.device) for ids in labels]
        self.timing = {'host_setup_ms': 1000 * (time.perf_counter() - t0), 'steps': len(plan['steps']),
                       'tokens': sum(len(st['tokens']) for st in plan['steps'])}
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        finals = []
        for k, st in enumerate(plan['steps']):
            if k:  # FlashInfer's plan() reuses a pinned host buffer that the previous step's queued copy still reads
                torch.cuda.current_stream().synchronize()
            finals.append(self._step(st, plan, pools, head, kinds, req))
        rows = torch.stack([finals[k][r] for k, r in plan['readout']])
        logits = rows.float() @ self.embed[union_rows].float().T
        logits = self.softcap * torch.tanh(logits / self.softcap)
        probs = [torch.softmax(logits[j, c], -1) for j, c in enumerate(cols)]
        end.record()
        self.enqueued = time.perf_counter()  # host done launching; the .tolist() below waits for the GPU
        out = [p.tolist() for p in probs]
        self.gpu_ms = start.elapsed_time(end)
        self.timing['gpu_ms'] = self.gpu_ms
        req.read.update(seqs)
        if req.retain or req.pinned:
            req.computed.update(need)
        if req.pinned:
            req.tables = {}
            for kk in 'gs':
                table = np.full(tree.num_tokens, -1)
                table[pinned[kk]] = plan['slot_of'][kk][pinned[kk]]
                req.tables[kk] = table
        return out

    def _grow(self, req, cap, kinds):
        """The request's pinned-mode pools, at least `cap` slots per kind; existing slots keep their contents."""
        old = req.pools
        if old is not None and all(old[i][0].shape[0] >= cap[kinds[k]] for i, k in enumerate(self.types)):
            return old
        pools = []
        for i, kind in enumerate(self.types):
            k, v = self._pool(kind, cap[kinds[kind]])
            if old is not None:
                n = min(old[i][0].shape[0], k.shape[0])
                k[:n] = old[i][0][:n]
                v[:n] = old[i][1][:n]
            pools.append((k, v))
        req.pools = pools
        return pools

    def _fill_head(self, k, v, slots, sk, sv):
        idx = torch.as_tensor(slots, device=self.device)
        k[idx] = sk
        v[idx] = sv

    def _step(self, st, plan, pools, head, kinds, req):
        """One forward over a planned step; returns the final-normed hidden states of its tokens (rows read out)."""
        g = lambda a, dt=torch.int32: torch.as_tensor(np.asarray(a), dtype=dt).to(self.device, non_blocking=True)
        h = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.int32)
        n = len(st['tokens'])
        qo = h(st['qo_indptr'])
        last_page = h(np.ones(len(st['qo_indptr']) - 1))
        slots = {kk: g(st['slots'][kk]) for kk in 'gs'}
        nb = st['n_bidir']  # leading image-block segments: no causal mask in sliding layers (as TensorRT-LLM's Gemma 4)
        rows_b = int(st['qo_indptr'][nb])
        for kind, (hk, d) in self.geom.items():
            kk = kinds[kind]
            qp, kvp, kvi = st['qo_indptr'], st['kv_indptr_' + kk], st['kv_indices_' + kk]
            window = self.window - 1 if kind == 'sliding_attention' else -1
            if kind == 'sliding_attention' and nb:
                cut = kvp[nb]
                segs = st['segs'][:nb]
                if max(b - a for _, a, b in segs) <= self.window:  # every in-block pair is inside the window
                    self.wrappers['bidir'].plan(
                        h(qp[:nb + 1]), h(kvp[:nb + 1]), g(kvi[:cut]), h(np.ones(nb)), self.hq, hk, d, 1, causal=False,
                        sm_scale=1.0, window_left=window, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
                else:  # a block longer than the window: explicit mask, in-block or within the window
                    masks = []
                    for i, (_, a, b) in enumerate(segs):
                        kd = np.arange(b - (kvp[i + 1] - kvp[i]), b)  # depths of the segment's sliding K/V
                        qd = np.arange(a, b)
                        masks.append(((kd[None, :] >= a) | (kd[None, :] > qd[:, None] - self.window)).reshape(-1))
                    self.wrappers['bidir'].plan(
                        h(qp[:nb + 1]), h(kvp[:nb + 1]), g(kvi[:cut]), h(np.ones(nb)), self.hq, hk, d, 1,
                        custom_mask=torch.as_tensor(np.concatenate(masks), device=self.device), sm_scale=1.0,
                        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
                qp, kvp, kvi = qp[nb:] - qp[nb], kvp[nb:] - cut, kvi[cut:]
                if len(qp) == 1:
                    continue
            self.wrappers[kind].plan(
                h(qp), h(kvp), g(kvi), h(np.ones(len(qp) - 1)), self.hq, hk, d, 1,
                causal=True, sm_scale=1.0, window_left=window, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        pos = g(st['positions']).float()
        rope = {}
        for kind, inv in self.inv_freq.items():
            f = pos[:, None] * inv[None, :]
            rope[kind] = (f.cos().contiguous(), f.sin().contiguous())
        x = self.embed[g(st['tokens'], torch.int64)] * self.embed_scale
        if req.features is not None:  # image soft tokens take the vision features
            rows = np.flatnonzero(st['tokens'] == self.image_id)
            if len(rows):
                depth = st['positions'][rows]
                k = np.searchsorted(req.blob_start, depth, side='right') - 1
                if (k < 0).any() or (depth >= req.blob_end[k]).any():
                    raise ValueError('image placeholder outside an image block')
                src = req.blob_off[k] + depth - req.blob_start[k]
                x[g(rows, torch.int64)] = req.features[g(src, torch.int64)]
        rows = sorted({r for k, r in plan['readout'] if plan['steps'][k] is st})
        H, eps, HB = self.hidden, self.eps, self.H_BLOCK
        hn = self.fi.rmsnorm(x, self.layers[0]['in_ln'], eps)
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            kind = layer['kind']
            hk, d = self.geom[kind]
            kk = kinds[kind]
            if pools is not None:
                k_pool, v_pool = pools[i]
            else:
                k_pool, v_pool = self._pool(kind, plan['cap'][kk])
                if len(head):
                    self._fill_head(k_pool, v_pool, plan['slot_of'][kk][head], *self.shared_kv[i])
            cos, sin = rope[kind]
            bh = 16 if d <= 256 else 8
            has_v = int(kind == 'sliding_attention')
            if i == last and self.prune_last:
                return self._last_layer(st, layer, hn, x, rows, k_pool, v_pool, slots[kk], cos, sin, hk, d, has_v, bh)
            qkv = mm(hn, layer['qkv'])
            q = torch.empty(n, self.hq, d, dtype=torch.bfloat16, device=self.device)
            _qkv_post[(n, triton.cdiv(self.hq + 2 * hk, bh))](
                qkv, q, k_pool, v_pool, slots[kk], layer['q_norm'], layer['k_norm'], cos, sin, qkv.stride(0),
                self.hq, hk, has_v, d, d // 2, bh, eps)
            if kind == 'sliding_attention' and nb:
                attn = torch.empty_like(q)
                self.wrappers['bidir'].run(q[:rows_b], (k_pool, v_pool), out=attn[:rows_b])
                if rows_b < n:
                    self.wrappers[kind].run(q[rows_b:], (k_pool, v_pool), out=attn[rows_b:])
            else:
                attn = self.wrappers[kind].run(q, (k_pool, v_pool))
            o = mm(attn.reshape(n, self.hq * d), layer['o'])
            h2 = torch.empty_like(x)
            _attn_epilogue[(n,)](o, x, h2, layer['post_attn'], layer['pre_ff'], H, eps, HB, num_warps=8)
            a = self.fi.activation.gelu_tanh_and_mul(mm(h2, layer['gu']))
            dn = mm(a, layer['down'])
            nxt = self.layers[i + 1]['in_ln'] if i < last else layer['in_ln']
            hn = torch.empty_like(x)
            _mlp_epilogue[(n,)](dn, x, hn, layer['post_ff'], layer['scalar'], nxt, H, eps, int(i < last), HB, num_warps=8)
        out = {}
        if rows:
            normed = self.fi.rmsnorm(x[torch.as_tensor(rows, device=self.device)], self.norm, eps)
            out = {r: normed[j] for j, r in enumerate(rows)}
        return out

    def _last_layer(self, st, layer, hn, x, rows, k_pool, v_pool, slots, cos, sin, hk, d, has_v, bh):
        """Last layer: K/V of every token (later readouts may attend to them), then queries, attention, o_proj and MLP
        for the readout rows only, each attending over its own path."""
        n, H, eps = len(st['tokens']), self.hidden, self.eps
        qd = self.hq * d
        kv = mm(hn, layer['qkv'][qd:])
        _qkv_post[(n, triton.cdiv(2 * hk, bh))](
            kv, kv, k_pool, v_pool, slots, layer['q_norm'], layer['k_norm'], cos, sin, kv.stride(0),
            0, hk, has_v, d, d // 2, bh, eps)
        if not rows:
            return {}
        r = torch.as_tensor(rows, device=self.device)
        q_lin = mm(hn[r], layer['qkv'][:qd])
        q = torch.empty(len(rows), self.hq, d, dtype=torch.bfloat16, device=self.device)
        _qkv_post[(len(rows), triton.cdiv(self.hq, bh))](
            q_lin, q, k_pool, v_pool, torch.zeros(len(rows), dtype=torch.int32, device=self.device), layer['q_norm'],
            layer['k_norm'], cos[r].contiguous(), sin[r].contiguous(), q_lin.stride(0), self.hq, 0, has_v, d, d // 2, bh, eps)
        # each readout row attends to its segment's path up to and including itself
        kk = 's' if has_v else 'g'
        seg = np.searchsorted(st['qo_indptr'], rows, side='right') - 1
        ind, ptr = [], [0]
        for row, sg in zip(rows, seg):
            lo, hi = st['kv_indptr_' + kk][sg], st['kv_indptr_' + kk][sg + 1]
            back = st['qo_indptr'][sg + 1] - 1 - row  # tokens of the segment after this row
            ind.append(st['kv_indices_' + kk][lo:hi - back])
            ptr.append(ptr[-1] + len(ind[-1]))
        i32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.int32)
        self.last_wrapper.plan(i32(np.arange(len(rows) + 1)), i32(ptr),
                               torch.as_tensor(np.concatenate(ind), dtype=torch.int32, device=self.device),
                               i32(np.ones(len(rows))), self.hq, hk, d, 1, causal=True, sm_scale=1.0,
                               window_left=self.window - 1 if has_v else -1,
                               q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
        attn = self.last_wrapper.run(q, (k_pool, v_pool))
        o = mm(attn.reshape(len(rows), qd), layer['o'])
        xr = x[r].contiguous()
        h2 = torch.empty_like(xr)
        _attn_epilogue[(len(rows),)](o, xr, h2, layer['post_attn'], layer['pre_ff'], H, eps, self.H_BLOCK, num_warps=8)
        dn = mm(self.fi.activation.gelu_tanh_and_mul(mm(h2, layer['gu'])), layer['down'])
        _mlp_epilogue[(len(rows),)](dn, xr, xr, layer['post_ff'], layer['scalar'], layer['in_ln'], H, eps, 0,
                                    self.H_BLOCK, num_warps=8)
        normed = self.fi.rmsnorm(xr, self.norm, eps)
        return {row: normed[j] for j, row in enumerate(rows)}

    def forward_tree(self, tree, label_ids):
        """Every sequence of an already built tree in one pass (no retention; the stage-0 interface)."""
        req = self.prepare([s.tolist() for s in tree.seqs], label_ids)
        return self.run(req, list(range(len(tree.seqs))))


def encoder_layers(encoder):
    """The vision tower's encoder layers on one image's real patches as a module: hidden [1, n, H], positions
    [1, n, 2] -> hidden. Without padding patches there is no attention mask. This is the module `vision_tune` hands to
    AITune. (Defined on use, so lean_core imports without torch.nn, as the CPU tree check needs.)"""

    class EncoderLayers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder

        def forward(self, hidden, positions):
            cos, sin = self.encoder.rotary_emb(hidden, positions)
            for layer in self.encoder.layers[:self.encoder.config.num_hidden_layers]:
                hidden = layer(hidden, attention_mask=None, position_embeddings=(cos, sin), position_ids=positions)
            return hidden

    return EncoderLayers().eval()


class VisionEncoder:
    """Gemma 4's image path: transformers' image processor (soft-token budget from the checkpoint's
    processor_config.json unless given), the vision tower and the multimodal embedder, BF16 on one GPU. `encode` returns
    one tensor (soft tokens x LM hidden) per image; the processor fixes each image's soft-token count.

    The tower runs each image on its real patches only: the processor pads every image to the budget's patch count
    with (-1, -1) positions after the real ones, which transformers' tower masks out at the cost of attention over the
    padding. The patch position embedding is a row lookup instead of transformers' one-hot matmul over 10,240 classes.
    Both give the same embeddings bit for bit. `engine`: 'eager' runs the encoder layers in PyTorch; 'aitune' runs them
    as the AITune-tuned (Torch-TensorRT) module `vision_tune` builds into the cache directory on a first start and loads
    on later ones."""

    def __init__(self, model_dir, device='cuda', batch=2, soft_tokens=None, engine='eager'):
        from safetensors import safe_open
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.gemma4 import modeling_gemma4 as mg
        cfg = AutoConfig.from_pretrained(model_dir)
        cfg.vision_config._attn_implementation = 'sdpa'
        self.processor = AutoProcessor.from_pretrained(model_dir).image_processor
        self.tower = mg.Gemma4VisionModel(cfg.vision_config).to(torch.bfloat16)
        self.embedder = mg.Gemma4MultimodalEmbedder(cfg.vision_config, cfg.text_config).to(torch.bfloat16)
        index = json.load(open(f'{model_dir}/model.safetensors.index.json'))['weight_map']
        for module, prefix in ((self.tower, 'model.vision_tower.'), (self.embedder, 'model.embed_vision.')):
            state = {}
            for k in (k for k in index if k.startswith(prefix)):
                with safe_open(f'{model_dir}/{index[k]}', 'pt') as f:
                    state[k[len(prefix):]] = f.get_tensor(k)
            missing, unexpected = module.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(f'{prefix}: missing {missing[:3]}, unexpected {unexpected[:3]}')
        self.tower, self.embedder = self.tower.to(device).eval(), self.embedder.to(device).eval()
        self.device, self.batch = device, batch  # batch: kept for the engine's signature; images run one at a time
        self.k = cfg.vision_config.pooling_kernel_size
        self.soft_tokens = soft_tokens or self.processor.max_soft_tokens
        self.engine, self.engine_info = engine, {'engine': engine}
        self.layers = encoder_layers(self.tower.encoder)
        if engine == 'aitune':
            from .vision_tune import tuned_layers
            self.layers, self.engine_info = tuned_layers(self.layers, model_dir, self.soft_tokens * self.k ** 2, device)
        elif engine != 'eager':
            raise ValueError(f"vision engine must be 'eager' or 'aitune', got {engine!r}")

    def _patches(self, image, soft):
        """One RGB PIL image -> (pixels [n, 3 * patch^2], positions [n, 2]) on the GPU, the real patches only."""
        p = self.processor([image], return_tensors='pt', max_soft_tokens=soft)
        n = int(p['num_soft_tokens_per_image'][0]) * self.k ** 2
        return (p['pixel_values'][0, :n].to(self.device, torch.bfloat16, non_blocking=True),
                p['image_position_ids'][0, :n].to(self.device, non_blocking=True))

    def _tower(self, pixels, positions):
        """One image's real patches -> its soft tokens [n / k^2, vision hidden], as Gemma4VisionModel.forward computes
        them for the padded batch."""
        tower = self.tower
        pixels, positions = pixels[None], positions[None]
        table = tower.patch_embedder.position_embedding_table
        hidden = tower.patch_embedder.input_proj(2 * (pixels - 0.5))
        hidden = hidden + (table[0][positions[..., 0]] + table[1][positions[..., 1]])
        hidden = self.layers(hidden, positions)
        padding = torch.zeros(positions.shape[:2], dtype=torch.bool, device=positions.device)
        hidden, mask = tower.pooler(hidden_states=hidden, pixel_position_ids=positions, padding_positions=padding,
                                    output_length=positions.shape[1] // self.k ** 2)
        hidden = hidden[mask]
        if tower.config.standardize:
            hidden = (hidden - tower.std_bias) * tower.std_scale
        return hidden

    @torch.inference_mode()
    def encode(self, images, soft_tokens=None):
        soft = soft_tokens or self.soft_tokens
        if soft * self.k ** 2 > self.engine_info.get('max_patches', soft * self.k ** 2):
            raise ValueError(f'{soft} soft tokens per image exceed the vision engine built for '
                             f'{self.engine_info["max_patches"] // self.k ** 2}')
        return [self.embedder(inputs_embeds=self._tower(*self._patches(image, soft))) for image in images]

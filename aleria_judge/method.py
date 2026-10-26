"""The decision method, independent of tokenizers and GPUs.

Every answer is read from a single next-token distribution; nothing is generated.
A request chooses how much work each question gets with an effort profile (see
`PROFILES`); the default is `high`. For each question at `high`:

1. Two candidate orders. The options are scored in their original order and
   rotated by n // 2. Each order's label probabilities are mapped back to the
   original options and the orders are averaged with equal weight.
2. Nine-bin rating. Every two-option question gets two extra branches asking
   how likely option (A) is on a 1-9 scale, one in each order. Each expected
   rating maps linearly to a probability; the two are averaged, and the result
   is averaged with the letter readout.
3. Gate. A question with three or more options whose two orders pick
   different options gets the two remaining four-order rotations
   (n // 4 and 3n // 4) in a second pass, and all scored orders are averaged.
4. Temperature. Letters of rated questions stay at T = 1. Every other
   question's per-order probabilities are softened with an option-count
   temperature T(n) before averaging. This never changes an order's argmax.

`xhigh` is `high` without the nine-bin rating, with the request's input read
twice: the shared prefix is the question list, the state, "Read the same input
again before answering:" and the state again, so every state token of the second
copy has seen the whole state. A request whose branches would not fit the context
limit that way reads the state once (capacity only; nothing is truncated).
"""
from dataclasses import dataclass
import math

FIRST_PASS_ORDERS = 2
GATED_ORDERS = 4
# T(n) = max(0.05, 6.9 - 0.7 ln n), fitted by negative log-likelihood on 512 examples from the training splits of
# BANKING77, CLINC150, WANLI and ARC-Challenge. BANKING77, CLINC150 and ARC are also Decision Index sources (the index
# uses their test splits). One fit example duplicates a CLINC150 test item up to punctuation; refitting without it
# gives the same constants.
TEMPERATURE_CURVE = (6.9, -0.7, 0.05)
RATING_INSTRUCTION = ('\n\nRate how likely option (A) is the correct answer on a scale from 1 to 9: '
                      '1 means almost certainly not, 5 means equally likely either way, 9 means almost certainly. '
                      'Return only the digit.')
RATING_PREFIX = 'Rating: '


@dataclass(frozen=True)
class Profile:
    """How much work one request's questions get."""
    name: str            # 'low', 'medium', 'high' or 'xhigh'
    orders: int          # first-pass candidate orders: 1 (original) or 2 (original and n // 2 rotation)
    rating: bool         # nine-digit rating of every two-option question, in both orders (needs two orders)
    gated: bool          # second pass with the n // 4 and 3n // 4 rotations where the two orders disagree (3+ options)
    temperature: bool    # option-count temperature on each order of unrated questions
    read_twice: bool = False  # the state written twice in the shared prefix, when every branch still fits

    def summary(self):
        return {'profile': self.name, 'first_pass_orders': self.orders, 'rating': self.rating,
                'gated_orders': self.gated, 'option_temperature': self.temperature, 'read_twice': self.read_twice}


# low = one order at T = 1: full Decision Index 0.2.1, 56.98 (one H100 PCIe 80 GB, BF16, sequential HTTP).
# medium = two orders, rating, option-count temperature; no full run. high = medium + gated orders: full
# Decision Index 0.3, 59.16 (BF16 on H100 PCIe 80 GB, six shards, sequential HTTP per shard).
# xhigh = high without the rating, input read twice: full Decision Index 0.3 public part, 64.72 against 63.43 for the
# 1.1.1 high (BF16 on H100 PCIe 80 GB, one sequential client per GPU; held-out accuracy unchanged).
PROFILES = {p.name: p for p in (
    Profile('low', orders=1, rating=False, gated=False, temperature=False),
    Profile('medium', orders=2, rating=True, gated=False, temperature=True),
    Profile('high', orders=2, rating=True, gated=True, temperature=True),
    Profile('xhigh', orders=2, rating=False, gated=True, temperature=True, read_twice=True),
)}
DEFAULT_EFFORT = 'xhigh'


def profile(effort=None):
    """The profile for an effort name: 'low', 'medium', 'high' or 'xhigh'."""
    name = effort or DEFAULT_EFFORT
    if name not in PROFILES:
        raise ValueError(f'Unknown effort {effort!r}; use one of {", ".join(PROFILES)}')
    return PROFILES[name]


HIGH = PROFILES['high']


@dataclass
class Branch:
    """One prompt scored by one prefill and a single readout step."""
    question: str
    input_ids: list[int]
    label_ids: tuple[int, ...]
    order: tuple[int, ...]   # original option index shown at each label position (rating: the nine digits)
    kind: str = 'answer'     # 'answer', 'rating' (option A of the original order) or 'rating_rev' (of the swapped order)
    segments: tuple = ()     # token counts (template head, image placeholders, state, question block, tail), for traces


def keys(q):
    """Option keys in request order: Choice names, Noul false/true, Score level indices."""
    if q['type'] == 'choice':
        return list(q['criteria'])
    if q['type'] == 'noul':
        return ['false', 'true']
    if q['type'] == 'score':
        return [str(i) for i in range(len(q['criteria']))]
    raise ValueError('Unsupported question type')


def orders(count, requested):
    """Cyclic candidate orders: the original and its rotation by n // 2, plus n // 4 and 3n // 4 for four."""
    if not 1 <= count <= 255 or requested not in (1, 2, 4):
        raise ValueError('Invalid option or order count')
    offsets = [0] if requested == 1 else [0, count // 2]
    if requested == 4:
        offsets += [count // 4, 3 * count // 4]
    return [tuple(range(offset, count)) + tuple(range(offset)) for offset in dict.fromkeys(offsets)]


def gated_orders(count):
    """Orders the second pass adds: the four-order rotations the first pass did not score."""
    first = set(orders(count, FIRST_PASS_ORDERS))
    return [order for order in orders(count, GATED_ORDERS) if order not in first]


def rating_kinds(count, profile=HIGH):
    """Rating branches for a question: two-option questions are rated in both orders when the profile rates."""
    return ['rating', 'rating_rev'] if count == 2 and profile.rating else []


def option_temperature(n):
    a, b, floor = TEMPERATURE_CURVE
    return max(floor, a + b * math.log(max(n, 2)))


def rating_probability(digit_probabilities):
    """P(option A) from the probabilities of the digits 1..9: 1 maps to 0.01, 9 to 0.99."""
    rating = sum(p * digit for digit, p in enumerate(digit_probabilities, 1))
    return min(.99, max(.01, .01 + (rating / 10 - .1) * (.98 / .8)))


def temper(probabilities, temperature):
    """softmax(z / T) recovered from one branch's softmax(z) over its answer labels."""
    floor = min((p for p in probabilities if p > 0), default=1.0)
    logs = [math.log(p if p > 0 else floor * 1e-6) / temperature for p in probabilities]
    top = max(logs)
    scaled = [math.exp(v - top) for v in logs]
    total = sum(scaled)
    return [v / total for v in scaled]


def probabilities(values, count):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError('Missing label probabilities')
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError('Invalid label probabilities')
    total = sum(values)
    if not math.isclose(total, 1.0, abs_tol=1e-3):
        raise ValueError('Probabilities are not normalized')
    return [p / total for p in values]


def disagreeing(branches, rows):
    """Questions with three or more options whose two answer orders pick different original options."""
    picks = {}
    for b, row in zip(branches, rows, strict=True):
        if b.kind != 'answer' or len(b.order) < 3:
            continue
        best = max(range(len(row)), key=row.__getitem__)
        picks.setdefault(b.question, []).append(b.order[best])
    return [q for q, p in picks.items() if len(p) == 2 and p[0] != p[1]]


def combine(questions, branches, rows, explain=None, profile=HIGH):
    """Typed answers from every scored branch. `explain`, if a dict, receives each question's steps."""
    if len(rows) != len(branches):
        raise ValueError('Missing branch results')
    accum, counts, ratings, reversed_ratings = {}, {}, {}, {}
    rated = {b.question for b in branches if b.kind.startswith('rating')}
    for b, row in zip(branches, rows, strict=True):
        if b.kind.startswith('rating'):
            digits = probabilities(row, len(b.label_ids))
            p = rating_probability(digits)
            (ratings if b.kind == 'rating' else reversed_ratings)[b.question] = p
            if explain is not None:
                steps = explain.setdefault(b.question, {'orders': []})
                steps.setdefault('ratings', []).append({
                    'kind': b.kind, 'digits': digits,
                    'expected_rating': sum(q * d for d, q in enumerate(digits, 1)), 'p_option_a': p})
            continue
        raw = probabilities(row, len(b.order))
        t = 1.0 if b.question in rated or not profile.temperature else option_temperature(len(b.order))
        values = temper(raw, t) if t != 1 else raw
        total = accum.setdefault(b.question, [0.0] * len(b.order))
        counts[b.question] = counts.get(b.question, 0) + 1
        for original, p in zip(b.order, values, strict=True):
            total[original] += p
        if explain is not None:
            explain.setdefault(b.question, {'orders': []})['orders'].append({
                'order': list(b.order), 'temperature': t, 'raw': raw, 'tempered': values,
                'pick': b.order[max(range(len(raw)), key=raw.__getitem__)]})
    values = {key: [p / counts[key] for p in accum[key]] for key in questions}
    if explain is not None:
        for key in questions:
            explain.setdefault(key, {'orders': []})['letters'] = values[key]
    for key, first in ratings.items():
        if key in reversed_ratings:
            first = (first + 1 - reversed_ratings[key]) / 2
        values[key] = [(values[key][0] + first) / 2, (values[key][1] + 1 - first) / 2]
        if explain is not None:
            explain[key]['rating'] = first
    if explain is not None:
        for key in questions:
            explain[key]['final'] = values[key]
    return {'answers': {key: answer(q, values[key]) for key, q in questions.items()}}


def answer(q, values):
    names = keys(q)
    best = max(range(len(values)), key=values.__getitem__)
    if q['type'] == 'noul':
        return {'type': 'noul', 'noul': values[1]}
    distribution = dict(zip(names, values, strict=True))
    if q['type'] == 'choice':
        confidence = 1.0 if len(values) == 1 else (values[best] - 1 / len(values)) / (1 - 1 / len(values))
        return {'type': 'choice', 'choice': names[best], 'probabilities': distribution,
                'confidence': max(0.0, min(1.0, confidence))}
    mid = (len(values) - 1) / 2
    baseline = sum(abs(i - mid) for i in range(len(values))) / len(values)
    distance = sum(p * abs(i - best) for i, p in enumerate(values))
    return {'type': 'score', 'score': sum(i * p for i, p in enumerate(values)),
            'probabilities': distribution, 'legend': dict(zip(names, q['criteria'], strict=True)),
            'confidence': max(0.0, 1 - distance / baseline)}

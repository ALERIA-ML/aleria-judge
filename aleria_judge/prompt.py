"""Prompt token IDs for every branch, in the Decider chat layout.

Every branch of a request starts with the same template head and rendered
state, so the engine computes that prefix once and reuses it. Inputs are
never truncated and options are never filtered.
"""
import threading
from types import SimpleNamespace

from .images import SOFT_TOKENS, image_tokens
from .method import HIGH, RATING_INSTRUCTION, RATING_PREFIX, Branch, gated_orders, orders, rating_kinds
from .vendor import prompt, systemone

MAX_QUESTIONS = 128
DEFAULT_INSTRUCTIONS = 'Which answer fits the context?'
# Every question of the request, listed before the state in the shared prefix (the "header" layout of jev-brain).
QUESTION_LIST = 'Questions that will be asked about the context:\n{}\n\n'
# Effort xhigh writes the state a second time after this line, so the second copy's tokens see the whole state.
READ_AGAIN = '\n\nRead the same input again before answering:\n'


class PlainText:
    """The tokenizer for request content (state, questions, options): text that spells a control token (<|image|>,
    <turn|>, <eos>, ...) is encoded as the characters it is, never as that token, so content cannot add image
    placeholders or close the user turn. Template pieces and image placeholders use the tokenizer itself."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, split_special_tokens=True, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


class FixedOrder:
    """Stands in for the vendored builder's RNG so it lays options out in a given order."""

    def __init__(self, order):
        self.order = order

    def shuffle(self, values):
        if sorted(values) != sorted(self.order):
            raise ValueError('Candidate filtering is forbidden')
        values[:] = self.order


class RatingPrompt:
    """Token pieces of the rating branch, validated against the chat template."""

    def __init__(self, tokenizer, chat):
        self.instruction = tokenizer.encode(RATING_INSTRUCTION, add_special_tokens=False)
        prefix = tokenizer.encode(RATING_PREFIX, add_special_tokens=False)
        digits = []
        for d in '123456789':
            ids = tokenizer.encode(RATING_PREFIX + d, add_special_tokens=False)
            if ids[:-1] != prefix:
                raise ValueError('Rating digit merges with its prefix')
            digits.append(ids[-1])
        if len(set(digits)) != 9:
            raise ValueError('Rating digits are not distinct tokens')
        self.digit_ids = tuple(digits)
        self.tail = list(chat.tail) + prefix
        if tokenizer.encode(chat.tail_text + RATING_PREFIX, add_special_tokens=False) != self.tail:
            raise ValueError('Chat tail merges with the rating prefix')
        self.head = list(chat.head)
        self.answer_tail = list(chat.tail) + chat.answer_ids(tokenizer, 0, False)

    def ids(self, answer_ids, context_ids):
        """The rating prompt: an answer branch's question and options, then the rating instruction."""
        start, end = len(self.head) + len(context_ids), len(answer_ids) - len(self.answer_tail)
        if (answer_ids[:len(self.head)] != self.head or answer_ids[len(self.head):start] != context_ids
                or answer_ids[end:] != self.answer_tail or end <= start):
            raise ValueError('Unexpected answer branch layout')
        return self.head + context_ids + answer_ids[start:end] + self.instruction + self.tail


class Builder:
    def __init__(self, tokenizer, context_length=65536, image_soft_tokens=SOFT_TOKENS):
        self.tokenizer, self.context_length = tokenizer, context_length
        self.text = PlainText(tokenizer)
        # One build at a time: the server builds prompts in a thread pool, and every encode sets the shared Rust
        # tokenizer's encode_special_tokens (True for request content, False for template pieces). Concurrent builds
        # raised 'Already borrowed' and could encode content with the other thread's setting (control tokens).
        self.lock = threading.Lock()
        self.image_tokens = image_tokens(image_soft_tokens)
        self.chat = prompt.chat_template(tokenizer)
        self.label_names, self.label_ids = prompt.label_table(tokenizer)[:2]
        self.rating = RatingPrompt(tokenizer, self.chat)
        image_id = tokenizer.convert_tokens_to_ids('<|image|>')
        # None when the tokenizer has no single image token: such a builder serves text only.
        self.image_id = image_id if tokenizer.encode('<|image|>', add_special_tokens=False) == [image_id] else None

    def prepare(self, state, questions, profile=HIGH, images=0, listed=None, read_twice=False):
        """First pass: the profile's orders per question, plus its rating branches of two-option questions.
        `listed`: every question of the request, shown before the state (default: `questions`). `read_twice`: the
        state written a second time after READ_AGAIN in the shared prefix."""
        with self.lock:
            return self._build(state, questions, gated=False, profile=profile, images=images, listed=listed,
                               read_twice=read_twice)

    def extra(self, state, questions, images=0, listed=None, read_twice=False):
        """Second pass: the remaining four-order rotations of gated questions; `listed` and `read_twice` as in
        prepare (a request's passes must share one prefix)."""
        with self.lock:
            return self._build(state, questions, gated=True, images=images, listed=listed, read_twice=read_twice)

    def listing(self, questions):
        """The question list before the state: one '- <question text>' line per question, in request order."""
        return QUESTION_LIST.format('\n'.join('- ' + self.render(spec).text for spec in questions.values()))

    def _build(self, state, questions, gated, profile=HIGH, images=0, listed=None, read_twice=False):
        if not 1 <= len(questions) <= MAX_QUESTIONS:
            raise ValueError(f'Requires 1–{MAX_QUESTIONS} questions')
        if images and self.image_id is None:
            raise ValueError('This tokenizer has no <|image|> token')
        context = systemone.render_state(state)
        # Tokenize the state once. Passing its true length as the context budget
        # disables the vendored builder's truncation; capacity is checked below.
        # Images come first, as the API places them before the state: one <|image|> placeholder each, which
        # TensorRT-LLM expands to the image's soft tokens (labelled when there are several). They sit in every
        # branch's shared prefix. The list of all the request's questions follows them, before the state, so every
        # pass and chunk of a request shares one prefix. With read_twice the state follows a second time, after
        # READ_AGAIN, still inside that shared prefix: the second copy's tokens attend to the whole first copy.
        pictures = self._images(images)
        text = self.listing(listed or questions) + 'Context:\n' + context
        if read_twice:
            text += READ_AGAIN + 'Context:\n' + context
        context_ids = pictures + self.text.encode(text, add_special_tokens=False)
        branches = []
        for key, spec in questions.items():
            question = self.render(spec, self.label_names[:len(spec.get('criteria') or ())] if spec['type'] == 'choice' else ())
            count = len(question.options)
            own = []
            for order in gated_orders(count) if gated else orders(count, profile.orders):
                item = prompt.build_chat(SimpleNamespace(context=context, qs=[question]), self.text, self.chat,
                                         FixedOrder(order), max_options=255, max_ctx_tokens=len(context_ids),
                                         context_ids=context_ids)
                if tuple(item['perms'][0]) != order or item['nopts'] != [count]:
                    raise ValueError('Runtime changed the full candidate permutation')
                own.append(self._branch(key, item['ids'], tuple(self.label_ids[:count]), order, context_ids,
                                        len(self.rating.answer_tail), images=images, pictures=len(pictures)))
            branches += own
            for kind in [] if gated else rating_kinds(count, profile):
                source = own[0] if kind == 'rating' else next(b for b in own if b.order == (1, 0))
                ids = self.rating.ids(source.input_ids, context_ids)
                branches.append(self._branch(key, ids, self.rating.digit_ids, tuple(range(9)), context_ids,
                                             len(self.rating.tail), kind, images=images, pictures=len(pictures)))
        return branches

    def _images(self, count):
        """The image placeholders. With two or more, each is introduced as 'Image k:' so the model keeps their order
        (unlabelled, it lost images in the middle of eight); a single image stays a bare placeholder."""
        if count < 2:
            return [self.image_id] * count
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        return [i for k in range(1, count + 1) for i in encode(f'Image {k}:\n') + [self.image_id] + encode('\n')]

    def _branch(self, key, ids, label_ids, order, context_ids, tail, kind='answer', images=0, pictures=0):
        """`pictures` is the length of the image placeholders (and their labels) at the start of `context_ids`."""
        if len(ids) + 1 + images * (self.image_tokens - 1) > self.context_length:
            raise ValueError('Prompt exceeds context limit; no truncation')
        head, state = len(self.chat.head), len(context_ids) - pictures
        return Branch(key, ids, label_ids, order, kind,
                      (head, pictures, state, len(ids) - head - pictures - state - tail, tail))

    @staticmethod
    def render(spec, labels=()):
        """The question text and options; `labels` are the label names the question is shown with (see systemone)."""
        if spec.get('instructions') in (None, ''):
            spec = {**spec, 'instructions': DEFAULT_INSTRUCTIONS}  # optional for every type in the API
        if spec['type'] == 'choice' and len(spec['criteria']) == 1:
            # The API allows a single option; the vendored renderer requires two.
            name, description = next(iter(spec['criteria'].items()))
            rendered = {'question': systemone._txt(spec['instructions']),
                        'options': [name if description in (None, '') else f'{name}: {systemone._txt(description)}']}
        else:
            rendered = systemone.render_question(spec, labels)
        return SimpleNamespace(text=rendered['question'], options=rendered['options'], gold=0)

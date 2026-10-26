"""`POST /v1/systemone` over the TensorRT-LLM engine or the lean tree-prefill engine (JUDGE_ENGINE=lean).

A request is one task: prepare (tokenize and validate), wait for an
inference slot, then score under a deadline. The request's effort profile
(`X-Decision-Effort` header or `options.effort`; default `high`) sets the
first pass (orders plus ratings) and whether questions whose orders disagree
get a second pass with two more orders. The task races the client connection;
whatever happens, its own `finally` returns the slot.
"""
import asyncio
from contextlib import asynccontextmanager
import faulthandler
import sys
from functools import partial
import hmac
import logging
import os
from pathlib import Path
import signal
import tempfile
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from . import trace as tracing
from .checkpoint import say
from .images import decode as decode_images
from .config import Settings, differences
from .metrics import ServiceMetrics
from .middleware import BodyLimit, RequestContext
from . import method
from .method import combine, disagreeing
from .prompt import MAX_QUESTIONS
from .schema import EvaluationRequest

log = logging.getLogger(__name__)
CANCEL_GRACE = 5
READINESS_PROBE = ('The recorded color is blue.',
                   {'q': {'type': 'choice', 'instructions': 'What color is recorded?',
                          'criteria': {'red': None, 'blue': None}}})


class QueueTimeout(Exception):
    pass


class NativeFailure(Exception):
    """The engine raised: its executor may be unusable, so the service drains and exits."""


class Gate:
    """`active` requests run inference; up to `queued` more prepare or wait."""

    def __init__(self, active, queued, timeout):
        self._slots = asyncio.Semaphore(active)
        self.capacity, self.timeout = active + queued, timeout
        self.admitted = self.inflight = 0

    @property
    def waiting(self):
        return self.admitted - self.inflight

    def admit(self):
        """Take a place at arrival, before anything awaits, so a burst cannot pass the capacity check together;
        `leave` gives it back when the request ends, however it ends."""
        if self.admitted >= self.capacity:
            return False
        self.admitted += 1
        return True

    def leave(self):
        self.admitted -= 1

    async def acquire(self):
        try:
            await asyncio.wait_for(self._slots.acquire(), self.timeout)
        except TimeoutError:
            raise QueueTimeout from None
        self.inflight += 1
        return True

    def release(self):
        self.inflight -= 1
        self._slots.release()


class Service:
    def __init__(self, settings, engine=None, builder=None):
        self.settings, self.engine, self.builder = settings, engine, builder
        self.owns_engine = engine is None
        self.gate = Gate(settings.active_requests, settings.max_queue, settings.queue_timeout)
        self.ready = self.failed = False
        self.exit_task = None

    # Lifecycle ---------------------------------------------------------------

    async def start(self):
        if self.builder is None:
            from transformers import AutoTokenizer
            from .prompt import Builder
            tokenizer = await run_in_threadpool(AutoTokenizer.from_pretrained, self.settings.tokenizer_path,
                                                local_files_only=True, trust_remote_code=False)
            self.builder = Builder(tokenizer, self.settings.context_length, self.settings.image_soft_tokens)
        if self.engine is None and self.settings.engine == 'lean':
            from .lean_engine import LeanEngine
            self.engine = await run_in_threadpool(LeanEngine, self.settings.model_path, shared_prefix=self.template_head(),
                                                  vision=self.settings.vision,
                                                  image_soft_tokens=self.settings.image_soft_tokens,
                                                  vision_batch=self.settings.vision_encoder_batch)
        if self.engine is None:
            from .engine import Engine
            # TRT-LLM's executor must be created on the process's main thread.
            self.engine = Engine(self.settings.model_path, context_length=self.settings.context_length,
                                 batch_size=self.settings.branch_slots,
                                 max_num_tokens=self.settings.prefill_tokens,
                                 cache_fraction=self.settings.cache_fraction,
                                 nvfp4_backends=[b for b in self.settings.nvfp4_backends.split(',') if b],
                                 host_kv_cache_gb=self.settings.host_kv_cache_gb,
                                 vision=self.settings.vision,
                                 image_soft_tokens=self.settings.image_soft_tokens,
                                 vision_encoder_batch=self.settings.vision_encoder_batch)
        state, questions = READINESS_PROBE
        branches = self.builder.prepare(state, questions)
        combine(questions, branches, (await self.engine.score_request(branches))[0])
        if self.settings.vision and getattr(self.engine, 'needs_image_warmup', False):
            await self.warm_images()
        self.ready = True

    async def warm_images(self):
        """One request with two images before the service is ready: otherwise the first image request compiles the
        vision path's kernels (25-29 s on an H100) while a user waits. The processor scales every image to the soft-token
        budget, so small images run the same shapes."""
        from PIL import Image
        state, questions = READINESS_PROBE
        images = [Image.new('RGB', (64, 64), color) for color in ((200, 30, 30), (30, 30, 200))]
        branches = self.builder.prepare(state, questions, images=len(images))
        combine(questions, branches, (await self.engine.score_request(branches, images))[0])

    async def watch_engine(self):
        """An engine pass running far past the request deadline means the engine is stuck (its worker never returns, so
        every later request would wait out its deadline while /health still answered): dump every thread's stack, fail
        the service (/health and /live turn 503) and drain and exit for the supervisor."""
        limit = self.settings.request_timeout + STALL_GRACE_SECONDS
        while not self.failed:
            await asyncio.sleep(STALL_CHECK_SECONDS)
            busy = getattr(self.engine, 'busy_seconds', None)
            seconds = busy() if busy else None
            if seconds is not None and seconds > limit:
                faulthandler.dump_traceback(file=STACKS[-1] if STACKS else sys.stderr, all_threads=True)
                self.fail('-', f'Engine stalled: one pass has run for {seconds:.0f} s')

    def template_head(self):
        """Token prefix every prompt starts with: the chat template head (system turn, user turn opening), which the
        builder checks on every branch; images and the state follow it."""
        return list(self.builder.chat.head)

    def stop(self):
        self.ready = False
        if self.owns_engine and self.engine is not None:
            self.engine.shutdown()

    @property
    def available(self):
        return self.ready and not self.failed

    def fail(self, request_id, reason):
        """A failed native executor cannot serve again: drain, then exit for the supervisor."""
        log.error('%s (request %s); service marked unavailable', reason, request_id)
        if self.failed:
            return
        self.failed = True
        if self.owns_engine:
            self.exit_task = asyncio.get_running_loop().create_task(self._exit_after_drain())

    async def _exit_after_drain(self):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.drain_timeout
        while (self.gate.inflight or self.gate.waiting) and loop.time() < deadline:
            await asyncio.sleep(.05)
        await asyncio.sleep(.5)  # let the last responses flush
        os._exit(1)

    # Errors ------------------------------------------------------------------

    def unavailable(self, detail='Model is not ready'):
        return HTTPException(503, detail, headers={'Retry-After': str(self.settings.retry_after)})

    def overloaded(self, detail):
        return HTTPException(529, detail, headers={'Retry-After': str(self.settings.retry_after)})

    # Request path ------------------------------------------------------------

    async def prepare(self, state, questions, gated=False, profile=method.HIGH, images=0, listed=None, read_twice=None):
        """First-pass branches of the profile, or with `gated` the second-pass orders of these questions. `listed`:
        every question of the request, shown before the state in every chunk and pass (default: `questions`).
        `read_twice`: the state written twice in the shared prefix; None leaves it to the profile, and a profile that
        reads twice falls back to reading once when a branch would not fit the context limit that way. Returns the
        branches and the choice made, which the request's later passes reuse (one shared prefix per request)."""
        keys = list(questions)
        listed = listed or questions
        twice = profile.read_twice if read_twice is None else read_twice

        async def build_all(twice):
            build = partial(self.builder.extra, images=images, listed=listed, read_twice=twice) if gated else partial(
                self.builder.prepare, profile=profile, images=images, listed=listed, read_twice=twice)
            branches = []
            for start in range(0, len(keys), MAX_QUESTIONS):  # the builder takes at most 128 per call
                chunk = {k: questions[k] for k in keys[start:start + MAX_QUESTIONS]}
                branches += await run_in_threadpool(build, state, chunk)
            return branches

        try:
            try:
                return await build_all(twice), twice
            except ValueError as error:
                if not (twice and read_twice is None and 'context limit' in str(error)):
                    raise
            return await build_all(False), False  # capacity only: the state once, nothing truncated
        except ValueError as error:
            raise HTTPException(422, f'{error}. Every question branch must fit in {self.settings.context_length} '
                                     'tokens including the state, instructions and all options; inputs are never '
                                     'truncated.') from error

    async def decide(self, state, questions, stages, decision, trace=False, profile=method.HIGH, urls=()):
        """Prepare, wait for a slot, score the profile's passes. Validation fails before any queue wait."""
        clock = time.perf_counter()
        acquired = False
        try:
            try:
                images = await run_in_threadpool(decode_images, list(urls)) if urls else []
            except ValueError as error:
                raise HTTPException(422, str(error)) from None
            branches, read_twice = await self.prepare(state, questions, profile=profile, images=len(images))
            if profile.read_twice:
                decision['read_twice'] = read_twice
            stages['prepare'], clock = ms(clock), time.perf_counter()
            acquired = await self.gate.acquire()
            stages['queue'], clock = ms(clock), time.perf_counter()
            if self.failed:
                raise self.unavailable('Inference engine failed; retry after the service restarts')
            started = clock
            # One cache salt for all passes of this request: the gated pass reuses the first pass's template, images
            # and state, while other requests (each with its own salt) can never reuse this request's KV blocks.
            salt = uuid.uuid4().hex
            future = {}
            if profile.gated and getattr(self.engine, 'wants_future', False):
                # every possible gated order, so the engine lays them into the request's tree and keeps its K/V
                future = {'future': (await self.prepare(state, questions, gated=True, profile=profile, images=len(images),
                                                        read_twice=read_twice))[0]}
            try:
                async with asyncio.timeout(self.settings.request_timeout):
                    rows, stats = await self.score(branches, images, salt, **future)
                    passes = [(branches, rows, stats)]
                    gated = disagreeing(branches, rows) if profile.gated else []
                    if gated:  # the orders disagree: score the other two rotations of those questions
                        extra, _ = await self.prepare(state, {k: questions[k] for k in gated}, gated=True,
                                                      profile=profile, images=len(images), listed=questions,
                                                      read_twice=read_twice)
                        if extra:
                            passes.append((extra, *await self.score(extra, images, salt)))
                            decision['gated_questions'] = len(gated)
            except TimeoutError:
                raise HTTPException(504, f'Inference deadline of {self.settings.request_timeout:g} seconds exceeded')
            except MemoryError as error:
                raise HTTPException(413, f'Request does not fit the GPU: {error}')
            finally:
                if hasattr(self.engine, 'release'):
                    self.engine.release(salt)
            stages['inference'] = ms(clock)
        except QueueTimeout:
            raise self.overloaded('Queue deadline exceeded') from None
        finally:
            if acquired:
                self.gate.release()
        branches = [b for p in passes for b in p[0]]
        rows = [r for p in passes for r in p[1]]
        steps = {} if trace else None
        result = combine(questions, branches, rows, steps, profile)
        # Prompt tokens as TensorRT-LLM counts them: each image placeholder is the image's soft tokens.
        cached = sum(c for p in passes for c in p[2]['cached_tokens'])
        image_tokens = len(images) * self.settings.image_soft_tokens
        # input_tokens counts every branch's whole prompt; cached_tokens is the part served from the K/V cache.
        # image_tokens: the image soft tokens in one prompt (images x the per-image budget), also at the top level,
        # where the Decision Index vision kit reads a run's image budget.
        result['usage'] = {'input_tokens': sum(n for p in passes for n in p[2]['prompt_tokens']), 'output_tokens': 0,
                           'cached_tokens': cached, 'image_tokens': image_tokens,
                           'effort': profile.name, 'branches': len(branches), 'images': len(images)}
        if images:
            result['image_tokens'] = image_tokens
        decision.update(branches=len(branches), input_tokens=result['usage']['input_tokens'], cached_tokens=cached)
        if trace:
            result['trace'] = await run_in_threadpool(tracing.build, self.builder, questions, passes, steps, gated,
                                                      started, dict(stages), profile)
            result['trace']['method'] = tracing.method_info(self.settings.default_effort)
        return result

    async def score(self, branches, images, salt, **kwargs):
        """Engine errors become NativeFailure; a request too large for the GPU (MemoryError, the engine's own planning)
        stays that request's 413, and errors in preparing, combining or tracing stay the request's."""
        try:
            return await self.engine.score_request(branches, images, salt=salt, **kwargs)
        except MemoryError:
            raise
        except Exception as error:  # cancellation (deadline, disconnect) is a BaseException and passes through
            raise NativeFailure(repr(error)) from error

    def effort(self, body, request):
        """The request's profile: `X-Decision-Effort` header or `options.effort`, else the service default."""
        header = request.headers.get('x-decision-effort')
        field = body.options.effort if body.options else None
        try:
            chosen = [method.profile(e) for e in (header, field) if e]
            if len({p.name for p in chosen}) > 1:
                raise ValueError(f'X-Decision-Effort ({header!r}) and options.effort ({field!r}) disagree')
            return chosen[0] if chosen else method.profile(self.settings.default_effort)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    async def handle(self, body, request, trace=False):
        profile = self.effort(body, request)
        if not self.available:
            raise self.unavailable()
        if body.images and not self.settings.vision:
            raise HTTPException(422, 'This deployment is text-only; omit images or send an empty list')
        if not self.gate.admit():
            raise self.overloaded('Queue is full')
        try:
            return await self._admitted(body, request, trace, profile)
        finally:
            self.gate.leave()

    async def _admitted(self, body, request, trace, profile):
        context = request.scope['state']
        questions = {key: question.model_dump() for key, question in body.questions.items()}
        context['decision'].update(questions=len(questions), effort=profile.name)
        task = asyncio.create_task(self.decide(body.state, questions, context['stages'], context['decision'], trace,
                                               profile, body.images))
        stop = asyncio.Event()
        watcher = asyncio.create_task(watch_disconnect(request, stop))
        bound = self.settings.queue_timeout + self.settings.request_timeout + 10
        try:
            await asyncio.wait((task, watcher), timeout=bound, return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                result = task.result()
                return result
            if watcher.done():
                raise HTTPException(499, 'Client disconnected')
            raise HTTPException(504, 'Request exceeded its overall deadline')
        except HTTPException:
            raise
        except NativeFailure:
            log.exception('Native inference failed')
            self.fail(context['request_id'], 'Native inference failed')
            raise self.unavailable('Inference engine failed; retry after the service restarts') from None
        except Exception:
            log.exception('Request %s failed outside the engine', context['request_id'])
            raise HTTPException(500, 'Internal error in this request; the service keeps serving') from None
        finally:
            stop.set()
            await self._settle((task, watcher), context['request_id'])

    async def _settle(self, tasks, request_id):
        for task in tasks:
            task.cancel()
        done, pending = await asyncio.wait(tasks, timeout=CANCEL_GRACE)
        for task in done:
            if not task.cancelled():
                task.exception()  # retrieved; already reported above
        if pending:
            self.fail(request_id, f'Native cancellation exceeded {CANCEL_GRACE} seconds')


def ms(since):
    return 1000 * (time.perf_counter() - since)


async def watch_disconnect(request, stop):
    # Starlette checks receive() inside an AnyIO cancel scope that can swallow
    # an outer cancellation, so poll with an explicit stop flag instead.
    while not stop.is_set():
        if await request.is_disconnected():
            return
        await asyncio.sleep(.1)


STACKS = []
STALL_CHECK_SECONDS = 10
STALL_GRACE_SECONDS = 60  # past request_timeout


def register_stack_dump():
    """`docker kill -s USR1 <container>` appends every thread's stack to $TMPDIR/aleria-judge-stacks.log (/tmp, mounted from
    the host cache folder). faulthandler writes from the signal handler without the GIL or stdout, so even a process
    frozen inside a blocked log write can be diagnosed. Registered once the engine is up: its native libraries
    install their own SIGUSR1 handler while it loads."""
    STACKS.append(open(Path(tempfile.gettempdir()) / 'aleria-judge-stacks.log', 'a', buffering=1))
    faulthandler.register(signal.SIGUSR1, file=STACKS[-1], all_threads=True)


def create_app(settings=None, *, engine=None, builder=None):
    settings = settings or Settings.from_env()
    service = Service(settings, engine, builder)
    created = int(time.time())

    @asynccontextmanager
    async def lifespan(app):
        try:
            await service.start()
            register_stack_dump()
            watcher = asyncio.get_running_loop().create_task(service.watch_engine())
            host = 'localhost' if settings.host in ('127.0.0.1', '0.0.0.0', '::') else settings.host
            say(f'ready. Playground: http://{host}:{settings.port}/   API: POST http://{host}:{settings.port}/v1/systemone')
            yield
            watcher.cancel()
        finally:
            service.stop()

    app = FastAPI(title='Aleria Judge decision service', lifespan=lifespan)
    app.state.service = service
    app.state.metrics = metrics = ServiceMetrics(settings, service, access_log=settings.access_log)
    app.add_middleware(BodyLimit, limit=settings.max_body_bytes)
    app.add_middleware(RequestContext, on_response=metrics.observe)

    async def authenticate(authorization: str | None = Header(default=None)):
        expected = f'Bearer {settings.api_key}'.encode()
        if settings.api_key and not hmac.compare_digest((authorization or '').encode(), expected):
            raise HTTPException(401, 'Missing or invalid API key', headers={'WWW-Authenticate': 'Bearer'})

    @app.exception_handler(RequestValidationError)
    async def invalid(request, error):
        return JSONResponse(status_code=422, content={'detail': [
            {k: item[k] for k in ('loc', 'msg', 'type')} for item in error.errors()]})

    @app.post('/v1/systemone', dependencies=[Depends(authenticate)])
    async def systemone(body: EvaluationRequest, request: Request):
        return await service.handle(body, request)

    if settings.trace:
        @app.post('/v1/trace', dependencies=[Depends(authenticate)])
        async def trace(body: EvaluationRequest, request: Request):
            """Same answers as /v1/systemone, plus what the engine did for every branch."""
            return await service.handle(body, request, trace=True)

    @app.get('/health')
    async def health():
        if not service.available:
            raise service.unavailable()
        return {'status': 'ready', 'model': settings.served_name, 'engine': settings.engine,
                'cpu_threads': settings.cpu_threads, 'checkpoint': settings.checkpoint,
                'checkpoint_path': settings.model_path, 'precision': settings.spec.precision,
                'modalities': ['text', 'image'] if settings.vision else ['text'], 'authentication': bool(settings.api_key),
                'image_soft_tokens': settings.image_soft_tokens if settings.vision else None,
                'context_length': settings.context_length, 'prefill_tokens': settings.prefill_tokens,
                'nvfp4_backends': settings.nvfp4_backends or 'default', 'branch_slots': settings.branch_slots,
                'host_kv_cache_gb': settings.host_kv_cache_gb, 'differs_from_evaluated': differences(settings),
                'default_effort': method.profile(settings.default_effort).name,
                'method': tracing.method_info(settings.default_effort),
                'trace': settings.trace}

    @app.get('/live')
    async def live():
        if service.failed:
            raise service.unavailable('Inference engine failed; restart pending')
        return {'status': 'alive', 'ready': service.ready}

    @app.get('/metrics', include_in_schema=False)
    async def prometheus():
        body, content_type = metrics.render()
        return Response(body, media_type=content_type)


    @app.get('/v1/models', dependencies=[Depends(authenticate)])
    async def models():
        """The served model, in the OpenAI model-list format vLLM uses."""
        return {'object': 'list', 'data': [{
            'id': settings.served_name, 'object': 'model', 'created': created, 'owned_by': 'aleria',
            'root': settings.spec.repo, 'parent': None, 'max_model_len': settings.context_length}]}

    @app.get('/', response_class=HTMLResponse, include_in_schema=False)
    async def playground():
        return Path(__file__).with_name('playground.html').read_text()

    return app

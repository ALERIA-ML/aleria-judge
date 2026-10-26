import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from aleria_judge import server
from aleria_judge.config import Settings
from aleria_judge.method import option_temperature
from aleria_judge.server import create_app
from tests.fakes import IMAGE, IMAGE_TOKENS, PAYLOAD, Builder, Engine, serving, until


def app_with(engine=None, builder=None, **settings):
    return create_app(Settings(**settings), engine=engine or Engine(), builder=builder or Builder())


def idle(app):
    return app.state.service.gate.inflight == app.state.service.gate.waiting == 0


# Contract -------------------------------------------------------------------

def test_answers_usage_and_headers():
    with TestClient(app_with()) as c:
        r = c.post('/v1/systemone', json=PAYLOAD, headers={'x-request-id': 'client-42:a.b'})
        assert r.status_code == 200
        # the default effort, xhigh: two orders of three tokens, no rating
        assert set(r.json()['answers']) == {'q'}
        assert r.json()['usage'] == {'input_tokens': 6, 'output_tokens': 0, 'cached_tokens': 2, 'image_tokens': 0,
                                     'effort': 'xhigh', 'branches': 2, 'images': 0}
        assert r.headers['x-request-id'] == 'client-42:a.b'
        assert all(f'{stage};dur=' in r.headers['server-timing'] for stage in ('prepare', 'queue', 'inference'))
        unsafe = c.post('/v1/systemone', json=PAYLOAD, headers={'x-request-id': 'bad id\twith space'})
        assert len(unsafe.headers['x-request-id']) == 32


def test_authentication_protects_inference_and_admin_endpoints(monkeypatch):
    monkeypatch.setenv('JUDGE_API_KEY', 'secret')
    with TestClient(create_app(Settings.from_env(), engine=Engine(), builder=Builder())) as c:
        assert c.post('/v1/systemone', json=PAYLOAD).status_code == 401
        assert c.get('/v1/models').status_code == 401
        auth = {'Authorization': 'Bearer secret'}
        assert c.post('/v1/systemone', json={**PAYLOAD, 'model': 'any-name'}, headers=auth).status_code == 200
        listing = c.get('/v1/models', headers=auth).json()
        assert listing['object'] == 'list' and isinstance(listing['data'][0].pop('created'), int)
        assert listing['data'] == [{'id': 'gemma-4-31b-trt', 'object': 'model', 'owned_by': 'aleria',
                                    'root': 'google/gemma-4-31B-it', 'parent': None, 'max_model_len': 131072}]
        assert {'status', 'model', 'checkpoint', 'authentication', 'modalities', 'method'} <= set(c.get('/health').json())
        assert c.post('/v1/trace', json=PAYLOAD).status_code == 401
        assert c.get('/live').status_code == c.get('/metrics').status_code == 200
        assert '<html' in c.get('/').text.lower()


@pytest.mark.parametrize('change, message', [
    ({'temperature': 1}, 'Extra inputs'),
    ({'images': ['data:image/png;base64,AAAA']}, 'could not be decoded'),
    ({'images': ['data:image/png;base64,@@']}, 'not valid base64'),
    ({'questions': {}}, 'at least one question'),
    ({'questions': {'q': {'type': 'choice', 'criteria': {}}}}, '1–255 options'),
    ({'questions': {'s': {'type': 'score', 'criteria': ['low', None]}}}, 'score'),
    ({'state': 'too long'}, 'never truncated'),
])
def test_invalid_requests_explain_the_problem(change, message):
    with TestClient(app_with()) as c:
        r = c.post('/v1/systemone', json={**PAYLOAD, **change})
        assert r.status_code == 422 and message in r.text
        assert c.get('/health').status_code == 200


def test_body_limit_uses_declared_length_and_actual_size():
    with TestClient(app_with(max_body_bytes=256)) as c:
        declared = c.post('/v1/systemone', content=b'{}',
                          headers={'content-length': '100000', 'content-type': 'application/json'})
        assert declared.status_code == 413 and '256 bytes' in declared.text
        assert c.post('/v1/systemone', json={**PAYLOAD, 'state': 'x' * 500}).status_code == 413


def test_requests_above_builder_limit_are_split_and_fully_answered():
    builder = Builder()
    questions = {f'q{i}': {'type': 'choice', 'criteria': {'a': None, 'b': None}} for i in range(300)}
    with TestClient(app_with(builder=builder)) as c:
        builder.calls.clear()
        r = c.post('/v1/systemone', json={'state': 'x', 'questions': questions})
        assert r.status_code == 200 and list(r.json()['answers']) == list(questions)
        assert builder.calls == [128, 128, 44]
        assert list(builder.listed) == list(questions)  # every chunk lists all 300 questions before the state


GATED = {'split': {'type': 'choice', 'criteria': {'a': None, 'b': None, 'c': None, 'd': None}},
         'agree': {'type': 'choice', 'criteria': {'x': None, 'y': None, 'z': None}},
         'pair': {'type': 'choice', 'criteria': {'l': None, 'r': None}}}


def pick_first_shown_except_agree(branch):
    """'split' picks whichever option each order shows first, so its orders disagree; 'agree' always picks x."""
    return branch.order.index(0) if branch.question == 'agree' else 0


def test_second_pass_scores_two_more_orders_only_where_orders_disagree():
    engine, builder = Engine(pick=pick_first_shown_except_agree), Builder()
    with TestClient(app_with(engine=engine, builder=builder, default_effort='high')) as c:
        engine.passes.clear()
        r = c.post('/v1/systemone', json={'state': 'x', 'questions': GATED})
        assert r.status_code == 200 and list(r.json()['answers']) == list(GATED)
        # First pass: two orders per question plus the pair's two ratings. Second: split's n/4 and 3n/4 rotations.
        assert engine.passes == [8, 2]
        assert r.json()['usage']['input_tokens'] == 6 * 3 + 2 * 4 + 2 * 3
        assert r.json()['answers']['agree']['choice'] == 'x'
        assert list(builder.listed) == list(GATED)  # the second pass scores 'split' but lists every question of the request
        engine.passes.clear()
        assert c.post('/v1/systemone', json={'state': 'x', 'questions': {'agree': GATED['agree']}}).status_code == 200
        assert engine.passes == [2]


def test_passes_of_one_request_share_a_cache_salt_that_no_other_request_gets():
    engine, builder = Engine(pick=pick_first_shown_except_agree), Builder()
    with TestClient(app_with(engine=engine, builder=builder)) as c:
        engine.salts.clear()
        for _ in range(2):
            assert c.post('/v1/systemone', json={'state': 'x', 'questions': GATED}).status_code == 200
        first, gated, again, again_gated = engine.salts
        assert first and first == gated and again == again_gated and first != again


def test_trace_explains_the_same_answers():
    with TestClient(app_with(engine=Engine(pick=pick_first_shown_except_agree), default_effort='high')) as c:
        payload = {'state': 'x', 'questions': GATED}
        plain = c.post('/v1/systemone', json=payload).json()
        traced = c.post('/v1/trace', json=payload).json()
        assert traced['answers'] == plain['answers'] and traced['usage'] == plain['usage']
        trace = traced['trace']
        branches = trace['branches']
        assert [b['pass'] for b in branches] == [1] * 8 + [2] * 2
        assert [b['seed'] for b in branches].count(True) == 2 and branches[0]['seed'] and branches[8]['seed']
        assert trace['tokens'] == {'prompt': 32, 'cached': 16, 'computed': 16}
        assert {'queued_ms', 'start_ms', 'end_ms', 'segments', 'question_text', 'probabilities'} <= set(branches[1])
        assert [b['rated_option'] for b in branches if b['kind'] != 'answer'] == ['l', 'r']
        split, agree, pair = (trace['questions'][k] for k in GATED)
        assert split['gated'] and len(split['orders']) == 4 and not agree['gated'] and not pair['gate_eligible']
        assert agree['orders'][0]['temperature'] == pytest.approx(option_temperature(3))
        assert pair['orders'][0]['temperature'] == 1 and [r['kind'] for r in pair['ratings']] == ['rating', 'rating_rev']
        assert pair['final'] == pytest.approx([(pair['letters'][0] + pair['rating']) / 2,
                                               (pair['letters'][1] + 1 - pair['rating']) / 2])
    with TestClient(app_with(trace=False)) as c:
        assert c.post('/v1/trace', json=PAYLOAD).status_code in (404, 405)


# Admission ------------------------------------------------------------------

def test_overload_is_explicit_and_retryable():
    async def run():
        app = app_with(engine=Engine(delay=.2), max_requests=1, max_queue=0)
        async with serving(app) as c:
            first = asyncio.create_task(c.post('/v1/systemone', json=PAYLOAD))
            await until(lambda: app.state.service.gate.inflight)
            busy = await c.post('/v1/systemone', json=PAYLOAD)
            assert busy.status_code == 529 and busy.headers['retry-after'] == '1'
            assert (await first).status_code == 200
            assert idle(app)
    asyncio.run(run())


def test_invalid_request_fails_before_queueing():
    async def run():
        app = app_with(engine=Engine(delay=1), max_requests=1, max_queue=2, queue_timeout=5)
        async with serving(app) as c:
            first = asyncio.create_task(c.post('/v1/systemone', json=PAYLOAD))
            await until(lambda: app.state.service.gate.inflight)
            started = time.monotonic()
            r = await c.post('/v1/systemone', json={**PAYLOAD, 'state': 'too long'})
            assert r.status_code == 422 and time.monotonic() - started < .5
            assert (await first).status_code == 200
    asyncio.run(run())


def test_queue_deadline_returns_529():
    async def run():
        app = app_with(engine=Engine(delay=.5), max_requests=1, max_queue=1, queue_timeout=.05)
        async with serving(app) as c:
            first = asyncio.create_task(c.post('/v1/systemone', json=PAYLOAD))
            await until(lambda: app.state.service.gate.inflight)
            late = await c.post('/v1/systemone', json=PAYLOAD)
            assert late.status_code == 529 and 'deadline' in late.text
            assert (await first).status_code == 200
    asyncio.run(run())


# Deadlines, disconnects and failures ----------------------------------------

def test_inference_deadline_cancels_native_work_and_frees_the_slot():
    engine = Engine(delay=1)
    app = app_with(engine=engine, request_timeout=.02)
    with TestClient(app) as c:
        r = c.post('/v1/systemone', json=PAYLOAD)
        assert r.status_code == 504 and engine.cancelled == 1
        engine.delay = 0
        assert c.post('/v1/systemone', json=PAYLOAD).status_code == 200
        assert app.state.service.gate.inflight == 0


def test_disconnect_while_running_cancels_native_work():
    async def run():
        engine = Engine(delay=10)
        app = app_with(engine=engine)
        async with app.router.lifespan_context(app):
            incoming, outgoing = asyncio.Queue(), []
            await incoming.put({'type': 'http.request', 'body': json.dumps(PAYLOAD).encode(), 'more_body': False})
            scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'method': 'POST', 'path': '/v1/systemone',
                     'raw_path': b'/v1/systemone', 'query_string': b'', 'root_path': '', 'scheme': 'http',
                     'http_version': '1.1', 'headers': [(b'content-type', b'application/json')],
                     'server': ('test', 80), 'client': ('client', 1234)}

            async def send(message):
                outgoing.append(message)
            task = asyncio.create_task(app(scope, incoming.get, send))
            await until(lambda: app.state.service.gate.inflight)
            await incoming.put({'type': 'http.disconnect'})
            await asyncio.wait_for(task, 1)
            assert engine.cancelled == 1 and outgoing[0]['status'] == 499
            assert idle(app) and not app.state.service.failed
    asyncio.run(run())


def test_disconnect_while_queued_returns_the_slot():
    async def run():
        app = app_with(engine=Engine(delay=.5), max_requests=1, max_queue=4, queue_timeout=10)
        async with serving(app) as c:
            first = asyncio.create_task(c.post('/v1/systemone', json=PAYLOAD))
            await until(lambda: app.state.service.gate.inflight)
            queued = asyncio.create_task(c.post('/v1/systemone', json=PAYLOAD))
            await until(lambda: app.state.service.gate.waiting)
            queued.cancel()
            await asyncio.gather(queued, return_exceptions=True)
            assert (await first).status_code == 200
            app.state.service.engine.delay = 0
            assert (await c.post('/v1/systemone', json=PAYLOAD)).status_code == 200
            assert idle(app)
    asyncio.run(run())


def test_disconnect_probe_swallowing_cancellation_cannot_hold_a_finished_response(monkeypatch):
    from starlette.requests import Request

    async def probe(request):
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            return False
    monkeypatch.setattr(Request, 'is_disconnected', probe)
    app = app_with(engine=Engine(delay=.01))
    with TestClient(app) as c:
        started = time.monotonic()
        assert c.post('/v1/systemone', json=PAYLOAD).status_code == 200
        assert time.monotonic() - started < .5 and not app.state.service.failed


def test_native_failure_drains_active_requests_then_exits(monkeypatch):
    exits = []
    monkeypatch.setattr(server.os, '_exit', exits.append)

    async def run():
        engine = Engine(delay=.3)
        engine.fail_on = 'fail'
        app = app_with(engine=engine, max_requests=2, drain_timeout=5)
        app.state.service.owns_engine = True
        monkeypatch.setattr(engine, 'shutdown', lambda: None, raising=False)
        async with serving(app) as c:
            healthy = asyncio.create_task(c.post('/v1/systemone', json=PAYLOAD))
            await until(lambda: app.state.service.gate.inflight)
            bad = await c.post('/v1/systemone', json={**PAYLOAD, 'questions': {'fail': PAYLOAD['questions']['q']}})
            assert bad.status_code == 503 and bad.headers['retry-after'] == '1'
            assert (await healthy).status_code == 200  # the concurrent request still gets its answer
            late = await c.post('/v1/systemone', json=PAYLOAD)
            assert late.status_code == 503
            assert (await c.get('/health')).status_code == (await c.get('/live')).status_code == 503
            await asyncio.wait_for(app.state.service.exit_task, 3)
        assert exits == [1]
    asyncio.run(run())


def png(color, size=(32, 32)):
    import base64, io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', size, color).save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def test_images_reach_the_builder_and_the_engine_in_order():
    builder, engine = Builder(), Engine()
    with TestClient(app_with(builder=builder, engine=engine)) as c:
        r = c.post('/v1/systemone', json={**PAYLOAD, 'images': [png('red'), png('blue', (48, 16))]})
        assert r.status_code == 200 and r.json()['usage']['images'] == 2
        # each of the two branches: its text, two placeholders, and what the engine expanded each image to
        assert r.json()['usage']['input_tokens'] == 6 + 2 * 2 * IMAGE_TOKENS
        assert builder.images == 2
        assert [im.getpixel((0, 0)) for im in engine.images] == [(255, 0, 0), (0, 0, 255)]
        assert c.get('/health').json()['modalities'] == ['text', 'image']


def test_trace_counts_images_as_the_engine_expanded_them():
    with TestClient(app_with()) as c:
        trace = c.post('/v1/trace', json={**PAYLOAD, 'images': [png('red'), png('blue')]}).json()['trace']
    expansion = 2 * (IMAGE_TOKENS - 1)
    for b in trace['branches']:
        assert b['segments']['head'] == 1 and b['segments']['images'] == 2 + expansion
        assert sum(b['segments'].values()) == b['prompt_tokens'] and b['shared_prefix_tokens'] >= 3 + expansion
        assert str(IMAGE) not in b['question_text'].split()
    assert trace['tokens']['prompt'] == sum(b['prompt_tokens'] for b in trace['branches'])
    assert trace['tokens']['computed'] == trace['tokens']['prompt'] - trace['tokens']['cached'] > 0


def test_text_only_deployment_refuses_images():
    with TestClient(app_with(vision=False)) as c:
        r = c.post('/v1/systemone', json={**PAYLOAD, 'images': [png('red')]})
        assert r.status_code == 422 and 'text-only' in r.text
        assert c.get('/health').json()['modalities'] == ['text']


def test_image_format_must_match_its_data_url():
    jpeg_label = png('red').replace('image/png', 'image/jpeg')
    with TestClient(app_with()) as c:
        r = c.post('/v1/systemone', json={**PAYLOAD, 'images': [jpeg_label]})
        assert r.status_code == 422 and 'not what its data URL declares' in r.text


# Effort profiles ------------------------------------------------------------

THREE = {'state': 'A fact', 'questions': {
    'pick': {'type': 'choice', 'criteria': {'a': None, 'b': None, 'c': None, 'd': None}},
    'pair': {'type': 'choice', 'criteria': {'x': None, 'y': None}},
    'flag': {'type': 'noul'}}}


def disagree(branch):
    """The two first-pass orders pick different original options for every 3+ option question."""
    return 0 if branch.order[0] == 0 else 1


@pytest.mark.parametrize('effort, profile, branches, passes', [
    (None, 'xhigh', 8, 2),
    ('high', 'high', 2 + 2 + 2 + 2 + 2 + 2, 2),  # pick: 2 orders + 2 gated; pair and flag: 2 orders + 2 ratings
    ('medium', 'medium', 2 + 2 + 2 + 2 + 2, 1),  # no gate; both two-option questions rated in both orders
    ('low', 'low', 1 + 1 + 1, 1),                # one order per question, nothing else
    ('xhigh', 'xhigh', 2 + 2 + 2 + 2, 2),        # pick: 2 orders + 2 gated; pair and flag: 2 orders, no rating
])
def test_effort_profiles_set_branches_and_passes(effort, profile, branches, passes):
    engine = Engine(pick=disagree)
    with TestClient(app_with(engine)) as c:
        body = {**THREE, 'options': {'effort': effort}} if effort else THREE
        r = c.post('/v1/systemone', json=body)
        assert r.status_code == 200, r.text
        usage = r.json()['usage']
        assert (usage['effort'], usage['branches'], len(engine.passes) - 1) == (profile, branches, passes)
        assert set(r.json()['answers']) == {'pick', 'pair', 'flag'}


def test_effort_header_body_agreement_and_validation():
    with TestClient(app_with()) as c:
        header = c.post('/v1/systemone', json=THREE, headers={'X-Decision-Effort': 'low'})
        assert header.status_code == 200 and header.json()['usage']['effort'] == 'low'
        same = c.post('/v1/systemone', json={**THREE, 'options': {'effort': 'low'}}, headers={'X-Decision-Effort': 'low'})
        assert same.status_code == 200
        clash = c.post('/v1/systemone', json={**THREE, 'options': {'effort': 'high'}}, headers={'X-Decision-Effort': 'low'})
        assert clash.status_code == 422 and 'disagree' in clash.text
        unknown = c.post('/v1/systemone', json={**THREE, 'options': {'effort': 'maximal'}})
        assert unknown.status_code == 422 and 'Unknown effort' in unknown.text
        versioned = c.post('/v1/systemone', json={**THREE, 'options': {'effort': 'high@2'}})
        assert versioned.status_code == 422 and 'Unknown effort' in versioned.text
        extra = c.post('/v1/systemone', json={**THREE, 'options': {'effort': 'low', 'temperature': 2}})
        assert extra.status_code == 422


def test_service_default_effort_applies_only_without_a_request_choice():
    with TestClient(app_with(default_effort='low')) as c:
        assert c.post('/v1/systemone', json=THREE).json()['usage']['effort'] == 'low'
        assert c.post('/v1/systemone', json={**THREE, 'options': {'effort': 'high'}}).json()['usage']['effort'] == 'high'
        assert c.get('/health').json()['default_effort'] == 'low'


def test_xhigh_reads_the_input_twice_in_every_pass_and_once_when_twice_does_not_fit():
    builder = Builder()
    with TestClient(app_with(Engine(pick=disagree), builder)) as c:
        for state, effort, status, passes in [
                ('A fact', 'xhigh', 200, [True, True]),           # first pass, then the gated pass, same prefix
                ('fits once', 'xhigh', 200, [True, False, False]),  # twice fails the context limit: once, both passes
                ('A fact', 'high', 200, [False, False]),
                ('too long', 'xhigh', 422, [True, False])]:       # once does not fit either: inputs are never cut
            start = len(builder.twice)
            r = c.post('/v1/systemone', json={**THREE, 'state': state, 'options': {'effort': effort}})
            assert r.status_code == status, r.text
            assert builder.twice[start:] == passes


def test_low_effort_answers_at_temperature_one():
    with TestClient(app_with(Engine(peak=.9))) as c:
        probabilities = c.post('/v1/systemone', json={**THREE, 'options': {'effort': 'low'}}).json()['answers']['pick']['probabilities']
        assert probabilities['a'] == pytest.approx(.9)  # one order, no option-count temperature

"""The decision method without tokenizers or GPUs."""
import json
import math

import pytest

from aleria_judge.method import (PROFILES, Branch, combine, disagreeing, gated_orders, option_temperature, orders,
                                    rating_kinds, rating_probability, temper)

DIGITS = tuple(range(10, 19))


def test_two_orders_rotate_by_half_and_the_gate_adds_the_quarter_rotations():
    assert orders(5, 2) == [(0, 1, 2, 3, 4), (2, 3, 4, 0, 1)]
    assert orders(2, 2) == [(0, 1), (1, 0)] and orders(1, 2) == [(0,)]
    assert gated_orders(5) == [(1, 2, 3, 4, 0), (3, 4, 0, 1, 2)]
    assert gated_orders(3) == [(2, 0, 1)]  # n // 4 coincides with the original order
    assert gated_orders(2) == gated_orders(1) == []
    assert sorted(orders(8, 2) + gated_orders(8)) == sorted(orders(8, 4))


def test_every_two_option_question_is_rated_in_both_orders_when_the_profile_rates():
    assert rating_kinds(2) == rating_kinds(2, PROFILES['medium']) == ['rating', 'rating_rev']
    assert rating_kinds(1) == rating_kinds(3) == rating_kinds(2, PROFILES['low']) == rating_kinds(2, PROFILES['xhigh']) == []


def test_xhigh_is_high_without_the_rating_reading_the_input_twice():
    high, xhigh = PROFILES['high'], PROFILES['xhigh']
    assert (xhigh.orders, xhigh.gated, xhigh.temperature) == (high.orders, high.gated, high.temperature)
    assert not xhigh.rating and xhigh.read_twice
    assert [name for name, p in PROFILES.items() if p.read_twice] == ['xhigh']
    assert xhigh.summary()['read_twice'] and not high.summary()['read_twice']


def test_option_count_temperature_curve():
    assert option_temperature(2) == pytest.approx(6.9 - 0.7 * math.log(2))
    assert option_temperature(1) == option_temperature(2)
    assert option_temperature(255) == pytest.approx(6.9 - 0.7 * math.log(255))


def test_rating_probability_maps_expected_digit_and_clamps():
    assert rating_probability([0] * 4 + [1] + [0] * 4) == pytest.approx(.01 + .4 * .98 / .8)  # rating 5
    assert rating_probability([1] + [0] * 8) == .01 and rating_probability([0] * 8 + [1]) == .99


def test_temperature_is_softmax_of_scaled_log_probabilities():
    assert temper([.9, .1], 1) == pytest.approx([.9, .1])
    soft = temper([.99, .009, .001], 4)
    expected = [math.exp(math.log(p) / 4) for p in (.99, .009, .001)]
    assert soft == pytest.approx([v / sum(expected) for v in expected]) and soft.index(max(soft)) == 0


def test_disagreeing_needs_three_options_and_two_different_picks():
    branches = [Branch('a', [1], (1, 2, 3), (0, 1, 2)), Branch('a', [1], (1, 2, 3), (1, 2, 0)),
                Branch('b', [1], (1, 2, 3), (0, 1, 2)), Branch('b', [1], (1, 2, 3), (1, 2, 0)),
                Branch('c', [1], (1, 2), (0, 1)), Branch('c', [1], (1, 2), (1, 0)),
                Branch('c', [1], DIGITS, tuple(range(9)), 'rating')]
    rows = [[.8, .1, .1], [.8, .1, .1],      # a: original 0 vs original 1 -> disagree
            [.8, .1, .1], [.1, .1, .8],      # b: original 0 vs original 0 -> agree
            [.9, .1], [.9, .1], [1 / 9] * 9]  # c: two options are never gated
    assert disagreeing(branches, rows) == ['a']


def test_orders_are_mapped_back_and_averaged():
    questions = {'q': {'type': 'choice', 'criteria': {'a': None, 'b': None, 'c': None}}}
    branches = [Branch('q', [1], (1, 2, 3), (0, 1, 2)), Branch('q', [2], (1, 2, 3), (1, 2, 0))]
    steps = {}
    out = combine(questions, branches, [[.7, .2, .1], [.6, .3, .1]], steps)['answers']['q']
    t = option_temperature(3)
    first, second = temper([.7, .2, .1], t), temper([.6, .3, .1], t)
    expected = [(first[0] + second[2]) / 2, (first[1] + second[0]) / 2, (first[2] + second[1]) / 2]
    assert list(out['probabilities'].values()) == pytest.approx(expected) and out['choice'] == 'b'
    assert [o['pick'] for o in steps['q']['orders']] == [0, 1] and steps['q']['final'] == pytest.approx(expected)


def test_noul_rating_is_averaged_with_untempered_letters():
    questions = {'yes': {'type': 'noul'}}
    branches = [Branch('yes', [1], (1, 2), (0, 1)), Branch('yes', [2], (1, 2), (1, 0)),
                Branch('yes', [3], DIGITS, tuple(range(9)), 'rating'),
                Branch('yes', [4], DIGITS, tuple(range(9)), 'rating_rev')]
    # orders: false .9 and .8; ratings: (A) = false at .99, then (A) = true at .01
    rows = [[.9, .1], [.2, .8], [0] * 8 + [1], [1] + [0] * 8]
    steps = {}
    out = combine(questions, branches, rows, steps)['answers']['yes']
    assert out['noul'] == pytest.approx((.15 + (1 - .99)) / 2)
    assert [o['temperature'] for o in steps['yes']['orders']] == [1, 1] and steps['yes']['rating'] == pytest.approx(.99)


def test_pair_rating_averages_both_orders():
    questions = {'p': {'type': 'choice', 'criteria': {'left': None, 'right': None}}}
    answer = [Branch('p', [1], (1, 2), (0, 1)), Branch('p', [2], (1, 2), (1, 0))]
    fwd = Branch('p', [3], DIGITS, tuple(range(9)), 'rating')
    rev = Branch('p', [4], DIGITS, tuple(range(9)), 'rating_rev')
    rows = [[.5, .5], [.5, .5], [0] * 8 + [1], [0] * 8 + [1]]  # both ratings say "(A) almost certainly"
    out = combine(questions, answer + [fwd, rev], rows)['answers']['p']
    assert out['probabilities']['left'] == pytest.approx(.5)   # forward .99 for left, reversed .99 for right


def test_score_answer_reports_expected_level_and_legend():
    questions = {'s': {'type': 'score', 'criteria': ['low', 'mid', 'high']}}
    branches = [Branch('s', [1], (1, 2, 3), (0, 1, 2)), Branch('s', [2], (1, 2, 3), (1, 2, 0))]
    out = combine(questions, branches, [[0., 0., 1.], [0., 1., 0.]])['answers']['s']
    assert out['legend'] == {'0': 'low', '1': 'mid', '2': 'high'} and out['probabilities']['2'] > .5
    assert out['score'] == pytest.approx(sum(int(k) * p for k, p in out['probabilities'].items()))


def test_option_names_that_repeat_a_label_are_dropped():
    """A name equal to one of the question's labels would contradict it after rotation, e.g. "(A) B: no"."""
    from aleria_judge.vendor import systemone
    render = lambda criteria, labels='AB': systemone.render_question(
        {'type': 'choice', 'instructions': 'Which?', 'criteria': criteria}, tuple(labels))['options']
    assert render({'A': 'yes', 'B': 'no'}) == ['yes', 'no']
    assert render({'(A)': 'first', '(B)': 'second'}) == ['first', 'second']
    assert render({'refund': 'Money back', 'resend': 'Ship again'}) == ['refund: Money back', 'resend: Ship again']
    assert render({'A': None, 'B': ''}) == ['A', 'B']                      # no description: the name stays
    assert render({'a': 'yes', 'b': 'no'}) == ['a: yes', 'b: no']           # exact match only
    assert render({'C': 'x', 'D': 'y'}) == ['C: x', 'D: y']                 # not a label of this question
    assert render({'A': 'same', 'B': 'same'}) == ['A: same', 'B: same']     # repeated descriptions keep names
    assert render({'A': 'yes', 'other': 'no'}) == ['yes', 'other: no']
    assert render({'Negative': 'Negative', 'Positive': 'Upbeat'}) == ['Negative', 'Positive: Upbeat']   # exact copy: once
    assert render({'key': 'key.     ', 'Key': 'key', 'k': ' k'}, 'ABC') == ['key: key.     ', 'Key: key', 'k:  k']
    assert render({'A': 'Neutral', 'Neutral': 'Neutral'}) == ['A: Neutral', 'Neutral: Neutral']  # alike: both in full
    assert systemone.render_question({'type': 'noul', 'instructions': 'Ok?', 'criteria': {'true': 'fine'}})['options'] \
        == ['no', 'yes: fine']


def test_json_states_are_indented_and_sent_unchanged():
    """No fields are added to a client's state (the former "_index" tags could collide with its own keys)."""
    from aleria_judge.vendor import systemone
    state = {'items': [{'_index': 'client value', 'n': i} for i in range(10)], 'note': 'café'}
    assert systemone.render_state(state) == json.dumps(state, ensure_ascii=False, indent=2)
    assert systemone.render_state('plain text') == 'plain text'

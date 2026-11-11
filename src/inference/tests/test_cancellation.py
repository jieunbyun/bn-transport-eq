"""Cancellation reaches real sampling and scoring loops without leaking into later computations."""

import threading

import pytest

from src.builder.bn_builder import build_tbn
from src.cancellation import InferenceCancelled, cancellable, check_cancelled, report_progress
from src.inference import cliques, engine
from src.inference.session import Session


@pytest.fixture
def toy_model():
    variables, probs = build_tbn(engine.compile_query({"assembly": "toy-ca/demo"}))
    return cliques.tree(probs), probs, variables


def test_cancellation_context_is_reset_and_does_not_cancel_other_threads():
    event = threading.Event()
    checked = []
    with cancellable(event):
        event.set()
        with pytest.raises(InferenceCancelled):
            check_cancelled()
        worker = threading.Thread(target=lambda: (check_cancelled(), checked.append(True)))
        worker.start()
        worker.join()
    check_cancelled()
    assert checked == [True]


def test_forward_sampling_stops_between_bridges(toy_model, monkeypatch):
    t, _, _ = toy_model
    event = threading.Event()
    original_draw = cliques.field_draw

    def draw_then_stop(*args, **kwargs):
        result = original_draw(*args, **kwargs)
        event.set()
        return result

    monkeypatch.setattr(cliques, "field_draw", draw_then_stop)
    with cancellable(event), pytest.raises(InferenceCancelled):
        cliques.sample(t, {}, n_chain=20, n_keep=2)


def test_inspection_scoring_stops_before_evaluating_candidates(toy_model, monkeypatch):
    t, probs, variables = toy_model
    session = Session(t, probs, variables, sampler={"n_chain": 20, "n_keep": 2})
    event = threading.Event()
    target = session._target

    def target_then_stop(*args, **kwargs):
        result = target(*args, **kwargs)
        event.set()
        return result

    monkeypatch.setattr(session, "_target", target_then_stop)
    with cancellable(event), pytest.raises(InferenceCancelled):
        session._values({"target": {"kind": "cut_off"}, "decision": "inspect"})


def test_an_observation_says_it_reweights_and_why_it_resamples(toy_model):
    # A likely edge state keeps most of the weight: reweighted only. A town cut off is rare: too little weight is
    # left, and the message gives the effective sample size against the rows before resampling.
    t, probs, variables = toy_model
    session = Session(t, probs, variables, sampler={"n_chain": 20, "n_burn": 5, "n_keep": 50})
    said = []
    with cancellable(threading.Event(), said.append):
        session.observe("EdgeOpen::e6", 1)
    assert said == ["Reweighting…"]
    said.clear()
    with cancellable(threading.Event(), said.append):
        entry = session.observe("SystemState::n4", 0)
    assert entry["update"] == "sampled again"
    assert said[0] == "Reweighting…"
    assert said[1] == f"ESS {entry['ess'] * 1000:,.0f} / 1,000. Too low, so resampling…"
    report_progress("outside any computation: nowhere to go")    # and no error

"""Queries are validated, interventions applied, only the needed bindings run, refusals explained."""

from __future__ import annotations

from math import log, sqrt

import pytest
from scipy.special import ndtr

from src.builder.kernels.hazard import campbell_2003_ln_median, great_circle_km
from src.inference.engine import QueryError, answer, compile_query, describe, library, statement

EMA = "ema-highway/gmm"
SHAKEMAP = "hebei-transport/shakemap"


def _epicentre_at(bridge_id: str) -> dict:
    """A rows intervention putting the scenario's epicentre on one bridge, so its shaking is strong."""
    bridge = compile_query({"assembly": EMA}).network.row(bridge_id)
    return {"ema-reference": {"lat": bridge["lat"], "lon": bridge["lon"]}}


def test_event_probability_matches_the_closed_form():
    """GMM then fragility: ln SA(1.0) ~ N(ln median, sigma^2), so P(DS >= extensive) = Phi((ln median - ln m) / sqrt(sigma^2 + beta^2))."""
    rows = _epicentre_at("bridge-e0001")
    network = compile_query({"assembly": EMA, "rows": rows}).network
    event, bridge = network.row("ema-reference"), network.row("bridge-e0001")
    gmm = library().parameter_sets["campbell-2003.ema-script-sa1"]["values"]
    r = sqrt(float(great_circle_km(event["lat"], event["lon"], bridge["lat"], bridge["lon"])) ** 2 + event["depth_km"] ** 2)
    ln_median = float(campbell_2003_ln_median(event["mw"], r, gmm["coefficients"]))
    s = gmm["sigma"]
    sigma = sqrt((s["a"] + s["b"] * event["mw"] if event["mw"] < s["m_break"] else s["sigma_above"]) ** 2
                 + gmm["sigma_epistemic"] ** 2)
    curve = library().parameter_sets[f"hazus61.{bridge['hazus_class']}"]["values"]["parameters"]
    p = float(ndtr((ln_median - log(curve["medians"]["extensive"])) / sqrt(sigma ** 2 + curve["beta"] ** 2)))

    result = answer({"assembly": EMA, "rows": rows, "samples": 20000, "seed": 1,
                     "events": [{"node": "EdgeOpen::e0001", "op": "==", "value": 0}]})
    (estimate,) = result["events"]
    assert 0.05 < p < 0.95                                   # a case the test can actually tell apart
    assert estimate["ci95"][0] <= p <= estimate["ci95"][1]


def test_only_the_bindings_a_target_needs_are_run():
    r = answer({"assembly": EMA, "targets": ["EdgeOpen::e0001"], "samples": 100})
    assert r["computed"] == ["hazard.gmm", "fragility.bridge", "edge.open"]


def test_patterns_expand_and_discrete_summaries_sum_to_one():
    r = answer({"assembly": EMA, "targets": ["EdgeOpen::*"], "samples": 200})
    assert len(r["targets"]) == 129
    assert all(abs(sum(t["summary"]["P"].values()) - 1) < 1e-9 for t in r["targets"])


def test_interventions_change_the_answer_in_the_right_direction():
    near = _epicentre_at("bridge-e0001")

    def p_closed(rows=None, **do):
        r = answer({"assembly": EMA, "rows": {**near, **(rows or {})}, "samples": 4000, "seed": 2, **do,
                    "events": [{"node": "EdgeOpen::e0001", "op": "==", "value": 0}]})
        return r["events"][0]["p"]

    base = p_closed()
    assert p_closed(rows={"ema-reference": {**near["ema-reference"], "mw": 6.5}}) < base
    assert p_closed(parameter_sets={"edge.open": "policy.closed-at-moderate"}) > base
    assert p_closed(rows={"bridge-e0001": {"hazus_class": "HWB5"}}) > base   # Extensive at 0.45 g, not 1.1 g


def test_statement_reads_back_the_query():
    s = statement({"assembly": SHAKEMAP, "events": [{"node": "SystemState::n1", "op": "==", "value": 0}],
                   "parameters": {"hazard.shakemap": {"event_id": "us7000abcd"}}})
    assert s.startswith("P(SystemState[n1] = 0) | do(hazard.shakemap.event_id = us7000abcd)")


@pytest.mark.parametrize("query, message", [
    ({"assembly": EMA, "targets": ["SA(1.0)::*"], "evidence": {"SA(1.0)::bridge-e0001": .5}}, "continuous observations"),
    ({"assembly": EMA, "targets": ["Nope::*"]}, "no node matches 'Nope::\\*'"),
    ({"assembly": EMA}, "query"),
    ({"assembly": "nowhere/x", "targets": ["SA(1.0)::*"]}, "no assembly 'nowhere/x'"),
    ({"assembly": EMA, "targets": ["SA(1.0)::*"], "rows": {"bridge-e0001": {"height": 3}}}, "no settable attribute 'height'"),
    ({"assembly": EMA, "targets": ["SA(1.0)::*"], "rows": {"bridge-e0001": {"hazus_class": "HWB99"}}}, "HWB99"),
    ({"assembly": EMA, "targets": ["SA(1.0)::*"], "parameters": {"fragility.bridge": {"beta": 1}}},
     "parameter set"),
    ({"assembly": SHAKEMAP, "targets": ["SA(1.0)::*"], "parameters": {"hazard.shakemap": {"event_id": ""}}},
     "event_id"),
    ({"assembly": EMA, "events": [{"node": "EdgeOpen::e0001", "op": "==", "value": "ajar"}]}, "not a state"),
    ({"assembly": EMA, "targets": ["SA(1.0)::*"], "samples": 5}, "minimum"),
])
def test_bad_queries_are_refused_with_a_reason(query, message):
    with pytest.raises(QueryError, match=message):
        answer(query)


def test_describe_lists_what_can_be_asked_and_set():
    menu = describe(EMA)
    assert {q["id"] for q in menu["quantities"]} >= {"SA(1.0)", "EdgeOpen"}
    assert {(r["row"], r["attribute"]) for r in menu["settable"]["rows"]} >= {("ema-reference", "mw")}
    choices = {c["function"]: c for c in menu["settable"]["parameter_sets"]}
    assert len(choices["edge.open"]["choices"]) == 2
    gmm = choices["hazard.gmm"]                       # chosen by location, still overridable by a query
    assert gmm["current"] is None and gmm["by_location"] == {"campbell-2003.ema-script-sa1": 129}
    shakemap = describe(SHAKEMAP)["settable"]
    assert "event_id" in {p["name"] for p in shakemap["parameters"]}
    correlation = {c["function"]: c for c in shakemap["parameter_sets"]}["spatial-correlation"]
    assert correlation["current"] == "jayaram-baker-2009.sa1p0" and len(correlation["choices"]) == 5

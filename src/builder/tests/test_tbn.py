"""Checks of the generated toy-ca model against independent calculations."""

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import stats
from tbnpy import inference

from src.builder.bn_builder import (
    _network_inline, build_tbn, compile_path, generate_tbn, load_generated, system_nodes, tbn_data,
)
from src.builder.load import assembly_path
from src.inference.engine import compile_query

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "examples/toy-ca/tbnpy/demo"


def _toy_tables():
    """The toy-ca network's rows, as written in its network.json."""
    return json.loads((ROOT / "examples/toy-ca/network.json").read_text(encoding="utf-8"))["tables"]


@pytest.fixture
def model():
    data = json.loads((MODEL / "model.json").read_text(encoding="utf-8"))
    variables, probs = load_generated(MODEL)
    return data, variables, probs


def test_generated_snapshot_matches_sources_and_owns_joint_hazard(model):
    data, variables, probs = model
    expected = tbn_data(compile_path(assembly_path("toy-ca/demo")))
    actual = _network_inline(data, MODEL)
    for key in ("source_sha256", "versions"):
        actual.pop(key)
    for node in system_nodes(actual):
        node["kwargs"].pop("rules", None)
    assert actual == expected
    owned = [v.name for p in probs.values() for v in p.childs]
    assert len(owned) == len(set(owned)) == len(variables) == 46
    hazard = [p for p in probs.values() if p.spec_metadata["function"] == "hazard.gmm"]
    assert len(hazard) == 1
    assert len(hazard[0].childs) == 10
    assert len(hazard[0].parents) == 0                 # the event is fixed data, not a parent
    assert set(inference.get_ancestor_order(probs, list(probs))) == set(probs)


def test_model_json_is_a_tbnpy_spec_that_tbnpy_compiles_alone():
    """No repository loader needed: tbnpy builds every object from the spec and the files it names."""
    import tbnpy
    from tbnpy.compile import compile_file

    spec, varis, probs = compile_file(MODEL / "model.json")
    assert {n.type for n in spec.nodes} == {"custom"}
    assert list(probs) == list(load_generated(MODEL)[1])
    assert all(p.rules is not None for p in probs.values() if p.spec_metadata["role"] == "system")
    assert tbnpy.check_model(probs, n_sample=32).ok


def test_the_gmm_is_written_once_and_the_network_once():
    text = (MODEL / "model.json").read_text(encoding="utf-8")
    assert text.count('"c1"') == 1                     # the coefficients once, for every site
    data = json.loads(text)
    assert {n["kwargs"]["network"] for n in system_nodes(data)} == {"system-network.json"}


def test_each_parameter_set_is_written_once_and_named_by_its_nodes(model):
    data, _, probs = model
    assert sorted(data["parameter_sets"]) == ["demo.campbell-2003-in-california", "hazus61.HWB11",
                                              "hazus61.HWB5", "policy.closed-at-extensive",
                                              "relative-sd-10pct.sa1p0"]
    bridges = [n for n in data["nodes"] if n["metadata"]["function"] == "fragility.bridge"]
    assert len(bridges) == 10 and all("kwargs" not in n for n in bridges)   # the curve only in its set
    assert sum(n["parameter_set"] == "hazus61.HWB5" for n in bridges) == 3
    assert probs["DamageState.HazusBridge::b2"].spec.parameter_set == "hazus61.HWB5"
    assert data["parameter_sets"]["hazus61.HWB5"]["medians"] == probs["DamageState.HazusBridge::b2"].medians.tolist()


def test_every_object_is_a_tbnpy_probability_that_reads_as_its_distribution(model):
    import tbnpy

    _, _, probs = model
    assert all(isinstance(p, tbnpy.Probability) for p in probs.values())
    assert repr(probs["DamageState.HazusBridge::b4"]).startswith(
        "Fragility(P(DamageState.HazusBridge::b4 | SA(1.0)::b4): P(DS >= k) = Phi(ln(SA(1.0)::b4 / median_k)")
    assert repr(probs["EdgeOpen::e4"]) == ("EdgeOpen(P(EdgeOpen::e4 | DamageState.HazusBridge::b4): True if "
                                           "DamageState.HazusBridge::b4 is none, slight, moderate, else False "
                                           "[policy.closed-at-extensive])")
    assert "object at 0x" not in repr(probs)
    assert "rules: " in str(probs["SystemState::n4"])
    assert "note: hazard.gmm draws each Bridge's ln IM independently" in str(probs["hazard.gmm"])
    assert tbnpy.check_model(probs, n_sample=64).ok


def test_samples_and_log_probabilities_agree(model):
    _, _, probs = model
    torch.manual_seed(19)
    results = inference.sample(probs, list(probs), n_sample=100, batch_size=37)
    values = {v.name: results[key].Cs[:, j] for key, p in probs.items() for j, v in enumerate(p.childs)}
    for key, prob in probs.items():
        full = torch.stack([values[v.name] for v in prob.childs + prob.parents], 1)
        logp = prob.log_prob(full)
        assert logp.shape == (100,)
        assert torch.isfinite(logp).all()
        torch.testing.assert_close(logp, results[key].ps, check_dtype=False)
        if prob.spec_metadata["function"] not in ("hazard.gmm", "fragility.bridge", "observation.intensity"):
            invalid = full.clone()
            invalid[:, 0] = -1
            assert torch.isneginf(prob.log_prob(invalid)).all()


# ---------------------------------------------------------------------------
# Each kind of object against an independent calculation from the raw inputs
# ---------------------------------------------------------------------------
def test_gmm_matches_campbell_2003_computed_by_hand(model):
    """ln SA = c1 + c2 M + c3 (8.5 - M)^2 + c4 ln R + (c5 + c6 M) r + f3(r),  R = sqrt(r^2 + (c7 exp(c8 M))^2),
    r the hypocentral distance; total sigma^2 = sigma_aleatory^2 + sigma_epistemic^2; sites independent."""
    _, _, probs = model
    tables = _toy_tables()
    (event,) = tables["Event"]
    gmm = json.loads((ROOT / "instances/hazard/demo-gmm.json").read_text(encoding="utf-8"))["sets"][0]["values"]
    c, s, m = gmm["coefficients"], gmm["sigma"], event["mw"]

    lat0, lon0 = np.radians(event["lat"]), np.radians(event["lon"])
    lat, lon = np.radians([b["lat"] for b in tables["Bridge"]]), np.radians([b["lon"] for b in tables["Bridge"]])
    surface_km = 2 * 6371.0 * np.arcsin(np.sqrt(np.sin((lat - lat0) / 2) ** 2
                                                + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2))
    r = np.hypot(surface_km, event["depth_km"])
    f3 = np.where(r <= 70, 0.0, c["c7"] * np.log(r / 70) + np.where(r <= 130, 0.0, c["c8"] * np.log(r / 130)))
    ln_median = (c["c1"] + c["c2"] * m + c["c3"] * (8.5 - m) ** 2
                 + c["c4"] * np.log(np.hypot(r, c["c7"] * np.exp(c["c8"] * m))) + (c["c5"] + c["c6"] * m) * r + f3)
    sigma_aleatory = s["a"] + s["b"] * m if m < s["m_break"] else s["sigma_above"]
    sigma = np.hypot(sigma_aleatory, gmm["sigma_epistemic"])

    hazard = probs["hazard.gmm"]
    assert [v.name for v in hazard.childs] == [f"SA(1.0)::{b['id']}" for b in tables["Bridge"]]
    im = np.exp(ln_median) * np.array([[0.5], [1.0], [3.0]])           # three fields, each over the ten sites
    expected = stats.lognorm.logpdf(im, s=sigma, scale=np.exp(ln_median)).sum(axis=1)
    np.testing.assert_allclose(hazard.log_prob(torch.tensor(im)).numpy(), expected, rtol=1e-10)


@pytest.mark.parametrize("bridge", ["b1", "b2"])                       # an HWB11 and an HWB5 bridge
def test_bridge_damage_matches_its_hazus_curve(model, bridge):
    """P(DS >= k | sa) = Phi(ln(sa / median_k) / beta), and P(DS = k) the difference of neighbouring exceedances."""
    _, _, probs = model
    hazus_class = next(b["hazus_class"] for b in _toy_tables()["Bridge"] if b["id"] == bridge)
    sets = json.loads((ROOT / "instances/fragility/hazus61-bridge-sa1.json").read_text(encoding="utf-8"))["sets"]
    curve = next(s for s in sets if s["id"] == f"hazus61.{hazus_class}")["values"]["parameters"]
    medians = np.array([curve["medians"][k] for k in ("slight", "moderate", "extensive", "complete")])

    sa = np.array([0.05, 0.3, 0.9, 1.4, 4.0])
    exceed = stats.norm.cdf(np.log(sa[:, None] / medians) / curve["beta"])                   # (5 sa, 4)
    expected = -np.diff(np.hstack([np.ones((5, 1)), exceed, np.zeros((5, 1))]), axis=1)    # (5 sa, 5 states)

    fragility = probs[f"DamageState.HazusBridge::{bridge}"]
    for k in range(5):
        Cs = torch.tensor(np.column_stack([np.full(5, k), sa]))
        np.testing.assert_allclose(fragility.log_prob(Cs).exp().numpy(), expected[:, k], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("edge", ["e1", "e2"])
def test_edge_is_open_exactly_when_its_bridges_are_usable(model, edge):
    """Under the assembly's policy (closed-at-extensive), an edge is True (open) iff every bridge on it is usable."""
    _, _, probs = model
    bridges = [b["id"] for b in _toy_tables()["Bridge"] if b["edge"] == edge]
    policies = json.loads((ROOT / "instances/edge/closure-policies.json").read_text(encoding="utf-8"))["sets"]
    usable = next(p for p in policies if p["id"] == "policy.closed-at-extensive")["values"]["usable_states"]

    opening = probs[f"EdgeOpen::{edge}"]
    assert [v.name for v in opening.parents] == [f"DamageState.HazusBridge::{b}" for b in bridges]
    damage_states = opening.parents[0].values                          # the labels of the state indices
    for damage in itertools.product(range(len(damage_states)), repeat=len(bridges)):
        is_open = all(damage_states[d] in usable for d in damage)
        for value in (0, 1):                                           # False, True
            logp = opening.log_prob(torch.tensor([[value, *damage]], dtype=torch.float64))
            assert logp.item() == (0.0 if value == is_open else -np.inf)


@pytest.mark.parametrize("origin", ["n4", "n7"])
def test_system_failure_matches_enumeration_over_every_edge_state(model, origin):
    """With every bridge's SA fixed (0.5 g), the edges are independent, so P(Fail) is exact as a sum over all
    2^10 edge states x:  P(x) = prod_i p_i^x_i (1 - p_i)^(1 - x_i),  p_i = P(edge i open) = P(DS_i < extensive | SA_i)
    from the Hazus curve, and S(x) from the system object (its rsr rules, then ndtools).
    tbnpy's sampled P(Fail) under the same fixed SA must agree within its Monte Carlo error."""
    from src.inference.interventions import Constant

    _, _, probs = model
    tables = _toy_tables()
    sets = json.loads((ROOT / "instances/fragility/hazus61-bridge-sa1.json").read_text(encoding="utf-8"))["sets"]
    curves = {s["id"]: s["values"]["parameters"] for s in sets}
    sa = 0.5

    system = probs[f"SystemState::{origin}"]
    p_open = []
    for edge in system.parents:                                          # EdgeOpen::e*, in the parents' order
        (bridge,) = [b for b in tables["Bridge"] if b["edge"] == edge.name.split("::")[1]]
        curve = curves[f"hazus61.{bridge['hazus_class']}"]
        p_open.append(1 - stats.norm.cdf(np.log(sa / curve["medians"]["extensive"]) / curve["beta"]))
    p_open = np.array(p_open)

    x = np.array(list(itertools.product((0, 1), repeat=len(p_open))))                  # (1024, 10) edge states
    p_x = np.prod(np.where(x == 1, p_open, 1 - p_open), axis=1)
    fails = system.evaluate(torch.tensor(x, dtype=torch.float64))[:, 0].numpy() == 0
    exact = p_x[fails].sum()

    hazard = probs["hazard.gmm"]
    fixed = {**probs, "hazard.gmm": Constant(hazard.childs, [sa] * len(hazard.childs))}    # do(SA = 0.5 g)
    n = 20_000
    torch.manual_seed(11)
    S = inference.sample(fixed, [f"SystemState::{origin}"], n_sample=n)[f"SystemState::{origin}"].Cs[:, 0]
    estimate = (S == 0).double().mean().item()
    assert abs(estimate - exact) < 4 * np.sqrt(exact * (1 - exact) / n)


def test_system_rules_match_ndtools_for_every_edge_configuration(model):
    _, _, probs = model
    states = [p for p in probs.values() if p.spec_metadata["role"] == "system"]
    assert len(states) == 6 and all(len(s.childs) == 1 and s.rules is not None for s in states)
    for state in states:                  # one object per origin, each with its own rules
        edges = list(itertools.product((0, 1), repeat=len(state.parents)))
        pars = torch.tensor(edges, dtype=torch.float64)
        classified = state.evaluate(pars)
        evaluate = state.system_function()
        names = [v.name.split("::", 1)[1] for v in state.parents]
        expected = [evaluate(dict(zip(names, row))) for row in edges]
        torch.testing.assert_close(classified[:, 0], torch.tensor([e[1] for e in expected], dtype=torch.float64))
        # Force every state to be unclassified: the ndtools fallback must give identical results.
        state.rules = (torch.empty((0, len(state.parents), 2), dtype=torch.int32),) * 2
        torch.testing.assert_close(state.evaluate(pars), classified)


def test_every_child_outside_the_hazard_field_is_its_own_object(model):
    _, _, probs = model
    for key, prob in probs.items():
        if prob.spec_metadata["function"] == "hazard.gmm":
            assert len(prob.childs) == 10                     # the field: every bridge's intensity together
        else:
            assert [v.name for v in prob.childs] == [key]     # one child, keyed by it
    assert [v.name for v in probs["DamageState.HazusBridge::b4"].parents] == ["SA(1.0)::b4"]
    assert [v.name for v in probs["EdgeOpen::e4"].parents] == ["DamageState.HazusBridge::b4"]   # no Usable layer


@pytest.fixture
def stored_at_network_threshold(tmp_path, monkeypatch):
    """The toy-ca rules as the repository ships them, at the network's threshold only, whatever else
    the app may have extracted since: a private copy the model directory points at."""
    import shutil

    from src.builder import bn_builder

    model = Path(__file__).resolve().parents[3] / "examples/toy-ca/tbnpy/demo"
    copy = tmp_path / "model"
    for folder in (model / "rsr").glob("*/target_add_h=0.5"):
        shutil.copytree(folder, copy / "rsr" / folder.parent.name / folder.name)
    monkeypatch.setattr(bn_builder, "model_dir", lambda assembly_file: copy)
    return copy


def _sources(query):
    from src.builder.bn_builder import rules_source

    _, probs = build_tbn(compile_query(query))
    systems = [p for p in probs.values() if p.spec_metadata["role"] == "system"]
    return {p.origin: rules_source(p.rules_reference if p.rules is not None else None, p.target_add_h)
            for p in systems}


def test_the_engine_reuses_stored_rules_under_another_scenario(stored_at_network_threshold):
    """Rules depend on the network and the system event, not the hazard: a changed Event keeps them."""
    for rows in ({}, {"demo-san-andreas": {"mw": 7.0}}):
        assert set(_sources({"assembly": "toy-ca/demo", "rows": rows}).values()) == {"exact"}


def test_rules_from_another_threshold_are_reused_where_they_still_hold(stored_at_network_threshold):
    # Stricter (0.25 h): failing within 0.5 h means failing within 0.25 h, so only the failure rules hold.
    stricter = _sources({"assembly": "toy-ca/demo", "rows": {"reach-key-destination": {"target_add_h": 0.25}}})
    assert set(stricter.values()) == {"failure rules from target_add_h=0.5"}
    # Looser (1 h): surviving within 0.5 h means surviving within 1 h, so only the survival rules hold.
    looser = _sources({"assembly": "toy-ca/demo", "rows": {"reach-key-destination": {"target_add_h": 1.0}}})
    assert set(looser.values()) == {"survival rules from target_add_h=0.5"}


def test_choose_rules_takes_the_nearest_threshold_on_each_side():
    from src.builder.bn_builder import choose_rules

    stored = [(0.25, Path("a")), (0.5, Path("b")), (1.0, Path("c"))]
    ref = choose_rules(stored, 0.75)
    assert (ref["upper_at"], ref["lower_at"]) == (0.5, 1.0)              # survival from below, failure from above
    assert choose_rules(stored, 0.5)["upper_at"] == choose_rules(stored, 0.5)["lower_at"] == 0.5
    assert choose_rules(stored, 2.0)["lower"] is None and choose_rules([], 0.5) is None


def test_borrowed_rules_give_the_answer_ndtools_gives_and_report_what_is_left(stored_at_network_threshold, monkeypatch):
    from src.builder import bn_builder
    from src.inference import engine

    query = {"assembly": "toy-ca/demo", "events": [{"node": "SystemState::*", "op": "==", "value": 0}],
             "samples": 4000, "seed": 3, "rows": {"reach-key-destination": {"target_add_h": 0.25}}}
    borrowed = engine.answer(query)
    assert borrowed["rules"]["target_add_h"] == 0.25 and borrowed["rules"]["extract"]       # mostly unclassified
    monkeypatch.setattr(bn_builder, "attach_rules", lambda data, directory: 0)
    assert [e["p"] for e in engine.answer(query)["events"]] == [e["p"] for e in borrowed["events"]]


def test_extracted_rules_are_stored_by_threshold_and_then_used(tmp_path, monkeypatch):
    from src.builder import bn_builder

    compiled = compile_query({"assembly": "toy-ca/demo", "rows": {"reach-key-destination": {"target_add_h": 0.25}}})
    assert bn_builder.extract_system_rules(compiled, tmp_path, {"n4"}, log=lambda _: None) == ["n4"]
    assert (tmp_path / "rsr" / "n4" / "target_add_h=0.25" / "refs_up_1.pt").is_file()
    data = bn_builder.tbn_data(compiled)
    assert bn_builder.attach_rules(data, tmp_path) == 1                  # n4 only: the others have none stored here
    n4 = next(n for n in bn_builder.system_nodes(data) if n["kwargs"]["origin"] == "n4")
    assert bn_builder.rules_source(n4["kwargs"]["rules"], 0.25) == "exact"


def test_generation_without_rules_is_importable_and_rejects_stale_rules(tmp_path):
    assembly = assembly_path("toy-ca/demo")
    (tmp_path / "toy-ca").mkdir()
    local = tmp_path / "toy-ca" / assembly.name
    local.write_text(assembly.read_text(encoding="utf-8"), encoding="utf-8")
    # Generated files normally sit inside the repo. For this import check,
    # retain their documented directory structure beneath a temporary root.
    (tmp_path / "src/builder").mkdir(parents=True)
    (tmp_path / "src/builder/bn_builder.py").touch()
    out = generate_tbn(local, extract_rules=False)
    spec = importlib.util.spec_from_file_location("generated_toy", out / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    variables, probs = module.define_model()
    assert len(inference.sample(probs, list(probs), 10)) == 37      # 1 hazard field + 10 observations + 10 damage + 10 edges + 6

    stale = tmp_path / "stale"
    stale.mkdir()
    data = json.loads((MODEL / "model.json").read_text(encoding="utf-8"))
    data["source_sha256"] = "different model"
    (stale / "model.json").write_text(json.dumps(data), encoding="utf-8")
    (stale / "system-network.json").write_text((MODEL / "system-network.json").read_text(encoding="utf-8"),
                                               encoding="utf-8")
    with pytest.raises(ValueError, match="different generated model"):
        load_generated(stale)


def test_every_threshold_without_its_own_rules_gets_them(stored_at_network_threshold):
    """A threshold is part of the system function: borrowed rules carry the answer meanwhile, but every
    origin without rules of its own is listed for extraction, however little it leaves to ndtools."""
    from src.inference import engine

    result = engine.answer({"assembly": "toy-ca/demo", "events": [{"node": "SystemState::*", "op": "==", "value": 0}],
                            "samples": 2000, "rows": {"reach-key-destination": {"target_add_h": 1.0}}})
    assert sorted(result["rules"]["extract"]) == ["n3", "n4", "n5", "n6", "n7", "n8"]
    assert min(o["unclassified"] for o in result["rules"]["origins"]) < 0.05       # even the nearly covered ones

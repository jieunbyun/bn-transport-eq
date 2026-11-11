"""The schema and instances are consistent, compile, and the kernels compute what they claim."""

from __future__ import annotations

import copy
import json
import shutil
from math import inf, log

import numpy as np
import pytest
import torch
from scipy.special import ndtr
from tbnpy.variable import Variable

from src.builder.bn_builder import compile_assembly, compile_path
from src.builder.kernels.fragility import Fragility
from src.builder.kernels.hazard import (GaussianField, ShakeMap, campbell_2003_ln_median,
                                        great_circle_km, jayaram_baker_2009_range_km)
from src.builder.kernels.edge import EdgeOpen
from src.builder.kernels.system import TravelTimeToNearest
from src.builder.load import (INSTANCES, NETWORKS, SCHEMA, RegistryError, assembly_path, assembly_paths,
                              load_assembly, load_library, load_network)
from src.inference import engine


@pytest.fixture(scope="module")
def lib():
    return load_library()


def _compile(lib, assembly_id, edit=None, network_edit=None):
    assembly = load_assembly(assembly_path(assembly_id))
    network = load_network(assembly["network"], lib)
    if edit:
        assembly = copy.deepcopy(assembly)
        edit(assembly)
    if network_edit:
        network = copy.deepcopy(network)
        network_edit(network)
    return compile_assembly(assembly, lib, network)


# ---- the instances as committed ---------------------------------------------------


def test_every_assembly_compiles(lib):
    paths = assembly_paths()
    assert len(paths) >= 2
    for path in paths:
        assert compile_path(path, lib).n_nodes > 0


def test_the_schema_diagram_is_current(lib):
    from src.builder import diagram

    assert diagram.is_current(lib=lib), "schema/diagram.md is stale; run: python -m src diagram"


def test_the_hazus_transcription_is_the_one_the_curves_cite(lib):
    import hashlib

    source = lib.parameter_sets["hazus61.HWB1"]["source"]
    text = (SCHEMA.parent / source["transcription"]).read_bytes()
    assert hashlib.sha256(text).hexdigest() == source["transcription_sha256"]


def test_few_parameter_sets_describe_every_bridge(lib):
    c = _compile(lib, "ema-highway/gmm")
    fragility = next(b for b in c.summary() if b["function"] == "fragility.bridge")
    assert fragility["rows"] == 129
    assert fragility["parameter_sets"] == {"hazus61.HWB11": 90, "hazus61.HWB4": 28, "hazus61.HWB2": 11}


# ---- consistency errors --------------------------------------------------------


def test_two_producers_for_one_node_is_refused(lib):
    def add_second_hazard(a):
        a["bindings"].insert(1, {"function": "hazard.shakemap", "parameters": {"event_id": "us7000abcd", "imt": "SA(1.0)"}})

    with pytest.raises(RegistryError, match="already produced by binding"):
        _compile(lib, "ema-highway/gmm", add_second_hazard)


def test_missing_producer_is_named(lib):
    def drop_hazard(a):
        a["bindings"] = [b for b in a["bindings"] if b["function"] != "hazard.gmm"]

    with pytest.raises(RegistryError, match=r"no binding produces SA\(1\.0\) for Bridge bridge-e0001"):
        _compile(lib, "ema-highway/gmm", drop_hazard)


def test_unknown_parameter_set_names_the_rows(lib):
    def unknown_class(network):
        network.tables["Bridge"][0]["hazus_class"] = "HWB99"

    with pytest.raises(RegistryError, match="no parameter set 'hazus61.HWB99'.*bridge-e0001"):
        _compile(lib, "ema-highway/gmm", network_edit=unknown_class)


def test_inline_parameters_are_validated(lib):
    def no_event(a):
        next(b for b in a["bindings"] if b["function"] == "hazard.shakemap")["parameters"]["event_id"] = ""

    with pytest.raises(RegistryError, match="event_id"):
        _compile(lib, "hebei-transport/shakemap", no_event)


def test_network_rows_are_checked(lib, tmp_path):
    shutil.copytree(NETWORKS / "hebei-transport", tmp_path / "hebei-transport")
    path = tmp_path / "hebei-transport" / "network.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tables"]["Bridge"][0]["edge"] = "nowhere"
    data["tables"]["Bridge"][1]["span_count"] = 3
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RegistryError) as exc:
        load_network("hebei-transport", lib, root=tmp_path)
    text = str(exc.value)
    assert "edge = 'nowhere' is not a Edge row" in text
    assert "unknown attribute 'span_count'" in text


def test_schema_checks_ports_against_plates(tmp_path):
    shutil.copytree(SCHEMA, tmp_path / "schema")
    path = tmp_path / "schema" / "functions" / "fragility.bridge.json"
    fn = json.loads(path.read_text(encoding="utf-8"))
    fn["inputs"][0]["via"] = "edge"               # Bridge.edge points at Edge, not at a Bridge
    fn["capabilities"].append("Gradient")         # not implemented
    path.write_text(json.dumps(fn), encoding="utf-8")
    with pytest.raises(RegistryError) as exc:
        load_library(tmp_path / "schema", INSTANCES)
    text = str(exc.value)
    assert "via 'edge' needs Bridge.edge to be a foreign key to Bridge" in text
    assert "OVERCLAIM Gradient" in text


# ---- a GMM chosen by location, an IM left open --------------------------------------


def _with_gmm_sets(tmp_path, *sets):
    """The library with extra hazard.gmm parameter sets, written to a scratch copy of instances/."""
    shutil.copytree(INSTANCES, tmp_path / "instances")
    campbell = json.loads((INSTANCES / "hazard" / "campbell-2003.json").read_text(encoding="utf-8"))
    campbell["sets"] = list(sets)
    (tmp_path / "instances" / "hazard" / "test-gmm.json").write_text(json.dumps(campbell), encoding="utf-8")
    return load_library(SCHEMA, tmp_path / "instances")


def _campbell_like(lib, set_id, **change):
    """Campbell's numbers under another id: a test fixture for selection, not a ground-motion model."""
    base = lib.parameter_sets["campbell-2003.ema-script-sa1"]
    return {"id": set_id, "label": "TEST ONLY", "values": {**base["values"], **change.pop("values", {})}, **change}


def _gmm_by(*set_ids):
    def edit(a):
        b = next(b for b in a["bindings"] if b["function"] == "hazard.gmm")
        b.clear()
        b.update(function="hazard.gmm", parameter_set_by_location=list(set_ids))
    return edit


def _move_first_bridge_to_beijing(network):
    network.tables["Bridge"][0].update(lat=39.9, lon=116.4)


def test_every_gmm_form_named_in_the_schema_has_a_kernel(lib):
    from src.builder.kernels.hazard import GMM_FORMS

    assert set(lib.functions["hazard.gmm"]["instance_schema"]["properties"]["form"]["enum"]) == set(GMM_FORMS)


def test_each_bridge_gets_the_gmm_whose_region_contains_it(lib, tmp_path):
    china = {"name": "test box around Beijing", "polygon": [[110.0, 35.0], [120.0, 35.0], [120.0, 45.0], [110.0, 45.0]]}
    lib2 = _with_gmm_sets(tmp_path, _campbell_like(lib, "test.china", region=china))
    c = _compile(lib2, "ema-highway/gmm", _gmm_by("campbell-2003.ema-script-sa1", "test.china"),
                 network_edit=_move_first_bridge_to_beijing)
    gmm = next(b for b in c.summary() if b["function"] == "hazard.gmm")
    assert gmm["parameter_sets"] == {"campbell-2003.ema-script-sa1": 128, "test.china": 1}


def test_a_bridge_no_listed_region_covers_is_refused_by_name(lib):
    with pytest.raises(RegistryError, match="no listed parameter set covers Bridge bridge-e0001 "
                                            r"\(campbell-2003.ema-script-sa1: Central and eastern North America\)"):
        _compile(lib, "ema-highway/gmm", _gmm_by("campbell-2003.ema-script-sa1"),
                 network_edit=_move_first_bridge_to_beijing)


def test_a_hazard_on_the_wrong_im_does_not_wire_to_the_curves(lib, tmp_path):
    pga = _campbell_like(lib, "test.pga", values={"imt": "PGA"},
                         region=lib.parameter_sets["campbell-2003.ema-script-sa1"]["region"])
    lib2 = _with_gmm_sets(tmp_path, pga)
    with pytest.raises(RegistryError, match=r"no binding produces SA\(1\.0\) for Bridge bridge-e0001.*"
                                            r"Bridge bridge-e0001 has .*PGA"):
        _compile(lib2, "ema-highway/gmm", _gmm_by("test.pga"))


def test_a_templated_port_needs_a_required_enum_of_quantities(tmp_path):
    shutil.copytree(SCHEMA, tmp_path / "schema")
    path = tmp_path / "schema" / "functions" / "hazard.gmm.json"
    fn = json.loads(path.read_text(encoding="utf-8"))
    fn["instance_schema"]["properties"]["imt"]["enum"].append("SA(9.9)")       # not a quantity
    path.write_text(json.dumps(fn), encoding="utf-8")
    with pytest.raises(RegistryError, match="parameter 'imt' must be an enum of quantity ids"):
        load_library(tmp_path / "schema", INSTANCES)


# ---- kernels compute what they claim -----------------------------------------------

# ln median SA1 (g) from the EMA dataset script's Campbell (2003), as reproduced by the
# forward pilot's hazard._gmpe_ln_sa1 at commit 784e6cc.
CAMPBELL_REFERENCE = [
    (5.0, 5.0, -3.0411925158948376),
    (6.5, 60.0, -3.364020778289433),
    (7.5, 100.0, -2.8219992023780347),
    (8.0, 200.0, -2.9479540856474182),
]


@pytest.mark.parametrize("mw,r,expected", CAMPBELL_REFERENCE)
def test_campbell_reproduces_the_dataset_script(lib, mw, r, expected):
    c = lib.parameter_sets["campbell-2003.ema-script-sa1"]["values"]["coefficients"]
    assert float(campbell_2003_ln_median(mw, r, c)) == pytest.approx(expected)


DAMAGE_STATES = ["none", "slight", "moderate", "extensive", "complete"]


def _fragility(values):
    """One bridge's Fragility, from a fragility parameter set's values."""
    curve = values["parameters"]
    medians = [curve["medians"][s] for s in DAMAGE_STATES[1:]]
    betas = [curve["betas"][s] for s in DAMAGE_STATES[1:]] if "betas" in curve else curve["beta"]
    return Fragility([Variable("DS", DAMAGE_STATES)], [Variable("IM", (0.0, inf))], medians, betas)


def test_fragility_density_and_sampling(lib):
    params = lib.parameter_sets["hazus61.HWB11"]["values"]
    im = torch.tensor([0.0, 0.5, 1.1, 3.0], dtype=torch.float64)
    p = _fragility(params).distribution(im).numpy()               # (4 intensities, 5 states)
    assert np.allclose(p.sum(-1), 1) and np.all(p >= 0)
    assert p[0, 0] == 1.0                                         # no shaking, no damage
    extensive_or_worse = p[:, 3:].sum(-1)
    assert np.allclose(extensive_or_worse[1:], ndtr(np.log(im[1:].numpy() / 1.1) / 0.6))

    torch.manual_seed(7)
    ds, logp = _fragility(params).sample(torch.full((20000, 1), 1.1, dtype=torch.float64))
    assert (ds[:, 0] >= 3).double().mean().item() == pytest.approx(0.5, abs=0.02)      # the median of Extensive
    assert torch.isfinite(logp).all()


# ---- fragility: one schema, curves as instances ---------------------------------------


def test_every_fragility_form_named_in_the_schema_has_a_kernel(lib):
    from src.builder.kernels.fragility import FRAGILITY_FORMS

    assert set(lib.functions["fragility.bridge"]["instance_schema"]["properties"]["form"]["enum"]) == set(FRAGILITY_FORMS)


def test_per_state_betas_and_crossing_curves_are_capped_with_a_note(lib):
    hazus = lib.parameter_sets["hazus61.HWB11"]["values"]
    params = {**hazus, "parameters": {"medians": hazus["parameters"]["medians"],
                                      "betas": {"slight": 0.3, "moderate": 0.4, "extensive": 0.9, "complete": 0.9}}}
    prob = _fragility(params)
    p = prob.distribution(torch.tensor([0.05, 1.1, 5.0], dtype=torch.float64)).numpy()
    assert np.allclose(p.sum(-1), 1) and np.all(p >= 0)
    assert p[1, 3:].sum() == pytest.approx(ndtr(0.0))           # Extensive's own beta, at its median
    # Moderate (beta 0.4) and Extensive (beta 0.9) cross at low im: P(DS >= extensive) is capped there.
    assert any("cross" in w for w in prob.notes)


def _library_with(tmp_path, folder, name, data):
    shutil.copytree(INSTANCES, tmp_path / "instances")
    (tmp_path / "instances" / folder).mkdir(exist_ok=True)
    (tmp_path / "instances" / folder / name).write_text(json.dumps(data), encoding="utf-8")
    return load_library(SCHEMA, tmp_path / "instances")


def _curve(set_id, **values):
    base = {"form": "lognormal", "imt": "PGA", "damage_scale": "DamageState.HazusBridge",
            "parameters": {"medians": {"slight": 0.1, "moderate": 0.2, "extensive": 0.3, "complete": 0.4}, "beta": 0.5}}
    return {"function": "fragility.bridge", "source": {"note": "TEST ONLY"},
            "sets": [{"id": set_id, "label": "TEST ONLY", "values": {**base, **values}}]}


def test_a_new_curve_is_an_instance_file_with_no_schema_change(lib, tmp_path):
    lib2 = _library_with(tmp_path, "fragility", "test-curves.json", _curve("test.pga-curve"))
    assert lib2.parameter_sets["test.pga-curve"]["function"] == "fragility.bridge"


def test_a_curve_that_does_not_fit_its_damage_scale_is_refused(tmp_path):
    bad = _curve("test.bad", parameters={"medians": {"minor": 0.1, "collapse": 0.4}, "beta": 0.5})
    with pytest.raises(RegistryError) as exc:
        _library_with(tmp_path, "fragility", "test-curves.json", bad)
    text = str(exc.value)
    assert "test.bad" in text and "'slight' is a required property" in text


def test_a_set_filed_under_another_role_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="role 'fragility', so its sets belong in"):
        _library_with(tmp_path, "hazard", "test-curves.json", _curve("test.misfiled"))


def test_the_gmm_gives_each_bridge_its_own_median_and_sigma(lib):
    """ln SA(1.0) at a bridge is N(Campbell median at its hypocentral distance, total sigma)."""
    c = _compile(lib, "ema-highway/gmm")
    gmm = next(b for b in c.bindings if b.function["id"] == "hazard.gmm")
    samples = engine.sample_nodes(c, 20000, seed=3, bindings=c.ancestors({gmm.index}))
    event = c.network.row("ema-reference")
    values = lib.parameter_sets["campbell-2003.ema-script-sa1"]["values"]
    s = values["sigma"]
    sigma = np.hypot(s["a"] + s["b"] * event["mw"] if event["mw"] < s["m_break"] else s["sigma_above"],
                     values["sigma_epistemic"])
    for bridge in c.network.rows("Bridge")[:5]:
        r = np.hypot(float(great_circle_km(event["lat"], event["lon"], bridge["lat"], bridge["lon"])), event["depth_km"])
        x = np.log(samples[f"SA(1.0)::{bridge['id']}"])
        assert x.mean() == pytest.approx(float(campbell_2003_ln_median(event["mw"], r, values["coefficients"])), abs=0.02)
        assert x.std() == pytest.approx(sigma, rel=0.03)


def test_an_edge_closes_with_any_bridge_and_is_open_without_one():
    damage = [Variable("DamageState.HazusBridge::x", DAMAGE_STATES), Variable("DamageState.HazusBridge::y", DAMAGE_STATES)]
    policy = ["none", "slight", "moderate"]                       # closed at extensive or complete
    two_bridges = EdgeOpen([Variable("EdgeOpen::a", [0, 1])], damage, usable_states=policy)
    no_bridge = EdgeOpen([Variable("EdgeOpen::b", [0, 1])], [], usable_states=policy)
    #                    both moderate  one extensive  slight + complete
    ds = torch.tensor([[2.0, 2.0], [0.0, 3.0], [1.0, 4.0]], dtype=torch.float64)
    assert two_bridges.evaluate(ds)[:, 0].tolist() == [1, 0, 0]
    assert no_bridge.sample(n_sample=2)[0][:, 0].tolist() == [1, 1]


def _diamond(target_add_h=0.5, directed=False, key="t", origins="sab"):
    """{origin: its TravelTimeToNearest}. s-a-t is 1 h and s-b-t 2 h at 80 km/h; ``key`` is the key destination."""
    nodes = [{"id": x, "key_destination": x == key} for x in "sabt"]
    edges = [{"id": eid, "source": u, "target": v, "directed": directed, "length_km": km, "avg_speed_kph": 80.0}
             for eid, u, v, km in (("sa", "s", "a", 40.0), ("at", "a", "t", 40.0), ("sb", "s", "b", 80.0),
                                   ("bt", "b", "t", 80.0))]
    edge_open = [Variable(f"EdgeOpen::{e['id']}", [0, 1]) for e in edges]
    return {origin: TravelTimeToNearest([Variable(f"SystemState::{origin}", [0, 1])], edge_open,
                                        origin=origin, edge_ids=[e["id"] for e in edges],
                                        target_add_h=target_add_h, network={"Node": nodes, "Edge": edges})
            for origin in origins}


def test_each_origin_is_judged_against_its_intact_travel_time():
    #                          sa at sb bt
    states = torch.tensor([[1, 1, 1, 1],      # intact: s 1.0 h (via a)
                           [0, 1, 1, 1],      # sa closed: s goes via b, 2.0 h, 1.0 h late
                           [0, 1, 0, 1]], dtype=torch.float64)     # s cut off
    systems = _diamond()
    assert systems["s"].evaluate(states)[:, 0].tolist() == [1, 0, 0]  # 1.0 h late > target_add_h 0.5 h
    assert systems["a"].evaluate(states)[:, 0].tolist() == [1, 1, 1]  # a reaches t directly throughout
    assert not any(system.notes for system in systems.values())

    lenient = _diamond(target_add_h=1.0)["s"].evaluate(states[1:2])
    assert lenient[0, 0] == 1                                          # 1.0 h late is within 1.0 h


def test_an_origin_with_no_route_even_intact_is_reported():
    # Every edge points away from s, so only s itself reaches s.
    systems = _diamond(directed=True, key="s", origins="sabt")
    intact = torch.ones((1, 4), dtype=torch.float64)
    assert [systems[o].evaluate(intact)[0, 0].item() for o in "sabt"] == [1, 0, 0, 0]
    assert not systems["s"].notes
    assert "no route to a key destination even with every edge open" in systems["a"].notes[0]


def test_every_system_node_reads_every_edge(lib):
    c = _compile(lib, "ema-highway/gmm")
    nodes = {n["id"]: n for n in c.nodes()}
    origins = [r["id"] for r in c.network.rows("Node") if not r["key_destination"]]
    assert len(origins) == 72 and "SystemState::n22" not in nodes             # key destinations are not origins
    assert len(nodes[f"SystemState::{origins[0]}"]["parents"]) == 129


def test_shakemap_kernel_is_a_field_of_the_maps_median_and_sigma_and_bounds_a_site_off_it_by_its_edge(monkeypatch):
    from src.builder import shakemap

    grid = shakemap.ShakeMapGrid(
        event_id="test", version="1", meta={}, lon_min=0.0, lat_max=1.0, dlon=1.0, dlat=1.0,
        ln_im=np.full((2, 2), log(0.4)), std_im=np.full((2, 2), 0.5),
    )
    monkeypatch.setattr(shakemap, "load", lambda *a, **k: grid)

    def shakemap_at(sites, correlation=None):
        rows = [{"id": f"s{i}", "lat": lat, "lon": lon} for i, (lat, lon) in enumerate(sites)]
        return ShakeMap([Variable(f"SA(1.0)::{r['id']}", (0.0, inf)) for r in rows], [],
                        event_id="test", imt="SA(1.0)", sites=rows, correlation=correlation)

    prob = shakemap_at([(0.5, 0.5), (0.2, 0.7)])
    assert torch.allclose(prob.ln_median, torch.tensor(log(0.4), dtype=torch.float64))
    assert torch.allclose(prob.sigma, torch.tensor(0.5, dtype=torch.float64))
    torch.manual_seed(3)
    sa, logp = prob.sample(n_sample=40000)
    ln_sa = sa.log().numpy()
    assert np.allclose(ln_sa.mean(axis=0), log(0.4), atol=0.01) and np.allclose(ln_sa.std(axis=0), 0.5, atol=0.01)
    assert abs(np.corrcoef(ln_sa.T)[0, 1]) < 0.03                         # no correlation named: independent
    assert any("no seismic station" in w for w in prob.notes)

    # Off the map (it spans 0-1 degrees), with another site on it: the value at the nearest edge point, (1, 1).
    from dataclasses import replace
    corner = replace(grid, ln_im=np.array([[log(0.4), log(0.3)], [log(0.4), log(0.4)]]))   # row 0 is lat 1: (1, 1) is 0.3 g
    monkeypatch.setattr(shakemap, "load", lambda *a, **k: corner)
    off = shakemap_at([(0.5, 0.5), (1.5, 1.5)])
    assert np.isclose(off.ln_median[1], log(0.3)) and np.isclose(off.sigma[1], 0.5)
    assert off.outside_km[0] == 0 and 70 < off.outside_km[1] < 80
    assert any("1 of 2 sites lie off ShakeMap test v1 (up to 79 km" in w and "upper bound" in w for w in off.notes)
    with pytest.raises(ValueError, match="reaches none of the 1 sites"):
        shakemap_at([(5.0, 5.0)])


@pytest.mark.parametrize("period, clustered, expected", [
    (0.0, False, 8.5), (0.0, True, 40.7), (0.5, False, 17.1), (0.5, True, 33.2),
    (1.0, False, 25.7), (1.0, True, 25.7), (2.0, False, 29.4),
])
def test_jayaram_baker_ranges(period, clustered, expected):
    assert jayaram_baker_2009_range_km(period, clustered) == pytest.approx(expected)


def _field(lib, set_id, rows, ln_median=log(0.2), sigma=0.6):
    """A Gaussian field of ln SA at the rows, correlated by the set."""
    prob = GaussianField([Variable(f"SA(1.0)::{r['id']}", (0.0, inf)) for r in rows])
    n = len(rows)
    prob._field(rows, torch.full((n,), ln_median, dtype=torch.float64), torch.full((n,), sigma, dtype=torch.float64),
                {"id": set_id, **lib.parameter_sets[set_id]["values"]})
    return prob


def _draw(prob, n=40000):
    torch.manual_seed(7)
    sa, logp = prob.sample(n_sample=n)
    torch.testing.assert_close(logp, prob.log_prob(sa))                 # sample and likelihood agree
    return np.log(sa.numpy())


def test_the_field_reproduces_the_correlation_model(lib):
    # 0.08993 deg of latitude is 10.0 km; the third site sits on the first.
    rows = [{"id": "a", "lat": 0.0, "lon": 0.0}, {"id": "b", "lat": 0.08993, "lon": 0.0},
            {"id": "c", "lat": 0.0, "lon": 0.0}]
    prob = _field(lib, "jayaram-baker-2009.sa1p0", rows)
    ln_sa = _draw(prob)
    r = np.corrcoef(ln_sa.T)
    assert np.allclose(ln_sa.mean(axis=0), log(0.2), atol=0.01) and np.allclose(ln_sa.std(axis=0), 0.6, atol=0.01)
    assert r[0, 1] == pytest.approx(np.exp(-3 * 10.0 / 25.7), abs=0.02)
    assert r[0, 2] == pytest.approx(1.0, abs=1e-6)                     # co-located: no failure, fully correlated
    assert not prob.notes


def test_the_fields_density_is_the_lognormal_one(lib):
    # Two distinct sites: ln SA ~ N(mu, D R D), so the density of SA is the bivariate normal's of ln SA over SA_1 SA_2.
    rows = [{"id": "a", "lat": 0.0, "lon": 0.0}, {"id": "b", "lat": 0.08993, "lon": 0.0}]
    prob = _field(lib, "jayaram-baker-2009.sa1p0", rows)
    sa = torch.tensor([[0.1, 0.3], [0.2, 0.2], [1.0, 0.05]], dtype=torch.float64)
    rho = torch.exp(-3 * great_circle_km(0.0, 0.0, 0.08993, 0.0) / 25.7).item()
    cov = 0.36 * torch.tensor([[1.0, rho], [rho, 1.0]], dtype=torch.float64)
    mvn = torch.distributions.MultivariateNormal(torch.full((2,), log(0.2), dtype=torch.float64), cov)
    torch.testing.assert_close(prob.log_prob(sa), mvn.log_prob(sa.log()) - sa.log().sum(1))
    assert prob.log_prob(torch.tensor([[0.0, 0.1]], dtype=torch.float64)).item() == -inf


def test_independent_sites_are_uncorrelated_and_say_so(lib):
    rows = [{"id": "a", "lat": 0.0, "lon": 0.0}, {"id": "b", "lat": 0.0, "lon": 0.0}]
    prob = _field(lib, "independent.any-im", rows)
    assert abs(np.corrcoef(_draw(prob).T)[0, 1]) < 0.03
    assert "independently" in prob.notes[0]
    sa = torch.tensor([[0.1, 0.3]], dtype=torch.float64)                # the product of the two lognormals
    torch.testing.assert_close(prob.log_prob(sa), prob.distribution(1).log_prob(sa).sum(1))


# ---- the system event ---------------------------------------------------------------


def test_a_system_event_takes_the_defaults_it_leaves_out(lib, tmp_path):
    shutil.copytree(NETWORKS / "toy-ca", tmp_path / "toy-ca", ignore=shutil.ignore_patterns("tbnpy", "*_files"))
    path = tmp_path / "toy-ca" / "network.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tables"]["SystemEvent"] = [{"id": "reach"}]
    path.write_text(json.dumps(data), encoding="utf-8")
    event = load_network("toy-ca", lib, root=tmp_path).row("reach")
    assert event["analysis"] == "system.travel-time-to-nearest" and event["target_add_h"] == 0.5


def test_the_bound_system_function_must_be_the_events_analysis(lib):
    def other_analysis(network):
        network.tables["SystemEvent"][0]["analysis"] = "system.max-flow"

    with pytest.raises(RegistryError, match="judged by system.max-flow, not system.travel-time-to-nearest"):
        _compile(lib, "toy-ca/demo", network_edit=other_analysis)

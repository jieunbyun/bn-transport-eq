"""The page's endpoints answer, and the view data shows every BN and what connects."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.app.server import make_server
from src.app.views import view_data


@pytest.fixture(scope="module")
def base_url():
    server = make_server(port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url):
    with urlopen(url, timeout=60) as response:
        return response.status, response.read()


@pytest.fixture(autouse=True)
def isolated_server_state(tmp_path, monkeypatch):
    """HTTP tests use private sessions and never start workers that write model artifacts."""
    from types import SimpleNamespace
    from src.app import server
    import subprocess
    monkeypatch.setattr(server, "subprocess", SimpleNamespace(
        Popen=lambda *a, **kw: SimpleNamespace(poll=lambda: None), STDOUT=subprocess.STDOUT,
        CREATE_NEW_PROCESS_GROUP=0))
    monkeypatch.setattr(server, "_extractions", {})
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_inference_jobs", {})
    monkeypatch.setattr(server, "_early_stops", deque(maxlen=128))
    monkeypatch.setattr(server, "SESSIONS", tmp_path / "sessions")


def post(url, body):
    request = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_view_data_has_the_schema_and_every_bn():
    views = {v["id"]: v for v in view_data()["views"]}
    assert set(views) == {"library", "ema-highway/gmm", "ema-highway/shakemap", "toy-ca/demo", "toy-ca/shakemap"}
    links = {(x["from"], x["to"]) for x in views["library"]["links"]}
    for hazard in ("hazard.gmm", "hazard.shakemap"):
        assert (hazard, "fragility.bridge") in links          # alternatives, all type-compatible
    assert views["ema-highway/gmm"]["n_nodes"] == 588            # 387 + 129 SA measurements + a system state for each of 72 origins
    assert views["ema-highway/gmm"]["menu"]["settable"]["rows"]


def test_the_export_is_one_file_with_the_views_inside(tmp_path):
    from src.app.views import export_page

    html = export_page(tmp_path / "schema.html").read_text(encoding="utf-8")
    assert html.count("window.BN_DATA = ") == 1 and '"ema-highway/gmm"' in html
    assert 'src="/diagram-navigation.js"' not in html
    assert "function diagramNavigation(" in html
    assert "</script>" not in html.split("window.BN_DATA = ", 1)[1].split(";</script>", 1)[0]


def test_the_schema_page_is_current():
    from src.app.views import schema_page_is_current

    assert schema_page_is_current(), "schema/diagram.html is stale; run: python -m src diagram"


def test_the_page_and_its_data_are_served(base_url):
    status, body = get(base_url + "/")
    assert status == 200 and b"<title>quakenet</title>" in body and b"/api/example?assembly=" in body
    assert b'id="panel-query"' in body and b'id="panel-evidence"' in body     # the two panels under the map
    with urlopen(base_url + "/diagram-navigation.js", timeout=10) as response:
        assert response.status == 200 and response.headers.get_content_type() == "text/javascript"
        assert b"function diagramNavigation(" in response.read()
    status, body = get(base_url + "/bn")
    assert status == 200 and b"/api/views" in body                # the BN drawing, embedded in quakenet's BN tab
    status, body = get(base_url + "/api/views")
    assert status == 200 and json.loads(body)["networks"]["ema-highway"]["rows"]["Bridge"] == 129
    assert "hebei-transport" not in json.loads(body)["networks"]


def test_a_query_is_answered_and_a_bad_one_explained(base_url):
    status, body = post(base_url + "/api/query", {"assembly": "ema-highway/gmm",
                                                  "targets": ["EdgeOpen::e0001"], "samples": 200})
    assert status == 200 and body["targets"][0]["node"] == "EdgeOpen::e0001"
    status, body = post(base_url + "/api/query", {"assembly": "ema-highway/gmm",
                                                  "targets": ["SA(1.0)::*"], "evidence": {"x": 1}})
    assert status == 400 and "no node 'x'" in body["error"]


def test_a_session_runs_and_drafting_without_a_model_says_why(base_url, monkeypatch):
    status, body = post(base_url + "/api/session/draft", {"assembly": "toy-ca/demo", "kind": "evidence", "text": "hi"})
    assert status == 400 and "Run" in body["error"]                   # no session yet
    status, body = post(base_url + "/api/session", {"assembly": "toy-ca/demo", "samples": 10000})
    assert status == 200 and body["n"] == 10000 and body["result"]["events"]
    assert body["timeline"][0]["elapsed_s"] >= 0
    status, body = post(base_url + "/api/session/query", {"assembly": "toy-ca/demo", "query": {
        "name": "Towns cut off", "target": {"kind": "cut_off"}, "decision": "track"}})
    assert status == 200 and body["timeline"][-1]["kind"] == "query"
    assert body["timeline"][-1]["elapsed_s"] >= 0 and body["timeline"][-1]["values"]["q1"]
    node_url = base_url + "/api/session/node?assembly=toy-ca/demo&node=EdgeOpen::e6"
    _, before = get(node_url)
    before = json.loads(before)
    status, body = post(base_url + "/api/session/evidence", {"assembly": "toy-ca/demo", "variable": "EdgeOpen::e6",
                                                             "value": "closed"})
    assert status == 200 and body["timeline"][-1]["kind"] == "evidence"
    timings = [entry["elapsed_s"] for entry in body["timeline"]]
    assert len(timings) == 3 and all(elapsed >= 0 for elapsed in timings)
    from src.app import server
    server._sessions.clear()                         # reopen the saved session, as after a restart
    _, saved = get(base_url + "/api/session?assembly=toy-ca/demo")
    assert [entry["elapsed_s"] for entry in json.loads(saved)["session"]["timeline"]] == timings
    status, comparison = get(node_url)
    comparison = json.loads(comparison)
    assert status == 200 and comparison["p"] == pytest.approx([1, 0])
    assert comparison["previous"]["p"] == before["p"]
    assert comparison["previous"]["at"] == before["at"]
    monkeypatch.setenv("BN_LLM", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    status, body = post(base_url + "/api/session/draft", {"assembly": "toy-ca/demo", "kind": "evidence", "text": "hi"})
    assert status == 503 and "GEMINI_API_KEY" in body["error"]


@pytest.mark.parametrize("failure,status,expected", [
    (ModuleNotFoundError("No module named 'rsr.rsr'", name="rsr.rsr"), 503, "rsr.rsr"),
    (RuntimeError("sampling failed"), 500, "RuntimeError"),
])
def test_inference_failures_return_json_instead_of_dropping_the_connection(
        base_url, monkeypatch, failure, status, expected):
    from src.app import server

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(server.engine, "answer", fail)
    code, body = post(base_url + "/api/query", {"assembly": "ema-highway/gmm", "samples": 200})
    assert code == status and expected in body["error"]
    if status == 503:
        assert "conda run -n quakenet python -m src serve" in body["error"]
    assert get(base_url + "/api/examples")[0] == 200


def test_an_example_carries_its_network_menu_and_rules(base_url):
    status, body = get(base_url + "/api/examples")
    examples = {e["id"]: e for e in json.loads(body)["examples"]}
    assert status == 200 and set(examples) == {"toy-ca/demo", "toy-ca/shakemap", "ema-highway/gmm", "ema-highway/shakemap"}
    assert {i for i, e in examples.items() if e["shakemap"]} == {"toy-ca/shakemap", "ema-highway/shakemap"}
    status, body = get(base_url + "/api/example?assembly=toy-ca/demo")
    ex = json.loads(body)
    assert status == 200 and set(ex["network"]["tables"]) >= {"Node", "Edge", "Bridge", "Event", "SystemEvent"}
    assert ex["objects"] == 37 and ex["rules"] == {"stored": 6, "origins": 6}
    assert len(ex["nodes"]) == 46 and ex["joint"] == ["hazard.gmm"]          # what the BN tab draws, layer by layer
    settable = {(r["entity"], r["attribute"]) for r in ex["menu"]["settable"]["rows"]}
    assert {("Event", "mw"), ("SystemEvent", "target_add_h")} <= settable
    with pytest.raises(HTTPError) as missing:
        get(base_url + "/api/example?assembly=nowhere/x")
    assert missing.value.code == 404 and "nowhere/x" in json.loads(missing.value.read())["error"]


def test_a_map_run_returns_every_family_and_says_rules_were_used(base_url):
    events = [{"node": f"{q}::*", "op": "==", "value": 0} for q in ("SystemState", "EdgeOpen")]
    status, body = post(base_url + "/api/query", {"assembly": "toy-ca/demo", "events": events, "samples": 500})
    assert status == 200 and len(body["events"]) == 6 + 10
    assert any("rsr rules exact for 6 of 6" in w for w in body["warnings"])


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


def test_a_run_left_mostly_unclassified_starts_extraction_in_the_background(base_url, monkeypatch,
                                                                         stored_at_network_threshold):
    from src.app import server

    started = []
    import subprocess
    from types import SimpleNamespace

    job = type("Job", (), {"poll": lambda self: None})
    # Only the server's own subprocess: git, which the query records call, keeps the real one.
    monkeypatch.setattr(server, "subprocess", SimpleNamespace(
        Popen=lambda cmd, **kw: started.append(cmd) or job(), STDOUT=subprocess.STDOUT, CREATE_NEW_PROCESS_GROUP=0))
    monkeypatch.setattr(server, "_extractions", {})
    query = {"assembly": "toy-ca/demo", "events": [{"node": "SystemState::*", "op": "==", "value": 0}], "samples": 500,
             "rows": {"reach-key-destination": {"target_add_h": 0.25}}}
    status, body = post(base_url + "/api/query", query)
    assert status == 200 and any("in the background" in w for w in body["warnings"])
    assert started[0][2:6] == ["src", "rules", "toy-ca/demo", "--target-add-h"] and started[0][6] == "0.25"
    status, body = post(base_url + "/api/query", query)                 # the same scenario again: read from its bank
    assert len(started) == 1 and body["samples_from"] == "the scenario's stored samples"
    other = {**query, "rows": {**query["rows"], "demo-san-andreas": {"mw": 7.0}}}
    status, body = post(base_url + "/api/query", other)                 # new samples, same threshold: one job only
    assert len(started) == 1 and any("still being extracted" in w for w in body["warnings"])


def test_a_negative_threshold_is_refused(base_url):
    status, body = post(base_url + "/api/query", {"assembly": "toy-ca/demo", "targets": ["SystemState::n4"],
                                                  "rows": {"reach-key-destination": {"target_add_h": -0.1}}})
    assert status == 400 and "below its minimum" in body["error"]


@pytest.mark.parametrize("operation", ["start", "query", "evidence"])
def test_stop_interrupts_computation_and_keeps_the_previous_session(base_url, monkeypatch, operation):
    import time
    from src.app import server
    from src.cancellation import check_cancelled
    from src.inference.session import Session

    assembly = "toy-ca/demo"
    assert post(base_url + "/api/session", {"assembly": assembly, "samples": 10000})[0] == 200
    assert post(base_url + "/api/session/query", {"assembly": assembly, "query": {
        "name": "Towns cut off", "target": {"kind": "cut_off"}, "decision": "track"}})[0] == 200
    previous = server._sessions[assembly]
    saved = server._session_path(assembly).read_bytes()
    evidence, queries, timeline, seed = dict(previous.evidence), dict(previous.queries), list(previous.timeline), previous.seed
    entered = threading.Event()

    def wait_for_stop(*args, **kwargs):
        entered.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            check_cancelled()
            time.sleep(0.01)
        raise AssertionError("Stop did not reach the computation")

    monkeypatch.setattr(Session, "_draw" if operation == "start" else "_values", wait_for_stop)
    request_id = f"stop-test-{operation}"
    body = {"assembly": assembly, "request_id": request_id, "samples": 10000}
    path = {"start": "/api/session", "query": "/api/session/query", "evidence": "/api/session/evidence"}[operation]
    if operation == "query":
        body["query"] = {"name": "Another target", "target": {"kind": "closed", "edge": "e9"}, "decision": "track"}
    if operation == "evidence":
        body.update(variable="EdgeOpen::e6", value="open")
    reply = []
    worker = threading.Thread(target=lambda: reply.append(post(base_url + path, body)), daemon=True)
    worker.start()
    try:
        assert entered.wait(5), "The computation did not start"
        # Stop must answer while the computation still owns _sessions_lock.
        assert post(base_url + "/api/session/stop", {"request_id": request_id}) == (200, {"stopping": True})
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert reply[0][0] == 409 and reply[0][1]["cancelled"] is True
        assert server._sessions[assembly] is previous
        assert previous.evidence == evidence and previous.queries == queries and previous.timeline == timeline
        assert previous.seed == seed
        assert server._session_path(assembly).read_bytes() == saved
        assert not server._inference_jobs
    finally:
        server.stop_inference(request_id)
        worker.join(timeout=6)


def test_stop_can_arrive_before_its_inference_request(base_url, monkeypatch):
    from src.app import server

    def must_not_start(*args, **kwargs):
        raise AssertionError("A previously cancelled request must not start inference")

    monkeypatch.setattr(server, "start_session", must_not_start)
    assert post(base_url + "/api/session/stop", {"request_id": "stopped-before-start"})[0] == 200
    status, body = post(base_url + "/api/session", {
        "assembly": "toy-ca/demo", "request_id": "stopped-before-start", "samples": 10000})
    assert status == 409 and body["cancelled"] is True
    assert not server._inference_jobs


def test_stop_requires_a_request_identifier(base_url):
    status, body = post(base_url + "/api/session/stop", {})
    assert status == 400 and "request_id" in body["error"]


def test_stop_interrupts_a_queued_update_before_the_session_lock_is_available(base_url, monkeypatch):
    from contextlib import contextmanager
    from src.app import server

    entered, reply = threading.Event(), []
    original = server.inference_job

    @contextmanager
    def registered(request_id):
        with original(request_id):
            entered.set()
            yield

    monkeypatch.setattr(server, "inference_job", registered)
    body = {"assembly": "toy-ca/demo", "request_id": "queued-update", "samples": 10000}
    worker = threading.Thread(target=lambda: reply.append(post(base_url + "/api/session", body)), daemon=True)
    try:
        with server._sessions_lock:
            worker.start()
            assert entered.wait(3)
            assert post(base_url + "/api/session/stop", {"request_id": "queued-update"})[0] == 200
            worker.join(timeout=3)
            assert not worker.is_alive(), "A queued update should stop without waiting for the other computation"
            assert reply[0][0] == 409 and reply[0][1]["cancelled"] is True
    finally:
        server.stop_inference("queued-update")
        worker.join(timeout=6)


# ---- real earthquakes: USGS ShakeMaps ------------------------------------------------------


@pytest.fixture
def shakecast_grid(tmp_path, monkeypatch):
    """One ShakeMap (two versions) covering Hebei, read from a ShakeCast directory: no network."""
    from src.builder import shakemap
    from src.builder.tests.test_shakemap import write_grid

    root = tmp_path / ".shakecast"
    for version in (1, 2):
        write_grid(root / "data" / "test001" / f"test001-{version}", "test001", version)
    monkeypatch.setenv("SHAKECAST_USER_DIRECTORY", str(root))
    monkeypatch.setenv("BN_SHAKEMAP_SOURCE", "shakecast")
    shakemap._load_cached.cache_clear()
    shakemap._resolved.clear()
    yield root
    shakemap._load_cached.cache_clear()
    shakemap._resolved.clear()


def test_a_shakemap_session_pins_the_version_it_ran_on(base_url, shakecast_grid):
    from src.app import server

    scenario = {"parameters": {"hazard.shakemap": {"event_id": "test001"}}}
    code, body = post(base_url + "/api/session", {"assembly": "hebei-transport/shakemap", "scenario": scenario,
                                                  "samples": 500})
    assert code == 200, body
    # The newest version was used, and is written into the scenario, so a reopened session rebuilds the same map.
    assert body["scenario"]["parameters"]["hazard.shakemap"] == {"event_id": "test001", "version": 2}
    assert body["shakemap"]["event_id"] == "test001" and body["shakemap"]["version"] == 2
    assert body["shakemap"]["magnitude"] == 6.8 and body["shakemap"]["source"] == "shakecast"
    server._sessions.clear()                                 # as after a restart: reopened from disk
    status, raw = get(base_url + "/api/session?assembly=hebei-transport/shakemap")
    assert status == 200 and json.loads(raw)["session"]["shakemap"]["version"] == 2

    scenario["parameters"]["hazard.shakemap"]["version"] = 1  # an earlier version, asked for
    code, body = post(base_url + "/api/session", {"assembly": "hebei-transport/shakemap", "scenario": scenario,
                                                  "samples": 500})
    assert code == 200 and body["shakemap"]["version"] == 1


def test_a_gmm_session_carries_no_shakemap(base_url):
    code, body = post(base_url + "/api/session", {"assembly": "toy-ca/demo", "samples": 500})
    assert code == 200 and body["shakemap"] is None


def test_a_shakemap_that_cannot_be_had_is_a_readable_refusal(base_url, shakecast_grid):
    code, body = post(base_url + "/api/session", {"assembly": "hebei-transport/shakemap", "samples": 500,
                                                  "scenario": {"parameters": {"hazard.shakemap": {"event_id": "nothere"}}}})
    assert code == 400 and "ShakeCast has no ShakeMap for 'nothere'" in body["error"]


def test_recent_earthquakes_merge_the_networks_searches(base_url, monkeypatch):
    from src.app import server
    from src.builder import shakemap

    calls = []

    def search(self, bbox, start="1900-01-01", end=None, min_magnitude=None):
        calls.append((bbox, min_magnitude))
        lon_min = bbox[0]
        shared = {"event_id": "both", "magnitude": 4.4, "time": "2026-11-01T00:00:00+00:00", "lat": 0, "lon": 0}
        own = {"event_id": "ca1" if lon_min < -100 else "ne1", "magnitude": 5.0,
               "time": "2026-11-02T00:00:00+00:00", "lat": 0, "lon": 0}
        return [own, shared]

    monkeypatch.setattr(shakemap.ComcatSource, "search", search)
    monkeypatch.setattr(server, "_recent", {})
    status, body = get(base_url + "/api/recent-earthquakes")
    events = {e["event_id"]: e for e in json.loads(body)["events"]}
    assert status == 200 and len(calls) == 2 and all(m == 4.0 for _, m in calls)
    assert events["both"]["networks"] == ["ema-highway", "toy-ca"]
    assert events["both"]["assemblies"] == ["ema-highway/shakemap", "toy-ca/shakemap"]
    assert events["ca1"]["assemblies"] == ["toy-ca/shakemap"] and events["ne1"]["networks"] == ["ema-highway"]
    get(base_url + "/api/recent-earthquakes")
    assert len(calls) == 2                                   # asked again within a minute: from memory

    def offline(self, *a, **k):
        raise shakemap.ShakeMapError("could not reach the USGS catalogue: offline")

    monkeypatch.setattr(shakemap.ComcatSource, "search", offline)
    monkeypatch.setattr(server, "_recent", {})
    with pytest.raises(HTTPError) as failed:
        get(base_url + "/api/recent-earthquakes")
    assert failed.value.code == 502 and "offline" in json.loads(failed.value.read())["error"]


def test_the_page_can_ask_for_a_shakemaps_current_version(base_url, monkeypatch):
    from src.builder import shakemap

    def current(self, event_id):
        if event_id == "gone":
            raise shakemap.UnknownEvent("USGS has no event with id 'gone'")
        return 7

    monkeypatch.setattr(shakemap.ComcatSource, "current_version", current)
    status, body = get(base_url + "/api/shakemap-version/nc216859")
    assert status == 200 and json.loads(body) == {"event_id": "nc216859", "version": 7}
    for event_id, code in (("gone", 404), ("bad%20id", 400)):
        with pytest.raises(HTTPError) as failed:
            get(base_url + f"/api/shakemap-version/{event_id}")
        assert failed.value.code == code


def test_past_shakemaps_say_how_many_bridges_they_reach_and_those_reaching_none_are_not_offered(base_url, monkeypatch):
    # toy-ca's bridges run from 36.1 to 36.9 N: a map stopping at 36.7 N (the 1922 Parkfield Atlas map) reaches
    # some; one stopping at 35 N reaches none.
    from src.builder import shakemap

    events = [{"event_id": e, "source": "comcat", "updated": 1} for e in ("whole", "south", "away")]
    monkeypatch.setattr(shakemap, "search_events", lambda rows, **k: list(events))
    extents = {"whole": (34.0, 39.0), "south": (31.7, 36.7), "away": (31.0, 35.0)}
    monkeypatch.setattr(shakemap.ComcatSource, "extent", lambda self, event_id, updated=None: {
        "lat_min": extents[event_id][0], "lat_max": extents[event_id][1], "lon_min": -125.0, "lon_max": -115.0})
    status, body = get(base_url + "/api/shakemap-events?network=toy-ca")
    body = json.loads(body)
    assert status == 200 and body["outside"] == 1
    reached = {e["event_id"]: (e["covered"], e["sites"]) for e in body["events"]}
    assert reached["whole"] == (10, 10) and 0 < reached["south"][0] < 10


def test_a_browser_that_leaves_mid_reply_is_not_reported_as_a_server_fault(capsys):
    from src.app.server import _Server

    server = _Server.__new__(_Server)                 # handle_error needs no socket
    for exc in (BrokenPipeError(32, "Broken pipe"), ConnectionResetError(104, "reset")):
        try:
            raise exc
        except OSError:
            server.handle_error(None, ("127.0.0.1", 1))
    assert capsys.readouterr().err == ""
    try:
        raise RuntimeError("a real fault")
    except RuntimeError:
        server.handle_error(None, ("127.0.0.1", 1))
    assert "a real fault" in capsys.readouterr().err


def test_a_new_session_archives_the_old_one_and_computes_nothing(base_url, tmp_path):
    from src.app import server

    status, body = post(base_url + "/api/session", {"assembly": "toy-ca/demo", "samples": 1000})
    assert status == 200 and server._session_path("toy-ca/demo").is_file()
    status, body = post(base_url + "/api/session/new", {"assembly": "toy-ca/demo"})
    assert status == 200 and body == {"session": None}
    assert not server._session_path("toy-ca/demo").is_file() and "toy-ca/demo" not in server._sessions
    assert len(list((server.SESSIONS / "archive").glob("toy-ca__demo__*.pt"))) == 1      # kept, not deleted
    _, saved = get(base_url + "/api/session?assembly=toy-ca/demo")
    assert json.loads(saved)["session"] is None                                           # as on first opening
    status, body = post(base_url + "/api/session/new", {"assembly": "nowhere/x"})
    assert status == 400 and "nowhere/x" in body["error"]

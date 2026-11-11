"""quakenet: one local page that runs an example network's BN, by form or in words.

Standard library only. Endpoints:

    GET  /                        quakenet: scenario, map, results, BN, questions in words
    GET  /bn                      the BN drawing (also embedded in quakenet's BN tab)
    GET  /api/examples            every example assembly: id, label, network
    GET  /api/example?assembly=<id>
                                  one example: its network's rows, its query menu, its stored rules
    GET  /api/views               the schema and every compiled BN, each with its query menu
    POST /api/query               a probabilistic query -> its answer          (src/inference)
    GET  /api/chat-status         whether a language-model backend is ready
    GET  /api/shakemap-events?network=<id>&min_magnitude=4
                                  USGS events since 1900 whose ShakeMap reaches a bridge of the
                                  network (each with how many), and how many nearby reach none
    GET  /api/recent-earthquakes  events of the last 30 days, M4+, with a ShakeMap reaching a network the
                                  page offers a ShakeMap assembly for; straight from USGS, cached a minute
    GET  /api/shakemap-version/<event id>
                                  the version of the event's current ShakeMap, asked of USGS now

    The posterior session of an example (src/inference/session.py), one per example, saved to
    records/sessions/ after every change and reopened from there after a restart:
    GET  /api/session?assembly=<id>               its evidence, decision queries and timeline
    POST /api/session            {assembly, scenario, samples}
                                                  start it (Run): sample with no evidence, under the
                                                  page's scenario settings
    POST /api/session/query      {assembly, query, at}
                                                  add a decision query
    POST /api/session/evidence   {assembly, variable, value, at, source, text}
                                                  add an observation: reweight or sample again
    GET  /api/session/node?assembly=<id>&node=<variable>
                                                  one variable's distribution given the evidence so far
    POST /api/session/draft      {assembly, kind: query | evidence, text}
                                                  the assistant's draft from a sentence; nothing is added
    POST /api/session/new        {assembly}         archive the example's session and keep none, as before its first Run
    POST /api/session/stop       {request_id}       stop a running update, keeping its previous session
    GET  /api/session/progress?request_id=<id>      what a running update is doing now (reweighting, resampling)
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import threading
import traceback
import webbrowser
from collections import deque
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from src.builder import shakemap
from src.builder.bn_builder import attach_rules, compile_path, model_dir, tbn_data
from src.builder.load import (ROOT, RegistryError, assembly_id_of, assembly_path, load_assembly,
                              load_network)
from src.cancellation import InferenceCancelled, cancellable, check_cancelled
from src.inference import engine
from src.llm import chat
from src.monitor.watch import shakemap_parameters

from .views import app_assembly_paths, view_data

PAGE = Path(__file__).with_name("quakenet.html")
BN_PAGE = Path(__file__).with_name("page.html")

# A raw SDK traceback is no use on the page. Say what went wrong and what to do;
# the full text still goes to the server log.
_API_ERRORS = {
    "AuthenticationError": ("The Anthropic credential was rejected. Check ANTHROPIC_API_KEY, or run "
                            "ant auth status to see which credential the server picked up, then restart it."),
    "PermissionDeniedError": "That credential is not permitted to use this model.",
    "BadRequestError": ("The API rejected the request. A new account with no credit balance "
                        "fails this way; the server log carries the reason it gave."),
    "RateLimitError": "Rate limited by the API. Wait a moment and ask again.",
    "APIConnectionError": "Could not reach the Anthropic API. Check the network connection.",
    "APITimeoutError": "The API call timed out. Ask again, or use the query form.",
    "NotFoundError": "The configured model is not available to this credential.",
}


def _readable_api_error(exc: Exception) -> str:
    name = type(exc).__name__
    print(f"[bn] the assistant failed: {name}: {exc}", flush=True)
    return _API_ERRORS.get(name, f"The request failed ({name}). See the server log for detail.")


_views: dict[str, Any] = {}
_views_lock = threading.Lock()


def views() -> dict[str, Any]:
    """Compiled once per server: the schema and instances do not change while it runs."""
    with _views_lock:
        if not _views:
            _views.update(view_data())
        return _views


def examples() -> list[dict[str, Any]]:
    out = []
    for path in app_assembly_paths():
        a = load_assembly(path)
        out.append({"id": assembly_id_of(path), "label": a["label"], "network": a["network"],
                    "shakemap": shakemap_parameters(assembly_id_of(path)) is not None})
    return out


def example(assembly_id: str) -> dict[str, Any]:
    """What quakenet needs to draw one example and its controls, before anything is sampled."""
    path = assembly_path(assembly_id)
    compiled = compile_path(path, engine.library())
    network = compiled.network
    data = tbn_data(compiled)
    systems = [n for n in data["nodes"] if n["metadata"]["role"] == "system"]
    stored = attach_rules(data, model_dir(path))
    return {
        "assembly": compiled.assembly, "menu": engine.describe(assembly_id),
        "network": {"id": network.id, "title": network.title, "assumptions": network.assumptions,
                    "source": network.source, "tables": network.tables},
        "functions": [compiled.bindings[i].function["id"] for i in compiled.order],
        "objects": len(data["nodes"]), "variables": len(data["variables"]),
        # Every BN node and its parents, for the layered drawing; one joint object per stochastic Joint binding.
        "nodes": [{k: n[k] for k in ("id", "quantity", "entity", "row", "function", "parameter_set", "parents")}
                  for n in compiled.nodes()],
        "joint": sorted({n["metadata"]["function"] for n in data["nodes"] if len(n["childs"]) > 1}),
        "rules": {"stored": stored, "origins": len(systems)},
    }


# Posterior sessions, one per example, kept while the server runs. Sampling takes seconds to minutes, so a
# session's requests are served one at a time.
_sessions: dict[str, Any] = {}
_sessions_lock = threading.Lock()
_inference_jobs: dict[str, threading.Event] = {}
_inference_progress: dict[str, str] = {}           # request id -> what its computation is doing now
_inference_jobs_lock = threading.Lock()
# A Stop click can reach the server before its computation request registers.
_early_stops: deque[str] = deque(maxlen=128)


@contextmanager
def inference_job(request_id: str):
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise ValueError("request_id must be a nonempty string of at most 128 characters")
    event = threading.Event()
    with _inference_jobs_lock:
        if request_id in _inference_jobs:
            raise ValueError("that inference request is already running")
        if request_id in _early_stops:
            _early_stops.remove(request_id)
            event.set()
        _inference_jobs[request_id] = event

    def progress(text: str) -> None:
        with _inference_jobs_lock:
            _inference_progress[request_id] = text

    try:
        with cancellable(event, progress):
            yield
    finally:
        with _inference_jobs_lock:
            _inference_jobs.pop(request_id, None)
            _inference_progress.pop(request_id, None)


def inference_progress(request_id: str) -> str | None:
    """What the running request is doing now (report_progress), or None if it has said nothing or has ended."""
    with _inference_jobs_lock:
        return _inference_progress.get(request_id)


def stop_inference(request_id: str) -> None:
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise ValueError("request_id must be a nonempty string of at most 128 characters")
    # Deliberately independent of _sessions_lock: the computation holds that lock.
    with _inference_jobs_lock:
        event = _inference_jobs.get(request_id)
        if event is not None:
            event.set()
        elif request_id not in _early_stops:
            _early_stops.append(request_id)


@contextmanager
def session_update_lock():
    """Queued computations can also be stopped while another update owns the session lock."""
    while not _sessions_lock.acquire(timeout=0.05):
        check_cancelled()
    try:
        check_cancelled()
        yield
    finally:
        _sessions_lock.release()


SESSIONS = ROOT / "records" / "sessions"            # each example's latest session, to pick up after a restart


def _session_path(assembly_id: str) -> Path:
    return SESSIONS / (assembly_id.replace("/", "__") + ".pt")


def _session_bn(assembly_id: str, settings: dict[str, Any]) -> tuple[Any, Any, Any, dict[str, float], list[str]]:
    """The example's BN under the scenario settings: the clique tree's pieces, the tbnpy objects and variables,
    the towns' populations, and the functions computed."""
    from src.builder.bn_builder import build_tbn
    from src.inference import cliques

    compiled = engine.compile_query({"assembly": assembly_id, **settings})
    variables, probs = build_tbn(compiled)
    population = {r["id"]: r["population"] for r in compiled.network.rows("Node") if r.get("population") is not None}
    return (cliques.tree(probs), probs, variables, population,
            [compiled.bindings[i].function["id"] for i in compiled.order])


def save_session(assembly_id: str, s: Any) -> None:
    """Keep the session on disk after each change, so a restarted server opens it where it was."""
    s.save(_session_path(assembly_id), scenario=s.scenario, computed=s.computed)


def start_session(assembly_id: str, scenario: dict[str, Any] | None = None, samples: int | None = None) -> Any:
    """Build a candidate session (committed by the request handler): its BN under the scenario settings (rows, parameters,
    parameter_sets), sampled by the clique tree with no evidence; samples rows in all, 100 chains."""
    from src.inference import session

    settings = {k: v for k, v in (scenario or {}).items() if k in ("rows", "parameters", "parameter_sets")}
    settings = pin_shakemap(assembly_id, settings)
    t, probs, variables, population, computed = _session_bn(assembly_id, settings)
    n_keep = max(100, -(-int(samples or 200_000) // session.SAMPLER["n_chain"]))
    s = session.Session(t, probs, variables, population=population, sampler={"n_keep": n_keep})
    s.computed, s.scenario = computed, settings      # the assumptions the session was started under
    s.shakemap = shakemap_info(assembly_id, settings)
    return s


def pin_shakemap(assembly_id: str, settings: dict[str, Any]) -> dict[str, Any]:
    """The settings with the ShakeMap version they will use written in, when the assembly binds a
    ShakeMap and the settings leave its version open ("the current one").

    A session saved on disk is reopened by rebuilding its BN from these settings. Left open, a server
    restarted after USGS revised the map would rebuild it on the new version beneath samples drawn
    from the old one. Pinned, the session is the map it was run on, and the page offers the new one.
    """
    params = shakemap_parameters(assembly_id, settings)
    if params is None or params.get("version") is not None:
        return settings
    grid = shakemap.load(params["event_id"], params.get("source"), None, params.get("imt", "SA(1.0)"))
    version = shakemap._int(grid.version)
    if version is None:
        return settings
    pinned = copy.deepcopy(settings)
    pinned.setdefault("parameters", {}).setdefault("hazard.shakemap", {})["version"] = version
    return pinned


def shakemap_info(assembly_id: str, settings: dict[str, Any]) -> dict[str, Any] | None:
    """Which ShakeMap the settings use, for the page: the event, its version and where it came from."""
    params = shakemap_parameters(assembly_id, settings)
    if params is None:
        return None
    grid = shakemap.load(params["event_id"], params.get("source"), params.get("version"), params.get("imt", "SA(1.0)"))
    keys = ("magnitude", "lat", "lon", "depth_km", "time", "description", "originator", "map_status",
            "seismic_stations", "source")
    info = {k: grid.meta.get(k) for k in keys}
    # How many bridges the map reaches: the rest are bounded by its edge (kernels.hazard.ShakeMap).
    from src.monitor.watch import sites
    network = load_network(load_assembly(assembly_path(assembly_id))["network"], engine.library())
    on = sites(network)
    info["covered"], info["sites"] = int(grid.sample([s["lat"] for s in on], [s["lon"] for s in on])[2].sum()), len(on)
    if info["description"]:
        info["description"] = " ".join(info["description"].split())
    # The id asked for, not grid.xml's: a regional network's map may record it without its prefix
    # ("75337442" for nc75337442), and the catalogue knows only the full one.
    return {"event_id": params["event_id"], "version": shakemap._int(grid.version), **info}


RECENT_DAYS = 30
RECENT_TTL = 60.0                 # seconds; the page asks every 2 minutes, the USGS feed updates every minute
_recent: dict[str, Any] = {}
_recent_lock = threading.Lock()


def recent_earthquakes() -> list[dict[str, Any]]:
    """Events of the last ``RECENT_DAYS`` with a ShakeMap, at the watcher's magnitude floor or above,
    around a network the page offers a ShakeMap assembly for, whose map reaches that network's bridges;
    newest first. Each names its networks and the assemblies that can run it. Asked of USGS directly,
    so no watcher is needed for it."""
    from datetime import datetime, timedelta, timezone
    import time as _time

    from src.monitor.feedwatch import DEFAULT_MIN_MAGNITUDE
    from src.monitor.watch import shakemap_assemblies

    with _recent_lock:
        if _recent and _time.monotonic() - _recent["at"] < RECENT_TTL:
            return _recent["events"]
        networks: dict[str, dict[str, Any]] = {}
        for assembly_id, w in shakemap_assemblies([assembly_id_of(p) for p in app_assembly_paths()]).items():
            networks.setdefault(w["network"], {"bbox": w["bbox"], "sites": w["sites"], "assemblies": []})["assemblies"].append(assembly_id)
        start = (datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d")
        source, found = shakemap.ComcatSource(timeout=30), {}
        for network_id, n in sorted(networks.items()):
            near = source.search(n["bbox"], start=start, min_magnitude=DEFAULT_MIN_MAGNITUDE)
            for e in shakemap.covering(near, n["sites"], cache_dir=source.cache_dir)[0]:
                item = found.setdefault(e["event_id"], {**e, "networks": [], "assemblies": []})
                item["networks"].append(network_id)
                item["assemblies"] += n["assemblies"]
        events = sorted(found.values(), key=lambda e: e.get("time") or "", reverse=True)
        _recent.update(at=_time.monotonic(), events=events)
        return events


def past_earthquakes(network_id: str, min_magnitude: float | None = None) -> tuple[list[dict[str, Any]], int]:
    """(USGS events since 1900 whose ShakeMap reaches a bridge of the network, how many nearby reach none).

    The watcher's magnitude floor by default, so anything it emails can also be picked here. Near the network
    is not enough: a map that reaches no bridge cannot be run. Each event says how many bridges it reaches;
    the rest are given the map's edge value, an upper bound (kernels.hazard.ShakeMap).
    """
    from src.monitor.feedwatch import DEFAULT_MIN_MAGNITUDE
    from src.monitor.watch import sites

    network = load_network(network_id, engine.library())
    floor = DEFAULT_MIN_MAGNITUDE if min_magnitude is None else min_magnitude
    events = shakemap.search_events(network.rows("Node"), min_magnitude=floor)
    return shakemap.covering(events, sites(network))


def _warm_past_earthquakes() -> None:
    """Ask each ShakeMap network's past earthquakes once at start, so the map extents are on disk before the
    Earthquakes panel first opens (cold, toy-ca's 141 nearby maps take some 20 s; cached, under one)."""
    from src.monitor.watch import shakemap_assemblies

    try:
        networks = {w["network"] for w in shakemap_assemblies([assembly_id_of(p) for p in app_assembly_paths()]).values()}
        for network_id in sorted(networks):
            past_earthquakes(network_id)
    except Exception:                     # offline, or USGS busy: the panel asks again when opened
        pass


def archive_session(assembly_id: str) -> None:
    """Move the example's saved session to records/sessions/archive/, stamped with when it was last saved, so a new
    session never overwrites an old one."""
    import time as _time

    path = _session_path(assembly_id)
    if path.is_file():
        stamp = _time.strftime("%Y%m%d-%H%M%S", _time.localtime(path.stat().st_mtime))
        target = SESSIONS / "archive" / f"{path.stem}__{stamp}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)


def session_of(assembly_id: str, required: bool = True) -> Any:
    """The example's session: in memory, else the one saved on disk (after a restart), else none."""
    s = _sessions.get(assembly_id)
    path = _session_path(assembly_id)
    if s is None and path.is_file():
        from src.inference import session
        try:
            meta = torch_load_meta(path)
            t, probs, variables, _, _ = _session_bn(assembly_id, meta.get("scenario") or {})
            s, meta = session.Session.load(path, t, probs, variables)
            s.computed, s.scenario = meta.get("computed", []), meta.get("scenario") or {}
            s.shakemap = shakemap_info(assembly_id, s.scenario)
            _sessions[assembly_id] = s
        except InferenceCancelled:
            raise
        except Exception as exc:                       # e.g. the model changed since: start afresh with Run
            print(f"[bn] could not reopen the saved session of {assembly_id}: {type(exc).__name__}: {exc}", flush=True)
            s = None
    if s is None and required:
        raise ValueError(f"no session for {assembly_id!r}: press Run first")
    return s


def torch_load_meta(path: Path) -> dict[str, Any]:
    import torch
    return torch.load(path, weights_only=False)["meta"]


def session_reply(s: Any) -> dict[str, Any]:
    """The session's state for the page, with the map's numbers given the evidence so far."""
    return {**s.state(), "result": s.result(computed=s.computed), "scenario": getattr(s, "scenario", {}),
            "shakemap": getattr(s, "shakemap", None)}


# Rule extractions running in the background, one per (assembly, threshold). A separate process, so a
# long extraction neither holds up queries nor disturbs the random state their seeds fix.
_extractions: dict[tuple[str, float], subprocess.Popen] = {}
_extractions_lock = threading.Lock()


def extract_in_background(assembly: str, threshold: float, origins: list[str]) -> str:
    """Start ``python -m src rules`` for origins without rules of their own at this threshold; say so."""
    key = (assembly, threshold)
    with _extractions_lock:
        job = _extractions.get(key)
        if job is not None and job.poll() is None:
            return (f"rsr rules for target_add_h={threshold:g} are still being extracted in the background; "
                    "later runs will use them.")
        log = ROOT / ".cache" / f"rules-{assembly.replace('/', '_')}-{threshold:g}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as out:
            _extractions[key] = subprocess.Popen(
                [sys.executable, "-m", "src", "rules", assembly, "--target-add-h", repr(threshold), "--origins", *origins],
                cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    print(f"[quakenet] extracting rsr rules for {assembly} at target_add_h={threshold:g}: {', '.join(origins)}", flush=True)
    return (f"Extracting rsr rules for target_add_h={threshold:g} in the background, for {len(origins)} origin(s); "
            f"later runs will use them (log: .cache/{log.name}).")


class Handler(BaseHTTPRequestHandler):
    def _json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_request(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def _page(self, path: Path, content_type: str = "text/html; charset=utf-8") -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        url = urlparse(self.path)
        if url.path in {"/", "/index.html"}:
            return self._page(PAGE)
        if url.path == "/diagram-navigation.js":
            return self._page(PAGE.with_name("diagram-navigation.js"), "text/javascript; charset=utf-8")
        if url.path == "/bn":
            return self._page(BN_PAGE)
        if url.path == "/api/examples":
            return self._json({"examples": examples()})
        if url.path == "/api/example":
            try:
                return self._json(example((parse_qs(url.query).get("assembly") or [""])[0]))
            except (RegistryError, engine.QueryError) as exc:
                problems = exc.problems if isinstance(exc, RegistryError) else [str(exc)]
                return self._json({"error": "; ".join(problems)}, HTTPStatus.NOT_FOUND)
        if url.path == "/api/views":
            return self._json(views())
        if url.path in {"/api/session", "/api/session/node"}:
            params = parse_qs(url.query)
            assembly = (params.get("assembly") or [""])[0]
            with _sessions_lock:
                if url.path == "/api/session":
                    s = session_of(assembly, required=False)
                    return self._json({"session": session_reply(s) if s else None})
                try:
                    return self._json(session_of(assembly).distribution((params.get("node") or [""])[0]))
                except ValueError as exc:
                    return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        if url.path == "/api/session/progress":
            return self._json({"progress": inference_progress((parse_qs(url.query).get("request_id") or [""])[0])})
        if url.path == "/api/inference":
            # The clique tree's message passing as src/inference/cliques.py declares it, for the Clique tree tab.
            from src.inference import cliques
            return self._json(cliques.describe())
        if url.path == "/api/chat-status":
            ok, reason = chat.available()
            return self._json({"available": ok, "reason": reason, "provider": chat.provider(),
                               "model": chat.model_name()})
        if url.path == "/api/shakemap-events":
            params = parse_qs(url.query)
            try:
                floor = params.get("min_magnitude")
                events, outside = past_earthquakes((params.get("network") or [""])[0],
                                                   float(floor[0]) if floor else None)
            except RegistryError as exc:
                return self._json({"error": "; ".join(exc.problems)}, HTTPStatus.NOT_FOUND)
            except ValueError as exc:                  # a ShakeMapError, or a min_magnitude that is no number
                return self._json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            return self._json({"events": events, "outside": outside})
        if url.path == "/api/recent-earthquakes":
            try:
                return self._json({"days": RECENT_DAYS, "events": recent_earthquakes()})
            except shakemap.ShakeMapError as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
        if url.path.startswith("/api/shakemap-version/"):
            event_id = url.path.removeprefix("/api/shakemap-version/")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", event_id):
                return self._json({"error": f"{event_id!r} is not a USGS event id"}, HTTPStatus.BAD_REQUEST)
            try:
                return self._json({"event_id": event_id, "version": shakemap.ComcatSource(timeout=30).current_version(event_id)})
            except shakemap.UnknownEvent as exc:
                return self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            except shakemap.ShakeMapError as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/api/session/stop":
            try:
                stop_inference(self._read_request().get("request_id"))
                return self._json({"stopping": True})
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        if path == "/api/query":
            try:
                query, notes = self._read_request(), []
                # Started before sampling, from what is stored: a slow run does not hold the extraction up.
                result = engine.answer(query, on_missing_rules=lambda threshold, origins: notes.append(
                    extract_in_background(query.get("assembly", ""), threshold, origins)), record={"source": "page"})
                result["warnings"] += notes
                return self._json(result)
            except (engine.QueryError, json.JSONDecodeError) as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except ImportError as exc:
                traceback.print_exc()
                module = f" ({exc.name})" if exc.name else ""
                return self._json({"error": (
                    f"The server's Python environment cannot load a required inference dependency{module}. "
                    "Restart the app with: conda run -n quakenet python -m src serve. "
                    "See the server log for the full error.")}, HTTPStatus.SERVICE_UNAVAILABLE)
            except Exception as exc:
                traceback.print_exc()
                return self._json({"error": (
                    f"The query could not be computed ({type(exc).__name__}). "
                    "See the server log for the full error.")}, HTTPStatus.INTERNAL_SERVER_ERROR)
        if path in {"/api/session", "/api/session/query", "/api/session/evidence"}:
            return self._session(path)
        if path == "/api/session/new":
            return self._new_session()
        if path == "/api/session/draft":
            return self._draft()
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _draft(self) -> None:
        """The assistant's draft of a decision query or an observation, from a sentence; nothing is added."""
        from src.llm import drafts
        try:
            body = self._read_request()
            assembly = body.get("assembly", "")
            with _sessions_lock:
                state = session_of(assembly).state()
            network = load_network(load_assembly(assembly_path(assembly))["network"], engine.library())
            bridges: dict[str, list[str]] = {}
            for b in network.rows("Bridge"):
                bridges.setdefault(b["edge"], []).append(b["id"])
            info = {"title": network.title, "description": load_assembly(assembly_path(assembly)).get("description", ""),
                    "assumptions": network.assumptions,
                    "edges": [{"id": e["id"], "ends": [e["source"], e["target"]], "bridges": bridges.get(e["id"], [])}
                              for e in network.rows("Edge")]}
            return self._json(drafts.draft(body.get("kind", ""), body.get("text", ""), state, info, body.get("history") or []))
        except chat.ChatUnavailable as exc:
            return self._json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except (RegistryError, ValueError, json.JSONDecodeError) as exc:
            problems = exc.problems if isinstance(exc, RegistryError) else [str(exc)]
            return self._json({"error": "; ".join(problems)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            return self._json({"error": _readable_api_error(exc)}, HTTPStatus.BAD_GATEWAY)

    def _new_session(self) -> None:
        """New session: the example's session is archived and none is kept, so the page is as on first opening, with
        nothing computed until Run."""
        try:
            assembly = self._read_request().get("assembly", "")
            assembly_path(assembly)                            # an example the server knows, or a RegistryError
            with session_update_lock():
                archive_session(assembly)
                _sessions.pop(assembly, None)
            return self._json({"session": None})
        except (RegistryError, ValueError, json.JSONDecodeError) as exc:
            problems = exc.problems if isinstance(exc, RegistryError) else [str(exc)]
            return self._json({"error": "; ".join(problems)}, HTTPStatus.BAD_REQUEST)

    def _session(self, path: str) -> None:
        """Start a session, or add a decision query or an observation to it; reply with its state."""
        try:
            body = self._read_request()
            assembly = body.get("assembly", "")
            with inference_job(body.get("request_id") or str(uuid4())), session_update_lock():
                check_cancelled()
                if path == "/api/session":
                    s = start_session(assembly, body.get("scenario"), body.get("samples"))
                else:
                    # Samples are read-only inputs to an update; mutable session metadata belongs to the candidate.
                    # A cancelled candidate is discarded, without touching the live or saved session.
                    s = copy.copy(session_of(assembly))
                    s.evidence, s.queries, s.timeline = dict(s.evidence), dict(s.queries), list(s.timeline)
                    if path == "/api/session/query":
                        s.add_query(body.get("query") or {}, at=body.get("at"))
                    else:
                        s.observe(body.get("variable", ""), body.get("value"), at=body.get("at"),
                                  source=body.get("source"), text=body.get("text"))
                reply = session_reply(s)
                check_cancelled()
                # Commit only complete results. A stop arriving during this final save is too late to cancel.
                if path == "/api/session":
                    archive_session(assembly)
                save_session(assembly, s)
                _sessions[assembly] = s
                return self._json(reply)
        except InferenceCancelled as exc:
            return self._json({"error": str(exc), "cancelled": True}, HTTPStatus.CONFLICT)
        except (RegistryError, ValueError, json.JSONDecodeError, engine.QueryError) as exc:
            problems = exc.problems if isinstance(exc, RegistryError) else [str(exc)]
            return self._json({"error": "; ".join(problems)}, HTTPStatus.BAD_REQUEST)
        except ImportError as exc:
            traceback.print_exc()
            return self._json({"error": (f"The session needs {exc.name or 'a module'}, which this Python "
                                         f"({sys.executable}) cannot load. Start the server in the quakenet "
                                         "environment (conda activate quakenet), with the rsr working tree on "
                                         "PYTHONPATH (see .env).")},
                              HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception as exc:
            traceback.print_exc()
            return self._json({"error": f"The session could not be updated ({type(exc).__name__}). "
                                        "See the server log."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[bn] {self.address_string()} {format % args}")


class _Server(ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets a second server bind a port that one is already listening on,
    # and requests then reach either: an old server keeps answering beside a new one. Refuse instead.
    allow_reuse_address = sys.platform != "win32"

    def handle_error(self, request, client_address):
        # A page reloaded or closed while its request was still being answered (reopening a session can take
        # a while) leaves nobody to read the reply: that is the browser leaving, not a server fault.
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return _Server((host, port), Handler)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False, example: str | None = None) -> None:
    try:
        server = make_server(host, port)
    except OSError as exc:
        raise SystemExit(f"Port {port} is already in use, probably by another quakenet server. Stop that one "
                         f"(Ctrl+C in its window), or start this one with --port {port + 1}. ({exc})") from None
    url = f"http://{host}:{server.server_address[1]}" + (f"/#example={example}" if example else "")
    views()                               # compile everything now, so a broken instance fails at start
    ok, reason = chat.available()
    print(f"quakenet: {url}", flush=True)
    print(f"Language model: {chat.provider()} ({chat.model_name()}), " + ("ready" if ok else f"not available. {reason}"),
          flush=True)
    if open_browser:
        webbrowser.open(url)
    if chat.provider() == "ollama":
        threading.Thread(target=lambda: print(f"[bn] {chat.preload_ollama()}", flush=True), daemon=True).start()
    threading.Thread(target=_warm_past_earthquakes, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

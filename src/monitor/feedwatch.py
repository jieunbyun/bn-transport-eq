"""Note new ShakeMaps near the example networks: the feed-only watcher.

The always-on part of monitoring, and nothing else. It polls the USGS real-time
feed, keeps events with a ShakeMap whose epicentre lies in one of the networks'
regions, and records each new ShakeMap *version* of them. It downloads no grid
and runs no BN: USGS keeps every version, and the machine running quakenet
fetches the one it needs when an event is opened.

What it adds over quakenet's own Earthquakes panel: it runs while quakenet is
closed. An earthquake near a network is emailed once, when its first ShakeMap
appears (``notify``; in the Lambda, an SNS topic). Every event found in one poll
goes in one email, so an aftershock sequence does not arrive as a flood.
Revisions of a ShakeMap are not emailed: the open page shows those.

It keeps two small files, which a Lambda has to keep in S3, since it holds
nothing between runs:

* ``watch/state.json`` -- each event, its networks and its ShakeMap versions,
  and whether it has been emailed;
* ``watch/heartbeat.json`` -- the last poll, read by ``shakemap heartbeat``.

``regions.json`` holds each network's box, written by ``python -m src shakemap
lambda-zip`` from the ShakeMap assemblies, so this module needs neither the
registry nor numpy.

Standard library only, plus boto3 when the store is S3 -- the AWS Lambda Python
runtime provides it. The zip is this file and ``regions.json``; the handler is
``feedwatch.lambda_handler``. Keep this module free of imports from the rest of
``src``, or the Lambda will not start.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{feed}.geojson"
DETAIL_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USER_AGENT = "quakenet (transport seismic BN; ShakeMap feed watcher)"
DEFAULT_FEED = "2.5_day"          # a day of history, so a watcher that was down a while catches up
DEFAULT_MIN_MAGNITUDE = 4.0       # below this no bridge is at risk, and New England fires M2.5s weekly
STATE_KEY = "watch/state.json"
HEARTBEAT_KEY = "watch/heartbeat.json"
METRIC_NAMESPACE = "quakenet"


class WatchError(RuntimeError):
    """A poll could not complete."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get(url: str, timeout: float = 30):
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)


def fetch_feed(feed: str = DEFAULT_FEED) -> dict[str, Any]:
    url = FEED_URL.format(feed=feed)
    try:
        with _get(url) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise WatchError(f"could not read the USGS feed {url}: {exc}") from exc


def fetch_detail(event_id: str) -> dict[str, Any]:
    """The event's catalogue entry, with its products and their download URLs."""
    with _get(f"{DETAIL_URL}?{urlencode({'eventid': event_id, 'format': 'geojson'})}") as response:
        return json.loads(response.read().decode("utf-8"))


# ---- storage ---------------------------------------------------------------------


class DirStore:
    """Keys as files. Each top-level key folder maps to a directory, so the local
    layout can follow quakenet's settings (``QUAKENET_WATCH_DIR``)."""

    def __init__(self, roots: dict[str, Path | str]):
        self.roots = {name: Path(path) for name, path in roots.items()}

    def _path(self, key: str) -> Path:
        head, _, rest = key.partition("/")
        return self.roots[head] / rest

    def read_json(self, key: str, default: Any = None) -> Any:
        try:
            return json.loads(self._path(key).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default

    def write_json(self, key: str, value: Any) -> None:
        self.put(key, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        partial.write_bytes(data)
        partial.replace(path)            # a reader never sees half a file


class S3Store:
    """Keys as objects under ``prefix`` in one bucket. An S3 write is atomic.
    Reading a missing key needs ``s3:ListBucket``: without it S3 answers
    "access denied" rather than "no such key"."""

    def __init__(self, bucket: str, prefix: str = "", client: Any = None):
        if client is None:
            import boto3                 # in the Lambda runtime; locally only when S3 is used

            client = boto3.client("s3")
        self.bucket, self.prefix, self.client = bucket, prefix.strip("/"), client

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def read_json(self, key: str, default: Any = None) -> Any:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return default
        return json.loads(body.decode("utf-8"))

    def write_json(self, key: str, value: Any) -> None:
        self.put(key, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)


# ---- one poll --------------------------------------------------------------------


def _inside(bbox, lon: float, lat: float) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def current_version(detail: dict[str, Any]) -> int:
    """The version of the event's preferred ShakeMap, from its catalogue entry."""
    products = detail.get("properties", {}).get("products", {}).get("shakemap") or []
    if not products:
        raise WatchError("the catalogue lists no ShakeMap for it")
    return int(float(products[0].get("properties", {}).get("version", 1)))   # preferred first


def poll(
    store,
    regions: dict[str, list[float]],
    *,
    feed: str = DEFAULT_FEED,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    fetch: Callable[[str], dict[str, Any]] = fetch_feed,
    detail: Callable[[str], dict[str, Any]] = fetch_detail,
    notify: Callable[[list[dict[str, Any]]], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """One poll: feed, match, note new versions, notify, record. Returns the heartbeat.

    The heartbeat is written whatever happens. A failure to read the feed is
    re-raised after it is recorded; a failure on one event is recorded against
    that event and the poll carries on with the others. A failed notification
    is retried on the next poll: an event counts as notified only once
    ``notify`` has returned.
    """
    started = time.monotonic()
    heartbeat: dict[str, Any] = {"last_poll_utc": _now(), "feed": feed, "status": "ok", "mode": "fetch-only",
                                 "networks": sorted(regions), "events_in_feed": 0, "events_matched": 0,
                                 "updates": [], "notified": [], "errors": []}
    try:
        state = store.read_json(STATE_KEY, {"events": {}})
        features = fetch(feed).get("features", [])
        heartbeat["events_in_feed"] = len(features)

        for feature in features:
            props = feature.get("properties") or {}
            event_id = feature.get("id")
            lon, lat = (feature.get("geometry") or {}).get("coordinates", [None, None])[:2]
            if (not event_id or lon is None or "shakemap" not in (props.get("types") or "").split(",")
                    or (props.get("mag") or 0) < min_magnitude):
                continue
            matched = sorted(name for name, bbox in regions.items() if _inside(bbox, lon, lat))
            if not matched:
                continue
            heartbeat["events_matched"] += 1
            record = state["events"].setdefault(event_id, {})
            if not record:
                record["notified"] = False        # first seen now; events kept from before were not
            record.update(magnitude=props.get("mag"), place=props.get("place"), lat=lat, lon=lon,
                          time=props.get("time"), url=props.get("url"), networks=matched)
            if record.get("updated") == props.get("updated") and not record.get("error"):
                continue                          # nothing new since the last poll

            try:
                version = current_version(detail(event_id))
            except (OSError, ValueError, WatchError) as exc:
                heartbeat["errors"].append(f"{event_id}: {exc}")
                record["error"] = str(exc)
                continue
            new = version not in record.get("versions", [])
            versions = sorted({*record.get("versions", []), version})
            record.update(updated=props.get("updated"), version=version, versions=versions, error=None)
            if new:
                heartbeat["updates"].append({"event_id": event_id, "version": version, "networks": matched})
                log(f"[feedwatch] {event_id} v{version} M{props.get('mag')} {props.get('place')}: "
                    f"new ShakeMap for {', '.join(matched)}")

        # Only events whose ShakeMap is known: an email should point at a map that exists.
        pending = [{"event_id": event_id, **record} for event_id, record in state["events"].items()
                   if record.get("notified") is False and record.get("version") is not None]
        if pending and notify:
            try:
                notify(pending)
            except Exception as exc:              # noqa: BLE001 - any failure: keep them pending, retry
                heartbeat["errors"].append(f"notification failed: {exc}")
            else:
                for item in pending:
                    state["events"][item["event_id"]]["notified"] = True
                heartbeat["notified"] = [item["event_id"] for item in pending]

        state["last_poll_utc"] = heartbeat["last_poll_utc"]
        store.write_json(STATE_KEY, state)
    except WatchError as exc:
        heartbeat.update(status="error", errors=[*heartbeat["errors"], str(exc)])
        raise
    finally:
        if heartbeat["errors"] and heartbeat["status"] == "ok":
            heartbeat["status"] = "degraded"      # the feed worked, some events did not
        heartbeat["duration_s"] = round(time.monotonic() - started, 1)
        store.write_json(HEARTBEAT_KEY, heartbeat)
    return heartbeat


# ---- the email -------------------------------------------------------------------


def _when(ms: Any) -> str:
    # Not datetime.fromtimestamp, which fails on Windows for times before 1970.
    when = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=ms) if ms else None
    return when.strftime("%Y-%m-%d %H:%M UTC") if when else "time unknown"


def compose_email(events: list[dict[str, Any]]) -> tuple[str, str]:
    """(subject, body) for one or more new earthquakes."""
    events = sorted(events, key=lambda e: e.get("time") or 0)
    networks = sorted({n for e in events for n in e.get("networks", [])})
    if len(events) == 1:
        e = events[0]
        subject = f"M{e.get('magnitude')} earthquake near {', '.join(networks)}: {e.get('place') or e['event_id']}"
    else:
        strongest = max(e.get("magnitude") or 0 for e in events)
        subject = f"{len(events)} earthquakes near {', '.join(networks)}, up to M{strongest}"
    blocks = []
    for e in events:
        blocks.append("\n".join([
            f"M{e.get('magnitude')}  {e.get('place') or ''}",
            f"  {_when(e.get('time'))}, epicentre {e.get('lat')}, {e.get('lon')}",
            f"  Networks: {', '.join(e.get('networks', []))}",
            f"  USGS event id: {e['event_id']} (ShakeMap version {e.get('version')})",
            f"  {e.get('url') or 'https://earthquake.usgs.gov/earthquakes/eventpage/' + e['event_id']}",
        ]))
    body = ("A ShakeMap is available for the following earthquake(s) near quakenet's example networks.\n\n"
            + "\n\n".join(blocks)
            + "\n\nTo run one: start quakenet (python -m src serve), open Earthquakes and pick the event, "
              "or run python -m src shakemap watch --once. Revised ShakeMaps and later events are shown "
              "on the page while it is open; this email is sent once per event.\n")
    return subject[:100], body               # SNS caps an email subject at 100 characters


def sns_notifier(topic_arn: str, client: Any = None) -> Callable[[list[dict[str, Any]]], None]:
    if client is None:
        import boto3

        client = boto3.client("sns")

    def notify(events: list[dict[str, Any]]) -> None:
        subject, body = compose_email(events)
        client.publish(TopicArn=topic_arn, Subject=subject, Message=body)

    return notify


# ---- AWS Lambda ------------------------------------------------------------------


def load_regions(path: Path | str | None = None) -> dict[str, list[float]]:
    path = Path(path or os.environ.get("FEEDWATCH_REGIONS") or Path(__file__).with_name("regions.json"))
    regions = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in regions.items() if not k.startswith("_")}


def _put_health(healthy: bool, cloudwatch: Any = None) -> None:
    """1 per good poll, 0 per failed one. The alarm treats a missing value as a
    failure too, so a schedule that stops firing is caught as well."""
    if cloudwatch is None:
        import boto3

        cloudwatch = boto3.client("cloudwatch")
    cloudwatch.put_metric_data(Namespace=os.environ.get("FEEDWATCH_METRIC_NAMESPACE", METRIC_NAMESPACE),
                               MetricData=[{"MetricName": "WatcherHealthy", "Value": 1.0 if healthy else 0.0}])


def lambda_handler(event: Any = None, context: Any = None, *, store: Any = None, cloudwatch: Any = None,
                   **poll_kwargs: Any) -> dict[str, Any]:
    """Entry point for a scheduled Lambda. Configured by environment:
    ``FEEDWATCH_BUCKET`` (required), ``FEEDWATCH_TOPIC_ARN`` (the SNS topic new
    earthquakes are emailed to; none, no email), ``FEEDWATCH_PREFIX``,
    ``FEEDWATCH_FEED``, ``FEEDWATCH_MIN_MAGNITUDE``. Run it with reserved
    concurrency 1, so polls never overlap."""
    if store is None:
        store = S3Store(os.environ["FEEDWATCH_BUCKET"], os.environ.get("FEEDWATCH_PREFIX", ""))
    if "notify" not in poll_kwargs and os.environ.get("FEEDWATCH_TOPIC_ARN"):
        poll_kwargs["notify"] = sns_notifier(os.environ["FEEDWATCH_TOPIC_ARN"])
    poll_kwargs.setdefault("feed", os.environ.get("FEEDWATCH_FEED", DEFAULT_FEED))
    poll_kwargs.setdefault("min_magnitude", float(os.environ.get("FEEDWATCH_MIN_MAGNITUDE", DEFAULT_MIN_MAGNITUDE)))
    try:
        beat = poll(store, load_regions(), **poll_kwargs)
    except WatchError:
        _put_health(False, cloudwatch)
        raise                             # also counts in the function's own Errors metric
    _put_health(True, cloudwatch)
    return {k: beat[k] for k in ("status", "events_in_feed", "events_matched", "updates", "notified", "errors")}

"""USGS ShakeMap as a hazard input: find events, fetch grids, sample at sites.

A ShakeMap is the ground motion an earthquake actually produced, estimated on a
regular latitude/longitude grid from recordings, felt reports and a ground-motion
model. Its ``grid.xml`` carries the median of each IM (``PGA``, ``PGV``, ``PSA03``,
``PSA10``, ``PSA30``) and its ``uncertainty.xml`` the log standard deviation of each
(``STDPGA``, ``STDPSA10``, ...) at each
grid point. Those two numbers are what the ``hazard.shakemap`` function samples at each
site, so a ShakeMap replaces the scenario-plus-GMM path rather than adding to it.

Two sources, one reader:

* ``ComcatSource`` -- the USGS earthquake catalogue (ComCat). Searches events
  that have a ShakeMap and downloads the preferred one. Downloads are cached on
  disk in the same layout ShakeCast uses, so a later run needs no network.
* ``ShakeCastSource`` -- the data directory of a running USGS ShakeCast server
  (v4). ShakeCast already polls ComCat and saves every ShakeMap version as
  ``<data>/<event_id>/<event_id>-<version>/grid.xml`` (and ``uncertainty.xml``
  when its ``eq_pref_products`` include it, as the default configuration does).
  Reading that folder means the registry follows whatever ShakeCast has received,
  with no second poller. ShakeCast's own root is ``SHAKECAST_USER_DIRECTORY``,
  else ``~/.shakecast``.

By default (``auto``) ShakeCast is tried first and the catalogue fills the
gaps: events ShakeCast never kept, or has already deleted.

Standard library and numpy only.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

COMCAT_QUERY = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USER_AGENT = "bn-on-demand (transport seismic BN; ShakeMap hazard input)"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = Path(
    os.environ.get("BN_SHAKEMAP_CACHE")
    or ROOT / ".cache" / "shakemaps"
)

# grid.xml unit strings for spectral acceleration, by ShakeMap generation.
_TO_G = {"%g": 0.01, "pctg": 0.01, "g": 1.0}
_TO_CMS = {"cms": 1.0, "cm/s": 1.0}

# Our IM quantity -> (grid.xml field, uncertainty.xml field, unit conversions to the quantity's unit).
IMT_FIELDS = {
    "PGA": ("PGA", "STDPGA", _TO_G),
    "PGV": ("PGV", "STDPGV", _TO_CMS),
    "SA(0.3)": ("PSA03", "STDPSA03", _TO_G),
    "SA(1.0)": ("PSA10", "STDPSA10", _TO_G),
    "SA(3.0)": ("PSA30", "STDPSA30", _TO_G),
}


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def utc_from_ms(ms: float) -> datetime:
    """A catalogue time (milliseconds since 1970, UTC) as a datetime. Not datetime.fromtimestamp, which
    fails on Windows for times before 1970, and the catalogue goes back to 1900."""
    return _EPOCH + timedelta(milliseconds=ms)


class ShakeMapError(ValueError):
    """A ShakeMap could not be found, fetched or read."""


class UnknownEvent(ShakeMapError):
    """The catalogue has no event with that id."""


# ---- the grid --------------------------------------------------------------


@dataclass(frozen=True)
class ShakeMapGrid:
    """One IM on a regular grid, rows north to south, columns west to east."""

    event_id: str
    version: str
    meta: dict[str, Any]
    lon_min: float
    lat_max: float
    dlon: float
    dlat: float
    ln_im: np.ndarray                        # (nlat, nlon), ln of the IM in its quantity's unit
    std_im: np.ndarray | None = None         # (nlat, nlon), ln units; None if no uncertainty.xml
    imt: str = "SA(1.0)"
    warnings: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.ln_im.shape

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(lon_min, lat_min, lon_max, lat_max) of the grid points."""
        nlat, nlon = self.shape
        return (self.lon_min, self.lat_max - (nlat - 1) * self.dlat,
                self.lon_min + (nlon - 1) * self.dlon, self.lat_max)

    def sample(self, lats: Sequence[float], lons: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bilinear (ln median IM in its unit, sigma, inside-grid flag) at each site.

        Interpolation is done on ln SA, the scale ShakeMap's medians and
        uncertainties live on. A site outside the grid gets NaN and ``False``;
        the caller decides what that means rather than this method guessing.
        """
        lats, lons = np.asarray(lats, float), np.asarray(lons, float)
        nlat, nlon = self.shape
        row = (self.lat_max - lats) / self.dlat
        col = (lons - self.lon_min) / self.dlon
        inside = (row >= 0) & (row <= nlat - 1) & (col >= 0) & (col <= nlon - 1)
        r0 = np.clip(np.floor(row).astype(int), 0, max(nlat - 2, 0))
        c0 = np.clip(np.floor(col).astype(int), 0, max(nlon - 2, 0))
        fr, fc = np.clip(row - r0, 0, 1), np.clip(col - c0, 0, 1)

        def interp(field_: np.ndarray) -> np.ndarray:
            r1, c1 = np.minimum(r0 + 1, nlat - 1), np.minimum(c0 + 1, nlon - 1)
            top = field_[r0, c0] * (1 - fc) + field_[r0, c1] * fc
            bottom = field_[r1, c0] * (1 - fc) + field_[r1, c1] * fc
            return top * (1 - fr) + bottom * fr

        ln_sa = np.where(inside, interp(self.ln_im), np.nan)
        sigma = (np.where(inside, interp(self.std_im), np.nan)
                 if self.std_im is not None else np.where(inside, 0.0, np.nan))
        return ln_sa, sigma, inside

    def nearest_on_map(self, lats: Sequence[float], lons: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Each site moved to the nearest point of the map's rectangle (unchanged if it lies on it)."""
        lon_min, lat_min, lon_max, lat_max = self.extent
        return np.clip(np.asarray(lats, float), lat_min, lat_max), np.clip(np.asarray(lons, float), lon_min, lon_max)


def _parse_grid_xml(path: Path, wanted: str) -> tuple[dict[str, Any], dict[str, float], np.ndarray, str]:
    """Header attributes, grid spec, the ``wanted`` field as (nlat, nlon), and its units.

    The header is parsed as XML; the data block, which is most of a ~30 MB file,
    is read as one whitespace-separated array, which is far faster than parsing
    it element by element.
    """
    text = path.read_text(encoding="utf-8")
    start, end = text.find("<grid_data>"), text.find("</grid_data>")
    if start < 0 or end < 0:
        raise ShakeMapError(f"{path} is not a ShakeMap grid: no <grid_data> block")
    header = ET.fromstring(text[:start] + "</shakemap_grid>")
    strip = lambda tag: tag.rsplit("}", 1)[-1]          # drop the XML namespace
    root_attrs = dict(header.attrib)
    event_attrs, spec, fields = {}, {}, {}
    for child in header:
        name = strip(child.tag)
        if name == "event":
            event_attrs = dict(child.attrib)
        elif name == "grid_specification":
            spec = {k: float(v) for k, v in child.attrib.items()}
        elif name == "grid_field":
            fields[child.attrib["name"].upper()] = (int(child.attrib["index"]) - 1, child.attrib.get("units", ""))
    if wanted not in fields:
        raise ShakeMapError(f"{path.name} has no {wanted} field; it carries {sorted(fields)}")

    data = np.array(text[start + len("<grid_data>"):end].split(), dtype=float)
    nlat, nlon = int(spec["nlat"]), int(spec["nlon"])
    if data.size != nlat * nlon * len(fields):
        raise ShakeMapError(f"{path.name}: expected {nlat}x{nlon} rows of {len(fields)} values, got {data.size} values")
    table = data.reshape(nlat * nlon, len(fields))

    # Do not trust the row order to a convention: check it against the LAT/LON columns.
    lat, lon = table[:, fields["LAT"][0]], table[:, fields["LON"][0]]
    if not (np.isclose(lat[0], spec["lat_max"], atol=1e-3) and np.isclose(lon[0], spec["lon_min"], atol=1e-3)
            and lon[1] > lon[0] and lat[nlon] < lat[0]):
        raise ShakeMapError(f"{path.name}: grid rows are not ordered north-west first as expected")
    index, units = fields[wanted]
    meta = {**root_attrs, "event": event_attrs, "grid": spec}
    return meta, spec, table[:, index].reshape(nlat, nlon), units


def read_grid(directory: Path | str, imt: str = "SA(1.0)") -> ShakeMapGrid:
    """Read ``grid.xml`` (required) and ``uncertainty.xml`` (if present) from one folder."""
    directory = Path(directory)
    grid_path = directory / "grid.xml"
    if not grid_path.is_file():
        raise ShakeMapError(f"no grid.xml in {directory}")
    if imt not in IMT_FIELDS:
        raise ShakeMapError(f"no ShakeMap grid for {imt!r}; it has {', '.join(IMT_FIELDS)}")
    field_name, std_name, units_to = IMT_FIELDS[imt]
    meta, spec, psa, units = _parse_grid_xml(grid_path, field_name)
    scale = units_to.get(units.strip().lower())
    if scale is None:
        raise ShakeMapError(f"grid.xml gives {field_name} in {units!r}; expected {', '.join(units_to)}")
    # Zero shaking is legal in a grid; floor it so the log stays finite and tiny.
    ln_psa = np.log(np.maximum(psa * scale, 1e-6))

    warnings: list[str] = []
    std = None
    unc_path = directory / "uncertainty.xml"
    if unc_path.is_file():
        _, unc_spec, std, _ = _parse_grid_xml(unc_path, std_name)
        if (unc_spec["nlat"], unc_spec["nlon"]) != (spec["nlat"], spec["nlon"]):
            raise ShakeMapError("uncertainty.xml and grid.xml are on different grids")
    else:
        warnings.append(
            "This ShakeMap has no uncertainty.xml, so ground-motion variability is omitted "
            "and damage probabilities are understated."
        )

    event = meta.get("event", {})
    return ShakeMapGrid(
        event_id=meta.get("event_id") or event.get("event_id", directory.parent.name),
        version=str(meta.get("shakemap_version", "")),
        meta={
            "event_id": meta.get("event_id"),
            "version": meta.get("shakemap_version"),
            "originator": meta.get("shakemap_originator"),
            "map_status": meta.get("map_status"),
            "process_timestamp": meta.get("process_timestamp"),
            "magnitude": _float(event.get("magnitude")),
            "depth_km": _float(event.get("depth")),
            "lat": _float(event.get("lat")),
            "lon": _float(event.get("lon")),
            "time": event.get("event_timestamp"),
            "description": event.get("event_description"),
            "seismic_stations": _int(event.get("seismic_stations")),
            "intensity_observations": _int(event.get("intensity_observations")),
            "directory": str(directory),
        },
        # Spacing from the bounds, not the "nominal" attributes: those are rounded
        # (0.0167 for 1/60 degree), which drifts ~2 km across a 600-row grid.
        lon_min=spec["lon_min"], lat_max=spec["lat_max"],
        dlon=(spec["lon_max"] - spec["lon_min"]) / (spec["nlon"] - 1),
        dlat=(spec["lat_max"] - spec["lat_min"]) / (spec["nlat"] - 1),
        ln_im=ln_psa, std_im=std, imt=imt, warnings=warnings,
    )


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _float(value)
    return None if number is None else int(number)


# ---- where grids come from ---------------------------------------------------


def _version_dirs(event_dir: Path) -> list[tuple[int, Path]]:
    """``<id>-<version>`` folders holding a grid.xml, oldest first (ShakeCast layout)."""
    found = []
    for child in event_dir.iterdir() if event_dir.is_dir() else []:
        match = re.fullmatch(re.escape(event_dir.name) + r"-(\d+)", child.name)
        if match and (child / "grid.xml").is_file():
            found.append((int(match.group(1)), child))
    return sorted(found)


class ShakeCastSource:
    """ShakeMaps a ShakeCast server has already downloaded."""

    name = "shakecast"

    def __init__(self, root: Path | str | None = None):
        user_dir = Path(root or os.environ.get("SHAKECAST_USER_DIRECTORY") or Path.home() / ".shakecast")
        # Accept either ShakeCast's user directory or its data/ folder.
        self.data_dir = user_dir / "data" if (user_dir / "data").is_dir() else user_dir

    def events(self) -> list[dict[str, Any]]:
        """Every event with at least one grid, newest version of each."""
        if not self.data_dir.is_dir():
            raise ShakeMapError(f"no ShakeCast data directory at {self.data_dir}")
        out = []
        for event_dir in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            versions = _version_dirs(event_dir)
            if versions:
                out.append(_summary_from_grid_header(versions[-1][1] / "grid.xml", event_dir.name))
        return out

    def directory(self, event_id: str, version: int | None = None) -> Path:
        versions = _version_dirs(self.data_dir / event_id)
        if not versions:
            raise ShakeMapError(f"ShakeCast has no ShakeMap for {event_id!r} in {self.data_dir}")
        if version is None:
            return versions[-1][1]
        for number, path in versions:
            if number == int(version):
                return path
        raise ShakeMapError(f"ShakeCast has {event_id!r} versions {[n for n, _ in versions]}, not {version}")


def _summary_from_grid_header(path: Path, event_id: str) -> dict[str, Any]:
    head = path.read_text(encoding="utf-8")[:4000]
    attr = lambda name: (re.search(rf'\b{name}="([^"]*)"', head) or [None, None])[1]
    return {"event_id": event_id, "version": attr("shakemap_version"), "magnitude": _float(attr("magnitude")),
            "time": attr("event_timestamp"), "place": attr("event_description"),
            "lat": _float(attr("lat")), "lon": _float(attr("lon")), "source": "shakecast"}


def _product_version(product: dict[str, Any]) -> int:
    return int(float(product.get("properties", {}).get("version", 1)))


class ComcatSource:
    """The USGS earthquake catalogue, with downloads cached in ShakeCast's layout."""

    name = "comcat"

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE, timeout: float = 60):
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout

    def _get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{COMCAT_QUERY}?{urlencode(params)}"
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # The catalogue answers an unknown event id with 404: that is "no such
            # event", not an outage, and saying otherwise sends a reader astray.
            if exc.code == 404 and "eventid" in params:
                raise UnknownEvent(f"USGS has no event with id {params['eventid']!r}") from exc
            raise ShakeMapError(f"the USGS catalogue refused the request: {exc}") from exc
        except OSError as exc:
            raise ShakeMapError(f"could not reach the USGS catalogue: {exc}") from exc

    def search(
        self,
        bbox: tuple[float, float, float, float],
        start: str = "1900-01-01",
        end: str | None = None,
        min_magnitude: float | None = None,
    ) -> list[dict[str, Any]]:
        """Events with a ShakeMap inside (lon_min, lat_min, lon_max, lat_max), newest first."""
        lon_min, lat_min, lon_max, lat_max = bbox
        params: dict[str, Any] = {
            "format": "geojson", "producttype": "shakemap", "orderby": "time",
            "minlatitude": lat_min, "maxlatitude": lat_max,
            "minlongitude": lon_min, "maxlongitude": lon_max, "starttime": start,
        }
        if end:
            params["endtime"] = end
        if min_magnitude is not None:
            params["minmagnitude"] = min_magnitude
        events = []
        for feature in self._get_json(params).get("features", []):
            props = feature["properties"]
            lon, lat, depth = feature["geometry"]["coordinates"]
            events.append({
                "event_id": feature["id"], "magnitude": props.get("mag"),
                "time": utc_from_ms(props["time"]).isoformat(timespec="seconds"),
                "place": props.get("place"), "lat": lat, "lon": lon, "depth_km": depth,
                "mmi": props.get("mmi"), "updated": props.get("updated"), "source": "comcat",
            })
        return events

    def directory(self, event_id: str, version: int | None = None) -> Path:
        """Folder holding one of the event's ShakeMaps, downloading it if not cached.

        With no ``version``, the catalogue is asked which version is current, so
        a revised ShakeMap replaces an older cached one. If the catalogue cannot
        be reached, the newest cached version is used instead. An older
        ``version`` is fetched from the catalogue's superseded products: USGS
        keeps every version, so none needs archiving here.
        """
        cached = _version_dirs(self.cache_dir / event_id)
        if version is not None:
            for number, path in cached:
                if number == int(version):
                    return path
        try:
            detail = self._get_json({"eventid": event_id, "format": "geojson"})
        except UnknownEvent:
            raise
        except ShakeMapError:
            if cached and version is None:
                return cached[-1][1]
            raise
        products = detail.get("properties", {}).get("products", {}).get("shakemap") or []
        if not products:
            raise ShakeMapError(f"USGS has no ShakeMap for event {event_id!r}")
        product = products[0]            # ComCat lists the preferred product first
        current = _product_version(product)
        if version is not None and int(version) != current:
            product = self._superseded(event_id, product, int(version))
        return self._save(event_id, product)

    def extent(self, event_id: str, updated: Any = None) -> dict[str, Any]:
        """The preferred ShakeMap's {lat_min, lat_max, lon_min, lon_max, version}, as the catalogue lists it.

        Kept on disk next to the grids, and asked again only when ``updated`` (the
        catalogue's last change to the event, from ``search``) differs, so a revised
        map is re-read and a list of a hundred events is not a hundred requests.
        """
        path = self.cache_dir / event_id / "extent.json"
        if updated is not None:
            try:
                kept = json.loads(path.read_text(encoding="utf-8"))
                if kept.get("updated") == updated:
                    return kept
            except (OSError, ValueError):
                pass
        detail = self._get_json({"eventid": event_id, "format": "geojson"})
        products = detail.get("properties", {}).get("products", {}).get("shakemap") or []
        if not products:
            raise ShakeMapError(f"USGS has no ShakeMap for event {event_id!r}")
        props = products[0].get("properties", {})
        try:
            out = {name: float(props[key]) for name, key in (
                ("lat_min", "minimum-latitude"), ("lat_max", "maximum-latitude"),
                ("lon_min", "minimum-longitude"), ("lon_max", "maximum-longitude"))}
        except (KeyError, TypeError, ValueError) as exc:
            raise ShakeMapError(f"the catalogue lists no extent for the ShakeMap of {event_id!r}") from exc
        out.update(version=_product_version(products[0]),
                   updated=updated if updated is not None else detail.get("properties", {}).get("updated"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out) + "\n", encoding="utf-8")
        except OSError:
            pass                          # a read-only cache only costs a request next time
        return out

    def current_version(self, event_id: str) -> int:
        """The version of the event's preferred ShakeMap, asked of the catalogue now."""
        detail = self._get_json({"eventid": event_id, "format": "geojson"})
        products = detail.get("properties", {}).get("products", {}).get("shakemap") or []
        if not products:
            raise ShakeMapError(f"USGS has no ShakeMap for event {event_id!r}")
        return _product_version(products[0])

    def _superseded(self, event_id: str, preferred: dict[str, Any], version: int) -> dict[str, Any]:
        """An earlier version of the preferred ShakeMap (same contributing network and code)."""
        detail = self._get_json({"eventid": event_id, "format": "geojson", "includesuperseded": "true"})
        products = [p for p in detail.get("properties", {}).get("products", {}).get("shakemap") or []
                    if p.get("source") == preferred.get("source") and p.get("code") == preferred.get("code")]
        for product in products:
            if _product_version(product) == version:
                return product
        known = sorted({_product_version(p) for p in products})
        raise ShakeMapError(f"USGS has versions {known} of the ShakeMap for {event_id!r}, not {version}")

    def _save(self, event_id: str, product: dict[str, Any]) -> Path:
        version = _product_version(product)
        target = self.cache_dir / event_id / f"{event_id}-{version}"
        if (target / "grid.xml").is_file():
            return target
        contents = product.get("contents", {})
        if "download/grid.xml" not in contents:
            raise ShakeMapError(f"version {version} of the ShakeMap for {event_id!r} has no grid.xml")
        target.mkdir(parents=True, exist_ok=True)
        # grid.xml last: its presence is what marks a cached version complete.
        for name in ("uncertainty.xml", "grid.xml"):
            entry = contents.get(f"download/{name}")
            if entry:
                self._download(entry["url"], target / name)
        (target / "info.json").write_text(json.dumps({
            "event_id": event_id, "version": version, "source": product.get("source"),
            "code": product.get("code"), "update_time": product.get("updateTime"),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=2) + "\n", encoding="utf-8")
        return target

    def _download(self, url: str, path: Path) -> None:
        partial = path.with_suffix(path.suffix + ".part")
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=self.timeout) as response:
                partial.write_bytes(response.read())
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise ShakeMapError(f"could not download {url}: {exc}") from exc
        partial.replace(path)            # never leave a truncated grid where a reader would find it


class FallbackSource:
    """ShakeCast first, then the USGS catalogue.

    A ShakeCast server covering the network has every recent event it kept, on
    local disk, without a network call; it has nothing older than its retention
    period (``keep_eq_for``) and nothing outside its group regions. The catalogue
    has everything, so it fills those gaps. With no ShakeCast directory at all
    this behaves exactly like the catalogue alone.

    ShakeCast can trail the catalogue by a version while it polls, so the
    version used is whichever ShakeCast has; results record which source it was.
    """

    name = "auto"

    def __init__(self, shakecast_root: Path | str | None = None, cache_dir: Path | str = DEFAULT_CACHE):
        self.shakecast = ShakeCastSource(shakecast_root)
        self.comcat = ComcatSource(cache_dir)
        self.last_provider: str | None = None

    def directory(self, event_id: str, version: int | None = None) -> Path:
        try:
            path = self.shakecast.directory(event_id, version)
            self.last_provider = self.shakecast.name
            return path
        except ShakeMapError:
            path = self.comcat.directory(event_id, version)
            self.last_provider = self.comcat.name
            return path

    def search(self, bbox, *, start="1900-01-01", end=None, min_magnitude=None) -> list[dict[str, Any]]:
        """Both lists, one entry per event, ShakeCast's where both have it.

        Either side may be missing -- no ShakeCast here, or no network -- and the
        other still answers; only when both fail is that an error.
        """
        try:
            local = _filter_box(self.shakecast.events(), bbox, min_magnitude)
        except ShakeMapError:
            local = []
        try:
            remote = self.comcat.search(bbox, start=start, end=end, min_magnitude=min_magnitude)
        except ShakeMapError:
            if not local:
                raise
            remote = []
        seen = {e["event_id"] for e in local}
        merged = local + [e for e in remote if e["event_id"] not in seen]
        return sorted(merged, key=lambda e: e.get("time") or "", reverse=True)


def _filter_box(events: Iterable[dict[str, Any]], bbox, min_magnitude) -> list[dict[str, Any]]:
    lon_min, lat_min, lon_max, lat_max = bbox
    return [
        e for e in events
        if e.get("lat") is not None and lat_min <= e["lat"] <= lat_max and lon_min <= e["lon"] <= lon_max
        and (min_magnitude is None or (e.get("magnitude") or 0) >= min_magnitude)
    ]


SOURCES = ("auto", "shakecast", "comcat")


def get_source(name: str | None = None, **kwargs: Any) -> ComcatSource | ShakeCastSource | FallbackSource:
    """``auto`` (default: ShakeCast, then ComCat), ``shakecast`` or ``comcat``.

    ``BN_SHAKEMAP_SOURCE`` overrides the default.
    """
    chosen = (name or os.environ.get("BN_SHAKEMAP_SOURCE") or "auto").strip().lower()
    if chosen == "auto":
        return FallbackSource(**kwargs)
    if chosen == "comcat":
        return ComcatSource(**kwargs)
    if chosen == "shakecast":
        return ShakeCastSource(**kwargs)
    raise ShakeMapError(f"unknown ShakeMap source {chosen!r}; choose from {', '.join(SOURCES)}")


@lru_cache(maxsize=8)
def _load_cached(directory: str, imt: str) -> ShakeMapGrid:
    return read_grid(directory, imt)


_RESOLVE_TTL = 300.0     # seconds; a ShakeMap revision is minutes apart at the fastest
_resolved: dict[tuple[str, str, int | None], tuple[float, Path, str]] = {}


def load(event_id: str, source: str | None = None, version: int | None = None, imt: str = "SA(1.0)") -> ShakeMapGrid:
    """The ShakeMap grid for one event, parsed once per process per folder.

    Which folder is current is remembered for a few minutes, so one query does
    not ask the catalogue twice, while a revised ShakeMap is still picked up.
    """
    if not event_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(event_id)):
        raise ShakeMapError(f"{event_id!r} is not a USGS event id")
    src = get_source(source)
    key = (src.name, str(event_id), None if version is None else int(version))
    now = time.monotonic()
    hit = _resolved.get(key)
    if hit is None or now - hit[0] > _RESOLVE_TTL:
        path = src.directory(str(event_id), version)
        hit = (now, path, getattr(src, "last_provider", None) or src.name)
        _resolved[key] = hit
    grid = _load_cached(str(hit[1]), imt)
    # One folder always comes from one source, so recording it on the shared grid is safe.
    grid.meta["source"] = hit[2]
    return grid


def bbox(rows: Iterable[dict[str, Any]], margin_deg: float = 1.0) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) around rows that carry lat/lon, for event search."""
    points = [(float(r["lat"]), float(r["lon"])) for r in rows if r.get("lat") is not None]
    if not points:
        raise ShakeMapError("no rows with lat/lon to search around")
    lats, lons = zip(*points)
    return (min(lons) - margin_deg, min(lats) - margin_deg, max(lons) + margin_deg, max(lats) + margin_deg)


def covered(extent: dict[str, Any], sites: Iterable[dict[str, Any]]) -> int:
    """How many of the sites (rows with lat/lon) lie on the map."""
    return sum(extent["lat_min"] <= float(r["lat"]) <= extent["lat_max"]
               and extent["lon_min"] <= float(r["lon"]) <= extent["lon_max"]
               for r in sites if r.get("lat") is not None)


def covering(events: list[dict[str, Any]], sites: Iterable[dict[str, Any]], *,
             cache_dir: Path | str = DEFAULT_CACHE, workers: int = 8) -> tuple[list[dict[str, Any]], int]:
    """(the events whose ShakeMap reaches at least one site, how many were left out).

    An epicentre near the network says nothing about how far its map reaches. A map
    that reaches none of the sites cannot be run; one that reaches some can, its
    edge bounding the shaking at the rest (``kernels.hazard.ShakeMap``). Each kept
    event says how many sites its map reaches (``covered``, of ``sites``). The
    catalogue's extent of each map decides, asked in parallel and cached
    (``ComcatSource.extent``). An event whose extent cannot be had (a ShakeCast-only
    event, or the catalogue unreachable) is kept with ``covered`` None: running it
    is the check of last resort, and says why if it fails.
    """
    from concurrent.futures import ThreadPoolExecutor

    sites = [r for r in sites if r.get("lat") is not None]
    comcat = ComcatSource(cache_dir)

    def count(event: dict[str, Any]) -> int | None:
        if event.get("source") != "comcat":
            return None
        try:
            return covered(comcat.extent(event["event_id"], event.get("updated")), sites)
        except ShakeMapError:
            return None

    if not events or not sites:
        return list(events), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        counts = list(pool.map(count, events))
    kept = [{**e, "covered": n, "sites": len(sites)} for e, n in zip(events, counts) if n != 0]
    return kept, len(events) - len(kept)


def search_events(rows: Iterable[dict[str, Any]], source: str | None = None, *,
                  start: str = "1900-01-01", end: str | None = None,
                  min_magnitude: float | None = None) -> list[dict[str, Any]]:
    """Events with a ShakeMap around these rows, newest first, from either source."""
    box = bbox(rows)
    src = get_source(source)
    if isinstance(src, (ComcatSource, FallbackSource)):
        return src.search(box, start=start, end=end, min_magnitude=min_magnitude)
    return sorted(_filter_box(src.events(), box, min_magnitude), key=lambda e: e.get("time") or "", reverse=True)

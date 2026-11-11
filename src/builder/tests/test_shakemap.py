"""ShakeMap grids are read, found and cached correctly, and feed an IM through the engine."""

from __future__ import annotations

import json
from math import exp, log

import numpy as np
import pytest

from src.builder import shakemap
from src.builder.kernels.hazard import great_circle_km
from src.inference import engine

NS = 'xmlns="http://earthquake.usgs.gov/eqcenter/shakemap"'


def ln_sa_field(lat, lon):
    """ln SA1 (g): strongest in the south-west, linear so interpolation is exact."""
    return log(0.4) - 0.3 * (lat - 37.0) - 0.2 * (lon - 115.0)


def write_grid(folder, event_id="test001", version=1, extent=(112.0, 35.0, 121.0, 43.0),
               step=0.5, sigma=0.6, units="%g", with_uncertainty=True, reverse_rows=False):
    lon_min, lat_min, lon_max, lat_max = extent
    lons = np.arange(lon_min, lon_max + step / 2, step)
    lats = np.arange(lat_max, lat_min - step / 2, -step)          # north to south, as ShakeMap writes
    if reverse_rows:
        lats = lats[::-1]
    scale = 100.0 if units in ("%g", "pctg") else 1.0
    head = (f'<?xml version="1.0" encoding="UTF-8"?><shakemap_grid {NS} event_id="{event_id}" '
            f'shakemap_id="{event_id}" shakemap_version="{version}" shakemap_originator="test" '
            f'map_status="RELEASED">\n<event event_id="{event_id}" magnitude="6.8" depth="20" lat="37.5" '
            f'lon="115.0" event_timestamp="1966-03-22T08:19:36" event_description="Test, China" '
            f'seismic_stations="0" intensity_observations="0" />\n'
            f'<grid_specification lon_min="{lon_min}" lat_min="{lat_min}" lon_max="{lon_max}" '
            f'lat_max="{lat_max}" nominal_lon_spacing="{step}" nominal_lat_spacing="{step}" '
            f'nlon="{len(lons)}" nlat="{len(lats)}"/>\n')
    rows = [(lat, lon) for lat in lats for lon in lons]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "grid.xml").write_text(
        head + '<grid_field index="1" name="LON" units="dd" />\n<grid_field index="2" name="LAT" units="dd" />\n'
        f'<grid_field index="3" name="PSA10" units="{units}" />\n<grid_data>\n'
        + "\n".join(f"{lon:.4f} {lat:.4f} {exp(ln_sa_field(lat, lon)) * scale:.8g}" for lat, lon in rows)
        + "\n</grid_data>\n</shakemap_grid>\n", encoding="utf-8")
    if with_uncertainty:
        (folder / "uncertainty.xml").write_text(
            head + '<grid_field index="1" name="LON" units="dd" />\n<grid_field index="2" name="LAT" units="dd" />\n'
            '<grid_field index="3" name="STDPSA10" units="ln(g)" />\n<grid_data>\n'
            + "\n".join(f"{lon:.4f} {lat:.4f} {sigma}" for lat, lon in rows)
            + "\n</grid_data>\n</shakemap_grid>\n", encoding="utf-8")
    return folder


@pytest.fixture(autouse=True)
def fresh_cache():
    """Grids and resolved folders are memoised per process; tests must not share them."""
    shakemap._load_cached.cache_clear()
    shakemap._resolved.clear()
    yield
    shakemap._load_cached.cache_clear()
    shakemap._resolved.clear()


@pytest.fixture
def shakecast_dir(tmp_path, monkeypatch):
    """A ShakeCast user directory, selected the way a deployment would select it."""
    root = tmp_path / ".shakecast"
    (root / "data").mkdir(parents=True)
    monkeypatch.setenv("SHAKECAST_USER_DIRECTORY", str(root))
    monkeypatch.setenv("BN_SHAKEMAP_SOURCE", "shakecast")
    return root / "data"


# ---- reading a grid --------------------------------------------------------


def test_grid_values_are_read_in_g_and_interpolated_on_ln_sa(tmp_path):
    grid = shakemap.read_grid(write_grid(tmp_path / "ev"))
    lat, lon = 38.37, 116.11                     # off-grid on purpose
    ln_sa, sigma, inside = grid.sample([lat], [lon])
    assert inside[0]
    assert ln_sa[0] == pytest.approx(ln_sa_field(lat, lon), abs=1e-6)
    assert sigma[0] == pytest.approx(0.6)
    assert grid.meta["magnitude"] == 6.8 and grid.version == "1"


def test_older_pctg_units_mean_the_same_as_percent_g(tmp_path):
    a = shakemap.read_grid(write_grid(tmp_path / "a", units="%g"))
    b = shakemap.read_grid(write_grid(tmp_path / "b", units="pctg"))
    assert np.allclose(a.ln_im, b.ln_im)


def test_a_site_outside_the_grid_is_flagged_not_extrapolated(tmp_path):
    grid = shakemap.read_grid(write_grid(tmp_path / "ev"))
    ln_sa, _, inside = grid.sample([44.0], [116.0])
    assert not inside[0] and np.isnan(ln_sa[0])


def test_another_im_is_read_from_its_own_field_or_refused(tmp_path):
    folder = write_grid(tmp_path / "ev")
    assert shakemap.read_grid(folder, "SA(1.0)").imt == "SA(1.0)"
    with pytest.raises(shakemap.ShakeMapError, match="has no PGA field"):
        shakemap.read_grid(folder, "PGA")                  # this test grid carries PSA10 only
    with pytest.raises(shakemap.ShakeMapError, match="no ShakeMap grid for 'MMI'"):
        shakemap.read_grid(folder, "MMI")


def test_rows_in_an_unexpected_order_are_refused(tmp_path):
    with pytest.raises(shakemap.ShakeMapError, match="ordered"):
        shakemap.read_grid(write_grid(tmp_path / "ev", reverse_rows=True))


def test_missing_uncertainty_gives_zero_sigma_and_says_so(tmp_path):
    grid = shakemap.read_grid(write_grid(tmp_path / "ev", with_uncertainty=False))
    _, sigma, _ = grid.sample([38.0], [116.0])
    assert sigma[0] == 0.0
    assert "understated" in grid.warnings[0]


# ---- where grids come from ---------------------------------------------------


def test_shakecast_source_uses_the_newest_version_unless_told(shakecast_dir):
    write_grid(shakecast_dir / "ev1" / "ev1-1", "ev1", 1)
    write_grid(shakecast_dir / "ev1" / "ev1-3", "ev1", 3)
    source = shakemap.ShakeCastSource()
    assert source.directory("ev1").name == "ev1-3"
    assert source.directory("ev1", 1).name == "ev1-1"
    with pytest.raises(shakemap.ShakeMapError, match="not 2"):
        source.directory("ev1", 2)
    with pytest.raises(shakemap.ShakeMapError, match="no ShakeMap"):
        source.directory("nope")


def test_shakecast_events_are_filtered_to_the_rows(shakecast_dir):
    write_grid(shakecast_dir / "near" / "near-1", "near", 1)
    far = write_grid(shakecast_dir / "far" / "far-1", "far", 1)
    text = (far / "grid.xml").read_text(encoding="utf-8").replace('lat="37.5"', 'lat="10.0"')
    (far / "grid.xml").write_text(text, encoding="utf-8")
    rows = [{"id": "a", "lat": 37.0, "lon": 115.0}, {"id": "b", "lat": 38.0, "lon": 116.0}]
    assert [e["event_id"] for e in shakemap.search_events(rows)] == ["near"]


def test_comcat_source_downloads_into_the_shakecast_layout(tmp_path, monkeypatch):
    upstream = write_grid(tmp_path / "upstream", "ev9", 4)
    source = shakemap.ComcatSource(cache_dir=tmp_path / "cache")
    detail = {"properties": {"products": {"shakemap": [{
        "source": "us", "code": "ev9", "updateTime": 0, "properties": {"version": "4"},
        "contents": {f"download/{n}": {"url": (upstream / n).as_uri()} for n in ("grid.xml", "uncertainty.xml")},
    }]}}}
    monkeypatch.setattr(source, "_get_json", lambda params: detail)

    folder = source.directory("ev9")
    assert folder == tmp_path / "cache" / "ev9" / "ev9-4"
    assert (folder / "grid.xml").is_file() and (folder / "uncertainty.xml").is_file()
    assert json.loads((folder / "info.json").read_text())["version"] == 4
    # The cache is ShakeCast's layout, so the ShakeCast reader finds it as-is.
    assert shakemap.ShakeCastSource(tmp_path / "cache").directory("ev9") == folder


def test_an_earlier_version_is_fetched_from_the_catalogues_superseded_products(tmp_path, monkeypatch):
    def product(version, code="ev9", source="us"):
        upstream = write_grid(tmp_path / "upstream" / f"{source}{version}", "ev9", version)
        return {"source": source, "code": code, "updateTime": version, "properties": {"version": str(version)},
                "contents": {f"download/{n}": {"url": (upstream / n).as_uri()} for n in ("grid.xml", "uncertainty.xml")}}

    current = product(3)
    history = [current, product(2), product(1), product(2, code="other", source="at")]
    calls = []

    def catalogue(params):
        calls.append(params)
        return {"properties": {"products": {"shakemap": history if params.get("includesuperseded") else [current]}}}

    source = shakemap.ComcatSource(cache_dir=tmp_path / "cache")
    monkeypatch.setattr(source, "_get_json", catalogue)
    folder = source.directory("ev9", version=2)
    assert folder == tmp_path / "cache" / "ev9" / "ev9-2"
    assert json.loads((folder / "info.json").read_text())["source"] == "us"   # the preferred network's v2
    assert [c.get("includesuperseded") for c in calls] == [None, "true"]
    assert source.directory("ev9", version=2) == folder and len(calls) == 2   # cached now
    with pytest.raises(shakemap.ShakeMapError, match=r"versions \[1, 2, 3\].*not 5"):
        source.directory("ev9", version=5)


def test_comcat_falls_back_to_the_cache_when_offline(tmp_path, monkeypatch):
    write_grid(tmp_path / "cache" / "ev9" / "ev9-2", "ev9", 2)
    source = shakemap.ComcatSource(cache_dir=tmp_path / "cache")

    def offline(params):
        raise shakemap.ShakeMapError("could not reach the USGS catalogue")

    monkeypatch.setattr(source, "_get_json", offline)
    assert source.directory("ev9").name == "ev9-2"
    with pytest.raises(shakemap.ShakeMapError, match="could not reach"):
        source.directory("other")


# ---- ShakeCast first, then the catalogue -----------------------------------


@pytest.fixture
def fallback(tmp_path, monkeypatch):
    """An ``auto`` source over a ShakeCast directory and a catalogue whose only
    event is ``remote1``. Returns (source, shakecast data dir, upstream calls)."""
    data = tmp_path / "sc" / "data"
    data.mkdir(parents=True)
    upstream = write_grid(tmp_path / "upstream", "remote1", 7)
    calls = []

    def catalogue(params):
        calls.append(params)
        if params.get("eventid") not in (None, "remote1"):
            raise shakemap.UnknownEvent(f"USGS has no event with id {params['eventid']!r}")
        if "eventid" not in params:            # a search
            return {"features": [
                {"id": "remote1", "properties": {"mag": 6.0, "time": 0, "place": "catalogue"},
                 "geometry": {"coordinates": [115.0, 37.5, 10.0]}},
                {"id": "both", "properties": {"mag": 5.0, "time": 1000, "place": "catalogue copy"},
                 "geometry": {"coordinates": [115.0, 37.5, 10.0]}}]}
        return {"properties": {"products": {"shakemap": [{
            "source": "us", "code": "remote1", "updateTime": 0, "properties": {"version": "7"},
            "contents": {f"download/{n}": {"url": (upstream / n).as_uri()} for n in ("grid.xml", "uncertainty.xml")},
        }]}}}

    source = shakemap.FallbackSource(shakecast_root=tmp_path / "sc", cache_dir=tmp_path / "cache")
    monkeypatch.setattr(source.comcat, "_get_json", catalogue)
    return source, data, calls


def test_auto_is_the_default_source(monkeypatch):
    monkeypatch.delenv("BN_SHAKEMAP_SOURCE", raising=False)
    assert shakemap.get_source().name == "auto"


def test_shakecast_is_used_when_it_has_the_event(fallback):
    source, data, calls = fallback
    write_grid(data / "both" / "both-2", "both", 2)
    assert source.directory("both") == data / "both" / "both-2"
    assert source.last_provider == "shakecast" and calls == []      # no network call at all


def test_the_catalogue_fills_in_what_shakecast_lacks(fallback, tmp_path):
    source, _, _ = fallback
    assert source.directory("remote1") == tmp_path / "cache" / "remote1" / "remote1-7"
    assert source.last_provider == "comcat"


def test_a_version_shakecast_lacks_is_looked_for_upstream(fallback):
    source, data, _ = fallback
    write_grid(data / "remote1" / "remote1-6", "remote1", 6)
    assert source.directory("remote1").name == "remote1-6"           # ShakeCast's newest
    assert source.directory("remote1", 7).name == "remote1-7"        # asked for, so fetched
    assert source.last_provider == "comcat"


def test_an_event_neither_has_is_reported_by_the_catalogue(fallback):
    source, _, _ = fallback
    with pytest.raises(shakemap.UnknownEvent, match="no event with id 'nope'"):
        source.directory("nope")


def test_search_merges_both_lists_once_each(fallback):
    source, data, _ = fallback
    write_grid(data / "both" / "both-1", "both", 1)
    events = source.search((112.0, 35.0, 121.0, 43.0))
    assert sorted(e["event_id"] for e in events) == ["both", "remote1"]
    assert next(e for e in events if e["event_id"] == "both")["source"] == "shakecast"


def test_an_event_before_1970_is_read_on_every_platform(tmp_path, monkeypatch):
    # The catalogue's time is milliseconds since 1970: negative before it, which datetime.fromtimestamp
    # refuses on Windows. The 1966 Xingtai earthquake, the Hebei example's event, is one.
    xingtai = {"id": "iscgem849480", "properties": {"mag": 6.8, "time": -119052562000, "place": "Hebei"},
               "geometry": {"coordinates": [114.9, 37.5, 10.0]}}
    source = shakemap.ComcatSource(cache_dir=tmp_path)
    monkeypatch.setattr(source, "_get_json", lambda params: {"features": [xingtai]})
    (event,) = source.search((112.0, 35.0, 121.0, 43.0))
    assert event["time"] == "1966-03-25T01:50:38+00:00"
    assert shakemap.utc_from_ms(-2208988800000).isoformat() == "1900-01-01T00:00:00+00:00"


def test_an_unknown_event_is_reported_as_such_not_as_an_outage(tmp_path, monkeypatch):
    from urllib.error import HTTPError

    def not_found(request, timeout=None):
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(shakemap, "urlopen", not_found)
    with pytest.raises(shakemap.UnknownEvent, match="no event with id 'ev9x'"):
        shakemap.ComcatSource(cache_dir=tmp_path / "cache").directory("ev9x")


def test_an_event_id_that_is_not_one_is_refused_before_any_lookup():
    with pytest.raises(shakemap.ShakeMapError, match="not a USGS event id"):
        shakemap.load("../../etc/passwd")


# ---- through the engine --------------------------------------------------------


def test_a_shakemap_supplies_sa1_to_the_hebei_bn(shakecast_dir):
    write_grid(shakecast_dir / "ev1" / "ev1-1", "ev1", 1, sigma=0.0)
    network = engine.compile_query({"assembly": "hebei-transport/shakemap"}).network
    site = network.rows("Bridge")[0]
    result = engine.answer({"assembly": "hebei-transport/shakemap", "targets": [f"SA(1.0)::{site['id']}"],
                            "parameters": {"hazard.shakemap": {"event_id": "ev1"}}, "samples": 100})
    summary = result["targets"][0]["summary"]
    assert summary["p50"] == pytest.approx(exp(ln_sa_field(site["lat"], site["lon"])), rel=1e-3)
    assert result["computed"] == ["hazard.shakemap"]                     # only what the target needs
    assert any("ShakeMap ev1 version 1" in w for w in result["warnings"])
    assert any("no seismic station" in w for w in result["warnings"])


def test_nearby_sites_share_their_shaking_unless_told_otherwise(shakecast_dir):
    write_grid(shakecast_dir / "ev3" / "ev3-1", "ev3", 1, sigma=0.6)
    sites = engine.compile_query({"assembly": "hebei-transport/shakemap"}).network.rows("Bridge")
    a, b = min(((x, y) for i, x in enumerate(sites) for y in sites[i + 1:]),
               key=lambda pair: (pair[0]["lat"] - pair[1]["lat"]) ** 2 + (pair[0]["lon"] - pair[1]["lon"]) ** 2)

    def correlation(set_id):
        compiled = engine.compile_query({"assembly": "hebei-transport/shakemap",
                                         "parameters": {"hazard.shakemap": {"event_id": "ev3"}},
                                         "parameter_sets": {"spatial-correlation": set_id}})
        x = engine.sample_nodes(compiled, 20000, seed=5)
        return np.corrcoef(np.log(x[f"SA(1.0)::{a['id']}"]), np.log(x[f"SA(1.0)::{b['id']}"]))[0, 1]

    distance = float(great_circle_km(a["lat"], a["lon"], b["lat"], b["lon"]))
    assert correlation("jayaram-baker-2009.sa1p0") == pytest.approx(np.exp(-3 * distance / 25.7), abs=0.03)
    assert abs(correlation("independent.any-im")) < 0.03


def test_a_network_outside_the_shakemap_is_refused_not_zeroed(shakecast_dir):
    write_grid(shakecast_dir / "ev2" / "ev2-1", "ev2", 1, extent=(100.0, 20.0, 105.0, 25.0))
    with pytest.raises(engine.QueryError, match="ShakeMap ev2 v1 reaches none of the 33 sites"):
        engine.answer({"assembly": "hebei-transport/shakemap", "targets": ["SA(1.0)::*"],
                       "parameters": {"hazard.shakemap": {"event_id": "ev2"}}, "samples": 100})


def _detail(lat=(36.0, 38.0), lon=(-123.0, -121.0), version=3):
    props = {"minimum-latitude": str(lat[0]), "maximum-latitude": str(lat[1]),
             "minimum-longitude": str(lon[0]), "maximum-longitude": str(lon[1]), "version": str(version)}
    return {"properties": {"updated": 1, "products": {"shakemap": [{"properties": props}]}}}


def test_a_maps_extent_is_kept_until_the_catalogue_changes_the_event(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(shakemap.ComcatSource, "_get_json", lambda self, params: asked.append(params) or _detail())
    source = shakemap.ComcatSource(tmp_path)
    extent = source.extent("ev1", updated=10)
    assert extent == {"lat_min": 36.0, "lat_max": 38.0, "lon_min": -123.0, "lon_max": -121.0, "version": 3, "updated": 10}
    assert source.extent("ev1", updated=10) == extent and len(asked) == 1     # from disk
    source.extent("ev1", updated=11)
    assert len(asked) == 2                                                    # revised: asked again


def test_maps_that_reach_a_site_are_offered_with_how_many_they_reach(tmp_path, monkeypatch):
    # ev_small stops at 37 N, short of the northern site; ev_far is elsewhere; ev_cast has no catalogue
    # extent; ev_gone cannot be asked.
    def get_json(self, params):
        if params["eventid"] == "ev_gone":
            raise shakemap.ShakeMapError("could not reach the USGS catalogue: offline")
        return {"ev_small": _detail(lat=(36.0, 37.0)), "ev_far": _detail(lat=(10.0, 12.0))}.get(params["eventid"], _detail())

    monkeypatch.setattr(shakemap.ComcatSource, "_get_json", get_json)
    sites = [{"lat": 36.5, "lon": -122.0}, {"lat": 37.5, "lon": -122.0}]
    events = [{"event_id": i, "source": s, "updated": 1} for i, s in (("ev_big", "comcat"), ("ev_small", "comcat"),
              ("ev_far", "comcat"), ("ev_cast", "shakecast"), ("ev_gone", "comcat"))]
    kept, outside = shakemap.covering(events, sites, cache_dir=tmp_path)
    assert {e["event_id"]: e["covered"] for e in kept} == {"ev_big": 2, "ev_small": 1, "ev_cast": None, "ev_gone": None}
    assert outside == 1 and all(e["sites"] == 2 for e in kept)

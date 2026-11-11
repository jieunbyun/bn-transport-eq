"""Ground-motion kernels: everything that produces an IM.

Every hazard function gives ln IM at its sites as one Gaussian field,

    ln IM ~ N(mu, Sigma),    Sigma = D R D,    D = diag(sigma),

drawn jointly by one object owning every site (``GaussianField``), so the dependence between
sites lives inside that object and is never an implicit product of independent objects. The
function supplies mu and sigma (``Gmm`` from a ground-motion model, ``ShakeMap`` from a USGS
map); the binding's correlation supplies R (``correlation_matrix``), the identity without one.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.distributions import LogNormal

from . import Probability

_EARTH_RADIUS_KM = 6371.0


def _t(x: Any) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def _sites(n: int) -> str:
    return f"{n} site" if n == 1 else f"{n} sites"


def great_circle_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> torch.Tensor:
    lat1, lon1, lat2, lon2 = (torch.deg2rad(_t(v)) for v in (lat1, lon1, lat2, lon2))
    a = torch.sin((lat2 - lat1) / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))


def campbell_2003_ln_median(mw: Any, r_km: Any, c: dict[str, Any]) -> torch.Tensor:
    """ln median SA in g, term for term as the EMA dataset script implements it, for the coefficients given.

    Coefficients may be numbers or tensors (one per row), broadcasting with ``mw`` and ``r_km``.
    """
    mw, r = _t(mw), _t(r_km).clamp_min(1e-9)
    c = {k: _t(v) for k, v in c.items()}
    f1 = c["c2"] * mw + c["c3"] * (8.5 - mw) ** 2
    radius = torch.sqrt(r**2 + (c["c7"] * torch.exp(c["c8"] * mw)) ** 2)
    f2 = c["c4"] * torch.log(radius) + (c["c5"] + c["c6"] * mw) * r
    near = c["c7"] * (torch.log(r) - torch.log(_t(70.0)))
    f3 = torch.where(r <= 70.0, 0.0,
                     torch.where(r <= 130.0, near, near + c["c8"] * (torch.log(r) - torch.log(_t(130.0)))))
    return c["c1"] + f1 + f2 + f3


def campbell_2003(mw: torch.Tensor, r_km: torch.Tensor, params: list[dict[str, Any]]):
    """Campbell (2003) form for each row's set: ln median in g, and aleatory sigma (magnitude-dependent)."""
    c = {k: _t([p["coefficients"][k] for p in params]) for k in params[0]["coefficients"]}
    s = {k: _t([p["sigma"][k] for p in params]) for k in params[0]["sigma"]}
    return (campbell_2003_ln_median(mw, r_km, c),
            torch.where(mw < s["m_break"], s["a"] + s["b"] * mw, s["sigma_above"]))


# Functional forms, by the parameter set's "form". A new GMM with an existing form is only
# a parameter set; a new form is a function here plus its branch in schema/functions/hazard.gmm.json.
GMM_FORMS = {
    "campbell-2003": campbell_2003,
}


def fixed_event(table: list[dict[str, Any]]) -> dict[str, Any]:
    """The one Event row, with the fields a GMM needs."""
    if len(table) != 1:
        raise ValueError(f"a GMM needs exactly one Event row, got {len(table)}")
    event = table[0]
    missing = [f for f in ("mw", "lat", "lon", "depth_km") if event.get(f) is None]
    if missing:
        raise ValueError(f"Event {event['id']!r} lacks {', '.join(missing)}, which the GMM needs")
    return event


# ======================================================================
def jayaram_baker_2009_range_km(period_s: float, vs30_clustered: bool) -> float:
    """Range b of rho(h) = exp(-3h / b): case 1 (eq. 17) or, with Vs30 clustering, case 2 (eq. 18) below 1 s."""
    if period_s < 1.0:
        return 40.7 - 15.0 * period_s if vs30_clustered else 8.5 + 17.2 * period_s
    return 22.0 + 3.7 * period_s


def correlation_range_km(correlation: dict[str, Any] | None) -> float | None:
    """The range b of a spatial-correlation set (schema/models/spatial-correlation.json); None if independent."""
    if correlation is None or correlation["model"] == "independent":
        return None
    return jayaram_baker_2009_range_km(correlation["period_s"], correlation["vs30_clustered"])


def correlation_matrix(lat: torch.Tensor, lon: torch.Tensor, range_km: float | None) -> torch.Tensor:
    """R_ij = exp(-3 h_ij / range_km), h_ij the distance between sites i and j; the identity without a range."""
    if range_km is None:
        return torch.eye(len(lat), dtype=torch.float64)
    distance = great_circle_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return torch.exp(-3.0 * distance / range_km)


# ======================================================================
class GaussianField(Probability):
    """ln IM at n sites as one Gaussian field, drawn jointly:

        ln IM ~ N(mu, Sigma),    Sigma = D R D,    D = diag(sigma)
        R_ij = exp(-3 h_ij / b)  (Jayaram & Baker 2009), or R = I without a correlation

    h_ij is the great-circle distance between sites i and j. A subclass sets mu and sigma
    (``_field``); a root, with no parents. Sigma's root comes from its eigendecomposition rather
    than a Cholesky factor: co-located sites (R_ij = 1) and sites with sigma = 0 make Sigma
    singular, and the field then lives on the span of the directions with positive variance,
    where log_prob is its density.
    """

    _TOL = 1e-10
    function_id = "the hazard"

    def _field(self, sites, ln_median, sigma, correlation):
        """Set the field: sites [{"lat", "lon"}], one per child; ln_median and sigma (n,); correlation a
        spatial-correlation set's values (with its "id"), or None for independent sites."""
        self.ln_median = self.tensor(ln_median)                              # mu, (n,)
        self.sigma = self.tensor(sigma)                                      # (n,)
        self.correlation_id = (correlation or {}).get("id")
        self.range_km = correlation_range_km(correlation)
        lat, lon = _t([s["lat"] for s in sites]), _t([s["lon"] for s in sites])
        self.corr = correlation_matrix(lat, lon, self.range_km).to(self.device)            # R, (n, n)
        values, vectors = torch.linalg.eigh(self.sigma[:, None] * self.corr * self.sigma[None, :])
        keep = values > self._TOL                                            # directions with positive variance
        self.root = vectors[:, keep] * values[keep].sqrt()                  # ln IM = mu + root z, z ~ N(0, I)
        self.whiten = vectors[:, keep] / values[keep].sqrt()                # z = (ln IM - mu) whiten
        self.log_det = values[keep].log().sum()
        if self.range_km is None and len(self.childs) > 1:
            self.notes.add(f"{self.function_id} draws each Bridge's ln IM independently: no spatial correlation, "
                           "so joint outcomes across the network are likely understated")

    # ------------------------------------------------------------------
    def _sigma_text(self) -> str:
        if torch.allclose(self.sigma, self.sigma[0]):
            return f"{self.sigma[0].item():.3g}"
        return f"{self.sigma.min().item():.3g} to {self.sigma.max().item():.3g}"

    def _correlation_text(self) -> str:
        if self.range_km is None:
            return "independent sites"
        named = f" [{self.correlation_id}]" if self.correlation_id else ""
        return f"R_ij = exp(-3 h_ij / {self.range_km:.3g} km){named}"

    def form(self):
        return (f"ln IM ~ N(mu, D R D) at {_sites(len(self.childs))}, sigma {self._sigma_text()}, "
                f"{self._correlation_text()}{self.parameter_sets()}")

    # ------------------------------------------------------------------
    def distribution(self, n_sample: int) -> LogNormal:
        """Each site's IM on its own, for n_sample samples: LogNormal of shape (n_sample, n) (the EvaluateDensity
        capability). The marginals only: the sites' dependence is in sample and log_prob."""
        shape = (n_sample, len(self.childs))
        return LogNormal(self.ln_median.expand(shape), self.sigma.expand(shape))

    def sample(self, Cs_pars=None, n_sample=None):
        """
        n_sample : number of samples (a root: no Cs_pars)

        Returns
        -------
        Cs   : (n_sample, n) IM at each site
        logp : (n_sample,) log density of each sample
        """
        if n_sample is None:
            n_sample = len(Cs_pars)
        z = torch.randn(n_sample, self.root.shape[1], dtype=torch.float64, device=self.device)
        Cs = torch.exp(self.ln_median + z @ self.root.T)
        return Cs, self.log_prob(Cs)

    def log_prob(self, Cs):
        """
        Cs : (N, n) IM at each site (no parent columns: a root)

        Returns (N,) the joint log density of the IMs on the field's support; -inf where an IM is not
        positive and finite
        """
        x = Cs.to(device=self.device, dtype=torch.float64)[:, :len(self.childs)]
        valid = (x > 0).all(1) & torch.isfinite(x).all(1)
        ln_x = x.clamp_min(torch.finfo(x.dtype).tiny).log()
        z = (ln_x - self.ln_median) @ self.whiten
        logp = (-0.5 * z**2).sum(1) - 0.5 * z.shape[1] * math.log(2 * math.pi) - 0.5 * self.log_det - ln_x.sum(1)
        return torch.where(valid, logp, -torch.inf)


# ======================================================================
class Gmm(GaussianField):
    """Ground motion at n sites from one fixed earthquake, by a ground-motion model:

        ln IM ~ N(mu, D R D),    mu_i = ln median at site i,    D = diag(sigma)

    mu and sigma from the GMM at each site's hypocentral distance; R from the binding's correlation,
    the identity without one. A root: the earthquake is fixed data, not a parent.
    """

    function_id = "hazard.gmm"

    def __init__(self, childs, parents, event, gmm, sites, correlation=None, device="cpu"):
        """
        childs      : [Variable IM_1, ..., Variable IM_n], one per site, in g
        parents     : [] -- this node does not have parents: the EQ scenario is a fixed parameter
        event       : the earthquake, {"id", "mw", "lat", "lon", "depth_km"}
        gmm         : the ground-motion model, one parameter set for every site: its "form" (a key of
                      GMM_FORMS), "coefficients", aleatory "sigma", "sigma_epistemic",
                      "magnitude_range" and "distance_metric"
        sites       : [{"id", "lat", "lon"}], one per child, in the same order
        correlation : a spatial-correlation set's values (and its "id"), or None for independent sites
        """
        super().__init__(childs, parents, device)
        assert not parents, "Gmm is a root: the EQ scenario is a fixed parameter, not a parent"
        assert len(sites) == len(childs), "one site per child"
        if gmm["form"] not in GMM_FORMS:
            raise ValueError(f"GMM form {gmm['form']!r} is not implemented; have {sorted(GMM_FORMS)}")
        if gmm.get("distance_metric", "hypocentral") != "hypocentral":
            raise ValueError("only the hypocentral distance metric is implemented")
        low, high = gmm["magnitude_range"]
        if not low <= event["mw"] <= high:
            raise ValueError(f"Mw {event['mw']} outside the ground-motion set's magnitude range [{low}, {high}]")
        self.event = dict(event)
        self.gmm_form = gmm["form"]

        # Hypocentral distance from the event to each site, in km.
        lat, lon = _t([s["lat"] for s in sites]), _t([s["lon"] for s in sites])
        r = torch.sqrt(great_circle_km(event["lat"], event["lon"], lat, lon) ** 2 + _t(event["depth_km"]) ** 2)

        # GMM parameters: the ln median and total sigma at each site.
        ln_median, sigma_aleatory = GMM_FORMS[gmm["form"]](_t(event["mw"]), r, [gmm])
        sigma = torch.sqrt(sigma_aleatory**2 + _t(gmm["sigma_epistemic"]) ** 2)
        self._field(sites, ln_median.expand(len(childs)), sigma.expand(len(childs)), correlation)

    def form_details(self):
        e = self.event
        return [f"mu by {self.gmm_form} for event {e['id']} (Mw {e['mw']:g}, {e['depth_km']:g} km deep, "
                f"at ({e['lat']:g}, {e['lon']:g}); hypocentral distance); sigma the GMM's total"] + super().form_details()


# ======================================================================
class ShakeMap(GaussianField):
    """Ground motion at n sites from a USGS ShakeMap, as a Gaussian field:

        ln IM ~ N(mu, D R D),    D = diag(sigma)

    mu is the map's ln median at each site (grid.xml, e.g. PSA10) and sigma its uncertainty
    (uncertainty.xml, e.g. STDPSA10); R from the binding's correlation, the identity without one.

    A site off the map is given the map's value at the nearest point of its edge, sigma included. USGS sizes
    a map to reach where the shaking it predicts has faded to light (about MMI IV), and shaking keeps falling
    with distance, so that is an upper bound on the site's shaking, not zero and not a guess at it. A map that
    reaches none of the sites is refused: it is some other region's.
    """

    function_id = "hazard.shakemap"

    def __init__(self, childs, parents, event_id, imt, sites, source=None, version=None, correlation=None,
                 device="cpu"):
        """
        childs      : [Variable IM_1, ..., Variable IM_n], one per site, in g
        parents     : [] (a root)
        event_id    : the ShakeMap's USGS event id
        imt         : the intensity measure, e.g. "SA(1.0)"
        sites       : [{"id", "lat", "lon"}], one per child, in the same order
        source      : where to fetch the map ("shakecast" or "comcat"); None for shakemap.load's default
        version     : the map version; None for the current one
        correlation : a spatial-correlation set's values (and its "id"), or None for independent sites
        """
        from .. import shakemap   # imported late: only this kernel fetches data

        super().__init__(childs, parents, device)
        assert not parents and len(sites) == len(childs), "a root, with one site per child"

        # shakemap.load caches the parsed grid, and re-asks which version is current
        # every few minutes, so a long-running server picks up a revised map.
        grid = shakemap.load(event_id, source, version, imt)
        lat, lon = [s["lat"] for s in sites], [s["lon"] for s in sites]
        ln_median, sigma, inside = grid.sample(lat, lon)
        if not inside.any():
            raise ValueError(f"ShakeMap {grid.event_id} v{grid.version} reaches none of the {len(sites)} sites; "
                             f"it maps another region's shaking")
        self.outside_km = torch.zeros(len(sites), dtype=torch.float64)
        if not inside.all():
            off = (~inside).nonzero()[0]
            edge_lat, edge_lon = grid.nearest_on_map([lat[i] for i in off], [lon[i] for i in off])
            ln_median[off], sigma[off], _ = grid.sample(edge_lat, edge_lon)
            self.outside_km[off] = great_circle_km(_t([lat[i] for i in off]), _t([lon[i] for i in off]),
                                                   _t(edge_lat), _t(edge_lon))
            far = int(self.outside_km.argmax())
            self.notes.add(f"{len(off)} of {len(sites)} sites lie off ShakeMap {grid.event_id} v{grid.version} (up to "
                           f"{self.outside_km[far]:.0f} km, {childs[far].name}): each is given the map's value at the "
                           f"nearest edge, an upper bound, since USGS maps reach to where shaking has faded to light and "
                           f"it falls further with distance.")
        self.source = f"ShakeMap {grid.event_id} v{grid.version}, {imt}"
        self._field(sites, _t(ln_median), _t(sigma), correlation)

        meta = grid.meta
        self.notes.add(f"Ground motion from USGS ShakeMap {grid.event_id} version {grid.version} "
                       f"({meta.get('originator')}, {meta.get('map_status')}, via {meta.get('source')}).")
        if not meta.get("seismic_stations"):
            self.notes.add("This ShakeMap has no seismic station recordings, so its shaking is a ground-motion-model "
                           "estimate (constrained by felt reports at most) with the larger uncertainty that implies.")
        for warning in grid.warnings:
            self.notes.add(warning)

    def form_details(self):
        return [f"mu and sigma from {self.source}"] + super().form_details()

"""The clique-tree sampler (src/inference/cliques.py) against Monte Carlo simulation, on an example network.

Four cases of evidence per example. MCS draws forward samples, keeps those that match the discrete
evidence and weights each by the likelihood of an observed SAObs, N(o; SA, (epsilon SA)^2)
(likelihood weighting: a continuous value is never matched exactly). Compared, for every
unobserved variable: E[ln SA], E[ln SAObs], P(D = each state), P(E closed) and P(S Fail), with
standard errors from 20 groups of chains (clique tree) and from the weighted samples (MCS); and
the posterior density of each bridge's ln SA, on BINS.

The ln SA residuals are correlated by rho(h) = exp(-3h / b) between the bridges' sites
(hazard.correlation_matrix), b = --range-km (the example's default below). Writes
examples/<example>/clique-sampling/results_range<b>km.json.

    PYTHONPATH=<rsr repository>;<this repository> python examples/clique-sampling/compare.py --example ema-highway
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.builder.bn_builder import build_tbn, compile_path
from src.builder.kernels.hazard import correlation_matrix
from src.inference import cliques, session

EXAMPLES = Path(__file__).resolve().parents[1]
SETTINGS = {   # D 2 = moderate, 3 = extensive; EdgeOpen 0 closed, 1 open; SystemState 0 Fail, 1 Survival
    "toy-ca": {
        "assembly": "demo.assembly.json", "range_km": 200.0, "n_mcs": 10**7, "chunk": 10**6,
        "cases": {
            "none": {},
            "1 damage, system": {"DamageState.HazusBridge::b4": 3, "SystemState::n4": 0},
            "2 system, edge": {"SystemState::n7": 0, "EdgeOpen::e7": 1},
            "3 SAObs, edge": {"SAObs(1.0)::b6": 0.8, "EdgeOpen::e9": 0},
        }},
    "ema-highway": {   # Jayaram & Baker (2009) range for SA(1.0 s), b = 40.7 - 15.0 T
        "assembly": "gmm.assembly.json", "range_km": 25.7, "n_mcs": 10**8, "chunk": 10**5,
        "cases": {
            "none": {},
            "1 SAObs, edge": {"SAObs(1.0)::bridge-e0125": 0.6, "EdgeOpen::e0124": 0},
            "2 damage, system": {"DamageState.HazusBridge::bridge-e0125": 2, "SystemState::n62": 0},
            "3 two systems": {"SystemState::n64": 0, "SystemState::n62": 1},
        }},
}
N_GROUP = 20
SAMPLER = {"n_chain": 100, "n_burn": 200, "n_keep": 10_000, "bridge": 8}
BINS = torch.linspace(-4.5, 1.0, 111, dtype=torch.float64)          # ln SA, for the posterior densities

parser = argparse.ArgumentParser()
parser.add_argument("--example", choices=list(SETTINGS), default="toy-ca")
parser.add_argument("--range-km", type=float, help="the example's default if not given")
parser.add_argument("--n-mcs", type=float, help="forward samples, the example's default if not given")
parser.add_argument("--n-keep", type=int, default=SAMPLER["n_keep"], help="kept sweeps per chain")
args = parser.parse_args()
S = SETTINGS[args.example]
RANGE_KM = args.range_km or S["range_km"]
N_MCS, CHUNK = int(args.n_mcs or S["n_mcs"]), min(S["chunk"], int(args.n_mcs or S["n_mcs"]))
SAMPLER["n_keep"] = args.n_keep


def ln_sa_counts(t, rows, w):
    """(n, len(BINS) - 1) weight of the rows in each ln SA bin, per edge's bridge."""
    out = torch.zeros(len(t["edges"]), len(BINS) - 1, dtype=torch.float64)
    for i, c in enumerate(t["edges"]):
        k = torch.bucketize(rows[c["SA"]].log(), BINS) - 1
        inside = (k >= 0) & (k < len(BINS) - 1)
        out[i].index_add_(0, k[inside], w[inside])
    return out


def features(t, rows):
    """(N, Q) columns and their names: ln SA, ln SAObs, D one-hot, E closed per edge; S Fail per system."""
    cols, names = [], []
    for c in t["edges"]:
        cols.append(rows[c["SA"]].log()[:, None]); names.append(f"E[ln {c['SA']}]")
        cols.append(rows[c["Obs"]].log()[:, None]); names.append(f"E[ln {c['Obs']}]")
        K = len(c["to_e"])
        cols.append(torch.nn.functional.one_hot(rows[c["D"]], K).double())
        names += [f"P({c['D']} = {s})" for s in c["frag"].states]
        cols.append((rows[c["E"]] == 0).double()[:, None]); names.append(f"P({c['E']} closed)")
    for k in t["systems"]:
        cols.append((rows[f"SystemState::{k}"] == 0).double()[:, None]); names.append(f"P(SystemState::{k} Fail)")
    return torch.cat(cols, dim=1), names


def mcs(t, seed):
    """Forward samples in chunks of CHUNK, N_MCS in all."""
    g = torch.Generator().manual_seed(seed)
    fld = cliques.field(t)
    for _ in range(N_MCS // CHUNK):
        rows = {}
        sa_all = cliques.field_draw(fld, CHUNK, g)
        for i, c in enumerate(t["edges"]):
            sa = sa_all[:, i]
            d = torch.multinomial(c["frag"].distribution(sa), 1, generator=g)[:, 0]
            rows[c["SA"]], rows[c["D"]], rows[c["E"]] = sa, d, c["to_e"][d].argmax(dim=1)
            rows[c["Obs"]] = c["obs"].distribution(sa).icdf(torch.rand(CHUNK, dtype=torch.float64, generator=g))
        for k, s in session.systems(t, rows).items():                  # each distinct edge vector once
            rows[f"SystemState::{k}"] = s
        yield rows


def mcs_estimates(t, cases, seed):
    """Per case, from the same samples: the weighted mean, its standard error, the number of samples
    matching the discrete evidence, the effective sample size and the ln SA density per edge's bridge.
    A sample's weight is 0 unless it matches the discrete evidence, times the likelihood of each
    observed SAObs."""
    obs = {c["Obs"]: c for c in t["edges"]}
    acc = dict.fromkeys(cases)                                          # sums: w, w f, w^2, w^2 f, w^2 f^2
    kept, counts = dict.fromkeys(cases, 0), dict.fromkeys(cases, 0.0)
    t0, n_chunks = time.perf_counter(), N_MCS // CHUNK
    for j, rows in enumerate(mcs(t, seed)):
        F, names = features(t, rows)
        for name, ev in cases.items():
            w, Fc = torch.ones(len(F), dtype=torch.float64), F.clone()
            for v, val in ev.items():
                if v in obs:                                            # continuous: weight, and show it observed
                    w = w * obs[v]["obs"].distribution(rows[obs[v]["SA"]]).log_prob(torch.tensor(float(val))).exp()
                    Fc[:, names.index(f"E[ln {v}]")] = torch.log(torch.tensor(float(val)))
                else:
                    w = w * (rows[v] == val)
            kept[name] += int((w > 0).sum())
            counts[name] = counts[name] + ln_sa_counts(t, rows, w)
            w2 = w * w
            s = torch.stack([w.sum().expand(F.shape[1]), w @ Fc, w2.sum().expand(F.shape[1]), w2 @ Fc, w2 @ Fc ** 2])
            acc[name] = s if acc[name] is None else acc[name] + s
        if (j + 1) % max(1, n_chunks // 10) == 0:
            print(f"  MCS {(j + 1) * CHUNK:.0e} of {N_MCS:.0e}, {time.perf_counter() - t0:.0f} s", flush=True)
    out = {}
    for name, (sw, swf, sw2, sw2f, sw2f2) in acc.items():
        mean = swf / sw
        se = (sw2f2 - 2 * mean * sw2f + mean ** 2 * sw2).clamp(min=0).sqrt() / sw
        density = counts[name] / (sw[0] * (BINS[1] - BINS[0]))
        out[name] = (mean, se, kept[name], (sw[0] ** 2 / sw2[0]).item(), density)
    return out


def clique_estimates(t, rows, n_chain):
    """The mean of every feature over the rows and its standard error from N_GROUP groups of chains,
    in chunks of CHUNK rows; and the ln SA density per edge's bridge."""
    N = len(rows[t["edges"][0]["SA"]])
    sums, n = None, torch.zeros(N_GROUP, dtype=torch.float64)
    for s in range(0, N, CHUNK):
        F, names = features(t, {k: v[s:s + CHUNK] for k, v in rows.items()})
        group = (torch.arange(s, s + len(F)) % n_chain) % N_GROUP
        sums = torch.zeros(N_GROUP, F.shape[1], dtype=torch.float64) if sums is None else sums
        sums.index_add_(0, group, F)
        n.index_add_(0, group, torch.ones(len(F), dtype=torch.float64))
    means = sums / n[:, None]
    density = ln_sa_counts(t, rows, torch.ones(N, dtype=torch.float64)) / (N * (BINS[1] - BINS[0]))
    return sums.sum(dim=0) / n.sum(), means.std(dim=0) / N_GROUP ** 0.5, names, N, density


compiled = compile_path(EXAMPLES / args.example / S["assembly"])
probs = build_tbn(compiled)[1]
bridges = {b["id"]: b for b in json.loads((EXAMPLES / args.example / "network.json").read_text())["tables"]["Bridge"]}
sites = [bridges[v.name.split("::")[1]] for v in probs["hazard.gmm"].childs]
corr = correlation_matrix(torch.tensor([s["lat"] for s in sites], dtype=torch.float64),
                          torch.tensor([s["lon"] for s in sites], dtype=torch.float64), RANGE_KM)
t = cliques.tree(probs, corr)
OUT = EXAMPLES / args.example / "clique-sampling" / f"results_range{RANGE_KM:g}km.json"
OUT.parent.mkdir(exist_ok=True)
print(f"{args.example}: {len(t['edges'])} edges, {len(t['systems'])} systems, range {RANGE_KM:g} km", flush=True)
t0 = time.perf_counter()
ref = mcs_estimates(t, S["cases"], 1)
sec_mcs = time.perf_counter() - t0
print(f"MCS: {N_MCS:.0e} forward samples, {sec_mcs:.0f} s\n", flush=True)

results = {"example": args.example, "mcs": {"n": N_MCS, "runs": 1, "sec": sec_mcs}, "range_km": RANGE_KM,
           "sampler": {"function": cliques.sample.__name__, **SAMPLER}, "cases": {}}
for name, ev in S["cases"].items():
    rows, cost = cliques.sample(t, ev, **SAMPLER)
    est, se, names, n_rows, density = clique_estimates(t, rows, SAMPLER["n_chain"])
    del rows
    m_est, m_se, n_kept, ess, m_density = ref[name]
    err, both = est - m_est, (se ** 2 + m_se ** 2).sqrt()
    observed = torch.tensor([any(f"{v} " in n or f"{v}]" in n for v in ev) for n in names])
    free = (both > 0) & ~observed                                # an observed variable is not estimated
    z = err[free] / both[free]
    is_sa = torch.tensor([n.startswith("E[ln") for n in names])
    summary = {
        "mcs_kept": n_kept, "mcs_ess": ess, "rows": n_rows,
        "max_abs_err_prob": err[free & ~is_sa].abs().max().item(),
        "max_abs_err_ln_sa": err[free & is_sa].abs().max().item(),
        "mean_z2": (z ** 2).mean().item(), "n_abs_z_over_3": int((z.abs() > 3).sum()), "n_compared": int(free.sum()),
    }
    results["cases"][name] = {
        "evidence": ev, "summary": summary, "cost": cost, "sec_total": sum(c["sec"] for c in cost.values()),
        "estimates": {n: {"clique": est[j].item(), "clique_se": se[j].item(), "mcs": m_est[j].item(),
                          "mcs_se": m_se[j].item()} for j, n in enumerate(names)},
        "ln_sa_density": {"bins": BINS.tolist(),
                          **{c["SA"]: {"clique": density[i].tolist(), "mcs": m_density[i].tolist()}
                             for i, c in enumerate(t["edges"])}},
    }
    print(f"{name}: {ev}")
    print(f"  MCS kept {n_kept} of {N_MCS:.0e} (effective {ess:.0f}); clique rows {n_rows}; "
          f"max |err| P {summary['max_abs_err_prob']:.4f}, E[ln SA] {summary['max_abs_err_ln_sa']:.4f}; "
          f"mean z^2 {summary['mean_z2']:.2f}, |z| > 3: {summary['n_abs_z_over_3']} of {summary['n_compared']}")
    print("  cost:  " + "  ".join(f"{k} {v['sec']:.3f} s / {v['n_ops']} ops" for k, v in cost.items()), flush=True)

OUT.write_text(json.dumps(results, indent=2) + "\n")
print(f"\nwrote {OUT}")

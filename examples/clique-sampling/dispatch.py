"""Peri-disaster inspection dispatch, as a posterior session (src/inference/session.py) takes in evidence.

A storyline per example: observations arrive one at a time; at each, the session reweights its
samples or samples again, and the dispatch table is read off: per edge, P(closed | evidence), the
population cut off now, E[L | evidence], and if the edge were opened, E[L | do(open), evidence].
At the steps marked for checking, the same quantities come from MCS with the cumulative evidence
(forward samples weighted by the likelihood of the evidence; do by substitution, as the session).
Writes examples/<example>/clique-sampling/dispatch.json.

    PYTHONPATH=<rsr repository>;<this repository> python examples/clique-sampling/dispatch.py --example toy-ca
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch

from src.builder.bn_builder import build_tbn, compile_path
from src.builder.kernels.hazard import correlation_matrix
from src.inference import cliques, session

EXAMPLES = Path(__file__).resolve().parents[1]
SETTINGS = {   # D 4 = complete (the edge closed); EdgeOpen 0 closed; SystemState 0 Fail
    "toy-ca": {
        "assembly": "demo.assembly.json", "range_km": 200.0, "n_mcs": 10**7, "chunk": 10**6,
        "steps": [   # (label, variable, value, check against MCS); T4 inspects the edge T3's table puts first
            ("T1 station reading", "SAObs(1.0)::b6", 0.8, True),
            ("T2 report: e9 closed", "EdgeOpen::e9", 0, True),
            ("T3 report: n4 cut off", "SystemState::n4", 0, True),
            ("T4 inspection: b4 complete", "DamageState.HazusBridge::b4", 4, True),
        ]},
}
SAMPLER = {"n_chain": 100, "n_burn": 200, "n_keep": 10_000, "bridge": 8}

parser = argparse.ArgumentParser()
parser.add_argument("--example", choices=list(SETTINGS), default="toy-ca")
parser.add_argument("--n-mcs", type=float, help="forward samples per check, the example's default if not given")
parser.add_argument("--n-keep", type=int, default=SAMPLER["n_keep"], help="kept sweeps per chain")
args = parser.parse_args()
S = SETTINGS[args.example]
N_MCS = int(args.n_mcs or S["n_mcs"])
CHUNK = min(S["chunk"], N_MCS)
SAMPLER["n_keep"] = args.n_keep


def forward(t, n, g):
    """n forward samples of every variable."""
    sa = cliques.field_draw(cliques.field(t), n, g)
    rows = {}
    for i, c in enumerate(t["edges"]):
        d = torch.multinomial(c["frag"].distribution(sa[:, i]), 1, generator=g)[:, 0]
        rows[c["SA"]], rows[c["D"]], rows[c["E"]] = sa[:, i], d, c["to_e"][d].argmax(dim=1)
        rows[c["Obs"]] = c["obs"].distribution(sa[:, i]).icdf(torch.rand(n, dtype=torch.float64, generator=g))
    for k, s in session.systems(t, rows).items():
        rows[f"SystemState::{k}"] = s
    return rows


def mcs(t, ev, population, seed=1):
    """{name: (estimate, standard error)} of session.features() given ev, by likelihood weighting, and the ESS."""
    g = torch.Generator().manual_seed(seed)
    obs = {c["Obs"]: c for c in t["edges"]}
    acc = None
    for _ in range(N_MCS // CHUNK):
        rows = forward(t, CHUNK, g)
        lw = torch.zeros(CHUNK, dtype=torch.float64)
        for v, x in ev.items():
            if v in obs:
                lw = lw + obs[v]["obs"].distribution(rows[obs[v]["SA"]]).log_prob(torch.tensor(float(x)))
                rows[v] = torch.full((CHUNK,), float(x), dtype=torch.float64)
            else:
                lw = torch.where(rows[v] == x, lw, -math.inf)
        keep = torch.isfinite(lw)
        rows, w = {k: v[keep] for k, v in rows.items()}, lw[keep].exp()
        F, names = session.features(t, rows, population)
        w2 = w * w
        s = torch.stack([w.sum().expand(F.shape[1]), w @ F, w2.sum().expand(F.shape[1]), w2 @ F, w2 @ F ** 2])
        acc = s if acc is None else acc + s
    sw, swf, sw2, sw2f, sw2f2 = acc
    mean = swf / sw
    se = (sw2f2 - 2 * mean * sw2f + mean ** 2 * sw2).clamp(min=0).sqrt() / sw
    return {n: (mean[j].item(), se[j].item()) for j, n in enumerate(names)}, (sw[0] ** 2 / sw2[0]).item()


def table(t, sess, est, ref=None):
    """The dispatch table as text, with the MCS estimate and z beside each number when ref is given."""
    def cell(name, x=None):
        e = est[name] if x is None else x
        if ref is None:
            return f"{e[0]:7.3f} ±{e[1]:.3f}"
        r = ref[name] if x is None else (ref["E[L]"][0] - ref[name][0], math.hypot(ref["E[L]"][1], ref[name][1]))
        z = (e[0] - r[0]) / math.hypot(e[1], r[1]) if math.hypot(e[1], r[1]) > 0 else 0.0
        return f"{e[0]:7.3f} | {r[0]:7.3f} z{z:+5.1f}"
    head = "clique ±SE" if ref is None else "clique | MCS     z"
    out = [f"    {'origin':10s} P(Fail): {head}"]
    out += [f"    {k:10s} {cell(f'P(SystemState::{k} Fail)')}" for k in t["systems"]]
    out.append(f"    E[L] {cell('E[L]')}")
    out.append(f"    {'edge':10s} P(closed)               restored by do(open)       E[L | do(open)]")
    for row in sess.dispatch(est):
        e, name = f"EdgeOpen::{row['edge']}", f"restored by do(EdgeOpen::{row['edge']} open)"
        mark = " (observed)" if row["observed"] else ""
        out.append(f"    {row['edge'] + mark:10s} {cell(f'P({e} closed)')}   {cell(name)}   {cell(None, row['L_do_open']) if ref is None else ''}")
    return "\n".join(out)


compiled = compile_path(EXAMPLES / args.example / S["assembly"])
varis, probs = build_tbn(compiled)
network = json.loads((EXAMPLES / args.example / "network.json").read_text())["tables"]
bridges = {b["id"]: b for b in network["Bridge"]}
sites = [bridges[v.name.split("::")[1]] for v in probs["hazard.gmm"].childs]
corr = correlation_matrix(torch.tensor([s["lat"] for s in sites], dtype=torch.float64),
                          torch.tensor([s["lon"] for s in sites], dtype=torch.float64), S["range_km"])
t = cliques.tree(probs, corr)
population = {r["id"]: r["population"] for r in network["Node"] if r.get("population") is not None}
unit = "people" if population else "towns"
print(f"{args.example}: L counts {unit} cut off from a key destination", flush=True)

t0 = time.perf_counter()
sess = session.Session(t, probs, varis, population=population, sampler=SAMPLER)
sec, est = time.perf_counter() - t0, sess.estimates()
print(f"\nT0 no evidence: sampled, {sec:.0f} s")
print(table(t, sess, est), flush=True)
out = {"example": args.example, "unit": unit, "sampler": SAMPLER, "n_mcs": N_MCS,
       "steps": [{"label": "T0 no evidence", "variable": None, "value": None, "update": "sampled", "ess": 1.0,
                  "sec": sec, "estimates": est}]}
for label, var, value, check in S["steps"]:
    t0 = time.perf_counter()
    entry = sess.observe(var, value)
    sec = time.perf_counter() - t0
    est = sess.estimates()
    print(f"\n{label}: {var} = {value}: {entry['update']} (ESS {entry['ess']:.1%} of the rows before), {sec:.0f} s")
    step = {"label": label, "variable": var, "value": value, "update": entry["update"], "ess": entry["ess"],
            "sec": sec, "estimates": est}
    if check:
        t0 = time.perf_counter()
        ref, ess = mcs(t, sess.evidence, population)
        z = [(est[n][0] - ref[n][0]) / math.hypot(est[n][1], ref[n][1]) for n in est
             if math.hypot(est[n][1], ref[n][1]) > 0]
        print(f"  checked against {N_MCS:.0e} MCS (effective {ess:.0f}), {time.perf_counter() - t0:.0f} s: "
              f"mean z^2 {sum(x * x for x in z) / len(z):.2f}, |z| > 3: {sum(abs(x) > 3 for x in z)} of {len(z)}")
        print(table(t, sess, est, ref))
        step.update(mcs=ref, mcs_ess=ess)
    else:
        print(table(t, sess, est))
    out["steps"].append(step)

path = EXAMPLES / args.example / "clique-sampling" / "dispatch.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print(f"\nwrote {path}")

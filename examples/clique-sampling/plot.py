"""Plot a compare.py results file: the clique-tree estimate against MCS for every unobserved variable,
per case, coloured by the step that drew the variable (its up messages shape every estimate).
Rows: probabilities (P(D = state), P(E closed), P(S Fail)); E[ln SA] and E[ln SAObs];
z = difference / combined SE. Writes comparison_range<b>km.png beside results_range<b>km.json.

    python examples/clique-sampling/plot.py examples/<example>/clique-sampling/results_range<b>km.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SOURCE = Path(sys.argv[1])
EXAMPLE = SOURCE.resolve().parents[1].name
results = json.loads(SOURCE.read_text())
STEPS = {   # the step that draws each kind of variable: colour (categorical slots 1-5) and marker
    "sa_e_down (SA | E)": ("E[ln SA(", "#2a78d6", "o"),
    "obs_down (SAObs | SA)": ("E[ln SAObs(", "#e87ba4", "v"),
    "edge_down (D | SA, E)": ("P(DamageState", "#eb6834", "s"),
    "root_absorb (E | SA)": ("P(EdgeOpen", "#1baf7a", "^"),   # the root
    "system_down (S | E)": ("P(SystemState", "#eda100", "D"),
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

cases = results["cases"]
fig, axes = plt.subplots(3, len(cases), figsize=(4.2 * len(cases), 11), squeeze=False)
for col, (name, case) in enumerate(cases.items()):
    est = case["estimates"]
    root = "certificate MCMC" if any(v.startswith("SystemState") for v in case["evidence"]) else "direct draw"
    axes[0, col].set_title(f"{name}\nroot (E): {root}", fontsize=10, color=INK)
    z_x = 0
    for label, (prefix, colour, marker) in STEPS.items():
        rows = [v for k, v in est.items() if k.startswith(prefix) and (v["clique_se"] ** 2 + v["mcs_se"] ** 2) > 0
                and not any(f"{x} " in k or f"{x}]" in k for x in case["evidence"])]   # observed: not estimated
        if not rows:
            continue
        ax = axes[1, col] if prefix.startswith("E[ln") else axes[0, col]
        ax.errorbar([r["mcs"] for r in rows], [r["clique"] for r in rows],
                    xerr=[2 * r["mcs_se"] for r in rows], yerr=[2 * r["clique_se"] for r in rows],
                    fmt=marker, ms=6, color=colour, mec="white", mew=0.8, elinewidth=1, label=label)
        z = [(r["clique"] - r["mcs"]) / (r["clique_se"] ** 2 + r["mcs_se"] ** 2) ** 0.5 for r in rows]
        axes[2, col].plot(range(z_x, z_x + len(z)), z, marker, ms=6, color=colour, mec="white", mew=0.8)
        z_x += len(z)
    for ax, lo_hi in ((axes[0, col], (0, 1)), (axes[1, col], None)):
        lo, hi = lo_hi or (min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1]))
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"MCS ({results['mcs']['n']:.0e} forward samples)", fontsize=9, color=MUTED)
    axes[0, col].set_ylabel("clique tree", fontsize=9, color=MUTED)
    axes[1, col].set_ylabel("clique tree: E[ln SA], E[ln SAObs]", fontsize=9, color=MUTED)
    axes[2, col].axhspan(-3, 3, color=GRID, zorder=0)
    axes[2, col].axhline(0, color=MUTED, lw=1, zorder=0)
    axes[2, col].set_xlabel("variable", fontsize=9, color=MUTED)
    axes[2, col].set_ylabel("z = (clique - MCS) / SE", fontsize=9, color=MUTED)
    s = case["summary"]
    axes[2, col].set_title(f"max |err| P {s['max_abs_err_prob']:.4f}, mean z² {s['mean_z2']:.2f}",
                           fontsize=9, color=INK)
    for ax in axes[:, col]:
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

handles = [plt.Line2D([], [], ls="", marker=m, ms=7, color=c, mec="white") for _, c, m in STEPS.values()]
fig.legend(handles, list(STEPS), loc="lower center", ncol=len(STEPS), frameon=False, fontsize=10,
           title="drawn by (error bars: 2 SE; grey band: |z| < 3)", title_fontsize=9)
bridge = results["sampler"]["bridge"]                     # older results: (partner tuples, candidates)
bridge = bridge[0] if isinstance(bridge, (tuple, list)) else bridge
fig.suptitle(f"Clique-tree sampler against MCS on {EXAMPLE} (ln SA correlated, range {results['range_km']:g} km): "
             f"{results['sampler']['n_chain']} chains x "
             f"{results['sampler']['n_keep']} kept sweeps, tuple jump m={bridge}", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
out = SOURCE.with_name(SOURCE.stem.replace("results", "comparison") + ".png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")

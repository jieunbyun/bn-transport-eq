"""A posterior session: samples of the whole BN given the evidence so far, updated as observations arrive.

The rows start as cliques.sample's draws from P(everything | evidence), each with log weight 0. A
new observation X = x is taken in by importance weighting: each row's weight is multiplied by the
probability of x given X's Markov blanket in that row, from the tbnpy objects' log_prob over
(childs, parents), and X is set to x; the rest of the row stays as it is:

    deterministic X (an edge, a system)   1[X_r = x]
    a leaf (an SAObs reading)             p(x | parents_r), the measurement's likelihood
    discrete X with children (a damage    p(x | parents_r) prod_children p(child_r | x, ...), normalised
    state, whose child is its edge)       over X's states: P(D = d | SA_r, E_r)

When the effective sample size (sum w)^2 / sum w^2 falls below `threshold` of the rows, the
session samples again with all the evidence, its chains starting from the current rows' SA and E,
drawn by weight; the burn-in stays as for a cold start. With no evidence the sampler needs no chains:
cliques.sample passes once down the tree.

The queries are weighted means over the rows, with standard errors from groups of chains: the
probability of each system's Fail and each edge's closure, L, the population cut off (the
population of each origin whose system fails; 1 each without populations, so towns), and, per
edge, E[L | do(E_i = open), evidence]: E_i set open in every row and the systems evaluated again,
the rest of the row kept as the evidence left it.
"""

from __future__ import annotations

import math
import time

import torch
from src.cancellation import check_cancelled

from src.builder.kernels import Probability
from src.builder.kernels.observation import ObservedIntensity
from . import cliques

N_GROUP = 20
SAMPLER = {"n_chain": 100, "n_burn": 200, "n_keep": 2_000, "bridge": 8}


WORDS = {"EdgeOpen": {"False": "closed", "True": "open"}}     # states as an operator says them


def _now():
    return time.strftime("%H:%M:%S")


def _deterministic(prob) -> bool:
    return type(prob).evaluate is not Probability.evaluate


def _columns(rows, variables):
    return torch.stack([rows[v.name].double() for v in variables], dim=1)


def evaluate(prob, rows):
    """A deterministic object's childs (N, n_childs) from the rows' parents, each distinct parent row
    evaluated once (edge states repeat across most rows)."""
    pars = _columns(rows, prob.parents)
    unique, inverse = torch.unique(pars, dim=0, return_inverse=True)
    return prob.evaluate(unique)[inverse]


def systems(t, rows):
    """{origin: (N,) SystemState} from the rows' edge states, each distinct edge vector evaluated once:
    the systems all read the edges in the same order."""
    check_cancelled()
    probs = list(t["systems"].values())
    names = [v.name for v in probs[0].parents]
    assert all([v.name for v in p.parents] == names for p in probs), "systems read the edges in one order"
    unique, inverse = torch.unique(_columns(rows, probs[0].parents), dim=0, return_inverse=True)
    return {k: p.evaluate(unique)[inverse, 0].long() for k, p in t["systems"].items()}


def cut_off(t, states, population):
    """(N,) L: the population of the origins whose system fails, in each row."""
    return sum(population.get(k, 1.0) * (s == 0).double() for k, s in states.items())


def features(t, rows, population):
    """(N, Q) per-row quantities and their names: P(S_k Fail), P(E_i closed), L, and per edge
    L - L under do(E_i = open), the population that edge's opening would give access back to."""
    states = {k: rows[f"SystemState::{k}"] for k in t["systems"]}
    L = cut_off(t, states, population)
    cols = [(states[k] == 0).double() for k in t["systems"]] + [(rows[c["E"]] == 0).double() for c in t["edges"]] + [L]
    names = ([f"P(SystemState::{k} Fail)" for k in t["systems"]] + [f"P({c['E']} closed)" for c in t["edges"]]
             + ["E[L]"])
    for c in t["edges"]:
        opened = {**rows, c["E"]: torch.ones_like(rows[c["E"]])}
        cols.append(L - cut_off(t, systems(t, opened), population))
        names.append(f"restored by do({c['E']} open)")
    return torch.stack(cols, dim=1), names


class Session:
    """Posterior samples given the evidence so far; observe() adds to it."""

    def __init__(self, t, probs, variables, evidence=None, population=None, threshold=0.1, sampler=None, seed=0):
        self.t, self.probs, self.variables = t, probs, variables
        self.population = population or {}
        self.threshold, self.sampler, self.seed = threshold, {**SAMPLER, **(sampler or {})}, seed
        self.evidence = dict(evidence or {})
        self.producer = {v.name: p for p in probs.values() for v in p.childs}
        self.order = list(probs.values())
        self.queries, self.timeline = {}, []
        self.rows = self.lw = None
        self.previous = None
        self._sample()
        self.timeline.append({"at": _now(), "kind": "start", "text": "Scenario started: no evidence yet",
                              "update": "sampled", "ess": 1.0, "values": {}, "elapsed_s": self.elapsed})

    # ---- kept on disk ---------------------------------------------------------------------------
    def save(self, path, **extra):
        """Everything needed to pick the session up again after a restart: the evidence, the decision queries,
        the timeline with every decision query's values at each step, the samples and their weights (discrete
        values as small integers, continuous ones in single precision), and what the caller adds (extra)."""
        meta = {"evidence": self.evidence, "queries": self.queries, "timeline": self.timeline, "sampler": self.sampler,
                "seed": self.seed, "threshold": self.threshold, "population": self.population,
                "elapsed": getattr(self, "elapsed", 0.0), **extra}
        packed = {}

        def pack(rows):
            # A reweighted round shares most columns with its predecessor; serialize those tensors only once.
            out = {}
            for k, v in rows.items():
                if id(v) not in packed:
                    packed[id(v)] = v.float() if v.is_floating_point() else v.to(torch.int16)
                out[k] = packed[id(v)]
            return out

        rows = pack(self.rows)
        previous = {**self.previous, "rows": pack(self.previous["rows"])} if self.previous else None
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        torch.save({"meta": meta, "rows": rows, "lw": self.lw, "previous": previous}, part)
        part.replace(path)

    @classmethod
    def load(cls, path, t, probs, variables):
        """A session saved by save(), on the BN rebuilt as it was (t, probs, variables); and its extra."""
        data = torch.load(path, weights_only=False)          # written by save(), not from elsewhere
        meta = data["meta"]
        s = cls.__new__(cls)
        s.t, s.probs, s.variables = t, probs, variables
        s.population, s.threshold, s.sampler, s.seed = meta["population"], meta["threshold"], meta["sampler"], meta["seed"]
        if isinstance(s.sampler.get("bridge"), (tuple, list)):        # saved as (partner tuples, candidates)
            s.sampler["bridge"] = s.sampler["bridge"][0]                # before rsr 0.3.1: keep the partners
        s.evidence, s.queries, s.timeline, s.elapsed = meta["evidence"], meta["queries"], meta["timeline"], meta["elapsed"]
        s.producer = {v.name: p for p in probs.values() for v in p.childs}
        s.order = list(probs.values())
        unpacked = {}

        def unpack(rows):
            out = {}
            for k, v in rows.items():
                if id(v) not in unpacked:
                    unpacked[id(v)] = v.double() if v.is_floating_point() else v.long()
                out[k] = unpacked[id(v)]
            return out

        s.rows = unpack(data["rows"])
        previous = data.get("previous")
        s.previous = {**previous, "rows": unpack(previous["rows"])} if previous else None
        s.lw, s.cost = data["lw"], {}
        return s, meta

    # ---- the samples --------------------------------------------------------------------------
    @property
    def n(self):
        return len(self.lw)

    def ess(self, lw=None):
        """The effective sample size as a fraction of the rows."""
        lw = self.lw if lw is None else lw
        w = (lw - lw.max()).exp()
        return (w.sum() ** 2 / (w ** 2).sum() / len(w)).item()

    def _sample(self):
        started = time.perf_counter()
        self._draw()
        self.elapsed = time.perf_counter() - started

    def _draw(self):
        sampler, sa0, e0 = dict(self.sampler), None, None
        if self.rows is not None:                             # warm start: SA and E of the rows, drawn by weight
            w = (self.lw - self.lw.max()).exp()
            pick = torch.multinomial(w, self.sampler["n_chain"], replacement=True,
                                     generator=torch.Generator().manual_seed(self.seed))
            sa0 = torch.stack([self.rows[c["SA"]][pick] for c in self.t["edges"]], dim=1)
            e0 = torch.stack([self.rows[c["E"]][pick] for c in self.t["edges"]], dim=1)
        self.rows, self.cost = cliques.sample(self.t, self.evidence, seed=self.seed, sa0=sa0, e0=e0, **sampler)
        self.lw = torch.zeros(len(self.rows[self.t["edges"][0]["SA"]]), dtype=torch.float64)
        self.seed += 1

    def _reweight(self, var, value):
        """The change in log weight for the observation var = value: log P(X = value | X's Markov blanket in
        each row), and the rows with X set to value; the rest of each row stays as it is."""
        check_cancelled()
        rows, n = dict(self.rows), self.n
        own = self.producer[var]
        children = [p for p in self.order if any(v.name == var for v in p.parents)]
        x = torch.full((n,), value, dtype=rows[var].dtype)
        if _deterministic(own):                    # P(X | parents) is 1 at the row's value only
            dlw = torch.where(rows[var] == x, 0.0, -math.inf).double()
        elif not children:                         # a leaf: its likelihood, a density for SAObs
            dlw = own.log_prob(_columns({**rows, var: x}, own.childs + own.parents))
        else:                                      # discrete, with children: normalised over its states
            def log_joint(state):
                r = {**rows, var: torch.full((n,), state, dtype=rows[var].dtype)}
                return sum(p.log_prob(_columns(r, p.childs + p.parents)) for p in [own, *children])
            states = [log_joint(s) for s in range(len(self.variables[var].values))]
            dlw = states[value] - torch.logsumexp(torch.stack(states), dim=0)
        rows[var] = x
        return rows, dlw

    def _state(self, var, value):
        """A discrete variable's state index, from an index or a state name; a float for a continuous one."""
        if var not in self.variables:
            raise ValueError(f"no variable {var!r} in this BN")
        states = self.variables[var].values
        if isinstance(states, tuple):                                       # continuous: (lower, upper)
            return float(value)
        words = WORDS.get(var.split("::")[0], {})
        if isinstance(value, str) and value in words.values():
            value = next(s for s, w in words.items() if w == value)
        if isinstance(value, str) and value in states:
            return states.index(value)
        if isinstance(value, (int, float)) and int(value) == value and 0 <= value < len(states):
            return int(value)
        raise ValueError(f"{var} has states {states}, not {value!r}")

    def observe(self, var, value, at=None, source=None, text=None):
        """Take in the observation var = value: reweight, or sample again if too little weight is left.
        Adds an evidence entry to the timeline, with every decision query's values after it."""
        if var in self.evidence:
            raise ValueError(f"{var} is already in the evidence")
        if var in self.variables and isinstance(self.variables[var].values, tuple) \
                and not isinstance(self.producer[var], ObservedIntensity):        # SA
            raise ValueError(f"{var} cannot be observed directly: ground motion is observed only through a "
                             "station's reading (SAObs), through its measurement likelihood")
        value = self._state(var, value)
        previous = self._snapshot()
        self.evidence[var] = value
        started = time.perf_counter()
        rows, dlw = self._reweight(var, value)
        lw = self.lw + dlw
        ess = self.ess(lw) if torch.isfinite(lw).any() else 0.0
        if ess < self.threshold:
            self._sample()
            update = "sampled again"
        else:
            self.rows, self.lw = rows, lw
            update = "reweighted"
            self.elapsed = time.perf_counter() - started
        entry = {"at": at or _now(), "kind": "evidence", "text": text or f"{var} = {self._label(var, value)}",
                 "evidence": {"variable": var, "value": value, "label": self._label(var, value), "source": source},
                 "update": update, "ess": ess, "values": {q: self._values(s) for q, s in self.queries.items()}}
        entry["elapsed_s"] = time.perf_counter() - started
        self.timeline.append(entry)
        self.previous = previous
        return entry

    def _label(self, var, value):
        states = self.variables[var].values
        if isinstance(states, tuple):
            return f"{value:g} g"
        return WORDS.get(var.split("::")[0], {}).get(states[value], states[value])

    # ---- decision queries -------------------------------------------------------------------------
    # A decision query: {"name", "target", "decision"}. The target is a quantity per row, summarised by its
    # weighted mean (TARGETS). The decision says what is computed of it:
    #   "inspect"  for each bridge not yet observed, what inspecting it gives back, repairing it if found
    #              damaged: VOI_i = E[T | ev] - E[T | do(E_i = open), ev], the edge set open in every row
    #              and the systems evaluated again, the rest of the row as the evidence left it
    #   "track"    the target itself, E[T | ev]
    TARGETS = {
        "cut_off": "E[L], the population cut off from a key destination (towns where none is given)",
        "cut_off_at_least": "P(L >= at_least)",
        "fail": "P(the origin's system Fail)",
        "closed": "P(the edge closed)",
    }
    DECISIONS = {
        "inspect": "which bridge to inspect first (and repair if damaged): what each gives back",
        "track": "the target itself, as the evidence comes in",
    }

    def _check_query(self, spec):
        target = dict(spec.get("target") or {})
        kind = target.get("kind")
        if kind not in self.TARGETS:
            raise ValueError(f"target kind must be one of {sorted(self.TARGETS)}, not {kind!r}")
        if kind == "fail" and target.get("origin") not in self.t["systems"]:
            raise ValueError(f"no origin {target.get('origin')!r}; have {sorted(self.t['systems'])}")
        if kind == "closed" and f"EdgeOpen::{target.get('edge')}" not in self.variables:
            raise ValueError(f"no edge {target.get('edge')!r}")
        if kind == "cut_off_at_least":
            target["at_least"] = float(target.get("at_least", 1))
        decision = spec.get("decision") or "track"
        if decision not in self.DECISIONS:
            raise ValueError(f"decision must be one of {sorted(self.DECISIONS)}, not {decision!r}")
        return {"name": spec.get("name") or self.TARGETS[kind], "target": target, "decision": decision}

    def add_query(self, spec, at=None):
        """Add a decision query; its timeline entry carries every query's values, the new one's included."""
        started = time.perf_counter()
        previous = self._snapshot()
        spec = {"id": f"q{len(self.queries) + 1}", **self._check_query(spec)}
        self.queries[spec["id"]] = spec
        values = {**self.timeline[-1]["values"], spec["id"]: self._values(spec)}
        entry = {"at": at or _now(), "kind": "query", "text": spec["name"], "query": spec, "values": values,
                 "elapsed_s": time.perf_counter() - started}
        self.timeline.append(entry)
        self.previous = previous
        return entry

    def _snapshot(self):
        """Retain one prior round. Updates replace sample columns and weights rather than mutating them."""
        return {"rows": self.rows, "lw": self.lw, "evidence": dict(self.evidence), "at": self.timeline[-1]["at"]}

    def _intervene(self, do, rows=None):
        """The rows (the session's, if not given) with do's variables set and what they decide evaluated again:
        deterministic objects only."""
        rows = dict(self.rows if rows is None else rows)
        changed, n = set(do), len(next(iter(rows.values())))
        for v, x in do.items():
            rows[v] = torch.full((n,), x, dtype=rows[v].dtype)
        system_objects = {id(p) for p in self.t["systems"].values()}
        again = False
        for prob in self.order:
            if not any(v.name in changed for v in prob.parents):
                continue
            if id(prob) in system_objects:
                again = True
                continue
            if not _deterministic(prob):
                raise ValueError(f"do({', '.join(do)}) would change {prob.childs[0].name}, which is drawn, not decided")
            out = evaluate(prob, rows)
            for j, v in enumerate(prob.childs):
                if v.name not in do:
                    rows[v.name] = out[:, j].to(rows[v.name].dtype)
                    changed.add(v.name)
        if again:
            rows.update({f"SystemState::{k}": s for k, s in systems(self.t, rows).items() if f"SystemState::{k}" not in do})
        return rows

    def _target(self, target, rows):
        kind = target["kind"]
        if kind in ("cut_off", "cut_off_at_least"):
            L = cut_off(self.t, {k: rows[f"SystemState::{k}"] for k in self.t["systems"]}, self.population)
            return L if kind == "cut_off" else (L >= target["at_least"]).double()
        if kind == "fail":
            return (rows[f"SystemState::{target['origin']}"] == 0).double()
        return (rows[f"EdgeOpen::{target['edge']}"] == 0).double()

    def _values(self, spec):
        """{label: [estimate, standard error]}: for "track", the target itself ("value"); for "inspect", per
        bridge not yet observed, what inspecting it and repairing it if damaged gives back, VOI_i."""
        check_cancelled()
        now = self._target(spec["target"], self.rows)
        if spec["decision"] == "track":
            est, se = self._means(now[:, None])
            return {"value": [est[0].item(), se[0].item()]}
        candidates = [c for c in self.t["edges"] if c["E"] not in self.evidence and c["D"] not in self.evidence]
        if not candidates:
            return {}
        # A row adds to VOI_i only where edge i is closed (opening an open edge changes nothing): evaluate those rows only.
        F = torch.zeros(self.n, len(candidates), dtype=torch.float64)
        for j, c in enumerate(candidates):
            check_cancelled()
            closed = self.rows[c["E"]] == 0
            if closed.any():
                sub = {k: v[closed] for k, v in self.rows.items()}
                F[closed, j] = now[closed] - self._target(spec["target"], self._intervene({c["E"]: 1}, sub))
        est, se = self._means(F)
        return {c["D"].split("::")[1]: [est[j].item(), se[j].item()] for j, c in enumerate(candidates)}

    def _means(self, F, lw=None):
        """Weighted means of the columns of F (N, Q), with standard errors from N_GROUP groups of chains."""
        lw = self.lw if lw is None else lw
        w = (lw - lw.max()).exp()
        group = (torch.arange(len(lw)) % self.sampler["n_chain"]) % N_GROUP
        sw = torch.zeros(N_GROUP, dtype=torch.float64).index_add_(0, group, w)
        swf = torch.zeros(N_GROUP, F.shape[1], dtype=torch.float64).index_add_(0, group, w[:, None] * F)
        means = swf[sw > 0] / sw[sw > 0, None]
        return swf.sum(0) / sw.sum(), means.std(0) / math.sqrt(len(means))

    def distribution(self, var, bins=40):
        """Current and previous-round distributions, using the same states or histogram bins for comparison."""
        if var not in self.rows:
            raise ValueError(f"no variable {var!r} in the samples")
        previous = self.previous if self.previous and var in self.previous["rows"] else None
        edges = None
        if isinstance(self.variables[var].values, tuple):
            samples = [self.rows[var]] + ([previous["rows"][var]] if previous else [])
            lo = min(x.log().min().item() for x in samples)
            hi = max(x.log().max().item() for x in samples)
            edges = torch.linspace(lo, hi + 1e-9, bins + 1, dtype=torch.float64)
        out = self._distribution(var, self.rows, self.lw, self.evidence, self.timeline[-1]["at"], edges)
        out["previous"] = self._distribution(var, previous["rows"], previous["lw"], previous["evidence"], previous["at"], edges) if previous else None
        return out

    def _distribution(self, var, rows, lw, evidence, at, edges):
        x, states = rows[var], self.variables[var].values
        out = {"node": var, "at": at, "observed": var in evidence, "n": len(lw), "ess": self.ess(lw)}
        if not isinstance(states, tuple):
            F = torch.nn.functional.one_hot(x.long(), len(states)).double()
            est, se = self._means(F, lw)
            words = WORDS.get(var.split("::")[0], {})
            return {**out, "kind": "discrete", "states": [words.get(s, s) for s in states], "p": est.tolist(),
                    "se": se.tolist()}
        lx = x.log()
        n_bins = len(edges) - 1
        k = (torch.bucketize(lx, edges) - 1).clamp(0, n_bins - 1)
        F = torch.nn.functional.one_hot(k, n_bins).double() / (edges[1] - edges[0])
        est, se = self._means(F, lw)
        mean, _ = self._means(lx[:, None], lw)
        return {**out, "kind": "continuous", "unit": "g", "ln_edges": edges.tolist(), "density": est.tolist(),
                "se": se.tolist(), "median_g": math.exp(mean.item())}

    def result(self, computed=(), statement=""):
        """The map's numbers given the evidence so far, in the shape of an engine result: P(Fail) per
        origin and P(closed) per edge, each with a 95% interval from the groups of chains; the rsr rules
        report; the probability objects' notes."""
        from . import engine

        events = []
        names = ([(f"SystemState::{k}", "Fail") for k in self.t["systems"]]
                 + [(c["E"], "False") for c in self.t["edges"]])
        F = torch.stack([(self.rows[n] == 0).double() for n, _ in names], dim=1)
        est, se = self._means(F)
        for j, (node, value) in enumerate(names):
            p, half = est[j].item(), 1.96 * se[j].item()
            events.append({"node": node, "op": "==", "value": value, "p": p,
                           "ci95": [max(0.0, p - half), min(1.0, p + half)]})
        notes = []
        for prob in self.probs.values():
            notes.extend(w for w in prob.notes if w not in notes)
        report = engine.rules_report(list(self.t["systems"].values()))
        return {"events": events, "targets": [], "rules": {k: v for k, v in report.items() if k != "note"},
                "statement": statement or "Every origin's access and every edge's state, given the evidence so far",
                "samples": self.n, "elapsed_s": round(self.elapsed, 1), "computed": list(computed),
                "warnings": notes + [report["note"]], "record": None,
                "inference": {"method": "clique tree (src/inference/cliques.py)", "ess": self.ess()}}

    def state(self):
        """What the page shows: the evidence, the decision queries and the timeline."""
        kinds = {"SAObs": "reading", "DamageState": "damage", "EdgeOpen": "edge", "SystemState": "system"}
        variables = {}
        for name in self.rows:                                    # what can be observed or decided; SA cannot
            kind = kinds.get(name.split("::")[0].split("(")[0].split(".")[0])
            if kind:
                states = self.variables[name].values
                words = WORDS.get(name.split("::")[0], {})
                variables[name] = {"kind": kind, "states": None if isinstance(states, tuple)
                                   else [words.get(s, s) for s in states]}
        return {"evidence": {v: {"value": x, "label": self._label(v, x)} for v, x in self.evidence.items()},
                "queries": list(self.queries.values()), "timeline": self.timeline, "n": self.n, "ess": self.ess(),
                "unit": "people" if self.population else "towns", "targets": self.TARGETS, "decisions": self.DECISIONS, "variables": variables,
                "origins": list(self.t["systems"]), "edges": [c["E"].split("::")[1] for c in self.t["edges"]]}

    # ---- the dispatch table, for examples/clique-sampling/dispatch.py ---------------------------
    def estimates(self):
        """{name: (estimate, standard error)} of every quantity in features(): weighted means over the
        rows, the standard error from N_GROUP groups of chains."""
        F, names = features(self.t, self.rows, self.population)
        est, se = self._means(F)
        return {name: (est[j].item(), se[j].item()) for j, name in enumerate(names)}

    def dispatch(self, est=None):
        """Per edge: P(closed | evidence), E[L | evidence], E[L | do(open), evidence] and the population
        its opening gives access back to, with standard errors; observed edges flagged."""
        est = est or self.estimates()
        L = est["E[L]"]
        out = []
        for c in self.t["edges"]:
            back = est[f"restored by do({c['E']} open)"]
            out.append({"edge": c["E"].split("::")[1], "observed": c["E"] in self.evidence or c["D"] in self.evidence,
                        "p_closed": est[f"P({c['E']} closed)"], "L": L, "L_do_open": (L[0] - back[0], back[1]),
                        "restored": back})
        return out

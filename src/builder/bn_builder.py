"""Compile an assembly into a Bayesian network: replicate, then wire.

Each binding names a function and the rows it describes. Compiling it

1. replicates the function once per row of its `per` plate (filtered by `where`),
   giving one node ``<Quantity>::<row id>`` per output;
2. resolves each row's parameter set, so ``"hazus61.{hazus_class}"`` hands every
   bridge the curve for its own class, and ``parameter_set_by_location`` hands it the
   first listed set whose region contains it; then each "{imt}" port becomes the IM
   that set names;
3. resolves each input's parents by following its `via` through the network's
   foreign keys, then finds the one node producing that quantity on that row.

Wiring is by quantity, so an output reaches an input only when the types agree,
and every (quantity, row) must have exactly one producer: two hazard bindings
over the same sites is an error, as is a fragility curve whose sites nobody
shakes. The binding graph must be acyclic; that order is the sampling order.
"""

from __future__ import annotations

import copy
import re
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .load import (Library, Network, RegistryError, assembly_path, load_assembly, load_library, load_network,
                   resolve_quantities, via_kind)

_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def node_id(quantity: str, row_id: str) -> str:
    return f"{quantity}::{row_id}"


@dataclass
class Binding:
    index: int
    spec: dict[str, Any]                     # the assembly entry
    function: dict[str, Any]
    rows: list[dict[str, Any]]
    parameter_sets: list[str | None]         # per row; None for inline or no parameters
    parameters: list[dict[str, Any] | None]  # per row, resolved values
    quantities: list[dict[str, str]]         # per row, port name -> quantity, "{imt}" resolved
    parents: dict[str, list[list[str]]]      # input name -> per row, the parent row ids
    upstream: set[int]
    correlation: str | None = None           # the correlation_model's set correlating the rows; None: independent

    @property
    def label(self) -> str:
        return f"#{self.index} {self.function['id']}"

    def port_quantity(self, name: str) -> str:
        """The port's quantity across this binding's rows, e.g. "SA(1.0)", or "PGA | SA(1.0)" if they differ."""
        return " | ".join(dict.fromkeys(q[name] for q in self.quantities))


@dataclass
class Compiled:
    assembly: dict[str, Any]
    library: Library
    network: Network
    bindings: list[Binding]
    producer: dict[tuple[str, str], int]     # (quantity, row id) -> binding index
    order: list[int]

    @property
    def n_nodes(self) -> int:
        return len(self.producer)

    def ancestors(self, indices: set[int]) -> set[int]:
        """These bindings and every binding they read from, directly or not."""
        out, todo = set(), list(indices)
        while todo:
            i = todo.pop()
            if i not in out:
                out.add(i)
                todo.extend(self.bindings[i].upstream)
        return out

    def nodes(self) -> list[dict[str, Any]]:
        """Every BN node, with its parents, in sampling order."""
        out = []
        for i in self.order:
            b = self.bindings[i]
            for r, row in enumerate(b.rows):
                q = b.quantities[r]
                parents = [
                    node_id(q[port["name"]], pid)
                    for port in b.function["inputs"]
                    for pid in b.parents[port["name"]][r]
                ]
                for port in b.function["outputs"]:
                    out.append({
                        "id": node_id(q[port["name"]], row["id"]),
                        "quantity": q[port["name"]],
                        "entity": b.function["per"],
                        "row": row["id"],
                        "function": b.function["id"],
                        "parameter_set": b.parameter_sets[r],
                        "parents": parents,
                    })
        return out

    def summary(self) -> list[dict[str, Any]]:
        out = []
        for i in self.order:
            b = self.bindings[i]
            inputs = []
            for port in b.function["inputs"]:
                counts = [len(p) for p in b.parents[port["name"]]]
                inputs.append({**port, "quantity": b.port_quantity(port["name"]),
                               "parents_min": min(counts), "parents_max": max(counts)})
            out.append({
                "binding": i,
                "function": b.function["id"],
                "per": b.function["per"],
                "rows": len(b.rows),
                "nodes": len(b.rows) * len(b.function["outputs"]),
                "parameter_sets": dict(Counter(s for s in b.parameter_sets if s)),
                "correlation": b.correlation,
                "inline_parameters": b.spec.get("parameters"),
                "inputs": inputs,
                "upstream": sorted(b.upstream),
            })
        return out


def _matches(row: dict[str, Any], where: dict[str, Any]) -> bool:
    for attr, wanted in where.items():
        value = row.get(attr)
        if isinstance(wanted, list) and value not in wanted:
            return False
        if not isinstance(wanted, list) and value != wanted:
            return False
    return True


def _few(items: list[str], n: int = 3) -> str:
    return ", ".join(items[:n]) + (f" ... ({len(items)} in all)" if len(items) > n else "")


def contains(polygon: list[list[float]], lat: float, lon: float) -> bool:
    """Whether (lat, lon) lies inside a polygon of [lon, lat] vertices (ray casting; the ring closes itself)."""
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _by_location(choices: list[str], fn: dict[str, Any], rows: list[dict[str, Any]], lib: Library,
                 where_label: str, problems: list[str]) -> tuple[list[str | None], list[dict[str, Any] | None]]:
    """Each row's parameter set: the first of ``choices`` whose region contains the row's lat/lon."""
    sets: list[str | None] = [None] * len(rows)
    values: list[dict[str, Any] | None] = [None] * len(rows)
    before = len(problems)
    for set_id in choices:
        found = lib.parameter_sets.get(set_id)
        if found is None or found["function"] != fn["id"]:
            problems.append(f"{where_label}: no parameter set {set_id!r} for {fn['id']}")
        elif "region" not in found:
            problems.append(f"{where_label}: {set_id} has no region, so it cannot be chosen by location")
    attrs = lib.entities[fn["per"]]["attributes"]
    if not {"lat", "lon"} <= attrs.keys():
        problems.append(f"{where_label}: {fn['per']} has no lat/lon to choose a parameter set by")
    if len(problems) > before:
        return sets, values
    uncovered = []
    for r, row in enumerate(rows):
        chosen = next((s for s in choices
                       if contains(lib.parameter_sets[s]["region"]["polygon"], row["lat"], row["lon"])), None)
        if chosen is None:
            uncovered.append(row["id"])
        else:
            sets[r], values[r] = chosen, lib.parameter_sets[chosen]["values"]
    if uncovered:
        regions = "; ".join(f"{s}: {lib.parameter_sets[s]['region']['name']}" for s in choices)
        problems.append(f"{where_label}: no listed parameter set covers {fn['per']} {_few(uncovered)} ({regions})")
    return sets, values


def compile_assembly(assembly: dict[str, Any], lib: Library, network: Network) -> Compiled:
    problems: list[str] = []
    bindings: list[Binding] = []
    producer: dict[tuple[str, str], int] = {}

    # Pass 1: rows, parameters and outputs of every binding.
    for index, spec in enumerate(assembly["bindings"]):
        where_label = f"{assembly['id']} binding #{index} ({spec['function']})"
        fn = lib.functions.get(spec["function"])
        if fn is None:
            problems.append(f"{where_label}: function is not in the library")
            continue
        per = fn["per"]
        where = spec.get("where", {})
        for attr in where:
            if attr not in lib.entities[per]["attributes"]:
                problems.append(f"{where_label}: where {attr!r} is not an attribute of {per}")
        rows = [r for r in network.rows(per) if _matches(r, where)]
        if not rows:
            problems.append(f"{where_label}: matches no {per} rows in network {network.id}")
            continue

        for attr in fn.get("attributes", []):
            missing = [r["id"] for r in rows if r.get(attr) is None]
            if missing:
                problems.append(f"{where_label}: {fn['id']} reads {per}.{attr}, missing on {_few(missing)}")
        if fn["role"] == "system":
            named = {e.get("analysis") for e in network.rows("SystemEvent")}
            if named and fn["id"] not in named:
                problems.append(f"{where_label}: the network's system event is judged by {', '.join(sorted(named))}, "
                                f"not {fn['id']}")
        for table in fn.get("tables", []):
            if lib.entities[table].get("singleton") and len(network.rows(table)) != 1:
                problems.append(f"{where_label}: {fn['id']} reads the {table} row, network {network.id} has "
                                f"{len(network.rows(table))}")

        sets: list[str | None] = [None] * len(rows)
        values: list[dict[str, Any] | None] = [None] * len(rows)
        if "instance_schema" not in fn:
            if any(k in spec for k in ("parameter_set", "parameters", "parameter_set_by_location")):
                problems.append(f"{where_label}: {fn['id']} takes no parameters")
        elif "parameter_set_by_location" in spec:
            sets, values = _by_location(spec["parameter_set_by_location"], fn, rows, lib, where_label, problems)
        elif "parameters" in spec:
            errors = list(Draft202012Validator(fn["instance_schema"]).iter_errors(spec["parameters"]))
            for error in errors:
                at = "/".join(str(p) for p in error.absolute_path) or "(parameters)"
                problems.append(f"{where_label}: {at}: {error.message}")
            if not errors:                          # invalid values are reported, not used
                values = [spec["parameters"]] * len(rows)
        elif "parameter_set" in spec:
            template = spec["parameter_set"]
            unresolved: dict[str, list[str]] = defaultdict(list)
            for r, row in enumerate(rows):
                try:
                    set_id = _PLACEHOLDER.sub(lambda m: str(row[m.group(1)]), template)
                except KeyError as exc:
                    problems.append(f"{where_label}: {template!r} needs {per}.{exc.args[0]}, missing on {row['id']}")
                    continue
                found = lib.parameter_sets.get(set_id)
                if found is None or found["function"] != fn["id"]:
                    unresolved[set_id].append(row["id"])
                    continue
                sets[r], values[r] = set_id, found["values"]
            for set_id, ids in unresolved.items():
                problems.append(f"{where_label}: no parameter set {set_id!r} for {fn['id']}, needed by {_few(ids)}")
        else:
            problems.append(f"{where_label}: {fn['id']} needs parameter_set, parameter_set_by_location or parameters")

        correlation = spec.get("correlation")
        if correlation is not None:
            model = fn.get("correlation_model")
            found = lib.parameter_sets.get(correlation)
            if model is None:
                problems.append(f"{where_label}: {fn['id']} has no correlation between its rows to set")
            elif found is None or found.get("model") != model:
                sets = sorted(s for s, p in lib.parameter_sets.items() if p.get("model") == model)
                problems.append(f"{where_label}: no {model} set {correlation!r}; have {', '.join(sets)}")

        quantities = [resolve_quantities(fn, v) for v in values]
        for port in fn["outputs"]:
            for row, q in zip(rows, quantities):
                key = (q[port["name"]], row["id"])
                if key in producer:
                    other = assembly["bindings"][producer[key]]["function"]
                    problems.append(
                        f"{where_label}: {node_id(*key)} is already produced by binding "
                        f"#{producer[key]} ({other}); each node needs exactly one producer"
                    )
                else:
                    producer[key] = index

        bindings.append(Binding(index, spec, fn, rows, sets, values, quantities, {}, set(), correlation))

    # Pass 2: parents, now that every producer is known.
    for b in bindings:
        per = b.function["per"]
        for port in b.function["inputs"]:
            over, via = port["over"], port["via"]
            kind = via_kind(via)
            label = f"{assembly['id']} binding #{b.index} ({b.function['id']}) input {port['name']!r}"
            if kind == "all" and lib.entities[over].get("singleton") and len(network.rows(over)) != 1:
                problems.append(f"{label}: needs exactly one {over} row, network {network.id} has {len(network.rows(over))}")
            per_row: list[list[str]] = []
            orphans: dict[str, list[str]] = defaultdict(list)       # quantity -> parent rows nobody produces it for
            for row, q in zip(b.rows, b.quantities):
                quantity = q[port["name"]]
                if kind == "self":
                    pids = [row["id"]]
                elif kind == "fk":
                    pids = [row[via]] if row.get(via) is not None else []
                    if not pids:
                        problems.append(f"{label}: {per} {row['id']!r} has no {via}")
                elif kind == "reverse":
                    attr = via.partition(".")[2]
                    pids = [r["id"] for r in network.rows(over) if r.get(attr) == row["id"]]
                else:
                    pids = [r["id"] for r in network.rows(over)]
                for pid in pids:
                    source = producer.get((quantity, pid))
                    if source is None:
                        orphans[quantity].append(pid)
                    else:
                        b.upstream.add(source)
                per_row.append(pids)
            for quantity, pids in orphans.items():
                unique = list(dict.fromkeys(pids))
                made = sorted({q for (q, pid) in producer if pid in unique[:1]})
                hint = f"; {over} {unique[0]} has {', '.join(made)}" if made else ""
                problems.append(f"{label}: no binding produces {quantity} for {over} {_few(unique)}{hint}")
            b.parents[port["name"]] = per_row

    # Pass 3: a sampling order over bindings.
    by_index = {b.index: b for b in bindings}
    remaining = {i: set(b.upstream) for i, b in by_index.items()}
    order: list[int] = []
    while remaining:
        ready = sorted(i for i, deps in remaining.items() if not deps)
        if not ready:
            cycle = ", ".join(by_index[i].label for i in sorted(remaining))
            problems.append(f"{assembly['id']}: bindings form a cycle among {cycle}")
            break
        for i in ready:
            order.append(i)
            del remaining[i]
        for deps in remaining.values():
            deps.difference_update(ready)

    if problems:
        raise RegistryError(problems)
    indexed = [by_index[i] for i in sorted(by_index)]
    # Binding.index is the position in the assembly; keep the list addressable by it.
    full: list[Binding] = [None] * len(assembly["bindings"])  # type: ignore[list-item]
    for b in indexed:
        full[b.index] = b
    return Compiled(assembly, lib, network, full, producer, order)


def compile_path(path: Path | str, lib: Library | None = None) -> Compiled:
    lib = lib or load_library()
    assembly = load_assembly(path)
    return compile_assembly(assembly, lib, load_network(assembly["network"], lib))


# ---------------------------------------------------------------------------
# The model as a tbnpy spec
# ---------------------------------------------------------------------------
TBN_FORMAT = "quakenet.tbnpy.v5"
NETWORK_FILE = "system-network.json"       # the network the system objects share, written once beside model.json


def _sites(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": row["id"], "lat": row["lat"], "lon": row["lon"]} for row in rows]


def _gmm_arguments(rows, params, tables, parent_rows, states):
    """Gmm: the GMM's coefficients from its parameter set; the Event row and the sites its own."""
    from .kernels.hazard import fixed_event

    event = fixed_event(tables["Event"])
    return {"gmm": params[0]}, {"event": {k: event[k] for k in ("id", "mw", "lat", "lon", "depth_km")},
                                "sites": _sites(rows)}


def _shakemap_arguments(rows, params, tables, parent_rows, states):
    """ShakeMap: which map, from its parameters; the sites its own."""
    p = params[0]
    return ({"event_id": p["event_id"], "imt": p["imt"], "source": p.get("source"), "version": p.get("version")},
            {"sites": _sites(rows)})


def _observation_arguments(rows, params, tables, parent_rows, states):
    """ObservedIntensity: the relative standard deviation, from its parameter set."""
    (p,) = params
    return {"relative_sd": p["relative_sd"]}, {}


def _fragility_arguments(rows, params, tables, parent_rows, states):
    """Fragility: the curve's median and beta for each damage state after the first, all from its parameter set."""
    (p,) = params
    curve, limit_states = p["parameters"], states["damage"][1:]
    return {"medians": [curve["medians"][s] for s in limit_states],
            "betas": [curve["betas"][s] for s in limit_states] if "betas" in curve else curve["beta"],
            "curve": p["form"]}, {}


def _edge_arguments(rows, params, tables, parent_rows, states):
    """EdgeOpen: the damage states its closure policy keeps a bridge open in, from its parameter set."""
    (p,) = params
    return {"usable_states": p["usable_states"]}, {}


def _system_arguments(rows, params, tables, parent_rows, states):
    """TravelTimeToNearest: no parameter set; the origin, the edges in parent order, the threshold, the network."""
    events = tables["SystemEvent"]
    if len(events) != 1:
        raise ValueError(f"a system function judges exactly one system event, got {len(events)}")
    (row,) = rows
    return {}, {"origin": row["id"], "edge_ids": parent_rows["open"][0], "target_add_h": events[0]["target_add_h"],
                "network": {name: table for name, table in tables.items() if name != "SystemEvent"}}


# The constructor arguments of each kernel, from one group of a binding's rows: the rows, their
# parameter values, the function's tables, each input port's parent row ids, and the discrete states.
# Each returns (from the parameter set, the node's own): the first is written once in the spec's
# parameter_sets, under the set's id, and every node on that set names it.
KERNEL_ARGUMENTS = {
    "src.builder.kernels.hazard:Gmm": _gmm_arguments,
    "src.builder.kernels.hazard:ShakeMap": _shakemap_arguments,
    "src.builder.kernels.observation:ObservedIntensity": _observation_arguments,
    "src.builder.kernels.fragility:Fragility": _fragility_arguments,
    "src.builder.kernels.edge:EdgeOpen": _edge_arguments,
    "src.builder.kernels.system:TravelTimeToNearest": _system_arguments,
}


def tbn_data(compiled: Compiled, bindings: set[int] | None = None) -> dict[str, Any]:
    """The model as a tbnpy spec: its variables, and each probability object as a ``custom`` node.

    A node names the object's class (``module``, ``cls``), its ``childs`` and ``parents``,
    and its constructor's arguments, so ``tbnpy.compile`` builds the object from the spec.
    Arguments from a library parameter set (a fragility curve, a closure policy, a GMM) are
    written once, in ``parameter_sets`` under the set's id, and named by each node built
    from it (``parameter_set``); ``kwargs`` hold the node's own. ``metadata`` keeps what this
    repository's engine needs to know about it: the function, its role and its binding.

    A stochastic binding drawn jointly (``drawKind: Joint``, e.g. a hazard field) is one
    object per parameter set, whose childs are its rows' outputs: those sites are sampled
    together, by one model written once. Every other binding is one object per row, each
    child with its own parents, as a Bayesian network has it: a bridge's damage depends on
    its own intensity, an origin's state on the edge states. ``bindings`` restricts the
    spec to those bindings, which must be closed under ``upstream``. The spec is a portable
    snapshot of the selected inputs, not sampled data.
    """
    variables, nodes, parameter_sets = {}, [], {}
    lib = compiled.library

    for index in compiled.order:
        if bindings is not None and index not in bindings:
            continue
        b = compiled.bindings[index]
        fn = b.function

        # The states of each discrete port, which must agree across the binding's rows.
        states = {}
        for port in fn["inputs"] + fn["outputs"]:
            used = {b.quantities[r][port["name"]] for r in range(len(b.rows))}
            if {lib.quantities[q]["kind"] for q in used} == {"discrete"}:
                if len({tuple(lib.quantities[q]["states"]) for q in used}) != 1:
                    raise ValueError(f"{fn['id']}: {port['name']} has different states across rows")
                states[port["name"]] = lib.quantities[next(iter(used))]["states"]
        tables = {t: compiled.network.rows(t) for t in fn.get("tables", [])}

        arguments = KERNEL_ARGUMENTS.get(fn["implementation"])
        if arguments is None:
            raise ValueError(f"{fn['id']}: no constructor arguments are defined for {fn['implementation']}")
        module, cls = fn["implementation"].split(":")

        # A joint object is one model over its sites: rows on different parameter sets
        # (e.g. two GMM regions) become one object per set, each set written once.
        if fn["drawKind"] == "Joint" and not fn["deterministic"]:
            by_set: dict[str, list[int]] = {}
            for r in range(len(b.rows)):
                by_set.setdefault(b.parameter_sets[r] or json.dumps(b.parameters[r], sort_keys=True), []).append(r)
            groups, joint = list(by_set.values()), True
        else:
            groups, joint = [[r] for r in range(len(b.rows))], False

        for group in groups:
            parents = []
            for r in group:
                for port in fn["inputs"]:
                    for pid in b.parents[port["name"]][r]:
                        name = node_id(b.quantities[r][port["name"]], pid)
                        if name not in parents:
                            parents.append(name)

            childs = []
            for r in group:
                for port in fn["outputs"]:
                    q = lib.quantities[b.quantities[r][port["name"]]]
                    name = node_id(q["id"], b.rows[r]["id"])
                    childs.append(name)
                    variables[name] = _variable_spec(name, q)

            # A per-row object is keyed by its child, as tbnpy's examples key theirs.
            if joint:
                key = fn["id"] if len(groups) == 1 else f"{fn['id']}[{b.parameter_sets[group[0]] or index}]"
            else:
                key = childs[0] if len(childs) == 1 else node_id(fn["id"], b.rows[group[0]]["id"])
            if any(node["name"] == key for node in nodes):
                key = f"{key}#{index}"

            rows, params = [b.rows[r] for r in group], [b.parameters[r] for r in group]
            parent_rows = {port: [b.parents[port][r] for r in group] for port in b.parents}
            shared, own = arguments(rows, params, tables, parent_rows, states)
            if b.correlation:                       # the binding's correlation between the object's rows
                own = {**own, "correlation": {"id": b.correlation, **lib.parameter_sets[b.correlation]["values"]}}
            node = {"name": key, "type": "custom", "module": module, "cls": cls,
                    "childs": childs, "parents": parents}
            # One parameter set per object: a joint object is grouped by set, any other has one row.
            set_id = b.parameter_sets[group[0]]
            if set_id:
                if parameter_sets.setdefault(set_id, shared) != shared:
                    raise ValueError(f"{fn['id']}: parameter set {set_id} gives different arguments in two bindings")
                node["parameter_set"] = set_id
            else:                                   # inline parameters belong to this binding alone
                own = {**shared, **own}
            if own:
                node["kwargs"] = own
            node["metadata"] = {"function": fn["id"], "role": fn["role"], "binding": index}
            if "SystemEvent" in tables:
                node["metadata"]["analysis"] = tables["SystemEvent"][0].get("analysis") if tables["SystemEvent"] else None
            nodes.append(node)

    return {
        "format": TBN_FORMAT, "spec_version": 3,
        "name": compiled.assembly["id"], "description": compiled.assembly.get("label", ""),
        "assembly": compiled.assembly,
        "network": compiled.network.id,
        "network_assumptions": compiled.network.assumptions,
        "instance_sources": {
            s: {"file": lib.parameter_sets[s]["file"], "source": lib.parameter_sets[s]["source"]}
            for i, b in enumerate(compiled.bindings) if bindings is None or i in bindings
            for s in [*b.parameter_sets, b.correlation] if s
        },
        "variables": list(variables.values()), "parameter_sets": parameter_sets, "nodes": nodes,
    }


def _variable_spec(name: str, quantity: dict[str, Any]) -> dict[str, Any]:
    """A tbnpy variable: discrete with its states, or continuous with its support (null: unbounded)."""
    if quantity["kind"] == "discrete":
        return {"name": name, "kind": "discrete", "states": quantity["states"]}
    out = {"name": name, "kind": "continuous"}
    support = quantity.get("range", [None, None])
    if support != [None, None]:
        out["support"] = support
    if quantity.get("unit"):
        out["unit"] = quantity["unit"]
    return out


def tbn_nodes(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The spec's nodes by name."""
    return {node["name"]: node for node in data["nodes"]}


def compile_tbn(data: dict[str, Any], device: str = "cpu",
                directory: Path | str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """tbnpy's two dictionaries, variables and probability objects, compiled from the spec by tbnpy.
    ``directory`` is the folder the spec's file names (the system objects' rules) are relative to."""
    from tbnpy.compile import compile_model
    from tbnpy.spec import ModelSpec

    return compile_model(ModelSpec.from_dict(data), device=device, directory=directory)


def _tbn_fingerprint(data: dict[str, Any]) -> str:
    """The model's identity: the spec without its provenance and without which stored rules it uses."""
    snapshot = {k: v for k, v in data.items() if k not in ("source_sha256", "versions")}
    snapshot["nodes"] = [{**node, "kwargs": {k: v for k, v in node.get("kwargs", {}).items() if k != "rules"}}
                         for node in data["nodes"]]
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


# rsr's extraction stops when the unknown region's probability, under its 50/50 search proposal, falls
# below RSR_UNKNOWN_TARGET, or after RSR_MAX_ROUNDS rounds: each round resolves one unclassified state with
# ndtools and adds the rule it yields, so the rounds also bound the number of rules.
RSR_UNKNOWN_TARGET = 0.01
RSR_MAX_ROUNDS = 3000


def system_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The spec's system-state nodes, one per origin."""
    return [node for node in data["nodes"] if node["metadata"]["role"] == "system"]


def rules_threshold(node: dict[str, Any]) -> float:
    """The system event's threshold a system-state node is judged by: its target_add_h, in hours."""
    return float(node["kwargs"]["target_add_h"])


def rules_fingerprint(node: dict[str, Any]) -> str:
    """What an origin's system rules depend on, apart from the threshold: the graph, the destinations,
    the analysis, the origin and the edge order. Not the hazard: rules extracted once serve every
    scenario. The threshold is stored beside the rules, since some rules hold across thresholds.
    (The snapshot keeps the form it had before model.json became a tbnpy spec, so stored rules stay valid.)"""
    kwargs = node["kwargs"]
    snapshot = {"function": node["metadata"]["function"], "tables": kwargs["network"],
                "events": [{"analysis": node["metadata"].get("analysis")}],
                "origin": kwargs["origin"], "edges": kwargs["edge_ids"]}
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def _rules_meta(path: Path) -> dict[str, Any]:
    meta = path.with_name("model.json")
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}


def model_dir(assembly_file: Path | str) -> Path:
    """Where an assembly's generated tbnpy model lives: examples/<n>/tbnpy/<assembly name>/."""
    assembly_file = Path(assembly_file)
    return assembly_file.parent / "tbnpy" / assembly_file.name.removesuffix(".assembly.json")


def rules_folder(directory: Path, origin: str, threshold: float) -> Path:
    """rsr/<origin>/target_add_h=<value>/: one set of rules per origin and threshold."""
    return Path(directory) / "rsr" / origin / f"target_add_h={threshold:g}"


def stored_rules(directory: Path, node: dict[str, Any]) -> list[tuple[float, Path]]:
    """(threshold, folder) for every set of rules stored for this origin's definition, by threshold."""
    fingerprint, found = rules_fingerprint(node), []
    for folder in (Path(directory) / "rsr" / node["kwargs"]["origin"]).glob("target_add_h=*"):
        meta = _rules_meta(folder / "refs_up_1.pt")
        if meta.get("rules_sha256") == fingerprint and (folder / "refs_up_1.pt").is_file():
            found.append((float(meta["target_add_h"]), folder))
    return sorted(found)


_SAME = 1e-12


def choose_rules(stored: list[tuple[float, Path]], threshold: float) -> dict[str, Any] | None:
    """The rules that hold at ``threshold``, from those stored at any threshold.

    The state is monotone in the threshold: reaching a destination within t_s of the
    intact time means reaching it within any t >= t_s, and failing within t_f means
    failing within any t <= t_f. So survival rules hold when extracted at the nearest
    t_s <= t, and failure rules at the nearest t_f >= t. Only an exact match has both.
    """
    survival = [(t, f) for t, f in stored if t <= threshold + _SAME]
    failure = [(t, f) for t, f in stored if t >= threshold - _SAME]
    upper, lower = (survival[-1] if survival else None), (failure[0] if failure else None)
    if upper is None and lower is None:
        return None
    return {"upper": upper[1] / "refs_up_1.pt" if upper else None, "upper_at": upper[0] if upper else None,
            "lower": lower[1] / "refs_low_0.pt" if lower else None, "lower_at": lower[0] if lower else None}


def rules_source(ref: dict[str, Any] | None, threshold: float) -> str:
    """How a system state's rules were found: in words, for notes and the page."""
    if ref is None:
        return "no stored rules"
    up, low = ref.get("upper_at"), ref.get("lower_at")
    if up is not None and low is not None and abs(up - threshold) < _SAME and abs(low - threshold) < _SAME:
        return "exact"
    parts = ([f"survival rules from target_add_h={up:g}"] if up is not None else []) + \
            ([f"failure rules from target_add_h={low:g}"] if low is not None else [])
    return " and ".join(parts)


def attach_rules(data: dict[str, Any], directory: Path) -> int:
    """Give each system-state node the stored rules that hold at its threshold, as its ``rules``
    argument: file names relative to ``directory``. Returns how many origins have any."""
    found = 0
    for node in system_nodes(data):
        ref = choose_rules(stored_rules(directory, node), rules_threshold(node))
        if ref:
            ref = {k: (v.relative_to(directory).as_posix() if isinstance(v, Path) else v) for k, v in ref.items()}
        node["kwargs"]["rules"] = ref
        found += ref is not None
    return found


def _check_reference(directory: Path, node: dict[str, Any], ref: dict[str, Any]) -> None:
    threshold, fingerprint = rules_threshold(node), rules_fingerprint(node)
    for side, holds in (("upper", lambda t: t <= threshold + _SAME), ("lower", lambda t: t >= threshold - _SAME)):
        if ref.get(side) is None:
            continue
        meta = _rules_meta(Path(directory) / ref[side])
        if meta.get("rules_sha256") != fingerprint or not holds(float(meta.get("target_add_h", "nan"))):
            raise ValueError("RSR references belong to a different system definition; regenerate")


def build_tbn(compiled: Compiled, bindings: set[int] | None = None, device: str = "cpu",
              rules: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """tbnpy's two dictionaries, variables and probabilities, for a compiled assembly.

    With ``rules``, system states reuse the rsr rules stored with the assembly's
    generated model that hold at the query's threshold (see ``choose_rules``);
    whatever they leave unclassified is evaluated by ndtools.
    """
    data = tbn_data(compiled, bindings)
    directory = None
    if rules:
        try:
            candidate = model_dir(assembly_path(compiled.assembly["id"]))
        except RegistryError:
            candidate = None
        if candidate is not None and attach_rules(data, candidate):
            directory = candidate
    return compile_tbn(data, device=device, directory=directory)


def extract_system_rules(compiled: Compiled, directory: Path, origins: set[str] | None = None, *,
                  seed: int = 20261022, log: Callable[[str], None] = print) -> list[str]:
    """Run rsr for each origin (or those in ``origins``) at the system event's threshold, and store the
    rules in rules_folder(). Returns the origins extracted."""
    import random
    import torch
    from rsr.rsr import run_ref_extraction_by_mcs

    data = tbn_data(compiled)
    _, probs = compile_tbn(data)
    done = []
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            random.seed(seed)
            for node in system_nodes(data):
                origin, edges, threshold = node["kwargs"]["origin"], node["kwargs"]["edge_ids"], rules_threshold(node)
                if origins is not None and origin not in origins:
                    continue
                folder = rules_folder(directory, origin, threshold)
                folder.mkdir(parents=True, exist_ok=True)
                # RSR appends metrics; a new extraction starts a fresh record.
                (folder / "metrics.json").write_text("", encoding="utf-8")
                log(f"Extracting system rules for {origin} at target_add_h={threshold:g}...")
                evaluate = probs[node["name"]].system_function()

                def binary_system(comps, evaluate=evaluate):
                    # The system value rsr sees is the binary state itself; travel times
                    # (infinite when cut off) stay internal to ndtools' judgement.
                    _, state, _ = evaluate(comps)
                    return state, state, None

                # The independent 50/50 proposal explores topology; it does not replace
                # the BN's component distribution or its hazard-induced dependence.
                run_ref_extraction_by_mcs(
                    sfun=binary_system, probs=torch.full((len(edges), 2), 0.5),
                    row_names=edges, n_state=2, sys_upper_st=1,
                    unk_prob_thres=RSR_UNKNOWN_TARGET, unk_prob_opt="abs", max_rounds=RSR_MAX_ROUNDS,
                    prob_update_every=10, save_every=128, n_sample=4096,
                    sample_batch_size=4096, max_search_loops=2,
                    ref_update_verbose=False, output_dir=str(folder))
                (folder / "model.json").write_text(json.dumps({
                    "rules_sha256": rules_fingerprint(node), "target_add_h": threshold,
                    "origin": origin, "edge_order": edges, "seed": seed,
                    "unknown_target": RSR_UNKNOWN_TARGET, "max_rounds": RSR_MAX_ROUNDS,
                    "proposal": "independent Bernoulli(0.5), for rule search only",
                    "system_value": "binary state",
                    "unclassified": "evaluate with ndtools",
                }, indent=2) + "\n", encoding="utf-8")
                done.append(origin)
    finally:
        random.setstate(python_state)
    return done


def generate_tbn(path: Path | str, *, extract_rules: bool = True, seed: int = 20261022) -> Path:
    """Write a runnable tbnpy model beneath the assembly's example folder, with rules at its threshold."""
    import importlib.metadata

    path = Path(path).resolve()
    compiled = compile_path(path)
    data = tbn_data(compiled)
    compile_tbn(data)                       # every object builds before anything is written
    fingerprint = _tbn_fingerprint(data)
    out = model_dir(path)
    out.mkdir(parents=True, exist_ok=True)
    data["source_sha256"] = fingerprint
    data["versions"] = {n: importlib.metadata.version(n) for n in ("tbnpy", "torch", "ndtools-duco", "rsr-duco")}
    if extract_rules:
        extract_system_rules(compiled, out, seed=seed)
    attach_rules(data, out)                 # the stored rules that hold, extracted now or before
    written = _network_to_file(data, out)
    (out / "model.json").write_text(json.dumps(written, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    script = _GENERATED_MODEL.replace("python -m src build toy-ca/demo", f"python -m src build {compiled.assembly['id']}")
    (out / "model.py").write_text(script, encoding="utf-8")
    return out


def _network_to_file(data: dict[str, Any], directory: Path) -> dict[str, Any]:
    """The spec as written: the network every system node reads, moved to NETWORK_FILE once."""
    systems = system_nodes(data)
    networks = {json.dumps(node["kwargs"]["network"], sort_keys=True) for node in systems}
    if len(networks) != 1:
        return data
    (directory / NETWORK_FILE).write_text(json.dumps(systems[0]["kwargs"]["network"], indent=2) + "\n",
                                          encoding="utf-8")
    written = copy.deepcopy(data)
    for node in system_nodes(written):
        node["kwargs"]["network"] = NETWORK_FILE
    return written


def _network_inline(data: dict[str, Any], directory: Path) -> dict[str, Any]:
    """The spec with each system node's network file read back in: the form its fingerprint is of."""
    inline = copy.deepcopy(data)
    for node in system_nodes(inline):
        if isinstance(node["kwargs"]["network"], str):
            node["kwargs"]["network"] = json.loads((directory / node["kwargs"]["network"]).read_text(encoding="utf-8"))
    return inline


def load_generated(directory: Path | str, device: str = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    """A generated model's (varis, probs), after checking that model.json and its stored rules still
    belong together. (tbnpy.compile.compile_file loads the same file without the checks.)"""
    directory = Path(directory)
    data = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    inline = _network_inline(data, directory)
    if "source_sha256" in data and _tbn_fingerprint(inline) != data["source_sha256"]:
        raise ValueError("Data belong to a different generated model; regenerate")
    for node in system_nodes(inline):
        if node["kwargs"].get("rules"):
            _check_reference(directory, node, node["kwargs"]["rules"])
    return compile_tbn(data, device=device, directory=directory)


_GENERATED_MODEL = '''"""Generated tbnpy model. Regenerate with: python -m src build toy-ca/demo.

model.json is a tbnpy spec: each probability object is a custom node naming its class (in
this repository's src/builder/kernels), its childs and parents, and its constructor's
arguments, and tbnpy.compile builds every object from it. The system objects read the
network from system-network.json and their stored rules from rsr/, beside this file.

Run from any directory: python path/to/model.py --samples 2000 --device cpu
Or import define_model(device="cpu") for tbnpy's (varis, probs).
"""
from pathlib import Path
import argparse
import json
import sys

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "src/builder/bn_builder.py").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))        # so tbnpy can import the kernel classes the spec names

from tbnpy.compile import compile_file


def define_model(device="cpu"):
    """tbnpy's variables and probability objects, compiled from model.json."""
    _, varis, probs = compile_file(BASE / "model.json", device=device)
    return varis, probs


if __name__ == "__main__":
    import torch
    from tbnpy import inference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    varis, probs = define_model(device=args.device)
    sampled = inference.sample(probs, list(probs), n_sample=args.samples)
    summary = {}
    for prob in sampled.values():
        for j, var in enumerate(prob.childs):
            x = prob.Cs[:, j]
            if isinstance(var.values, list):
                summary[var.name] = {"P": {str(s): (x == k).double().mean().item()
                                          for k, s in enumerate(var.values)}}
            else:
                finite = x[torch.isfinite(x)]
                summary[var.name] = {"mean_finite": finite.mean().item() if len(finite) else None,
                                     "P_inf": torch.isinf(x).double().mean().item()}
    result = {"samples": args.samples, "seed": args.seed, "device": args.device, "variables": summary}
    output = BASE / "sample_summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\\n", encoding="utf-8")
    print(f"Sampled {len(varis)} variables in {len(probs)} probability objects; wrote {output}")
'''

"""Load the schema and the instances, and check they are consistent.

    schema/                     what can exist
      meta/*.schema.json        the JSON Schema every file below observes
      quantities.json           types that flow between functions
      entities.json             plates: tables of rows, with foreign keys
      functions/<id>.json       typed signatures, one file each
      models/<id>.json          parts of a function chosen on their own (a hazard field's correlation)
    instances/                  what does exist, filed by the role of the function it fills
      <role>/*.json             parameter sets: GMMs under hazard/, curves under fragility/, ...
    examples/                   networks the tools are shown and tested on
      <n>/network.json          one network's rows
      <n>/*.assembly.json       which function and parameters describe which rows

Schema validation catches a malformed file. The cross-reference checks here
catch what a schema cannot: a port naming a quantity nobody defined, a `via`
that follows a foreign key the plate does not have, parameter values that do not
fit their function, a capability the implementation does not provide. Every
problem is collected and reported together, so a contributor fixes a file in one
pass rather than one error at a time.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "schema"
META = SCHEMA / "meta"
INSTANCES = ROOT / "instances"
NETWORKS = ROOT / "examples"

# The method a kernel must have for each capability its function declares.
CAPABILITY_METHOD = {
    "SampleForward": "sample",
    "EvaluateLikelihood": "log_prob",
    "EvaluateDensity": "distribution",
    "BatchClassify": "batch_classify",
    "Gradient": "gradient",
    "NormalisingConstant": "normalising_constant",
}


class RegistryError(ValueError):
    """One or more problems, all listed."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__(f"{len(self.problems)} problem(s):\n  " + "\n  ".join(self.problems))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


@cache
def _validator(schema_file: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((META / schema_file).read_text(encoding="utf-8")))


def read_json(path: Path, schema_file: str, problems: list[str]) -> dict[str, Any] | None:
    """The file's content if it observes its schema; otherwise None, with problems recorded."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"{rel(path)}: not valid JSON: {exc}")
        return None
    errors = sorted(_validator(schema_file).iter_errors(data), key=lambda e: list(e.absolute_path))
    for error in errors:
        where = "/".join(str(p) for p in error.absolute_path) or "(root)"
        problems.append(f"{rel(path)}: {where}: {error.message}")
    return None if errors else data


# ---- schema and parameter sets --------------------------------------------------


@dataclass
class Library:
    quantities: dict[str, dict[str, Any]]
    entities: dict[str, dict[str, Any]]
    functions: dict[str, dict[str, Any]]
    parameter_sets: dict[str, dict[str, Any]]      # id -> set, plus "function" (or "model") and "file"
    implementations: dict[str, Any] = field(default_factory=dict)   # function id -> probability class
    models: dict[str, dict[str, Any]] = field(default_factory=dict)  # schema/models: id -> model

    def kernel(self, function_id: str) -> Any:
        return self.implementations[function_id]


def resolve_implementation(spec: str) -> type:
    """"module:Class" -> the tbnpy probability class a function's `implementation` names."""
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


_TEMPLATE = re.compile(r"^\{([a-z][a-z0-9_]*)\}$")


def quantity_template(quantity: str) -> str | None:
    """The parameter a port's quantity is taken from, for a port declared as "{imt}"; else None."""
    match = _TEMPLATE.match(quantity)
    return match.group(1) if match else None


def port_choices(fn: dict[str, Any], port: dict[str, Any]) -> list[str]:
    """The quantities a port can carry: its own, or every value its template's parameter may take."""
    param = quantity_template(port["quantity"])
    return [port["quantity"]] if param is None else list(fn["instance_schema"]["properties"][param]["enum"])


def resolve_quantities(fn: dict[str, Any], values: dict[str, Any] | None) -> dict[str, str]:
    """Port name -> quantity once the parameters are known: "{imt}" becomes, say, "SA(1.0)"."""
    out = {}
    for port in fn["inputs"] + fn["outputs"]:
        param = quantity_template(port["quantity"])
        out[port["name"]] = port["quantity"] if param is None or values is None else values.get(param, port["quantity"])
    return out


def via_kind(via: str) -> str:
    """'self', 'all', 'fk' (follow a key on this row) or 'reverse' (rows keyed to this one)."""
    if via == "self":
        return "self"
    if via == "*":
        return "all"
    return "reverse" if "." in via else "fk"


def _check_function(fn: dict[str, Any], lib: Library, where: str, problems: list[str]) -> None:
    per = fn["per"]
    if per not in lib.entities:
        problems.append(f"{where}: per {per!r} is not an entity type")
        return
    per_attrs = lib.entities[per]["attributes"]

    names = [p["name"] for p in fn["inputs"] + fn["outputs"]]
    for dup in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"{where}: port name {dup!r} is used twice")

    def check_quantity(port: dict[str, Any], label: str) -> None:
        param = quantity_template(port["quantity"])
        if param is None:
            if port["quantity"] not in lib.quantities:
                problems.append(f"{label} has unknown quantity {port['quantity']!r}")
            return
        # A templated port is only as safe as its parameter: required, and an enum of real quantities,
        # so that every parameter set that validates resolves to a type the wiring can check.
        params = fn.get("instance_schema", {})
        prop = params.get("properties", {}).get(param)
        if prop is None or param not in params.get("required", []):
            problems.append(f"{label}: quantity {{{param}}} needs a required parameter {param!r}")
        elif not prop.get("enum") or not set(prop["enum"]) <= lib.quantities.keys():
            problems.append(f"{label}: parameter {param!r} must be an enum of quantity ids")

    for port in fn["outputs"]:
        check_quantity(port, f"{where}: output {port['name']!r}")

    for port in fn["inputs"]:
        label = f"{where}: input {port['name']!r}"
        check_quantity(port, label)
        over, via = port["over"], port["via"]
        if over not in lib.entities:
            problems.append(f"{label}: over {over!r} is not an entity type")
            continue
        kind = via_kind(via)
        if kind == "self" and over != per:
            problems.append(f"{label}: via self needs over == per ({per}), got {over}")
        elif kind == "fk" and per_attrs.get(via, {}).get("ref") != over:
            problems.append(f"{label}: via {via!r} needs {per}.{via} to be a foreign key to {over}")
        elif kind == "reverse":
            entity, _, attr = via.partition(".")
            if entity != over:
                problems.append(f"{label}: via {via!r} must start with the over entity {over}")
            elif lib.entities[over]["attributes"].get(attr, {}).get("ref") != per:
                problems.append(f"{label}: via {via!r} needs {over}.{attr} to be a foreign key to {per}")

    for attr in fn.get("attributes", []):
        if attr not in per_attrs:
            problems.append(f"{where}: reads attribute {attr!r}, which {per} does not define")
    for table in fn.get("tables", []):
        if table not in lib.entities:
            problems.append(f"{where}: reads table {table!r}, which is not an entity type")

    if "instance_schema" in fn:
        try:
            Draft202012Validator.check_schema(fn["instance_schema"])
        except SchemaError as exc:
            problems.append(f"{where}: instance_schema is not a valid JSON Schema: {exc.message}")

    try:
        kernel = resolve_implementation(fn["implementation"])
    except Exception as exc:  # noqa: BLE001 - any import failure is a registry problem
        problems.append(f"{where}: implementation {fn['implementation']!r} does not load: {exc}")
        return
    for capability in fn["capabilities"]:
        method = CAPABILITY_METHOD[capability]
        if not callable(getattr(kernel, method, None)):
            problems.append(f"{where}: OVERCLAIM {capability}: {fn['implementation']} has no {method}()")
    lib.implementations[fn["id"]] = kernel


def load_library(schema: Path = SCHEMA, instances: Path = INSTANCES) -> Library:
    problems: list[str] = []
    lib = Library({}, {}, {}, {})

    for key, store, schema_file in (("quantities", lib.quantities, "quantities.schema.json"),
                                    ("entities", lib.entities, "entities.schema.json")):
        data = read_json(schema / f"{key}.json", schema_file, problems)
        for record in (data or {}).get(key, []):
            if record["id"] in store:
                problems.append(f"{key}.json: duplicate id {record['id']!r}")
            store[record["id"]] = record

    for name, entity in lib.entities.items():
        for attr, spec in entity["attributes"].items():
            if "ref" in spec and spec["ref"] not in lib.entities:
                problems.append(f"entities.json: {name}.{attr} refers to unknown entity {spec['ref']!r}")

    for path in sorted((schema / "functions").glob("*.json")):
        fn = read_json(path, "function.schema.json", problems)
        if fn is None:
            continue
        if path.stem != fn["id"]:
            problems.append(f"{rel(path)}: file must be named {fn['id']}.json")
        if fn["id"] in lib.functions:
            problems.append(f"{rel(path)}: duplicate function id {fn['id']!r}")
        lib.functions[fn["id"]] = fn
    for path in sorted((schema / "models").glob("*.json")):
        model = read_json(path, "model.schema.json", problems)
        if model is None:
            continue
        if path.stem != model["id"]:
            problems.append(f"{rel(path)}: file must be named {model['id']}.json")
        try:
            Draft202012Validator.check_schema(model["instance_schema"])
        except SchemaError as exc:
            problems.append(f"{rel(path)}: instance_schema is not a valid JSON Schema: {exc.message}")
            continue
        lib.models[model["id"]] = model
    for fn in lib.functions.values():
        _check_function(fn, lib, f"functions/{fn['id']}.json", problems)
        if "correlation_model" in fn and fn["correlation_model"] not in lib.models:
            problems.append(f"functions/{fn['id']}.json: correlation_model {fn['correlation_model']!r} is not in "
                            "schema/models/")

    for path in sorted(instances.glob("*/*.json")):
        data = read_json(path, "parameter-sets.schema.json", problems)
        if data is None:
            continue
        # The sets parameterise a function, or a model a function uses (its correlation).
        owner = lib.functions.get(data["function"]) if "function" in data else lib.models.get(data["model"])
        if owner is None:
            kind = "function" if "function" in data else "model"
            problems.append(f"{rel(path)}: {kind} {data[kind]!r} is not in the schema")
            continue
        if "instance_schema" not in owner:
            problems.append(f"{rel(path)}: {owner['id']} takes no parameters")
            continue
        if path.parent.name != owner["role"]:
            problems.append(f"{rel(path)}: {owner['id']} has role {owner['role']!r}, so its sets belong in "
                            f"{rel(instances / owner['role'])}/")
        validator = Draft202012Validator(owner["instance_schema"])
        for item in data["sets"]:
            if item["id"] in lib.parameter_sets:
                problems.append(f"{rel(path)}: duplicate parameter-set id {item['id']!r}")
            for error in validator.iter_errors(item["values"]):
                at = "/".join(str(p) for p in error.absolute_path) or "(values)"
                problems.append(f"{rel(path)}: {item['id']}: {at}: {error.message}")
            lib.parameter_sets[item["id"]] = {
                **item, "function": data.get("function"), "model": data.get("model"), "file": rel(path),
                "source": {**data["source"], **item.get("source", {})},
            }

    if problems:
        raise RegistryError(problems)
    return lib


# ---- networks -------------------------------------------------------------------


@dataclass
class Network:
    id: str
    title: str
    assumptions: dict[str, str]
    source: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]
    entity_of: dict[str, str]                 # row id -> entity type
    path: Path

    def rows(self, entity: str) -> list[dict[str, Any]]:
        return self.tables.get(entity, [])

    def row(self, row_id: str) -> dict[str, Any]:
        entity = self.entity_of[row_id]
        return next(r for r in self.tables[entity] if r["id"] == row_id)


_TYPES = {
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
}


def check_tables(tables: dict[str, list[dict[str, Any]]], lib: Library, where: str
                 ) -> tuple[dict[str, str], list[str]]:
    """Row id -> entity type, and every problem with the rows' ids, attributes and keys."""
    problems: list[str] = []
    entity_of: dict[str, str] = {}
    for entity, rows in tables.items():
        if entity not in lib.entities:
            problems.append(f"{where}: table {entity!r} is not an entity type")
            continue
        if lib.entities[entity].get("singleton") and len(rows) > 1:
            problems.append(f"{where}: {entity} is a singleton but has {len(rows)} rows")
        for row in rows:
            if row["id"] in entity_of:
                problems.append(f"{where}: row id {row['id']!r} is used twice (ids are unique across the network)")
            entity_of[row["id"]] = entity

    for entity, rows in tables.items():
        attrs = lib.entities.get(entity, {}).get("attributes", {})
        for row in rows:
            at = f"{where}: {entity} {row['id']!r}"
            for name in row.keys() - {"id"} - attrs.keys():
                problems.append(f"{at}: unknown attribute {name!r}")
            for name, spec in attrs.items():
                if name not in row or row[name] is None:
                    if spec.get("required", True):
                        problems.append(f"{at}: missing required attribute {name!r}")
                    continue
                value = row[name]
                if "ref" in spec:
                    if entity_of.get(value) != spec["ref"]:
                        problems.append(f"{at}: {name} = {value!r} is not a {spec['ref']} row")
                elif not _TYPES[spec["type"]](value):
                    problems.append(f"{at}: {name} = {value!r} is not a {spec['type']}")
                elif "minimum" in spec and value < spec["minimum"]:
                    problems.append(f"{at}: {name} = {value!r} is below its minimum, {spec['minimum']}")
    return entity_of, problems


def fill_defaults(tables: dict[str, list[dict[str, Any]]], lib: Library) -> None:
    """Give every row the default of each attribute it leaves out, where the schema names one."""
    for entity, rows in tables.items():
        for name, spec in lib.entities.get(entity, {}).get("attributes", {}).items():
            if "default" in spec:
                for row in rows:
                    if row.get(name) is None:
                        row[name] = spec["default"]


def load_network(network_id: str, lib: Library, root: Path = NETWORKS) -> Network:
    problems: list[str] = []
    path = root / network_id / "network.json"
    if not path.is_file():
        raise RegistryError([f"no network {network_id!r} (expected {rel(path)})"])
    data = read_json(path, "network.schema.json", problems)
    if data is None:
        raise RegistryError(problems)
    if data["network"] != network_id:
        problems.append(f"{rel(path)}: network is {data['network']!r}, but the folder is {network_id!r}")
    fill_defaults(data["tables"], lib)
    entity_of, row_problems = check_tables(data["tables"], lib, rel(path))
    problems += row_problems
    if problems:
        raise RegistryError(problems)
    return Network(data["network"], data["title"], data["assumptions"], data["source"],
                   data["tables"], entity_of, path)


def assembly_paths(root: Path = NETWORKS) -> list[Path]:
    return sorted(root.glob("*/*.assembly.json"))


def assembly_path(assembly_id: str, root: Path = NETWORKS) -> Path:
    """``ema-highway/gmm`` -> examples/ema-highway/gmm.assembly.json"""
    network, _, name = assembly_id.partition("/")
    path = root / network / f"{name}.assembly.json"
    if not name or not path.is_file():
        known = ", ".join(assembly_id_of(p) for p in assembly_paths(root))
        raise RegistryError([f"no assembly {assembly_id!r}; known: {known}"])
    return path


def assembly_id_of(path: Path) -> str:
    return f"{path.parent.name}/{path.name.removesuffix('.assembly.json')}"


def load_assembly(path: Path | str) -> dict[str, Any]:
    problems: list[str] = []
    path = Path(path)
    data = read_json(path, "assembly.schema.json", problems)
    if data is not None and path.parent.name != data["network"]:
        problems.append(f"{rel(path)}: network is {data['network']!r}, but the file is in {path.parent.name!r}")
    if data is not None and data["id"] != assembly_id_of(path):
        problems.append(f"{rel(path)}: id must be {assembly_id_of(path)}")
    if problems:
        raise RegistryError(problems)
    return data

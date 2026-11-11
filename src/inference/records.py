"""Every query and question asked, with what answered it, so any answer can be inspected or reproduced.

    records/queries.jsonl          one JSON record per line, appended, never rewritten
    records/samples/<bank>.npz     a scenario's sample bank: every variable of the whole BN, when they
                                   fit under SAMPLE_LIMIT_BYTES, with <bank>.json (its notes and rules)

A bank is named by the scenario's model fingerprint, the sample count and the
seed, so every question on one scenario reads the same draws.

A ``query`` record holds the query as computed (with its seed and sample count),
the statement, each probability with its interval, the time taken, what was
computed and with which parameter sets, how the system states' rsr rules served,
and the model's fingerprint, the code's commit and the libraries' versions. The
query, seed, fingerprint and commit reproduce the answer exactly
(``python -m src replay <id>``), so samples too large to keep are not lost.

An ``ask`` record holds a question in words, the model that translated it, its
answer, and the ids of the query records it computed.

``QUAKENET_RECORDS`` moves the directory (the tests point it at a scratch folder).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import threading
from collections import Counter
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.builder.load import ROOT

SAMPLE_LIMIT_BYTES = 50 * 2**20          # above this, a query keeps its recipe, not its samples
_LOCK = threading.Lock()                 # the server answers queries on several threads


def directory() -> Path:
    return Path(os.environ.get("QUAKENET_RECORDS") or ROOT / "records")


def _append(record: dict[str, Any]) -> None:
    path = directory() / "queries.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with _LOCK, path.open("a", encoding="utf-8") as out:
        out.write(line + "\n")


def new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


@cache
def _versions() -> dict[str, str]:
    import importlib.metadata

    out = {}
    for name in ("tbnpy", "torch", "ndtools-duco", "rsr-duco", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


def _code() -> dict[str, Any]:
    """The commit the code was at, and whether the working tree had changes on top of it."""
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD") or None, "uncommitted_changes": bool(status) if status is not None else None}


def bank_path(fingerprint: str, n: int, seed: int) -> Path:
    return directory() / "samples" / f"{fingerprint[:16]}-n{n}-s{seed}.npz"


def as_stored(samples: Mapping[str, np.ndarray], kinds: Mapping[str, str]) -> dict[str, np.ndarray]:
    """Samples at the precision a bank keeps: discrete states as uint8, the rest as float32. A query answered
    from fresh samples uses these too, so it gives the numbers a later read of the bank gives."""
    return {k: np.asarray(v, np.uint8 if kinds.get(k) == "discrete" else np.float32) for k, v in samples.items()}


def save_bank(path: Path, samples: Mapping[str, np.ndarray], kinds: Mapping[str, str], meta: Mapping[str, Any]) -> None:
    discrete = [k for k in samples if kinds.get(k) == "discrete"]
    continuous = [k for k in samples if kinds.get(k) != "discrete"]
    n = len(next(iter(samples.values())))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, discrete_names=np.array(discrete), continuous_names=np.array(continuous),
        discrete=np.stack([samples[k] for k in discrete], 1) if discrete else np.empty((n, 0), np.uint8),
        continuous=np.stack([samples[k] for k in continuous], 1) if continuous else np.empty((n, 0), np.float32))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def load_bank(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path) as data:
        out = {str(k): data["discrete"][:, i] for i, k in enumerate(data["discrete_names"])}
        out.update({str(k): data["continuous"][:, i] for i, k in enumerate(data["continuous_names"])})
    return out, json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def load_samples(record: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    """The sample bank a record was answered from, if it was and it is still there."""
    bank = (record.get("samples_file") or {}).get("bank")
    if not bank or not (directory() / bank).is_file():
        return None
    return load_bank(directory() / bank)[0]


def log_query(result: dict[str, Any], compiled: Any, bindings: set[int], bank: Mapping[str, Any],
              context: Mapping[str, Any]) -> dict[str, Any]:
    """Append a query record; returns what the result reports about it."""
    from src.builder.bn_builder import _tbn_fingerprint, tbn_data

    record_id = new_id()
    stored = {"bank": bank["path"].relative_to(directory()).as_posix() if bank["path"] else None,
              "reused": bank["reused"], "bytes": bank["bytes"],
              "reason": bank["reason"] or (None if bank["path"] else
                                            "sampled on the fly: replay the record to draw them again")}
    used = [compiled.bindings[i] for i in compiled.order if i in bindings]
    _append({
        "kind": "query", "id": record_id, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **{k: v for k, v in context.items() if v is not None},
        "assembly": result["assembly"], "query": result["query"], "statement": result["statement"],
        "events": result["events"], "targets": result["targets"],
        "samples": result["samples"], "seed": result["seed"], "elapsed_s": result["elapsed_s"],
        "inference": result.get("inference"),
        "computed": result["computed"], "warnings": result["warnings"], "rules": result["rules"],
        "scope": result["scope"], "samples_from": result["samples_from"],
        "provenance": {
            "model_sha256": _tbn_fingerprint({**tbn_data(compiled, bindings),
                              **({"node_interventions": result["query"]["do"]} if result["query"].get("do") else {})}),
            "parameter_sets": {**{b.function["id"]: dict(Counter(s for s in b.parameter_sets if s)) or None for b in used},
                               **{b.function["correlation_model"]: b.correlation for b in used if b.correlation}},
            "code": _code(), "versions": _versions(),
        },
        "samples_file": stored,
    })
    return {"id": record_id, "samples_file": stored}


def log_analysis(task, result, runs, context):
    _append({"kind": "analysis", "id": new_id(), "time": datetime.now(timezone.utc).isoformat(),
             "context": dict(context), "task": task, "result": result,
             "queries": [r.get("result", {}).get("record", {}).get("id") for r in runs]})


def log_ask(question: str, answer: str, llm: Mapping[str, Any], assembly: str, runs: list[dict[str, Any]], *,
            clarification=None, history=()) -> str:
    """Append an ask record: the question, its translation(s), the answer, and the queries behind it."""
    record_id = new_id()
    _append({
        "kind": "ask", "id": record_id, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "assembly": assembly, "question": question, "llm": dict(llm), "answer": answer,
        "clarification": clarification,
        "history": list(history),
        "translations": [{"statement": r["statement"], "query": r["query"],
                          "record": (r.get("result") or {}).get("record", {}).get("id"),
                          "refused": r.get("error")} for r in runs],
    })
    return record_id


def read(kind: str | None = None) -> list[dict[str, Any]]:
    path = directory() / "queries.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if kind is None or r["kind"] == kind]


def find(record_id: str) -> dict[str, Any]:
    for record in read():
        if record["id"] == record_id:
            return record
    raise KeyError(f"no record {record_id!r} in {directory() / 'queries.jsonl'}")


def digest(result: Mapping[str, Any]) -> str:
    """The numbers of a result, hashed, so a replay can say whether it reproduced them exactly."""
    numbers = {"events": [(e["node"], e["op"], str(e["value"]), e["p"]) for e in result["events"]],
               "targets": [(t["node"], json.dumps(t["summary"], sort_keys=True)) for t in result["targets"]]}
    return hashlib.sha256(json.dumps(numbers, sort_keys=True).encode()).hexdigest()

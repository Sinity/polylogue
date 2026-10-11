"""Private, staged native population inventory and receipt-based projection.

No transcript is decoded here. Source units are the benchmark sampler's units,
not asserted sessions or revisions. Normalized counts come only from supplied
production component receipts; filesystem bytes never stand in for those counts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import sqlite3
import stat
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any, cast

from devtools.fresh_build_bench.corpus import SampleSource, _unit_key, refuse_inside_checkout

FORMAT = "polylogue.denominator-stage.v1"
METRICS = ("sessions", "messages", "blocks", "events", "revisions", "jobs")
NORMALIZED_METRICS = ("sessions", "messages", "blocks", "events")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    refuse_inside_checkout(path, "private denominator input")
    with path.open(encoding="utf-8") as stream:
        return cast(dict[str, Any], json.load(stream))


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path.absolute()), "sha256": _digest(path)}


def _bound(ref: dict[str, str]) -> Path:
    path = Path(ref["path"])
    refuse_inside_checkout(path, "private denominator reference")
    if not path.is_absolute() or _digest(path) != ref["sha256"]:
        raise ValueError("immutable reference changed")
    return path


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _chain(path: Path) -> list[Path]:
    stages, seen = [], set()
    while True:
        refuse_inside_checkout(path, "private denominator stage")
        if str(path.absolute()) in seen:
            raise ValueError("stage reference cycle")
        seen.add(str(path.absolute()))
        head = next(_rows(path), {})
        if head.get("format") != FORMAT or head.get("kind") != "header":
            raise ValueError("not a denominator stage")
        stages.append(path)
        if head["previous"] is None:
            return list(reversed(stages))
        path = _bound(head["previous"])


def _plan(path: Path) -> dict[str, Any]:
    plan = _read(path)
    if plan.get("format") != "polylogue.denominator-plan.v1":
        raise ValueError("unsupported inventory plan")
    frontier = plan["frontier"]
    if not frontier.get("id") or not frontier.get("basis"):
        raise ValueError("declare frontier identity and its observation basis")
    roots = plan["roots"]
    if not roots or len({root["id"] for root in roots}) != len(roots):
        raise ValueError("declare uniquely named roots")
    strata: dict[str, tuple[str, str]] = {}
    for root in roots:
        if not Path(root["path"]).is_absolute():
            raise ValueError("source roots must be absolute")
        refuse_inside_checkout(Path(root["path"]), "private source root")
        if root["population_kind"] not in {"normal", "exceptional-hooks"} or not root["shape"]:
            raise ValueError("declare population kind and workload shape")
        shape = (root["shape"], root["population_kind"])
        if root["stratum"] in strata and strata[root["stratum"]] != shape:
            raise ValueError("one stratum must have one shape and population kind")
        strata[root["stratum"]] = shape
    return plan


def _source(root: dict[str, Any]) -> SampleSource:
    origin = root["origin"]
    return SampleSource(
        origin,
        Path(root["path"]),
        "",
        origin in {"claude-code", "gemini-cli"},
        unit_depth=1 if origin == "gemini-cli" else 2,
    )


def _coordinates(origin: str, path: Path) -> tuple[str, ...]:
    from polylogue.sources.retained_assembly import claude_code_sidecar_coordinates, codex_sidecar_coordinates

    resolver = {"claude-code": claude_code_sidecar_coordinates, "codex": codex_sidecar_coordinates}.get(origin)
    return (resolver(str(path)) or ()) if resolver else ()


def _observation(path: Path) -> dict[str, Any]:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return {"state": "absent"}
    except OSError:
        return {"state": "stat-fault"}
    if not stat.S_ISREG(observed.st_mode):
        return {"state": "nonregular"}
    return {
        "state": "present",
        "bytes": observed.st_size,
        "stamp": [observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns],
    }


def _physical_lines(path: Path, budget: int, expected: dict[str, Any]) -> dict[str, Any]:
    """Count complete nonblank physical lines; this is not JSON validation."""
    before = _observation(path)
    digest = hashlib.sha256()
    consumed, lines, nonblank = 0, 0, False
    complete = False
    try:
        with path.open("rb") as stream:
            while consumed < budget:
                chunk = stream.read(min(65536, budget - consumed))
                if not chunk:
                    complete = True
                    break
                consumed += len(chunk)
                digest.update(chunk)
                for part in chunk.splitlines(keepends=True):
                    nonblank |= bool(part.strip())
                    if part.endswith(b"\n"):
                        lines += int(nonblank)
                        nonblank = False
            if consumed == before.get("bytes"):
                complete = True
            if complete:
                lines += int(nonblank)
    except OSError:
        return {"bytes_read": consumed, "state": "read-fault", "complete": False}
    current = _observation(path)
    changed = current != before or before != expected
    return {
        "bytes_read": consumed,
        "state": "changed" if changed else "observed",
        "complete": complete and not changed,
        "nonblank_physical_lines": lines if not changed else None,
        "sha256": digest.hexdigest() if complete and not changed else None,
        "prefix_sha256": digest.hexdigest(),
        "not_normalized_messages": True,
    }


def inventory(
    plan_path: Path, output: Path, *, members: int, previous: Path | None = None, structural_bytes: int = 0
) -> dict[str, Any]:
    """Observe at most ``members`` admitted files; continuation never drops input.

    Parent directory listings and link/fault handling remain native. The budget
    bounds emitted members, not listing cost. Resume revisits ancestors; coverage
    is observational, not an atomic snapshot or a promise of unchanged live roots.
    """
    from polylogue.sources.live.discovery import _source_path_steps
    from polylogue.sources.live.watcher import WatchSource
    from polylogue.sources.walk_faults import WalkRefusedError

    if members <= 0:
        raise ValueError("stage member budget must be positive")
    if structural_bytes < 0:
        raise ValueError("structural read budget must be nonnegative")
    refuse_inside_checkout(output, "private denominator output")
    plan = _plan(plan_path)
    if any(output.resolve().is_relative_to(Path(root["path"]).resolve()) for root in plan["roots"]):
        raise ValueError("private output must not mutate a source root")
    plan_ref = _ref(plan_path)
    cursor: dict[str, Any] = {"root": 0, "after": None}
    prior_observations: dict[str, dict[str, Any]] = {}
    if previous is not None:
        last: dict[str, Any] = {}
        for stage in _chain(previous):
            last = {}
            for row in _rows(stage):
                if row["kind"] == "header" and row["plan"] != plan_ref:
                    raise ValueError("continuation belongs to another plan")
                if row["kind"] in {"member", "dependency"} and (
                    row["kind"] == "dependency" or row["path"] in prior_observations
                ):
                    prior_observations[row["path"]] = row["observation"]
                last = row
            if last.get("kind") != "trailer":
                raise ValueError("incomplete stage cannot be continued")
        if last["complete"]:
            raise ValueError("inventory already completed")
        cursor = last["cursor"]
    sources = tuple(WatchSource(name=root["origin"], root=Path(root["path"])) for root in plan["roots"])
    importlib.import_module("polylogue.sources.retained_assembly")
    faults: Counter[str] = Counter()
    emitted, inspected, changes, read_bytes = 0, 0, 0, 0
    source_dir = Path(__file__).resolve().parents[2]
    identity = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    authority = {
        name: {"path": str(Path(str(module.__file__)).resolve()), "sha256": _digest(Path(str(module.__file__)))}
        for name, module in list(sys.modules.items())
        if name.startswith("polylogue.") and getattr(module, "__file__", None)
    }
    if any(not Path(binding["path"]).is_relative_to(source_dir) for binding in authority.values()):
        raise ValueError("inventory product imports must belong to the collector checkout")

    def disposition(_path: Path, outcome: str, reason: str) -> None:
        if outcome != "accepted":
            faults[f"{outcome}:{reason}"] += 1

    def inspected_entry() -> None:
        nonlocal inspected
        inspected += 1

    with output.open("x", encoding="utf-8") as stream:

        def emit(row: dict[str, Any]) -> None:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

        emit(
            {
                "kind": "header",
                "format": FORMAT,
                "plan": plan_ref,
                "previous": _ref(previous) if previous else None,
                "candidate_git_sha": identity,
                "driver_sha256": _digest(Path(__file__)),
                "frontier": plan["frontier"],
                "runtime": {"path": sys.executable, "version": sys.version},
                "structural_byte_budget": structural_bytes,
                "source_authority": authority,
                "unit_grouping": _ref(Path(inspect.getfile(_unit_key))),
            }
        )
        root_index = cursor["root"]
        after = cursor["after"]
        while root_index < len(sources) and emitted < members:
            root = plan["roots"][root_index]
            source = sources[root_index]
            sample_source = _source(root)
            try:
                for path in _source_path_steps(
                    source, sources, after=after, on_disposition=disposition, on_inspected=inspected_entry
                ):
                    if path is None:
                        continue
                    observed = _observation(path)
                    structural = None
                    if path.suffix in {".jsonl", ".ndjson"} and read_bytes < structural_bytes:
                        structural = _physical_lines(path, structural_bytes - read_bytes, observed)
                        read_bytes += structural["bytes_read"]
                        changes += int(structural["state"] == "changed")
                    unit = root["id"] + ":" + _unit_key(sample_source, path)
                    emit(
                        {
                            "kind": "member",
                            "root_id": root["id"],
                            "path": str(path),
                            "unit_id": unit,
                            "stratum": root["stratum"],
                            "artifact_kind": source.artifact_kind(path),
                            "observation": observed,
                            "structural": structural,
                        }
                    )
                    if str(path) in prior_observations and prior_observations[str(path)] != observed:
                        changes += 1
                    if str(path) in prior_observations:
                        prior_observations[str(path)] = observed
                    for coordinate in _coordinates(root["origin"], path):
                        dependency = Path(coordinate)
                        observation = _observation(dependency)
                        prior = prior_observations.get(coordinate)
                        if prior is not None and prior != observation:
                            changes += 1
                        if prior is None or prior != observation:
                            emit(
                                {
                                    "kind": "dependency",
                                    "path": coordinate,
                                    "stratum": root["stratum"],
                                    "observation": observation,
                                }
                            )
                        prior_observations[coordinate] = observation
                    emitted += 1
                    after = str(path)
                    if emitted >= members:
                        break
                else:
                    root_index += 1
                    after = None
                    continue
                break
            except WalkRefusedError as exc:
                faults["fault:walk_refused"] += len(exc.faults)
                emit({"kind": "refused_root", "root_id": root["id"], "fault_count": len(exc.faults)})
                root_index += 1
                after = None
        summary = {
            "kind": "trailer",
            "complete": root_index == len(sources),
            "cursor": {"root": root_index, "after": after},
            "members": emitted,
            "inspected_entries": inspected,
            "dispositions": dict(faults),
            "changed_observations": changes,
            "structural_bytes_read": read_bytes,
            "limitations": [
                "No content parsing, independent-session or revision census; optional physical reads are reported.",
                "Live-root observations are not an atomic or currentness-certified snapshot.",
                "Faults and parent listings can recur across stages; counts are stage-local.",
            ],
        }
        emit(summary)
    return summary


def _calibration(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    selection = _read(_bound(row["selection"]))
    if any(selection[key] != row[key] for key in ("frontier_id", "unit_id", "stratum", "shape")):
        raise ValueError("calibration selection differs from unit/frontier binding")
    if not selection.get("complete_unit"):
        raise ValueError("calibration requires complete selected-unit coverage")
    _bound(selection["closure"])
    receipts = [_read(_bound(ref)) for ref in row["receipts"]]
    if not receipts:
        raise ValueError("calibration requires actual production normalization")
    inputs = [
        {"input_sha256": receipt["input_sha256"], "recipe_sha256": receipt["recipe_sha256"]} for receipt in receipts
    ]
    selected_inputs = selection["normalization_inputs"]
    if (
        inputs != [{key: member[key] for key in ("input_sha256", "recipe_sha256")} for member in selected_inputs]
        or any(not member.get("member_id") for member in selected_inputs)
        or len({member["member_id"] for member in selected_inputs}) != len(inputs)
        or len({(ref["path"], ref["sha256"]) for ref in row["receipts"]}) != len(receipts)
    ):
        raise ValueError("normalization receipts differ from selected inputs")
    per_input = []
    for receipt in receipts:
        if receipt.get("format") != "polylogue.component-receipt.v1":
            raise ValueError("calibration requires a production component receipt")
        if receipt.get("component") != "retained-preparation":
            raise ValueError("calibration requires the complete preparation normalization owner")
        for key in (
            "candidate",
            "product_tree",
            "runtime",
            "input_sha256",
            "recipe_sha256",
            "imported_product_modules",
        ):
            if not receipt.get(key):
                raise ValueError("calibration lacks candidate/runtime/input/import binding")
        if receipt["candidate"].get("dirty") is not False:
            raise ValueError("calibration must bind a clean candidate")
        output = receipt["output"]
        verdict = output.get("verdict")
        if not isinstance(verdict, dict) or not all(
            key in verdict for key in ("sample_count", "invalid_count", "error_count", "drift_count", "strict_refusal")
        ):
            raise ValueError("calibration requires the original complete validation prerequisite")
        if (
            output.get("status") != "passed"
            or verdict["error_count"]
            or verdict["invalid_count"]
            or verdict["strict_refusal"]
        ):
            raise ValueError("refused normalization cannot calibrate a population")
        normalized = output.get("normalized")
        if normalized is None:
            raise ValueError("schema sample_count is not a normalized workload count")
        if normalized.get("count_basis") != "production-preparation-before-lineage":
            raise ValueError("calibration counts must declare the preparation-work basis")
        preparation = output.get("preparation", {})
        if (
            preparation.get("state") != "complete"
            or preparation.get("complete_success") is not True
            or preparation.get("input_sha256") != receipt["input_sha256"]
            or "accepted_prefix_size" not in preparation
            or preparation["accepted_prefix_size"] is not None
            or "parsed_prefix_size" not in preparation
            or preparation["parsed_prefix_size"] is not None
            or preparation.get("consumed_session_count") != normalized.get("sessions")
            or len(output.get("session_content_hashes", [])) != normalized.get("sessions")
        ):
            raise ValueError("calibration requires complete original-input preparation consumption")
        metrics: dict[str, float] = {}
        for key in NORMALIZED_METRICS:
            value = normalized.get(key)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("normalized metrics must be nonnegative integer counts")
                metrics[key] = float(value)
        per_input.append(metrics)
    # Missing evidence for even one normalized input keeps that unit metric unknown.
    summed = {
        key: sum(values[key] for values in per_input) for key in METRICS if all(key in values for values in per_input)
    }
    return receipts, summed


def project(stage: Path, calibration_path: Path, output: Path) -> dict[str, Any]:
    """Ratio-weighted per-stratum projection; unknown coverage stays unknown."""
    refuse_inside_checkout(output, "private denominator projection")
    # This database is disposable private accounting scratch, never an archive.
    # Units and paths can outnumber memory; the source traversal remains streamed.
    with (
        tempfile.TemporaryDirectory(prefix="denominator-", dir=output.parent) as scratch,
        closing(sqlite3.connect(Path(scratch) / "accounting.sqlite")) as ledger,
    ):
        ledger.execute("CREATE TABLE units(stratum TEXT, unit TEXT PRIMARY KEY)")
        ledger.execute(
            "CREATE TABLE paths(path TEXT PRIMARY KEY, bytes INTEGER, state TEXT, member INTEGER, dependency INTEGER)"
        )
        return _project(stage, calibration_path, output, ledger)


def _project(stage: Path, calibration_path: Path, output: Path, ledger: sqlite3.Connection) -> dict[str, Any]:
    calibration = _read(calibration_path)
    if calibration.get("format") != "polylogue.denominator-calibration.v1":
        raise ValueError("unsupported calibration format")
    design = calibration["sampling_design"]
    if design["kind"] not in {"probability", "targeted-model"} or not design["mechanism"]:
        raise ValueError("declare sample selection mechanism")
    stages = _chain(stage)
    first_header = next(_rows(stages[0]))
    plan = _plan(_bound(first_header["plan"]))
    roots = {root["id"]: root for root in plan["roots"]}
    dispositions: Counter[str] = Counter()
    complete, changes = False, 0
    structural_coverage: Counter[str] = Counter()
    artifact_kinds: Counter[str] = Counter()
    for path in stages:
        last_kind = None
        for row in _rows(path):
            if last_kind == "trailer":
                raise ValueError("inventory records follow the terminal trailer")
            last_kind = row["kind"]
            if row["kind"] == "header" and any(
                row[key] != first_header[key] for key in ("plan", "candidate_git_sha", "driver_sha256", "unit_grouping")
            ):
                raise ValueError("inventory stages differ in plan or source authority")
            if row["kind"] == "header" and any(
                row["source_authority"].get(name) != binding
                for name, binding in first_header["source_authority"].items()
            ):
                raise ValueError("inventory source import binding changed")
            if row["kind"] == "member":
                artifact_kinds[row["artifact_kind"]] += 1
                ledger.execute("INSERT OR IGNORE INTO units VALUES (?, ?)", (row["stratum"], row["unit_id"]))
                structural = row.get("structural")
                structural_coverage["members_observed"] += 1
                if structural is not None:
                    structural_coverage["bytes_read"] += structural["bytes_read"]
                    structural_coverage["complete_members"] += int(structural["complete"])
                    structural_coverage["partial_or_faulted_members"] += int(not structural["complete"])
                    structural_coverage["nonblank_physical_lines"] += structural.get("nonblank_physical_lines") or 0
            if row["kind"] in {"member", "dependency"}:
                observed = row["observation"]
                ledger.execute(
                    """INSERT INTO paths VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET bytes=excluded.bytes, state=excluded.state,
                    member=MAX(paths.member, excluded.member), dependency=MAX(paths.dependency, excluded.dependency)""",
                    (
                        row["path"],
                        observed.get("bytes", 0),
                        observed["state"],
                        int(row["kind"] == "member"),
                        int(row["kind"] == "dependency"),
                    ),
                )
            if row["kind"] == "trailer":
                complete = row["complete"]
                changes += row["changed_observations"]
                dispositions.update(row["dispositions"])
        if last_kind != "trailer":
            raise ValueError("inventory stage lacks its terminal trailer")
    bindings: list[dict[str, Any]] = []
    samples: dict[str, list[tuple[float, dict[str, float]]]] = defaultdict(list)
    held_out: dict[str, list[dict[str, float]]] = defaultdict(list)
    seen_units: set[str] = set()
    for row in calibration["samples"]:
        stratum = row["stratum"]
        declared_shape = next((root["shape"] for root in roots.values() if root["stratum"] == stratum), None)
        if row["shape"] != declared_shape:
            raise ValueError("calibration shape differs from declared population stratum")
        if row["frontier_id"] != plan["frontier"]["id"]:
            raise ValueError("calibration belongs to another population frontier")
        if (
            not ledger.execute("SELECT 1 FROM units WHERE stratum=? AND unit=?", (stratum, row["unit_id"])).fetchone()
            or row["unit_id"] in seen_units
        ):
            raise ValueError("calibration must name distinct observed units in their declared stratum")
        seen_units.add(row["unit_id"])
        weight = row["inclusion_weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("explicit sample weights must be finite and positive")
        probability = row.get("inclusion_probability")
        if design["kind"] == "probability" and (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not 0 < probability <= 1
            or not math.isclose(weight * probability, 1.0)
        ):
            raise ValueError("probability samples need matching explicit inclusion probabilities and weights")
        receipts, metrics = _calibration(row)
        for receipt in receipts:
            binding = {key: receipt[key] for key in ("candidate", "product_tree", "runtime")}
            if bindings and binding != bindings[0]:
                raise ValueError("do not pool different candidate/runtime calibrations")
            bindings.append(binding)
        if row.get("held_out", False):
            held_out[stratum].append(metrics)
        else:
            samples[stratum].append((float(weight), metrics))
    results = {}
    member_states = dict(ledger.execute("SELECT state,COUNT(*) FROM paths WHERE member=1 GROUP BY state"))
    dependency_states = dict(ledger.execute("SELECT state,COUNT(*) FROM paths WHERE dependency=1 GROUP BY state"))
    covered = (
        complete
        and not changes
        and not any(k.startswith("fault:") for k in dispositions)
        and not any(state != "present" for state in member_states)
        and not any(state not in {"present", "absent"} for state in dependency_states)
    )
    present_members = member_states.get("present", 0)
    byte_distribution = {
        "present_members": present_members,
        "min": ledger.execute("SELECT MIN(bytes) FROM paths WHERE member=1 AND state='present'").fetchone()[0],
        "max": ledger.execute("SELECT MAX(bytes) FROM paths WHERE member=1 AND state='present'").fetchone()[0],
        "quantiles": {
            str(fraction): ledger.execute(
                "SELECT bytes FROM paths WHERE member=1 AND state='present' ORDER BY bytes LIMIT 1 OFFSET ?",
                (int((present_members - 1) * fraction),),
            ).fetchone()[0]
            if present_members
            else None
            for fraction in (0.5, 0.95)
        },
    }
    for stratum in {root["stratum"] for root in roots.values()}:
        root = next(root for root in roots.values() if root["stratum"] == stratum)
        population = ledger.execute("SELECT COUNT(*) FROM units WHERE stratum=?", (stratum,)).fetchone()[0]
        rows = samples[stratum]
        estimates: dict[str, float | None] = {}
        uncertainty: dict[str, Any] = {}
        errors: dict[str, Any] = {}
        unknown: list[str] = []
        for metric in METRICS:
            if not rows or any(metric not in values for _, values in rows):
                estimates[metric] = None
                unknown.append(metric)
                continue
            weight_sum = sum(weight for weight, _ in rows)
            mean = sum(weight * values[metric] for weight, values in rows) / weight_sum
            estimates[metric] = population * mean
            effective_n = weight_sum**2 / sum(weight**2 for weight, _ in rows)
            dispersion = sum(weight * (values[metric] - mean) ** 2 for weight, values in rows) / weight_sum
            uncertainty[metric] = {
                "sample_range": [min(v[metric] for _, v in rows), max(v[metric] for _, v in rows)],
                "observed_shape_scenario_range": [
                    population * min(v[metric] for _, v in rows),
                    population * max(v[metric] for _, v in rows),
                ],
                "effective_n": effective_n,
                "weighted_dispersion": dispersion,
                "confidence_interval": None,
            }
            values = [row[metric] for row in held_out[stratum] if metric in row]
            if values:
                actual = sum(values)
                predicted = mean * len(values)
                errors[metric] = {
                    "held_out_units": len(values),
                    "predicted": predicted,
                    "actual": actual,
                    "relative_error": (predicted - actual) / actual if actual else None,
                }
        results[stratum] = {
            "population_kind": root["population_kind"],
            "shape": root["shape"],
            "observed_source_units": population,
            "calibration_units": len(rows),
            "estimates": estimates,
            "uncertainty": uncertainty,
            "held_out_error": errors,
            "held_out_checked_metrics": sorted(errors) if covered else [],
            "unknown_metrics": unknown,
        }
    result = {
        "format": "polylogue.denominator-projection.v1",
        "inventory": _ref(stage),
        "calibration": _ref(calibration_path),
        "frontier": plan["frontier"],
        "traversal_complete": complete,
        "population_complete": covered,
        "dispositions": dict(dispositions),
        "sampling_design": design,
        "structural_coverage": dict(structural_coverage),
        "changed_observations": changes,
        "strata": results,
        "calibration_binding": bindings[:1],
        "count_basis": "production-preparation-before-lineage",
        "input_member_count": ledger.execute("SELECT COUNT(*) FROM paths WHERE member=1").fetchone()[0],
        "shared_dependency_count": ledger.execute("SELECT COUNT(*) FROM paths WHERE dependency=1").fetchone()[0],
        "shared_dependency_present_files": ledger.execute(
            "SELECT COUNT(*) FROM paths WHERE dependency=1 AND state='present'"
        ).fetchone()[0],
        "member_bytes": ledger.execute("SELECT COALESCE(SUM(bytes),0) FROM paths WHERE member=1").fetchone()[0],
        "additional_shared_bytes": ledger.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM paths WHERE dependency=1 AND member=0"
        ).fetchone()[0],
        "dependency_states": dependency_states,
        "member_states": member_states,
        "artifact_kinds": dict(artifact_kinds),
        "member_byte_distribution": byte_distribution,
        "unknowns": [
            "Source units are not independent sessions or revision chains.",
            "Prepared counts precede archive lineage resolution, deduplication and materialization; final Index rows and downstream amplification remain unknown.",
            "Declared shape is a caller stratum model, not an inferred content classification.",
            "Partial inventory estimates cover observed units only; no unseen-population extrapolation.",
            "No confidence interval without a verified sampling design; dispersion is descriptive.",
            "Unobserved long tails remain unknown; observed-shape ranges are not confidence bounds.",
            "Selection files attest closure and weights; inventory stats do not prove content equality.",
            "Acquisition, CPU, serialized publication, remote completion and settlement are not inferred.",
            "Embeddings excluded; exceptional hooks are a separate population.",
        ],
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("inventory")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--members", type=int, required=True)
    collect.add_argument("--previous", type=Path)
    collect.add_argument("--structural-bytes", type=int, default=0)
    estimate = commands.add_parser("project")
    estimate.add_argument("--inventory", type=Path, required=True)
    estimate.add_argument("--calibration", type=Path, required=True)
    estimate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        inventory(
            args.plan, args.output, members=args.members, previous=args.previous, structural_bytes=args.structural_bytes
        )
    else:
        project(args.inventory, args.calibration, args.output)
    return 0

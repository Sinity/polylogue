#!/usr/bin/env python3
"""Compose evidence-only execution dossiers for test-suite diet clusters.

The selector declares the behavioral responsibility and exact candidate scope.
This command joins disposable evidence already produced by pytest, testmon,
coverage, mutation campaigns, the coupling/fixture inventories, and git.  It
does not decide that a test may be deleted and is never a verification gate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import hashlib
import json
import sqlite3
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DEFAULT_SELECTORS = HERE / "clusters.json"
DEFAULT_OUTPUT = HERE.parent / "dossiers"
TESTMON_DB = ROOT / ".cache/testmon/testmondata"
COVERAGE_JSON = ROOT / ".cache/coverage/coverage.json"
COUPLING_JSON = HERE / "test-coupling-census.json"
INFRA_TSV = HERE / "test-infra-consumers.tsv"
PYTEST_RECEIPT_GLOBS = (
    ".cache/verify/*pytest*.json",
    ".cache/verify/**/*pytest*.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cluster_map(path: Path) -> dict[str, dict[str, Any]]:
    raw = _read_json(path)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError(f"unsupported selector document: {path}")
    clusters = raw.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError(f"clusters must be a list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for item in clusters:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"invalid cluster selector in {path}")
        cluster_id = item["id"]
        if cluster_id in result:
            raise ValueError(f"duplicate cluster id: {cluster_id}")
        result[cluster_id] = item
    return result


def _defined_symbols(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return set()
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
    return symbols


def _route_evidence(cluster: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    gaps: list[str] = []
    for route in cluster["authoritative_routes"]:
        rel = str(route["path"])
        path = ROOT / rel
        expected = [str(item) for item in route.get("symbols", [])]
        if not path.is_file():
            rows.append({"path": rel, "exists": False, "sha256": None, "symbols": {}})
            gaps.append(f"authoritative route missing: {rel}")
            continue
        defined = _defined_symbols(path)
        symbols = {name: name in defined for name in expected}
        for name, found in symbols.items():
            if not found:
                gaps.append(f"symbol not resolved: {rel}:{name}")
        rows.append({"path": rel, "exists": True, "sha256": _sha256(path), "symbols": symbols})
    return rows, gaps


def _scope_fingerprints(cluster: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = ("test_files", "helper_files", "planned_files", "avoid_files")
    rows: list[dict[str, Any]] = []
    for group in groups:
        for rel in cluster.get(group, []):
            path = ROOT / str(rel)
            rows.append(
                {
                    "group": group,
                    "path": str(rel),
                    "exists": path.is_file(),
                    "sha256": _sha256(path) if path.is_file() else None,
                }
            )
    return rows


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _strings(nested)


def _pytest_receipts(cluster: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = tuple(str(item) for item in cluster.get("test_files", []))
    paths: set[Path] = set()
    for pattern in PYTEST_RECEIPT_GLOBS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    receipts: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        nodeids = sorted(
            {text for text in _strings(payload) if "::" in text and (not prefixes or text.startswith(prefixes))}
        )
        if nodeids:
            receipts.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": _sha256(path),
                    "matching_nodeids": nodeids,
                }
            )
    return {"available": bool(receipts), "receipts": receipts}


def _testmon_evidence(cluster: Mapping[str, Any]) -> dict[str, Any]:
    if not TESTMON_DB.is_file():
        return {"available": False, "path": TESTMON_DB.relative_to(ROOT).as_posix(), "tests": []}
    route_paths = {str(route["path"]) for route in cluster["authoritative_routes"]}
    test_prefixes = tuple(str(item) for item in cluster.get("test_files", []))
    query = """
        SELECT te.test_name, te.duration, ff.filename
        FROM test_execution te
        JOIN test_execution_file_fp tefp ON tefp.test_execution_id = te.id
        JOIN file_fp ff ON ff.id = tefp.fingerprint_id
    """
    grouped: dict[str, dict[str, Any]] = {}
    connection = sqlite3.connect(f"file:{TESTMON_DB}?mode=ro", uri=True)
    try:
        for test_name, duration, filename in connection.execute(query):
            test = str(test_name)
            dependency = str(filename)
            if dependency not in route_paths and not test.startswith(test_prefixes):
                continue
            row = grouped.setdefault(test, {"nodeid": test, "duration_s": float(duration or 0), "dependencies": []})
            if dependency in route_paths:
                row["dependencies"].append(dependency)
    except sqlite3.DatabaseError as error:
        return {"available": False, "path": TESTMON_DB.relative_to(ROOT).as_posix(), "error": str(error), "tests": []}
    finally:
        connection.close()
    rows = sorted(grouped.values(), key=lambda item: item["nodeid"])
    for row in rows:
        row["dependencies"] = sorted(set(row["dependencies"]))
    return {
        "available": True,
        "path": TESTMON_DB.relative_to(ROOT).as_posix(),
        "sha256": _sha256(TESTMON_DB),
        "tests": rows,
    }


def _coverage_contexts(cluster: Mapping[str, Any]) -> dict[str, Any]:
    if not COVERAGE_JSON.is_file():
        return {"available": False, "path": COVERAGE_JSON.relative_to(ROOT).as_posix(), "routes": []}
    try:
        payload = _read_json(COVERAGE_JSON)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "available": False,
            "path": COVERAGE_JSON.relative_to(ROOT).as_posix(),
            "error": str(error),
            "routes": [],
        }
    files = payload.get("files", {}) if isinstance(payload, Mapping) else {}
    test_tokens = {Path(str(item)).name for item in cluster.get("test_files", [])}
    routes: list[dict[str, Any]] = []
    for route in cluster["authoritative_routes"]:
        rel = str(route["path"])
        entry = files.get(rel, {}) if isinstance(files, Mapping) else {}
        contexts = entry.get("contexts", {}) if isinstance(entry, Mapping) else {}
        matching: set[str] = set()
        if isinstance(contexts, Mapping):
            for names in contexts.values():
                if isinstance(names, list):
                    matching.update(str(name) for name in names if any(token in str(name) for token in test_tokens))
        routes.append(
            {
                "path": rel,
                "present": bool(entry),
                "summary": entry.get("summary") if isinstance(entry, Mapping) else None,
                "matching_contexts": sorted(matching),
            }
        )
    return {
        "available": True,
        "path": COVERAGE_JSON.relative_to(ROOT).as_posix(),
        "sha256": _sha256(COVERAGE_JSON),
        "routes": routes,
    }


def _coupling_evidence(cluster: Mapping[str, Any]) -> dict[str, Any]:
    if not COUPLING_JSON.is_file():
        return {"available": False, "findings": []}
    payload = _read_json(COUPLING_JSON)
    prefixes = tuple(str(item) for item in (*cluster.get("test_files", []), *cluster.get("helper_files", [])))
    findings = [
        item
        for item in payload.get("findings", [])
        if isinstance(item, Mapping) and str(item.get("path", "")).startswith(prefixes)
    ]
    return {
        "available": True,
        "path": COUPLING_JSON.relative_to(ROOT).as_posix(),
        "sha256": _sha256(COUPLING_JSON),
        "findings": findings,
    }


def _fixture_inventory(cluster: Mapping[str, Any]) -> dict[str, Any]:
    if not INFRA_TSV.is_file():
        return {"available": False, "rows": []}
    helper_modules = {
        str(item).removesuffix(".py").replace("/", ".")
        for item in cluster.get("helper_files", [])
        if str(item).startswith("tests/infra/")
    }
    with INFRA_TSV.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("module") in helper_modules]
    return {
        "available": True,
        "path": INFRA_TSV.relative_to(ROOT).as_posix(),
        "sha256": _sha256(INFRA_TSV),
        "rows": rows,
    }


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _history(cluster: Mapping[str, Any]) -> dict[str, Any]:
    paths = [str(item) for item in cluster.get("history_paths", [])]
    path_log = _git(["log", "--format=%H%x09%cs%x09%s", "-20", "--", *paths]) if paths else ""
    dogfood: dict[str, list[str]] = {}
    for term in cluster.get("dogfood_terms", []):
        output = _git(["log", "--all", "--regexp-ignore-case", f"--grep={term}", "--format=%H%x09%cs%x09%s", "-8"])
        dogfood[str(term)] = output.splitlines() if output else []
    return {
        "paths": paths,
        "commits": path_log.splitlines() if path_log else [],
        "dogfood_witnesses": dogfood,
    }


def _artifact_matches(patterns: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for raw in glob.glob(str(ROOT / pattern), recursive=True):
            path = Path(raw)
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            rows.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": _sha256(path),
                    "size": path.stat().st_size,
                }
            )
    return sorted(rows, key=lambda item: item["path"])


def _prerequisite_gaps(cluster: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    gaps: list[str] = []
    for prerequisite in cluster.get("incoming_prerequisites", []):
        row = dict(prerequisite)
        required = [str(item) for item in prerequisite.get("required_paths_after_merge", [])]
        if required:
            missing = [rel for rel in required if not (ROOT / rel).is_file()]
            row["missing_paths"] = missing
            if missing:
                gaps.append(f"incoming prerequisite not present: {', '.join(missing)}")
        elif prerequisite.get("cluster"):
            gaps.append(f"cluster prerequisite requires coordinator receipt: {prerequisite['cluster']}")
        elif prerequisite.get("bead"):
            gaps.append(f"upstream Bead contract requires merged-source coordinator receipt: {prerequisite['bead']}")
        rows.append(row)
    return rows, gaps


def build_dossier(cluster: Mapping[str, Any], *, selector_path: Path) -> dict[str, Any]:
    routes, gaps = _route_evidence(cluster)
    prerequisites, prerequisite_gaps = _prerequisite_gaps(cluster)
    gaps.extend(prerequisite_gaps)
    sensitivity_artifacts = _artifact_matches(
        pattern for witness in cluster.get("sensitivity_witnesses", []) for pattern in witness.get("evidence_globs", [])
    )
    if not sensitivity_artifacts:
        gaps.append("no realized sensitivity artifact found for the cluster")
    required_fields = (
        "obligations",
        "proposed_survivors",
        "sensitivity_witnesses",
        "deletion_candidates",
        "focused_tests",
    )
    for field in required_fields:
        if not cluster.get(field):
            gaps.append(f"selector field is empty: {field}")
    dossier = {
        "schema_version": 1,
        "cluster": {
            key: value
            for key, value in cluster.items()
            if key not in {"mutation_globs", "history_paths", "dogfood_terms"}
        },
        "readiness": "execution-grade" if not gaps else "prepared-not-execution-grade",
        "readiness_gaps": gaps,
        "source": {
            "selector": selector_path.relative_to(ROOT).as_posix(),
            "selector_sha256": _sha256(selector_path),
            "git_head": _git(["rev-parse", "HEAD"]),
        },
        "route_evidence": routes,
        "scope_fingerprints": _scope_fingerprints(cluster),
        "prerequisites": prerequisites,
        "pytest_receipts": _pytest_receipts(cluster),
        "testmon": _testmon_evidence(cluster),
        "coverage_contexts": _coverage_contexts(cluster),
        "coupling_census": _coupling_evidence(cluster),
        "fixture_inventory": _fixture_inventory(cluster),
        "history": _history(cluster),
        "mutation_artifacts": _artifact_matches(cluster.get("mutation_globs", [])),
        "sensitivity_artifacts": sensitivity_artifacts,
    }
    return dossier


def _bullets(values: Iterable[object]) -> str:
    rows = [f"- {value}" for value in values]
    return "\n".join(rows) if rows else "- none"


def _render_markdown(dossier: Mapping[str, Any]) -> str:
    cluster = dossier["cluster"]
    route_rows = []
    for route in dossier["route_evidence"]:
        resolved = [name for name, found in route["symbols"].items() if found]
        missing = [name for name, found in route["symbols"].items() if not found]
        route_rows.append(
            f"| `{route['path']}` | {'yes' if route['exists'] else 'no'} | "
            f"{', '.join(f'`{item}`' for item in resolved) or '—'} | "
            f"{', '.join(f'`{item}`' for item in missing) or '—'} |"
        )
    history_rows = dossier["history"]["commits"][:12]
    testmon = dossier["testmon"]
    coverage = dossier["coverage_contexts"]
    coupling = dossier["coupling_census"]
    fixtures = dossier["fixture_inventory"]
    prerequisites = dossier["prerequisites"]
    prerequisite_lines = [
        f"- `{item.get('bead') or item.get('cluster') or item.get('worktree')}`: "
        f"{item.get('reason', 'no rationale recorded')}"
        for item in prerequisites
    ]
    lines = [
        "---",
        f"cluster: {cluster['id']}",
        f"readiness: {dossier['readiness']}",
        f"git_head: {dossier['source']['git_head']}",
        "generated_by: census/dossier.py",
        "---",
        "",
        f"# {cluster['title']}",
        "",
        "> Evidence packet, not a deletion verdict or coverage gate.",
        "",
        "## Responsibility",
        "",
        str(cluster["responsibility"]),
        "",
        "## Readiness",
        "",
        f"`{dossier['readiness']}`",
        "",
        _bullets(dossier["readiness_gaps"]),
        "",
        "## Baseline dependencies",
        "",
        *(prerequisite_lines or ["- none"]),
        "",
        "## Authoritative routes",
        "",
        "| Path | Exists | Resolved symbols | Missing symbols |",
        "| --- | --- | --- | --- |",
        *route_rows,
        "",
        "## Independent obligations",
        "",
        _bullets(cluster["obligations"]),
        "",
        "## Proposed survivor tests",
        "",
        _bullets(cluster["proposed_survivors"]),
        "",
        "## Sensitivity witnesses",
        "",
        _bullets(f"{item['kind']}: {item['description']}" for item in cluster["sensitivity_witnesses"]),
        "",
        f"Realized artifacts: `{len(dossier['sensitivity_artifacts'])}`.",
        "",
        "## Candidate scope",
        "",
        f"Tests: {', '.join(f'`{item}`' for item in cluster['test_files']) or '—'}",
        "",
        f"Helpers: {', '.join(f'`{item}`' for item in cluster['helper_files']) or '—'}",
        "",
        f"Planned: {', '.join(f'`{item}`' for item in cluster['planned_files']) or '—'}",
        "",
        f"Avoid: {', '.join(f'`{item}`' for item in cluster['avoid_files']) or '—'}",
        "",
        "## Deletion candidates requiring dominance proof",
        "",
        _bullets(cluster["deletion_candidates"]),
        "",
        "## Evidence inventory",
        "",
        f"- pytest receipts: `{len(dossier['pytest_receipts']['receipts'])}`",
        f"- testmon available/matching tests: `{testmon['available']}` / `{len(testmon['tests'])}`",
        f"- coverage contexts available: `{coverage['available']}`",
        f"- coupling findings: `{len(coupling['findings'])}`",
        f"- fixture inventory rows: `{len(fixtures['rows'])}`",
        f"- mutation artifacts: `{len(dossier['mutation_artifacts'])}`",
        "",
        "## Recent path history",
        "",
        _bullets(f"`{item}`" for item in history_rows),
        "",
        "## Permitted worker checks",
        "",
        "```bash",
        *cluster["focused_tests"],
        "```",
        "",
        "The coordinator must refresh this dossier after the upstream merge, after collecting ",
        "per-test coverage contexts, and after sensitivity execution.",
        "",
    ]
    return "\n".join(lines)


def _write_dossier(cluster: Mapping[str, Any], *, selector_path: Path, output_dir: Path) -> tuple[Path, Path]:
    dossier = build_dossier(cluster, selector_path=selector_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{cluster['id']}.json"
    markdown_path = output_dir / f"{cluster['id']}.md"
    json_path.write_text(json.dumps(dossier, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(dossier), encoding="utf-8")
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selectors", type=Path, default=DEFAULT_SELECTORS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    render = subparsers.add_parser("render")
    render.add_argument("--cluster", action="append", default=[])
    render.add_argument("--all", action="store_true")
    render.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    selector_path = args.selectors.resolve()
    clusters = _cluster_map(selector_path)
    if args.command == "list":
        for cluster_id, cluster in clusters.items():
            print(f"{cluster_id}\t{cluster['title']}")
        return 0

    requested = list(clusters) if args.all else list(args.cluster)
    if not requested:
        parser.error("render requires --cluster ID or --all")
    unknown = [item for item in requested if item not in clusters]
    if unknown:
        parser.error(f"unknown cluster(s): {', '.join(unknown)}")
    for cluster_id in requested:
        json_path, markdown_path = _write_dossier(
            clusters[cluster_id],
            selector_path=selector_path,
            output_dir=args.output_dir.resolve(),
        )
        print(f"{cluster_id}: {json_path} {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Apply the declared GitHub/Beads authority map to the live Beads database.

The GitHub issue mutations are performed through GitHub itself. This command updates the
repo-local Beads authority from ``devtools/data/tracker-authority.json`` without trusting the
possibly stale checked-in JSONL snapshot:

1. export the current live Dolt-backed Beads state;
2. mutate only named rows;
3. give changed rows a fresh ``updated_at`` revision;
4. apply them through ``bd_reimport_guard.py reconcile`` so downgrades and incomparable rows
   are refused and a receipt is written;
5. let the guard re-export ``.beads/issues.jsonl`` from the resulting live state.

Default mode is a dry run. Pass ``--apply`` to mutate Beads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "devtools" / "data" / "tracker-authority.json"
GUARD_PATH = ROOT / "devtools" / "bd_reimport_guard.py"

_MARKER_START = "<!-- tracker-authority:v1 -->"
_MARKER_END = "<!-- /tracker-authority:v1 -->"
_RELATION_LABELS = {
    "gh_mirror": "tracker:gh-mirror",
    "gh_public_parent": "tracker:gh-public-parent",
    "gh_implements": "tracker:gh-implements",
    "gh_supersedes_scope": "tracker:gh-supersedes-scope",
    "internal_only": "tracker:internal-only",
}


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)


def _export_live_rows() -> dict[str, dict[str, Any]]:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as handle:
        export_path = Path(handle.name)
    try:
        _run([sys.executable, str(GUARD_PATH), "export", str(export_path)])
        rows: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(export_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            issue_id = row.get("id")
            if not isinstance(issue_id, str) or not issue_id:
                raise ValueError(f"export line {line_number} has no issue id")
            rows[issue_id] = row
        return rows
    finally:
        export_path.unlink(missing_ok=True)


def _load_manifest() -> dict[str, Any]:
    payload = json.loads(MANIFEST_PATH.read_text())
    if payload.get("version") != 1:
        raise ValueError(f"unsupported manifest version: {payload.get('version')!r}")
    bindings = payload.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("tracker authority manifest has no bindings")
    return payload


def _normalise_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, list):
        raise ValueError(f"unsupported labels shape: {type(value).__name__}")
    labels: list[str] = []
    for item in value:
        if isinstance(item, str):
            labels.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            labels.append(item["name"])
        else:
            raise ValueError(f"unsupported label entry: {item!r}")
    return labels


def _replace_authority_note(existing: Any, block: str) -> str:
    text = existing if isinstance(existing, str) else ""
    while _MARKER_START in text:
        start = text.index(_MARKER_START)
        end = text.find(_MARKER_END, start)
        if end < 0:
            text = text[:start].rstrip()
            break
        text = (text[:start] + text[end + len(_MARKER_END) :]).strip()
    if text:
        return f"{text.rstrip()}\n\n{block}\n"
    return f"{block}\n"


def _authority_block(binding: dict[str, Any]) -> str:
    issue = binding.get("github_issue")
    relation = binding["relation"]
    note = binding.get("note", "")
    lines = [
        _MARKER_START,
        f"Tracker relation: {relation}; public GitHub outcome: #{issue}.",
    ]
    if note:
        lines.append(str(note))
    lines.append(_MARKER_END)
    return "\n".join(lines)


def _apply_binding(row: dict[str, Any], binding: dict[str, Any], *, timestamp: str) -> list[str]:
    changes: list[str] = []
    relation = binding.get("relation")
    if relation not in _RELATION_LABELS:
        raise ValueError(f"unknown tracker relation {relation!r} for {binding.get('bead_id')}")

    labels = _normalise_labels(row.get("labels"))
    relation_labels = set(_RELATION_LABELS.values())
    desired_label = _RELATION_LABELS[relation]
    new_labels = [label for label in labels if label not in relation_labels]
    if desired_label not in new_labels:
        new_labels.append(desired_label)
    new_labels = sorted(dict.fromkeys(new_labels))
    if new_labels != labels:
        row["labels"] = new_labels
        changes.append(f"labels -> add {desired_label}")

    for field in ("external_ref", "title", "description", "acceptance_criteria"):
        if field not in binding:
            continue
        desired = binding[field]
        if row.get(field) != desired:
            row[field] = desired
            changes.append(f"{field} updated")

    block = _authority_block(binding)
    notes = _replace_authority_note(row.get("notes"), block)
    if notes != row.get("notes"):
        row["notes"] = notes
        changes.append("tracker authority note updated")

    if changes:
        row["updated_at"] = timestamp
    return changes


def _write_candidate(rows: dict[str, dict[str, Any]]) -> Path:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    path = Path(handle.name)
    try:
        for issue_id in sorted(rows):
            handle.write(json.dumps(rows[issue_id], sort_keys=True) + "\n")
    finally:
        handle.close()
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="apply the candidate through the monotonic Beads guard")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="authority manifest path")
    args = parser.parse_args(argv)

    global MANIFEST_PATH
    MANIFEST_PATH = args.manifest.resolve()

    manifest = _load_manifest()
    live_rows = _export_live_rows()
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    changed: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    report: list[tuple[str, list[str]]] = []
    for raw_binding in manifest["bindings"]:
        binding = dict(raw_binding)
        bead_id = binding.get("bead_id")
        if not isinstance(bead_id, str) or not bead_id:
            raise ValueError(f"invalid binding without bead_id: {binding!r}")
        original = live_rows.get(bead_id)
        if original is None:
            missing.append(bead_id)
            continue
        candidate = json.loads(json.dumps(original))
        changes = _apply_binding(candidate, binding, timestamp=timestamp)
        report.append((bead_id, changes))
        if changes:
            changed[bead_id] = candidate

    print(f"tracker authority: {len(changed)} changed, {len(report) - len(changed)} already coherent")
    for bead_id, changes in report:
        if changes:
            print(f"  {bead_id}: " + "; ".join(changes))
    if missing:
        print("missing target beads: " + ", ".join(sorted(missing)), file=sys.stderr)
        return 2

    if not changed:
        return 0
    if not args.apply:
        print("dry run only; rerun with --apply to mutate live Beads")
        return 0

    candidate_path = _write_candidate(changed)
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(GUARD_PATH),
                "reconcile",
                str(candidate_path),
                "--source",
                "tracker-authority-v1",
            ],
            cwd=ROOT,
            text=True,
        )
        if result.returncode != 0:
            return result.returncode
    finally:
        candidate_path.unlink(missing_ok=True)

    print("tracker authority applied; .beads/issues.jsonl re-exported by the guard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

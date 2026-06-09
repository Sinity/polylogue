"""Capture and compare archive golden-master envelopes for schema rewrites."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from devtools import repo_root
from polylogue import Polylogue
from polylogue.archive.query.spec import SessionQuerySpec, clamp_query_limit
from polylogue.core.json import JSONDocument, JSONValue, require_json_document
from polylogue.paths import active_index_db_path as default_db_path
from polylogue.paths import archive_root
from polylogue.surfaces.payloads import (
    SessionListRowPayload,
    SessionMessagePayload,
    SessionSearchHitPayload,
    build_search_envelope,
    model_json_document,
)

REPORT_VERSION = 1
DEFAULT_SEARCH_QUERIES: tuple[str, ...] = ("analysis", "error")
DEFAULT_STATS_GROUPS: tuple[str, ...] = ("provider", "day", "month", "year")


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    db_path: Path
    limit: int = 10
    message_limit: int = 8
    search_queries: tuple[str, ...] = DEFAULT_SEARCH_QUERIES
    stats_groups: tuple[str, ...] = DEFAULT_STATS_GROUPS


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _schema_version(db_path: Path) -> int | None:
    if not db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def _stable_mapping(value: Mapping[str, int]) -> dict[str, int]:
    return {key: int(value[key]) for key in sorted(value)}


def _archive_stats_payload(stats: Any) -> JSONDocument:
    recent = [
        model_json_document(SessionListRowPayload.from_session(session), exclude_none=True)
        for session in getattr(stats, "recent", [])
    ]
    return require_json_document(
        {
            "session_count": int(getattr(stats, "session_count", 0) or 0),
            "message_count": int(getattr(stats, "message_count", 0) or 0),
            "word_count": int(getattr(stats, "word_count", 0) or 0),
            "providers": _stable_mapping(cast(Mapping[str, int], getattr(stats, "providers", {}) or {})),
            "tags": _stable_mapping(cast(Mapping[str, int], getattr(stats, "tags", {}) or {})),
            "last_sync": getattr(stats, "last_sync", None),
            "recent": recent,
        },
        context="archive stats payload",
    )


def _query_case_payload(name: str, spec: SessionQuerySpec, rows: Sequence[JSONDocument], total: int) -> JSONDocument:
    return require_json_document(
        {
            "name": name,
            "spec": require_json_document(
                {
                    "query_terms": list(spec.query_terms),
                    "retrieval_lane": spec.retrieval_lane,
                    "sort": spec.sort,
                    "limit": spec.limit,
                    "offset": spec.offset,
                    "latest": spec.latest,
                    "filter_has_tool_use": spec.filter_has_tool_use,
                    "filter_has_thinking": spec.filter_has_thinking,
                    "filter_has_paste": spec.filter_has_paste,
                },
                context="self-verify query spec payload",
            ),
            "total": total,
            "items": list(rows),
        },
        context="self-verify query case payload",
    )


async def _capture_list_cases(poly: Polylogue, *, limit: int) -> list[JSONDocument]:
    cases: tuple[tuple[str, SessionQuerySpec], ...] = (
        ("latest", SessionQuerySpec(latest=True, limit=limit)),
        ("date", SessionQuerySpec(sort="date", limit=limit)),
        ("messages", SessionQuerySpec(sort="messages", limit=limit)),
        ("has_tool_use", SessionQuerySpec(filter_has_tool_use=True, limit=limit)),
    )
    payloads: list[JSONDocument] = []
    for name, spec in cases:
        sessions = await poly.list_sessions_for_spec(spec)
        total = await spec.count(poly.config)
        rows = [
            model_json_document(SessionListRowPayload.from_session(session), exclude_none=True) for session in sessions
        ]
        payloads.append(_query_case_payload(name, spec, rows, total))
    return payloads


async def _capture_search_cases(poly: Polylogue, *, queries: Sequence[str], limit: int) -> list[JSONDocument]:
    payloads: list[JSONDocument] = []
    for query in queries:
        spec = SessionQuerySpec(query_terms=(query,), retrieval_lane="dialogue", limit=limit)
        hits = await poly.search_session_hits(spec)
        total = await spec.count(poly.config)
        hit_payloads = [
            SessionSearchHitPayload.from_search_hit(hit, message_count=hit.summary.message_count) for hit in hits
        ]
        resolved_lane = hits[0].retrieval_lane if hits else spec.retrieval_lane
        envelope = build_search_envelope(
            hit_payloads,
            total=total,
            limit=limit,
            offset=0,
            query=query,
            retrieval_lane=resolved_lane,
            sort=spec.sort,
        )
        payloads.append(
            require_json_document(
                {
                    "name": query,
                    "spec": {"query_terms": [query], "retrieval_lane": spec.retrieval_lane, "limit": limit},
                    "envelope": envelope.model_dump(mode="json", exclude_none=True),
                },
                context="self-verify search case payload",
            )
        )
    return payloads


async def _capture_message_cases(
    poly: Polylogue,
    *,
    session_ids: Sequence[str],
    limit: int,
) -> list[JSONDocument]:
    payloads: list[JSONDocument] = []
    for session_id in session_ids:
        messages, total = await poly.get_messages_paginated(session_id, limit=limit, offset=0)
        payloads.append(
            require_json_document(
                {
                    "session_id": session_id,
                    "total": total,
                    "limit": limit,
                    "offset": 0,
                    "messages": [
                        model_json_document(
                            SessionMessagePayload.from_message(message, session_id=session_id),
                            exclude_none=True,
                        )
                        for message in messages
                    ],
                },
                context="self-verify messages payload",
            )
        )
    return payloads


async def _capture_topology_cases(poly: Polylogue, *, session_ids: Sequence[str]) -> list[JSONDocument]:
    from polylogue.mcp.payloads import session_topology_payload

    payloads: list[JSONDocument] = []
    for session_id in session_ids:
        topology = await poly.get_session_topology(session_id)
        payload: JSONValue
        if topology is None:
            payload = None
        else:
            payload = session_topology_payload(topology, session_id=session_id).model_dump(
                mode="json",
                exclude_none=True,
            )
        payloads.append(
            require_json_document(
                {"session_id": session_id, "topology": payload},
                context="self-verify topology payload",
            )
        )
    return payloads


async def capture_snapshot(config: CaptureConfig) -> JSONDocument:
    """Capture a stable read-surface snapshot from a v22 archive database."""
    db_path = config.db_path.expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"archive database not found: {db_path}")

    limit = clamp_query_limit(config.limit, default=10)
    message_limit = clamp_query_limit(config.message_limit, default=8)
    async with Polylogue(archive_root=archive_root(), db_path=db_path) as poly:
        stats = await poly.stats()
        list_cases = await _capture_list_cases(poly, limit=limit)
        top_ids = [
            str(item["id"])
            for case in list_cases
            for item in cast(list[dict[str, object]], case.get("items", []))
            if item.get("id")
        ]
        session_ids = tuple(dict.fromkeys(top_ids))[:limit]
        payload = {
            "report_version": REPORT_VERSION,
            "captured_at": _utc_now(),
            "git_head": _git_head(),
            "db_path": str(db_path),
            "schema_version": _schema_version(db_path),
            "limits": {"session_limit": limit, "message_limit": message_limit},
            "snapshot": {
                "archive_stats": _archive_stats_payload(stats),
                "stats_by": {group: _stable_mapping(await poly.get_stats_by(group)) for group in config.stats_groups},
                "lists": list_cases,
                "searches": await _capture_search_cases(poly, queries=config.search_queries, limit=limit),
                "messages": await _capture_message_cases(poly, session_ids=session_ids, limit=message_limit),
                "topology": await _capture_topology_cases(poly, session_ids=session_ids),
            },
        }
    return require_json_document(payload, context="self-verify snapshot")


def capture_snapshot_sync_for_test(
    db_path: Path,
    *,
    limit: int = 10,
    message_limit: int = 8,
    search_queries: tuple[str, ...] = DEFAULT_SEARCH_QUERIES,
) -> JSONDocument:
    """Synchronous test helper for focused unit tests."""
    return asyncio.run(
        capture_snapshot(
            CaptureConfig(
                db_path=db_path,
                limit=limit,
                message_limit=message_limit,
                search_queries=search_queries,
            )
        )
    )


def _canonical_snapshot(report: Mapping[str, object]) -> object:
    return report.get("snapshot", {})


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_snapshots(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    allowed_sections: Sequence[str] = (),
) -> JSONDocument:
    baseline_snapshot = _canonical_snapshot(baseline)
    candidate_snapshot = _canonical_snapshot(candidate)
    differing_sections: list[str] = []
    if isinstance(baseline_snapshot, Mapping) and isinstance(candidate_snapshot, Mapping):
        keys = sorted(set(baseline_snapshot) | set(candidate_snapshot))
        differing_sections = [str(key) for key in keys if baseline_snapshot.get(key) != candidate_snapshot.get(key)]
    elif baseline_snapshot != candidate_snapshot:
        differing_sections = ["snapshot"]
    allowed = sorted(set(allowed_sections))
    allowed_differing_sections = [section for section in differing_sections if section in allowed]
    unexpected_differing_sections = [section for section in differing_sections if section not in allowed]

    return require_json_document(
        {
            "ok": not unexpected_differing_sections,
            "report_version": REPORT_VERSION,
            "baseline_digest": _digest(baseline_snapshot),
            "candidate_digest": _digest(candidate_snapshot),
            "differing_sections": differing_sections,
            "allowed_sections": allowed,
            "allowed_differing_sections": allowed_differing_sections,
            "unexpected_differing_sections": unexpected_differing_sections,
        },
        context="self-verify compare result",
    )


def _load_report(path: Path) -> JSONDocument:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return require_json_document(loaded, context=f"{path} self-verify report")


def _write_or_print(payload: JSONDocument, *, output: Path | None, json_output: bool) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    if json_output or output is None:
        sys.stdout.write(rendered)
    elif bool(payload.get("ok", True)):
        print(f"wrote {output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Capture a v22 read-surface golden-master snapshot.")
    capture.add_argument("--db", type=Path, default=default_db_path(), help="Archive database path.")
    capture.add_argument("--out", type=Path, help="Path to write the captured JSON snapshot.")
    capture.add_argument("--limit", type=int, default=10, help="Session rows per list/search case.")
    capture.add_argument("--message-limit", type=int, default=8, help="Messages per selected session.")
    capture.add_argument(
        "--search-query",
        action="append",
        dest="search_queries",
        help="Search query to include. Repeatable; defaults to analysis and error.",
    )
    capture.add_argument(
        "--stats-group",
        action="append",
        dest="stats_groups",
        help="Stats grouping to include. Repeatable; defaults to provider/day/month/year.",
    )
    capture.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    compare = subparsers.add_parser("compare", help="Compare two self-verify snapshots.")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument(
        "--allow-section",
        action="append",
        dest="allowed_sections",
        help="Snapshot section allowed to differ. Repeat for each intended archive change.",
    )
    compare.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def _run_capture(args: argparse.Namespace) -> int:
    try:
        payload = asyncio.run(
            capture_snapshot(
                CaptureConfig(
                    db_path=cast(Path, args.db),
                    limit=int(args.limit),
                    message_limit=int(args.message_limit),
                    search_queries=tuple(args.search_queries or DEFAULT_SEARCH_QUERIES),
                    stats_groups=tuple(args.stats_groups or DEFAULT_STATS_GROUPS),
                )
            )
        )
    except Exception as exc:
        print(f"self-verify capture failed: {exc}", file=sys.stderr)
        return 1
    _write_or_print(payload, output=args.out, json_output=bool(args.json))
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    try:
        result = compare_snapshots(
            _load_report(args.baseline),
            _load_report(args.candidate),
            allowed_sections=tuple(args.allowed_sections or ()),
        )
    except Exception as exc:
        print(f"self-verify compare failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print(f"self-verify: snapshots match ({result['baseline_digest']})")
    else:
        unexpected = cast(list[str], result["unexpected_differing_sections"])
        print(f"self-verify: unexpected differences (sections: {', '.join(unexpected)})")
    return 0 if result["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = cast(Literal["capture", "compare"], args.command)
    if command == "capture":
        return _run_capture(args)
    if command == "compare":
        return _run_compare(args)
    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

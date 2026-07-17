"""Generate provider schema packages from the devtools surface."""

from __future__ import annotations

import argparse
import json
import resource
import sqlite3
import sys
from pathlib import Path

from polylogue.cli.shared.schema_command_support import build_schema_privacy_config
from polylogue.cli.shared.schema_rendering import render_schema_generate_result
from polylogue.config import get_config
from polylogue.core.json import JSONDocument
from polylogue.schemas.operator.models import SchemaInferRequest
from polylogue.schemas.operator.workflow import infer_schema


def _process_resources() -> JSONDocument:
    """Return aggregate process evidence without inspecting payload contents."""
    metrics: JSONDocument = {}
    try:
        for line in Path("/proc/self/smaps_rollup").read_text(encoding="ascii").splitlines():
            key, separator, raw_value = line.partition(":")
            if separator and key in {"Pss", "Pss_Anon", "Pss_File", "SwapPss"}:
                metrics[f"{key.lower()}_kb"] = int(raw_value.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    try:
        for line in Path("/proc/self/io").read_text(encoding="ascii").splitlines():
            key, separator, raw_value = line.partition(":")
            if separator and key in {"rchar", "wchar", "read_bytes", "write_bytes"}:
                metrics[key] = int(raw_value.strip())
    except (OSError, ValueError):
        pass
    metrics["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return metrics


def _source_input_summary(index_path: Path) -> JSONDocument:
    """Count the source-tier input without opening raw payloads."""
    source_path = index_path.with_name("source.db")
    if not source_path.exists():
        return {"source_db_bytes": None, "raw_row_count": None, "raw_blob_bytes": None}
    try:
        with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT COUNT(*), COALESCE(SUM(blob_size), 0) FROM raw_sessions").fetchone()
        return {
            "source_db_bytes": source_path.stat().st_size,
            "raw_row_count": int(row[0]) if row is not None else 0,
            "raw_blob_bytes": int(row[1]) if row is not None else 0,
        }
    except (OSError, sqlite3.Error):
        return {"source_db_bytes": None, "raw_row_count": None, "raw_blob_bytes": None}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate provider schema packages and optional evidence clusters.")
    parser.add_argument("--provider", required=True, help="Provider to generate schema for.")
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Also cluster observed samples by structural fingerprint.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for generation.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    parser.add_argument(
        "--privacy",
        choices=("strict", "standard", "permissive"),
        default=None,
        help="Privacy preset level. Defaults to standard.",
    )
    parser.add_argument("--privacy-config", type=Path, default=None, help="Path to TOML privacy config overrides.")
    parser.add_argument("--report", action="store_true", help="Write a redaction report alongside the schema.")
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Bypass all sample caps for full-corpus schema generation.",
    )
    parser.add_argument("--progress", action="store_true", help="Emit aggregate phase progress to stderr.")
    parser.add_argument(
        "--receipt", type=Path, default=None, help="Write an aggregate-only generation receipt as JSON."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    progress_events: list[JSONDocument] = []

    def on_progress(_phase: str, payload: JSONDocument) -> None:
        event = {**payload, "process": _process_resources()}
        progress_events.append(event)
        if args.progress:
            print(f"schema-generate: {json.dumps(event, sort_keys=True)}", file=sys.stderr, flush=True)

    try:
        privacy_config = build_schema_privacy_config(
            privacy=args.privacy,
            privacy_config_path=args.privacy_config,
        )
        result = infer_schema(
            SchemaInferRequest(
                provider=str(args.provider),
                db_path=get_config().db_path,
                max_samples=args.max_samples,
                privacy_config=privacy_config,
                cluster=bool(args.cluster),
                full_corpus=bool(args.full_corpus),
                progress_callback=on_progress if args.progress or args.receipt is not None else None,
            )
        )
    except ValueError as exc:
        print(f"schema-generate: {exc}", file=sys.stderr)
        return 1

    generation = result.generation
    if args.receipt is not None:
        archive_path = get_config().db_path
        try:
            archive_size_bytes = archive_path.stat().st_size
        except OSError:
            archive_size_bytes = None
        receipt = {
            "version": 1,
            "provider": str(args.provider),
            "full_corpus": bool(args.full_corpus),
            "max_samples": args.max_samples,
            "input": {"index_size_bytes": archive_size_bytes, **_source_input_summary(archive_path)},
            "generation": generation.phase_receipt,
            "progress_events": progress_events,
            "process_final": _process_resources(),
            "resume": {
                "status": "restart_from_acquisition",
                "reason": "the private observation journal is intentionally removed after interruption",
            },
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not generation.success:
        print(f"schema-generate: {generation.error or 'Schema generation failed'}", file=sys.stderr)
        return 1
    if args.cluster and result.manifest is None:
        print("schema-generate: No samples found for clustering", file=sys.stderr)
        return 1

    render_schema_generate_result(
        provider=str(args.provider),
        result=result,
        json_output=bool(args.json),
        report=bool(args.report),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

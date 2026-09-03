"""Promote schema evidence clusters from the devtools surface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from polylogue.cli.shared.schema_command_support import build_schema_privacy_config
from polylogue.cli.shared.schema_rendering import render_schema_promote_result
from polylogue.config import get_config
from polylogue.schemas.operator.models import SchemaPromoteRequest
from polylogue.schemas.operator.workflow import promote_schema_cluster


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote an evidence cluster to a registered schema package.")
    parser.add_argument("--provider", required=True, help="Provider name.")
    parser.add_argument("--cluster", dest="cluster_id", required=True, help="Evidence cluster ID to promote.")
    parser.add_argument("--with-samples", action="store_true", help="Re-load samples for full schema generation.")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples when using --with-samples.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    parser.add_argument(
        "--privacy",
        choices=("strict", "standard", "permissive"),
        default=None,
        help="Privacy preset level for the --with-samples candidate schema. Defaults to standard.",
    )
    parser.add_argument("--privacy-config", type=Path, default=None, help="Path to TOML privacy config overrides.")
    return parser


def _schema_registry_root() -> Path:
    """The schema package tree promotion writes to."""
    import polylogue.schemas

    return Path(next(iter(polylogue.schemas.__path__)))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        privacy_config = build_schema_privacy_config(
            privacy=args.privacy,
            privacy_config_path=args.privacy_config,
        )
        result = promote_schema_cluster(
            SchemaPromoteRequest(
                provider=str(args.provider),
                cluster_id=str(args.cluster_id),
                db_path=get_config().db_path,
                with_samples=bool(args.with_samples),
                max_samples=int(args.max_samples),
                privacy_config=privacy_config,
            )
        )
    except ValueError as exc:
        print(f"schema-promote: {exc}", file=sys.stderr)
        return 1

    json_output = bool(args.json)
    render_schema_promote_result(result=result, json_output=json_output)
    # Promotion evidence must be complete for what it just promoted; the audit
    # is a postcondition of this command, not of every static gate. It audits
    # the registry tree promotion actually writes to, which a relative literal
    # would only find from one working directory.
    audit = subprocess.run(
        [sys.executable, "-m", "polylogue.schemas.promotion_audit", str(_schema_registry_root())],
        check=False,
        # --json promises one document on stdout; the audit's own report goes
        # to stderr so it cannot be concatenated onto it.
        capture_output=json_output,
        text=True,
    )
    if json_output:
        for stream in (audit.stdout, audit.stderr):
            if stream:
                sys.stderr.write(stream)
    return audit.returncode


if __name__ == "__main__":
    raise SystemExit(main())

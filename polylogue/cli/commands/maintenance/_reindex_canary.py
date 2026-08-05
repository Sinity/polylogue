"""Run a bounded semantic reindex canary through the existing rebuild service."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("reindex-canary")
@click.option(
    "--archive-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False, readable=True),
    required=True,
    help="Explicit isolated archive containing durable source.db and the active index candidate.",
)
@click.option(
    "--review-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    default=None,
    help="JSON manifest containing one explicit review for every observed difference.",
)
@click.option(
    "--input-index",
    "--input",
    "input_index",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    default=None,
    help="Explicit active index.db to compare and select from. Defaults to the configured active index.",
)
@click.option(
    "--sessions-per-origin",
    "--sample",
    "sessions_per_origin",
    type=int,
    default=100,
    show_default=True,
    help="Newest replayable sessions selected automatically for each origin.",
)
@click.option(
    "--pathology-session-id",
    multiple=True,
    help="Explicit pathology session_id to include. May be supplied multiple times.",
)
@click.option(
    "--sample-session-id",
    multiple=True,
    help="Explicit sample session_id to include. May be supplied multiple times.",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Canary report destination, or existing report with --consume-report.",
)
@click.option(
    "--consume-report",
    is_flag=True,
    help="Validate and approve evidence under archive/rebuild ownership. This never authorizes promotion.",
)
@click.option(
    "--schema-inference-receipt",
    "schema_inference_receipt_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    required=False,
    help="Explicit fresh schema-inference PASS receipt consumed by the canary rebuild.",
)
@click.option("--output-format", type=click.Choice(["plain", "json"]), default="plain", show_default=True)
@click.option(
    "--no-promote",
    is_flag=True,
    help="Required safety gate: leave the rebuilt candidate inactive for comparison.",
)
def reindex_canary_command(
    archive_root: Path,
    review_manifest: Path | None,
    input_index: Path | None,
    sessions_per_origin: int,
    pathology_session_id: tuple[str, ...],
    sample_session_id: tuple[str, ...],
    report_path: Path,
    consume_report: bool,
    schema_inference_receipt_path: Path | None,
    output_format: str,
    no_promote: bool,
) -> None:
    """Replay representative raw inputs into an inactive generation and compare it."""

    if not no_promote:
        raise click.UsageError("reindex-canary requires --no-promote")
    if sessions_per_origin <= 0:
        raise click.BadParameter("sample size must be positive", param_hint="--sessions-per-origin")

    from polylogue.maintenance.reindex_canary import (
        CanaryRunResult,
        UnclassifiedCanaryDiffError,
        approve_canary_report,
        load_canary_report,
        load_canary_review_manifest,
        run_reindex_canary,
        write_canary_report,
    )

    if consume_report:
        if review_manifest is not None or input_index is not None or pathology_session_id or sample_session_id:
            raise click.UsageError("--consume-report cannot be combined with rebuild selection options")
        if not report_path.is_file():
            raise click.BadParameter("report file does not exist", param_hint="--report")
        try:
            payload = approve_canary_report(report_path, archive_root=archive_root)
        except UnclassifiedCanaryDiffError as exc:
            raise click.ClickException(str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        result_payload = {**payload, "decision": "evidence-approved", "promotion_authorized": False}
        if output_format == "json":
            click.echo(json.dumps(result_payload, indent=2, sort_keys=True))
            return
        click.echo("Decision:     evidence approved; promotion is not authorized")
        click.echo(f"Report:       {report_path}")
        return

    result: CanaryRunResult | None = None
    if schema_inference_receipt_path is None:
        raise click.UsageError("reindex-canary rebuild requires --schema-inference-receipt")
    try:
        result = run_reindex_canary(
            archive_root,
            input_index=input_index,
            sessions_per_origin=sessions_per_origin,
            pathology_session_ids=pathology_session_id,
            sample_session_ids=sample_session_id,
            no_promote=no_promote,
            schema_inference_receipt_path=schema_inference_receipt_path,
        )
        reviews = load_canary_review_manifest(review_manifest) if review_manifest is not None else ()
        durable = write_canary_report(
            report_path,
            selection=result.selection,
            comparison=result.comparison,
            rebuild_receipt=result.rebuild_receipt,
            reviews=reviews,
            allow_unreviewed=review_manifest is None,
        )
        load_canary_report(report_path, archive_root=archive_root)
        if result.comparison.differences and review_manifest is None:
            identities = [dict(item.identity) for item in result.comparison.differences]
            raise UnclassifiedCanaryDiffError(
                "unreviewed canary report written to "
                + str(report_path)
                + "; observed identities="
                + json.dumps(identities, sort_keys=True)
            )
    except UnclassifiedCanaryDiffError as exc:
        raise click.ClickException(str(exc)) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if result is None:
        raise click.ClickException("reindex canary produced no run result")
    payload = durable.to_dict()
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(f"Selected:     {len(result.selection.selected_raw_ids):,} raw row(s)")
    click.echo(f"Compared:     {len(result.comparison.differences):,} difference(s)")
    click.echo(f"Classified:   {durable.unclassified_count == 0}")
    click.echo(f"Report:       {report_path}")


__all__ = ["reindex_canary_command"]

"""Blind pairwise comparative judgment (rxdo.9.6/.9.7/.9.11/.9.12).

Wires the previously-unwired judgment mechanisms
(``polylogue/insights/judgment/``) into a real command surface: blinding
(rxdo.9.6, ``blinding.py``) masks provenance before verdict; the recorded
verdict persists through the fully-built but previously-uncalled storage
chokepoint (``upsert_comparative_judgment_assertion``, rxdo.9.11); and
calibration (rxdo.9.12, ``calibration.py``) reports per-judge agreement with
a designated gold actor over the recorded judgments.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.cli.shared.types import AppEnv
from polylogue.core.enums import ComparativeVerdict
from polylogue.core.refs import ActorRef, ExecutionContextRef
from polylogue.insights.judgment.blinding import assert_no_leak, blind_items, reveal
from polylogue.insights.judgment.calibration import compute_calibration
from polylogue.insights.judgment.comparative import build_comparative_judgment
from polylogue.insights.judgment.types import JudgeIdentity

_VERDICT_CHOICES = tuple(verdict.value for verdict in ComparativeVerdict)


def _parse_fields(pairs: tuple[str, ...]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise click.UsageError(f"--left-field/--right-field must be key=value, got {pair!r}")
        fields[key] = value
    return fields


def _judge_identity(actor_ref: str, exec_context_id: str) -> JudgeIdentity:
    return JudgeIdentity(
        actor=ActorRef.parse(actor_ref), execution_context=ExecutionContextRef.from_legacy_id(exec_context_id)
    )


@click.command("compare")
@click.option("--left", "left_ref", default=None, help="Left item ref.")
@click.option("--right", "right_ref", default=None, help="Right item ref.")
@click.option("--left-field", "left_fields", multiple=True, help="Provenance field on the left item, key=value.")
@click.option("--right-field", "right_fields", multiple=True, help="Provenance field on the right item, key=value.")
@click.option("--dimension", default=None, help="Comparison dimension (e.g. quality, correctness).")
@click.option("--rubric", "rubric_id", default=None, help="Rubric identifier the blinded receipt binds to.")
@click.option("--rubric-version", type=int, default=1, show_default=True)
@click.option(
    "--verdict",
    type=click.Choice(_VERDICT_CHOICES),
    default=None,
    help="Record this verdict (omit to only print the blinded pair without recording).",
)
@click.option("--rationale", default=None, help="Optional judge rationale (stored only if --rationale-visible).")
@click.option("--rationale-visible", is_flag=True, help="Store rationale visibly rather than redacted.")
@click.option("--evidence-ref", "evidence_refs", multiple=True, help="Additional evidence ref for the judgment.")
@click.option("--actor-ref", default="user:local", show_default=True)
@click.option("--exec-context-id", default="cli:compare", show_default=True)
@click.option("--calibration", "calibration_mode", is_flag=True, help="Report calibration instead of comparing.")
@click.option("--gold-actor", default=None, help="Actor ref treated as gold for --calibration.")
@click.option("--json", "output_format", flag_value="json", default=None, help="Shortcut for --format json.")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default=None)
@click.pass_obj
def compare_command(
    env: AppEnv,
    left_ref: str | None,
    right_ref: str | None,
    left_fields: tuple[str, ...],
    right_fields: tuple[str, ...],
    dimension: str | None,
    rubric_id: str | None,
    rubric_version: int,
    verdict: str | None,
    rationale: str | None,
    rationale_visible: bool,
    evidence_refs: tuple[str, ...],
    actor_ref: str,
    exec_context_id: str,
    calibration_mode: bool,
    gold_actor: str | None,
    output_format: str | None,
) -> None:
    """Blind pairwise comparative judgment, or a calibration report over recorded judgments."""

    output_format = output_format or "text"

    if calibration_mode:
        if gold_actor is None:
            raise click.UsageError("--calibration requires --gold-actor")
        judgments = run_coroutine_sync(env.polylogue.list_comparative_judgments())
        gold = [j for j in judgments if j.judge.actor_ref == gold_actor]
        candidates = [j for j in judgments if j.judge.actor_ref != gold_actor]
        reports = compute_calibration(candidates, gold)
        if output_format == "json":
            click.echo(
                json.dumps(
                    [
                        {
                            "actor_ref": key.actor_ref,
                            "execution_context_id": key.execution_context_id,
                            "dimension": key.dimension,
                            "n_gold_overlap": report.n_gold_overlap,
                            "agreement_rate": report.agreement_rate,
                            "tie_rate": report.tie_rate,
                            "incomparable_rate": report.incomparable_rate,
                            "abstain_rate": report.abstain_rate,
                            "insufficient_evidence_rate": report.insufficient_evidence_rate,
                            "n_total_verdicts": report.n_total_verdicts,
                        }
                        for key, report in reports.items()
                    ],
                    indent=2,
                )
            )
            return
        if not reports:
            click.echo("No comparative judgments to calibrate.")
            return
        for key, report in reports.items():
            agreement = "unknown (no gold overlap)" if report.agreement_rate is None else f"{report.agreement_rate:.2%}"
            click.echo(
                f"{key.actor_ref} @ {key.execution_context_id} / {key.dimension}: "
                f"agreement={agreement} n={report.n_total_verdicts} gold_overlap={report.n_gold_overlap}"
            )
        return

    if left_ref is None or right_ref is None or dimension is None or rubric_id is None:
        raise click.UsageError("--left, --right, --dimension, and --rubric are required (or use --calibration)")

    left_record: dict[str, object] = {"ref": left_ref, **_parse_fields(left_fields)}
    right_record: dict[str, object] = {"ref": right_ref, **_parse_fields(right_fields)}
    sealed_at_ms = int(time.time() * 1000)
    blinded, receipt = blind_items(
        [left_record, right_record],
        order=[0, 1],
        rubric_ref=rubric_id,
        sealed_at_ms=sealed_at_ms,
    )
    assert_no_leak(blinded)

    if verdict is None:
        payload = {
            "items": [asdict(item) for item in blinded],
            "receipt": asdict(receipt),
        }
        click.echo(json.dumps(payload, indent=2) if output_format == "json" else _render_blinded_text(blinded, receipt))
        return

    judge = _judge_identity(actor_ref, exec_context_id)
    verdict_enum = ComparativeVerdict.from_string(verdict)
    judgment = build_comparative_judgment(
        items=[left_ref, right_ref],
        dimension=dimension,
        verdict=verdict_enum,
        judge=judge,
        blinded=True,
        rubric_id=rubric_id,
        rubric_version=rubric_version,
        decided_at_ms=int(time.time() * 1000),
        evidence_refs=evidence_refs,
        rationale=rationale,
        rationale_visible=rationale_visible,
    )
    envelope = run_coroutine_sync(env.polylogue.record_comparative_judgment(judgment, author_kind="user"))
    revealed_receipt = reveal(receipt, revealed_at_ms=int(time.time() * 1000), verdict_recorded=True)

    if output_format == "json":
        click.echo(
            json.dumps(
                {
                    "judgment_id": judgment.judgment_id,
                    "assertion_id": envelope.assertion_id,
                    "status": envelope.status.value,
                    "verdict": verdict_enum.value,
                    "revealed": {"left": left_record, "right": right_record},
                    "receipt": asdict(revealed_receipt),
                },
                indent=2,
            )
        )
        return
    click.echo(f"Recorded {judgment.judgment_id} ({verdict_enum.value}) as assertion {envelope.assertion_id}.")
    click.echo(f"  revealed left:  {left_record}")
    click.echo(f"  revealed right: {right_record}")


def _render_blinded_text(blinded: object, receipt: object) -> str:
    from polylogue.insights.judgment.blinding import BlindedItem, BlindingReceipt

    assert isinstance(blinded, tuple)
    assert isinstance(receipt, BlindingReceipt)
    lines = [f"rubric: {receipt.rubric_ref}  masked_fields: {list(receipt.masked_fields)}"]
    for item in blinded:
        assert isinstance(item, BlindedItem)
        lines.append(f"  [{item.display_position}] {dict(item.visible_fields)}")
    return "\n".join(lines)


__all__ = ["compare_command"]

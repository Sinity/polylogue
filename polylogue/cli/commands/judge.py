"""Canonical terminal triage for the assertion-candidate lifecycle."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import click

from polylogue.api.sync.bridge import run_coroutine_sync
from polylogue.archive.query.spec import QuerySpecError, parse_query_date
from polylogue.cli.shared.types import AppEnv
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionBulkJudgmentItemEnvelope
from polylogue.surfaces.payloads import (
    AssertionBulkJudgmentPayload,
    AssertionCandidateQueueHealthPayload,
    AssertionCandidateReviewItemPayload,
    AssertionCandidateReviewListPayload,
    AssertionClaimPayload,
    serialize_surface_payload,
)


@dataclass(frozen=True, slots=True)
class JudgeCandidateRow:
    assertion_id: str
    kind: str
    target_ref: str
    body: str
    evidence_refs: tuple[str, ...] = ()
    review_status: str = "pending"
    evidence_lines: tuple[str, ...] = ()

    @classmethod
    def from_claim(cls, claim: AssertionClaimPayload) -> JudgeCandidateRow:
        """Compatibility constructor for callers that already hold a claim."""

        return cls(
            assertion_id=claim.assertion_id,
            kind=claim.kind.value,
            target_ref=claim.target_ref,
            body=str(claim.body_text or claim.value or ""),
            evidence_refs=claim.evidence_refs,
            evidence_lines=tuple(f"unresolved {ref}" for ref in claim.evidence_refs[:5]),
        )

    @classmethod
    def from_review(cls, item: AssertionCandidateReviewItemPayload) -> JudgeCandidateRow:
        evidence_lines = tuple(
            " ".join(
                part
                for part in (
                    preview.state,
                    preview.ref,
                    preview.title,
                    preview.excerpt,
                    preview.open_commands[0] if preview.open_commands else None,
                )
                if part
            )
            for preview in item.evidence_previews
        )
        return cls(
            assertion_id=item.candidate.assertion_id,
            kind=item.candidate.kind.value,
            target_ref=item.candidate.target_ref,
            body=item.claim_summary,
            evidence_refs=item.candidate.evidence_refs,
            review_status=item.review_status,
            evidence_lines=evidence_lines,
        )

    @property
    def ref(self) -> str:
        return f"assertion:{self.assertion_id}"

    @property
    def label(self) -> str:
        return f"{self.review_status:<10} {self.kind:<20} {self.target_ref:<36} {self.first_line[:80]}"

    @property
    def first_line(self) -> str:
        return next(iter(self.body.splitlines()), "")

    @property
    def preview(self) -> str:
        evidence = "\n".join(self.evidence_lines or tuple(f"unresolved {ref}" for ref in self.evidence_refs[:5]))
        evidence = evidence or "(no evidence refs)"
        return (
            f"{self.ref}\nstatus: {self.review_status}\nkind: {self.kind}\n"
            f"target: {self.target_ref}\n\n{self.body}\n\nevidence:\n{evidence}"
        )


def _choose_candidate(rows: Sequence[JudgeCandidateRow]) -> JudgeCandidateRow | None:
    if not rows or not sys.stdin.isatty() or not sys.stdout.isatty() or shutil.which("fzf") is None:
        return None
    content = "\n".join(f"{row.assertion_id}\t{row.label}\t{row.preview.replace(chr(10), ' | ')}" for row in rows)
    try:
        completed = subprocess.run(
            [
                "fzf",
                "--delimiter",
                "\t",
                "--with-nth",
                "2",
                "--height",
                "70%",
                "--reverse",
                "--preview",
                "echo {} | cut -f3- | tr '|' '\\n'",
                "--preview-window",
                "right:60%:wrap",
            ],
            input=content,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    selected_id = completed.stdout.strip().split("\t", 1)[0]
    return next((row for row in rows if row.assertion_id == selected_id), None)


def _render_rows(items: Sequence[AssertionCandidateReviewItemPayload]) -> None:
    if not items:
        click.echo("No assertion candidate review rows found.")
        return
    for item in items:
        age_hours = item.age_ms / (60 * 60 * 1000)
        click.echo(
            f"{item.candidate.assertion_id:<36} {item.review_status:<10} "
            f"{item.candidate.kind.value:<20} {item.candidate.target_ref} "
            f"age={age_hours:.1f}h {item.claim_summary[:100]}"
        )
        if not item.evidence_previews and item.evidence_total_count:
            click.echo(f"  evidence: {item.evidence_total_count} refs; previews unavailable")
        for preview in item.evidence_previews:
            detail = preview.title or preview.excerpt or preview.reason or ""
            click.echo(f"  evidence[{preview.state}]: {preview.ref} {detail}".rstrip())
            for command in preview.open_commands[:1]:
                click.echo(f"    open: {command}")
        if item.evidence_omitted_count:
            click.echo(f"  evidence: {item.evidence_omitted_count} additional refs omitted")


def _parse_time_bound(field: str, value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = parse_query_date(field, value)
    except QuerySpecError as exc:
        raise click.UsageError(str(exc)) from exc
    assert parsed is not None
    return int(parsed.timestamp() * 1000)


def _filter_reviews(
    items: Sequence[AssertionCandidateReviewItemPayload],
    *,
    since: str | None,
    until: str | None,
) -> list[AssertionCandidateReviewItemPayload]:
    lower_ms = _parse_time_bound("since", since)
    upper_ms = _parse_time_bound("until", until)
    return [
        item
        for item in items
        if (lower_ms is None or item.candidate.created_at_ms >= lower_ms)
        and (upper_ms is None or item.candidate.created_at_ms <= upper_ms)
    ]


def _judge(
    env: AppEnv,
    *,
    refs: Sequence[str],
    decision: str,
    reason: str | None,
    actor_ref: str,
    inject: bool,
    replacement_body_text: str | None = None,
    replacement_kind: str | None = None,
) -> AssertionBulkJudgmentPayload:
    return run_coroutine_sync(
        env.polylogue.judge_assertion_candidates(
            items=tuple(
                ArchiveAssertionBulkJudgmentItemEnvelope(
                    candidate_ref=ref,
                    decision=decision,
                    reason=reason,
                    actor_ref=actor_ref,
                    inject=inject,
                    replacement_body_text=replacement_body_text,
                    replacement_kind=replacement_kind,
                )
                for ref in refs
            )
        )
    )


def _edit_and_accept(
    env: AppEnv,
    *,
    selected: JudgeCandidateRow,
    edited_body: str,
    inject: bool,
    actor_ref: str = "user:local",
    reason: str = "operator edited candidate",
) -> AssertionBulkJudgmentPayload:
    """Supersede edited text while preserving the candidate's promoted claim kind.

    The lifecycle authority derives the promoted kind from candidate provenance
    when ``replacement_kind`` is omitted. Passing a wrapper kind such as
    ``transform_candidate`` here would incorrectly promote that wrapper.
    """

    return _judge(
        env,
        refs=(selected.ref,),
        decision="supersede",
        reason=reason,
        actor_ref=actor_ref,
        inject=inject,
        replacement_body_text=edited_body,
    )


def _emit_bulk_result(payload: AssertionBulkJudgmentPayload, decision: str, output_format: str) -> None:
    if output_format == "json":
        click.echo(serialize_surface_payload(payload, exclude_none=True))
        return
    click.echo(
        f"{payload.applied_count} {decision}, {payload.idempotent_count} unchanged, {payload.failed_count} failed"
    )
    for item in payload.items:
        if item.error:
            click.echo(f"  {item.candidate_ref}: {item.error}")


def _render_queue_health(payload: AssertionCandidateQueueHealthPayload, output_format: str) -> None:
    if output_format == "json":
        click.echo(serialize_surface_payload(payload, exclude_none=True))
        return
    click.echo(f"Assertion candidate queue: {payload.state}")
    click.echo(f"  pending: {payload.pending_count}")
    if payload.oldest_pending_age_ms is not None:
        click.echo(f"  oldest age: {payload.oldest_pending_age_ms / (24 * 60 * 60 * 1000):.1f}d")
    if payload.kind_counts:
        click.echo("  kinds: " + ", ".join(f"{key}={value}" for key, value in payload.kind_counts.items()))
    click.echo(
        f"  producer: {payload.producer_status or 'unobserved'}; "
        f"scheduler: {payload.scheduler_state}; debt: {payload.producer_debt_count}"
    )
    click.echo(f"  retention: {payload.retention_outcome}")
    for caveat in payload.caveats:
        click.echo(f"  caveat: {caveat}")


@click.command("judge")
@click.option("--list", "list_only", is_flag=True, help="List pending candidates without judging.")
@click.option("--review", is_flag=True, help="List durable review history, not only pending candidates.")
@click.option("--status", "show_status", is_flag=True, help="Show non-destructive queue and producer health.")
@click.option("--accept", "accept_refs", multiple=True, help="Accept one or more candidate assertion refs.")
@click.option("--reject", "reject_refs", multiple=True, help="Reject one or more candidate assertion refs.")
@click.option("--defer", "defer_refs", multiple=True, help="Durably defer one or more candidate assertion refs.")
@click.option("--supersede", "supersede_refs", multiple=True, help="Supersede one or more candidate assertion refs.")
@click.option(
    "--accept-all-of-kind",
    is_flag=True,
    help="Accept every pending candidate selected by --kind and time/target filters.",
)
@click.option("--target-ref", default=None, help="Limit review rows to one target object ref.")
@click.option("--kind", "kind_filter", type=click.Choice([kind.value for kind in AssertionKind]), default=None)
@click.option(
    "--candidate-status",
    "candidate_statuses",
    multiple=True,
    type=click.Choice(
        [
            AssertionStatus.CANDIDATE.value,
            AssertionStatus.ACCEPTED.value,
            AssertionStatus.REJECTED.value,
            AssertionStatus.DEFERRED.value,
            AssertionStatus.SUPERSEDED.value,
        ]
    ),
)
@click.option("--since", default=None, help="Lower candidate creation-time bound (ISO or relative date).")
@click.option("--until", default=None, help="Upper candidate creation-time bound (ISO or relative date).")
@click.option("--limit", type=click.IntRange(min=1, max=500), default=50, show_default=True)
@click.option("--reason", default=None)
@click.option("--actor-ref", default="user:local", show_default=True)
@click.option("--inject", is_flag=True, help="Authorize injection for accepted/superseding assertions.")
@click.option("--replacement-kind", type=click.Choice([kind.value for kind in AssertionKind]), default=None)
@click.option("--body", "replacement_body_text", default=None, help="Replacement body for --supersede.")
@click.option("--json", "output_format", flag_value="json", default=None, help="Shortcut for --format json.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default=None,
    help="Output format (default: text).",
)
@click.pass_obj
def judge_command(
    env: AppEnv,
    list_only: bool,
    review: bool,
    show_status: bool,
    accept_refs: tuple[str, ...],
    reject_refs: tuple[str, ...],
    defer_refs: tuple[str, ...],
    supersede_refs: tuple[str, ...],
    accept_all_of_kind: bool,
    target_ref: str | None,
    kind_filter: str | None,
    candidate_statuses: tuple[str, ...],
    since: str | None,
    until: str | None,
    limit: int,
    reason: str | None,
    actor_ref: str,
    inject: bool,
    replacement_kind: str | None,
    replacement_body_text: str | None,
    output_format: str | None,
) -> None:
    """Review and judge assertions through the sole public operator lifecycle."""

    output_format = output_format or "text"
    action_groups = [accept_refs, reject_refs, defer_refs, supersede_refs]
    action_count = sum(bool(group) for group in action_groups) + int(accept_all_of_kind)
    if action_count > 1:
        raise click.UsageError("choose one judgment action for one invocation")
    if show_status and (action_count or list_only or review):
        raise click.UsageError("--status cannot be combined with review or judgment actions")
    if accept_all_of_kind and kind_filter is None:
        raise click.UsageError("--accept-all-of-kind requires --kind")
    if supersede_refs and (replacement_kind is None or replacement_body_text is None):
        raise click.UsageError("--supersede requires --replacement-kind and --body")
    if not supersede_refs and (replacement_kind is not None or replacement_body_text is not None):
        raise click.UsageError("--replacement-kind and --body are only valid with --supersede")
    if inject and (reject_refs or defer_refs):
        raise click.UsageError("--inject is only valid for accept or supersede judgments")

    if show_status:
        health = run_coroutine_sync(env.polylogue.assertion_candidate_queue_health())
        _render_queue_health(health, output_format)
        return

    direct_actions = (
        ("accept", accept_refs),
        ("reject", reject_refs),
        ("defer", defer_refs),
        ("supersede", supersede_refs),
    )
    for decision, refs in direct_actions:
        if not refs:
            continue
        bulk_payload = _judge(
            env,
            refs=refs,
            decision=decision,
            reason=reason,
            actor_ref=actor_ref,
            inject=inject if decision in {"accept", "supersede"} else False,
            replacement_kind=replacement_kind if decision == "supersede" else None,
            replacement_body_text=replacement_body_text if decision == "supersede" else None,
        )
        _emit_bulk_result(bulk_payload, decision, output_format)
        return

    kinds = None if kind_filter is None else (AssertionKind.from_string(kind_filter),)
    statuses: Sequence[str | AssertionStatus]
    if candidate_statuses:
        statuses = candidate_statuses
    elif review:
        statuses = (
            AssertionStatus.CANDIDATE,
            AssertionStatus.ACCEPTED,
            AssertionStatus.REJECTED,
            AssertionStatus.DEFERRED,
            AssertionStatus.SUPERSEDED,
        )
    else:
        statuses = (AssertionStatus.CANDIDATE,)

    read_limit = None if since is not None or until is not None or accept_all_of_kind else limit
    review_payload = run_coroutine_sync(
        env.polylogue.list_assertion_candidate_reviews(
            target_ref=target_ref,
            kinds=kinds,
            statuses=statuses,
            limit=read_limit,
        )
    )
    filtered = _filter_reviews(review_payload.items, since=since, until=until)

    if accept_all_of_kind:
        bulk = _judge(
            env,
            refs=tuple(item.candidate_ref for item in filtered),
            decision="accept",
            reason=reason,
            actor_ref=actor_ref,
            inject=inject,
        )
        _emit_bulk_result(bulk, "accept", output_format)
        return

    filtered = filtered[:limit]
    visible_payload = AssertionCandidateReviewListPayload(
        items=tuple(filtered),
        total=len(filtered),
        limit=limit,
        target_ref=target_ref,
        candidate_statuses=tuple(AssertionStatus.from_string(status) for status in statuses),
    )
    if output_format == "json":
        click.echo(serialize_surface_payload(visible_payload, exclude_none=True))
        return
    if review or list_only or not sys.stdin.isatty() or not sys.stdout.isatty():
        _render_rows(filtered)
        return

    rows = [JudgeCandidateRow.from_review(item) for item in filtered if item.review_status == "pending"]
    selected = _choose_candidate(rows)
    if selected is None:
        _render_rows(filtered)
        return
    action = click.prompt("Judge [a]ccept/[r]eject/[d]efer/[e]dit-and-accept/[s]kip", default="s")
    if action in {"s", "d"}:
        durable_reason = reason or ("operator skipped interactive review" if action == "s" else None)
        _judge(
            env,
            refs=(selected.ref,),
            decision="defer",
            reason=durable_reason,
            actor_ref=actor_ref,
            inject=False,
        )
        click.echo("Deferred.")
        return
    if action == "r":
        rejection_reason = reason or click.prompt("Reason", default="")
        _judge(
            env,
            refs=(selected.ref,),
            decision="reject",
            reason=rejection_reason,
            actor_ref=actor_ref,
            inject=False,
        )
        click.echo("Rejected.")
        return
    if action == "e":
        edited = click.edit(selected.body, extension=".md")
        if edited is None:
            _judge(
                env,
                refs=(selected.ref,),
                decision="defer",
                reason="operator closed editor without saving",
                actor_ref=actor_ref,
                inject=False,
            )
            click.echo("Deferred; editor closed without saving.")
            return
        _edit_and_accept(
            env,
            selected=selected,
            edited_body=edited,
            inject=inject,
            actor_ref=actor_ref,
            reason=reason or "operator edited candidate",
        )
        click.echo("Edited candidate promoted.")
        return
    if action == "a":
        _judge(
            env,
            refs=(selected.ref,),
            decision="accept",
            reason=reason,
            actor_ref=actor_ref,
            inject=inject,
        )
        click.echo("Accepted.")
        return
    raise click.UsageError("judge action must be a, r, d, e, or s")


__all__ = ["JudgeCandidateRow", "_edit_and_accept", "judge_command"]

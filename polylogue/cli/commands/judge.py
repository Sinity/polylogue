"""Terminal triage for candidate assertion claims."""

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
from polylogue.core.enums import AssertionKind
from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionBulkJudgmentItemEnvelope
from polylogue.surfaces.payloads import (
    AssertionBulkJudgmentPayload,
    AssertionClaimListPayload,
    AssertionClaimPayload,
    serialize_surface_payload,
)


@dataclass(frozen=True, slots=True)
class JudgeCandidateRow:
    assertion_id: str
    kind: str
    target_ref: str
    body: str
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_claim(cls, claim: AssertionClaimPayload) -> JudgeCandidateRow:
        return cls(
            assertion_id=claim.assertion_id,
            kind=claim.kind.value,
            target_ref=claim.target_ref,
            body=str(claim.body_text or claim.value or ""),
            evidence_refs=claim.evidence_refs,
        )

    @property
    def ref(self) -> str:
        return f"assertion:{self.assertion_id}"

    @property
    def label(self) -> str:
        return f"{self.kind:<20} {self.target_ref:<36} {self.first_line[:80]}"

    @property
    def first_line(self) -> str:
        return next(iter(self.body.splitlines()), "")

    @property
    def preview(self) -> str:
        evidence = "\n".join(self.evidence_refs[:5]) or "(no evidence refs)"
        return f"{self.ref}\nkind: {self.kind}\ntarget: {self.target_ref}\n\n{self.body}\n\nevidence:\n{evidence}"


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


def _render_rows(rows: Sequence[JudgeCandidateRow]) -> None:
    if not rows:
        click.echo("No pending assertion candidates.")
        return
    for row in rows:
        click.echo(f"{row.assertion_id:<36} {row.kind:<20} {row.target_ref} {row.first_line[:100]}")


def _filter_candidates(
    claims: Sequence[AssertionClaimPayload],
    *,
    kind_filter: str | None,
    since: str | None,
) -> list[AssertionClaimPayload]:
    """Apply the terminal queue's kind and lower-time filters locally."""

    filtered = list(claims)
    if kind_filter is not None:
        filtered = [claim for claim in filtered if claim.kind.value == kind_filter]
    if since is None:
        return filtered
    try:
        lower = parse_query_date("since", since)
    except QuerySpecError as exc:
        raise click.UsageError(str(exc)) from exc
    assert lower is not None
    lower_ms = int(lower.timestamp() * 1000)
    return [claim for claim in filtered if claim.created_at_ms >= lower_ms]


def _judge(
    env: AppEnv,
    *,
    refs: Sequence[str],
    decision: str,
    reason: str | None,
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
                    inject=inject,
                    replacement_body_text=replacement_body_text,
                    replacement_kind=replacement_kind,
                )
                for ref in refs
            )
        )
    )


@click.command("judge")
@click.option("--list", "list_only", is_flag=True, help="List the pending queue without judging.")
@click.option("--accept", "accept_refs", multiple=True, help="Accept one or more candidate assertion refs.")
@click.option("--reject", "reject_refs", multiple=True, help="Reject one or more candidate assertion refs.")
@click.option(
    "--accept-all-of-kind",
    is_flag=True,
    help="Accept every pending candidate selected by --kind and --since.",
)
@click.option("--kind", "kind_filter", type=click.Choice([kind.value for kind in AssertionKind]), default=None)
@click.option("--since", default=None, help="Daily-triage lower timestamp bound (ISO or relative date).")
@click.option("--reason", default=None)
@click.option("--inject", is_flag=True, help="Authorize injection for accepted candidates.")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]), default="text", show_default=True)
@click.pass_obj
def judge_command(
    env: AppEnv,
    list_only: bool,
    accept_refs: tuple[str, ...],
    reject_refs: tuple[str, ...],
    accept_all_of_kind: bool,
    kind_filter: str | None,
    since: str | None,
    reason: str | None,
    inject: bool,
    output_format: str,
) -> None:
    """Interactively triage candidate assertions through the normal lifecycle."""

    if accept_refs and reject_refs or accept_all_of_kind and (accept_refs or reject_refs):
        raise click.UsageError("choose one judgment action for one invocation")
    if accept_all_of_kind and kind_filter is None:
        raise click.UsageError("--accept-all-of-kind requires --kind")
    if accept_refs or reject_refs:
        decision = "accept" if accept_refs else "reject"
        bulk_payload = _judge(
            env,
            refs=accept_refs or reject_refs,
            decision=decision,
            reason=reason,
            inject=inject if decision == "accept" else False,
        )
        if output_format == "json":
            click.echo(serialize_surface_payload(bulk_payload, exclude_none=True))
            return
        click.echo(
            f"{bulk_payload.applied_count} {decision}ed, {bulk_payload.idempotent_count} unchanged, "
            f"{bulk_payload.failed_count} failed"
        )
        return

    claims = run_coroutine_sync(env.polylogue.list_assertion_candidates(limit=50))
    claims = _filter_candidates(claims, kind_filter=kind_filter, since=since)
    if accept_all_of_kind:
        bulk_payload = _judge(
            env,
            refs=tuple(f"assertion:{claim.assertion_id}" for claim in claims),
            decision="accept",
            reason=reason,
            inject=inject,
        )
        if output_format == "json":
            click.echo(serialize_surface_payload(bulk_payload, exclude_none=True))
            return
        click.echo(
            f"{bulk_payload.applied_count} accepted, {bulk_payload.idempotent_count} unchanged, "
            f"{bulk_payload.failed_count} failed"
        )
        return
    rows = [JudgeCandidateRow.from_claim(claim) for claim in claims]
    if output_format == "json":
        list_payload = AssertionClaimListPayload(items=tuple(claims), total=len(rows), limit=50)
        click.echo(serialize_surface_payload(list_payload, exclude_none=True))
        return
    if list_only or not sys.stdin.isatty() or not sys.stdout.isatty():
        _render_rows(rows)
        return
    selected = _choose_candidate(rows)
    if selected is None:
        _render_rows(rows)
        return
    action = click.prompt("Judge [a]ccept/[r]eject/[e]dit-and-accept/[s]kip", default="s")
    if action == "s":
        click.echo("Skipped.")
        return
    if action == "r":
        _judge(env, refs=(selected.ref,), decision="reject", reason=click.prompt("Reason", default=""), inject=False)
        click.echo("Rejected.")
        return
    if action == "e":
        edited = click.edit(selected.body, extension=".md")
        if edited is None:
            click.echo("Skipped; editor closed without saving.")
            return
        _judge(
            env,
            refs=(selected.ref,),
            decision="supersede",
            reason="operator edited candidate",
            inject=inject,
            replacement_body_text=edited,
            replacement_kind=selected.kind,
        )
        click.echo("Edited candidate promoted.")
        return
    if action == "a":
        _judge(env, refs=(selected.ref,), decision="accept", reason=None, inject=inject)
        click.echo("Accepted.")
        return
    raise click.UsageError("judge action must be a, r, e, or s")


__all__ = ["JudgeCandidateRow", "judge_command"]

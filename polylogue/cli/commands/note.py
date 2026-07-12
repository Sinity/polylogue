"""Zero-friction terminal capture into the assertion judgment queue."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from polylogue.api import Polylogue
from polylogue.core.enums import AssertionKind
from polylogue.paths import archive_root
from polylogue.surfaces.payloads import AssertionClaimPayload

MAX_NOTE_STDIN_BYTES = 256 * 1024

_KIND_MAP: dict[str, AssertionKind] = {
    "note": AssertionKind.NOTE,
    "claim": AssertionKind.DECISION,
    "correction": AssertionKind.CORRECTION,
    "lesson": AssertionKind.LESSON,
}


def _scope_refs(repo: str | None, topic: str | None) -> tuple[str, ...]:
    refs: list[str] = []
    if repo:
        refs.append(repo if repo.startswith("repo:") else f"repo:{repo}")
    if topic:
        refs.append(topic if topic.startswith("insight:") else f"insight:{topic}")
    return tuple(refs)


def _stdin_note() -> str:
    payload = sys.stdin.buffer.read(MAX_NOTE_STDIN_BYTES + 1)
    if len(payload) > MAX_NOTE_STDIN_BYTES:
        raise click.UsageError(f"stdin note exceeds {MAX_NOTE_STDIN_BYTES} byte limit; save a bounded summary instead.")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise click.UsageError("stdin note must be valid UTF-8") from exc


@click.command("note")
@click.argument("text", required=False)
@click.option("--stdin", "from_stdin", is_flag=True, help="Read note text from stdin (bounded).")
@click.option("--ref", "refs", multiple=True, help="Evidence ref, or 'last' for the latest session in this cwd.")
@click.option("--repo", default=None, help="Attach repository scope (accepts repo:NAME or NAME).")
@click.option("--topic", default=None, help="Attach topic scope (accepts insight:NAME or NAME).")
@click.option("--kind", "kind_name", type=click.Choice(tuple(_KIND_MAP)), default="note", show_default=True)
@click.option("--format", "output_format", type=click.Choice(("text", "json")), default="text", show_default=True)
def note_command(
    text: str | None,
    from_stdin: bool,
    refs: tuple[str, ...],
    repo: str | None,
    topic: str | None,
    kind_name: str,
    output_format: str,
) -> None:
    """Capture one terminal memory candidate; judgment remains a separate step."""

    if from_stdin == (text is not None):
        raise click.UsageError("provide note TEXT or --stdin, but not both")
    body_text = _stdin_note() if from_stdin else text
    assert body_text is not None

    async def run() -> AssertionClaimPayload:
        async with Polylogue(archive_root=archive_root()) as poly:
            return await poly.capture_assertion_candidate(
                body_text=body_text,
                kind=_KIND_MAP[kind_name],
                refs=refs,
                scope_refs=_scope_refs(repo, topic),
                cwd=Path.cwd(),
            )

    try:
        captured = asyncio.run(run())
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    payload = captured.model_dump(mode="json")
    if output_format == "json":
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    anchor = "unanchored" if payload["value"]["unanchored"] else ", ".join(payload["evidence_refs"])
    click.echo(f"Captured candidate {payload['assertion_id']} ({kind_name}; {anchor}).")


__all__ = ["MAX_NOTE_STDIN_BYTES", "note_command"]

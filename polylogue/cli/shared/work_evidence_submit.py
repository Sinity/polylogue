"""Persist a CLI-computed work-evidence graph through the resident daemon.

``materialize-incident-evidence`` and ``reconcile-work-effects`` compute a
graph in this process, which only reads. The resident daemon is the one writer
of ``index.db``, so both submit the result to the declared
``mutation.work_evidence.graph.replace`` operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from polylogue.cli.shared.types import AppEnv

if TYPE_CHECKING:
    from polylogue.analysis.work_evidence import WorkEvidenceGraph


def submit_work_evidence_graph(
    env: AppEnv, graph: WorkEvidenceGraph, *, expected_base_digest: str | None
) -> dict[str, object]:
    """Submit one graph; a stored graph that moved since ``expected_base_digest`` is refused."""
    from polylogue.cli.archive_query import submit_cli_mutation

    envelope = submit_cli_mutation(
        env,
        "mutation.work_evidence.graph.replace",
        {"graph": graph.model_dump(mode="json"), "expected_base_digest": expected_base_digest},
    )
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise click.ClickException("daemon returned no work-evidence graph replacement receipt")
    return {str(key): value for key, value in result.items()}


def render_replacement(payload: dict[str, object]) -> None:
    """Print the daemon's replacement receipt, when one was requested."""
    replacement = payload.get("replacement")
    if not isinstance(replacement, dict):
        return
    state = "replaced" if replacement.get("changed") else "unchanged (already stored)"
    click.echo(f"Stored graph:        {state} {replacement.get('digest')}")


__all__ = ["render_replacement", "submit_work_evidence_graph"]

"""Production read-modify-write wrapper around work-effect reconciliation.

``insights.work_effects`` is pure: adapters read outside evidence, matching
derives judgments, ``reconcile_work_effects`` returns a new graph. Nothing in
that module touches the archive. This module is the one production caller
that does: it loads an existing work-evidence graph (e.g. one of the
``claude-workflow:<run-id>`` graphs the Claude Workflow materializer already
builds), reconciles it against the supplied effect adapters, and -- when
``apply=True`` -- persists the reconciled graph back through the same
repository route ``replace_work_evidence_graph`` already round-trips in
``tests/unit/insights/test_work_evidence.py``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from polylogue.core.errors import PolylogueError
from polylogue.insights.work_effects import (
    DEFAULT_WORK_ITEM_ID_PATTERN,
    RepositoryEffectAdapter,
    collect_repository_effects,
    derive_direct_identifier_judgments,
)
from polylogue.insights.work_reconciliation import reconcile_work_effects
from polylogue.storage.repository import SessionRepository


class WorkEvidenceGraphNotFoundError(PolylogueError):
    """Raised when the requested graph_id has no stored work-evidence graph."""

    def __init__(self, graph_id: str) -> None:
        super().__init__(f"no work-evidence graph stored for graph_id={graph_id!r}")
        self.graph_id = graph_id


@dataclass(frozen=True, slots=True)
class WorkEffectReconciliationSummary:
    """Quantified result of one reconciliation pass over one graph."""

    graph_id: str
    applied: bool
    claims_total: int
    claims_evaluated: int
    claims_unevaluated: int
    effect_count_by_authority: dict[str, int]
    judgment_count_by_evaluation: dict[str, int]
    adapter_failures: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


async def reconcile_graph_repository_effects(
    repository: SessionRepository,
    *,
    graph_id: str,
    adapters: Sequence[RepositoryEffectAdapter],
    since_ms: int | None = None,
    until_ms: int | None = None,
    id_pattern: re.Pattern[str] = DEFAULT_WORK_ITEM_ID_PATTERN,
    apply: bool = False,
) -> WorkEffectReconciliationSummary:
    """Reconcile one stored work-evidence graph against observed effects.

    Read-only unless ``apply=True``: dry-run callers (the CLI's default) get
    a full summary -- including which adapters could not run -- without
    mutating ``index.db``.
    """

    graph = await repository.get_work_evidence_graph(graph_id)
    if graph is None:
        raise WorkEvidenceGraphNotFoundError(graph_id)

    collection = collect_repository_effects(adapters, since_ms=since_ms, until_ms=until_ms)
    judgments = derive_direct_identifier_judgments(graph, collection.effects, pattern=id_pattern)
    reconciled = reconcile_work_effects(graph, effects=collection.effects, judgments=judgments)

    if apply:
        await repository.replace_work_evidence_graph(reconciled)

    claim_refs = {node.ref.format() for node in graph.nodes if node.kind == "claim"}
    evaluated_claim_refs = {judgment.claim_ref.format() for judgment in judgments}
    evaluated = claim_refs & evaluated_claim_refs

    return WorkEffectReconciliationSummary(
        graph_id=graph_id,
        applied=apply,
        claims_total=len(claim_refs),
        claims_evaluated=len(evaluated),
        claims_unevaluated=len(claim_refs - evaluated),
        effect_count_by_authority=dict(sorted(Counter(effect.authority for effect in collection.effects).items())),
        judgment_count_by_evaluation=dict(sorted(Counter(judgment.evaluation for judgment in judgments).items())),
        adapter_failures=tuple(
            {"authority": failure.authority, "reason": failure.reason} for failure in collection.unavailable
        ),
    )


__all__ = [
    "WorkEffectReconciliationSummary",
    "WorkEvidenceGraphNotFoundError",
    "reconcile_graph_repository_effects",
]

"""Freeze the exact insight target universe from one supplied index snapshot."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from polylogue.operations.insight_acceptance import (
    MAX_INSIGHT_ACCEPTED_PARTS,
    MAX_INSIGHT_PART_TARGETS,
    AcceptedInsightPart,
    AcceptedInsightTarget,
    InsightScopeKind,
    build_insight_page_context,
    insight_manifest_digest,
)
from polylogue.operations.mutation_transaction import (
    MutationPlan,
    MutationReceipt,
    MutationTarget,
    RecoveryDisposition,
    build_typed_plan,
)
from polylogue.storage.derived.session.threads import thread_root_ids_sync
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


@dataclass(frozen=True, slots=True)
class InsightManifest:
    scope_kind: InsightScopeKind
    index_generation: str
    recipe_version: str
    pages: tuple[tuple[AcceptedInsightTarget, ...], ...]
    digest: str


@dataclass(frozen=True, slots=True)
class AcceptedInsightActuator:
    """Use the shared authorization policy without a legacy materializer path."""

    operation: str = "mutate-rebuild-insights"
    destructive_class: str = "maintenance"
    required_confirmation: str = "role_only"

    def prepare(self, plan: MutationPlan) -> MutationPlan:
        if plan.operation != self.operation:
            raise ValueError("insight page belongs to another operation")
        return plan

    def apply(self, _plan: MutationPlan, _args: object) -> MutationReceipt:
        raise RuntimeError("accepted insight publication requires the shared staged owner")

    def inspect_recovery(self, _operation: object, _args: object) -> RecoveryDisposition:
        return RecoveryDisposition("unknown", "operator-blocking", "insight recovery requires exact output bindings")


def _required_index_connection(archive: ArchiveStore) -> sqlite3.Connection:
    """Return the pinned index handle required by insight planning.

    Acquire-only archive handles deliberately have no derived-tier connection.
    Planning a derived insight manifest against one would be semantically
    invalid, rather than a recoverable empty result.
    """
    connection = archive.index_connection
    if connection is None:
        raise RuntimeError("insight planning requires a pinned index-tier connection")
    return connection


def prepare_insight_manifest(
    archive: ArchiveStore,
    session_ids: Sequence[str] | None,
    *,
    index_generation: str,
    recipe_version: str,
    check_stop: Callable[[], None],
) -> InsightManifest:
    """Select once, including orphan partitions only for a full sweep.

    Every keyset page uses the supplied, already-pinned connection. Later
    accepted execution consumes these exact identifiers, never a new scan.
    """
    scope_kind: InsightScopeKind = "full" if session_ids is None else "explicit"
    index_connection = _required_index_connection(archive)
    targets: list[AcceptedInsightTarget] = []
    maximum = MAX_INSIGHT_ACCEPTED_PARTS * MAX_INSIGHT_PART_TARGETS
    if session_ids is None:
        cursor = ""
        while True:
            check_stop()
            rows = index_connection.execute(
                "SELECT session_id, disposition FROM ("
                "SELECT session_id, 'required' AS disposition FROM sessions "
                "UNION ALL SELECT p.session_id, 'excess' AS disposition FROM session_profiles p "
                "WHERE NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id=p.session_id)) "
                "WHERE session_id > ? ORDER BY session_id LIMIT ?",
                (cursor, MAX_INSIGHT_PART_TARGETS),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                targets.append(AcceptedInsightTarget(f"session:{row[0]}", row[1]))
            if len(targets) > maximum:
                raise ValueError("insight manifest exceeds its bounded page count")
            cursor = str(rows[-1][0])
    else:
        seen: set[str] = set()
        for requested in session_ids:
            check_stop()
            try:
                resolved = archive.resolve_session_id(requested)
            except KeyError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                targets.append(AcceptedInsightTarget(f"session:{resolved}", "required"))
            if len(targets) > maximum:
                raise ValueError("insight manifest exceeds its bounded page count")
    pages = tuple(
        tuple(targets[offset : offset + MAX_INSIGHT_PART_TARGETS])
        for offset in range(0, len(targets), MAX_INSIGHT_PART_TARGETS)
    ) or ((),)
    # The digest covers only immutable target facts, not subsequently allocated
    # preview references. Empty explicit scope remains one audited empty page.
    parts = tuple(
        AcceptedInsightPart(
            preview_ref="pending-preview",
            authorization_ref="pending-authorization",
            plan_hash="0" * 64,
            ordinal=ordinal,
            page_count=len(pages),
            manifest_digest="0" * 64,
            previous_preview_ref="pending-preview" if ordinal else None,
            scope_kind=scope_kind,
            index_generation=index_generation,
            recipe_version=recipe_version,
            targets=page,
        )
        for ordinal, page in enumerate(pages)
    )
    return InsightManifest(scope_kind, index_generation, recipe_version, pages, insight_manifest_digest(parts))


def insight_page_plan(
    manifest: InsightManifest,
    ordinal: int,
    *,
    previous_preview_ref: str | None,
    archive_instance_id: str,
    archive_identity_digest: str,
    now_ms: int,
    expires_at_ms: int,
) -> MutationPlan:
    """Bind a single immutable page to the existing maintenance policy."""
    page = manifest.pages[ordinal]
    context = build_insight_page_context(
        scope_kind=manifest.scope_kind,
        index_generation=manifest.index_generation,
        recipe_version=manifest.recipe_version,
        ordinal=ordinal,
        page_count=len(manifest.pages),
        manifest_digest=manifest.digest,
        previous_preview_ref=previous_preview_ref,
        targets=page,
    )
    parameter_digest = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    targets = tuple(
        MutationTarget(
            kind="session",
            ref=target.target_ref,
            policy_key="insights-rebuild",
            identity_digest=hashlib.sha256(
                json.dumps(
                    [target.target_ref, target.disposition, manifest.index_generation, manifest.recipe_version],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            effect_identity=f"mutate-rebuild-insights:{manifest.digest}:{target.target_ref}",
            durability="derived",
            recovery="rebuild",
        )
        for target in page
    )
    return build_typed_plan(
        operation="mutate-rebuild-insights",
        operation_version=1,
        archive_instance_id=archive_instance_id,
        archive_identity_digest=archive_identity_digest,
        targets=targets,
        affected_tiers=("index", "user"),
        parameter_digest=parameter_digest,
        required_capabilities=("archive.rebuild_insights",),
        destructive_class="maintenance",
        required_confirmation="role_only",
        prepared_at_ms=now_ms,
        expires_at_ms=expires_at_ms,
        context=context,
    )


def insight_terminal_view_counts(
    archive: ArchiveStore, targets: Sequence[AcceptedInsightTarget], *, check_stop: Callable[[], None]
) -> tuple[int, int]:
    """Observe shared views once; page totals must not double-count roots."""
    index_connection = _required_index_connection(archive)
    session_ids = tuple(
        target.target_ref.removeprefix("session:") for target in targets if target.disposition == "required"
    )
    if not session_ids:
        return 0, 0
    roots: set[str] = set()
    for offset in range(0, len(session_ids), MAX_INSIGHT_PART_TARGETS):
        check_stop()
        roots.update(
            thread_root_ids_sync(index_connection, session_ids[offset : offset + MAX_INSIGHT_PART_TARGETS]).values()
        )
    ordered_roots = tuple(sorted(roots))
    threads = 0
    for offset in range(0, len(ordered_roots), MAX_INSIGHT_PART_TARGETS):
        check_stop()
        page = ordered_roots[offset : offset + MAX_INSIGHT_PART_TARGETS]
        placeholders = ",".join("?" for _ in page)
        threads += int(
            index_connection.execute(
                f"SELECT COUNT(*) FROM threads WHERE thread_id IN ({placeholders})", page
            ).fetchone()[0]
        )
    # Tag rollups retain the existing archive-global public count semantics.
    tags = int(index_connection.execute("SELECT COUNT(*) FROM session_tag_rollups").fetchone()[0])
    return threads, tags

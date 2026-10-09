"""Freeze the exact insight target universe from one supplied index snapshot."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from itertools import batched
from pathlib import Path

from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.operations.insight_acceptance import (
    MAX_INSIGHT_PART_TARGETS,
    AcceptedInsightPart,
    AcceptedInsightTarget,
    InsightScopeKind,
    build_insight_page_context,
    insight_manifest_digest,
)
from polylogue.operations.mutation_transaction import (
    ConfirmationStrength,
    DestructiveClass,
    MutationPlan,
    MutationReceipt,
    MutationTarget,
    RecoveryResolution,
    ReplayHandles,
    build_typed_plan,
)
from polylogue.storage.derived.session.threads import thread_root_ids_sync
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


@dataclass(frozen=True, slots=True)
class InsightManifest:
    scope_kind: InsightScopeKind
    index_generation: str
    recipe_version: str
    pages: InsightManifestPages
    digest: str

    def close(self) -> None:
        self.pages.close()


class InsightManifestPages:
    """Private immutable target ordinals, with one page resident at a time."""

    def __init__(self, directory: tempfile.TemporaryDirectory[str], target_count: int) -> None:
        self.directory = directory
        self.target_count = target_count
        self.path = Path(directory.name) / "targets.db"

    def __len__(self) -> int:
        return max(1, (self.target_count + MAX_INSIGHT_PART_TARGETS - 1) // MAX_INSIGHT_PART_TARGETS)

    def __getitem__(self, ordinal: int) -> tuple[AcceptedInsightTarget, ...]:
        if ordinal < 0:
            ordinal += len(self)
        if not 0 <= ordinal < len(self):
            raise IndexError(ordinal)
        with closing(sqlite3.connect(self.path)) as connection:
            return tuple(
                AcceptedInsightTarget(str(row[0]), row[1])
                for row in connection.execute(
                    "SELECT target_ref, disposition FROM targets WHERE ordinal >= ? AND ordinal < ? ORDER BY ordinal",
                    (ordinal * MAX_INSIGHT_PART_TARGETS, (ordinal + 1) * MAX_INSIGHT_PART_TARGETS),
                )
            )

    def __iter__(self) -> Iterator[tuple[AcceptedInsightTarget, ...]]:
        for ordinal in range(len(self)):
            yield self[ordinal]

    def close(self) -> None:
        self.directory.cleanup()


@dataclass(frozen=True, slots=True)
class AcceptedInsightActuator:
    """Use the shared authorization policy without a legacy materializer path."""

    operation: str = "mutate-rebuild-insights"
    destructive_class: DestructiveClass = "maintenance"
    required_confirmation: ConfirmationStrength = "role_only"

    def prepare(self, args: object) -> MutationPlan:
        if not isinstance(args, MutationPlan):
            raise TypeError("accepted insight preparation requires its sealed mutation plan")
        plan = args
        if plan.operation != self.operation:
            raise ValueError("insight page belongs to another operation")
        return plan

    def apply(self, _plan: MutationPlan, _args: object) -> MutationReceipt:
        raise RuntimeError("accepted insight publication requires the shared staged owner")

    def recover(self, _handles: ReplayHandles, _plan: MutationPlan) -> RecoveryResolution:
        """Terminalize an interrupted page; see ``InsightsRebuildActuator.recover``."""
        return RecoveryResolution(
            "not-replayable", "an interrupted insight page is re-derived by convergence or a new rebuild request"
        )


def _required_index_connection(archive: ArchiveStore) -> sqlite3.Connection:
    """Return the pinned index handle required by insight planning.

    Acquire-only archive handles deliberately have no derived-tier connection.
    Planning a derived insight manifest against one would be semantically
    invalid, rather than a recoverable empty result.
    """
    connection = archive.index_connection
    if connection is None:
        raise ArchiveTierUnavailableError(
            tier="index",
            path=str(archive.archive_root / "index.db"),
            reason="insight planning was opened in source-tier acquisition mode",
            guidance="wait for ordinary daemon convergence to restore derived state, then retry",
        )
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
    directory = tempfile.TemporaryDirectory(prefix="polylogue-insight-manifest-")
    count = 0
    try:
        with closing(sqlite3.connect(Path(directory.name) / "targets.db")) as spool:
            spool.execute("PRAGMA cache_size = -64")
            spool.execute(
                "CREATE TABLE targets (ordinal INTEGER PRIMARY KEY, target_ref TEXT NOT NULL UNIQUE, "
                "disposition TEXT NOT NULL) STRICT"
            )
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
                        spool.execute("INSERT INTO targets VALUES (?, ?, ?)", (count, f"session:{row[0]}", row[1]))
                        count += 1
                    cursor = str(rows[-1][0])
            else:
                for requested in session_ids:
                    check_stop()
                    try:
                        resolved = archive.resolve_session_id(requested)
                    except KeyError:
                        continue
                    count += spool.execute(
                        "INSERT OR IGNORE INTO targets VALUES (?, ?, 'required')", (count, f"session:{resolved}")
                    ).rowcount
            spool.commit()
        pages = InsightManifestPages(directory, count)

        # No preview reference participates in the immutable target digest.
        def parts() -> Iterator[AcceptedInsightPart]:
            for ordinal, page in enumerate(pages):
                check_stop()
                yield AcceptedInsightPart(
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

        return InsightManifest(scope_kind, index_generation, recipe_version, pages, insight_manifest_digest(parts()))
    except BaseException:
        directory.cleanup()
        raise


def insight_page_plan(
    manifest: InsightManifest,
    ordinal: int,
    *,
    previous_preview_ref: str | None,
    archive_instance_id: str,
    archive_identity_digest: str,
    now_ms: int,
    expires_at_ms: int | None,
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
        expires_at_ms=now_ms if expires_at_ms is None else expires_at_ms,
        context=context,
    )


def insight_terminal_view_counts(
    archive: ArchiveStore, targets: Iterable[AcceptedInsightTarget], *, check_stop: Callable[[], None]
) -> tuple[int, int]:
    """Observe shared views once; page totals must not double-count roots."""
    index_connection = _required_index_connection(archive)
    session_ids = (target.target_ref.removeprefix("session:") for target in targets if target.disposition == "required")
    selected = 0
    threads = 0
    with (
        tempfile.TemporaryDirectory(prefix="polylogue-insight-roots-") as scratch,
        closing(sqlite3.connect(Path(scratch) / "roots.db")) as roots,
    ):
        roots.execute("PRAGMA cache_size = -64")
        roots.execute("CREATE TABLE roots (root_id TEXT PRIMARY KEY) STRICT")
        for page in batched(session_ids, MAX_INSIGHT_PART_TARGETS):
            check_stop()
            selected += len(page)
            roots.executemany(
                "INSERT OR IGNORE INTO roots VALUES (?)",
                ((root,) for root in thread_root_ids_sync(index_connection, page).values()),
            )
        with closing(roots.execute("SELECT root_id FROM roots ORDER BY root_id")) as cursor:
            while rows := cursor.fetchmany(MAX_INSIGHT_PART_TARGETS):
                check_stop()
                page = tuple(str(row[0]) for row in rows)
                placeholders = ",".join("?" for _ in page)
                threads += int(
                    index_connection.execute(
                        f"SELECT COUNT(*) FROM threads WHERE thread_id IN ({placeholders})", page
                    ).fetchone()[0]
                )
    if not selected:
        return 0, 0
    # Tag rollups retain the existing archive-global public count semantics.
    tags = int(index_connection.execute("SELECT COUNT(*) FROM session_tag_rollups").fetchone()[0])
    return threads, tags

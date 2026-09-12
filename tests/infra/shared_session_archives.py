"""The archives an xdist run shares across workers, and the one route that builds them.

A shared artifact's cold build holds a per-key exclusive flock, so under xdist
the first worker to reach it builds while every other worker wanting it blocks,
and the wait is charged to whichever test arrived first. Warming the whole set
on the controller before any worker starts costs the same once and nothing per
worker.

Every session-scoped fixture over a shared archive resolves it through a
builder declared here, and :func:`shared_session_archives` lists all of them:
an archive reachable from a fixture cannot be absent from the warm-up.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.infra.workload_artifacts import SeededArchiveArtifact

__all__ = [
    "SharedSessionArchive",
    "WarmedArchives",
    "named_workload_archive",
    "schema_coverage_archive",
    "semantic_archive",
    "shared_session_archives",
    "warm_shared_session_archives",
]


def semantic_archive() -> SeededArchiveArtifact:
    """The default single-corpus archive, built when a caller names no specs."""
    from tests.infra.workload_artifacts import build_seeded_archive

    return build_seeded_archive()


def schema_coverage_archive() -> SeededArchiveArtifact:
    """The all-provider schema-coverage archive behind ``seeded_archive``."""
    from tests.infra.workload_artifacts import build_seeded_archive
    from tests.infra.workload_declarations import schema_coverage_corpus_specs

    return build_seeded_archive(schema_coverage_corpus_specs())


def named_workload_archive(name: str) -> SeededArchiveArtifact:
    """One entry of the finite named workload catalog."""
    from tests.infra.workload_artifacts import build_seeded_archive
    from tests.infra.workload_declarations import named_corpus_specs

    return build_seeded_archive(named_corpus_specs(name))


@dataclass(frozen=True)
class SharedSessionArchive:
    """One artifact built once per run and read by tests on every worker."""

    name: str
    build: Callable[[], object]


_REGISTRY: tuple[SharedSessionArchive, ...] | None = None


def shared_session_archives() -> tuple[SharedSessionArchive, ...]:
    """Every shared archive, resolved on first call so import costs nothing."""
    global _REGISTRY
    if _REGISTRY is None:
        from tests.infra.workload_declarations import NAMED_WORKLOAD_PROFILES

        named = tuple(
            SharedSessionArchive(
                f"named-{profile.name}",
                # Bound per profile: a closure over the loop variable would
                # build the last profile once per entry.
                lambda name=profile.name: named_workload_archive(name),  # type: ignore[misc]
            )
            for profile in NAMED_WORKLOAD_PROFILES
        )
        _REGISTRY = (
            SharedSessionArchive("semantic", semantic_archive),
            *named,
            SharedSessionArchive("schema-coverage", schema_coverage_archive),
        )
    return _REGISTRY


@dataclass(frozen=True)
class WarmedArchives:
    """What the warm-up built, and what it left for a fixture to build."""

    warmed: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]


def warm_shared_session_archives() -> WarmedArchives:
    """Build every shared archive once, surviving any archive that cannot build.

    The warm-up is an accelerator: it decides only whether a build is charged
    to the controller or to the first worker that asks. A build that raises
    here must therefore raise again in the one test that needs the archive,
    never before collection for every test in the run.
    """
    warmed: list[str] = []
    failed: list[tuple[str, str]] = []
    for entry in shared_session_archives():
        try:
            entry.build()
        except Exception as exc:
            failed.append((entry.name, f"{type(exc).__name__}: {exc}"))
        else:
            warmed.append(entry.name)
    return WarmedArchives(tuple(warmed), tuple(failed))

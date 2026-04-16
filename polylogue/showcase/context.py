"""Showcase data context — clean abstraction over data modes.

A ``ShowcaseContext`` is an immutable reference to a database and archive
root, with the environment variables needed to point the CLI at them.
Four constructors handle the different data modes:

- ``synthetic()``: fresh temp workspace with generated data
- ``live()``: user's real archive, read-only
- ``source(names)``: fresh workspace ingesting specific real sources
- ``existing(db_path, archive_root)``: explicit paths, no setup

The data mode is orthogonal to what exercises or stages run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from polylogue.showcase.corpus_requests import showcase_corpus_request


@dataclass(frozen=True, slots=True)
class ShowcaseContext:
    """Immutable reference to a database and archive for showcase/QA runs."""

    db_path: Path
    archive_root: Path
    env_vars: dict[str, str] = field(default_factory=dict)

    @classmethod
    def synthetic(
        cls,
        *,
        count: int = 3,
        style: str = "showcase",
        workspace_dir: Path | None = None,
    ) -> ShowcaseContext:
        """Create a fresh temp workspace with synthetic generated data."""
        from polylogue.showcase.workspace import (
            create_verification_workspace,
            seed_workspace_from_corpus_request,
        )

        workspace = create_verification_workspace(workspace_dir)
        seed_workspace_from_corpus_request(
            workspace,
            request=showcase_corpus_request(count=count, style=style),
        )
        return cls(
            db_path=workspace.db_path,
            archive_root=workspace.archive_root,
            env_vars=dict(workspace.env_vars),
        )

    @classmethod
    def live(cls) -> ShowcaseContext:
        """Use the user's real archive (read-only)."""
        from polylogue.paths import archive_root, db_path

        return cls(
            db_path=db_path(),
            archive_root=archive_root(),
            env_vars={},
        )

    @classmethod
    def source(
        cls,
        names: list[str],
        *,
        workspace_dir: Path | None = None,
        regenerate_schemas: bool = False,
    ) -> ShowcaseContext:
        """Create a fresh workspace ingesting specific named real sources."""
        from polylogue.showcase.workspace import (
            create_verification_workspace,
            run_pipeline_for_configured_sources,
        )

        workspace = create_verification_workspace(workspace_dir)
        run_pipeline_for_configured_sources(
            workspace,
            source_names=names,
            regenerate_schemas=regenerate_schemas,
        )
        return cls(
            db_path=workspace.db_path,
            archive_root=workspace.archive_root,
            env_vars=dict(workspace.env_vars),
        )

    @classmethod
    def existing(
        cls,
        db_path: Path,
        archive_root: Path,
    ) -> ShowcaseContext:
        """Use explicit paths, no setup needed."""
        return cls(
            db_path=db_path,
            archive_root=archive_root,
            env_vars={
                "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
                "POLYLOGUE_FORCE_PLAIN": "1",
            },
        )


__all__ = ["ShowcaseContext"]

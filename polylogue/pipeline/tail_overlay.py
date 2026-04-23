"""Temporary archive overlays built from ahead-of-archive source state."""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Config, Source
from polylogue.lib.tail_overlay import TailOverlayInfo, with_tail_overlay_provider_meta
from polylogue.pipeline.prepare import prepare_bundle, save_bundle
from polylogue.services import RuntimeServices, build_runtime_services
from polylogue.sources.source_parsing import iter_source_conversations_with_raw

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

SUPPORTED_TAIL_SOURCE_NAMES = frozenset({"claude-code"})


class TailOverlayUnavailableError(ValueError):
    """Raised when a tail overlay cannot be constructed for the current config."""


def _snapshot_archive_db(source_db: Path, snapshot_db: Path) -> None:
    """Copy a consistent SQLite snapshot into a temporary overlay path."""
    if not source_db.exists():
        return

    source = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    target = sqlite3.connect(snapshot_db)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def _tail_sources(config: Config, *, source_names: Sequence[str] | None = None) -> tuple[Source, ...]:
    allowed_names = set(SUPPORTED_TAIL_SOURCE_NAMES)
    if source_names is not None:
        allowed_names &= {str(name) for name in source_names}

    return tuple(
        source
        for source in config.sources
        if source.name in allowed_names and source.path is not None and source.path.exists()
    )


async def _materialize_tail_sources(
    base_services: RuntimeServices,
    overlay_services: RuntimeServices,
    *,
    config: Config,
    sources: Sequence[Source],
) -> None:
    base_repository = base_services.get_repository()
    overlay_repository = overlay_services.get_repository()
    known_mtimes = await base_repository.get_known_source_mtimes()

    for source in sources:
        for raw_data, conversation in iter_source_conversations_with_raw(
            source,
            known_mtimes=known_mtimes,
            capture_raw=True,
        ):
            if raw_data is None:
                continue

            prepared = await prepare_bundle(
                conversation,
                source.name,
                archive_root=config.archive_root,
                repository=overlay_repository,
            )
            overlay_info = TailOverlayInfo(
                source_name=source.name,
                source_path=raw_data.source_path,
                archive_state="ahead_of_archive" if raw_data.source_path in known_mtimes else "unseen",
                file_mtime=raw_data.file_mtime,
            )
            prepared.bundle.conversation = prepared.bundle.conversation.model_copy(
                update={
                    "provider_meta": with_tail_overlay_provider_meta(
                        prepared.bundle.conversation.provider_meta,
                        overlay_info,
                    )
                }
            )
            await save_bundle(prepared.bundle, repository=overlay_repository)


@asynccontextmanager
async def tail_overlay_services(
    services: RuntimeServices,
    *,
    source_names: Sequence[str] | None = None,
) -> AsyncIterator[RuntimeServices]:
    """Yield runtime services backed by a tailed archive overlay."""
    base_config = services.get_config()
    sources = _tail_sources(base_config, source_names=source_names)
    if not sources:
        raise TailOverlayUnavailableError(
            "--tail currently supports configured Claude Code sources only; no readable claude-code source was found."
        )

    with tempfile.TemporaryDirectory(prefix="polylogue-tail-overlay-") as tmp:
        overlay_db_path = Path(tmp) / "tail-overlay.db"
        _snapshot_archive_db(services.get_backend().db_path, overlay_db_path)
        overlay_config = Config(
            archive_root=base_config.archive_root,
            render_root=base_config.render_root,
            sources=list(base_config.sources),
            db_path=overlay_db_path,
            drive_config=base_config.drive_config,
            index_config=base_config.index_config,
        )
        overlay_services = build_runtime_services(config=overlay_config, db_path=overlay_db_path)
        try:
            await _materialize_tail_sources(
                services,
                overlay_services,
                config=overlay_config,
                sources=sources,
            )
            yield overlay_services
        finally:
            await overlay_services.close()


__all__ = [
    "SUPPORTED_TAIL_SOURCE_NAMES",
    "TailOverlayUnavailableError",
    "tail_overlay_services",
]

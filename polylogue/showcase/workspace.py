"""Shared verification workspace lifecycle helpers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config, Source, get_config
from polylogue.scenarios import CorpusSourceKind, CorpusSpec, resolve_corpus_specs
from polylogue.sync_bridge import run_coroutine_sync


@dataclass(frozen=True, slots=True)
class VerificationWorkspace:
    """Filesystem layout and environment for an isolated verification run."""

    root: Path
    data_home: Path
    state_home: Path
    archive_root: Path
    render_root: Path
    fake_home: Path
    fixture_dir: Path
    inbox_dir: Path
    report_dir: Path
    env_vars: dict[str, str]


def create_verification_workspace(
    workspace_dir: Path | None = None,
    *,
    prefix: str = "polylogue-qa-",
) -> VerificationWorkspace:
    """Create an isolated verification workspace and its environment map."""
    if workspace_dir is None:
        workspace_dir = Path(tempfile.mkdtemp(prefix=prefix))

    data_home = workspace_dir / "data"
    state_home = workspace_dir / "state"
    archive_root = workspace_dir / "archive"
    render_root = archive_root / "render"
    fake_home = workspace_dir / "home"
    fixture_dir = workspace_dir / "fixtures"
    inbox_dir = data_home / "polylogue" / "inbox"
    report_dir = workspace_dir / "reports"

    for path in [data_home, state_home, archive_root, render_root, fake_home]:
        path.mkdir(parents=True, exist_ok=True)

    env_vars = {
        "HOME": str(fake_home),
        "XDG_DATA_HOME": str(data_home),
        "XDG_STATE_HOME": str(state_home),
        "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
        "POLYLOGUE_FORCE_PLAIN": "1",
    }

    return VerificationWorkspace(
        root=workspace_dir,
        data_home=data_home,
        state_home=state_home,
        archive_root=archive_root,
        render_root=render_root,
        fake_home=fake_home,
        fixture_dir=fixture_dir,
        inbox_dir=inbox_dir,
        report_dir=report_dir,
        env_vars=env_vars,
    )


def ensure_report_dir(
    workspace: VerificationWorkspace,
    report_dir: Path | None = None,
) -> Path:
    """Return a writable report directory for a verification run."""
    target = report_dir or workspace.report_dir
    target.mkdir(parents=True, exist_ok=True)
    return target


def generate_synthetic_fixtures(
    fixture_dir: Path,
    *,
    providers: tuple[str, ...] | None = None,
    count: int = 3,
    style: str = "showcase",
    corpus_source: CorpusSourceKind | str = CorpusSourceKind.DEFAULT,
) -> None:
    """Generate schema-driven synthetic fixtures for all providers."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    provider_names = providers
    if provider_names is None and CorpusSourceKind(corpus_source) is CorpusSourceKind.DEFAULT:
        provider_names = tuple(SyntheticCorpus.available_providers())
    specs = resolve_corpus_specs(
        providers=provider_names,
        source=corpus_source,
        count=count,
        messages_min=6,
        messages_max=19,
        seed=42,
        style=style,
    )
    generate_synthetic_fixtures_from_specs(fixture_dir, corpus_specs=specs)


def generate_synthetic_fixtures_from_specs(
    fixture_dir: Path,
    *,
    corpus_specs: tuple[CorpusSpec, ...],
) -> None:
    """Generate schema-driven synthetic fixtures for explicit corpus specs."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    fixture_dir.mkdir(parents=True, exist_ok=True)
    SyntheticCorpus.write_specs_artifacts(corpus_specs, fixture_dir, prefix="showcase")


def mirror_fixtures_to_inbox(fixture_dir: Path, inbox_dir: Path) -> None:
    """Copy generated fixture files into the inbox layout used by source discovery."""
    inbox_dir.mkdir(parents=True, exist_ok=True)
    for provider_dir in fixture_dir.iterdir():
        if not provider_dir.is_dir():
            continue
        dest = inbox_dir / provider_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for fixture in provider_dir.iterdir():
            if fixture.is_file():
                (dest / fixture.name).write_bytes(fixture.read_bytes())


@contextmanager
def override_workspace_env(env_vars: dict[str, str]) -> Iterator[None]:
    """Temporarily apply workspace env vars to the current process."""
    old_env: dict[str, str | None] = {}
    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        yield
    finally:
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def run_pipeline_for_fixture_workspace(
    workspace: VerificationWorkspace,
    *,
    regenerate_schemas: bool = False,
) -> None:
    """Ingest synthetic fixtures inside an isolated workspace."""
    mirror_fixtures_to_inbox(workspace.fixture_dir, workspace.inbox_dir)

    sources: list[Source] = []
    if workspace.fixture_dir.exists():
        for provider_dir in sorted(workspace.fixture_dir.iterdir()):
            if provider_dir.is_dir():
                sources.append(Source(name=provider_dir.name, path=provider_dir))

    _run_pipeline_with_sources(
        workspace,
        sources=sources,
        regenerate_schemas=regenerate_schemas,
    )


def run_pipeline_for_configured_sources(
    workspace: VerificationWorkspace,
    *,
    source_names: list[str] | None = None,
    regenerate_schemas: bool = False,
) -> None:
    """Ingest configured user sources inside an isolated workspace."""
    configured_sources = get_config().sources
    if source_names is None:
        sources = list(configured_sources)
    else:
        names = set(source_names)
        sources = [source for source in configured_sources if source.name in names]

    _run_pipeline_with_sources(
        workspace,
        sources=sources,
        regenerate_schemas=regenerate_schemas,
    )


def _run_pipeline_with_sources(
    workspace: VerificationWorkspace,
    *,
    sources: list[Source],
    regenerate_schemas: bool,
) -> None:
    """Run the ingestion pipeline under a workspace environment."""
    del regenerate_schemas
    from polylogue.pipeline.runner import run_sources

    config = Config(
        archive_root=workspace.archive_root,
        render_root=workspace.render_root,
        sources=sources,
    )

    with override_workspace_env(workspace.env_vars):
        run_coroutine_sync(
            run_sources(
                config=config,
                stage="all",
                plan=None,
                ui=None,
                source_names=None,
            )
        )


__all__ = [
    "VerificationWorkspace",
    "create_verification_workspace",
    "ensure_report_dir",
    "generate_synthetic_fixtures",
    "generate_synthetic_fixtures_from_specs",
    "mirror_fixtures_to_inbox",
    "override_workspace_env",
    "run_pipeline_for_configured_sources",
    "run_pipeline_for_fixture_workspace",
]

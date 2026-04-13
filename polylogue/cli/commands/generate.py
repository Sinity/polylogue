"""Generate synthetic conversations for demos, testing, and inspection."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click

from polylogue.cli.types import AppEnv
from polylogue.scenarios import build_default_corpus_specs


@contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables and restore previous values on exit."""
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@click.command("generate")
@click.option(
    "--provider",
    "-p",
    "providers",
    multiple=True,
    help="Providers to include (default: all). Can be repeated.",
)
@click.option("--count", "-n", default=3, show_default=True, help="Conversations per provider")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None, help="Output directory")
@click.option("--seed", is_flag=True, help="Run pipeline to produce a usable demo environment")
@click.option("--env-only", is_flag=True, help="Print shell export statements only (requires --seed)")
@click.pass_obj
def generate_command(
    env: AppEnv,
    providers: tuple[str, ...],
    count: int,
    output_dir: Path | None,
    seed: bool,
    env_only: bool,
) -> None:
    """Generate synthetic conversations for demos, testing, and inspection.

    \b
    Examples:
      polylogue audit generate                        # Raw wire-format files
      polylogue audit generate -p chatgpt -n 5        # ChatGPT only, 5 conversations
      polylogue audit generate -o /tmp/corpus         # Custom output directory
      polylogue audit generate --seed                 # Full demo environment
      eval "$(polylogue audit generate --seed --env-only)"  # Shell-friendly
    """
    if env_only and not seed:
        raise click.UsageError("--env-only requires --seed")

    from polylogue.schemas.synthetic import SyntheticCorpus

    available = SyntheticCorpus.available_providers()
    selected = list(providers) if providers else available

    invalid = set(selected) - set(available)
    if invalid:
        raise click.BadParameter(
            f"Unknown provider(s): {', '.join(sorted(invalid))}. Available: {', '.join(available)}",
            param_hint="--provider",
        )

    if seed:
        _do_seed(env, selected, count, output_dir, env_only)
    else:
        _do_corpus(env, selected, count, output_dir)


def _do_corpus(
    env: AppEnv,
    providers: list[str],
    count: int,
    output_dir: Path | None,
) -> None:
    """Generate raw wire-format files for each provider."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    out = output_dir or Path(tempfile.mkdtemp(prefix="polylogue-corpus-"))
    specs = build_default_corpus_specs(
        providers=providers,
        count=count,
        messages_min=4,
        messages_max=15,
        seed=42,
    )

    for spec in specs:
        provider_dir = out / spec.provider
        written = SyntheticCorpus.write_spec_artifacts(spec, provider_dir, prefix="sample")

        env.ui.console.print(
            f"  {spec.provider}: {written.batch.report.generated_count} files "
            f"({written.batch.report.element_kind or 'default'}) → {provider_dir}"
        )

    env.ui.console.print(f"\nCorpus written to: {out}")


def _do_seed(
    env: AppEnv,
    providers: list[str],
    count: int,
    output_dir: Path | None,
    env_only: bool,
) -> None:
    """Seed a full demo database via the pipeline."""
    from polylogue.config import Config, Source
    from polylogue.pipeline.runner import run_sources
    from polylogue.schemas.synthetic import SyntheticCorpus
    from polylogue.sync_bridge import run_coroutine_sync

    out = output_dir or Path(tempfile.mkdtemp(prefix="polylogue-demo-"))

    data_home = out / "data"
    state_home = out / "state"
    config_home = out / "config"
    cache_home = out / "cache"
    archive_root = out / "archive"
    render_root = archive_root / "render"
    fake_home = out / "home"
    fixture_dir = out / "fixtures"
    inbox_dir = data_home / "polylogue" / "inbox"

    for d in [data_home, state_home, config_home, cache_home, archive_root, render_root, fake_home, inbox_dir]:
        d.mkdir(parents=True, exist_ok=True)

    env_vars = {
        "HOME": str(fake_home),
        "XDG_CONFIG_HOME": str(config_home),
        "XDG_CACHE_HOME": str(cache_home),
        "XDG_DATA_HOME": str(data_home),
        "XDG_STATE_HOME": str(state_home),
        "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
        "POLYLOGUE_FORCE_PLAIN": "1",
    }

    sources: list[Source] = []
    specs = build_default_corpus_specs(
        providers=providers,
        count=count,
        messages_min=6,
        messages_max=19,
        seed=42,
    )
    for spec in specs:
        provider_dir = fixture_dir / spec.provider
        SyntheticCorpus.write_spec_artifacts(spec, provider_dir, prefix="demo")

        sources.append(Source(name=spec.provider, path=provider_dir))
        inbox_provider_dir = inbox_dir / spec.provider
        inbox_provider_dir.mkdir(parents=True, exist_ok=True)
        for fixture in provider_dir.iterdir():
            if fixture.is_file():
                (inbox_provider_dir / fixture.name).write_bytes(fixture.read_bytes())

    with _temporary_env(env_vars):
        config = Config(
            archive_root=archive_root,
            render_root=render_root,
            sources=sources,
        )

        result = run_coroutine_sync(
            run_sources(
                config=config,
                stage="all",
                plan=None,
                ui=None,
                source_names=None,
            )
        )

    if env_only:
        for key, value in env_vars.items():
            click.echo(f'export {key}="{value}"')
    else:
        c = result.counts
        env.ui.console.print(f"Seeded {c.get('conversations', 0)} conversations, {c.get('messages', 0)} messages")
        env.ui.console.print(f"\nDemo environment: {out}")
        env.ui.console.print("\nTo use:")
        for key, value in env_vars.items():
            env.ui.console.print(f'  export {key}="{value}"')


__all__ = ["generate_command"]

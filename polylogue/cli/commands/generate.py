"""Generate synthetic conversations for demos, testing, and inspection."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click

from polylogue.cli.types import AppEnv


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
    "--provider", "-p", "providers", multiple=True,
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
      polylogue generate                        # Raw wire-format files
      polylogue generate -p chatgpt -n 5        # ChatGPT only, 5 conversations
      polylogue generate -o /tmp/corpus         # Custom output directory
      polylogue generate --seed                 # Full demo environment
      polylogue generate --seed --env-only | eval  # Shell-friendly
    """
    if env_only and not seed:
        raise click.UsageError("--env-only requires --seed")

    from polylogue.schemas.synthetic import SyntheticCorpus

    available = SyntheticCorpus.available_providers()
    selected = list(providers) if providers else available

    invalid = set(selected) - set(available)
    if invalid:
        raise click.BadParameter(
            f"Unknown provider(s): {', '.join(sorted(invalid))}. "
            f"Available: {', '.join(available)}",
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

    for provider in providers:
        corpus = SyntheticCorpus.for_provider(provider)
        ext = ".json" if corpus.wire_format.encoding == "json" else ".jsonl"
        provider_dir = out / provider
        provider_dir.mkdir(parents=True, exist_ok=True)

        raw_items = corpus.generate(
            count=count,
            messages_per_conversation=range(4, 16),
            seed=42,
        )

        for idx, raw_bytes in enumerate(raw_items):
            dest = provider_dir / f"sample-{idx:02d}{ext}"
            dest.write_bytes(raw_bytes)

        env.ui.console.print(f"  {provider}: {count} files → {provider_dir}")

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
    archive_root = out / "archive"
    render_root = archive_root / "render"
    fixture_dir = out / "fixtures"

    for d in [data_home, state_home, archive_root, render_root]:
        d.mkdir(parents=True, exist_ok=True)

    env_vars = {
        "XDG_DATA_HOME": str(data_home),
        "XDG_STATE_HOME": str(state_home),
        "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
        "POLYLOGUE_RENDER_ROOT": str(render_root),
        "POLYLOGUE_FORCE_PLAIN": "1",
    }

    sources: list[Source] = []
    for provider in providers:
        corpus = SyntheticCorpus.for_provider(provider)
        ext = ".json" if corpus.wire_format.encoding == "json" else ".jsonl"
        provider_dir = fixture_dir / provider
        provider_dir.mkdir(parents=True, exist_ok=True)

        raw_items = corpus.generate(
            count=count,
            messages_per_conversation=range(6, 20),
            seed=42,
        )
        for idx, raw_bytes in enumerate(raw_items):
            (provider_dir / f"demo-{idx:02d}{ext}").write_bytes(raw_bytes)

        sources.append(Source(name=provider, path=provider_dir))

    with _temporary_env(env_vars):
        config = Config(
            archive_root=archive_root,
            render_root=render_root,
            sources=sources,
        )

        result = run_coroutine_sync(run_sources(
            config=config,
            stage="all",
            plan=None,
            ui=None,
            source_names=None,
        ))

    if env_only:
        for key, value in env_vars.items():
            click.echo(f'export {key}="{value}"')
    else:
        c = result.counts
        env.ui.console.print(f"Seeded {c.get('conversations', 0)} conversations, "
                             f"{c.get('messages', 0)} messages")
        env.ui.console.print(f"\nDemo environment: {out}")
        env.ui.console.print("\nTo use:")
        for key, value in env_vars.items():
            env.ui.console.print(f'  export {key}="{value}"')


__all__ = ["generate_command"]

"""Demo command for generating synthetic data and seeding demo environments."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import click

from polylogue.cli.types import AppEnv


@click.command("demo")
@click.option("--seed", "mode", flag_value="seed", help="Seed a demo database with synthetic data")
@click.option("--corpus", "mode", flag_value="corpus", help="Generate raw fixture files for inspection")
@click.option(
    "--provider", "-p", "providers", multiple=True,
    help="Providers to include (default: all). Can be repeated.",
)
@click.option("--count", "-n", default=3, show_default=True, help="Conversations per provider")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None, help="Output directory")
@click.option("--env-only", is_flag=True, help="Print export statements only (for eval)")
@click.pass_obj
def demo_command(
    env: AppEnv,
    mode: str | None,
    providers: tuple[str, ...],
    count: int,
    output_dir: Path | None,
    env_only: bool,
) -> None:
    """Generate synthetic conversations for demos, testing, and inspection.

    \b
    Two modes:
      --seed     Create a full demo environment (database + rendered files)
      --corpus   Write raw provider-format files to disk for inspection

    \b
    Examples:
      polylogue demo --seed                     # Seed demo DB, print env vars
      polylogue demo --seed --env-only | eval   # Shell-friendly
      polylogue demo --corpus -o /tmp/corpus    # Inspect raw wire formats
      polylogue demo --corpus -p chatgpt -n 5   # ChatGPT only, 5 conversations
    """
    from polylogue.schemas.synthetic import SyntheticCorpus

    if not mode:
        # Default to corpus if no mode specified
        mode = "corpus"

    available = SyntheticCorpus.available_providers()
    selected = list(providers) if providers else available

    # Validate providers
    invalid = set(selected) - set(available)
    if invalid:
        raise click.BadParameter(
            f"Unknown provider(s): {', '.join(sorted(invalid))}. "
            f"Available: {', '.join(available)}",
            param_hint="--provider",
        )

    if mode == "corpus":
        _do_corpus(env, selected, count, output_dir)
    elif mode == "seed":
        _do_seed(env, selected, count, output_dir, env_only)


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
    import asyncio

    from polylogue.config import Config, Source
    from polylogue.pipeline.async_runner import async_run_sources
    from polylogue.schemas.synthetic import SyntheticCorpus

    out = output_dir or Path(tempfile.mkdtemp(prefix="polylogue-demo-"))

    data_home = out / "data"
    state_home = out / "state"
    archive_root = out / "archive"
    render_root = archive_root / "render"
    fixture_dir = out / "fixtures"

    for d in [data_home, state_home, archive_root, render_root]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["XDG_DATA_HOME"] = str(data_home)
    os.environ["XDG_STATE_HOME"] = str(state_home)
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    os.environ["POLYLOGUE_RENDER_ROOT"] = str(render_root)
    os.environ["POLYLOGUE_FORCE_PLAIN"] = "1"

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

    config = Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=sources,
    )

    result = asyncio.run(async_run_sources(
        config=config,
        stage="all",
        plan=None,
        ui=None,
        source_names=None,
    ))

    env_vars = {
        "XDG_DATA_HOME": str(data_home),
        "XDG_STATE_HOME": str(state_home),
        "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
        "POLYLOGUE_RENDER_ROOT": str(render_root),
        "POLYLOGUE_FORCE_PLAIN": "1",
    }

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


__all__ = ["demo_command"]

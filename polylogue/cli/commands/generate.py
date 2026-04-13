"""Generate synthetic conversations for demos, testing, and inspection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import click

from polylogue.cli.types import AppEnv
from polylogue.scenarios import CorpusRequest, CorpusSourceKind


@click.command("generate")
@click.option(
    "--provider",
    "-p",
    "providers",
    multiple=True,
    help="Providers to include (default: all). Can be repeated.",
)
@click.option("--count", "-n", default=3, show_default=True, help="Conversations per provider")
@click.option(
    "--corpus-source",
    type=click.Choice([kind.value for kind in CorpusSourceKind], case_sensitive=False),
    default=CorpusSourceKind.DEFAULT.value,
    show_default=True,
    help="Corpus spec source to execute.",
)
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None, help="Output directory")
@click.option("--seed", is_flag=True, help="Run pipeline to produce a usable demo environment")
@click.option("--env-only", is_flag=True, help="Print shell export statements only (requires --seed)")
@click.pass_obj
def generate_command(
    env: AppEnv,
    providers: tuple[str, ...],
    count: int,
    corpus_source: str,
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
    source_kind = CorpusSourceKind(corpus_source)

    if source_kind is CorpusSourceKind.DEFAULT:
        available = SyntheticCorpus.available_providers()
        selected = list(providers) if providers else available
        invalid = set(selected) - set(available)
        if invalid:
            raise click.BadParameter(
                f"Unknown provider(s): {', '.join(sorted(invalid))}. Available: {', '.join(available)}",
                param_hint="--provider",
            )
    else:
        selected = list(providers)

    if seed:
        _do_seed(env, selected, count, source_kind, output_dir, env_only)
    else:
        _do_corpus(env, selected, count, source_kind, output_dir)


def _do_corpus(
    env: AppEnv,
    providers: list[str],
    count: int,
    corpus_source: CorpusSourceKind,
    output_dir: Path | None,
) -> None:
    """Generate raw wire-format files for each provider."""
    from polylogue.showcase.workspace import (
        build_synthetic_corpus_scenarios,
        generate_synthetic_fixtures_from_scenarios,
    )

    out = output_dir or Path(tempfile.mkdtemp(prefix="polylogue-corpus-"))
    request = CorpusRequest(
        providers=tuple(providers) or None,
        source=corpus_source,
        count=count,
        style="default",
        messages_min=4,
        messages_max=15,
        seed=42,
    )
    scenarios = build_synthetic_corpus_scenarios(request=request)
    if not scenarios:
        raise click.BadParameter(
            "No corpus scenarios matched the selected source/providers.",
            param_hint="--corpus-source",
        )
    written_batches = generate_synthetic_fixtures_from_scenarios(out, corpus_scenarios=scenarios, prefix="sample")
    specs = tuple(spec for scenario in scenarios for spec in scenario.corpus_specs)

    for spec, written in zip(specs, written_batches, strict=True):
        provider_dir = out / spec.provider

        env.ui.console.print(
            f"  {spec.provider}:{spec.scope_label}: {written.batch.report.generated_count} files "
            f"({written.batch.report.element_kind or 'default'}) → {provider_dir}"
        )

    env.ui.console.print(f"\nCorpus written to: {out}")


def _do_seed(
    env: AppEnv,
    providers: list[str],
    count: int,
    corpus_source: CorpusSourceKind,
    output_dir: Path | None,
    env_only: bool,
) -> None:
    """Seed a full demo database via the pipeline."""
    from polylogue.showcase.workspace import (
        build_synthetic_corpus_scenarios,
        create_verification_workspace,
        seed_workspace_from_scenarios,
    )

    out = output_dir or Path(tempfile.mkdtemp(prefix="polylogue-demo-"))
    workspace = create_verification_workspace(out)
    request = CorpusRequest(
        providers=tuple(providers) or None,
        source=corpus_source,
        count=count,
        style="default",
        messages_min=6,
        messages_max=19,
        seed=42,
    )
    scenarios = build_synthetic_corpus_scenarios(request=request)
    if not scenarios:
        raise click.BadParameter(
            "No corpus scenarios matched the selected source/providers.",
            param_hint="--corpus-source",
        )
    result = seed_workspace_from_scenarios(workspace, corpus_scenarios=scenarios, prefix="demo")

    if env_only:
        for key, value in workspace.env_vars.items():
            click.echo(f'export {key}="{value}"')
    else:
        c = result.counts
        env.ui.console.print(f"Seeded {c.get('conversations', 0)} conversations, {c.get('messages', 0)} messages")
        env.ui.console.print(f"\nDemo environment: {out}")
        env.ui.console.print("\nTo use:")
        for key, value in workspace.env_vars.items():
            env.ui.console.print(f'  export {key}="{value}"')


__all__ = ["generate_command"]

"""Verification-lab corpus generation helpers."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from polylogue.scenarios import CorpusRequest, CorpusSourceKind


@dataclass(frozen=True, slots=True)
class GeneratedCorpusBatch:
    """One generated provider/spec batch."""

    provider: str
    scope_label: str
    element_kind: str
    generated_count: int
    provider_dir: Path


@dataclass(frozen=True, slots=True)
class GeneratedCorpusResult:
    """Result of writing raw synthetic corpus files."""

    output_dir: Path
    batches: tuple[GeneratedCorpusBatch, ...]


@dataclass(frozen=True, slots=True)
class SeededDemoResult:
    """Result of seeding a complete demo archive workspace."""

    output_dir: Path
    counts: dict[str, int]
    env_vars: dict[str, str]


def resolve_lab_corpus_providers(
    *,
    providers: tuple[str, ...],
    corpus_source: CorpusSourceKind,
) -> tuple[str, ...] | None:
    """Resolve provider filters for lab corpus commands."""
    if corpus_source is not CorpusSourceKind.DEFAULT:
        return providers or None

    from polylogue.schemas.synthetic import SyntheticCorpus

    available = tuple(SyntheticCorpus.available_providers())
    selected = providers or available
    invalid = set(selected) - set(available)
    if invalid:
        raise ValueError(f"Unknown provider(s): {', '.join(sorted(invalid))}. Available: {', '.join(available)}")
    return selected


def generate_lab_corpus(
    *,
    providers: tuple[str, ...],
    count: int,
    corpus_source: CorpusSourceKind,
    output_dir: Path | None,
) -> GeneratedCorpusResult:
    """Generate raw wire-format files for verification-lab scenarios."""
    from polylogue.showcase.workspace import (
        build_synthetic_corpus_scenarios,
        generate_synthetic_fixtures_from_scenarios,
    )

    out = output_dir or Path(tempfile.mkdtemp(prefix="polylogue-corpus-"))
    selected_providers = resolve_lab_corpus_providers(providers=providers, corpus_source=corpus_source)
    request = CorpusRequest(
        providers=selected_providers,
        source=corpus_source,
        count=count,
        style="default",
        messages_min=4,
        messages_max=15,
        seed=42,
    )
    scenarios = build_synthetic_corpus_scenarios(request=request)
    if not scenarios:
        raise ValueError("No corpus scenarios matched the selected source/providers.")

    written_batches = generate_synthetic_fixtures_from_scenarios(out, corpus_scenarios=scenarios, prefix="sample")
    specs = tuple(spec for scenario in scenarios for spec in scenario.corpus_specs)
    batches = tuple(
        GeneratedCorpusBatch(
            provider=spec.provider,
            scope_label=spec.scope_label,
            element_kind=written.batch.report.element_kind or "default",
            generated_count=written.batch.report.generated_count,
            provider_dir=out / spec.provider,
        )
        for spec, written in zip(specs, written_batches, strict=True)
    )
    return GeneratedCorpusResult(output_dir=out, batches=batches)


def seed_lab_demo(
    *,
    providers: tuple[str, ...],
    count: int,
    corpus_source: CorpusSourceKind,
    output_dir: Path | None,
) -> SeededDemoResult:
    """Seed a full demo archive workspace for verification-lab scenarios."""
    from polylogue.showcase.workspace import (
        build_synthetic_corpus_scenarios,
        create_verification_workspace,
        seed_workspace_from_scenarios,
    )

    out = output_dir or Path(tempfile.mkdtemp(prefix="polylogue-demo-"))
    workspace = create_verification_workspace(out)
    selected_providers = resolve_lab_corpus_providers(providers=providers, corpus_source=corpus_source)
    request = CorpusRequest(
        providers=selected_providers,
        source=corpus_source,
        count=count,
        style="default",
        messages_min=6,
        messages_max=19,
        seed=42,
    )
    scenarios = build_synthetic_corpus_scenarios(request=request)
    if not scenarios:
        raise ValueError("No corpus scenarios matched the selected source/providers.")

    result = seed_workspace_from_scenarios(workspace, corpus_scenarios=scenarios, prefix="demo")
    return SeededDemoResult(
        output_dir=out,
        counts=dict(result.counts),
        env_vars=dict(workspace.env_vars),
    )


__all__ = [
    "GeneratedCorpusBatch",
    "GeneratedCorpusResult",
    "SeededDemoResult",
    "generate_lab_corpus",
    "resolve_lab_corpus_providers",
    "seed_lab_demo",
]

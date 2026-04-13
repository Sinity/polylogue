from __future__ import annotations

from devtools.large_archive_generator import ArchiveSpec, ScaleLevel


def _build_archive_spec() -> ArchiveSpec:
    return ArchiveSpec(
        level=ScaleLevel.SMALL,
        provider_mix={"chatgpt": 0.75, "codex": 0.25, "missing": 1.0},
        message_count=1_000,
        conversations=8,
        avg_messages_per_conv=10,
        content_blocks_ratio=0.3,
        seed=9,
    )


def test_archive_spec_compiles_per_provider_corpus_scenarios() -> None:
    spec = _build_archive_spec()

    corpus_scenarios = spec.corpus_scenarios(available_providers={"chatgpt", "codex"})

    assert tuple(scenario.provider for scenario in corpus_scenarios) == ("chatgpt", "codex")
    assert all(scenario.origin == "compiled.large-archive-scenario" for scenario in corpus_scenarios)
    assert all("scenario" in scenario.tags for scenario in corpus_scenarios)
    assert sum(len(scenario.corpus_specs) for scenario in corpus_scenarios) == 2


def test_archive_spec_compiles_per_provider_corpus_specs() -> None:
    spec = _build_archive_spec()

    corpus_specs = spec.corpus_specs(available_providers={"chatgpt", "codex"})

    assert tuple(corpus.provider for corpus in corpus_specs) == ("chatgpt", "codex")
    assert sum(corpus.count for corpus in corpus_specs) == 8
    assert all(corpus.messages_min >= 2 for corpus in corpus_specs)
    assert all(corpus.seed == 9 for corpus in corpus_specs)
    assert all(corpus.origin == "generated.large-archive" for corpus in corpus_specs)


def test_archive_spec_flattens_scenario_compilation_into_corpus_specs() -> None:
    spec = _build_archive_spec()

    corpus_specs = spec.corpus_specs(available_providers={"chatgpt", "codex"})
    corpus_scenarios = spec.corpus_scenarios(available_providers={"chatgpt", "codex"})

    assert corpus_specs == tuple(spec for scenario in corpus_scenarios for spec in scenario.corpus_specs)

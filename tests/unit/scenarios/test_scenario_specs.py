from __future__ import annotations

from typing import cast

from polylogue.scenarios import CorpusSpec, ScenarioProjectionSourceKind, ScenarioSpec


class _ScenarioFixture(ScenarioSpec):
    @property
    def projection_source_kind(self) -> ScenarioProjectionSourceKind:
        return ScenarioProjectionSourceKind.INFERRED_CORPUS_SCENARIO

    @property
    def projection_name(self) -> str:
        return "fixture"

    @property
    def projection_description(self) -> str:
        return "fixture scenario"


def test_scenario_spec_projection_payload_includes_corpus_specs() -> None:
    scenario = _ScenarioFixture(
        corpus_specs=(
            CorpusSpec.for_provider(
                "chatgpt",
                count=2,
                messages_min=4,
                messages_max=6,
                seed=7,
                origin="generated.test-suite",
                tags=("synthetic", "test"),
            ),
        ),
        origin="compiled.synthetic-corpus-scenario",
        tags=("synthetic", "scenario"),
    )

    projection = scenario.to_projection_entry()

    assert projection.source_kind is ScenarioProjectionSourceKind.INFERRED_CORPUS_SCENARIO
    assert projection.name == "fixture"
    assert projection.description == "fixture scenario"
    assert projection.origin == "compiled.synthetic-corpus-scenario"
    source_payload = projection.source_payload
    corpus_specs = cast(list[dict[str, object]], source_payload["corpus_specs"])
    assert corpus_specs[0]["provider"] == "chatgpt"
    assert projection.tags == ("synthetic", "scenario")


def test_scenario_spec_tracks_unique_corpus_providers() -> None:
    scenario = _ScenarioFixture(
        corpus_specs=(
            CorpusSpec.for_provider("chatgpt"),
            CorpusSpec.for_provider("chatgpt", package_version="v2"),
            CorpusSpec.for_provider("codex"),
        )
    )

    assert scenario.has_corpus_specs is True
    assert scenario.corpus_providers() == ("chatgpt", "codex")

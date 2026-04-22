"""Shared scenario-bearing models above metadata and projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from polylogue.authored_payloads import PayloadDict, PayloadMap

from .metadata import ScenarioMetadata
from .projections import ScenarioProjectionSource

if TYPE_CHECKING:
    from .corpus import CorpusSpec


@dataclass(frozen=True, kw_only=True)
class ScenarioSpec(ScenarioProjectionSource, ScenarioMetadata):
    """Shared scenario-bearing base with optional compiled corpus bundles."""

    corpus_specs: tuple[CorpusSpec, ...] = ()

    @property
    def has_corpus_specs(self) -> bool:
        return bool(self.corpus_specs)

    def corpus_providers(self) -> tuple[str, ...]:
        seen: set[str] = set()
        providers: list[str] = []
        for spec in self.corpus_specs:
            provider = spec.provider
            if provider not in seen:
                seen.add(provider)
                providers.append(provider)
        return tuple(providers)

    def scenario_payload(self) -> PayloadDict:
        payload = self.to_payload()
        if self.corpus_specs:
            payload["corpus_specs"] = [spec.to_payload() for spec in self.corpus_specs]
        return payload

    def projection_source_payload(self) -> PayloadMap:
        return self.scenario_payload()


__all__ = ["ScenarioSpec"]

"""Roundtrip suite orchestration for schema-and-evidence proofing."""

from __future__ import annotations

from polylogue.schemas.roundtrip_models import RoundtripProofSuiteReport
from polylogue.schemas.roundtrip_provider import _prove_provider_roundtrip
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.sync_bridge import run_coroutine_sync


async def _prove_roundtrip_suite_async(
    *,
    providers: list[str] | None = None,
    count: int = 1,
    style: str = "default",
    seed: int = 42,
) -> RoundtripProofSuiteReport:
    provider_list = providers or SyntheticCorpus.available_providers()
    if not provider_list:
        raise ValueError("No providers available for schema roundtrip proof")
    if count <= 0:
        raise ValueError("Roundtrip proof count must be positive")

    reports = {}
    for index, provider in enumerate(provider_list):
        reports[provider] = await _prove_provider_roundtrip(
            provider,
            count=count,
            style=style,
            seed=seed + index,
        )
    return RoundtripProofSuiteReport(provider_reports=reports)


def prove_schema_evidence_roundtrip_suite(
    *,
    providers: list[str] | None = None,
    count: int = 1,
    style: str = "default",
    seed: int = 42,
) -> RoundtripProofSuiteReport:
    return run_coroutine_sync(
        _prove_roundtrip_suite_async(
            providers=providers,
            count=count,
            style=style,
            seed=seed,
        )
    )


__all__ = ["_prove_roundtrip_suite_async", "prove_schema_evidence_roundtrip_suite"]

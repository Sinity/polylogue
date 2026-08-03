"""Regression test for polylogue-tu1f: gemini/aistudio-drive schema-catalog
identity backfill.

Live ``ops.db`` schema_drift_samples showed 100% of ``aistudio-drive``
records since 2026-07-01 classifying as ``unseen_shape``
(``polylogue.schemas.drift_sentinel.UNSEEN_SHAPE``). The committed
``session_document`` schema for the gemini provider (both v1 and v2)
already modeled the real ``{chunkedPrompt, runSettings, systemInstruction,
applets, citations}`` wire shape correctly -- the actual defect was that
neither package's element carried any ``exact_structure_ids`` or
``profile_tokens`` identity evidence (both were empty lists in the
committed ``catalog.json``/``package.json``/schema ``x-polylogue-*``
annotations), so ``SchemaRegistry.resolve_payload`` could never find a
real-candidate match (``exact_structure``/``bundle_scope``/
``profile_family`` -- see ``runtime_registry._resolve_observation``) and
always fell back to ``package_default``, which
``polylogue.schemas.drift_sentinel.classify_schema_drift`` reports as
``unseen_shape``.

These tests exercise the DEFAULT (bundled, no storage_root override)
``SchemaRegistry`` -- i.e. the exact registry
``polylogue.pipeline.services.ingest_worker._runtime_schema_registry()``
uses in production -- against a gemini/AI-Studio payload shaped exactly
like real cached Drive exports (a bare ``{chunkedPrompt, runSettings,
systemInstruction}`` document, no top-level ``id``/``title``/
``createTime``), to prove the committed catalog now resolves it as a real
candidate instead of ``package_default``.
"""

from __future__ import annotations

from polylogue.schemas.drift_sentinel import UNSEEN_SHAPE, classify_schema_drift
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.validator import validate_provider_export


def _real_shaped_gemini_payload(*, message_text: str) -> dict[str, object]:
    """A payload shaped like a real AI Studio / Drive cached export.

    Deliberately carries no top-level ``id``/``title``/``createTime`` --
    real cached ``chunkedPrompt`` documents observed at
    ``~/.local/share/polylogue/drive-cache/gemini/*.json`` are a bare
    ``{chunkedPrompt, runSettings, systemInstruction}`` object.
    """
    return {
        "runSettings": {
            "temperature": 1.0,
            "model": "models/gemini-2.5-pro",
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": 8192,
            "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}],
            "enableCodeExecution": True,
        },
        "systemInstruction": {},
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": message_text, "tokenCount": 5},
                {"role": "model", "text": "A reply with content the catalog never saw before.", "tokenCount": 9},
            ],
        },
    }


def test_gemini_catalog_elements_carry_real_identity_evidence() -> None:
    """Both committed gemini packages must carry non-empty identity evidence.

    Before the polylogue-tu1f backfill, ``exact_structure_ids`` and
    ``profile_tokens`` were empty lists on every element of every gemini
    package -- the concrete, on-disk symptom of the unseen_shape defect.
    """
    registry = SchemaRegistry()
    catalog = registry.load_package_catalog("gemini")
    assert catalog is not None, "gemini schema catalog must be committed"
    assert catalog.packages, "gemini catalog must have at least one package"

    for package in catalog.packages:
        for element in package.elements:
            assert element.exact_structure_ids, (
                f"gemini {package.version}/{element.element_kind} has no exact_structure_ids evidence"
            )
            assert element.profile_tokens, (
                f"gemini {package.version}/{element.element_kind} has no profile_tokens evidence"
            )


def test_real_shaped_gemini_payload_resolves_to_a_real_candidate_not_unseen_shape() -> None:
    """The production resolution path must not fall back to package_default.

    Uses the DEFAULT registry (no storage_root override) -- the same one
    ``ingest_worker._runtime_schema_registry()`` resolves payloads through
    in production -- against a payload whose message content never appeared
    in any generation/fixture corpus, proving the match generalizes via
    profile-family evidence rather than an exact-content coincidence.
    """
    registry = SchemaRegistry()
    payload = _real_shaped_gemini_payload(message_text="Inspect this never-before-seen fixture body.")

    resolution = registry.resolve_payload("gemini", payload)

    assert resolution is not None
    assert resolution.reason != "package_default", (
        "gemini payload with the real chunkedPrompt/runSettings/systemInstruction shape "
        "must not fall back to package_default (the unseen_shape resolution reason)"
    )


def test_real_shaped_gemini_payload_does_not_classify_as_unseen_shape() -> None:
    """End-to-end through the same classification helper the live drift
    health check (``polylogue.daemon.health._check_schema_drift_medium``)
    and ``schema_drift_samples`` writer consume."""
    registry = SchemaRegistry()
    payload = _real_shaped_gemini_payload(message_text="Another distinct, never-fixtured conversation turn.")

    resolution = registry.resolve_payload("gemini", payload)
    assert resolution is not None

    validation = validate_provider_export(payload, "gemini", strict=False)

    classification = classify_schema_drift(
        resolution_reason=resolution.reason,
        is_valid=validation.is_valid,
        drift_warnings=validation.drift_warnings,
    )

    assert classification != UNSEEN_SHAPE


def test_payload_with_top_level_applets_and_citations_still_resolves() -> None:
    """Real exports occasionally carry top-level ``applets``/``citations``
    (already modeled in the committed schema's ``properties``) -- these
    must not push resolution back to package_default either."""
    registry = SchemaRegistry()
    payload = _real_shaped_gemini_payload(message_text="Third distinct body, with attachments this time.")
    payload["applets"] = [{"name": "code-runner", "description": "Runs sandboxed code."}]
    payload["citations"] = []

    resolution = registry.resolve_payload("gemini", payload)

    assert resolution is not None
    assert resolution.reason != "package_default"

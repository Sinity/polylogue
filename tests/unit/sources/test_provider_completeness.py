from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from polylogue.core.enums import Origin
from polylogue.core.provider_identity import CORE_SCHEMA_PROVIDERS, canonical_schema_provider
from polylogue.core.schema_subjects import (
    CORE_SCHEMA_ORIGINS,
    INFERENCE_EXCLUDED_SUBJECTS,
    SCHEMA_PACKAGE_DIRECTORIES,
    SCHEMA_SUBJECTS,
)
from polylogue.schemas.registry import SCHEMA_DIR, schema_subject_diagnostics
from polylogue.sources import provider_completeness as module
from polylogue.sources.origin_specs import ORIGIN_SPECS
from polylogue.sources.provider_completeness import (
    accepted_blockers,
    provider_package_completeness,
)


def test_provider_completeness_reports_representative_modes() -> None:
    report = provider_package_completeness()
    by_ref = {row.package_ref: row for row in report.rows}

    codex = by_ref["provider-package:codex-session/session-jsonl@v1"]
    chatgpt = by_ref["provider-package:chatgpt-export/takeout-json@v1"]
    browser = by_ref["provider-package:browser-capture/live-receiver@v1"]
    hermes = by_ref["provider-package:hermes-session/state-db@v1"]
    grok = by_ref["provider-package:grok-export/export-json@v1"]

    assert codex.origin == "codex-session"
    assert codex.capture_mode == "session-jsonl"
    assert codex.detector.status == "complete"
    assert codex.parser.owner_path == "polylogue/sources/parsers/codex.py"
    assert codex.import_explain.status == "complete"

    assert chatgpt.origin == "chatgpt-export"
    assert chatgpt.schema_package.status == "complete"

    # Browser capture is a first-party, Polylogue-controlled envelope (not an
    # inferred third-party export shape): the receiver enforces
    # polylogue.browser_capture.models.BrowserCaptureEnvelope on every accepted
    # capture, so that authored model is the authoritative wire schema and the
    # row's schema evidence. The subject is declared outside the
    # schema-inference denominator and has no committed inferred package
    # (polylogue-cfz6, polylogue-tnqqt AC3).
    assert browser.maturity == "accepted"
    assert browser.status == "complete"
    assert browser.schema_package.owner_path == "polylogue/browser_capture/models.py"
    assert browser.schema_package.status == "complete"
    assert not browser.blockers

    assert hermes.schema_package.owner_path == ("polylogue/schemas/providers/hermes/state_db_v16.contract.json")
    assert hermes.schema_package.status == "complete"
    assert grok.maturity == "accepted"
    assert grok.status == "complete"
    assert grok.parser.status == "complete"
    assert grok.schema_package.status == "complete"
    assert not grok.blockers


def test_provider_completeness_is_a_projection_of_every_origin_spec() -> None:
    report = provider_package_completeness()

    assert {Origin.from_string(row.origin) for row in report.rows} == set(Origin)
    assert {(origin, mode.package_ref) for origin, mode in module.PACKAGE_MODE_SPECS} == {
        (spec.origin, mode.package_ref) for spec in ORIGIN_SPECS for mode in spec.completeness_modes
    }


def test_provider_completeness_origin_filter_accepts_origin_and_provider() -> None:
    by_origin = provider_package_completeness(origin="codex-session")
    by_provider = provider_package_completeness(origin="codex")

    assert [row.package_ref for row in by_origin.rows] == ["provider-package:codex-session/session-jsonl@v1"]
    assert [row.package_ref for row in by_provider.rows] == ["provider-package:codex-session/session-jsonl@v1"]


def test_provider_completeness_json_payload_shape() -> None:
    report = provider_package_completeness(origin="codex-session")
    payload = json.loads(report.to_json())

    assert payload["mode"] == "provider-package-completeness"
    assert payload["totals"]["total"] == 1
    assert payload["rows"][0]["query_units"]["status"] == "complete"
    assert payload["rows"][0]["provider_wire"] == "codex"


def test_provider_completeness_check_blocks_accepted_missing_required_item(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    report = provider_package_completeness(origin="codex-session")

    assert report.rows[0].maturity == "accepted"
    assert report.rows[0].blockers
    assert accepted_blockers(report)


def test_schema_subject_declaration_reaches_every_package_and_origin() -> None:
    """The one subject declaration is the executable vocabulary oracle."""
    assert (
        tuple(item.token for item in SCHEMA_SUBJECTS if item.provider is not None and item.requires_package)
        == CORE_SCHEMA_PROVIDERS
    )
    assert {item.package_dir for item in SCHEMA_SUBJECTS if item.requires_package} == set(SCHEMA_PACKAGE_DIRECTORIES)
    assert {path.name for path in SCHEMA_DIR.iterdir() if path.is_dir() and (path / "catalog.json").exists()} == set(
        SCHEMA_PACKAGE_DIRECTORIES
    )
    assert all(canonical_schema_provider(item.token) == item.token for item in SCHEMA_SUBJECTS)
    assert set(CORE_SCHEMA_ORIGINS) == {origin for item in SCHEMA_SUBJECTS for origin in item.origins}
    assert schema_subject_diagnostics() == ()


def test_inference_excluded_subject_has_no_package_to_refresh() -> None:
    """A subject declared outside the inference denominator is one whose wire
    format this repository authors. An inferred package for it would be a
    snapshot of our own model presented as an observation, and its presence is
    precisely what invites someone to "refresh" a subject that must never be
    inferred.

    Anti-vacuity: restore ``polylogue/schemas/providers/browser-capture/`` (or
    flip its ``requires_package``) and this goes red.
    """
    assert INFERENCE_EXCLUDED_SUBJECTS, "no subject declares an inference exclusion"
    for item in SCHEMA_SUBJECTS:
        if item.inference_excluded_reason is None:
            continue
        assert not item.requires_package, f"{item.token} is excluded from inference but still requires a package"
        assert not (SCHEMA_DIR / item.package_dir).exists(), (
            f"{item.token} is excluded from inference but a committed package remains"
        )


def test_schema_subject_diagnostics_catches_added_or_removed_package(tmp_path: Path) -> None:
    """An executable package-set mutation cannot self-authorize completeness."""
    for item in SCHEMA_SUBJECTS:
        if item.requires_package:
            (tmp_path / item.package_dir).mkdir()
            (tmp_path / item.package_dir / "catalog.json").write_text("{}", encoding="utf-8")
    (tmp_path / "unexpected").mkdir()
    (tmp_path / "unexpected" / "catalog.json").write_text("{}", encoding="utf-8")

    diagnostics = schema_subject_diagnostics(tmp_path)
    assert "undeclared schema package directory: unexpected" in diagnostics

    (tmp_path / "grok" / "catalog.json").unlink()
    assert "declared schema package is missing: grok" in schema_subject_diagnostics(tmp_path)

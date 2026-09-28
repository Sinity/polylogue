"""``schema commit`` publishes only packages that pass the promotion audit (polylogue-mdlft)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.schemas.operator import commit
from polylogue.schemas.promotion_audit import audit_schema_artifacts

_LEAKED_ID = "3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b"


def _element(annotation_key: str) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "x-polylogue-observed-distribution": {
                    "documents": 3,
                    "co_occurring_fields": {annotation_key: 3},
                },
            }
        },
    }


def _write_tree(root: Path, annotation_key: str) -> None:
    element = root / "versions" / "v1" / "elements" / "session_document.json"
    element.parent.mkdir(parents=True, exist_ok=True)
    element.write_text(json.dumps(_element(annotation_key)))


def test_nested_annotation_keys_are_held_to_the_property_name_bar(tmp_path: Path) -> None:
    """Anti-vacuity: walk only ``x-polylogue-values`` again and the leaked id publishes."""
    _write_tree(tmp_path / "leaky", _LEAKED_ID)
    _write_tree(tmp_path / "clean", "created_at")

    leaky = audit_schema_artifacts(tmp_path / "leaky")
    clean = audit_schema_artifacts(tmp_path / "clean")

    assert [finding.category for finding in leaky.blockers] == ["unsafe_annotation_key"]
    assert _LEAKED_ID not in json.dumps(leaky.to_payload())
    assert clean.blockers == ()


def test_commit_restores_the_prior_tree_when_the_written_package_fails_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: persist without the post-write audit and the leaky package stays published."""
    output_dir = tmp_path / "providers"
    _write_tree(output_dir / "chatgpt", "created_at")
    prior = (output_dir / "chatgpt" / "versions" / "v1" / "elements" / "session_document.json").read_text()

    def leaky_persist(root: Path, provider: str, _bundle: object) -> None:
        _write_tree(root / provider, _LEAKED_ID)

    monkeypatch.setattr(commit, "persist_generated_provider_bundle", leaky_persist)

    with pytest.raises(commit.SchemaCommitAuditError) as refused:
        commit._persist_audited(output_dir, "chatgpt", object())  # type: ignore[arg-type]

    assert [item.category for item in refused.value.blockers] == ["unsafe_annotation_key"]
    restored = output_dir / "chatgpt" / "versions" / "v1" / "elements" / "session_document.json"
    assert restored.read_text() == prior


def test_commit_keeps_a_package_that_passes_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "providers"

    def clean_persist(root: Path, provider: str, _bundle: object) -> None:
        _write_tree(root / provider, "updated_at")

    monkeypatch.setattr(commit, "persist_generated_provider_bundle", clean_persist)

    commit._persist_audited(output_dir, "chatgpt", object())  # type: ignore[arg-type]

    assert (output_dir / "chatgpt" / "versions" / "v1" / "elements" / "session_document.json").exists()


def test_high_entropy_annotation_value_is_a_blocker(tmp_path: Path) -> None:
    """Anti-vacuity: drop the entropy predicate for annotation values and this publishes."""
    element = tmp_path / "versions" / "v1" / "elements" / "session_document.json"
    element.parent.mkdir(parents=True)
    element.write_text(
        json.dumps(
            {
                "type": "object",
                "x-polylogue-observed-distribution": {"sample": "abc123XYZ987mnop"},
                "x-polylogue-mutually-exclusive": [{"fields": ["content_sha256", "inline_base64"], "parent": "$"}],
            }
        )
    )

    report = audit_schema_artifacts(tmp_path)

    assert [finding.category for finding in report.blockers] == ["unsafe_annotation_value"]
    assert "sample" in report.blockers[0].json_path


def test_a_secret_used_as_an_annotation_key_never_appears_in_the_finding_path(tmp_path: Path) -> None:
    """Anti-vacuity: build the key path before deciding secret status and the key publishes in ``json_path``."""
    secret_key = "sk-ant-api03-" + "b" * 40
    _write_tree(tmp_path, secret_key)

    report = audit_schema_artifacts(tmp_path)

    assert "anthropic_api_key" in {finding.category for finding in report.blockers}
    assert secret_key not in json.dumps(report.to_payload())


def test_commit_removes_a_first_publication_that_fails_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: skip rollback when no prior tree existed and the leaky package stays published."""
    output_dir = tmp_path / "providers"

    def leaky_persist(root: Path, provider: str, _bundle: object) -> None:
        _write_tree(root / provider, _LEAKED_ID)

    monkeypatch.setattr(commit, "persist_generated_provider_bundle", leaky_persist)

    with pytest.raises(commit.SchemaCommitAuditError):
        commit._persist_audited(output_dir, "chatgpt", object())  # type: ignore[arg-type]

    assert list(output_dir.iterdir()) == []


def test_commit_carries_the_live_providers_history_through_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity (Codex P1, #5704): persist into an empty stage and the
    merging persistence sees no prior catalog, so publishing the stage drops
    the provider's historical ``v1`` package."""
    output_dir = tmp_path / "providers"
    _write_tree(output_dir / "chatgpt", "created_at")
    prior_seen: list[bool] = []

    def merging_persist(root: Path, provider: str, _bundle: object) -> None:
        prior_seen.append((root / provider / "versions" / "v1").exists())
        element = root / provider / "versions" / "v2" / "elements" / "session_document.json"
        element.parent.mkdir(parents=True, exist_ok=True)
        element.write_text(json.dumps(_element("updated_at")))

    monkeypatch.setattr(commit, "persist_generated_provider_bundle", merging_persist)

    commit._persist_audited(output_dir, "chatgpt", object())  # type: ignore[arg-type]

    assert prior_seen == [True]
    versions = output_dir / "chatgpt" / "versions"
    assert sorted(path.name for path in versions.iterdir()) == ["v1", "v2"]


def test_an_audited_commit_retires_the_live_legacy_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The live ``<provider>.schema.json`` is removed once the audited tree is published.

    Anti-vacuity (Codex P2, #5704): leave legacy cleanup to the persistence
    step, which runs in the staging root, and the live legacy file survives.
    """
    output_dir = tmp_path / "providers"
    output_dir.mkdir()
    legacy = output_dir / "chatgpt.schema.json"
    legacy.write_text("{}", encoding="utf-8")

    def clean_persist(root: Path, provider: str, _bundle: object) -> None:
        _write_tree(root / provider, "updated_at")

    monkeypatch.setattr(commit, "persist_generated_provider_bundle", clean_persist)

    commit._persist_audited(output_dir, "chatgpt", object())  # type: ignore[arg-type]

    assert not legacy.exists()

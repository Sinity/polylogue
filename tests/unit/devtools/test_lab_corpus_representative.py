"""Tests for representative corpus generate and verify commands."""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.proof.corpus import CorpusManifest, representatives_dir


def test_corpus_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = CorpusManifest(
        provider="chatgpt",
        schema_version="v1",
        generator_command="devtools lab-corpus representative-generate -p chatgpt",
        generator_version="0.1.0",
        seed=42,
        source_mode="schema-only",
        sample_count=3,
        privacy_status="auto-generated-safe",
    )
    path = tmp_path / "corpus-manifest.json"
    manifest.write(path)
    loaded = CorpusManifest.from_path(path)
    assert loaded.provider == "chatgpt"
    assert loaded.schema_version == "v1"
    assert loaded.sample_count == 3
    assert loaded.seed == 42
    assert loaded.source_mode == "schema-only"
    assert loaded.privacy_status == "auto-generated-safe"


def test_representatives_dir_returns_schemas_providers_path() -> None:
    rep_dir = representatives_dir("chatgpt")
    assert "schemas" in str(rep_dir)
    assert "providers" in str(rep_dir)
    assert rep_dir.name == "representatives"
    assert rep_dir.parent.name == "chatgpt"


def test_representative_generate_produces_valid_json(tmp_path: Path) -> None:
    """Integration: generate representative corpus to tmp, verify output."""
    from devtools.lab_corpus import _representative_generate

    exit_code = _representative_generate(
        providers=("chatgpt",),
        count=1,
        seed=99,
        output_dir=tmp_path,
    )
    assert exit_code == 0

    rep_dir = tmp_path / "chatgpt"
    assert rep_dir.is_dir()
    manifest_path = rep_dir / "corpus-manifest.json"
    assert manifest_path.exists()

    manifest = CorpusManifest.from_path(manifest_path)
    assert manifest.sample_count == 1

    samples = sorted(rep_dir.glob("sample-*.json"))
    samples = [s for s in samples if s.name != "corpus-manifest.json"]
    assert len(samples) == 1
    data = json.loads(samples[0].read_text(encoding="utf-8"))
    assert isinstance(data, (dict, list))

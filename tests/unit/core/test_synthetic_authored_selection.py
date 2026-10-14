"""Curated fixture operands use the same original schema/profile driver."""

from __future__ import annotations

import base64
import json
import random
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.schemas.synthetic import SyntheticCorpus, WireFormat
from polylogue.schemas.synthetic.authored_selection import authored_schema_selection
from polylogue.schemas.synthetic.models import SchemaRecord
from polylogue.sources.dispatch import parse_payload


@pytest.mark.parametrize("provider", ["claude-code", "codex"])
@pytest.mark.parametrize("style", ["tool-heavy", "demo-tool-heavy"])
def test_authored_tools_parse_and_retain_original_fixture_custody(provider: str, style: str, tmp_path: Path) -> None:
    original_schema: SchemaRecord = {"type": "object", "properties": {"untouched": {"type": "boolean"}}}
    corpus = SyntheticCorpus(original_schema, WireFormat(encoding="jsonl"), provider)
    for seed in (7, 71, 203):
        batch = corpus.generate_batch(
            count=1, messages_per_session=range(8, 9), seed=seed, style=style, session_native_ids=(f"neutral-{seed}",)
        )
        selection = authored_schema_selection(provider, style)
        assert selection.workload_profile is not None
        assert selection.workload_profile["profile_kind"] == "authored-fixture"
        assert batch.report.workload_profile_id == selection.workload_profile["profile_id"]
        assert batch.report.generated_count == 1
        assert batch.artifacts[0].facts.tool_use_ids
        assert batch.artifacts[0].facts.paired_tool_ids
        records = [json.loads(line) for line in batch.raw_items[0].splitlines()]
        if provider == "claude-code":
            assert all(record["sessionId"] == f"neutral-{seed}" for record in records)
        sessions = parse_payload(Provider(provider), records, f"authored-{seed}")
        assert sessions
        assert any(session.messages for session in sessions)
        blocks = [
            block
            for record in records
            for block in record.get("message", record.get("payload", record)).get("content", [])
        ]
        assert any(block.get("type") == "tool_result" for block in blocks)
        assert all(
            block["tool_use_id"] in batch.artifacts[0].facts.tool_use_ids
            for block in blocks
            if block.get("type") == "tool_result"
        )
        if style == "demo-tool-heavy":
            assert any("AssertionError" in str(block.get("content", "")) for block in blocks)
        assert corpus.schema is original_schema
        assert corpus.workload_profile is None
    # File and explicitly collected fixtures have the exact same producer.
    baseline = corpus.generate_batch(count=1, messages_per_session=range(8, 9), seed=7, style=style)
    from polylogue.scenarios import CorpusSpec

    spec = CorpusSpec.for_provider(provider, count=1, messages_min=8, messages_max=8, seed=7, style=style)
    written = SyntheticCorpus.write_spec_artifacts(spec, tmp_path, prefix="authored")
    assert written.session_files[0].read_bytes() == baseline.raw_items[0]


def test_authored_attachment_is_declared_and_lossless_before_document_fields() -> None:
    selection = authored_schema_selection("gemini", "demo-attachments")
    corpus = SyntheticCorpus.from_selection(selection)
    batch = corpus.generate_batch(count=1, messages_per_session=range(4, 5), seed=71, style="demo-attachments")
    document = json.loads(batch.raw_items[0])
    chunks = document["chunkedPrompt"]["chunks"]
    assert len(chunks) == 4
    attachments = [chunk["inlineFile"] for chunk in chunks if "inlineFile" in chunk]
    assert len(attachments) == 1
    assert all(
        base64.b64decode(item["data"]) == b"demo attachment bytes for Polylogue fixture coverage\n"
        for item in attachments
    )
    assert all("inlineFile" not in chunk for chunk in chunks if chunk["role"] == "model")
    assert parse_payload(Provider.GEMINI, document, "authored-attachment")


def test_authored_demo_document_binds_native_identity_before_original_field_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.scenarios import CorpusSpec
    from polylogue.schemas.synthetic.demo_themes import _DEMO_THEMES

    original_schema: SchemaRecord = {"type": "object", "properties": {"untouched": {"type": "boolean"}}}
    corpus = SyntheticCorpus(original_schema, WireFormat(encoding="json"), "chatgpt")
    original_draw = corpus._generate_from_schema
    reached: list[set[str] | None] = []

    def observe_draw(
        schema: SchemaRecord, rng: random.Random, *, skip_keys: set[str] | None = None, path: str = "$"
    ) -> object:
        properties = schema.get("properties")
        if isinstance(properties, dict) and "mapping" in properties:
            reached.append(skip_keys)
        return original_draw(schema, rng, skip_keys=skip_keys, path=path)

    monkeypatch.setattr(corpus, "_generate_from_schema", observe_draw)
    batch = corpus.generate_batch(
        count=1, messages_per_session=range(12, 13), seed=71, style="demo", session_native_ids=("neutral-document",)
    )
    assert reached and all(fields is not None and "id" in fields for fields in reached)
    document = json.loads(batch.raw_items[0])
    assert document["id"] == "neutral-document"
    nodes = list(document["mapping"].values())
    assert len(nodes) == 12
    assert nodes[0]["parent"] is None
    assert all(node["parent"] in document["mapping"] for node in nodes[1:])
    assert all(child in document["mapping"] for node in nodes for child in node["children"])
    selection = authored_schema_selection("chatgpt", "demo", session_native_ids=("neutral-document",), theme_index=1)
    assert selection.workload_profile is not None and selection.workload_profile["profile_kind"] == "authored-fixture"
    # The seeded public theme is declared before sampling; authored prose is
    # preserved rather than appended to an unrelated measured operator draw.
    theme = _DEMO_THEMES[random.Random(71).randrange(len(_DEMO_THEMES))]
    for index, node in enumerate(nodes):
        text = (theme.assistant_turns if index % 2 else theme.user_turns)[(index // 2) % len(theme.user_turns)]
        assert node["message"]["content"]["parts"] == [text]
    sessions = parse_payload(Provider.CHATGPT, document, "authored-document")
    assert sessions and len(sessions[0].messages) == 12
    assert corpus.schema is original_schema and corpus.workload_profile is None
    spec = CorpusSpec.for_provider(
        "chatgpt",
        count=1,
        messages_min=12,
        messages_max=12,
        seed=71,
        style="demo",
        session_native_ids=("neutral-document",),
    )
    written = SyntheticCorpus.write_spec_artifacts(spec, tmp_path, prefix="authored")
    assert written.session_files[0].read_bytes() == batch.raw_items[0]

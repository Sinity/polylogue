"""Per-provider distribution evidence from real neutral inference/publication."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from polylogue.schemas.audit.workflow import audit_schema_bundle_privacy
from polylogue.schemas.field_stats.distributions import _bucket_index
from polylogue.schemas.generation.workflow import build_provider_bundle_from_sources, persist_generated_provider_bundle
from polylogue.schemas.registry import SchemaRegistry
from polylogue.schemas.source_inference import SchemaSourceInput
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.selection import select_synthetic_schema
from polylogue.sources.dispatch import parse_payload
from tests.infra.qualified_schema_registry import _neutral_sources


@pytest.mark.parametrize("provider", ("chatgpt", "claude-code", "gemini"))
def test_published_provider_body_distribution_matches_sampled_driver_within_tolerance(
    tmp_path: Path, provider: str
) -> None:
    sources = _neutral_sources(provider)
    original = next(source for source in sources if source.expected.filename in {"strings.jsonl", "conversation.json"})
    source_root = tmp_path / "sources"
    source_root.mkdir()
    for index in range(16):
        text = json.dumps(original.payload)
        # Authored source population is deliberately bimodal and skewed.
        # Counts and annotations are produced by inference, never inserted.
        length = 63 if index < 12 else 1023
        for body in original.expected.bodies:
            text = text.replace(json.dumps(body), json.dumps("x" * length))
        for slot, old in enumerate(
            (
                "11111111-1111-4111-8111-111111111111",
                "33333333-3333-4333-8333-333333333333",
                "44444444-4444-4444-8444-444444444444",
            )
        ):
            text = text.replace(old, f"{index + 1:08d}-1111-4111-8111-{slot + 1:012d}")
        payload = json.loads(text)
        if isinstance(payload, list):
            text = "".join(json.dumps(row) + "\n" for row in payload)
        (source_root / f"neutral-{index}.{'jsonl' if isinstance(payload, list) else 'json'}").write_text(text)
    bundle = build_provider_bundle_from_sources(
        provider,
        source_inputs=(SchemaSourceInput(provider, source_root),),
        cache_path=tmp_path / "evidence.sqlite3",
        max_workers=1,
        privacy_config=None,
        prior_catalog=None,
    )
    assert bundle.result.success, bundle.result.error
    registry_root = tmp_path / "registry"
    persist_generated_provider_bundle(registry_root, provider, bundle)
    registry = SchemaRegistry(storage_root=registry_root)
    assert audit_schema_bundle_privacy(registry=registry).failed == 0
    selection = select_synthetic_schema(provider, registry_factory=lambda: registry)
    pending: list[object] = [selection.schema]
    measured: list[dict[str, object]] = []
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            observed = value.get("x-polylogue-observed-distribution")
            length = observed.get("string_length") if isinstance(observed, dict) else None
            if isinstance(length, dict) and length.get("min") == 63 and length.get("max") == 1023:
                measured.append(length)
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    assert len(measured) == 1, (provider, measured)
    expected = Counter({int(bucket): int(count) for bucket, count in cast(list[list[int]], measured[0]["histogram"])})
    assert expected[_bucket_index(63.0)] / sum(expected.values()) == 0.75
    corpus = SyntheticCorpus.from_selection(selection)
    sessions = corpus.iter_sessions(count=256, seed=47)
    sampled: Counter[int] = Counter()
    try:
        for session in sessions:
            raw = b"".join(session.chunks)
            payload = (
                [json.loads(line) for line in raw.splitlines()]
                if selection.wire_format.encoding == "jsonl"
                else json.loads(raw)
            )
            parsed = parse_payload(provider, payload, "neutral")
            assert parsed
            for message in parsed[0].messages:
                texts = [block.text for block in message.blocks if block.text] or [message.text]
                for text in texts:
                    assert isinstance(text, str) and text
                    sampled[_bucket_index(float(len(text)))] += 1
    finally:
        sessions.close()
    assert sum(sampled.values()) >= 256, provider
    total = sum(sampled.values())
    distance = sum(
        abs(expected[bucket] / sum(expected.values()) - sampled[bucket] / total)
        for bucket in expected.keys() | sampled.keys()
    )
    assert distance < 0.12, (provider, expected, sampled, distance)

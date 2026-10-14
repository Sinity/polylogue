"""Declared Codex roles retain the original carrier form and count."""

import json

import pytest

from polylogue.core.json import JSONDocument
from polylogue.schemas.generation.schema_builder import generate_schema_from_samples
from polylogue.schemas.record_structure import classify_codex_record
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.wire_formats import PROVIDER_WIRE_FORMATS
from polylogue.sources.parsers.codex import parse_stream
from tests.infra.schema_workload_profiles import inferred_record_profile


@pytest.mark.parametrize("form", ["envelope", "flat", "role"])
def test_declared_system_and_tool_carriers_keep_original_form_and_count(form: str) -> None:
    rows: list[JSONDocument] = [
        {"type": "session_meta", "payload": {"id": "neutral-session"}}
        if form == "envelope"
        else {"id": "neutral-session", "timestamp": "2026-01-01T00:00:00Z"}
    ]
    for role in ("system", "tool"):
        message: JSONDocument = {"role": role, "content": [{"type": "input_text", "text": "neutral"}]}
        if form == "envelope":
            row: JSONDocument = {"type": "response_item", "payload": {"type": "message", **message}}
        elif form == "flat":
            row = {"type": "message", **message}
        else:
            row = message
        rows.append(row)
        assert classify_codex_record(row) == f"{role}_message"
    # The actual parser establishes the declared role law independently of the
    # workload classifier and before synthetic inference/generation.
    parsed = parse_stream(rows, "neutral-session")
    assert [message.role.value for message in parsed.messages] == ["system", "tool"]
    schema = generate_schema_from_samples(rows, provider="codex", max_stats_samples=0)
    profile = inferred_record_profile("codex", schema, rows, element="session_record_stream")
    corpus = SyntheticCorpus(schema, PROVIDER_WIRE_FORMATS["codex"], "codex", workload_profile=profile)
    try:
        batch = corpus.generate_batch(count=1, seed=31)
        generated = [json.loads(line) for line in batch.raw_items[0].splitlines()]
        assert [classify_codex_record(row) for row in generated] == ["session_meta", "system_message", "tool_message"]
        assert batch.artifacts[0].message_count == 2
        for row in generated[1:]:
            assert ("payload" in row) == (form == "envelope")
            if form != "envelope":
                assert ("type" in row) == (form == "flat")
    finally:
        corpus._relation_solver.close()

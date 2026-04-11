"""Contracts for schema-annotation summaries used by operator workflows."""

from __future__ import annotations

from polylogue.schemas.operator_annotations import collect_annotation_summary


def test_collect_annotation_summary_deduplicates_same_role_on_union_path() -> None:
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "oneOf": [
                    {
                        "type": "string",
                        "x-polylogue-semantic-role": "message_body",
                        "x-polylogue-score": 0.6,
                        "x-polylogue-evidence": {"branch": "a"},
                    },
                    {
                        "type": "string",
                        "x-polylogue-semantic-role": "message_body",
                        "x-polylogue-score": 0.9,
                        "x-polylogue-evidence": {"branch": "b"},
                    },
                ]
            }
        },
    }

    summary = collect_annotation_summary(schema)

    assert summary.semantic_count == 1
    assert len(summary.roles) == 1
    assert summary.roles[0].path == "$.payload"
    assert summary.roles[0].role == "message_body"
    assert summary.roles[0].confidence == 0.9
    assert summary.roles[0].evidence == {"branch": "b"}

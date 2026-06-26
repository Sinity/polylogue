"""SessionProfile cost/token round-trip contract.

Regression: ``SessionProfile.from_dict`` dropped the cost/token-attribution
fields that ``to_dict`` emits, so a round-tripped or batch-hydrated profile
reported zero tokens (and lost credit/provenance/per-model cost) even when the
columns were populated — which zeroed the postmortem token lanes.
"""

from __future__ import annotations

from polylogue.archive.session.models import SessionProfile

_COST_FIELDS: dict[str, object] = {
    "total_input_tokens": 40000,
    "total_output_tokens": 7500,
    "total_cache_read_tokens": 200000,
    "total_cache_write_tokens": 30000,
    "total_credit_cost": 1.5,
    "cost_provenance": "priced",
    "per_model_cost_json": '{"claude-opus-4-8": 2.025}',
}


def _payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "session_id": "claude-code-session:abc",
        "origin": "claude-code-session",
        "total_cost_usd": 2.025,
    }
    payload.update(_COST_FIELDS)
    return payload


def test_from_dict_reads_cost_and_token_fields() -> None:
    profile = SessionProfile.from_dict(_payload())
    for field, value in _COST_FIELDS.items():
        assert getattr(profile, field) == value, field
    assert profile.total_cost_usd == 2.025


def test_cost_and_token_fields_survive_roundtrip() -> None:
    profile = SessionProfile.from_dict(_payload())
    restored = SessionProfile.from_dict(profile.to_dict())
    for field, value in _COST_FIELDS.items():
        assert getattr(restored, field) == value, field

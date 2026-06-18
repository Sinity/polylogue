"""Reader action vocabulary contract (#1488)."""

from __future__ import annotations

import json
from typing import get_args

import pytest
from pydantic import ValidationError

from polylogue.surfaces.payloads import (
    READER_ACTION_IDS,
    ReaderActionAvailabilityPayload,
    ReaderActionState,
    reader_message_actions,
    reader_session_actions,
)

# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


def test_action_id_registry_is_a_tuple_of_str() -> None:
    """The registry is a tuple (immutable + order-preserving) of strings."""
    assert isinstance(READER_ACTION_IDS, tuple)
    assert all(isinstance(action_id, str) for action_id in READER_ACTION_IDS)


def test_action_id_registry_has_no_duplicates() -> None:
    """A duplicate would be a typo that two action ids collide on."""
    assert len(READER_ACTION_IDS) == len(set(READER_ACTION_IDS))


def test_action_id_registry_includes_the_canonical_actions_from_issue() -> None:
    """The actions enumerated in #1488 must all be registered."""
    required = {
        "copy_text",
        "copy_markdown",
        "copy_raw",
        "copy_permalink",
        "copy_selected_range",
        "copy_typed_only",
        "copy_paste_only",
        "open_raw",
        "open_source",
        "inspect_provenance",
        "mark",
        "annotate",
        "add_to_context",
        "compare",
        "open_stack",
        "continue_elsewhere",
    }
    missing = required - set(READER_ACTION_IDS)
    assert not missing, f"missing canonical action ids: {sorted(missing)}"


def test_default_session_action_ids_are_in_registry() -> None:
    """The defaults for session-level rendering use registered ids only."""
    for action_id in reader_session_actions():
        assert action_id in READER_ACTION_IDS, f"{action_id!r} not in READER_ACTION_IDS"


def test_default_message_action_ids_are_in_registry() -> None:
    """Same invariant for message-level defaults."""
    for action_id in reader_message_actions():
        assert action_id in READER_ACTION_IDS, f"{action_id!r} not in READER_ACTION_IDS"


# ---------------------------------------------------------------------------
# State enum contract
# ---------------------------------------------------------------------------


def test_action_state_covers_the_seven_required_values() -> None:
    """The seven states from #1488 are all accepted."""
    required = {"enabled", "disabled", "partial", "dangerous", "loading", "target", "unavailable"}
    assert set(get_args(ReaderActionState)) == required


@pytest.mark.parametrize(
    "state",
    ["enabled", "disabled", "partial", "dangerous", "loading", "target", "unavailable"],
)
def test_payload_accepts_every_state(state: str) -> None:
    payload = ReaderActionAvailabilityPayload(state=state)  # type: ignore[arg-type]
    assert payload.state == state


def test_payload_rejects_unknown_state() -> None:
    """A typo in the state string is a contract violation, not a free-form note."""
    with pytest.raises(ValidationError):
        ReaderActionAvailabilityPayload(state="totally-made-up")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Default payload contract
# ---------------------------------------------------------------------------


def test_default_payload_is_enabled_state() -> None:
    """Constructing with no arguments yields the enabled state."""
    payload = ReaderActionAvailabilityPayload()
    assert payload.enabled is True
    assert payload.state == "enabled"


def test_enabled_false_can_carry_disabled_reason() -> None:
    payload = ReaderActionAvailabilityPayload(enabled=False, disabled_reason="paste unknown")
    assert payload.enabled is False
    assert payload.disabled_reason == "paste unknown"


def test_payload_carries_repair_and_inspect_paths() -> None:
    payload = ReaderActionAvailabilityPayload(
        state="disabled",
        enabled=False,
        disabled_reason="vectors not ready",
        repair_path="polylogue ops embed backfill",
        inspect_path="/inspect/embeddings",
    )
    assert payload.repair_path == "polylogue ops embed backfill"
    assert payload.inspect_path == "/inspect/embeddings"


def test_payload_serializes_repair_and_inspect_paths_when_set() -> None:
    payload = ReaderActionAvailabilityPayload(
        state="disabled",
        enabled=False,
        disabled_reason="vectors not ready",
        repair_path="polylogue ops embed backfill",
        inspect_path="/inspect/embeddings",
    )
    blob = json.loads(payload.to_json(exclude_none=True))
    assert blob["repair_path"] == "polylogue ops embed backfill"
    assert blob["inspect_path"] == "/inspect/embeddings"
    assert blob["state"] == "disabled"


def test_payload_omits_repair_and_inspect_paths_when_unset() -> None:
    """exclude_none must keep the payload compact for the common case."""
    payload = ReaderActionAvailabilityPayload()
    blob = json.loads(payload.to_json(exclude_none=True))
    assert "repair_path" not in blob
    assert "inspect_path" not in blob
    assert "disabled_reason" not in blob

"""Unit tests for shared parser helpers (sources/parsers/base_support.py)."""

from __future__ import annotations

from polylogue.archive.message.roles import Role
from polylogue.sources.parsers.base import ParsedMessage, fill_linear_parent_chain


def _msg(pid: str, *, parent: str | None = None, active: bool | None = True) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=pid,
        role=Role.USER,
        text=pid,
        position=int(pid.rsplit("-", 1)[-1]) if "-" in pid else 0,
        variant_index=0,
        is_active_path=active,
        parent_message_provider_id=parent,
    )


def test_fill_linear_parent_chain_chains_previous_message() -> None:
    messages = [_msg("m-0"), _msg("m-1"), _msg("m-2")]
    filled = fill_linear_parent_chain(messages)
    assert [m.parent_message_provider_id for m in filled] == [None, "m-0", "m-1"]


def test_fill_linear_parent_chain_preserves_existing_parent_evidence() -> None:
    # A message that already carries real parent evidence (e.g. drive.py's
    # branch chunks) must not be overwritten.
    messages = [_msg("m-0"), _msg("m-1", parent="m-0-real-branch-parent"), _msg("m-2")]
    filled = fill_linear_parent_chain(messages)
    assert filled[1].parent_message_provider_id == "m-0-real-branch-parent"
    # m-2 still chains to the nearest preceding active-path message (m-1),
    # not to the overridden parent of m-1.
    assert filled[2].parent_message_provider_id == "m-1"


def test_fill_linear_parent_chain_skips_inactive_path_messages() -> None:
    # An inactive-path message is never used as a chain anchor for the
    # message that follows it.
    messages = [_msg("m-0"), _msg("m-1", active=False), _msg("m-2")]
    filled = fill_linear_parent_chain(messages)
    assert filled[1].parent_message_provider_id == "m-0"
    assert filled[2].parent_message_provider_id == "m-0"


def test_fill_linear_parent_chain_empty_list() -> None:
    assert fill_linear_parent_chain([]) == []


def test_fill_linear_parent_chain_single_message_stays_unparented() -> None:
    filled = fill_linear_parent_chain([_msg("m-0")])
    assert filled[0].parent_message_provider_id is None

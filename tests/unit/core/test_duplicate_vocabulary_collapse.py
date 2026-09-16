"""Vocabularies that model one concept have one name (polylogue-jglh).

Anti-vacuity: re-declaring either vocabulary locally, or letting a member set
drift from its owner, makes these red -- the MessageTypeName check raises at
import of its own module.
"""

from __future__ import annotations

from typing import get_args

from polylogue.core.enums import (
    SOURCE_FIDELITY_STATUS_VALUES,
    MessageType,
    SourceFidelityStatus,
)
from polylogue.storage.sqlite.queries.message_query_reads import MessageTypeName


def test_message_type_name_is_message_types_member_set() -> None:
    assert frozenset(get_args(MessageTypeName)) == frozenset(member.value for member in MessageType)


def test_source_fidelity_status_has_one_owner() -> None:
    import polylogue.sources.parsers.hermes_state as hermes_state
    import polylogue.surfaces.payloads as payloads

    assert not hasattr(hermes_state, "HermesFidelityStatus")
    assert not hasattr(payloads, "ImportFidelityStatus")
    assert payloads.SourceFidelityStatus is SourceFidelityStatus
    assert frozenset(get_args(SourceFidelityStatus)) == SOURCE_FIDELITY_STATUS_VALUES
    assert {"exact", "absent", "redacted", "degraded", "inferred"} == SOURCE_FIDELITY_STATUS_VALUES

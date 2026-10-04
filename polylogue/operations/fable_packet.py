"""Resident Fable packet compilation on the original selected archive snapshot."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from polylogue.analysis.fable_packet import regenerate_private_fable_packet
from polylogue.operations.fable_packet_contracts import FablePacketRequest, FablePacketResult
from polylogue.surfaces.outcome import decide_outcome

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def execute_fable_packet(
    payload: Mapping[str, object], *, archive: ArchiveStore, checkpoint: Callable[[], None]
) -> dict[str, object]:
    request = FablePacketRequest.model_validate(payload)
    checkpoint()
    packet = regenerate_private_fable_packet(archive, **request.model_dump(), checkpoint=checkpoint)
    checkpoint()
    gaps = packet.not_supported_reasons if packet.status == "not_supported" else ()
    return FablePacketResult(
        packet=packet, outcome=decide_outcome(matched=len(packet.selected_refs), degraded=gaps)
    ).model_dump(mode="json")

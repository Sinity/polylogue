"""Canonical resident request and result for a private descriptive Fable packet."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from polylogue.analysis.fable_packet_contracts import FableDelegationPacket
from polylogue.surfaces.outcome import OutcomeEnvelope


class FablePacketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    seed: str
    requested_size: int = Field(ge=0)
    schema_id: str = "delegation.discourse"
    schema_version: int = Field(default=1, ge=1)
    exact_template_cap: int = Field(default=1, ge=1)


class FablePacketResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    packet: FableDelegationPacket
    outcome: OutcomeEnvelope


def decode_fable_packet_result(value: object) -> FablePacketResult:
    from polylogue.operations.daemon_protocol import _json_result_validator

    return cast(FablePacketResult, _json_result_validator(FablePacketResult).validate_python(value))

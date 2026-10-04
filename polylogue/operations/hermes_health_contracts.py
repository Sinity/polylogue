"""Canonical request, report and strict JSON hydration for Hermes health."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from polylogue.analysis.hermes_health_contracts import HermesIntegrationHealth
from polylogue.surfaces.outcome import OutcomeEnvelope


class HermesHealthRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class HermesHealthResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    report: HermesIntegrationHealth
    outcome: OutcomeEnvelope


def decode_hermes_health_result(value: object) -> HermesHealthResult:
    """Hydrate the same strict JSON forms admitted by the resident protocol."""
    from typing import cast

    from polylogue.operations.daemon_protocol import _json_result_validator

    return cast(HermesHealthResult, _json_result_validator(HermesHealthResult).validate_python(value))

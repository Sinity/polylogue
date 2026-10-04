"""Closed resident insight export request and result."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict

from polylogue.analysis.export_bundle_contracts import InsightExportBundleRequest, InsightExportBundleResult
from polylogue.surfaces.outcome import OutcomeEnvelope


class InsightExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    request: InsightExportBundleRequest


class InsightExportResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    bundle: InsightExportBundleResult
    outcome: OutcomeEnvelope


def decode_insight_export_result(value: object) -> InsightExportResult:
    from polylogue.operations.daemon_protocol import _json_result_validator

    return cast(InsightExportResult, _json_result_validator(InsightExportResult).validate_python(value))


def decode_insight_export_request(value: object) -> InsightExportRequest:
    from polylogue.operations.daemon_protocol import _json_result_validator

    return cast(InsightExportRequest, _json_result_validator(InsightExportRequest).validate_python(value))

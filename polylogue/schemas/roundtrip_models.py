"""Typed reports and stage helpers for schema roundtrip proof."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_STAGE_ORDER = (
    "selection",
    "synthetic",
    "acquisition",
    "validation",
    "parse_dispatch",
    "prepare_persist",
    "corpus_verification",
    "artifact_proof",
)


@dataclass(frozen=True)
class RoundtripStageReport:
    name: str
    status: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "detail": self.detail,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class ProviderRoundtripProofReport:
    provider: str
    package_version: str
    element_kind: str | None
    wire_encoding: str
    stages: dict[str, RoundtripStageReport]

    @property
    def is_clean(self) -> bool:
        return all(stage.status == "ok" for stage in self.stages.values())

    @property
    def failed_stages(self) -> list[str]:
        return [name for name, stage in self.stages.items() if stage.status == "error"]

    @property
    def summary(self) -> dict[str, Any]:
        ok_stages = sum(1 for stage in self.stages.values() if stage.status == "ok")
        skipped_stages = sum(1 for stage in self.stages.values() if stage.status == "skip")
        error_stages = sum(1 for stage in self.stages.values() if stage.status == "error")
        artifact_count = self.stages.get("synthetic", RoundtripStageReport("", "skip", "")).detail.get(
            "generated_artifacts",
            0,
        )
        parsed_conversations = self.stages.get("parse_dispatch", RoundtripStageReport("", "skip", "")).detail.get(
            "parsed_conversations",
            0,
        )
        persisted_conversations = self.stages.get("prepare_persist", RoundtripStageReport("", "skip", "")).detail.get(
            "persisted_conversations",
            0,
        )
        return {
            "clean": self.is_clean,
            "ok_stages": ok_stages,
            "skipped_stages": skipped_stages,
            "error_stages": error_stages,
            "artifact_count": artifact_count,
            "parsed_conversations": parsed_conversations,
            "persisted_conversations": persisted_conversations,
            "failed_stages": self.failed_stages,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "package_version": self.package_version,
            "element_kind": self.element_kind,
            "wire_encoding": self.wire_encoding,
            "summary": self.summary,
            "stages": {
                name: self.stages[name].to_dict()
                for name in _STAGE_ORDER
                if name in self.stages
            },
        }


@dataclass(frozen=True)
class RoundtripProofSuiteReport:
    provider_reports: dict[str, ProviderRoundtripProofReport]

    @property
    def is_clean(self) -> bool:
        return all(report.is_clean for report in self.provider_reports.values())

    @property
    def summary(self) -> dict[str, Any]:
        total_providers = len(self.provider_reports)
        clean_providers = sum(1 for report in self.provider_reports.values() if report.is_clean)
        failed_providers = total_providers - clean_providers
        total_artifacts = sum(
            int(report.summary["artifact_count"])
            for report in self.provider_reports.values()
        )
        parsed_conversations = sum(
            int(report.summary["parsed_conversations"])
            for report in self.provider_reports.values()
        )
        persisted_conversations = sum(
            int(report.summary["persisted_conversations"])
            for report in self.provider_reports.values()
        )
        return {
            "clean": self.is_clean,
            "provider_count": total_providers,
            "clean_providers": clean_providers,
            "failed_providers": failed_providers,
            "artifact_count": total_artifacts,
            "parsed_conversations": parsed_conversations,
            "persisted_conversations": persisted_conversations,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "providers": {
                provider: report.to_dict()
                for provider, report in sorted(self.provider_reports.items())
            },
        }


def stage_ok(name: str, summary: str, **detail: Any) -> RoundtripStageReport:
    return RoundtripStageReport(name=name, status="ok", summary=summary, detail=detail)


def stage_error(name: str, error: Exception | str, **detail: Any) -> RoundtripStageReport:
    return RoundtripStageReport(
        name=name,
        status="error",
        summary=str(error),
        detail=detail,
        error=str(error),
    )


def stage_skip(name: str, summary: str) -> RoundtripStageReport:
    return RoundtripStageReport(name=name, status="skip", summary=summary)


def finalize_stages(
    stages: dict[str, RoundtripStageReport],
    *,
    last_completed: str | None = None,
    skip_after: str | None = None,
) -> dict[str, RoundtripStageReport]:
    terminal = skip_after or last_completed
    passed_terminal = terminal is None
    for stage_name in _STAGE_ORDER:
        if stage_name in stages:
            if terminal == stage_name:
                passed_terminal = True
            continue
        if skip_after is not None:
            stages[stage_name] = stage_skip(stage_name, f"Skipped after {skip_after}")
        elif last_completed is not None and not passed_terminal:
            stages[stage_name] = stage_skip(stage_name, f"Skipped after {last_completed}")
        else:
            stages[stage_name] = stage_skip(stage_name, "Not executed")
        if terminal == stage_name:
            passed_terminal = True
    return stages


__all__ = [
    "ProviderRoundtripProofReport",
    "RoundtripProofSuiteReport",
    "RoundtripStageReport",
    "_STAGE_ORDER",
    "finalize_stages",
    "stage_error",
    "stage_ok",
    "stage_skip",
]

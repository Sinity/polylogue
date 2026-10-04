"""Canonical read-only composition of the Hermes integration diagnostic."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from polylogue.analysis.hermes_integration_health import HermesIntegrationHealth, build_hermes_integration_health
from polylogue.surfaces.outcome import OutcomeEnvelope, decide_outcome

if TYPE_CHECKING:
    from polylogue.config import Config


class HermesHealthRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class HermesHealthResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    report: HermesIntegrationHealth
    outcome: OutcomeEnvelope


def read_hermes_health(
    archive_root: Path, *, hermes_root: Path, checkpoint: Callable[[], None] = lambda: None
) -> HermesIntegrationHealth:
    """Compose available evidence without requiring a derived Index reader."""
    from polylogue.daemon.convergence_debt_alert import source_family_for_subject, watchsource_name_to_family
    from polylogue.daemon.convergence_debt_status import convergence_debt_summary_info

    checkpoint()
    family = watchsource_name_to_family("hermes")
    debt = convergence_debt_summary_info(archive_root / "source.db")
    checkpoint()
    summary = next((item for item in debt.family_summaries if item.family == family), None)
    return build_hermes_integration_health(
        archive_root,
        hermes_root=hermes_root,
        convergence_debt_available=debt.available,
        convergence_debt_error=debt.error,
        convergence_debt_failed_count=summary.failed_count if summary else 0,
        convergence_debt_deferred_count=summary.deferred_count if summary else 0,
        convergence_debt_retry_due_count=sum(
            1
            for item in debt.recent
            if item.retry_due and source_family_for_subject(item.subject_type, item.subject_id) == family
        ),
        checkpoint=checkpoint,
    )


def configured_hermes_health(config: Config) -> HermesIntegrationHealth:
    from polylogue.config import active_archive_root
    from polylogue.paths import hermes_sessions_path

    root = next((source.path for source in config.sources if source.name == "hermes" and source.path is not None), None)
    return read_hermes_health(
        active_archive_root(config), hermes_root=root if root is not None else hermes_sessions_path()
    )


def execute_hermes_health(
    archive_root: Path, *, hermes_root: Path, checkpoint: Callable[[], None]
) -> dict[str, object]:
    report = read_hermes_health(archive_root, hermes_root=hermes_root, checkpoint=checkpoint)
    checkpoint()
    gaps = (f"hermes_health_{report.verdict}",) if report.verdict in {"degraded", "unavailable"} else ()
    return HermesHealthResult(report=report, outcome=decide_outcome(matched=1, degraded=gaps)).model_dump(mode="json")


def decode_hermes_health_result(value: object) -> HermesHealthResult:
    """Hydrate the same strict JSON forms admitted by the resident protocol."""
    from typing import cast

    from polylogue.operations.daemon_protocol import _json_result_validator

    return cast(HermesHealthResult, _json_result_validator(HermesHealthResult).validate_python(value))

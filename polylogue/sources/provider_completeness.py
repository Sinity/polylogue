"""Provider/importer package completeness report."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from polylogue.core.enums import Origin
from polylogue.sources.origin_specs import ORIGIN_SPECS, OriginCompletenessMode
from polylogue.surfaces.payloads import (
    ProviderCompletenessItemPayload,
    ProviderCompletenessStatus,
    ProviderPackageCompletenessPayload,
    ProviderPackageCompletenessRowPayload,
    ProviderPackageCompletenessTotalsPayload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_REQUIRED_ITEM_NAMES = (
    "detector",
    "raw_model",
    "parser",
    "normalizer",
    "fixtures",
    "schema_package",
    "query_units",
    "read_views",
    "import_explain",
    "privacy_caveats",
    "generated_docs",
)

# OriginSpec owns this inventory. Keep the exported name as a projection for
# report consumers without retaining a second admission registry.
PACKAGE_MODE_SPECS: tuple[tuple[Origin, OriginCompletenessMode], ...] = tuple(
    (origin_spec.origin, mode) for origin_spec in ORIGIN_SPECS for mode in origin_spec.completeness_modes
)


def provider_package_completeness(*, origin: str | None = None) -> ProviderPackageCompletenessPayload:
    """Compile the provider/importer package completeness report."""

    rows = tuple(
        row
        for row in (_row_for_spec(spec) for spec in PACKAGE_MODE_SPECS)
        if origin is None or row.origin == origin or row.provider_wire == origin
    )
    return ProviderPackageCompletenessPayload(
        generated_at=datetime.now(UTC).isoformat(),
        rows=rows,
        totals=_totals(rows),
        caveats=(
            "This report is a readiness map; runtime import is not blocked by partial rows.",
            "Rows are keyed by public origin plus capture mode. Provider-wire tokens are evidence fields only.",
        ),
    )


def accepted_blockers(report: ProviderPackageCompletenessPayload) -> tuple[str, ...]:
    """Return human-readable blockers for accepted package rows."""

    return tuple(
        f"{row.package_ref}: {blocker}" for row in report.rows if row.maturity == "accepted" for blocker in row.blockers
    )


def _row_for_spec(spec: tuple[Origin, OriginCompletenessMode]) -> ProviderPackageCompletenessRowPayload:
    origin, mode = spec
    items = {
        "detector": _item(mode.detector_paths),
        "raw_model": _item(mode.raw_model_paths),
        "parser": _item(mode.parser_paths),
        "normalizer": _item(mode.normalizer_paths),
        "fixtures": _item(mode.fixture_paths),
        "schema_package": _item(mode.schema_paths),
        "query_units": _item(("polylogue/archive/query/metadata.py", "polylogue/archive/query/unit_results.py")),
        "read_views": _item(("polylogue/archive/viewport/profiles.py", "polylogue/cli/read_view_handlers.py")),
        "import_explain": _item(("polylogue/sources/import_explain.py",)),
        "privacy_caveats": _item(mode.privacy_paths),
        "generated_docs": _item(mode.docs_paths),
        "debt_rows": ProviderCompletenessItemPayload(
            status="not_applicable",
            evidence=("tracked by #2179 unified archive debt views",),
            caveats=("Debt rows are reported at archive subsystem level, not per provider package yet.",),
        ),
    }
    blockers = tuple(
        f"{name} is {item.status}"
        for name, item in items.items()
        if name in _REQUIRED_ITEM_NAMES and item.status in {"missing", "partial"}
    )
    if mode.maturity == "proposed":
        status: ProviderCompletenessStatus = "proposed"
    elif mode.maturity == "reserved":
        status = "reserved"
    elif mode.maturity == "unsupported":
        status = "unsupported"
    elif blockers:
        status = "partial"
    else:
        status = "complete"
    evidence_refs = tuple(
        evidence for item in items.values() for evidence in ((item.owner_path,) if item.owner_path else item.evidence)
    )
    return ProviderPackageCompletenessRowPayload(
        package_ref=mode.package_ref,
        origin=origin.value,
        capture_mode=mode.capture_mode,
        provider_wire=mode.provider_wire.value if mode.provider_wire is not None else None,
        maturity=mode.maturity,
        detector=items["detector"],
        raw_model=items["raw_model"],
        parser=items["parser"],
        normalizer=items["normalizer"],
        fixtures=items["fixtures"],
        schema_package=items["schema_package"],
        query_units=items["query_units"],
        read_views=items["read_views"],
        import_explain=items["import_explain"],
        privacy_caveats=items["privacy_caveats"],
        generated_docs=items["generated_docs"],
        debt_rows=items["debt_rows"],
        status=status,
        blockers=() if mode.maturity in {"proposed", "reserved", "unsupported"} else blockers,
        evidence_refs=evidence_refs,
    )


def _item(paths: tuple[str, ...]) -> ProviderCompletenessItemPayload:
    if not paths:
        return ProviderCompletenessItemPayload(status="missing", caveats=("no owner path declared",))
    present = tuple(path for path in paths if (REPO_ROOT / path).exists())
    missing = tuple(path for path in paths if not (REPO_ROOT / path).exists())
    if not present:
        return ProviderCompletenessItemPayload(status="missing", caveats=tuple(f"missing {path}" for path in missing))
    if missing:
        return ProviderCompletenessItemPayload(
            status="partial",
            owner_path=present[0],
            evidence=present,
            caveats=tuple(f"missing {path}" for path in missing),
        )
    return ProviderCompletenessItemPayload(status="complete", owner_path=present[0], evidence=present)


def _totals(rows: tuple[ProviderPackageCompletenessRowPayload, ...]) -> ProviderPackageCompletenessTotalsPayload:
    return ProviderPackageCompletenessTotalsPayload(
        total=len(rows),
        complete=sum(1 for row in rows if row.status == "complete"),
        partial=sum(1 for row in rows if row.status == "partial"),
        missing=sum(1 for row in rows if row.status == "missing"),
        proposed=sum(1 for row in rows if row.status == "proposed"),
        reserved=sum(1 for row in rows if row.status == "reserved"),
        unsupported=sum(1 for row in rows if row.status == "unsupported"),
        accepted_blocked=sum(1 for row in rows if row.maturity == "accepted" and row.blockers),
    )


__all__ = [
    "PACKAGE_MODE_SPECS",
    "accepted_blockers",
    "provider_package_completeness",
]

"""Provider/importer package completeness report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from polylogue.core.enums import Origin, Provider
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


@dataclass(frozen=True, slots=True)
class _PackageModeSpec:
    package_ref: str
    origin: Origin
    capture_mode: str
    provider_wire: Provider | None
    maturity: Literal["accepted", "proposed"]
    detector_paths: tuple[str, ...]
    raw_model_paths: tuple[str, ...]
    parser_paths: tuple[str, ...]
    normalizer_paths: tuple[str, ...]
    fixture_paths: tuple[str, ...]
    schema_paths: tuple[str, ...]
    docs_paths: tuple[str, ...]
    privacy_paths: tuple[str, ...]
    caveats: tuple[str, ...] = ()


PACKAGE_MODE_SPECS: tuple[_PackageModeSpec, ...] = (
    _PackageModeSpec(
        package_ref="provider-package:claude-code-session/export-jsonl@v1",
        origin=Origin.CLAUDE_CODE_SESSION,
        capture_mode="export-jsonl",
        provider_wire=Provider.CLAUDE_CODE,
        maturity="accepted",
        detector_paths=("polylogue/sources/parsers/claude/code_detection.py", "polylogue/sources/dispatch.py"),
        raw_model_paths=("polylogue/sources/providers/claude_code_record.py",),
        parser_paths=("polylogue/sources/parsers/claude/code_parser.py",),
        normalizer_paths=("polylogue/sources/parsers/claude/common.py",),
        fixture_paths=(
            "tests/unit/sources/test_parsers_claude_code_artifacts.py",
            "tests/unit/sources/test_assembly_claude_code_history.py",
        ),
        schema_paths=("polylogue/schemas/providers/claude-code/catalog.json",),
        docs_paths=("docs/providers/claude-code.md",),
        privacy_paths=("docs/provider-origin-identity.md",),
    ),
    _PackageModeSpec(
        package_ref="provider-package:codex-session/session-jsonl@v1",
        origin=Origin.CODEX_SESSION,
        capture_mode="session-jsonl",
        provider_wire=Provider.CODEX,
        maturity="accepted",
        detector_paths=("polylogue/sources/parsers/codex.py", "polylogue/sources/dispatch.py"),
        raw_model_paths=("polylogue/sources/providers/codex.py",),
        parser_paths=("polylogue/sources/parsers/codex.py",),
        normalizer_paths=("polylogue/sources/parsers/base_support.py",),
        fixture_paths=("tests/unit/sources/test_parsers_codex.py", "tests/data/codex_event_stream"),
        schema_paths=("polylogue/schemas/providers/codex/catalog.json",),
        docs_paths=("docs/providers/openai-codex.md",),
        privacy_paths=("docs/provider-origin-identity.md",),
    ),
    _PackageModeSpec(
        package_ref="provider-package:chatgpt-export/takeout-json@v1",
        origin=Origin.CHATGPT_EXPORT,
        capture_mode="takeout-json",
        provider_wire=Provider.CHATGPT,
        maturity="accepted",
        detector_paths=("polylogue/sources/parsers/chatgpt.py", "polylogue/sources/dispatch.py"),
        raw_model_paths=("polylogue/sources/parsers/chatgpt.py",),
        parser_paths=("polylogue/sources/parsers/chatgpt.py",),
        normalizer_paths=("polylogue/sources/parsers/base_support.py",),
        fixture_paths=("tests/unit/sources/test_parsers_chatgpt.py", "tests/data/golden/chatgpt-simple.md"),
        schema_paths=("polylogue/schemas/providers/chatgpt/catalog.json",),
        docs_paths=("docs/providers/chatgpt.md",),
        privacy_paths=("docs/provider-origin-identity.md",),
    ),
    _PackageModeSpec(
        package_ref="provider-package:claude-ai-export/export-json@v1",
        origin=Origin.CLAUDE_AI_EXPORT,
        capture_mode="export-json",
        provider_wire=Provider.CLAUDE_AI,
        maturity="accepted",
        detector_paths=("polylogue/sources/parsers/claude/ai_parser.py", "polylogue/sources/dispatch.py"),
        raw_model_paths=("polylogue/sources/providers/claude_ai.py",),
        parser_paths=("polylogue/sources/parsers/claude/ai_parser.py",),
        normalizer_paths=("polylogue/sources/parsers/claude/common.py",),
        fixture_paths=("tests/unit/sources/test_parsers_claude_ai_catalog.py",),
        schema_paths=("polylogue/schemas/providers/claude-ai/catalog.json",),
        docs_paths=("docs/providers/claude-ai.md",),
        privacy_paths=("docs/provider-origin-identity.md",),
    ),
    _PackageModeSpec(
        package_ref="provider-package:aistudio-drive/drive-export@v1",
        origin=Origin.AISTUDIO_DRIVE,
        capture_mode="drive-like-export",
        provider_wire=Provider.GEMINI,
        maturity="accepted",
        detector_paths=("polylogue/sources/parsers/drive.py", "polylogue/sources/dispatch.py"),
        raw_model_paths=("polylogue/sources/providers/gemini_message.py",),
        parser_paths=("polylogue/sources/parsers/drive.py",),
        normalizer_paths=("polylogue/sources/parsers/drive_support.py",),
        fixture_paths=("tests/unit/sources/test_parsers_drive.py", "tests/data/gemini_chunked_prompt"),
        schema_paths=("polylogue/schemas/providers/gemini/catalog.json",),
        docs_paths=("docs/providers/gemini.md",),
        privacy_paths=("docs/provider-origin-identity.md",),
    ),
    _PackageModeSpec(
        package_ref="provider-package:browser-capture/live-receiver@v1",
        origin=Origin.UNKNOWN_EXPORT,
        capture_mode="browser-capture-live-receiver",
        provider_wire=None,
        maturity="proposed",
        detector_paths=("polylogue/sources/parsers/browser_capture.py", "polylogue/sources/dispatch.py"),
        raw_model_paths=("polylogue/sources/parsers/browser_capture.py",),
        parser_paths=("polylogue/sources/parsers/browser_capture.py",),
        normalizer_paths=("polylogue/sources/parsers/base_support.py",),
        fixture_paths=(
            "tests/unit/sources/test_browser_capture.py",
            "tests/data/witnesses/browser-capture-sequence.json",
        ),
        schema_paths=(),
        docs_paths=("docs/browser-capture.md",),
        privacy_paths=("docs/provider-origin-identity.md", "docs/daemon-threat-model.md"),
        caveats=("Browser capture maps captured page sessions onto provider-specific origins at parse time.",),
    ),
    _PackageModeSpec(
        package_ref="provider-package:antigravity-session/language-server-export@v1",
        origin=Origin.ANTIGRAVITY_SESSION,
        capture_mode="language-server-export",
        provider_wire=Provider.ANTIGRAVITY,
        maturity="accepted",
        detector_paths=("polylogue/sources/parsers/antigravity.py", "polylogue/sources/source_parsing.py"),
        raw_model_paths=("polylogue/sources/parsers/antigravity.py",),
        parser_paths=("polylogue/sources/parsers/antigravity.py",),
        normalizer_paths=("polylogue/sources/parsers/base_support.py",),
        fixture_paths=(
            "tests/unit/sources/test_antigravity_language_server.py",
            "tests/unit/sources/parsers/test_antigravity.py",
        ),
        schema_paths=("polylogue/schemas/providers/antigravity/catalog.json",),
        docs_paths=("docs/architecture.md",),
        privacy_paths=("docs/provider-origin-identity.md",),
    ),
    _PackageModeSpec(
        package_ref="provider-package:gemini-cli-session/local-agent-document@v1",
        origin=Origin.GEMINI_CLI_SESSION,
        capture_mode="local-agent-document",
        provider_wire=Provider.GEMINI_CLI,
        maturity="accepted",
        detector_paths=("polylogue/sources/parsers/local_agent.py", "polylogue/sources/dispatch.py"),
        raw_model_paths=("polylogue/sources/parsers/local_agent.py",),
        parser_paths=("polylogue/sources/parsers/local_agent.py",),
        normalizer_paths=("polylogue/sources/parsers/base_support.py",),
        fixture_paths=("tests/unit/sources/test_parsers_local_agent.py",),
        schema_paths=("polylogue/schemas/providers/gemini-cli/catalog.json",),
        docs_paths=("docs/providers/README.md",),
        privacy_paths=("docs/provider-origin-identity.md",),
    ),
    _PackageModeSpec(
        package_ref="provider-package:hermes-session/state-db@v1",
        origin=Origin.HERMES_SESSION,
        capture_mode="state-db",
        provider_wire=Provider.HERMES,
        maturity="accepted",
        detector_paths=(
            "polylogue/sources/parsers/hermes_state.py",
            "polylogue/sources/dispatch.py",
            "polylogue/sources/source_parsing.py",
        ),
        raw_model_paths=("polylogue/sources/parsers/hermes_state.py",),
        parser_paths=("polylogue/sources/parsers/hermes_state.py",),
        normalizer_paths=("polylogue/sources/parsers/base_support.py",),
        fixture_paths=("tests/unit/sources/test_parsers_local_agent.py",),
        schema_paths=("polylogue/schemas/providers/hermes/state_db_v16.contract.json",),
        docs_paths=("docs/providers/README.md", "docs/onboarding.md"),
        privacy_paths=("docs/provider-origin-identity.md",),
    ),
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


def _row_for_spec(spec: _PackageModeSpec) -> ProviderPackageCompletenessRowPayload:
    items = {
        "detector": _item(spec.detector_paths),
        "raw_model": _item(spec.raw_model_paths),
        "parser": _item(spec.parser_paths),
        "normalizer": _item(spec.normalizer_paths),
        "fixtures": _item(spec.fixture_paths),
        "schema_package": _item(spec.schema_paths),
        "query_units": _item(("polylogue/archive/query/metadata.py", "polylogue/archive/query/unit_results.py")),
        "read_views": _item(("polylogue/archive/viewport/profiles.py", "polylogue/cli/read_view_handlers.py")),
        "import_explain": _item(("polylogue/sources/import_explain.py",)),
        "privacy_caveats": _item(spec.privacy_paths),
        "generated_docs": _item(spec.docs_paths),
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
    if spec.maturity == "proposed":
        status: ProviderCompletenessStatus = "proposed"
    elif blockers:
        status = "partial"
    else:
        status = "complete"
    evidence_refs = tuple(
        evidence for item in items.values() for evidence in ((item.owner_path,) if item.owner_path else item.evidence)
    )
    return ProviderPackageCompletenessRowPayload(
        package_ref=spec.package_ref,
        origin=spec.origin.value,
        capture_mode=spec.capture_mode,
        provider_wire=spec.provider_wire.value if spec.provider_wire is not None else None,
        maturity=spec.maturity,
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
        blockers=() if spec.maturity == "proposed" else blockers,
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
        accepted_blocked=sum(1 for row in rows if row.maturity == "accepted" and row.blockers),
    )


__all__ = [
    "PACKAGE_MODE_SPECS",
    "accepted_blockers",
    "provider_package_completeness",
]

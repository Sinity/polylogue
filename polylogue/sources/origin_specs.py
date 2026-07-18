"""Executable source-admission declarations for public archive origins.

Provider adapters retain record-level parsing.  This module owns only the
cross-adapter admission contract: public origin vocabulary, acquisition modes,
detector tightness, registration/fixture evidence, fidelity, and reparse
consequences.  The initial pilots deliberately cover a streaming runtime, a
document export, and a reserved origin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

from polylogue.core.enums import Origin, Provider
from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationRegistry,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
    validate_registry,
)

OriginLifecycle = Literal["executable", "reserved", "unsupported", "compatibility-only"]
OriginCompletenessMaturity = Literal["accepted", "proposed", "reserved", "unsupported"]
ArtifactParsePolicy = Literal["session", "fact", "raw-only"]


@dataclass(frozen=True, slots=True)
class OriginArtifactRule:
    """Declared admission policy for one native artifact path family.

    These rules deliberately describe paths, not a second parser registry:
    the owning ``OriginSpec`` remains the one admission declaration and
    consumers can distinguish session streams from evidence-bearing sidecars.
    """

    kind: str
    path_pattern: str
    parse_policy: ArtifactParsePolicy
    parser_path: str | None
    coverage_role: str
    fidelity_note: str
    path_suffixes: tuple[str, ...]

    def matches(self, source_path: str) -> bool:
        return re.search(self.path_pattern, source_path.replace("\\", "/")) is not None


@dataclass(frozen=True, slots=True)
class OriginCompletenessMode:
    """One material import mode projected into provider-completeness reports."""

    package_ref: str
    capture_mode: str
    provider_wire: Provider | None
    maturity: OriginCompletenessMaturity
    detector_paths: tuple[str, ...]
    raw_model_paths: tuple[str, ...]
    parser_paths: tuple[str, ...]
    normalizer_paths: tuple[str, ...]
    fixture_paths: tuple[str, ...]
    schema_paths: tuple[str, ...]
    docs_paths: tuple[str, ...]
    privacy_paths: tuple[str, ...]
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OriginSpec:
    """One public-origin admission contract, independent of parser internals."""

    origin: Origin
    declaration: DeclarationSpec
    lifecycle: OriginLifecycle
    acquisition_modes: tuple[str, ...]
    provider_wires: tuple[Provider, ...]
    collision_policy: str | None
    detector_tightness: int | None
    parser_paths: tuple[str, ...]
    stream_parser_path: str | None
    assembly_paths: tuple[str, ...]
    fixture_paths: tuple[str, ...]
    coverage_refs: tuple[str, ...]
    fidelity_notes: tuple[str, ...]
    semantic_reparse: str
    artifact_rules: tuple[OriginArtifactRule, ...] = ()
    completeness_modes: tuple[OriginCompletenessMode, ...] = ()


@dataclass(frozen=True, slots=True)
class OriginSpecDiagnostic:
    """Actionable domain diagnostic layered over the shared declaration kernel."""

    code: str
    message: str
    origin: Origin
    owner_path: str
    repair_command: str


class OriginSpecRegistry:
    """Register origin declarations and reject incomplete admission edges."""

    def __init__(self) -> None:
        self._kernel = DeclarationRegistry()
        self._by_origin: dict[Origin, OriginSpec] = {}

    def register(self, spec: OriginSpec) -> OriginSpec:
        if spec.origin in self._by_origin:
            raise ValueError(f"duplicate OriginSpec for {spec.origin.value!r}")
        if spec.declaration.public_name != spec.origin.value:
            raise ValueError(f"{spec.origin.value}: declaration public name must equal the public Origin token")
        if not spec.acquisition_modes:
            raise ValueError(f"{spec.origin.value}: missing acquisition mode")
        if not spec.coverage_refs:
            raise ValueError(f"{spec.origin.value}: missing coverage declaration")
        if not spec.fixture_paths:
            raise ValueError(f"{spec.origin.value}: missing fixture declaration")
        if len(spec.provider_wires) > 1 and not spec.collision_policy:
            raise ValueError(f"{spec.origin.value}: multiple provider wires require an explicit collision policy")
        if spec.lifecycle == "executable":
            if spec.detector_tightness is None:
                raise ValueError(f"{spec.origin.value}: executable origin requires detector tightness")
            if not spec.parser_paths:
                raise ValueError(f"{spec.origin.value}: executable origin requires parser binding")
            for rule in spec.artifact_rules:
                if rule.parse_policy != "raw-only" and rule.parser_path is None:
                    raise ValueError(f"{spec.origin.value}: {rule.kind} requires a parser binding")
                if not rule.path_suffixes:
                    raise ValueError(f"{spec.origin.value}: {rule.kind} requires acquisition suffixes")
                if any(not suffix.startswith(".") or suffix != suffix.lower() for suffix in rule.path_suffixes):
                    raise ValueError(
                        f"{spec.origin.value}: {rule.kind} acquisition suffixes must be lowercase dot suffixes"
                    )
        elif spec.parser_paths or spec.stream_parser_path is not None:
            raise ValueError(f"{spec.origin.value}: non-executable origin cannot declare a parser binding")
        self._kernel.register(spec.declaration)
        self._by_origin[spec.origin] = spec
        return spec

    def specs(self) -> tuple[OriginSpec, ...]:
        return tuple(self._by_origin[origin] for origin in sorted(self._by_origin, key=lambda item: item.value))

    def diagnostics(self) -> tuple[OriginSpecDiagnostic, ...]:
        diagnostics = [
            OriginSpecDiagnostic(
                code=item.code,
                message=item.message,
                origin=Origin.from_string(item.declaration_id.removeprefix("origin.")),
                owner_path=item.owner_path,
                repair_command=item.repair_command,
            )
            for item in validate_registry(self._kernel)
        ]
        executable = [spec for spec in self.specs() if spec.lifecycle == "executable"]
        missing_origins = sorted(set(Origin) - set(self._by_origin), key=lambda origin: origin.value)
        diagnostics.extend(
            OriginSpecDiagnostic(
                code="missing_origin_spec",
                message=f"{origin.value}: public Origin has no admission declaration",
                origin=origin,
                owner_path="polylogue/sources/origin_specs.py",
                repair_command="devtools test tests/unit/sources/test_origin_specs.py",
            )
            for origin in missing_origins
        )
        tightness = [spec.detector_tightness for spec in executable]
        if len(tightness) != len(set(tightness)):
            diagnostics.extend(
                OriginSpecDiagnostic(
                    code="ambiguous_detector_tightness",
                    message=f"{spec.origin.value}: detector tightness must be unique among executable OriginSpecs",
                    origin=spec.origin,
                    owner_path=spec.declaration.owner_path,
                    repair_command=spec.declaration.repair_command,
                )
                for spec in executable
            )
        return tuple(sorted(diagnostics, key=lambda item: (item.origin.value, item.code)))


def _declaration(origin: Origin, *, lifecycle: OriginLifecycle, discovery: str) -> DeclarationSpec:
    return DeclarationSpec(
        declaration_id=f"origin.{origin.value}",
        family_id=f"source-origin-admission:{lifecycle}",
        public_name=origin.value,
        owner_path="polylogue/sources/origin_specs.py",
        compatibility=CompatibilityKey(
            identity="public-origin",
            lifecycle=lifecycle,
            authority="source-admission",
            access_result_shape="normalized-session-or-honest-nonexecution",
            durability="source-evidence",
        ),
        producer="polylogue.sources.origin_specs",
        role_gate="archive:read",
        schema_ref="polylogue.core.enums.Origin",
        discovery_text=discovery,
        repair_command="devtools test tests/unit/sources/test_origin_specs.py",
        handlers=(
            HandlerBinding(
                surface="source-admission",
                owner_path="polylogue/sources/origin_specs.py",
                symbol="ORIGIN_SPECS",
                binding_key=origin.value,
            ),
        ),
        outputs=(
            OutputSpec("coverage", "completeness-row", "ProviderPackageCompletenessPayload", "provider-completeness"),
            OutputSpec("docs", "origin-catalog", "Origin", "docs/provider-origin-identity.md"),
        ),
        examples=(ExampleSpec("discover", discovery),),
        completeness_edges=(
            CompletenessEdge(
                producer=f"origin.{origin.value}",
                consumer="polylogue.sources.dispatch",
                kind="detector-parity",
                owner_path="polylogue/sources/dispatch.py",
            ),
        ),
    )


def _claude_code_spec() -> OriginSpec:
    origin = Origin.CLAUDE_CODE_SESSION
    return OriginSpec(
        origin=origin,
        declaration=_declaration(origin, lifecycle="executable", discovery="Claude Code JSONL and sidecar admission."),
        lifecycle="executable",
        acquisition_modes=("export-jsonl", "live-jsonl", "sidecar", "workflow-orchestration"),
        provider_wires=(Provider.CLAUDE_CODE,),
        collision_policy=None,
        detector_tightness=60,
        parser_paths=("polylogue/sources/parsers/claude/code_parser.py",),
        stream_parser_path="polylogue/sources/parsers/claude/code_parser.py:parse_code_stream",
        assembly_paths=("polylogue/sources/dispatch.py:merge_parsed_session_chunks",),
        fixture_paths=(
            "tests/unit/sources/test_parsers_claude_code_artifacts.py",
            "tests/unit/sources/test_assembly_claude_code_history.py",
        ),
        coverage_refs=("provider-package:claude-code-session/export-jsonl@v1",),
        fidelity_notes=("Streaming JSONL retains source ordering; sidecars require separate authority admission.",),
        semantic_reparse=(
            "reparse Claude Code sessions and re-inventory workflow artifacts when the Claude parser, "
            "orchestration artifact parser, or sidecar assembly fingerprint changes"
        ),
        artifact_rules=(
            OriginArtifactRule(
                kind="workflow_run_snapshot",
                path_pattern=r"(?:^|/)workflows/[^/]+\.json$",
                parse_policy="fact",
                parser_path="polylogue/sources/parsers/claude/orchestration.py:parse_claude_orchestration_artifact",
                coverage_role="run_snapshot",
                fidelity_note="Authoritative mutable workflow run snapshot; every observed revision is retained.",
                path_suffixes=(".json",),
            ),
            OriginArtifactRule(
                kind="workflow_journal",
                path_pattern=r"(?:^|/)subagents/workflows/[^/]+/journal\.jsonl$",
                parse_policy="fact",
                parser_path="polylogue/sources/parsers/claude/orchestration.py:parse_claude_orchestration_artifact",
                coverage_role="journal",
                fidelity_note=(
                    "Append-only workflow journal; content keys and unresolved references remain provider evidence."
                ),
                path_suffixes=(".jsonl",),
            ),
            OriginArtifactRule(
                kind="agent_transcript",
                path_pattern=r"(?:^|/)subagents/(?:[^/]+/)*agent-[^/]+\.(?:jsonl|ndjson)$",
                parse_policy="session",
                parser_path="polylogue/sources/parsers/claude/code_parser.py:parse_code_stream",
                coverage_role="attempt_transcript",
                fidelity_note="Attempt transcript is a session only when provider workflow evidence links it to a run.",
                path_suffixes=(".jsonl", ".ndjson"),
            ),
            OriginArtifactRule(
                kind="agent_sidecar_meta",
                path_pattern=r"(?:^|/)subagents/(?:[^/]+/)*agent-[^/]+\.meta\.json$",
                parse_policy="fact",
                parser_path="polylogue/sources/parsers/claude/orchestration.py:parse_claude_orchestration_artifact",
                coverage_role="attempt_meta",
                fidelity_note="Agent metadata never fabricates a transcript pair; missing peers are coverage gaps.",
                path_suffixes=(".json",),
            ),
            OriginArtifactRule(
                kind="adopt_manifest",
                path_pattern=r"(?:^|/)jobs/[^/]+/adopt\.json$",
                parse_policy="fact",
                parser_path="polylogue/sources/parsers/claude/orchestration.py:parse_claude_orchestration_artifact",
                coverage_role="adopt_manifest",
                fidelity_note="Recovery manifest preserves resume/adoption evidence without asserting completed work.",
                path_suffixes=(".json",),
            ),
            OriginArtifactRule(
                kind="coordinator_session_stream",
                path_pattern=r"(?:^|/)projects/[^/]+/[^/]+\.(?:jsonl|ndjson)$",
                parse_policy="session",
                parser_path="polylogue/sources/parsers/claude/code_parser.py:parse_code_stream",
                coverage_role="coordinator_invocation_stream",
                fidelity_note=(
                    "Coordinator streams retain authored prompts and Workflow tool-use events; "
                    "child topology alone never establishes run membership."
                ),
                path_suffixes=(".jsonl", ".ndjson"),
            ),
        ),
    )


def artifact_rule_for_path(provider: Provider, source_path: str) -> OriginArtifactRule | None:
    """Return the owning OriginSpec artifact rule for a native source path."""

    for spec in ORIGIN_SPECS:
        if provider not in spec.provider_wires:
            continue
        for rule in spec.artifact_rules:
            if rule.matches(source_path):
                return rule
    return None


def artifact_suffixes_for_provider(
    provider: Provider,
    *,
    defaults: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Project live-acquisition suffixes from the owning OriginSpec rules.

    Acquisition may add a generic default, but provider artifact families must
    not maintain a second suffix inventory beside OriginSpec.
    """

    suffixes = list(defaults)
    for spec in ORIGIN_SPECS:
        if provider not in spec.provider_wires:
            continue
        for rule in spec.artifact_rules:
            suffixes.extend(rule.path_suffixes)
    return tuple(dict.fromkeys(suffix.lower() for suffix in suffixes))


def _chatgpt_spec() -> OriginSpec:
    origin = Origin.CHATGPT_EXPORT
    return OriginSpec(
        origin=origin,
        declaration=_declaration(
            origin, lifecycle="executable", discovery="ChatGPT document and bundled export admission."
        ),
        lifecycle="executable",
        acquisition_modes=("takeout-json", "bundle", "browser-capture"),
        provider_wires=(Provider.CHATGPT,),
        collision_policy=None,
        detector_tightness=70,
        parser_paths=("polylogue/sources/parsers/chatgpt.py",),
        stream_parser_path=None,
        assembly_paths=("polylogue/sources/dispatch.py:_lower_payload_specs",),
        fixture_paths=("tests/unit/sources/test_parsers_chatgpt.py", "tests/data/golden/chatgpt-simple.md"),
        coverage_refs=("provider-package:chatgpt-export/takeout-json@v1",),
        fidelity_notes=("Browser capture remains an acquisition mode and is not a new public origin.",),
        semantic_reparse="reparse when ChatGPT document parsing fingerprints change",
    )


def _grok_spec() -> OriginSpec:
    origin = Origin.GROK_EXPORT
    return OriginSpec(
        origin=origin,
        declaration=_declaration(
            origin, lifecycle="reserved", discovery="Reserved Grok export origin; no parser is admitted."
        ),
        lifecycle="reserved",
        acquisition_modes=("export-unknown",),
        provider_wires=(Provider.GROK,),
        collision_policy=None,
        detector_tightness=None,
        parser_paths=(),
        stream_parser_path=None,
        assembly_paths=(),
        fixture_paths=("tests/unit/sources/test_origin_specs.py",),
        coverage_refs=("origin:grok-export:reserved",),
        fidelity_notes=("No parser is wired; discovery must report reserved status rather than infer completeness.",),
        semantic_reparse="admission requires a separate parser and fixture contract",
    )


def _executable_spec(
    origin: Origin,
    *,
    provider: Provider,
    tightness: int,
    discovery: str,
    acquisition_modes: tuple[str, ...],
    parser_paths: tuple[str, ...],
    fixture_paths: tuple[str, ...],
    stream_parser_path: str | None = None,
    assembly_paths: tuple[str, ...] = (),
    fidelity_notes: tuple[str, ...] = (),
) -> OriginSpec:
    return OriginSpec(
        origin=origin,
        declaration=_declaration(origin, lifecycle="executable", discovery=discovery),
        lifecycle="executable",
        acquisition_modes=acquisition_modes,
        provider_wires=(provider,),
        collision_policy=None,
        detector_tightness=tightness,
        parser_paths=parser_paths,
        stream_parser_path=stream_parser_path,
        assembly_paths=assembly_paths,
        fixture_paths=fixture_paths,
        coverage_refs=(f"origin:{origin.value}:admitted",),
        fidelity_notes=fidelity_notes,
        semantic_reparse=f"reparse when {origin.value} parser fingerprints change",
    )


def _codex_spec() -> OriginSpec:
    return _executable_spec(
        Origin.CODEX_SESSION,
        provider=Provider.CODEX,
        tightness=50,
        discovery="Codex session JSONL admission.",
        acquisition_modes=("session-jsonl",),
        parser_paths=("polylogue/sources/parsers/codex.py",),
        fixture_paths=("tests/unit/sources/test_parsers_codex.py", "tests/data/codex_event_stream"),
        stream_parser_path="polylogue/sources/parsers/codex.py:parse_codex_stream",
    )


def _gemini_cli_spec() -> OriginSpec:
    return _executable_spec(
        Origin.GEMINI_CLI_SESSION,
        provider=Provider.GEMINI_CLI,
        tightness=10,
        discovery="Gemini CLI local-agent document admission.",
        acquisition_modes=("local-agent-document",),
        parser_paths=("polylogue/sources/parsers/local_agent.py",),
        fixture_paths=("tests/unit/sources/test_parsers_local_agent.py",),
    )


def _hermes_spec() -> OriginSpec:
    return _executable_spec(
        Origin.HERMES_SESSION,
        provider=Provider.HERMES,
        tightness=20,
        discovery="Hermes state database plus NeMo Relay ATIF/ATOF observer admission.",
        acquisition_modes=("state-db", "atif-spans", "atof-jsonl"),
        parser_paths=("polylogue/sources/parsers/hermes_state.py", "polylogue/sources/parsers/hermes_spans.py"),
        fixture_paths=(
            "tests/unit/sources/test_parsers_local_agent.py",
            "tests/unit/sources/parsers/test_hermes_spans.py",
            "tests/fixtures/hermes/atif/nemo_relay_atif_v1.7_real_redacted.json",
            "tests/fixtures/hermes/atof/nemo_relay_atof_v0.1_real_redacted.jsonl",
        ),
        stream_parser_path="polylogue/sources/parsers/hermes_spans.py:parse_atof_stream",
    )


def _antigravity_spec() -> OriginSpec:
    return _executable_spec(
        Origin.ANTIGRAVITY_SESSION,
        provider=Provider.ANTIGRAVITY,
        tightness=30,
        discovery="Antigravity language-server export admission.",
        acquisition_modes=("language-server-export",),
        parser_paths=("polylogue/sources/parsers/antigravity.py",),
        fixture_paths=(
            "tests/unit/sources/test_antigravity_language_server.py",
            "tests/unit/sources/parsers/test_antigravity.py",
        ),
    )


def _beads_spec() -> OriginSpec:
    return _executable_spec(
        Origin.BEADS_ISSUE,
        provider=Provider.BEADS,
        tightness=40,
        discovery="Beads issue export admission.",
        acquisition_modes=("issue-jsonl",),
        parser_paths=("polylogue/sources/parsers/beads.py",),
        fixture_paths=("tests/unit/sources/parsers/test_beads.py",),
        stream_parser_path="polylogue/sources/parsers/beads.py:parse_beads_stream",
    )


def _claude_ai_spec() -> OriginSpec:
    return _executable_spec(
        Origin.CLAUDE_AI_EXPORT,
        provider=Provider.CLAUDE_AI,
        tightness=80,
        discovery="Claude AI document export admission.",
        acquisition_modes=("export-json",),
        parser_paths=("polylogue/sources/parsers/claude/ai_parser.py",),
        fixture_paths=("tests/unit/sources/test_parsers_claude_ai_catalog.py",),
    )


def _aistudio_drive_spec() -> OriginSpec:
    origin = Origin.AISTUDIO_DRIVE
    return OriginSpec(
        origin=origin,
        declaration=_declaration(origin, lifecycle="executable", discovery="AI Studio and Drive export admission."),
        lifecycle="executable",
        acquisition_modes=("drive-like-export",),
        provider_wires=(Provider.GEMINI, Provider.DRIVE),
        collision_policy="Gemini and Drive wire families intentionally normalize to one public AI Studio origin.",
        detector_tightness=90,
        parser_paths=("polylogue/sources/parsers/drive.py",),
        stream_parser_path=None,
        assembly_paths=("polylogue/sources/dispatch.py:_lower_payload_specs",),
        fixture_paths=("tests/unit/sources/test_parsers_drive.py", "tests/data/gemini_chunked_prompt"),
        coverage_refs=("origin:aistudio-drive:admitted",),
        fidelity_notes=("Provider reverse mapping remains intentionally non-injective.",),
        semantic_reparse="reparse when Drive parser fingerprints change",
    )


def _unknown_spec() -> OriginSpec:
    origin = Origin.UNKNOWN_EXPORT
    return OriginSpec(
        origin=origin,
        declaration=_declaration(
            origin, lifecycle="compatibility-only", discovery="Unknown fallback origin admission."
        ),
        lifecycle="compatibility-only",
        acquisition_modes=("fallback", "browser-capture"),
        provider_wires=(Provider.UNKNOWN,),
        collision_policy=None,
        detector_tightness=None,
        parser_paths=(),
        stream_parser_path=None,
        assembly_paths=(),
        fixture_paths=("tests/unit/sources/test_origin_specs.py",),
        coverage_refs=("origin:unknown-export:fallback",),
        fidelity_notes=(
            "Fallback is explicit; browser capture resolves a provider-specific origin before archive materialization.",
        ),
        semantic_reparse="no direct parser; retain unknown evidence until a concrete source adapter is admitted",
    )


def _completeness_mode(
    package_ref: str,
    capture_mode: str,
    provider_wire: Provider | None,
    maturity: OriginCompletenessMaturity,
    *,
    detector_paths: tuple[str, ...],
    raw_model_paths: tuple[str, ...],
    parser_paths: tuple[str, ...],
    normalizer_paths: tuple[str, ...],
    fixture_paths: tuple[str, ...],
    schema_paths: tuple[str, ...],
    docs_paths: tuple[str, ...],
    privacy_paths: tuple[str, ...] = ("docs/provider-origin-identity.md",),
    caveats: tuple[str, ...] = (),
) -> OriginCompletenessMode:
    return OriginCompletenessMode(
        package_ref=package_ref,
        capture_mode=capture_mode,
        provider_wire=provider_wire,
        maturity=maturity,
        detector_paths=detector_paths,
        raw_model_paths=raw_model_paths,
        parser_paths=parser_paths,
        normalizer_paths=normalizer_paths,
        fixture_paths=fixture_paths,
        schema_paths=schema_paths,
        docs_paths=docs_paths,
        privacy_paths=privacy_paths,
        caveats=caveats,
    )


_ORIGIN_COMPLETENESS_MODES: dict[Origin, tuple[OriginCompletenessMode, ...]] = {
    Origin.CLAUDE_CODE_SESSION: (
        _completeness_mode(
            "provider-package:claude-code-session/export-jsonl@v1",
            "export-jsonl",
            Provider.CLAUDE_CODE,
            "accepted",
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
        ),
    ),
    Origin.CODEX_SESSION: (
        _completeness_mode(
            "provider-package:codex-session/session-jsonl@v1",
            "session-jsonl",
            Provider.CODEX,
            "accepted",
            detector_paths=("polylogue/sources/parsers/codex.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/providers/codex.py",),
            parser_paths=("polylogue/sources/parsers/codex.py",),
            normalizer_paths=("polylogue/sources/parsers/base_support.py",),
            fixture_paths=("tests/unit/sources/test_parsers_codex.py", "tests/data/codex_event_stream"),
            schema_paths=("polylogue/schemas/providers/codex/catalog.json",),
            docs_paths=("docs/providers/openai-codex.md",),
        ),
    ),
    Origin.GEMINI_CLI_SESSION: (
        _completeness_mode(
            "provider-package:gemini-cli-session/local-agent-document@v1",
            "local-agent-document",
            Provider.GEMINI_CLI,
            "accepted",
            detector_paths=("polylogue/sources/parsers/local_agent.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/parsers/local_agent.py",),
            parser_paths=("polylogue/sources/parsers/local_agent.py",),
            normalizer_paths=("polylogue/sources/parsers/base_support.py",),
            fixture_paths=("tests/unit/sources/test_parsers_local_agent.py",),
            schema_paths=("polylogue/schemas/providers/gemini-cli/catalog.json",),
            docs_paths=("docs/providers/README.md",),
        ),
    ),
    Origin.HERMES_SESSION: (
        _completeness_mode(
            "provider-package:hermes-session/state-db@v1",
            "state-db",
            Provider.HERMES,
            "accepted",
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
        ),
    ),
    Origin.ANTIGRAVITY_SESSION: (
        _completeness_mode(
            "provider-package:antigravity-session/language-server-export@v1",
            "language-server-export",
            Provider.ANTIGRAVITY,
            "accepted",
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
        ),
    ),
    Origin.BEADS_ISSUE: (
        _completeness_mode(
            "provider-package:beads-issue/issue-jsonl@v1",
            "issue-jsonl",
            Provider.BEADS,
            "accepted",
            detector_paths=("polylogue/sources/parsers/beads.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/parsers/beads.py",),
            parser_paths=("polylogue/sources/parsers/beads.py",),
            normalizer_paths=("polylogue/sources/parsers/base_support.py",),
            fixture_paths=("tests/unit/sources/parsers/test_beads.py",),
            schema_paths=(),
            docs_paths=("docs/architecture.md",),
            caveats=("Beads is a non-chat issue artifact origin.",),
        ),
    ),
    Origin.GROK_EXPORT: (
        _completeness_mode(
            "provider-package:grok-export/reserved@v1",
            "reserved",
            Provider.GROK,
            "reserved",
            detector_paths=(),
            raw_model_paths=(),
            parser_paths=(),
            normalizer_paths=(),
            fixture_paths=("tests/unit/sources/test_origin_specs.py",),
            schema_paths=(),
            docs_paths=("docs/provider-origin-identity.md",),
            caveats=("Reserved origin: parser admission has not been declared.",),
        ),
    ),
    Origin.CHATGPT_EXPORT: (
        _completeness_mode(
            "provider-package:chatgpt-export/takeout-json@v1",
            "takeout-json",
            Provider.CHATGPT,
            "accepted",
            detector_paths=("polylogue/sources/parsers/chatgpt.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/parsers/chatgpt.py",),
            parser_paths=("polylogue/sources/parsers/chatgpt.py",),
            normalizer_paths=("polylogue/sources/parsers/base_support.py",),
            fixture_paths=("tests/unit/sources/test_parsers_chatgpt.py", "tests/data/golden/chatgpt-simple.md"),
            schema_paths=("polylogue/schemas/providers/chatgpt/catalog.json",),
            docs_paths=("docs/providers/chatgpt.md",),
        ),
    ),
    Origin.CLAUDE_AI_EXPORT: (
        _completeness_mode(
            "provider-package:claude-ai-export/export-json@v1",
            "export-json",
            Provider.CLAUDE_AI,
            "accepted",
            detector_paths=("polylogue/sources/parsers/claude/ai_parser.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/providers/claude_ai.py",),
            parser_paths=("polylogue/sources/parsers/claude/ai_parser.py",),
            normalizer_paths=("polylogue/sources/parsers/claude/common.py",),
            fixture_paths=("tests/unit/sources/test_parsers_claude_ai_catalog.py",),
            schema_paths=("polylogue/schemas/providers/claude-ai/catalog.json",),
            docs_paths=("docs/providers/claude-ai.md",),
        ),
    ),
    Origin.AISTUDIO_DRIVE: (
        _completeness_mode(
            "provider-package:aistudio-drive/drive-export@v1",
            "drive-like-export",
            Provider.GEMINI,
            "accepted",
            detector_paths=("polylogue/sources/parsers/drive.py", "polylogue/sources/dispatch.py"),
            raw_model_paths=("polylogue/sources/providers/gemini_message.py",),
            parser_paths=("polylogue/sources/parsers/drive.py",),
            normalizer_paths=("polylogue/sources/parsers/drive_support.py",),
            fixture_paths=("tests/unit/sources/test_parsers_drive.py", "tests/data/gemini_chunked_prompt"),
            schema_paths=("polylogue/schemas/providers/gemini/catalog.json",),
            docs_paths=("docs/providers/gemini.md",),
        ),
    ),
    Origin.UNKNOWN_EXPORT: (
        _completeness_mode(
            "provider-package:browser-capture/live-receiver@v1",
            "browser-capture-live-receiver",
            None,
            "proposed",
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
    ),
}


def _with_completeness_modes(spec: OriginSpec) -> OriginSpec:
    return replace(spec, completeness_modes=_ORIGIN_COMPLETENESS_MODES[spec.origin])


ORIGIN_SPEC_REGISTRY = OriginSpecRegistry()
for _spec in (
    _claude_code_spec(),
    _codex_spec(),
    _gemini_cli_spec(),
    _hermes_spec(),
    _antigravity_spec(),
    _beads_spec(),
    _grok_spec(),
    _chatgpt_spec(),
    _claude_ai_spec(),
    _aistudio_drive_spec(),
    _unknown_spec(),
):
    ORIGIN_SPEC_REGISTRY.register(_with_completeness_modes(_spec))
ORIGIN_SPECS = ORIGIN_SPEC_REGISTRY.specs()


def origin_specs() -> tuple[OriginSpec, ...]:
    """Return the stable public-origin admission projection."""

    return ORIGIN_SPECS


def validate_dispatch_precedence(provider_order: tuple[Provider, ...]) -> tuple[OriginSpecDiagnostic, ...]:
    """Check that the legacy detector branch order honors declared tightness."""

    positions = {provider: index for index, provider in enumerate(provider_order)}
    diagnostics: list[OriginSpecDiagnostic] = []
    executable = [spec for spec in ORIGIN_SPECS if spec.lifecycle == "executable"]
    for spec in executable:
        if not spec.provider_wires or spec.provider_wires[0] not in positions:
            diagnostics.append(
                OriginSpecDiagnostic(
                    code="missing_dispatch_provider",
                    message=f"{spec.origin.value}: provider is absent from dispatch precedence",
                    origin=spec.origin,
                    owner_path=spec.declaration.owner_path,
                    repair_command=spec.declaration.repair_command,
                )
            )
    ordered = sorted(executable, key=lambda spec: spec.detector_tightness or 0)
    for left, right in zip(ordered, ordered[1:], strict=False):
        left_position = positions.get(left.provider_wires[0])
        right_position = positions.get(right.provider_wires[0])
        if left_position is not None and right_position is not None and left_position > right_position:
            diagnostics.append(
                OriginSpecDiagnostic(
                    code="dispatch_tightness_mismatch",
                    message=(
                        f"{left.origin.value}: declared tighter than {right.origin.value} but appears later in dispatch"
                    ),
                    origin=left.origin,
                    owner_path="polylogue/sources/dispatch.py",
                    repair_command="devtools test tests/unit/sources/test_origin_specs.py",
                )
            )
    return tuple(sorted(diagnostics, key=lambda item: (item.origin.value, item.code)))


__all__ = [
    "ORIGIN_SPECS",
    "ORIGIN_SPEC_REGISTRY",
    "OriginLifecycle",
    "OriginArtifactRule",
    "ArtifactParsePolicy",
    "OriginSpec",
    "OriginSpecDiagnostic",
    "OriginSpecRegistry",
    "origin_specs",
    "artifact_rule_for_path",
    "artifact_suffixes_for_provider",
    "validate_dispatch_precedence",
]

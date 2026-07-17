"""Executable source-admission declarations for public archive origins.

Provider adapters retain record-level parsing.  This module owns only the
cross-adapter admission contract: public origin vocabulary, acquisition modes,
detector tightness, registration/fixture evidence, fidelity, and reparse
consequences.  The initial pilots deliberately cover a streaming runtime, a
document export, and a reserved origin.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        acquisition_modes=("export-jsonl", "live-jsonl", "sidecar"),
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
        semantic_reparse="reparse when Claude Code parser or sidecar assembly fingerprints change",
    )


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


ORIGIN_SPEC_REGISTRY = OriginSpecRegistry()
for _spec in (_claude_code_spec(), _chatgpt_spec(), _grok_spec()):
    ORIGIN_SPEC_REGISTRY.register(_spec)
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
    "OriginSpec",
    "OriginSpecDiagnostic",
    "OriginSpecRegistry",
    "origin_specs",
    "validate_dispatch_precedence",
]

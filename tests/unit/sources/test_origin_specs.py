"""OriginSpec admission-kernel laws (polylogue-2qx.1.1)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.sources.assembly import get_assembly_spec
from polylogue.sources.dispatch import RECORD_DETECTOR_PROVIDER_ORDER, STREAM_RECORD_PROVIDERS
from polylogue.sources.origin_specs import (
    ORIGIN_SPEC_REGISTRY,
    ORIGIN_SPECS,
    OriginSpecRegistry,
    artifact_suffixes_for_provider,
    validate_assembly_spec_parity,
    validate_dispatch_precedence,
    validate_stream_parser_parity,
)


def test_origin_specs_cover_the_public_enum_and_admission_lifecycles() -> None:
    """Production dependency: source admission is one typed public-origin registry.

    Anti-vacuity mutation: removing a pilot's parser, fixture, coverage, or
    lifecycle binding makes registration reject its owning OriginSpec.
    """

    by_origin = {spec.origin: spec for spec in ORIGIN_SPECS}

    claude = by_origin[Origin.CLAUDE_CODE_SESSION]
    chatgpt = by_origin[Origin.CHATGPT_EXPORT]
    grok = by_origin[Origin.GROK_EXPORT]

    assert claude.stream_parser_path is not None
    assert {rule.kind for rule in claude.artifact_rules} == {
        "workflow_run_snapshot",
        "workflow_journal",
        "agent_transcript",
        "agent_sidecar_meta",
        "adopt_manifest",
        "coordinator_session_stream",
    }
    assert artifact_suffixes_for_provider(Provider.CLAUDE_CODE) == (".json", ".jsonl", ".ndjson")
    assert claude.detector_tightness == 60
    assert chatgpt.detector_tightness == 70
    assert chatgpt.acquisition_modes == ("takeout-json", "bundle", "browser-capture")
    assert grok.lifecycle == "executable"
    assert grok.parser_paths == ("polylogue/sources/parsers/grok.py",)
    assert grok.detector_tightness == 85
    assert set(by_origin) == set(Origin)
    assert by_origin[Origin.UNKNOWN_EXPORT].lifecycle == "compatibility-only"
    assert by_origin[Origin.AISTUDIO_DRIVE].provider_wires == (Provider.GEMINI, Provider.DRIVE)
    assert ORIGIN_SPEC_REGISTRY.diagnostics() == ()


def test_origin_specs_are_parity_checked_against_current_dispatch_order() -> None:
    assert validate_dispatch_precedence(RECORD_DETECTOR_PROVIDER_ORDER) == ()


def test_origin_spec_reports_source_locatable_missing_dispatch_provider() -> None:
    diagnostics = validate_dispatch_precedence((Provider.CODEX,))

    assert {item.code for item in diagnostics} == {"missing_dispatch_provider"}
    assert {item.origin for item in diagnostics} == {
        spec.origin
        for spec in ORIGIN_SPECS
        if spec.lifecycle == "executable" and spec.origin is not Origin.CODEX_SESSION
    }
    assert {item.owner_path for item in diagnostics} == {"polylogue/sources/origin_specs.py"}


def test_origin_spec_rejects_missing_fixture_and_noninjective_collision_without_policy() -> None:
    claude = next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.CLAUDE_CODE_SESSION)
    registry = OriginSpecRegistry()

    with pytest.raises(ValueError, match="missing fixture"):
        registry.register(replace(claude, fixture_paths=()))
    with pytest.raises(ValueError, match="collision policy"):
        registry.register(replace(claude, provider_wires=(Provider.CLAUDE_CODE, Provider.DRIVE)))


def test_origin_spec_rejects_undeclared_coverage() -> None:
    """Production dependency: registration requires a non-empty coverage_refs.

    Anti-vacuity mutation: an OriginSpec with no coverage evidence must be
    rejected rather than silently admitted with an unproven coverage claim.
    """
    claude = next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.CLAUDE_CODE_SESSION)
    registry = OriginSpecRegistry()

    with pytest.raises(ValueError, match="missing coverage declaration"):
        registry.register(replace(claude, coverage_refs=()))


def test_origin_spec_rejects_leaked_provider_token_as_public_name() -> None:
    """Production dependency: public origin names must not collide with Provider-wire tokens.

    A public origin name equal to a raw Provider-wire spelling (e.g.
    ``"claude-code"`` instead of ``"claude-code-session"``) would let
    provider-wire vocabulary leak onto the public origin surface, violating
    the doctrine in docs/provider-origin-identity.md. Constructed directly
    against a colliding declaration (bypassing the module's ``_declaration``
    helper, which always derives ``public_name`` from ``origin.value``) since
    no real ``Origin`` member currently collides with a ``Provider`` member.
    """
    claude = next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.CLAUDE_CODE_SESSION)
    registry = OriginSpecRegistry()
    colliding = replace(claude, declaration=replace(claude.declaration, public_name=Provider.CLAUDE_CODE.value))

    with pytest.raises(ValueError, match="leaks a"):
        registry.register(colliding)


def test_origin_spec_supports_reserved_lifecycle_without_parser_or_tightness() -> None:
    """Production dependency: the reserved lifecycle state admits an origin with no parser yet.

    ``lifecycle="reserved"`` is the state OriginSpec offers for an origin whose
    public token is claimed but has no confirmed export shape (the original
    Grok pilot before polylogue-611/#3201 shipped a real parser). Every
    current ``Origin`` member is admitted as executable or compatibility-only,
    so this proves the reserved path against a synthetic variant of a real
    spec rather than a live production origin.

    Anti-vacuity mutation: dropping ``lifecycle="reserved"`` back to
    ``"executable"`` on this synthetic spec without also supplying
    ``detector_tightness``/``parser_paths`` makes registration reject it
    (see ``test_origin_specs_cover_the_public_enum_and_admission_lifecycles``'s
    sibling executable-path checks), proving the two lifecycles are genuinely
    different admission contracts, not a cosmetic label.
    """
    grok = next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.GROK_EXPORT)
    reserved_variant = replace(
        grok,
        lifecycle="reserved",
        detector_tightness=None,
        parser_paths=(),
        stream_parser_path=None,
        assembly_paths=(),
    )
    registry = OriginSpecRegistry()

    registered = registry.register(reserved_variant)

    assert registered.lifecycle == "reserved"
    assert registered.parser_paths == ()
    assert registered.detector_tightness is None
    # The reserved variant still carries real coverage/fixture evidence --
    # reserved means "no parser yet", not "no admission evidence at all".
    assert registered.coverage_refs
    assert registered.fixture_paths


def test_origin_specs_are_parity_checked_against_stream_record_providers() -> None:
    """Production dependency: declared stream_parser_path presence matches dispatch's stream-record set.

    Anti-vacuity mutation: passing an empty stream-record-provider set makes
    every stream-capable executable OriginSpec (Claude Code, Codex, Beads,
    Hermes) report a ``stream_parser_parity_mismatch`` diagnostic.
    """
    assert validate_stream_parser_parity(STREAM_RECORD_PROVIDERS) == ()

    diagnostics = validate_stream_parser_parity(frozenset())

    assert {item.code for item in diagnostics} == {"stream_parser_parity_mismatch"}
    stream_origins = {
        spec.origin
        for spec in ORIGIN_SPECS
        if spec.lifecycle == "executable" and any(p in STREAM_RECORD_PROVIDERS for p in spec.provider_wires)
    }
    assert {item.origin for item in diagnostics} == stream_origins
    assert stream_origins  # sanity: the production set is non-empty today


def test_origin_specs_declare_the_claude_and_codex_assembly_extension_hooks() -> None:
    """Production dependency: Claude Code and Codex admit their sidecar/title enrichment hook via OriginSpec.

    This is the one typed admission point polylogue-2qx.2, polylogue-j2zz, and
    polylogue-ih67 build their assembly/orchestration/title/action extensions
    on, rather than a private inventory.
    """
    by_origin = {spec.origin: spec for spec in ORIGIN_SPECS}

    assert by_origin[Origin.CLAUDE_CODE_SESSION].assembly_spec_path == (
        "polylogue/sources/assembly_claude_code.py:ClaudeCodeAssemblySpec"
    )
    assert by_origin[Origin.CODEX_SESSION].assembly_spec_path == (
        "polylogue/sources/assembly_codex.py:CodexAssemblySpec"
    )
    assert by_origin[Origin.AISTUDIO_DRIVE].assembly_spec_path == (
        "polylogue/sources/assembly_gemini.py:GeminiAssemblySpec"
    )
    assert by_origin[Origin.GEMINI_CLI_SESSION].assembly_spec_path is None
    assert by_origin[Origin.CHATGPT_EXPORT].assembly_spec_path is None


def test_origin_specs_are_parity_checked_against_the_live_assembly_registry() -> None:
    """Production dependency: declared assembly_spec_path matches polylogue.sources.assembly.get_assembly_spec.

    Anti-vacuity mutation: a resolver that never returns an assembly spec makes
    every origin that declares assembly_spec_path (Claude Code, Codex, AI
    Studio/Drive) report a mismatch.
    """
    assert validate_assembly_spec_parity(get_assembly_spec) == ()

    diagnostics = validate_assembly_spec_parity(lambda _provider: None)

    assert {item.code for item in diagnostics} == {"assembly_spec_parity_mismatch"}
    declared_origins = {spec.origin for spec in ORIGIN_SPECS if spec.assembly_spec_path is not None}
    assert {item.origin for item in diagnostics} == declared_origins
    assert declared_origins  # sanity: the production set is non-empty today


def test_every_origin_spec_declares_a_display_description() -> None:
    """Production dependency: CLI --origin shell completion derives its help text from OriginSpec.

    Anti-vacuity: blanking a display_description (or dropping a spec) shrinks
    the derived completion inventory below the full Origin vocabulary.
    """
    from polylogue.cli.shell_completion_values import _ORIGIN_DESCRIPTIONS

    for spec in ORIGIN_SPECS:
        assert spec.display_description.strip(), spec.origin.value

    assert set(_ORIGIN_DESCRIPTIONS) == {origin.value for origin in Origin}
    assert all(text.strip() for text in _ORIGIN_DESCRIPTIONS.values())

"""OriginSpec admission-kernel laws (polylogue-2qx.1.1)."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.sources.assembly import get_assembly_spec
from polylogue.sources.detection import DetectorBinding, DetectorBindingError, compile_detector_registry
from polylogue.sources.dispatch import STREAM_RECORD_PROVIDERS
from polylogue.sources.origin_specs import (
    DROPPED_VALUE_VOCABULARIES,
    ORIGIN_SPEC_REGISTRY,
    ORIGIN_SPECS,
    DroppedValueVocabulary,
    OriginSpecRegistry,
    artifact_suffixes_for_provider,
    check_dropped_value_vocabularies,
    detector_registry,
    lowering_fingerprint,
    parser_fingerprint_for_origin,
    public_origin_descriptions,
    public_origin_meanings,
    public_origin_tokens,
    schema_observed_leaf_values,
    undeclared_schema_values,
    validate_assembly_spec_parity,
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
        "tool_result_sidecar",
        "workflow_run_snapshot",
        "workflow_journal",
        "agent_transcript",
        "agent_sidecar_meta",
        "adopt_manifest",
        "coordinator_session_stream",
        "todo_snapshot",
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


def test_public_origin_projections_cover_declared_specs_coherently() -> None:
    """Public vocabulary and capability projections share OriginSpec ownership."""
    public = set(public_origin_tokens())
    meanings = dict(public_origin_meanings())
    descriptions = public_origin_descriptions()

    assert public <= {spec.origin.value for spec in ORIGIN_SPECS}
    assert set(meanings) == set(descriptions) == {spec.origin.value for spec in ORIGIN_SPECS}
    assert public == {spec.origin.value for spec in ORIGIN_SPECS if spec.public_filter}
    assert all(description for description in descriptions.values())
    assert all(spec.completeness_modes for spec in ORIGIN_SPECS)
    assert all(mode.package_ref and mode.capture_mode for spec in ORIGIN_SPECS for mode in spec.completeness_modes)


def test_origin_spec_metadata_change_propagates_to_public_manual_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual payloads read current declaration metadata instead of a copied list."""
    import polylogue.sources.origin_specs as origin_specs
    from polylogue.agent_integration.spec import integration_spec_payload

    target = next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.CODEX_SESSION)
    changed = replace(target, display_description="Codex sessions (updated declaration)")
    monkeypatch.setattr(
        origin_specs, "ORIGIN_SPECS", tuple(changed if spec is target else spec for spec in ORIGIN_SPECS)
    )

    assert dict(public_origin_meanings())[target.origin.value] == "Codex sessions (updated declaration)"
    payload = integration_spec_payload()
    origins = cast(list[dict[str, object]], payload["origins"])
    row = next(item for item in origins if item["token"] == target.origin.value)
    assert row["meaning"] == "Codex sessions (updated declaration)"


def test_parser_fingerprint_changes_when_a_normalizing_parser_helper_changes(tmp_path: Path) -> None:
    """Parser helper behavior is part of the persisted normalized-output contract.

    Mutation proof: changing the helper from stripping to case-folding changes
    the parser fingerprint, so a candidate stamped before that semantic change
    cannot satisfy the current-fingerprint comparison.
    """
    spec = next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.CODEX_SESSION)
    parser_source = tmp_path / "parser.py"
    helper_source = tmp_path / "support.py"
    parser_source.write_text(
        "from .support import normalize\n\ndef parse(payload):\n    return {'title': normalize(payload['title'])}\n",
        encoding="utf-8",
    )
    helper_source.write_text("def normalize(value):\n    return value.strip()\n", encoding="utf-8")
    synthetic = replace(spec, parser_paths=(str(parser_source),))

    before = synthetic.parser_fingerprint()
    helper_source.write_text("def normalize(value):\n    return value.casefold()\n", encoding="utf-8")
    after = synthetic.parser_fingerprint()

    assert before != after


def test_parser_fingerprint_changes_when_a_declared_assembly_helper_changes(tmp_path: Path) -> None:
    """Assembly enrichment is part of the origin's normalized output contract."""
    spec = next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.AISTUDIO_DRIVE)
    parser_source = tmp_path / "parser.py"
    assembly_source = tmp_path / "assembly.py"
    helper_source = tmp_path / "support.py"
    parser_source.write_text("def parse(payload):\n    return payload\n", encoding="utf-8")
    assembly_source.write_text(
        "from .support import enrich\n\n"
        "class AssemblySpec:\n"
        "    def enrich_session(self, session):\n"
        "        return enrich(session)\n",
        encoding="utf-8",
    )
    helper_source.write_text("def enrich(value):\n    return value.strip()\n", encoding="utf-8")
    synthetic = replace(
        spec,
        parser_paths=(str(parser_source),),
        assembly_spec_path=f"{assembly_source}:AssemblySpec",
    )

    before = synthetic.parser_fingerprint()
    helper_source.write_text("def enrich(value):\n    return value.strip().casefold()\n", encoding="utf-8")
    after = synthetic.parser_fingerprint()

    assert before != after


def test_lowering_fingerprint_changes_when_session_emitter_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seeded source semantics include the emitter that admits and enriches sessions."""
    import polylogue.sources.origin_specs as origin_specs

    source_root = tmp_path / "source-root"
    source_dir = source_root / "polylogue" / "sources"
    source_dir.mkdir(parents=True)
    emitter = source_dir / "emitter.py"
    emitter.write_text("def emit(payload):\n    return payload\n", encoding="utf-8")
    monkeypatch.setattr(origin_specs, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(origin_specs, "_LOWERING_FINGERPRINT_PATHS", ("polylogue/sources/emitter.py",))
    origin_specs._fingerprint_sources_cached.cache_clear()

    before = origin_specs.lowering_fingerprint()
    emitter.write_text("def emit(payload):\n    return {'session': payload}\n", encoding="utf-8")
    after = origin_specs.lowering_fingerprint()

    assert before != after


def test_production_fingerprints_are_stable_across_a_fresh_interpreter() -> None:
    current_parser = parser_fingerprint_for_origin(Origin.CODEX_SESSION)
    command = (
        "from polylogue.core.enums import Origin; "
        "from polylogue.sources.origin_specs import parser_fingerprint_for_origin; "
        "print(parser_fingerprint_for_origin(Origin.CODEX_SESSION))"
    )
    restarted = subprocess.check_output([sys.executable, "-c", command], text=True, cwd=Path.cwd()).strip()

    assert restarted == current_parser
    assert len(lowering_fingerprint()) == 64


def test_origin_specs_compile_the_production_detector_registry() -> None:
    registry = detector_registry()

    assert registry.by_mode
    assert all(spec.detector_bindings for spec in ORIGIN_SPECS if spec.lifecycle == "executable")


def test_detector_registry_rejects_broken_declarations_with_the_binding_id() -> None:
    codex = next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.CODEX_SESSION)
    broken = replace(
        codex,
        detector_bindings=(
            replace(codex.detector_bindings[0], predicate_path="polylogue.sources.dispatch:not_a_detector"),
            *codex.detector_bindings[1:],
        ),
    )
    duplicate = replace(
        codex,
        detector_bindings=(
            *codex.detector_bindings,
            DetectorBinding(
                binding_id="codex-record-pydantic",
                mode=codex.detector_bindings[0].mode,
                predicate_path=codex.detector_bindings[0].predicate_path,
                local_rank=99,
                evidence_label="duplicate",
                fixed_provider=Provider.CODEX,
            ),
        ),
    )

    with pytest.raises(DetectorBindingError, match="codex-record-pydantic"):
        compile_detector_registry(tuple(broken if spec is codex else spec for spec in ORIGIN_SPECS))
    with pytest.raises(DetectorBindingError, match="duplicate detector binding id"):
        compile_detector_registry(tuple(duplicate if spec is codex else spec for spec in ORIGIN_SPECS))


def test_origin_spec_rejects_missing_fixture_and_noninjective_collision_without_policy() -> None:
    claude = next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.CLAUDE_CODE_SESSION)
    registry = OriginSpecRegistry()

    with pytest.raises(ValueError, match="missing fixture"):
        registry.register(replace(claude, fixture_paths=()))
    with pytest.raises(ValueError, match="collision policy"):
        registry.register(replace(claude, provider_wires=(Provider.CLAUDE_CODE, Provider.DRIVE)))
    with pytest.raises(ValueError, match="detector binding"):
        registry.register(replace(claude, detector_bindings=()))


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
    # bd polylogue-0hwv / polylogue-dt5s: ChatGPT gained an assembly hook
    # (asset-name/sandbox-file sidecar resolution) -- it is no longer in the
    # "no assembly extension" cohort with Gemini CLI.
    assert by_origin[Origin.CHATGPT_EXPORT].assembly_spec_path == (
        "polylogue/sources/assembly_chatgpt.py:ChatGPTAssemblySpec"
    )
    assert by_origin[Origin.GEMINI_CLI_SESSION].assembly_spec_path is None


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


def test_dropped_value_vocabularies_match_the_real_parser_constant() -> None:
    """Production dependency: local_agent.py:_status_is_error's guessed success set.

    Anti-vacuity: the DroppedValueVocabulary declaration is a second copy of
    the parser's hardcoded set, not an import of it (origin_specs.py stays
    free of parser-internal imports, per this module's own docstring). This
    is what keeps that duplication honest -- if a future edit to
    local_agent.py's set diverges from the declared vocabulary without
    updating both, this test catches it instead of the declaration silently
    describing a set the parser no longer uses.
    """
    from polylogue.sources.parsers.local_agent import _status_is_error

    gemini_cli_vocab = next(vocab for vocab in DROPPED_VALUE_VOCABULARIES if vocab.schema_provider == "gemini-cli")
    for value in gemini_cli_vocab.declared_values:
        assert _status_is_error(value) is False, value
    # Everything outside the declared set that also isn't an error-marker
    # substring is classified None (unknown), not silently swept into "ok".
    assert _status_is_error("some_new_outcome_string") is None


def test_dropped_value_vocabularies_have_no_drift_against_the_committed_schema() -> None:
    """Production dependency: DROPPED_VALUE_VOCABULARIES stays honest against real evidence.

    Anti-vacuity: mutating this test to assert against a stale, hand-copied
    expected set (rather than calling check_dropped_value_vocabularies,
    which re-derives observed values from the live committed schema package
    on disk) would defeat the entire point of polylogue-2qx's ask -- this
    must read the real gzip schema file, not a fixture standing in for it.
    """
    assert check_dropped_value_vocabularies() == {}


def test_schema_observed_leaf_values_walks_array_and_scalar_segments() -> None:
    observed = schema_observed_leaf_values("gemini-cli", "messages[].toolCalls[].status")
    assert observed == {"success"}
    # A path with no committed schema evidence resolves to empty, not an error.
    assert schema_observed_leaf_values("gemini-cli", "no.such.path") == frozenset()
    assert schema_observed_leaf_values("no-such-provider", "status") == frozenset()


def test_undeclared_schema_values_flags_a_value_the_declaration_does_not_cover() -> None:
    narrow_vocab = DroppedValueVocabulary(
        field="test-only narrow gemini-cli status vocabulary",
        schema_provider="gemini-cli",
        schema_field_path="messages[].toolCalls[].status",
        declared_values=frozenset(),
        parser_path="test-only",
        reason="Anti-vacuity fixture: an empty declared set must show the real observed value as drift.",
    )
    assert undeclared_schema_values(narrow_vocab) == {"success"}

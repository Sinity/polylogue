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
    TopologyCapabilities,
    TopologyCapability,
    artifact_suffixes_for_provider,
    check_dropped_value_vocabularies,
    detector_registry,
    lowering_fingerprint,
    parser_fingerprint_for_origin,
    public_origin_descriptions,
    public_origin_meanings,
    public_origin_tokens,
    recognize_source_class,
    schema_observed_leaf_values,
    topology_capability_census,
    undeclared_schema_values,
    validate_assembly_spec_parity,
    validate_stream_parser_parity,
)
from polylogue.sources.source_walk import census_source_root


def test_hermes_source_class_recognition_is_structural_and_fails_closed(tmp_path: Path) -> None:
    """Suffixes enumerate candidates; declared Hermes shapes alone admit sessions.

    Anti-vacuity: replacing this recognizer with a suffix check would admit the
    renamed template and the unrelated JSON document below.
    """

    atif = tmp_path / "renamed-template.json"
    atif.write_text(
        '{"schema_version":"ATIF-v1.7","session_id":"s-1","steps":[]}',
        encoding="utf-8",
    )
    template = tmp_path / "config.json"
    template.write_text('{"name":"optional skill","version":1}', encoding="utf-8")
    unrelated = tmp_path / "session.json"
    unrelated.write_text('{"session_id":"copied","messages":[]}', encoding="utf-8")

    for path, expected in ((atif, "session"), (template, "unsupported"), (unrelated, "unsupported")):
        recognition = recognize_source_class(Provider.HERMES, path)
        assert recognition is not None
        assert recognition.source_class == expected


def test_hermes_source_class_recognition_accepts_atof_jsonl(tmp_path: Path) -> None:
    """The real ATOF envelope remains admitted independently of its basename."""

    path = tmp_path / "moved-events.jsonl"
    path.write_text(
        '{"atof_version":"0.1","kind":"mark","uuid":"u-1",'
        '"timestamp":"2026-08-26T00:00:00Z","name":"hermes.turn.start"}\n',
        encoding="utf-8",
    )
    result = recognize_source_class(Provider.HERMES, path)
    assert result is not None
    assert result.source_class == "session"


def test_source_class_recognition_defers_zip_members_to_archive_extraction(tmp_path: Path) -> None:
    """A provider archive is classified after its members are extracted."""

    archive = tmp_path / "export.zip"
    archive.write_bytes(b"not inspected during source-class admission")

    assert recognize_source_class(Provider.CODEX, archive) is None


def test_hermes_root_census_accounts_for_every_candidate_without_parsing(tmp_path: Path) -> None:
    """A broad root has one declared disposition for every enumerated file.

    Anti-vacuity: dropping a candidate from the walk or admitting every JSON
    by suffix changes the denominator or the typed disposition counts.
    """

    for index in range(40):
        (tmp_path / f"template-{index}.json").write_text('{"name":"optional skill"}', encoding="utf-8")
    (tmp_path / "moved-atif.json").write_text(
        '{"schema_version":"ATIF-v1.7","session_id":"s-1","steps":[]}', encoding="utf-8"
    )
    (tmp_path / "moved-atof.jsonl").write_text(
        '{"atof_version":"0.1","kind":"mark","uuid":"u-1",'
        '"timestamp":"2026-08-26T00:00:00Z","name":"hermes.turn.start"}\n',
        encoding="utf-8",
    )
    (tmp_path / "cache.sqlite").write_bytes(b"not sqlite")

    census = census_source_root(tmp_path, provider=Provider.HERMES)

    assert census.candidate_count == 43
    assert census.disposition_counts == {"session": 2, "non_session": 0, "unsupported": 41}
    assert census.accounted_count == census.candidate_count
    assert census.unexplained_candidates == ()
    assert census.is_complete
    assert census.inspection_bytes > 0
    assert census.inspection_seconds >= 0


def test_origin_specs_cover_the_public_enum_and_admission_lifecycles() -> None:
    """Production dependency: source admission is one typed public-origin registry.

    Anti-vacuity mutation: removing a pilot's parser, fixture, coverage, or
    lifecycle binding makes registration reject its owning OriginSpec.
    """

    by_origin = {spec.origin: spec for spec in ORIGIN_SPECS}

    claude = by_origin[Origin.CLAUDE_CODE_SESSION]
    chatgpt = by_origin[Origin.CHATGPT_EXPORT]
    grok = by_origin[Origin.GROK_EXPORT]
    antigravity = by_origin[Origin.ANTIGRAVITY_SESSION]
    beads = by_origin[Origin.BEADS_ISSUE]

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
    assert beads.lifecycle == "reserved"
    assert beads.public_filter is False
    assert beads.detector_tightness is None
    assert beads.detector_bindings == ()
    assert beads.parser_paths == ()
    assert beads.fixture_paths == ("tests/unit/sources/test_origin_specs.py",)
    assert beads.completeness_modes[0].maturity == "reserved"
    assert beads.completeness_modes[0].fixture_paths == beads.fixture_paths
    assert {rule.coverage_role for rule in antigravity.artifact_rules} == {
        "conversation_protobuf",
        "brain_metadata_sidecar",
        "brain_document",
    }
    assert artifact_suffixes_for_provider(Provider.ANTIGRAVITY) == (".pb", ".metadata.json", ".md")
    assert set(by_origin) == set(Origin)
    assert by_origin[Origin.UNKNOWN_EXPORT].lifecycle == "compatibility-only"
    assert by_origin[Origin.AISTUDIO_DRIVE].provider_wires == (Provider.GEMINI, Provider.DRIVE)
    assert ORIGIN_SPEC_REGISTRY.diagnostics() == ()


def test_topology_capability_census_is_complete_and_typed() -> None:
    """Every current origin has an explicit disposition for every dimension."""
    census = topology_capability_census()
    dimensions = {
        "message_parent",
        "message_branch_state",
        "session_parent_target",
        "inheritance_branch_point",
        "parent_dispatch",
    }
    assert set(census) == set(Origin)
    assert all(set(rows) == dimensions for rows in census.values())
    assert all(
        cell["state"] in {"carried", "positive-derived", "structurally-absent"}
        and cell["evidence"]
        and (cell["state"] != "structurally-absent" or cell["reason"])
        for rows in census.values()
        for cell in rows.values()
    )

    codex = census[Origin.CODEX_SESSION.value]
    claude = census[Origin.CLAUDE_CODE_SESSION.value]
    chatgpt = census[Origin.CHATGPT_EXPORT.value]
    hermes = census[Origin.HERMES_SESSION.value]
    claude_ai = census[Origin.CLAUDE_AI_EXPORT.value]
    aistudio_drive = census[Origin.AISTUDIO_DRIVE.value]
    assert codex["session_parent_target"]["state"] == "carried"
    assert claude["message_parent"]["state"] == "carried"
    assert claude["session_parent_target"]["state"] == "positive-derived"
    assert claude["message_branch_state"]["state"] == "positive-derived"
    assert chatgpt["message_parent"]["state"] == "carried"
    assert chatgpt["message_branch_state"]["state"] == "carried"
    assert hermes["session_parent_target"]["state"] == "carried"
    assert claude_ai["message_parent"]["state"] == "carried"
    assert claude_ai["message_branch_state"]["state"] == "positive-derived"
    assert aistudio_drive["message_parent"]["state"] == "carried"
    assert aistudio_drive["message_branch_state"]["state"] == "positive-derived"


def test_topology_capability_census_rejects_missing_or_duplicate_origins() -> None:
    """The census requires exactly one declaration for every current origin."""
    with pytest.raises(ValueError, match="cover every current Origin exactly once"):
        topology_capability_census(ORIGIN_SPECS[:-1])
    with pytest.raises(ValueError, match="cover every current Origin exactly once"):
        topology_capability_census((*ORIGIN_SPECS, ORIGIN_SPECS[0]))


def test_topology_capability_census_rejects_missing_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every topology declaration exposes all five dimensions."""
    original_as_dict = TopologyCapabilities.as_dict

    def missing_parent_dispatch(capabilities: TopologyCapabilities) -> dict[str, TopologyCapability]:
        return {
            name: capability for name, capability in original_as_dict(capabilities).items() if name != "parent_dispatch"
        }

    monkeypatch.setattr(TopologyCapabilities, "as_dict", missing_parent_dispatch)
    with pytest.raises(ValueError, match="topology capability census is incomplete"):
        topology_capability_census()


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    [
        ("state", "unknown", "capability state is not complete"),
        ("evidence", (), "topology capability lacks evidence"),
        ("reason", "", "structural absence lacks a reason"),
    ],
    ids=["unknown-state", "missing-evidence", "missing-absence-reason"],
)
def test_topology_capability_census_rejects_invalid_capability_cells(attribute: str, value: object, match: str) -> None:
    """The census validates cells even when a malformed declaration bypasses construction checks."""
    capability = TopologyCapability("structurally-absent", ("mutation",), "mutation")
    object.__setattr__(capability, attribute, value)
    specs = tuple(
        replace(
            spec,
            topology_capabilities=replace(spec.topology_capabilities, message_parent=capability),
        )
        if spec.origin is Origin.CODEX_SESSION
        else spec
        for spec in ORIGIN_SPECS
    )

    with pytest.raises(ValueError, match=match):
        topology_capability_census(specs)


def test_public_origin_projections_cover_declared_specs_coherently() -> None:
    """Public vocabulary and capability projections share OriginSpec ownership."""
    public = set(public_origin_tokens())
    meanings = dict(public_origin_meanings())
    descriptions = public_origin_descriptions()

    assert public <= {spec.origin.value for spec in ORIGIN_SPECS}
    assert set(meanings) == set(descriptions) == public
    assert public == {spec.origin.value for spec in ORIGIN_SPECS if spec.public_filter}
    assert all(description for description in descriptions.values())
    assert all(spec.completeness_modes for spec in ORIGIN_SPECS)
    assert all(mode.package_ref and mode.capture_mode for spec in ORIGIN_SPECS for mode in spec.completeness_modes)


def test_projection_only_origin_spec_changes_do_not_change_lowering_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public declarations are not executable lowering semantics."""
    import polylogue.sources.origin_specs as origin_specs

    before = lowering_fingerprint()
    changed_specs = tuple(
        replace(
            spec, display_description=f"{spec.display_description} (reworded)", public_filter=not spec.public_filter
        )
        for spec in ORIGIN_SPECS
    )
    monkeypatch.setattr(origin_specs, "ORIGIN_SPECS", changed_specs)
    origin_specs._fingerprint_sources_cached.cache_clear()

    assert origin_specs.lowering_fingerprint() == before
    target = next(spec for spec in changed_specs if spec.origin is Origin.CODEX_SESSION)
    assert target.parser_fingerprint() == parser_fingerprint_for_origin(Origin.CODEX_SESSION)


def test_source_ast_projection_mutation_is_closed_over_all_fingerprint_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reload-style source mutation changes only executable semantics."""
    source_root = tmp_path / "source-root"
    source_dir = source_root / "polylogue" / "sources"
    source_dir.mkdir(parents=True)
    origin_source = source_dir / "origin_specs.py"
    semantic_source = source_dir / "semantic.py"
    origin_source.write_text(
        "class OriginSpec:\n"
        "    def __init__(self, *, display_description, public_filter):\n"
        "        self.display_description = display_description\n"
        "        self.public_filter = public_filter\n"
        "DECLARATION = OriginSpec(display_description='before', public_filter=True)\n",
        encoding="utf-8",
    )
    semantic_source.write_text(
        "from .origin_specs import DECLARATION\n\n"
        "def execute(value):\n"
        "    return value + DECLARATION.display_description\n",
        encoding="utf-8",
    )
    import polylogue.sources.origin_specs as origin_specs_module

    monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(origin_specs_module, "_LOWERING_FINGERPRINT_PATHS", ("polylogue/sources/semantic.py",))
    monkeypatch.setattr(origin_specs_module, "_REPLAY_ROUTING_FINGERPRINT_PATHS", ("polylogue/sources/semantic.py",))
    monkeypatch.setattr(origin_specs_module, "_MATERIALIZER_FINGERPRINT_PATHS", ("polylogue/sources/semantic.py",))
    origin_specs_module._fingerprint_sources_cached.cache_clear()

    before = (
        origin_specs_module.lowering_fingerprint(),
        origin_specs_module.replay_routing_fingerprint(),
        origin_specs_module.materializer_fingerprint(),
    )
    origin_source.write_text(
        origin_source.read_text(encoding="utf-8").replace("before", "after").replace("True", "False"), encoding="utf-8"
    )
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    assert (
        origin_specs_module.lowering_fingerprint(),
        origin_specs_module.replay_routing_fingerprint(),
        origin_specs_module.materializer_fingerprint(),
    ) == before

    semantic_source.write_text(
        semantic_source.read_text(encoding="utf-8").replace("value + DECLARATION.display_description", "value"),
        encoding="utf-8",
    )
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    assert (
        origin_specs_module.lowering_fingerprint(),
        origin_specs_module.replay_routing_fingerprint(),
        origin_specs_module.materializer_fingerprint(),
    ) != before


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


def test_parser_fingerprints_ignore_diagnostic_module_but_lowering_and_materializer_do_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Diagnostic implementation changes do not stale parser cursors.

    Mutation proof: changing the diagnostic helper leaves both origin parser
    fingerprints unchanged, while the same change remains visible to the
    explicitly unfiltered lowering and materializer routes. Changing parser
    output logic still changes each origin's parser fingerprint.
    """
    source_root = tmp_path / "source-root"
    source_dir = source_root / "polylogue" / "sources"
    source_dir.mkdir(parents=True)
    logging_source = source_root / "polylogue" / "logging.py"
    logging_source.write_text("def get_logger():\n    return 'before'\n", encoding="utf-8")
    parser_a = source_dir / "parser_a.py"
    parser_b = source_dir / "parser_b.py"
    parser_a.write_text(
        "from polylogue.logging import get_logger\n\ndef parse(payload):\n    return payload\n", encoding="utf-8"
    )
    parser_b.write_text(
        "from polylogue.logging import get_logger\n\ndef parse(payload):\n    return {'b': payload}\n", encoding="utf-8"
    )

    import polylogue.sources.origin_specs as origin_specs

    monkeypatch.setattr(origin_specs, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(
        origin_specs,
        "_LOWERING_FINGERPRINT_PATHS",
        ("polylogue/sources/parser_a.py",),
    )
    monkeypatch.setattr(
        origin_specs,
        "_MATERIALIZER_FINGERPRINT_PATHS",
        ("polylogue/sources/parser_a.py",),
    )
    first = replace(
        next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.CODEX_SESSION),
        parser_paths=(str(parser_a),),
        stream_parser_path=None,
        assembly_paths=(),
        assembly_spec_path=None,
        artifact_rules=(),
    )
    second = replace(
        next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.CHATGPT_EXPORT),
        parser_paths=(str(parser_b),),
        stream_parser_path=None,
        assembly_paths=(),
        assembly_spec_path=None,
        artifact_rules=(),
    )
    origin_specs._fingerprint_sources_cached.cache_clear()
    parser_before = (first.parser_fingerprint(), second.parser_fingerprint())
    lowering_before = origin_specs.lowering_fingerprint()
    materializer_before = origin_specs.materializer_fingerprint()

    logging_source.write_text("def get_logger():\n    return 'after'\n", encoding="utf-8")
    origin_specs._fingerprint_sources_cached.cache_clear()
    assert (first.parser_fingerprint(), second.parser_fingerprint()) == parser_before
    assert origin_specs.lowering_fingerprint() != lowering_before
    assert origin_specs.materializer_fingerprint() != materializer_before

    parser_a.write_text(
        "from polylogue.logging import get_logger\n\ndef parse(payload):\n    return {'a': payload}\n", encoding="utf-8"
    )
    origin_specs._fingerprint_sources_cached.cache_clear()
    assert first.parser_fingerprint() != parser_before[0]


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
    every stream-capable executable OriginSpec (Claude Code, Codex, Hermes)
    report a ``stream_parser_parity_mismatch`` diagnostic.
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

    Anti-vacuity: blanking a display_description (or dropping a public spec)
    shrinks the derived completion inventory below the accepted filter vocabulary.
    """
    from polylogue.sources.origin_specs import public_origin_descriptions

    for spec in ORIGIN_SPECS:
        assert spec.display_description.strip(), spec.origin.value

    descriptions = public_origin_descriptions()
    assert set(descriptions) == set(public_origin_tokens())
    assert all(text.strip() for text in descriptions.values())


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


def test_source_fingerprint_memoizes_on_disk_by_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: a second process-level computation reads the memo instead of parsing.

    Dropping the memo makes the second call recompute (the patched compute
    raises); editing a source changes its signature and forces a recompute.
    """
    import polylogue.sources.origin_specs as origin_specs_module

    source_root = tmp_path / "source-root"
    source_dir = source_root / "polylogue" / "sources"
    source_dir.mkdir(parents=True)
    emitter = source_dir / "emitter.py"
    emitter.write_text("def emit(payload):\n    return payload\n", encoding="utf-8")
    monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(origin_specs_module, "_LOWERING_FINGERPRINT_PATHS", ("polylogue/sources/emitter.py",))
    origin_specs_module._fingerprint_sources_cached.cache_clear()

    first = origin_specs_module.lowering_fingerprint()
    memos = list((source_root / ".cache" / "source-fingerprints").glob("*.txt"))
    assert len(memos) == 1 and len(memos[0].read_text(encoding="utf-8")) == 64

    origin_specs_module._fingerprint_sources_cached.cache_clear()
    monkeypatch.setattr(
        origin_specs_module,
        "_fingerprint_sources_compute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("recomputed despite memo")),
    )
    assert origin_specs_module.lowering_fingerprint() == first

    monkeypatch.undo()
    monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(origin_specs_module, "_LOWERING_FINGERPRINT_PATHS", ("polylogue/sources/emitter.py",))
    emitter.write_text("def emit(payload):\n    return {'session': payload}\n", encoding="utf-8")
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    assert origin_specs_module.lowering_fingerprint() != first

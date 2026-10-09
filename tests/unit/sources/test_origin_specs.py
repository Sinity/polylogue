"""OriginSpec admission-kernel laws (polylogue-2qx.1.1)."""

from __future__ import annotations

import ast
import gzip
import io
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from ijson.backends import python as exact_backend

from polylogue.archive.revision_authority import raw_authority_parser_fingerprint
from polylogue.core.enums import Origin, Provider
from polylogue.sources.assembly import get_assembly_spec
from polylogue.sources.detection import (
    DetectionMode,
    DetectorBinding,
    DetectorBindingError,
    compile_detector_registry,
)
from polylogue.sources.dispatch import STREAM_RECORD_PROVIDERS
from polylogue.sources.origin_specs import (
    DROPPED_VALUE_VOCABULARIES,
    ORIGIN_SPEC_REGISTRY,
    ORIGIN_SPECS,
    DroppedValueVocabulary,
    OriginSpecRegistry,
    TopologyCapabilities,
    TopologyCapability,
    check_dropped_value_vocabularies,
    database_capability_for_provider,
    detector_registry,
    lowering_fingerprint,
    parser_fingerprint_for_origin,
    parser_semantic_authority_fingerprint,
    public_origin_descriptions,
    public_origin_meanings,
    public_origin_tokens,
    recognize_source_class,
    schema_observed_leaf_values,
    topology_capability_census,
    undeclared_schema_values,
    unobserved_value_vocabularies,
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


def test_hermes_source_class_recognition_reads_jsonl_txt_as_records(tmp_path: Path) -> None:
    """A ``.jsonl.txt`` stream is JSONL by the shared suffix rule, not one JSON document.

    Anti-vacuity: probing by ``path.suffix`` sees ``.txt``, reads the two
    records as a single JSON document, and refuses the file.
    """

    path = tmp_path / "moved-events.jsonl.txt"
    path.write_text(
        '{"atof_version":"0.1","kind":"mark","uuid":"u-1",'
        '"timestamp":"2026-08-26T00:00:00Z","name":"hermes.turn.start"}\n'
        '{"atof_version":"0.1","kind":"mark","uuid":"u-2",'
        '"timestamp":"2026-08-26T00:00:01Z","name":"hermes.turn.end"}\n',
        encoding="utf-8",
    )
    result = recognize_source_class(Provider.HERMES, path)
    assert result is not None
    assert result.source_class == "session"


def test_jsonl_recognition_streams_giant_fields_and_keeps_foreign_mapping_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3
    from collections.abc import Generator
    from contextlib import contextmanager

    from polylogue.schemas import observation_spill
    from polylogue.schemas.observation_spill import SpilledKey, _ScalarTokenStore
    from polylogue.storage.sqlite.connection_profile import scratch_connection_context

    record = {
        "atof_version": "0.1",
        "kind": "mark",
        "uuid": "u" * (4 * 1024 * 1024),
        "timestamp": "t",
        "name": "n",
        "k" * (4 * 1024 * 1024): [{"cell": index} for index in range(10000)],
    }
    path = tmp_path / "neutral.jsonl"
    valid = json.dumps(record).encode()
    path.write_bytes(b" " * 65536 + valid + b"\n")

    @contextmanager
    def small_cells(**kwargs: Any) -> Generator[sqlite3.Connection, None, None]:
        with scratch_connection_context(**kwargs) as connection:
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 32768)
            yield connection

    def no_key_read(_self: SpilledKey) -> str:
        raise AssertionError("original key demand")

    def no_scalar_read(*_args: object) -> object:
        raise AssertionError("selected type-only field demand")

    monkeypatch.setattr(observation_spill, "scratch_connection_context", small_cells)
    monkeypatch.setattr(SpilledKey, "read", no_key_read)
    monkeypatch.setattr(_ScalarTokenStore, "read", no_scalar_read)
    result = recognize_source_class(Provider.HERMES, path)
    assert result is not None and result.source_class == "session"
    path.write_bytes(valid + b'\n{"' + b"x" * 65536 + b'":1}\n')
    result = recognize_source_class(Provider.HERMES, path)
    assert result is not None and result.source_class == "unsupported"


def test_source_class_recognition_defers_zip_members_to_archive_extraction(tmp_path: Path) -> None:
    """A provider archive is classified after its members are extracted."""

    archive = tmp_path / "export.zip"
    archive.write_bytes(b"not inspected during source-class admission")

    assert recognize_source_class(Provider.CODEX, archive) is None


def test_hermes_root_census_accounts_for_every_candidate_without_parsing(tmp_path: Path) -> None:
    """A broad root has one declared disposition for every file its layout places.

    Anti-vacuity: dropping a candidate from the walk or admitting every JSON
    by suffix (the 40 shipped skill templates, the stray cache database)
    changes the denominator or the typed disposition counts.
    """

    templates = tmp_path / "hermes-agent" / "optional-skills" / "skill" / "templates"
    templates.mkdir(parents=True)
    for index in range(40):
        (templates / f"template-{index}.json").write_text('{"name":"optional skill"}', encoding="utf-8")
    relay = tmp_path / "observability" / "nemo-relay"
    (relay / "atif").mkdir(parents=True)
    (relay / "atof").mkdir(parents=True)
    (relay / "atif" / "trajectory-1.json").write_text(
        '{"schema_version":"ATIF-v1.7","session_id":"s-1","steps":[]}', encoding="utf-8"
    )
    (relay / "atof" / "events.jsonl").write_text(
        '{"atof_version":"0.1","kind":"mark","uuid":"u-1",'
        '"timestamp":"2026-08-26T00:00:00Z","name":"hermes.turn.start"}\n',
        encoding="utf-8",
    )
    (tmp_path / "cache.sqlite").write_bytes(b"not sqlite")
    (tmp_path / "state.db").write_bytes(b"not sqlite")

    census = census_source_root(tmp_path, provider=Provider.HERMES)

    assert census.candidate_count == 3
    assert census.disposition_counts == {"session": 2, "non_session": 0, "unsupported": 1}
    assert census.accounted_count == census.candidate_count
    assert census.unexplained_candidates == ()
    assert census.is_complete
    assert census.candidate_bytes > 0
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
        # polylogue-rovf5 / polylogue-ximhz: harness-authored memory
        # documents and the two retained assembly inputs.
        "agent_memory_document",
        "session_index",
        "prompt_history_log",
        # polylogue-k3ahm (#5225): the per-process NDJSON carrier a hook
        # producer appends one event per line to. Retained bytes only --
        # ``parse_policy="raw-only"`` -- so it must never be probed as a
        # session stream. It is listed here because this assertion is the
        # registry's completeness gate: a production rule the expected set
        # omits makes the gate red, which is how this omission surfaced.
        "hook_event_carrier",
    }
    assert {rule.kind for rule in chatgpt.artifact_rules} == {"export_asset_index", "export_asset"}
    assert {rule.kind for rule in by_origin[Origin.CODEX_SESSION].artifact_rules} == {
        "agent_memory_document",
        "session_index",
        "prompt_history_log",
        "hook_event_carrier",
    }
    tool_result_rule = next(rule for rule in claude.artifact_rules if rule.kind == "tool_result_sidecar")
    assert tool_result_rule.path_suffixes == (".json", ".txt", ".html", "")
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
    assert set(by_origin) == set(Origin)
    assert by_origin[Origin.UNKNOWN_EXPORT].lifecycle == "compatibility-only"
    assert by_origin[Origin.AISTUDIO_DRIVE].provider_wires == (Provider.GEMINI, Provider.DRIVE)
    assert ORIGIN_SPEC_REGISTRY.diagnostics() == ()


def test_database_origins_declare_snapshot_and_member_disposition() -> None:
    """Database acquisition policy is projected from OriginSpec, not filename sets."""

    codex = database_capability_for_provider(Provider.CODEX)
    hermes = database_capability_for_provider(Provider.HERMES)
    assert codex is not None
    assert hermes is not None
    for capability in (codex, hermes):
        assert capability.snapshot_method == "logical_export"
        assert "read transaction" in capability.consistency_fence
        assert "logical export" in capability.revision_identity
        assert capability.full_snapshot_per_revision
        assert capability.snapshot_lineage_policy
        assert capability.filenames

    assert codex.member("state_5.sqlite").disposition == "acquire"  # type: ignore[union-attr]
    assert codex.member("logs_2.sqlite").disposition == "out-of-scope"  # type: ignore[union-attr]
    assert hermes.member("state.db").disposition == "acquire"  # type: ignore[union-attr]


def test_database_member_lookup_is_origin_owned() -> None:
    """A declared member can be discovered without a second source filename registry."""

    capability = database_capability_for_provider(Provider.CODEX)
    assert capability is not None
    assert capability.member("new-database.sqlite") is None
    assert {member.filename for member in capability.members} == capability.filenames


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
    assert claude["parent_dispatch"]["state"] == "positive-derived"
    assert "parentToolUseID" in str(claude["parent_dispatch"]["evidence"])
    # polylogue-esvzb: a forked Claude Code session's own records name the
    # parent message it diverged at, so this dimension is carried, not absent.
    assert claude["inheritance_branch_point"]["state"] == "carried"
    assert "forkedFrom.messageUuid" in str(claude["inheritance_branch_point"]["evidence"])
    assert codex["parent_dispatch"]["state"] == "structurally-absent"
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
    origin_specs._invalidate_source_signatures()

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
    origin_specs_module._invalidate_source_signatures()

    before = (
        origin_specs_module.lowering_fingerprint(),
        origin_specs_module.replay_routing_fingerprint(),
        origin_specs_module.materializer_fingerprint(),
    )
    origin_source.write_text(
        origin_source.read_text(encoding="utf-8").replace("before", "after").replace("True", "False"), encoding="utf-8"
    )
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()
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
    origin_specs_module._invalidate_source_signatures()
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
    from polylogue.sources import origin_specs as origin_specs_module

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
    origin_specs_module._invalidate_source_signatures()
    after = synthetic.parser_fingerprint()

    assert before != after


def test_dispatch_closure_traverses_implicit_parser_namespace_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Namespace-imported parser modules contribute to shared lowering identity."""
    import polylogue.sources.origin_specs as origin_specs

    source_root = tmp_path / "source-root"
    sources = source_root / "polylogue" / "sources"
    parser_namespace = sources / "parsers"
    parser_namespace.mkdir(parents=True)
    dispatch = sources / "dispatch.py"
    dispatch.write_text(
        "from .parsers import alpha, beta\n\ndef route(value):\n    return alpha.parse(value), beta.parse(value)\n"
    )
    alpha = parser_namespace / "alpha.py"
    beta = parser_namespace / "beta.py"
    alpha.write_text("def parse(value):\n    return value\n", encoding="utf-8")
    beta.write_text("def parse(value):\n    return {'beta': value}\n", encoding="utf-8")

    monkeypatch.setattr(origin_specs, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(origin_specs, "_LOWERING_FINGERPRINT_PATHS", ("polylogue/sources/dispatch.py",))
    origin_specs._fingerprint_sources_cached.cache_clear()
    origin_specs._invalidate_source_signatures()

    closure = set(origin_specs._semantic_source_paths(("polylogue/sources/dispatch.py",)))
    assert alpha.resolve() in closure
    assert beta.resolve() in closure
    before = origin_specs.lowering_fingerprint()

    beta.write_text("def parse(value):\n    return {'beta': value, 'changed': True}\n", encoding="utf-8")
    origin_specs._fingerprint_sources_cached.cache_clear()
    origin_specs._invalidate_source_signatures()
    assert origin_specs.lowering_fingerprint() != before


def test_composed_raw_authority_fingerprint_tracks_origin_spec_parser_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persisted authority stamp includes the executable OriginSpec parser closure."""
    import polylogue.sources.origin_specs as origin_specs

    parser = tmp_path / "parser.py"
    helper = tmp_path / "helper.py"
    parser.write_text(
        "from .helper import normalize\n\ndef parse(value):\n    return normalize(value)\n", encoding="utf-8"
    )
    helper.write_text("def normalize(value):\n    return value.strip()\n", encoding="utf-8")
    target = next(spec for spec in ORIGIN_SPECS if spec.origin is Origin.CODEX_SESSION)
    monkeypatch.setattr(
        origin_specs,
        "ORIGIN_SPECS",
        tuple(replace(spec, parser_paths=(str(parser),)) if spec is target else spec for spec in ORIGIN_SPECS),
    )
    origin_specs.parser_semantic_authority_fingerprint.cache_clear()
    origin_specs._fingerprint_sources_cached.cache_clear()
    origin_specs._invalidate_source_signatures()
    before = parser_semantic_authority_fingerprint()

    helper.write_text("def normalize(value):\n    return value.casefold()\n", encoding="utf-8")
    origin_specs.parser_semantic_authority_fingerprint.cache_clear()
    origin_specs._fingerprint_sources_cached.cache_clear()
    origin_specs._invalidate_source_signatures()
    assert parser_semantic_authority_fingerprint() != before


def test_composed_raw_authority_fingerprint_tracks_declared_database_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed database member contract changes which retained source bytes mean parser input."""
    import polylogue.sources.origin_specs as origin_specs

    target = next(spec for spec in ORIGIN_SPECS if spec.lifecycle == "executable" and spec.database_capability)
    assert target.database_capability is not None
    changed_capability = replace(
        target.database_capability,
        revision_identity=f"{target.database_capability.revision_identity}:changed",
    )
    monkeypatch.setattr(
        origin_specs,
        "ORIGIN_SPECS",
        tuple(
            replace(spec, database_capability=changed_capability) if spec is target else spec for spec in ORIGIN_SPECS
        ),
    )
    origin_specs.parser_semantic_authority_fingerprint.cache_clear()
    origin_specs._fingerprint_sources_cached.cache_clear()
    origin_specs._invalidate_source_signatures()
    changed = parser_semantic_authority_fingerprint()

    monkeypatch.undo()
    origin_specs.parser_semantic_authority_fingerprint.cache_clear()
    origin_specs._fingerprint_sources_cached.cache_clear()
    origin_specs._invalidate_source_signatures()
    current = parser_semantic_authority_fingerprint()
    assert changed != current


def test_database_consumer_implementation_is_in_origin_parser_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Executable database consumers are parser routes even when their module is not imported by dispatch."""
    import polylogue.sources.origin_specs as origin_specs

    source_root = tmp_path / "source-root"
    package = source_root / "polylogue" / "sources"
    package.mkdir(parents=True)
    reader = package / "database_reader.py"
    reader.write_text("def read(connection):\n    return connection.execute('select 1')\n", encoding="utf-8")
    capability_origin = next(spec for spec in ORIGIN_SPECS if spec.database_capability is not None)
    assert capability_origin.database_capability is not None
    # The synthetic closure contains only the synthetic consumer: other members'
    # consumers and the origin's own parser modules are absent from this root.
    capability = replace(
        capability_origin.database_capability,
        members=(
            replace(
                capability_origin.database_capability.members[0], consumer="polylogue/sources/database_reader.py:read"
            ),
            *(replace(member, consumer=None) for member in capability_origin.database_capability.members[1:]),
        ),
    )
    synthetic = replace(
        capability_origin,
        database_capability=capability,
        parser_paths=(),
        stream_parser_path=None,
        assembly_paths=(),
        assembly_spec_path=None,
        artifact_rules=(),
    )
    monkeypatch.setattr(origin_specs, "_SOURCE_ROOT", source_root)
    origin_specs._invalidate_source_signatures()

    before = synthetic.parser_fingerprint()
    reader.write_text("def read(connection):\n    return connection.execute('select 2')\n", encoding="utf-8")
    origin_specs._invalidate_source_signatures()
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
        # Database consumers are parser routes; this synthetic closure has none.
        database_capability=None,
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
    origin_specs._invalidate_source_signatures()
    parser_before = (first.parser_fingerprint(), second.parser_fingerprint())
    lowering_before = origin_specs.lowering_fingerprint()
    materializer_before = origin_specs.materializer_fingerprint()

    logging_source.write_text("def get_logger():\n    return 'after'\n", encoding="utf-8")
    origin_specs._fingerprint_sources_cached.cache_clear()
    origin_specs._invalidate_source_signatures()
    assert (first.parser_fingerprint(), second.parser_fingerprint()) == parser_before
    assert origin_specs.lowering_fingerprint() != lowering_before
    assert origin_specs.materializer_fingerprint() != materializer_before

    parser_a.write_text(
        "from polylogue.logging import get_logger\n\ndef parse(payload):\n    return {'a': payload}\n", encoding="utf-8"
    )
    origin_specs._fingerprint_sources_cached.cache_clear()
    origin_specs._invalidate_source_signatures()
    assert first.parser_fingerprint() != parser_before[0]


def test_parser_fingerprint_changes_when_a_declared_assembly_helper_changes(tmp_path: Path) -> None:
    """Assembly enrichment is part of the origin's normalized output contract."""
    from polylogue.sources import origin_specs as origin_specs_module

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
    origin_specs_module._invalidate_source_signatures()
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
    origin_specs._invalidate_source_signatures()

    before = origin_specs.lowering_fingerprint()
    emitter.write_text("def emit(payload):\n    return {'session': payload}\n", encoding="utf-8")
    origin_specs._invalidate_source_signatures()
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
    assert raw_authority_parser_fingerprint() == parser_semantic_authority_fingerprint()
    assert raw_authority_parser_fingerprint() != "revision-membership-v5"


def test_origin_specs_compile_the_production_detector_registry() -> None:
    registry = detector_registry()

    assert registry.by_mode
    assert all(spec.detector_bindings for spec in ORIGIN_SPECS if spec.lifecycle == "executable")


def test_singleton_sequence_detection_projects_and_tests_once() -> None:
    registry = detector_registry()
    compiled = next(
        item
        for item in registry.by_mode[DetectionMode.SEQUENCE_DOCUMENT]
        if item.binding.binding_id == "chatgpt-sequence-document"
    )
    calls: list[object] = []

    def predicate(payload: object) -> bool:
        calls.append(payload)
        return True

    single_binding = replace(compiled, predicate=predicate)
    one_binding_registry = replace(
        registry,
        by_mode={
            DetectionMode.SEQUENCE_DOCUMENT: (single_binding,),
            DetectionMode.SEQUENCE_RECORD_STREAM: (),
            DetectionMode.RECORD: (),
        },
    )

    result = list(one_binding_registry.iter_record_detections({"messages": []}))[1]

    assert result == (Provider.CHATGPT, compiled.binding.evidence_label)
    assert calls == [[{"messages": []}]]


def test_record_detection_views_share_only_identical_declared_projections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two independent classification orders reuse each declaration once per record."""
    from polylogue.sources import detection_projection

    registry = detector_registry()
    original = detection_projection.project_detection_root
    calls: list[object] = []

    def observe(value: object, rule: detection_projection.DetectorProjection) -> object:
        calls.append(rule)
        return original(value, rule)

    monkeypatch.setattr(detection_projection, "project_detection_root", observe)
    value = {"unrelated": {"nested": [[{"opaque": "synthetic"}]]}}
    paths = {
        compiled.binding.stream_projection_path for candidates in registry.by_mode.values() for compiled in candidates
    }

    assert list(registry.iter_record_detections(value)) == [(None, None), (None, None)]
    assert len(calls) == len(paths)
    # A different record must produce new views even when it has the same shape.
    assert list(registry.iter_record_detections(dict(value))) == [(None, None), (None, None)]
    assert len(calls) == 2 * len(paths)


def test_record_stream_shares_only_identical_projections_within_each_record(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.sources import detection_projection

    registry = detector_registry()
    original = detection_projection.project_detection_value
    calls: list[object] = []

    def observe(value: object, rule: detection_projection.DetectorProjection) -> object:
        calls.append(rule)
        return original(value, rule)

    monkeypatch.setattr(detection_projection, "project_detection_value", observe)
    candidates = (
        *registry.by_mode[DetectionMode.SEQUENCE_DOCUMENT],
        *registry.by_mode[DetectionMode.SEQUENCE_RECORD_STREAM],
    )
    paths = {compiled.binding.stream_projection_path for compiled in candidates}
    assert len(paths) < len(candidates)
    value = {"unrelated": {"nested": [[{"opaque": "synthetic"}]]}}
    assert registry.detect_record_stream([value, dict(value)]) == (None, None)
    assert len(calls) == 2 * len(paths)


def test_record_stream_shares_json_read_conversion_within_each_record(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.sources import detection_projection
    from polylogue.sources.dispatch import _payload_record

    registry = detector_registry()
    compiled = next(
        item
        for item in registry.by_mode[DetectionMode.SEQUENCE_DOCUMENT]
        if item.binding.binding_id == "claude-ai-sequence-chat-messages"
    )
    from polylogue.core.json import json_document_or_none

    original = json_document_or_none
    calls: list[object] = []

    def observe(value: object) -> object:
        calls.append(value)
        return original(value)

    def predicate(payload: object) -> bool:
        assert isinstance(payload, list)
        assert _payload_record(payload[0]) is not None
        return False

    monkeypatch.setattr(detection_projection, "json_document_or_none", observe)
    candidates = (replace(compiled, predicate=predicate), replace(compiled, predicate=predicate))
    selected = replace(registry, by_mode={DetectionMode.SEQUENCE_DOCUMENT: candidates})
    assert selected.detect_record_stream([{}, {}]) == (None, None)
    assert len(calls) == 2


def test_record_stream_keeps_order_last_witness_and_complete_consumption() -> None:
    registry = detector_registry()
    compiled = next(
        item
        for item in registry.by_mode[DetectionMode.SEQUENCE_DOCUMENT]
        if item.binding.binding_id == "browser-capture-sequence"
    )
    observed: list[object] = []
    resolved: list[object] = []

    def first(payload: object) -> bool:
        assert isinstance(payload, list)
        record = payload[0]
        assert isinstance(record, dict)
        return record.get("session") == {"provider": "chatgpt"}

    def resolver(payload: object) -> Provider:
        resolved.append(payload)
        return Provider.CHATGPT

    tighter = replace(compiled, predicate=first, provider_resolver=resolver)
    looser = replace(compiled, predicate=lambda _payload: True)
    selected = replace(registry, by_mode={DetectionMode.SEQUENCE_DOCUMENT: (tighter, looser)})
    values = [
        {"session": {"provider": "codex"}},
        {"session": {"provider": "chatgpt"}},
        {"session": {"provider": "chatgpt"}, "schema_version": "last"},
    ]

    def records() -> Iterator[object]:
        for value in values:
            observed.append(value)
            yield value

    assert selected.detect_record_stream(records()) == (Provider.CHATGPT, compiled.binding.evidence_label)
    assert observed == values
    assert resolved == [[{"session": {"provider": "chatgpt"}, "schema_version": "last"}]]

    def broken() -> Iterator[object]:
        yield values[1]
        raise ValueError("synthetic late syntax refusal")

    with pytest.raises(ValueError, match="synthetic late syntax refusal"):
        selected.detect_record_stream(broken())

    invalid = replace(
        tighter,
        binding=replace(tighter.binding, dynamic_provider_allowlist=(Provider.CHATGPT,)),
        provider_resolver=lambda _payload: Provider.CODEX,
    )
    with pytest.raises(DetectorBindingError, match="invalid projected dynamic provider"):
        replace(registry, by_mode={DetectionMode.SEQUENCE_DOCUMENT: (invalid,)}).detect_record_stream(values)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"type": "user", "sessionId": "synthetic", "message": {"content": ["complete"]}},
        {"type": "session_meta", "payload": {"id": "synthetic", "timestamp": "2026-01-01T00:00:00Z"}},
        [{"type": "summary"}, {"type": "session_meta", "payload": {"id": "synthetic"}}],
        {"messages": [{"role": "user", "content": "synthetic"}], "metadata": {"deep": [[{}]]}},
        {"mapping": {"node": {"id": "node", "message": None, "parent": None, "children": []}}},
        {"schema_version": "ATIF-v1.7", "session_id": "synthetic", "steps": []},
        {"source": "antigravity_language_server", "cascadeId": "synthetic", "markdown": "# Synthetic"},
    ],
)
def test_record_detection_views_match_complete_event_route(value: object) -> None:
    registry = detector_registry()

    def events_factory() -> Iterator[tuple[str, object]]:
        return iter(exact_backend.basic_parse(io.BytesIO(json.dumps(value).encode("utf-8"))))

    expected = [registry.detect_record_events(events_factory, sequence=sequence) for sequence in (False, True)]
    assert list(registry.iter_record_detections(value)) == expected


def test_singleton_sequence_detection_matches_event_route_for_nested_and_dynamic_shapes() -> None:
    registry = detector_registry()
    capture = json.loads(
        (Path(__file__).parents[2] / "fixtures/chatgpt/native-browser-capture-v1.json").read_text(encoding="utf-8")
    )
    capture["provenance"]["operator_metadata"] = {"nested": [[{"value": "synthetic"}]]}
    values_and_expected = (
        (capture, (Provider.CHATGPT, "browser_capture.looks_like (any complete sequence envelope)")),
        ([capture], (None, None)),
        ({"unrelated": [[{"opaque": "synthetic"}]]}, (None, None)),
    )

    for value, expected in values_and_expected:

        def events_factory(value: object = value) -> Iterator[tuple[str, object]]:
            encoded = json.dumps(value, separators=(",", ":")).encode("utf-8")
            return iter(exact_backend.basic_parse(io.BytesIO(encoded)))

        decoded = list(registry.iter_record_detections(value))
        streamed = [registry.detect_record_events(events_factory, sequence=sequence) for sequence in (False, True)]
        assert decoded == streamed
        assert decoded[1] == expected


def test_record_detection_views_keep_dynamic_provider_allowlist() -> None:
    registry = detector_registry()
    compiled = next(
        item for item in registry.by_mode[DetectionMode.RECORD] if item.binding.binding_id == "browser-capture-record"
    )
    invalid = replace(
        compiled,
        binding=replace(compiled.binding, dynamic_provider_allowlist=(Provider.CHATGPT,)),
        predicate=lambda _payload: True,
        provider_resolver=lambda _payload: Provider.CODEX,
    )
    one_binding_registry = replace(registry, by_mode={DetectionMode.RECORD: (invalid,)})

    with pytest.raises(DetectorBindingError, match="invalid projected dynamic provider"):
        list(one_binding_registry.iter_record_detections({}))


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


def _write_status_package(root: Path, *, values: list[str] | None) -> Path:
    """Write one committed-shaped gemini-cli element with a status leaf.

    The vocabulary readers must be exercised against a real gzip package file
    rather than an in-memory document, but reading whichever package happens to
    be committed also makes the assertion depend on live corpus content. A
    synthetic package keeps the file format real and the expected values fixed.
    """

    leaf: dict[str, object] = {"type": "string"}
    if values is not None:
        leaf["x-polylogue-semantic-role"] = "message_role"
        leaf["x-polylogue-values"] = list(values)
    document = {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "toolCalls": {
                            "type": "array",
                            "items": {"type": "object", "properties": {"status": leaf}},
                        }
                    },
                },
            }
        },
    }
    elements = root / "gemini-cli" / "versions" / "v1" / "elements"
    elements.mkdir(parents=True, exist_ok=True)
    (elements / "session_document.schema.json.gz").write_bytes(gzip.compress(json.dumps(document).encode("utf-8")))
    return root


def test_dropped_value_vocabularies_have_no_drift_against_the_committed_schema(tmp_path: Path) -> None:
    """Production dependency: DROPPED_VALUE_VOCABULARIES stays honest against real evidence.

    Anti-vacuity: the fixture publishes ``success``, which the declared set
    covers, so the empty result is earned. Publishing a value the declaration
    omits (see the drift test below) turns it red.
    """
    assert check_dropped_value_vocabularies(schema_root=_write_status_package(tmp_path, values=["success"])) == {}


def test_schema_observed_leaf_values_walks_array_and_scalar_segments(tmp_path: Path) -> None:
    root = _write_status_package(tmp_path, values=["success"])
    assert schema_observed_leaf_values("gemini-cli", "messages[].toolCalls[].status", schema_root=root) == {"success"}
    # A path with no committed schema evidence resolves to empty, not an error.
    assert schema_observed_leaf_values("gemini-cli", "no.such.path", schema_root=root) == frozenset()
    assert schema_observed_leaf_values("no-such-provider", "status", schema_root=root) == frozenset()


def test_undeclared_schema_values_flags_a_value_the_declaration_does_not_cover(tmp_path: Path) -> None:
    narrow_vocab = DroppedValueVocabulary(
        field="test-only narrow gemini-cli status vocabulary",
        schema_provider="gemini-cli",
        schema_field_path="messages[].toolCalls[].status",
        declared_values=frozenset(),
        parser_path="test-only",
        reason="Anti-vacuity fixture: an empty declared set must show the observed value as drift.",
    )
    root = _write_status_package(tmp_path, values=["success"])
    assert undeclared_schema_values(narrow_vocab, schema_root=root) == {"success"}


def test_a_leaf_publishing_nothing_is_reported_as_unobserved_not_as_agreement(tmp_path: Path) -> None:
    """A vocabulary with no published evidence must not read as a clean drift check.

    A committed package publishes a member list only from a declared protocol
    slot, so the drift comparison can silently lose its evidence. Anti-vacuity:
    give the leaf a published vocabulary again and the field disappears from
    ``unobserved_value_vocabularies``.
    """
    empty_root = _write_status_package(tmp_path / "empty", values=None)
    assert check_dropped_value_vocabularies(schema_root=empty_root) == {}
    assert "gemini-cli tool-result status" in unobserved_value_vocabularies(schema_root=empty_root)

    published_root = _write_status_package(tmp_path / "published", values=["success"])
    assert unobserved_value_vocabularies(schema_root=published_root) == ()


def test_source_fingerprint_memoizes_on_disk_by_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: a second process-level computation reads the memo instead of parsing.

    Dropping the memo makes the second call recompute (the patched compute
    raises); editing a source changes its signature and forces a recompute.
    """
    import polylogue.sources.origin_specs as origin_specs_module

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "shared-cache"))
    source_root = tmp_path / "source-root"
    source_dir = source_root / "polylogue" / "sources"
    source_dir.mkdir(parents=True)
    emitter = source_dir / "emitter.py"
    emitter.write_text("def emit(payload):\n    return payload\n", encoding="utf-8")
    monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(origin_specs_module, "_LOWERING_FINGERPRINT_PATHS", ("polylogue/sources/emitter.py",))
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()

    first = origin_specs_module.lowering_fingerprint()
    memo_root = origin_specs_module._source_memo_root()
    assert memo_root is not None
    memos = list(memo_root.glob("fingerprint-*.txt"))
    assert len(memos) == 1 and len(memos[0].read_text(encoding="utf-8")) == 64

    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()
    monkeypatch.setattr(
        origin_specs_module,
        "_fingerprint_sources_compute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("recomputed despite memo")),
    )
    assert origin_specs_module.lowering_fingerprint() == first

    monkeypatch.undo()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "shared-cache"))
    monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(origin_specs_module, "_LOWERING_FINGERPRINT_PATHS", ("polylogue/sources/emitter.py",))
    emitter.write_text("def emit(payload):\n    return {'session': payload}\n", encoding="utf-8")
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()
    assert origin_specs_module.lowering_fingerprint() != first


def test_generated_build_provenance_is_not_a_semantic_fingerprint_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Package-only build metadata cannot invalidate parser/lowering code.

    Anti-vacuity: restoring the local import edge into the source closure
    makes the second fingerprint differ when only BUILD_COMMIT/BUILD_DIRTY
    changes.  The real helper remains in the closure below, proving that this
    is a narrow generated-file exclusion rather than closure-wide suppression.
    """
    import polylogue.sources.origin_specs as origin_specs_module

    source_dir = tmp_path / "polylogue" / "sources"
    source_dir.mkdir(parents=True)
    emitter = source_dir / "emitter.py"
    emitter.write_text(
        "from polylogue._build_info import BUILD_COMMIT\n"
        "from polylogue.sources.helper import shape\n\n"
        "def emit(payload):\n    return shape(payload)\n",
        encoding="utf-8",
    )
    helper = source_dir / "helper.py"
    helper.write_text("def shape(payload):\n    return payload\n", encoding="utf-8")
    build_info = tmp_path / "polylogue" / "_build_info.py"
    build_info.write_text('BUILD_COMMIT = "commit-a"\nBUILD_DIRTY = False\n', encoding="utf-8")

    monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(origin_specs_module, "_LOWERING_FINGERPRINT_PATHS", ("polylogue/sources/emitter.py",))
    origin_specs_module._semantic_source_closure.cache_clear()
    origin_specs_module._local_import_paths.cache_clear()
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()

    first = origin_specs_module.lowering_fingerprint()
    members = origin_specs_module._semantic_source_paths(("polylogue/sources/emitter.py",))
    assert build_info.resolve() not in members
    assert helper.resolve() in members

    build_info.write_text('BUILD_COMMIT = "commit-b"\nBUILD_DIRTY = True\n', encoding="utf-8")
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()
    assert origin_specs_module.lowering_fingerprint() == first

    helper.write_text("def shape(payload):\n    return {'session': payload}\n", encoding="utf-8")
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()
    assert origin_specs_module.lowering_fingerprint() != first


def test_index_ddl_formatting_is_normalized_in_the_production_source_hash_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DDL comments/formatting do not leak through the lowering source hash."""
    import polylogue.sources.origin_specs as origin_specs_module

    path = tmp_path / "polylogue" / "storage" / "sqlite" / "archive_tiers" / "index.py"
    path.parent.mkdir(parents=True)
    source = 'INDEX_DDL = """CREATE TABLE x ( a TEXT /* note */ )"""\n'
    path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", tmp_path)
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()

    first = origin_specs_module._fingerprint_sources(
        ("polylogue/storage/sqlite/archive_tiers/index.py",), namespace="index-ddl-format"
    )
    path.write_text('INDEX_DDL = """ CREATE  TABLE x(a TEXT) """\n', encoding="utf-8")
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()
    assert (
        origin_specs_module._fingerprint_sources(
            ("polylogue/storage/sqlite/archive_tiers/index.py",), namespace="index-ddl-format"
        )
        == first
    )

    path.write_text('INDEX_DDL = """CREATE TABLE x(a BLOB)"""\n', encoding="utf-8")
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()
    assert (
        origin_specs_module._fingerprint_sources(
            ("polylogue/storage/sqlite/archive_tiers/index.py",), namespace="index-ddl-format"
        )
        != first
    )


def test_imported_fts_ddl_formatting_is_normalized_but_semantics_move_the_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Imported FTS DDL follows the same production lowering closure rule."""
    import polylogue.sources.origin_specs as origin_specs_module

    index_path = tmp_path / "polylogue" / "storage" / "sqlite" / "archive_tiers" / "index.py"
    fts_path = tmp_path / "polylogue" / "storage" / "fts" / "sql.py"
    index_path.parent.mkdir(parents=True)
    fts_path.parent.mkdir(parents=True)
    index_path.write_text(
        "from polylogue.storage.fts.sql import BLOCKS_FTS_TRIGGER_DDL\n"
        "INDEX_DDL = 'CREATE TABLE x (id INTEGER);' + ';'.join(BLOCKS_FTS_TRIGGER_DDL)\n",
        encoding="utf-8",
    )
    fts_path.write_text(
        "BLOCKS_FTS_TRIGGER_DDL = [\n"
        '    """CREATE TRIGGER x /* maintenance note */ AFTER INSERT ON blocks\n'
        '    BEGIN SELECT 1; END"""\n'
        "]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(
        origin_specs_module,
        "_LOWERING_FINGERPRINT_PATHS",
        ("polylogue/storage/sqlite/archive_tiers/index.py",),
    )
    origin_specs_module._semantic_source_closure.cache_clear()
    origin_specs_module._local_import_paths.cache_clear()
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()

    first = origin_specs_module.lowering_fingerprint()
    fts_path.write_text(
        'BLOCKS_FTS_TRIGGER_DDL = [\n    """ CREATE  TRIGGER x AFTER INSERT ON blocks BEGIN SELECT 1; END """\n]\n',
        encoding="utf-8",
    )
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()
    assert origin_specs_module.lowering_fingerprint() == first

    fts_path.write_text(
        fts_path.read_text(encoding="utf-8").replace("AFTER INSERT", "AFTER UPDATE"),
        encoding="utf-8",
    )
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()
    assert origin_specs_module.lowering_fingerprint() != first


def test_runtime_index_ddl_formatting_is_normalized_but_semantics_move_the_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runtime indexes are DDL contributors despite their ``*_SQL`` name."""
    import polylogue.sources.origin_specs as origin_specs_module

    path = tmp_path / "polylogue" / "storage" / "sqlite" / "runtime_indexes.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        '_RUNTIME_INDEX_SQL = (\n    """CREATE INDEX idx_x /* maintenance note */ ON blocks (session_id)""",\n)\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(
        origin_specs_module,
        "_LOWERING_FINGERPRINT_PATHS",
        ("polylogue/storage/sqlite/runtime_indexes.py",),
    )
    origin_specs_module._semantic_source_closure.cache_clear()
    origin_specs_module._local_import_paths.cache_clear()
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()

    first = origin_specs_module.lowering_fingerprint()
    path.write_text(
        '_RUNTIME_INDEX_SQL = ("CREATE  INDEX idx_x ON blocks(session_id)",)\n',
        encoding="utf-8",
    )
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()
    assert origin_specs_module.lowering_fingerprint() == first

    path.write_text(
        path.read_text(encoding="utf-8").replace("session_id", "message_id"),
        encoding="utf-8",
    )
    origin_specs_module._fingerprint_sources_cached.cache_clear()
    origin_specs_module._invalidate_source_signatures()
    assert origin_specs_module.lowering_fingerprint() != first


class TestSemanticSourceClosureMemo:
    """Closure membership and signatures are memoized for the process lifetime.

    Production source files cannot change during a process. Source-mutating
    tests invalidate signatures explicitly after writing their fixtures.
    """

    def test_overlapping_large_closures_reuse_parsed_imports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second large closure must reuse its shared members' parsed imports."""
        import polylogue.sources.origin_specs as origin_specs_module

        source_dir = tmp_path / "polylogue"
        source_dir.mkdir()
        paths = tuple(f"polylogue/module_{index}.py" for index in range(600))
        for path in paths:
            (tmp_path / path).write_text("value = 1\n", encoding="utf-8")
        monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", tmp_path)
        origin_specs_module._semantic_source_closure.cache_clear()
        origin_specs_module._local_import_paths.cache_clear()

        first = origin_specs_module._semantic_source_paths(paths)
        assert len(first) == len(paths)
        monkeypatch.setattr(
            ast,
            "parse",
            lambda *_args, **_kwargs: pytest.fail("shared import graph was parsed again"),
        )
        assert origin_specs_module._semantic_source_paths(tuple(reversed(paths))) == first

    def test_membership_is_walked_once_per_argument_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anti-vacuity: drop the memo and the repeat calls re-stat every member."""
        import polylogue.sources.origin_specs as origin_specs_module

        paths = origin_specs_module._LOWERING_FINGERPRINT_PATHS
        origin_specs_module._semantic_source_closure.cache_clear()

        first = origin_specs_module._semantic_source_paths(paths)
        assert len(first) > 1

        real_signature = origin_specs_module._source_signature
        walked: list[Path] = []

        def counting_signature(path: Path) -> tuple[str, str, int]:
            walked.append(path)
            return real_signature(path)

        monkeypatch.setattr(origin_specs_module, "_source_signature", counting_signature)
        for _ in range(20):
            assert origin_specs_module._semantic_source_paths(paths) is first
        assert walked == []

    def test_fingerprint_reads_each_member_once_per_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anti-vacuity: clearing the signature memo restores content reads."""
        import polylogue.sources.origin_specs as origin_specs_module

        paths = origin_specs_module._LOWERING_FINGERPRINT_PATHS
        # Prime advisory AST memos so observed reads measure signature work.
        origin_specs_module._fingerprint_sources(paths, namespace="closure-memo-law")
        origin_specs_module._fingerprint_sources_cached.cache_clear()
        origin_specs_module._invalidate_source_signatures()

        real_read = Path.read_bytes
        walked: list[Path] = []

        def counting_read(path: Path) -> bytes:
            walked.append(path)
            return real_read(path)

        monkeypatch.setattr(Path, "read_bytes", counting_read)
        first = origin_specs_module._fingerprint_sources(paths, namespace="closure-memo-law")
        cold_read_count = len(walked)
        members = origin_specs_module._semantic_source_paths(paths)
        assert cold_read_count == len(members)
        for _ in range(20):
            assert origin_specs_module._fingerprint_sources(paths, namespace="closure-memo-law") == first
        assert len(walked) == cold_read_count

        origin_specs_module._invalidate_source_signatures()
        origin_specs_module._fingerprint_sources(paths, namespace="closure-memo-law")
        assert len(walked) == cold_read_count * 2

    def test_edited_member_changes_the_fingerprint_under_a_warm_memo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transitively imported source is re-read even though membership is memoized.

        Anti-vacuity: move the memo up to ``_fingerprint_sources`` -- or key it
        on anything but the members' content signatures -- and the edited
        helper no longer moves the fingerprint.
        """
        import polylogue.sources.origin_specs as origin_specs_module

        source_root = tmp_path / "source-root"
        source_dir = source_root / "polylogue" / "sources"
        source_dir.mkdir(parents=True)
        (source_dir / "emitter.py").write_text(
            "from polylogue.sources.helper import shape\n\n\ndef emit(payload):\n    return shape(payload)\n",
            encoding="utf-8",
        )
        helper = source_dir / "helper.py"
        helper.write_text("def shape(payload):\n    return payload\n", encoding="utf-8")
        monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", source_root)
        monkeypatch.setattr(origin_specs_module, "_LOWERING_FINGERPRINT_PATHS", ("polylogue/sources/emitter.py",))
        origin_specs_module._semantic_source_closure.cache_clear()
        origin_specs_module._fingerprint_sources_cached.cache_clear()
        origin_specs_module._invalidate_source_signatures()

        first = origin_specs_module.lowering_fingerprint()
        members = origin_specs_module._semantic_source_paths(("polylogue/sources/emitter.py",))
        assert helper.resolve() in members

        # Edit a member's body without touching any import: membership is
        # unchanged and the memo stays warm, so only the re-read can move this.
        helper.write_text("def shape(payload):\n    return {'session': payload}\n", encoding="utf-8")
        origin_specs_module._invalidate_source_signatures()
        assert origin_specs_module._semantic_source_paths(("polylogue/sources/emitter.py",)) == members
        assert origin_specs_module.lowering_fingerprint() != first

    def test_invalidation_rebuilds_changed_import_membership(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit edit signal refreshes both signatures and import edges."""
        import polylogue.sources.origin_specs as origin_specs_module

        source_dir = tmp_path / "polylogue" / "sources"
        source_dir.mkdir(parents=True)
        emitter = source_dir / "emitter.py"
        emitter.write_text(
            "from polylogue.sources.first import shape\n\ndef emit(value):\n    return shape(value)\n",
            encoding="utf-8",
        )
        first = source_dir / "first.py"
        first.write_text("def shape(value):\n    return value\n", encoding="utf-8")
        second = source_dir / "second.py"
        second.write_text("def shape(value):\n    return {'value': value}\n", encoding="utf-8")

        monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", tmp_path)
        paths = ("polylogue/sources/emitter.py",)
        origin_specs_module._invalidate_source_signatures()
        before = origin_specs_module._semantic_source_paths(paths)
        assert first.resolve() in before
        assert second.resolve() not in before

        emitter.write_text(
            "from polylogue.sources.second import shape\n\ndef emit(value):\n    return shape(value)\n",
            encoding="utf-8",
        )
        origin_specs_module._invalidate_source_signatures()
        after = origin_specs_module._semantic_source_paths(paths)
        assert second.resolve() in after
        assert first.resolve() not in after

    def test_a_substituted_source_root_does_not_reuse_another_root_membership(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Declared paths are relative, so the root is part of the memo key."""
        import polylogue.sources.origin_specs as origin_specs_module

        roots = []
        for name in ("alpha", "beta"):
            source_dir = tmp_path / name / "polylogue" / "sources"
            source_dir.mkdir(parents=True)
            (source_dir / "emitter.py").write_text("def emit(payload):\n    return payload\n", encoding="utf-8")
            roots.append(tmp_path / name)

        monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", roots[0])
        alpha = origin_specs_module._semantic_source_paths(("polylogue/sources/emitter.py",))
        monkeypatch.setattr(origin_specs_module, "_SOURCE_ROOT", roots[1])
        beta = origin_specs_module._semantic_source_paths(("polylogue/sources/emitter.py",))

        assert alpha == ((roots[0] / "polylogue" / "sources" / "emitter.py").resolve(),)
        assert beta == ((roots[1] / "polylogue" / "sources" / "emitter.py").resolve(),)


# -----------------------------------------------------------------------------
# BOUNDED ADMISSION PROBES (bd polylogue-dhkuu, finding B)
# -----------------------------------------------------------------------------


def test_large_antigravity_export_is_recognized_by_streaming_envelope(tmp_path: Path) -> None:
    """A large Antigravity export is a session whatever its size.

    Anti-vacuity: reinstate a size ceiling on the whole-document probe (the
    removed 64 MiB refusal) and this candidate is classified ``unsupported``.
    """
    from polylogue.core.enums import Provider
    from polylogue.core.json_envelope import ENVELOPE_TEXT_PREFIX_CHARS
    from polylogue.sources.origin_specs import recognize_source_class

    candidate = tmp_path / "export.json"
    markdown = "## User\n" + "x" * (ENVELOPE_TEXT_PREFIX_CHARS * 4)
    candidate.write_text(
        json.dumps({"source": "antigravity_language_server", "cascadeId": "c-1", "markdown": markdown}),
        encoding="utf-8",
    )

    recognition = recognize_source_class(Provider.ANTIGRAVITY, candidate)

    assert recognition is not None
    assert recognition.source_class == "session"


def test_hermes_jsonl_probe_reads_a_long_record_instead_of_skipping_it(tmp_path: Path) -> None:
    """A long leading JSONL record is classified, not skipped.

    Anti-vacuity: reinstate the per-record byte skip and the long non-ATOF
    record is dropped from the sample, so the second file is wrongly admitted.
    Recognition applies the artifact route's all-record predicate.
    """
    from polylogue.core.enums import Provider
    from polylogue.sources.origin_specs import recognize_source_class

    candidate = tmp_path / "events.jsonl"
    record = {
        "atof_version": "0.1",
        "kind": "mark",
        "uuid": "u-1",
        "timestamp": "2026-01-01T00:00:00Z",
        "name": "event",
        "data": {"pad": "x" * (4 * 1024 * 1024)},
    }
    short = {key: value for key, value in record.items() if key != "data"} | {"uuid": "u-2"}
    candidate.write_text(json.dumps(record) + "\n" + json.dumps(short) + "\n", encoding="utf-8")

    recognition = recognize_source_class(Provider.HERMES, candidate)

    assert recognition is not None
    assert recognition.source_class == "session"

    # The long record is read, not skipped: when it is not ATOF, the file is
    # refused even though the short record is.
    record.pop("atof_version")
    candidate.write_text(json.dumps(record) + "\n" + json.dumps(short) + "\n", encoding="utf-8")
    refused = recognize_source_class(Provider.HERMES, candidate)
    assert refused is not None
    assert refused.source_class == "unsupported"


def test_hermes_jsonl_first_line_byte_order_mark_is_stripped_as_the_decoder_strips_it(tmp_path: Path) -> None:
    """The JSONL decoder strips a BOM from the first line and keeps that record.

    Anti-vacuity: leave the BOM in place and the tokenizer skips the first
    line, so only the ATOF record is sampled and the mixed file is admitted.
    """
    from polylogue.sources.origin_specs import recognize_source_class

    atof = b'{"atof_version": "0.1", "kind": "mark", "uuid": "u", "timestamp": "t", "name": "n"}'
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_bytes(b"\xef\xbb\xbf" + b'{"other": 1}\n' + atof + b"\n")
    pure = tmp_path / "pure.jsonl"
    pure.write_bytes(b"\xef\xbb\xbf" + atof + b"\n" + atof + b"\n")

    mixed_recognition = recognize_source_class(Provider.HERMES, mixed)
    pure_recognition = recognize_source_class(Provider.HERMES, pure)
    assert mixed_recognition is not None and mixed_recognition.source_class == "unsupported"
    assert pure_recognition is not None and pure_recognition.source_class == "session"


def test_array_document_recognition_samples_the_head_and_validates_the_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON array's signature is read from its leading records; the rest is
    streamed only to prove it parses, as the record parser reads the whole array.

    Anti-vacuity: stop at the sample and a document with a malformed tail is
    admitted although the record parser refuses it; read the root unexpanded
    and the whole array is materialized as one envelope.
    """
    from polylogue.sources import origin_specs

    record = '{"atof_version": "0.1", "kind": "mark", "uuid": "u", "timestamp": "t", "name": "n"}'
    document = tmp_path / "spans.json"
    document.write_text("[" + ",".join([record] * 200) + "]", encoding="utf-8")
    drawn: list[int] = []

    real = origin_specs._signature_envelope

    def guarded(value: object, fields: frozenset[str]) -> object:
        assert isinstance(value, dict), "array root read unexpanded"
        drawn.append(len(drawn))
        return real(value, fields)

    monkeypatch.setattr(origin_specs, "_signature_envelope", guarded)
    recognition = origin_specs.recognize_source_class(Provider.HERMES, document)
    assert recognition is not None and recognition.source_class == "session"
    assert len(drawn) == 200

    truncated = tmp_path / "truncated.json"
    truncated.write_text("[" + ",".join([record] * 40) + ",", encoding="utf-8")
    refused = origin_specs.recognize_source_class(Provider.HERMES, truncated)
    assert refused is not None and refused.source_class == "unsupported"


def test_whole_document_source_recognition_keeps_valid_large_unknown_integer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.schemas.observation_spill import _ScalarTokenStore

    path = tmp_path / "neutral.json"
    path.write_bytes(_ATOF_RECORD[:-1] + b',"unknown":' + b"9" * 65537 + b"}")

    def unselected(*_args: object) -> object:
        raise AssertionError("unknown scalar materialization")

    monkeypatch.setattr(_ScalarTokenStore, "read", unselected)
    result = recognize_source_class(Provider.HERMES, path)
    assert result is not None and result.source_class == "session"


def test_hermes_jsonl_recognition_requires_every_record_to_be_atof(tmp_path: Path) -> None:
    """Anti-vacuity: go back to ``any`` and the mixed file is recognized as a session."""
    from polylogue.sources.origin_specs import recognize_source_class

    atof = '{"atof_version": "0.1", "kind": "mark", "uuid": "u", "timestamp": "t", "name": "n"}'
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(atof + "\n" + '{"other": 1}\n', encoding="utf-8")
    pure = tmp_path / "pure.jsonl"
    pure.write_text(atof + "\n" + atof + "\n", encoding="utf-8")

    mixed_recognition = recognize_source_class(Provider.HERMES, mixed)
    pure_recognition = recognize_source_class(Provider.HERMES, pure)
    assert mixed_recognition is not None and mixed_recognition.source_class == "unsupported"
    assert pure_recognition is not None and pure_recognition.source_class == "session"


def test_jsonl_integer_observation_has_no_interpreter_digit_cap() -> None:
    """Valid JSONL integers retain their exact value beyond Python's string cap."""
    import io
    from decimal import Decimal

    from polylogue.sources.origin_specs import _jsonl_signature_envelopes

    fields = frozenset({"n", "atof_version"})
    lines = b'{"n": ' + b"9" * 4301 + b'}\n{"atof_version": "0.1"}'
    assert list(_jsonl_signature_envelopes(io.BytesIO(lines), fields=fields)) == [
        {"n": int(Decimal("9" * 4301))},
        {"atof_version": "0.1"},
    ]


def _reset_closure_caches(module: object) -> None:
    """Drop every in-process closure cache, leaving only what the disk memo holds."""
    module._semantic_source_closure.cache_clear()  # type: ignore[attr-defined]
    module._local_import_paths.cache_clear()  # type: ignore[attr-defined]
    module._invalidate_source_signatures()  # type: ignore[attr-defined]


def test_the_import_closure_memo_outlives_the_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Parsing the closure is paid once per file version, not once per process.

    Re-deriving the import graph cost about 9.6 s of a 14 s single-file pytest
    collection and was paid again by every xdist worker and every CLI start.

    Anti-vacuity: delete the memo lookup in ``_local_import_paths`` and this
    goes red -- the second walk re-parses, which the sabotaged ``_import_bases``
    turns into a failure.
    """
    from polylogue.sources import origin_specs as module

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text("from .b import B\n", encoding="utf-8")
    (package / "b.py").write_text("B = 1\n", encoding="utf-8")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "shared-cache"))
    monkeypatch.setattr(module, "_SOURCE_ROOT", tmp_path)
    _reset_closure_caches(module)
    first = module._semantic_source_paths(("pkg/a.py",))
    assert {path.name for path in first} == {"a.py", "b.py"}
    memo_root = module._source_memo_root()
    assert memo_root is not None
    assert list(memo_root.glob("edges-*.json"))

    _reset_closure_caches(module)

    def _refuse(signature: tuple[str, str, int]) -> tuple[str, ...]:
        raise AssertionError(f"re-parsed {signature[0]} despite an unchanged file")

    monkeypatch.setattr(module, "_import_bases", _refuse)
    assert module._semantic_source_paths(("pkg/a.py",)) == first


def test_a_memoized_closure_still_sees_a_module_that_appeared_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memo holds what a file's text says, never which imports resolved then.

    ``from .c import C`` against an absent ``c.py`` contributes nothing to the
    closure; the day ``c.py`` lands, the unedited importer's closure must grow,
    or a semantic file joins the fingerprint without moving it.

    Anti-vacuity: memoize resolved paths instead of lexical bases -- the shape
    this replaced -- and the second assertion still reports two members.
    """
    from polylogue.sources import origin_specs as module

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text("from .b import B\nfrom .c import C\n", encoding="utf-8")
    (package / "b.py").write_text("B = 1\n", encoding="utf-8")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "shared-cache"))
    monkeypatch.setattr(module, "_SOURCE_ROOT", tmp_path)
    _reset_closure_caches(module)
    assert {path.name for path in module._semantic_source_paths(("pkg/a.py",))} == {"a.py", "b.py"}

    (package / "c.py").write_text("C = 1\n", encoding="utf-8")
    _reset_closure_caches(module)

    assert {path.name for path in module._semantic_source_paths(("pkg/a.py",))} == {"a.py", "b.py", "c.py"}


def test_hermes_skill_asset_templates_are_not_admitted_as_sessions() -> None:
    """polylogue-6d7fx: the hermes-agent checkout bundled under the watched
    Hermes root ships prompt templates that are bare role/content message
    lists -- message-shaped by construction, so only their path can refuse
    them. The declared ``skill_asset`` rule must make
    ``path_declaration_refuses_session`` true for them while leaving a real
    ``~/.hermes/sessions`` transcript admissible.

    Anti-vacuity: deleting the ``skill_asset`` rule from ``_hermes_spec``
    (or loosening its ``parse_policy`` off ``raw-only``) makes the first
    assertion False.
    """
    from polylogue.sources.origin_specs import artifact_rule_for_path, path_declaration_refuses_session

    template = "/home/operator/.hermes/hermes-agent/optional-skills/security/godmode/templates/prefill.json"
    transcript = "/home/operator/.hermes/sessions/2026-09-01-session.json"

    assert path_declaration_refuses_session(Provider.HERMES, template) is True
    assert path_declaration_refuses_session(Provider.HERMES, transcript) is False

    rule = artifact_rule_for_path(Provider.HERMES, template)
    assert rule is not None
    assert rule.kind == "skill_asset"
    assert rule.parse_policy == "raw-only"


_ATOF_RECORD = b'{"atof_version": "0.1", "kind": "mark", "uuid": "u", "timestamp": "t", "name": "n"}'


def test_jsonl_storage_strategy_keeps_every_valid_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Spilling changes storage only; a foreign mapping remains census evidence."""
    from contextlib import closing

    from polylogue.archive.raw_payload.decode import _decode_jsonl_payload
    from polylogue.sources import decoder_json
    from polylogue.sources.decoder_json import DecodedRecordSequence

    monkeypatch.setattr(decoder_json, "_JSONL_MEMORY_BYTES", 256)
    long_record = b'{"other": "' + b"x" * 400 + b'"}'
    payload = long_record + b"\n" + _ATOF_RECORD + b"\n"
    path = tmp_path / "events.jsonl"
    path.write_bytes(payload)
    recognition = recognize_source_class(Provider.HERMES, path)
    assert recognition is not None and recognition.source_class == "unsupported"
    with closing(
        DecodedRecordSequence.from_jsonl(io.BytesIO(payload), "events.jsonl", fail_on_decode_error=True)
    ) as records:
        assert len(records) == 2
        assert isinstance(records[0], dict) and records[0]["other"] == "x" * 400
        assert isinstance(records[1], dict) and records[1]["uuid"] == "u"
    records, malformed, detail = _decode_jsonl_payload(payload)
    with closing(records):
        assert len(records) == 2
        assert isinstance(records[0], dict) and records[0]["other"] == "x" * 400
        assert isinstance(records[1], dict) and records[1]["uuid"] == "u"
        assert malformed == 0 and detail is None


def test_jsonl_byte_order_mark_is_stripped_from_the_first_decodable_line(tmp_path: Path) -> None:
    """Replay strips the BOM from the first line it can decode, not byte zero.

    Anti-vacuity: strip only at byte zero and the BOM-prefixed second record
    is skipped, so only the ATOF record is sampled and the mixed file is
    admitted although replay decodes ``{"other": 1}`` too.
    """
    from polylogue.archive.raw_payload.decode import _decode_jsonl_payload
    from polylogue.sources.origin_specs import recognize_source_class

    payload = b"\xff\xfe not utf-8\n" + b"\xef\xbb\xbf" + b'{"other": 1}\n' + _ATOF_RECORD + b"\n"
    path = tmp_path / "mixed.jsonl"
    path.write_bytes(payload)

    records, malformed, _detail = _decode_jsonl_payload(payload)
    assert records[0] == {"other": 1} and malformed == 1
    recognition = recognize_source_class(Provider.HERMES, path)
    assert recognition is not None and recognition.source_class == "unsupported"


def test_json_document_recognition_matches_the_record_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recognition drains the complete record stream without a document-size cap.

    A foreign tail or scalar root still refuses; aggregate width does not
    turn valid individually bounded records into an unsupported document.
    """
    from polylogue.sources import origin_specs
    from polylogue.sources.origin_specs import recognize_source_class

    record = '{"atof_version": "0.1", "kind": "mark", "uuid": "u", "timestamp": "t", "name": "n"}'
    mixed = tmp_path / "mixed.json"
    mixed.write_text("[" + ",".join([record] * 40 + ['{"other": 1}']) + "]", encoding="utf-8")
    refused = recognize_source_class(Provider.HERMES, mixed)
    assert refused is not None and refused.source_class == "unsupported"

    scalar = tmp_path / "scalar.json"
    scalar.write_bytes(b'"' + b"x" * 64)
    opened: list[object] = []
    real = origin_specs._signature_envelope

    def tracked(*args: Any, **kwargs: Any) -> Any:
        opened.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(origin_specs, "_signature_envelope", tracked)
    refused = recognize_source_class(Provider.HERMES, scalar)
    assert refused is not None and refused.source_class == "unsupported" and not opened

    import json

    from tests.infra.json_values import iter_owned_json_values

    large = tmp_path / "large.json"
    large.write_text("[" + ",".join([record] * 1201) + "]", encoding="utf-8")
    recognition = recognize_source_class(Provider.HERMES, large)
    assert recognition is not None and recognition.source_class == "session"
    with large.open("rb") as handle:
        records = iter_owned_json_values(handle, str(large))
        assert sum(1 for value in records if value == json.loads(record)) == 1201


def test_array_decoder_reads_provider_surrogates_through_the_stdlib_fallback() -> None:
    """Anti-vacuity: raise the partial-stream error when ijson refuses a later
    element's surrogate bytes and the whole valid document yields nothing."""
    import io

    from tests.infra.json_values import iter_owned_json_values

    first = b'{"atof_version": "0.1", "kind": "mark", "uuid": "a", "timestamp": "t", "name": "n"}'
    second = b'{"atof_version": "0.1", "kind": "mark", "uuid": "b", "timestamp": "t", "name": "n\xed\xa0\x80"}'
    records = list(iter_owned_json_values(io.BytesIO(b"[" + first + b"," + second + b"]"), "spans.json"))
    assert [record["uuid"] for record in records] == ["a", "b"]  # type: ignore[index,call-overload]


def test_orchestration_identity_with_a_surrogate_is_refused() -> None:
    """Anti-vacuity: return the parsed artifact and the lone surrogate reaches a
    SQLite binding in the workflow projector."""
    from polylogue.sources.parsers.claude.orchestration import parse_claude_orchestration_artifact

    payload = b'{"runId": "run\xed\xa0\x80", "status": "done"}\n'
    with pytest.raises(ValueError, match="surrogate"):
        parse_claude_orchestration_artifact(
            "/home/u/.claude/projects/p/s/subagents/workflows/run-1/journal.jsonl", payload
        )


def test_a_session_native_id_with_a_surrogate_is_refused_by_name() -> None:
    """Anti-vacuity: substitute the surrogate and two distinct provider sessions
    share one stored row; bind it raw and SQLite raises an unnamed error."""
    from polylogue.storage.sqlite.archive_tiers.write import _stored_session_native_id

    with pytest.raises(ValueError, match="surrogate"):
        _stored_session_native_id(" s\ud800 ")
    assert _stored_session_native_id(" s\ufffd ") == "s\ufffd"


def test_root_census_includes_a_directly_selected_file(tmp_path: Path) -> None:
    """The directory-only walk used to report an existing source as zero candidates."""
    source = tmp_path / "direct.json"
    source.write_text('{"session_id":"direct","messages":[],"platform":"cli"}', encoding="utf-8")

    census = census_source_root(source, provider=Provider.HERMES)

    assert census.candidate_count == 1
    assert census.accounted_count == 1
    assert census.is_complete
    assert census.candidate_bytes == source.stat().st_size


def test_root_census_accounts_for_rejected_nonregular_candidates(tmp_path: Path) -> None:
    """The regular-file filter used to remove links and FIFOs from the denominator."""
    target = tmp_path / "target.txt"
    target.write_text("neutral", encoding="utf-8")
    project = tmp_path / "-home-user-repo"
    project.mkdir()
    (project / "linked.jsonl").symlink_to(target)
    os.mkfifo(project / "pipe.jsonl")

    census = census_source_root(tmp_path, provider=Provider.CLAUDE_CODE)

    assert census.candidate_count == 2
    assert census.disposition_counts == {"session": 0, "non_session": 0, "unsupported": 2}
    assert census.is_complete


def test_source_walk_keeps_uninspectable_candidate_and_census_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old lstat exception handler returns an empty success and loses the candidate.

    Anti-vacuity: skip the entry on an lstat fault and the walk returns no
    path, so no per-file read can record the failure.
    """
    from polylogue.config import Source
    from polylogue.sources.source_walk import _resolve_source_paths

    source = tmp_path / "-home-user-repo" / "unreadable.jsonl"
    source.parent.mkdir()
    source.write_text("{}\n", encoding="utf-8")
    real_stat = os.stat

    def failed_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == source and kwargs.get("follow_symlinks") is False:
            raise PermissionError("synthetic transient inspection fault", str(source))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", failed_stat)
    assert _resolve_source_paths(Source(name="claude-code", path=tmp_path)) == [source]
    census = census_source_root(tmp_path, provider=Provider.CLAUDE_CODE)
    assert census.candidate_count == 1
    assert census.unexplained_candidates == (source,)
    assert not census.is_complete


def test_claude_history_rule_only_admits_the_install_root() -> None:
    from polylogue.core.enums import Provider
    from polylogue.sources.origin_specs import artifact_rule_for_path

    rule = artifact_rule_for_path(Provider.CLAUDE_CODE, "/neutral/install/.claude/history.jsonl")
    assert rule is not None and rule.kind == "prompt_history_log" and rule.parse_policy == "raw-only"
    for path in (
        "/neutral/install/.claude/plugins/example/history.jsonl",
        "/neutral/install/.claude/projects/example/history.jsonl",
        "/neutral/history.jsonl",
    ):
        found = artifact_rule_for_path(Provider.CLAUDE_CODE, path)
        assert found is None or found.kind != "prompt_history_log"

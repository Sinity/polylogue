from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import pytest

from devtools import verify_classifier_fingerprints as vcf
from polylogue.sources.origin_specs import OriginArtifactRule
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.lifecycle import index_delta_declaration_report

_CLAUDE_CODE_QUALNAME = "polylogue/sources/parsers/claude/code_detection.py:looks_like_code"
_RECORD_ENTRY_QUALNAME = "polylogue/archive/artifact_taxonomy/support.py:looks_like_record_entry"


def test_live_repo_classifier_fingerprint_manifest_is_current(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed manifest matches the live repo state: no missing/orphaned/drifted entries."""
    assert vcf.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["missing"] == []
    assert payload["orphaned"] == []
    assert payload["drifted"] == []
    assert payload["invalid_covered_by"] == []


def test_manifest_covers_the_pr_3428_functions() -> None:
    """Anti-vacuity: the two functions PR #3428 actually changed are tracked."""
    manifest = vcf.load_manifest()
    assert _CLAUDE_CODE_QUALNAME in manifest
    assert _RECORD_ENTRY_QUALNAME in manifest


def test_collect_classifier_functions_only_matches_looks_like_and_classify_artifact_names() -> None:
    functions = vcf.collect_classifier_functions()
    assert functions, "expected at least one classification function to be discovered"
    for qualname in functions:
        _, _, name = qualname.rpartition(":")
        assert name.startswith("looks_like") or name.startswith("classify_artifact")


def test_fingerprint_ignores_docstring_and_formatting_changes() -> None:
    src_a = '''
def looks_like_x(payload):
    """Original docstring."""
    return "a" in payload
'''
    src_b = '''
def looks_like_x(payload):
    """A totally different docstring, reworded for clarity."""
    return (
        "a"
        in
        payload
    )
'''
    fn_a = ast.parse(src_a).body[0]
    fn_b = ast.parse(src_b).body[0]
    assert isinstance(fn_a, ast.FunctionDef)
    assert isinstance(fn_b, ast.FunctionDef)
    assert vcf._fingerprint_function(fn_a) == vcf._fingerprint_function(fn_b)


def test_fingerprint_changes_when_logic_changes() -> None:
    src_a = "def looks_like_x(payload):\n    return 'a' in payload\n"
    src_b = "def looks_like_x(payload):\n    return 'a' in payload or 'b' in payload\n"
    fn_a = ast.parse(src_a).body[0]
    fn_b = ast.parse(src_b).body[0]
    assert isinstance(fn_a, ast.FunctionDef)
    assert isinstance(fn_b, ast.FunctionDef)
    assert vcf._fingerprint_function(fn_a) != vcf._fingerprint_function(fn_b)


def test_fingerprint_changes_when_a_decorator_is_added() -> None:
    src_a = "def looks_like_x(payload):\n    return 'a' in payload\n"
    src_b = "@functools.lru_cache\ndef looks_like_x(payload):\n    return 'a' in payload\n"
    fn_a = ast.parse(src_a).body[0]
    fn_b = ast.parse(src_b).body[0]
    assert isinstance(fn_a, ast.FunctionDef)
    assert isinstance(fn_b, ast.FunctionDef)
    assert vcf._fingerprint_function(fn_a) != vcf._fingerprint_function(fn_b)


def test_collect_classifier_functions_skips_undecodable_file(tmp_path: Path) -> None:
    bad_file = tmp_path / "broken.py"
    bad_file.write_bytes(b"def looks_like_x(payload):\n    return b'\xff\xfe' in payload\n")
    found = vcf.collect_classifier_functions(roots=(tmp_path,))
    assert not found


def test_undeclared_drift_fails_and_declared_drift_passes() -> None:
    current = {
        "mod.py:looks_like_x": vcf.ClassifierFunction(
            qualname="mod.py:looks_like_x", path=vcf.ROOT / "mod.py", lineno=1, fingerprint="new-hash"
        )
    }
    undeclared_manifest: dict[str, vcf.ManifestEntry] = {
        "mod.py:looks_like_x": vcf.ManifestEntry(
            fingerprint="old-hash",
            covered_by=vcf.CoveredBy(kind="acknowledged_safe", reason="prior unrelated acknowledgment", ref="#1"),
        )
    }
    report = vcf.compute_drift_report(current=current, manifest=undeclared_manifest)
    assert report.ok is False
    assert "mod.py:looks_like_x" in report.drifted

    declared_manifest: dict[str, vcf.ManifestEntry] = {
        "mod.py:looks_like_x": vcf.ManifestEntry(
            fingerprint="new-hash",
            covered_by=vcf.CoveredBy(kind="acknowledged_safe", reason="prior unrelated acknowledgment", ref="#1"),
        )
    }
    ok_report = vcf.compute_drift_report(current=current, manifest=declared_manifest)
    assert ok_report.ok is True


def test_missing_and_orphaned_entries_fail() -> None:
    current = {
        "mod.py:looks_like_new": vcf.ClassifierFunction(
            qualname="mod.py:looks_like_new", path=vcf.ROOT / "mod.py", lineno=1, fingerprint="hash-1"
        )
    }
    manifest = {
        "mod.py:looks_like_gone": vcf.ManifestEntry(
            fingerprint="hash-2",
            covered_by=vcf.CoveredBy(kind="acknowledged_safe", reason="an old, now-removed classifier", ref="#2"),
        )
    }
    report = vcf.compute_drift_report(current=current, manifest=manifest)
    assert report.missing == ("mod.py:looks_like_new",)
    assert report.orphaned == ("mod.py:looks_like_gone",)
    assert report.ok is False


def test_semantic_reparse_covered_by_must_reference_a_real_declared_version() -> None:
    current = {
        "mod.py:looks_like_x": vcf.ClassifierFunction(
            qualname="mod.py:looks_like_x", path=vcf.ROOT / "mod.py", lineno=1, fingerprint="hash-1"
        )
    }
    manifest = {
        "mod.py:looks_like_x": vcf.ManifestEntry(
            fingerprint="hash-1",
            covered_by=vcf.CoveredBy(
                kind="semantic_reparse_version",
                reason="declared alongside a version bump that does not exist",
                ref="polylogue-zzzz",
                version=999_999,
            ),
        )
    }
    report = vcf.compute_drift_report(current=current, manifest=manifest)
    assert report.ok is False
    assert any(qualname == "mod.py:looks_like_x" for qualname, _ in report.invalid_covered_by)


def test_covered_by_reason_and_ref_shape_are_validated() -> None:
    too_short_reason = vcf.CoveredBy(kind="acknowledged_safe", reason="short", ref="polylogue-abcd")
    assert vcf._validate_covered_by(too_short_reason) is not None

    bad_ref = vcf.CoveredBy(
        kind="acknowledged_safe", reason="a long enough explanation of safety", ref="not-a-real-ref"
    )
    assert vcf._validate_covered_by(bad_ref) is not None

    good = vcf.CoveredBy(kind="acknowledged_safe", reason="a long enough explanation of safety", ref="polylogue-abcd")
    assert vcf._validate_covered_by(good) is None


def test_collect_origin_artifact_rules_covers_claude_code_workflow_kinds() -> None:
    """Anti-vacuity: the artifact-rule table (not just functions) is fingerprinted."""
    rules = vcf.collect_origin_artifact_rules()
    assert rules, "expected at least one OriginArtifactRule to be discovered"
    for qualname in rules:
        assert qualname.startswith("polylogue/sources/origin_specs.py:artifact_rule:")
    assert "polylogue/sources/origin_specs.py:artifact_rule:claude-code-session:agent_transcript" in rules


def test_artifact_rule_fingerprint_changes_when_parse_policy_changes() -> None:
    """The #3088 failure shape: parse_policy drift for identical bytes must move the hash."""
    rule_a = OriginArtifactRule(
        kind="agent_transcript",
        path_pattern=r"subagents/.*\.jsonl$",
        parse_policy="session",
        parser_path="polylogue/sources/parsers/claude/code_parser.py:parse_code_stream",
        coverage_role="attempt_transcript",
        fidelity_note="Attempt transcript is a session only when linked to a run.",
        path_suffixes=(".jsonl",),
    )
    rule_b = OriginArtifactRule(
        kind="agent_transcript",
        path_pattern=r"subagents/.*\.jsonl$",
        parse_policy="fact",
        parser_path="polylogue/sources/parsers/claude/code_parser.py:parse_code_stream",
        coverage_role="attempt_transcript",
        fidelity_note="Attempt transcript is a session only when linked to a run.",
        path_suffixes=(".jsonl",),
    )
    assert vcf._fingerprint_artifact_rule(rule_a) != vcf._fingerprint_artifact_rule(rule_b)


def test_artifact_rule_fingerprint_ignores_fidelity_note_changes() -> None:
    """fidelity_note is prose documentation, stripped the same way a docstring is."""
    rule_a = OriginArtifactRule(
        kind="agent_transcript",
        path_pattern=r"subagents/.*\.jsonl$",
        parse_policy="session",
        parser_path="polylogue/sources/parsers/claude/code_parser.py:parse_code_stream",
        coverage_role="attempt_transcript",
        fidelity_note="Original wording.",
        path_suffixes=(".jsonl",),
    )
    rule_b = OriginArtifactRule(
        kind="agent_transcript",
        path_pattern=r"subagents/.*\.jsonl$",
        parse_policy="session",
        parser_path="polylogue/sources/parsers/claude/code_parser.py:parse_code_stream",
        coverage_role="attempt_transcript",
        fidelity_note="A totally different, reworded explanation.",
        path_suffixes=(".jsonl",),
    )
    assert vcf._fingerprint_artifact_rule(rule_a) == vcf._fingerprint_artifact_rule(rule_b)


def test_gate_would_have_caught_pr_3088_artifact_rule_drift_regression() -> None:
    """Historical regression check (polylogue-qs4b verification requirement).

    Reconstructs the manifest state as if this gate had existed before PR
    #3088 changed ``parse_as_session`` for Claude Workflow artifact kinds by
    editing ``OriginArtifactRule.parse_policy`` in ``origin_specs.py`` with
    no ``INDEX_SCHEMA_VERSION`` bump (retroactively declared as the missing
    v48 delta by polylogue-lzh8). Asserts the gate flags undeclared drift
    when a rule's ``parse_policy`` is mutated for identical input bytes,
    proving this lint -- unlike ``devtools lab policy schema-versioning`` --
    would have failed such a PR green.
    """
    qualname = "polylogue/sources/origin_specs.py:artifact_rule:claude-code-session:agent_transcript"
    current = vcf.collect_origin_artifact_rules()
    assert qualname in current

    mutated_rule = OriginArtifactRule(
        kind="agent_transcript",
        path_pattern=r"(?:^|/)subagents/(?:[^/]+/)*agent-[^/]+\.(?:jsonl|ndjson)$",
        parse_policy="fact",
        parser_path="polylogue/sources/parsers/claude/code_parser.py:parse_code_stream",
        coverage_role="attempt_transcript",
        fidelity_note="Attempt transcript is a session only when provider workflow evidence links it to a run.",
        path_suffixes=(".jsonl", ".ndjson"),
    )
    stale_fingerprint = vcf._fingerprint_artifact_rule(mutated_rule)
    assert current[qualname].fingerprint != stale_fingerprint, (
        "the live rule's parse_policy is expected to differ from the mutated ('fact') one"
    )

    manifest = dict(vcf.load_manifest())
    manifest[qualname] = vcf.ManifestEntry(fingerprint=stale_fingerprint, covered_by=manifest[qualname].covered_by)

    report = vcf.compute_drift_report(current=vcf.collect_all_classification_surfaces(), manifest=manifest)
    assert qualname in report.drifted
    assert report.ok is False

    # And devtools lab policy schema-versioning genuinely does not see this class of
    # change -- the whole point of this bead: only the fingerprint gate catches it.
    schema_report = index_delta_declaration_report(INDEX_SCHEMA_VERSION)
    assert schema_report["ok"] is True


def test_gate_would_have_caught_pr_3428_classifier_drift_regression() -> None:
    """Historical regression check (polylogue-gucv verification requirement).

    Reconstructs the manifest state as if this gate had existed the day
    before PR #3428 (ab8a92c1a) landed -- pinning ``looks_like_code``'s
    pre-PR fingerprint -- and asserts the gate flags undeclared drift
    against the current (post-PR) source, proving this lint would have
    failed that PR green today.
    """
    old_source = subprocess.run(
        ["git", "show", "ab8a92c1a~1:polylogue/sources/parsers/claude/code_detection.py"],
        cwd=vcf.ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    old_tree = ast.parse(old_source)
    old_fn = next(
        node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == "looks_like_code"
    )
    old_fingerprint = vcf._fingerprint_function(old_fn)

    current = vcf.collect_classifier_functions()
    assert _CLAUDE_CODE_QUALNAME in current
    assert current[_CLAUDE_CODE_QUALNAME].fingerprint != old_fingerprint, (
        "PR #3428 is expected to have changed looks_like_code's classification logic"
    )

    manifest = dict(vcf.load_manifest())
    manifest[_CLAUDE_CODE_QUALNAME] = vcf.ManifestEntry(
        fingerprint=old_fingerprint,
        covered_by=manifest[_CLAUDE_CODE_QUALNAME].covered_by,
    )

    report = vcf.compute_drift_report(current=current, manifest=manifest)
    assert _CLAUDE_CODE_QUALNAME in report.drifted
    assert report.ok is False

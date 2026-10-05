"""Worker-one regression witnesses for actual test-infrastructure routes.

Each test names the behavior that was previously vacuous or incorrect. No
production result is replaced with an independently reimplemented result.
"""

from __future__ import annotations

import json
import socket
import sqlite3
import subprocess
import tempfile
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


def test_catalog_rejects_fixture_keys_that_normalize_to_one_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two distinct declarations cannot silently share one normalized fixture."""
    from tests.infra import continuity

    declarations = tuple(
        SimpleNamespace(scenario_id=key, fixture_key=key, required_facts=("count",)) for key in ("foo-bar", "foo_bar")
    )
    monkeypatch.setattr(continuity, "CONTINUITY_SCENARIOS", declarations)
    payload = {
        "schema_version": 2,
        "fixture_id": "normalization-collision",
        "corpus": {"foo_bar": {"count": 1}},
        "oracles": {
            item.fixture_key: {"facts": {"count": 1}, "source_refs": ["fixture:synthetic"]} for item in declarations
        },
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="collide after snake_case normalization"):
        continuity.load_continuity_catalog(path)


@pytest.mark.parametrize("name", ["capped-pseudo-total", "identical-call-topology-replay"])
def test_incident_mutations_leave_negative_membership_queries_untouched(name: str) -> None:
    """A negative membership query must neither mutate nor consume a fault's state."""
    from tests.infra.continuity_mutations import continuity_mutation

    mutate = continuity_mutation(name).response_mutator
    assert mutate is not None
    negative: dict[str, object] = {"expression": 'text:parallel-child AND NOT text:"workflow_run:run"'}
    positive: dict[str, object] = {"expression": 'text:parallel-child AND text:"workflow_run:run"'}
    first = json.dumps({"rows": [{"id": "first"}], "continuation": "page-2", "next_offset": 1})
    second = json.dumps({"rows": [{"id": "second"}], "continuation": None, "next_offset": None})
    assert mutate("query", negative, 0, first) == first
    assert mutate("query", negative, 1, second) == second
    changed_first = mutate("query", positive, 2, first)
    if name == "capped-pseudo-total":
        assert json.loads(changed_first)["continuation"] is None
    else:
        assert changed_first == first
        changed_second = mutate("query", positive, 3, second)
        assert json.loads(changed_second)["rows"] == json.loads(first)["rows"]


def test_archive_green_gate_refuses_an_ordinary_verification_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing verification input cannot masquerade as a green archive."""
    from polylogue.core.outcomes import OutcomeStatus
    from polylogue.maintenance.archive_verification import ArchiveVerificationCheck, ArchiveVerificationReport
    from tests.infra import convergence_harness

    report = ArchiveVerificationReport(
        checks=[ArchiveVerificationCheck(name="missing-input", status=OutcomeStatus.SKIP)]
    )
    monkeypatch.setattr(convergence_harness, "verify_archive", lambda _root: report)
    with pytest.raises(AssertionError, match="not green"):
        convergence_harness.assert_archive_verification_green(tmp_path)


def test_convergence_oracle_identity_is_independent_of_production_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken provider-to-origin mapping must disagree with the oracle."""
    from polylogue.core.enums import Origin, Provider
    from polylogue.pipeline import ids
    from tests.infra.convergence_laws import AuthoritativeSession

    monkeypatch.setattr(ids, "origin_from_provider", lambda _provider: Origin.CLAUDE_CODE_SESSION)
    expected = AuthoritativeSession(native_id="w1-session", messages=()).session_id
    assert expected == "codex-session:w1-session"
    assert ids.session_id(Provider.CODEX, "w1-session") != expected


def test_convergence_plan_refuses_an_empty_execution() -> None:
    """A plan cannot report success without opening any archive."""
    from tests.infra.convergence_laws import ConvergenceLaw, build_convergence_run_plan, execute_convergence_plan

    with pytest.raises(ValueError, match="at least one archive root"):
        execute_convergence_plan(build_convergence_run_plan(), (), law=ConvergenceLaw.PERMUTATION)


def test_convergence_plan_reads_its_declared_probe_set(tmp_path: Path) -> None:
    """The real write/converge/read route must use a custom plan's search terms."""
    from tests.infra.convergence_harness import build_converged_archive
    from tests.infra.convergence_laws import (
        ConvergenceLaw,
        build_convergence_run_plan,
        execute_convergence_plan,
        generated_convergence_workload,
    )

    workload = replace(generated_convergence_workload(), probe_terms=("shared",))
    plan = build_convergence_run_plan(workload)
    root = tmp_path / "archive"
    root.mkdir()
    # Fixture ingest prepares lease-free and takes the writer lease itself.
    archive = build_converged_archive(root, workload.sources)
    execute_convergence_plan(plan, (archive.root,), law=ConvergenceLaw.PERMUTATION)


def test_daemon_operation_stack_can_rebind_a_stale_explicit_socket(tmp_path: Path) -> None:
    """The permission probe must not bind the stale path before the real server."""
    from tests.infra.daemon_operations import running_daemon_operations

    with tempfile.TemporaryDirectory(prefix="plg-w1-", dir="/tmp") as directory:
        path = Path(directory) / "daemon.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
            try:
                stale.bind(str(path))
            except PermissionError:
                pytest.skip("sandbox denies AF_UNIX listeners")
        assert path.exists()
        with running_daemon_operations(tmp_path / "archive", socket_path=path) as stack:
            assert stack.server.socket.getsockname() == str(path)


def test_ordering_revision_pilot_distinguishes_last_seen_and_sorted_mutants() -> None:
    """The actual representation projection rejects both stated shortcut mutants."""
    from tests.infra.pilot_scenarios import ordering_revision_relations, project_representation

    case = ordering_revision_relations()
    for representation in case.representations:
        assert project_representation(case, representation) == case.expected
    histories = case.representations[1]["revisions"]
    last_seen = tuple((native_id, revisions[-1]) for native_id, revisions in histories.items())
    assert last_seen != case.expected.revision_order
    assert tuple(sorted(case.expected.revision_order)) != case.expected.revision_order


def test_census_copy_discards_a_failed_reflinks_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed cp that already made a directory must not prevent the byte copy."""
    from tests.infra import query_census

    source, destination = tmp_path / "source", tmp_path / "snapshot"
    source.mkdir()
    for tier in query_census.REQUIRED_TIERS:
        with closing(sqlite3.connect(source / tier)) as conn:
            conn.execute("CREATE TABLE witness (value TEXT)")
            conn.execute("INSERT INTO witness VALUES ('synthetic')")
            conn.commit()

    def failed_reflink(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        destination.mkdir()
        (destination / "partial-copy").write_text("incomplete", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 1, b"", b"reflink unavailable")

    monkeypatch.setattr(subprocess, "run", failed_reflink)
    snapshot = query_census.reflink_archive_snapshot(source, destination)
    assert not snapshot.reflinked
    assert not (destination / "partial-copy").exists()
    with closing(sqlite3.connect(snapshot.index_path)) as conn:
        assert conn.execute("SELECT value FROM witness").fetchall() == [("synthetic",)]


@pytest.mark.parametrize("alias,classified", [("a", True), ("ap", False), ("archive_rollup", False)])
def test_census_scan_allowance_matches_sqlite_alias_tokens(alias: str, classified: bool) -> None:
    """An allowance for SCAN a must not exempt a different SQLite scan alias."""
    from tests.infra.query_census import PlanStep, _classify
    from tests.infra.query_contract import CENSUS_FAMILIES, ScanAllowance

    family = CENSUS_FAMILIES[0]
    allowance = ScanAllowance(family.family_id, "SCAN a", "expected", "w1-regression", "exact alias")
    family = replace(family, scan_allowances=(allowance,))
    statement = f"SELECT * FROM sample AS {alias}"
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE sample (value TEXT)")
        plans = [PlanStep(str(row[3])) for row in conn.execute("EXPLAIN QUERY PLAN " + statement)]
    scans = [step for step in plans if step.is_scan]
    assert len(scans) == 1
    assert _classify(statement, scans[0], family).classified is classified


@settings(max_examples=8, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_valid_cli_strategy_quotes_round_trip_through_the_expression_parser(
    monkeypatch: pytest.MonkeyPatch, data: st.DataObject
) -> None:
    """Quotes and backslashes in a valid generated value cannot break the DSL."""
    from polylogue.archive.query.expression import _FieldToken, parse_expression_ast
    from tests.infra.strategies import cli

    value = 'alpha"beta\\gamma東京'
    monkeypatch.setattr(cli, "_UNICODE_WORD", st.just(value))
    case = data.draw(
        cli.cli_interaction_case_strategy().filter(lambda item: not item.near_miss and item.query.startswith("title:"))
    )
    parsed = parse_expression_ast(case.query)
    clause = parsed.clauses[0]
    assert isinstance(clause, _FieldToken)
    assert clause.raw_value == value


@pytest.mark.parametrize(
    "actual,expected",
    [
        ("abc\rX", "Xbc"),
        ("abc\r\nz", "abc\nz"),
        ("a\x1b]0;first\x1b\\b\x1b]0;second\x1b\\c", "abc"),
    ],
)
def test_terminal_oracle_preserves_visible_text_and_applies_overwrites(actual: str, expected: str) -> None:
    """OSC stripping cannot eat visible text, and CR cannot leave duplicate cells."""
    from tests.infra.terminal_cells import normalize_terminal_text

    assert normalize_terminal_text(actual, columns=40) == normalize_terminal_text(expected, columns=40)


def test_terminal_overwrite_removes_the_whole_overlapping_wide_grapheme() -> None:
    """Overwriting one column of a wide cell cannot leave an overlapping cluster."""
    from tests.infra.terminal_cells import normalize_terminal_text

    frame = normalize_terminal_text("界b\rX", columns=40)
    assert [(cell.column, cell.grapheme, cell.display_width) for cell in frame.cells] == [(0, "X", 1), (2, "b", 1)]


@pytest.mark.parametrize("bad_entry", [None, 17, "not-an-object"])
def test_immutable_tree_rebuilds_a_manifest_with_non_mapping_entries(tmp_path: Path, bad_entry: object) -> None:
    """Syntactically valid corrupt JSON must be a cache miss, not AttributeError."""
    from tests.infra.workload_artifacts import build_immutable_tree

    builds = 0

    def build(root: Path) -> None:
        nonlocal builds
        builds += 1
        (root / "witness.txt").write_text("synthetic", encoding="utf-8")

    first = build_immutable_tree(cache_root=tmp_path, key="w1-malformed-entry", builder=build)
    path = first.root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["files"] = [bad_entry]
    path.chmod(0o600)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o444)
    rebuilt = build_immutable_tree(cache_root=tmp_path, key="w1-malformed-entry", builder=build)
    assert builds == 2
    assert (rebuilt.root / "witness.txt").read_text(encoding="utf-8") == "synthetic"


def test_convergence_provider_filter_preserves_the_canonical_seed_offset() -> None:
    """Selecting Codex must not regenerate the preceding provider's seed slot."""
    from tests.infra.workload_declarations import convergence_corpus_specs

    full = convergence_corpus_specs("xs-tiny", seed=911)
    selected = convergence_corpus_specs("xs-tiny", provider="codex", seed=911)
    assert len(full) > len(selected) == 1
    assert selected == tuple(spec for spec in full if spec.provider == "codex")
    assert selected[0].seed == 912


def test_convergence_workloads_keep_the_declared_default_style() -> None:
    """A benchmark formerly using default must not silently become tool-heavy."""
    from tests.infra.workload_declarations import convergence_corpus_specs

    specs = convergence_corpus_specs("xs-tiny")
    assert specs
    assert {spec.style for spec in specs} == {"default"}

from __future__ import annotations

from pathlib import Path

import pytest

from devtools.verification_graph import (
    GRAPH_IDENTITY_INPUTS,
    VerificationGraphError,
    attest_corpus,
    eligible_root,
    graph_identity,
    publish_complete_root,
    publish_selected_child,
)


def _identity(**overrides: str) -> str:
    values: dict[str, str] = {name: f"{name}-value" for name in GRAPH_IDENTITY_INPUTS}
    values.update(overrides)
    return graph_identity(**values)


def test_graph_identity_changes_for_every_named_environment_input() -> None:
    baseline = _identity()
    for name in GRAPH_IDENTITY_INPUTS:
        changed = _identity(**{name: f"changed-{name}"})
        assert changed != baseline, name


def test_graph_identity_excludes_irrelevant_nondeterminism() -> None:
    assert _identity() == _identity()


def test_corpus_attestation_detects_collection_and_fixture_drift(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.json"
    fixture.write_text("{}\n", encoding="utf-8")
    first = attest_corpus(("tests/test_one.py::test_a",), fixture_files=(fixture,), root=tmp_path)
    fixture.write_text('{"changed": true}\n', encoding="utf-8")
    fixture_changed = attest_corpus(("tests/test_one.py::test_a",), fixture_files=(fixture,), root=tmp_path)
    collection_changed = attest_corpus(("tests/test_one.py::test_b",), fixture_files=(fixture,), root=tmp_path)
    assert first.digest != fixture_changed.digest
    assert fixture_changed.digest != collection_changed.digest


def test_only_successful_complete_run_publishes_root(tmp_path: Path) -> None:
    corpus = attest_corpus(("tests/test_one.py::test_a",))
    digest = _identity(corpus_attestation=corpus.digest)
    assert (
        publish_complete_root(
            tmp_path, graph_digest=digest, corpus=corpus, run_id="failed", terminal_status="failed", complete=True
        )
        is None
    )
    assert eligible_root(tmp_path, digest) is None
    assert (
        publish_complete_root(
            tmp_path, graph_digest=digest, corpus=corpus, run_id="partial", terminal_status="success", complete=False
        )
        is None
    )
    assert eligible_root(tmp_path, digest) is None
    published = publish_complete_root(
        tmp_path, graph_digest=digest, corpus=corpus, run_id="complete", terminal_status="success", complete=True
    )
    assert published is not None
    assert eligible_root(tmp_path, digest) == published


def test_selected_child_requires_parent_and_is_never_root(tmp_path: Path) -> None:
    corpus = attest_corpus(("tests/test_one.py::test_a",))
    parent = _identity(corpus_attestation=corpus.digest)
    child = _identity(corpus_attestation="child-corpus")
    with pytest.raises(VerificationGraphError, match="parent root"):
        publish_selected_child(
            tmp_path,
            parent_digest=parent,
            graph_digest=child,
            selection=corpus.nodeids,
            change_lineage={"base": parent, "head": "child"},
            outcome={"exit_code": 1},
            run_id="child",
        )
    publish_complete_root(
        tmp_path, graph_digest=parent, corpus=corpus, run_id="parent", terminal_status="success", complete=True
    )
    path = publish_selected_child(
        tmp_path,
        parent_digest=parent,
        graph_digest=child,
        selection=corpus.nodeids,
        change_lineage={"base": parent, "head": "child"},
        outcome={"exit_code": 0},
        run_id="child",
    )
    assert path.exists()
    assert eligible_root(tmp_path, child) is None

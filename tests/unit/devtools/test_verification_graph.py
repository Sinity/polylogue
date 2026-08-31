from __future__ import annotations

from pathlib import Path

import pytest

from devtools.verification_graph import (
    GRAPH_IDENTITY_INPUTS,
    VerificationGraphError,
    attest_corpus,
    eligible_root,
    graph_identity,
    latest_eligible_root,
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


def test_worktree_reads_roots_through_the_parent_indirection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worktree with no local roots resolves the main checkout's sealed root."""
    main_checkout = tmp_path / "main"
    worktree = tmp_path / "worktree"
    main_checkout.mkdir()
    worktree.mkdir()
    corpus = attest_corpus(("tests/test_one.py::test_a",))
    digest = _identity(corpus_attestation=corpus.digest)
    published = publish_complete_root(
        main_checkout,
        graph_digest=digest,
        corpus=corpus,
        run_id="sealed",
        terminal_status="success",
        complete=True,
    )
    assert published is not None
    assert eligible_root(worktree, digest) is None
    monkeypatch.setenv("POLYLOGUE_VERIFY_GRAPH_PARENT", str(main_checkout))
    resolved = eligible_root(worktree, digest)
    assert resolved is not None
    assert main_checkout in resolved.parents
    latest = latest_eligible_root(worktree)
    assert latest is not None and latest[0] == digest
    # The indirection is read-only: publication still lands locally.
    local = publish_complete_root(
        worktree,
        graph_digest=digest,
        corpus=corpus,
        run_id="sealed",
        terminal_status="success",
        complete=True,
    )
    assert local is not None and worktree in local.parents

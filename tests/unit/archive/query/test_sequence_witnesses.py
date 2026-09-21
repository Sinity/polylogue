"""``seq()`` returns the actions that matched, not only that something did.

polylogue-4p1.7: the sequence predicate lowered to an ``EXISTS`` over a join
chain and discarded the matched span, so a session could be reported as
containing ``seq(action:file_edit -> action:shell)`` with no way to project,
group or cite *which* edit and *which* shell command matched.

The existential predicate and the witness read are now two uses of one
relation (``_action_sequence_witness_relation``): a semijoin and a
projection. These tests run the production ``ArchiveStore`` routes.

Anti-vacuity: make ``_action_sequence_witness_relation`` drop its per-step
bindings -- have ``binding_columns`` project a constant instead of
``seq_a{i}.tool_use_block_id`` -- and
``test_witness_bindings_are_the_actual_matching_actions`` goes red, because
it compares the returned action ids against the ids the archive actually
stores for those steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.query.expression import parse_expression_ast
from polylogue.archive.query.predicate import QuerySequencePredicate
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder


def _tool_use(tool_name: str, tool_id: str, semantic_type: str, command: str) -> dict[str, object]:
    return {
        "type": "tool_use",
        "tool_name": tool_name,
        "tool_id": tool_id,
        "input": {"command": command},
        "semantic_type": semantic_type,
    }


def _seed(index_db: Path, session: str, steps: tuple[tuple[str, str, str], ...]) -> None:
    builder = SessionBuilder(index_db, session).provider("claude-code")
    for position, (tool_name, tool_id, semantic_type) in enumerate(steps):
        builder = builder.add_message(
            f"m{position}",
            role="assistant",
            text="step",
            blocks=[_tool_use(tool_name, tool_id, semantic_type, f"cmd-{tool_id}")],
        )
    builder.save()


def _sequence_predicate(expression: str) -> QuerySequencePredicate:
    predicate = parse_expression_ast(expression).boolean_predicate
    assert isinstance(predicate, QuerySequencePredicate), predicate
    return predicate


def _stored_action_ids(archive: ArchiveStore, session_native: str) -> dict[str, str]:
    """Map tool_id -> tool_use_block_id for one session, straight from SQL."""
    rows = archive._conn.execute(
        "SELECT tool_id, tool_use_block_id FROM action_pairs WHERE session_id LIKE ?",
        (f"%{session_native}%",),
    ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def test_witness_bindings_are_the_actual_matching_actions(workspace_env: dict[str, Path]) -> None:
    index_db = workspace_env["archive_root"] / "index.db"
    _seed(
        index_db,
        "seq-witness",
        (("Edit", "edit-1", "file_edit"), ("Bash", "shell-1", "shell")),
    )
    predicate = _sequence_predicate("seq(action:file_edit -> action:shell)")

    with ArchiveStore(workspace_env["archive_root"], initialize=False, read_only=True) as archive:
        witnesses = archive.query_action_sequence_witnesses(predicate, limit=50)
        stored = _stored_action_ids(archive, "seq-witness")

    assert len(witnesses) == 1
    witness = witnesses[0]
    assert [binding.step_index for binding in witness.bindings] == [0, 1]
    # The witness names the archive's own action ids, not a re-derived set.
    assert witness.action_ids == (stored["edit-1"], stored["shell-1"])
    assert witness.span == (stored["edit-1"], stored["shell-1"])
    # Every binding resolves to the message that carried the action.
    assert all(binding.message_id for binding in witness.bindings)
    assert witness.bindings[0].message_position < witness.bindings[1].message_position


def test_multiplicity_is_all_pairs(workspace_env: dict[str, Path]) -> None:
    """Two edits before two shells yield four matches, not one."""
    index_db = workspace_env["archive_root"] / "index.db"
    _seed(
        index_db,
        "seq-pairs",
        (
            ("Edit", "edit-1", "file_edit"),
            ("Edit", "edit-2", "file_edit"),
            ("Bash", "shell-1", "shell"),
            ("Bash", "shell-2", "shell"),
        ),
    )
    predicate = _sequence_predicate("seq(action:file_edit -> action:shell)")

    with ArchiveStore(workspace_env["archive_root"], initialize=False, read_only=True) as archive:
        witnesses = archive.query_action_sequence_witnesses(predicate, limit=100)

    assert len(witnesses) == 4
    assert len({witness.action_ids for witness in witnesses}) == 4


def test_limit_pages_the_matches_rather_than_sampling_them(workspace_env: dict[str, Path]) -> None:
    """A limit returns the first matches under the declared order.

    Without a total order over the binding coordinates, ``limit=1`` would
    return an arbitrary member of the all-pairs relation.
    """
    index_db = workspace_env["archive_root"] / "index.db"
    _seed(
        index_db,
        "seq-page",
        (
            ("Edit", "edit-1", "file_edit"),
            ("Edit", "edit-2", "file_edit"),
            ("Bash", "shell-1", "shell"),
            ("Bash", "shell-2", "shell"),
        ),
    )
    predicate = _sequence_predicate("seq(action:file_edit -> action:shell)")

    with ArchiveStore(workspace_env["archive_root"], initialize=False, read_only=True) as archive:
        everything = archive.query_action_sequence_witnesses(predicate, limit=100)
        first = archive.query_action_sequence_witnesses(predicate, limit=1)
        second = archive.query_action_sequence_witnesses(predicate, limit=1, offset=1)
        remainder = archive.query_action_sequence_witnesses(predicate, limit=100, offset=1)

    assert [w.action_ids for w in first] == [everything[0].action_ids]
    assert [w.action_ids for w in second] == [everything[1].action_ids]
    assert [w.action_ids for w in remainder] == [w.action_ids for w in everything[1:]]


def test_witnesses_agree_with_the_existential_predicate(workspace_env: dict[str, Path]) -> None:
    """A session yields witnesses exactly when the predicate selects it.

    This is the semijoin property: the predicate is ``SELECT 1`` over the same
    relation this read projects, so neither can claim a match the other denies.
    """
    index_db = workspace_env["archive_root"] / "index.db"
    _seed(index_db, "seq-yes", (("Edit", "e1", "file_edit"), ("Bash", "s1", "shell")))
    _seed(index_db, "seq-no", (("Bash", "s2", "shell"), ("Edit", "e2", "file_edit")))
    predicate = _sequence_predicate("seq(action:file_edit -> action:shell)")

    with ArchiveStore(workspace_env["archive_root"], initialize=False, read_only=True) as archive:
        selected = {row.session_id for row in archive.list_summaries(limit=100, boolean_predicate=predicate)}
        witnessed = {w.session_id for w in archive.query_action_sequence_witnesses(predicate, limit=100)}

    assert witnessed == selected
    assert any("seq-yes" in session_id for session_id in witnessed)
    assert not any("seq-no" in session_id for session_id in witnessed)


def test_next_adjacency_is_unchanged(workspace_env: dict[str, Path]) -> None:
    """``[next]`` stays alphabet-relative: no *action* may sit between."""
    index_db = workspace_env["archive_root"] / "index.db"
    _seed(
        index_db,
        "seq-next",
        (
            ("Edit", "edit-1", "file_edit"),
            ("Read", "read-1", "file_read"),
            ("Bash", "shell-1", "shell"),
        ),
    )
    ordered = _sequence_predicate("seq(action:file_edit -> action:shell)")
    adjacent = _sequence_predicate("seq(action:file_edit ->[next] action:shell)")

    with ArchiveStore(workspace_env["archive_root"], initialize=False, read_only=True) as archive:
        assert len(archive.query_action_sequence_witnesses(ordered, limit=10)) == 1
        assert archive.query_action_sequence_witnesses(adjacent, limit=10) == []


def test_a_one_step_pattern_is_refused(workspace_env: dict[str, Path]) -> None:
    with ArchiveStore(workspace_env["archive_root"], initialize=False, read_only=True) as archive:
        with pytest.raises(ValueError, match="at least two steps"):
            archive.query_action_sequence_witnesses(QuerySequencePredicate(action_terms=("file_edit",)), limit=10)

from pathlib import Path

from polylogue.branch_explorer import _load_branch_overview, BranchNodeSummary
from polylogue.db import open_connection


def _seed_branch_data(tmp_path: Path):
    db_path = tmp_path / "state.db"
    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO conversations(provider, conversation_id, slug)
            VALUES('p', 'c1', 'slug')
            """
        )
        conn.executemany(
            """
            INSERT INTO branches(provider, conversation_id, branch_id, parent_branch_id, depth, is_current, metadata_json)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("p", "c1", "branch-000", None, 2, 1, '{"divergence_index": 0, "message_count": 2}'),
                ("p", "c1", "branch-001", "branch-000", 1, 0, '{"divergence_index": 0, "message_count": 1}'),
            ],
        )
        conn.executemany(
            """
            INSERT INTO messages(provider, conversation_id, branch_id, message_id, parent_id, position, role, rendered_text)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("p", "c1", "branch-000", "m0", None, 0, "user", "hello"),
                ("p", "c1", "branch-000", "m1", "m0", 1, "assistant", "world"),
                ("p", "c1", "branch-001", "m2", "m0", 0, "assistant", "branch"),
            ],
        )
    return db_path


def test_branch_overview_batches_queries(tmp_path: Path):
    db_path = _seed_branch_data(tmp_path)
    summary = _load_branch_overview(
        provider="p",
        conversation_id="c1",
        slug="slug",
        title="t",
        current_branch="branch-000",
        last_updated=None,
        conversation_path=None,
        db_path=db_path,
    )
    assert summary.canonical_branch_id == "branch-000"
    assert summary.nodes["branch-000"].divergence_role == "user"
    assert summary.nodes["branch-001"].divergence_role == "assistant"
    assert isinstance(summary.nodes["branch-000"], BranchNodeSummary)

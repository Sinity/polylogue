"""A per-session row allowance must bound the database work, not just the rows.

``query_session_messages(per_session_limit=...)`` used to rank every matching
row of every selected session with ``ROW_NUMBER() OVER (PARTITION BY ...)`` and
discard the ranks above the allowance afterwards, so a session with a very long
history sorted its whole history to hand back a 201-row page. The declared row
ceiling then bounded the answer but not the cost of producing it.

Anti-vacuity: restore the window subquery in
``polylogue/storage/sqlite/archive_tiers/archive_query_reads.py`` and
``test_allowance_cost_does_not_follow_history`` goes red -- the measured VDBE
step count grows with the partition instead of staying flat. The
``test_allowance_still_...`` cases pin the opposite direction, so a fix that
simply stopped bounding per session (or returned fewer rows) cannot pass.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.storage_records import SessionBuilder

#: Partition sizes whose ratio is large enough that a per-partition scan is
#: unmistakable: an O(history) plan costs ~4x more at the larger size, while a
#: bounded indexed read costs the same at both.
_SMALL_HISTORY = 1_500
_LARGE_HISTORY = 6_000
_ALLOWANCE = 201


def _seed(root: Path, name: str, count: int) -> str:
    builder = SessionBuilder(root / "index.db", name).provider("claude-code").title(name)
    for index in range(count):
        builder = builder.add_message(
            f"m-{index:06d}",
            role="user" if index % 2 == 0 else "assistant",
            text=f"body {index}",
        )
    builder.save()
    return f"claude-code-session:ext-{name}"


def _measure_steps(root: Path, session_ids: list[str], *, per_session_limit: int | None) -> tuple[int, tuple[str, ...]]:
    """Return VDBE step units and message ids for one bounded page read."""
    with ArchiveStore.open_existing(root) as archive:
        conn: sqlite3.Connection = archive._conn
        steps = 0

        def _handler() -> int:
            nonlocal steps
            steps += 1
            return 0

        conn.set_progress_handler(_handler, 1000)
        try:
            rows = archive.query_session_messages(
                session_ids,
                limit=_ALLOWANCE * len(session_ids),
                offset=0,
                sort_direction="asc",
                per_session_limit=per_session_limit,
            )
        finally:
            conn.set_progress_handler(None, 1000)
        return steps, tuple(row.message_id for row in rows)


def _measure_previous_window(root: Path, session_ids: list[str]) -> tuple[int, tuple[str, ...]]:
    """Measure the rank-before-bound query from the parent of d9a756de7.

    Selecting only ids makes this a lower bound on the old query's work: its
    full row projection, repo lookup, and block text would cost more.
    """
    placeholders = ", ".join("?" for _ in session_ids)
    with ArchiveStore.open_existing(root) as archive:
        conn: sqlite3.Connection = archive._conn
        steps = 0

        def _handler() -> int:
            nonlocal steps
            steps += 1
            return 0

        conn.set_progress_handler(_handler, 1000)
        try:
            rows = conn.execute(
                f"""
                SELECT message_id FROM (
                    SELECT m.message_id, m.position, m.variant_index,
                           ROW_NUMBER() OVER (
                               PARTITION BY m.session_id
                               ORDER BY m.position ASC, m.variant_index ASC, m.message_id ASC
                           ) AS unit_rank
                    FROM messages m INDEXED BY idx_messages_session_position
                    WHERE m.session_id IN ({placeholders})
                )
                WHERE unit_rank <= ?
                ORDER BY position ASC, variant_index ASC, message_id ASC
                LIMIT ? OFFSET 0
                """,
                [*session_ids, _ALLOWANCE, _ALLOWANCE * len(session_ids)],
            ).fetchall()
        finally:
            conn.set_progress_handler(None, 1000)
        return steps, tuple(str(row["message_id"]) for row in rows)


class TestAllowanceBoundsTheWork:
    def test_allowance_cost_does_not_follow_history(self, tmp_path: Path) -> None:
        """The same allowance over a 4x longer session costs the same work.

        Measured before the fix: 1,500 rows cost ~322k VDBE step units and
        6,000 rows cost ~1.19M -- the full partition, ranked and sorted, to
        return 201 rows. A bounded indexed read costs the same at both sizes.
        """
        small_root = tmp_path / "small"
        large_root = tmp_path / "large"
        small_root.mkdir()
        large_root.mkdir()
        small_id = _seed(small_root, "small", _SMALL_HISTORY)
        large_id = _seed(large_root, "large", _LARGE_HISTORY)

        small_steps, small_rows = _measure_steps(small_root, [small_id], per_session_limit=_ALLOWANCE)
        large_steps, large_rows = _measure_steps(large_root, [large_id], per_session_limit=_ALLOWANCE)

        assert len(small_rows) == _ALLOWANCE
        assert len(large_rows) == _ALLOWANCE
        # A 4x partition may cost a little more (the page's own projection is
        # unchanged) but it must not cost proportionally more.
        assert large_steps <= small_steps * 3 // 2, (
            f"allowance cost follows history: {small_steps} -> {large_steps} step units "
            f"for the same {_ALLOWANCE}-row allowance over a {_SMALL_HISTORY}- vs "
            f"{_LARGE_HISTORY}-message session"
        )

    def test_skewed_page_costs_less_than_the_previous_window(self, tmp_path: Path) -> None:
        """One long and one short partition retain the old answer with bounded work.

        The comparison executes the pre-d9a756de7 rank-before-bound shape on
        the same archive. Restoring that shape as the product query makes the
        work comparison fail even though the returned page remains correct.
        """
        short_id = _seed(tmp_path, "short", 5)
        long_id = _seed(tmp_path, "long", _LARGE_HISTORY)
        session_ids = [short_id, long_id]
        previous_steps, previous_ids = _measure_previous_window(tmp_path, session_ids)
        bounded_steps, bounded_ids = _measure_steps(tmp_path, session_ids, per_session_limit=_ALLOWANCE)

        assert bounded_ids == previous_ids
        assert sum(message_id.startswith(f"{short_id}:") for message_id in bounded_ids) == 5
        assert sum(message_id.startswith(f"{long_id}:") for message_id in bounded_ids) == _ALLOWANCE
        assert bounded_steps * 2 < previous_steps, (
            f"rank-before-bound: {previous_steps}k VDBE steps; bounded indexed read: "
            f"{bounded_steps}k for the same {len(bounded_ids)}-row skewed page"
        )

    def test_allowance_still_bounds_each_session_separately(self, tmp_path: Path) -> None:
        """Opposite direction: a cheaper read that stopped bounding per session fails."""
        long_id = _seed(tmp_path, "long", 600)
        short_id = _seed(tmp_path, "short", 5)
        with ArchiveStore.open_existing(tmp_path) as archive:
            rows = archive.query_session_messages(
                [long_id, short_id],
                limit=10_000,
                offset=0,
                sort_direction="asc",
                per_session_limit=50,
            )
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.session_id] = counts.get(row.session_id, 0) + 1
        assert counts[long_id] == 50, "the long session must be cut to its own allowance"
        assert counts[short_id] == 5, "a session below the allowance keeps every row"

    def test_allowance_page_matches_the_unbounded_read(self, tmp_path: Path) -> None:
        """Opposite direction: the bounded read must return the same head rows.

        A bound that changed *which* rows a session contributes -- or dropped
        the page order -- would pass the cost assertion while answering a
        different question.
        """
        session_id = _seed(tmp_path, "order", 400)
        with ArchiveStore.open_existing(tmp_path) as archive:
            bounded = archive.query_session_messages(
                [session_id], limit=25, offset=0, sort_direction="asc", per_session_limit=25
            )
            plain = archive.query_session_messages([session_id], limit=25, offset=0, sort_direction="asc")
            tail = archive.query_session_messages(
                [session_id], limit=5, offset=0, sort_direction="desc", per_session_limit=5
            )
            plain_tail = archive.query_session_messages([session_id], limit=5, offset=0, sort_direction="desc")
        assert [row.message_id for row in bounded] == [row.message_id for row in plain]
        assert [row.message_id for row in tail] == [row.message_id for row in plain_tail]

    def test_allowance_offset_pages_through_the_bounded_set(self, tmp_path: Path) -> None:
        """Opposite direction: ``offset`` must still page the bounded rows."""
        session_id = _seed(tmp_path, "paging", 200)
        with ArchiveStore.open_existing(tmp_path) as archive:
            first = archive.query_session_messages(
                [session_id], limit=10, offset=0, sort_direction="asc", per_session_limit=30
            )
            second = archive.query_session_messages(
                [session_id], limit=10, offset=10, sort_direction="asc", per_session_limit=30
            )
            whole = archive.query_session_messages(
                [session_id], limit=30, offset=0, sort_direction="asc", per_session_limit=30
            )
        assert [row.message_id for row in first] == [row.message_id for row in whole[:10]]
        assert [row.message_id for row in second] == [row.message_id for row in whole[10:20]]

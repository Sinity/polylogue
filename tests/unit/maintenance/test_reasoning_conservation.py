"""Red twins for the ``reasoning-conservation`` owner check of ``verify-archive``.

The cohort is generated and mixed: one Claude Code session carrying the three
declared thinking variants and one Codex session carrying summary-bearing and
opaque reasoning, both materialized through the production parser and the
production archive writer. Each negative twin then breaks exactly one thing in
the index and must turn the check red on the term that names it.

Anti-vacuity: every twin leaves the source bytes untouched, so nothing but the
index changes. A check that read its denominator from the index, or that
compared one global total against another, would stay green on all four --
particularly ``test_offsetting_losses_do_not_pass_as_an_aggregate``, whose
index keeps the archive-wide thinking-block count exactly right.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.archive_verification import (
    ArchiveVerificationCheck,
    archive_verification_domain_adapters,
    verify_archive,
)
from polylogue.maintenance.reasoning_conservation import (
    DECLARED_VARIANTS,
    ORIGIN_CLAUDE_CODE,
    ORIGIN_CODEX,
    REASONING_ORIGINS,
    VARIANT_CONTENT,
    VARIANT_EMPTY,
    VARIANT_OPAQUE,
    VARIANT_SIGNATURE_ONLY,
    VARIANT_SUMMARY,
    VARIANT_TEXT,
    claude_code_witnesses,
    codex_witnesses,
)
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.claude.code_parser import parse_code
from polylogue.sources.parsers.codex import parse_stream
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

CHECK = "reasoning-conservation"

CLAUDE_SESSION_ID = "generated-claude-session"
CODEX_SESSION_ID = "generated-codex-session"

THINKING_TEXT = "weighing the two orderings"
SIGNATURE_ONLY_VALUE = "generated-signature-only-value"
CODEX_SUMMARY_TEXT = "planning the extraction"

CLAUDE_RECORDS: tuple[dict[str, object], ...] = (
    {
        "type": "user",
        "uuid": "u1",
        "sessionId": CLAUDE_SESSION_ID,
        "timestamp": "2026-01-01T00:00:00Z",
        "message": {"role": "user", "content": [{"type": "text", "text": "the question"}]},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "sessionId": CLAUDE_SESSION_ID,
        "timestamp": "2026-01-01T00:00:01Z",
        "message": {
            "id": "msg_text_bearing",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": THINKING_TEXT, "signature": "sig-with-text"},
                {"type": "text", "text": "the first answer"},
            ],
        },
    },
    {
        "type": "assistant",
        "uuid": "a2",
        "sessionId": CLAUDE_SESSION_ID,
        "timestamp": "2026-01-01T00:00:02Z",
        "message": {
            "id": "msg_signature_only",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "", "signature": SIGNATURE_ONLY_VALUE},
                {"type": "text", "text": "the second answer"},
            ],
        },
    },
    {
        "type": "assistant",
        "uuid": "a3",
        "sessionId": CLAUDE_SESSION_ID,
        "timestamp": "2026-01-01T00:00:03Z",
        "message": {
            "id": "msg_empty",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "the third answer"},
            ],
        },
    },
)

CODEX_RECORDS: tuple[dict[str, object], ...] = (
    {
        "type": "session_meta",
        "payload": {"id": CODEX_SESSION_ID, "timestamp": "2026-01-01T00:00:00Z", "cwd": "/generated"},
    },
    {
        "type": "response_item",
        "timestamp": "2026-01-01T00:00:01Z",
        "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "the ask"}]},
    },
    {
        "type": "response_item",
        "timestamp": "2026-01-01T00:00:02Z",
        "payload": {
            "type": "reasoning",
            "id": "rs_summary",
            "summary": [{"type": "summary_text", "text": CODEX_SUMMARY_TEXT}],
        },
    },
    {
        "type": "response_item",
        "timestamp": "2026-01-01T00:00:03Z",
        "payload": {"type": "reasoning", "id": "rs_opaque", "summary": [], "encrypted_content": "AAAABBBB"},
    },
    {
        "type": "response_item",
        "timestamp": "2026-01-01T00:00:04Z",
        "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
    },
)


def _jsonl(records: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(json.dumps(record).encode() + b"\n" for record in records)


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _insert_raw(root: Path, *, raw_id: str, origin: str, native_id: str, payload: bytes) -> None:
    blob_hash = BlobStore(root / "blob").write_from_bytes(payload)[0]
    conn = _connect(root / "source.db")
    try:
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, parsed_at_ms, logical_source_key
            ) VALUES (?, ?, ?, ?, ?, ?, 1000, 1000, ?)
            """,
            (
                raw_id,
                origin,
                native_id,
                f"/generated/{raw_id}.jsonl",
                bytes.fromhex(blob_hash),
                len(payload),
                f"{origin}:{native_id}",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _materialize(root: Path, *, raw_id: str, session: ParsedSession) -> None:
    conn = _connect(root / "index.db")
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        write_parsed_session_to_archive(
            conn, session, content_hash=hashlib.sha256(raw_id.encode()).hexdigest(), raw_id=raw_id
        )
        conn.commit()
    finally:
        conn.close()


def _seed_mixed_cohort(root: Path) -> None:
    """One Claude and one Codex session through the production parse/write route."""
    initialize_active_archive_root(root)
    claude_payload = _jsonl(CLAUDE_RECORDS)
    codex_payload = _jsonl(CODEX_RECORDS)
    _insert_raw(
        root,
        raw_id="raw-claude",
        origin=ORIGIN_CLAUDE_CODE,
        native_id=CLAUDE_SESSION_ID,
        payload=claude_payload,
    )
    _insert_raw(root, raw_id="raw-codex", origin=ORIGIN_CODEX, native_id=CODEX_SESSION_ID, payload=codex_payload)
    _materialize(root, raw_id="raw-claude", session=parse_code(list(CLAUDE_RECORDS), CLAUDE_SESSION_ID))
    _materialize(root, raw_id="raw-codex", session=parse_stream(list(CODEX_RECORDS), CODEX_SESSION_ID))


def _run(root: Path) -> ArchiveVerificationCheck:
    report = verify_archive(root, checks=(CHECK,))
    assert len(report.checks) == 1
    check = report.checks[0]
    assert isinstance(check, ArchiveVerificationCheck)
    return check


def _index(root: Path, statement: str, parameters: tuple[object, ...] = ()) -> None:
    conn = _connect(root / "index.db")
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(statement, parameters)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Structural selection
# ---------------------------------------------------------------------------


def test_witnesses_are_selected_structurally_not_by_identity() -> None:
    """Renaming every identity in the bytes leaves the denominator unchanged.

    A denominator that moved with a session id, a path or an operator name
    would be selecting witnesses by identity rather than by shape.
    """
    original = _jsonl(CLAUDE_RECORDS)
    renamed = original.replace(CLAUDE_SESSION_ID.encode(), b"a-completely-different-session")
    baseline = [witness.variant for witness in claude_code_witnesses(original)]

    assert baseline == [VARIANT_TEXT, VARIANT_SIGNATURE_ONLY, VARIANT_EMPTY]
    assert [witness.variant for witness in claude_code_witnesses(renamed)] == baseline
    assert [witness.variant for witness in codex_witnesses(_jsonl(CODEX_RECORDS))] == [
        VARIANT_SUMMARY,
        VARIANT_OPAQUE,
    ]


def test_declared_variants_cover_both_coding_origins() -> None:
    assert set(DECLARED_VARIANTS) == set(REASONING_ORIGINS)
    assert VARIANT_SIGNATURE_ONLY in DECLARED_VARIANTS[ORIGIN_CLAUDE_CODE]
    assert VARIANT_OPAQUE in DECLARED_VARIANTS[ORIGIN_CODEX]


def test_reasoning_conservation_is_declared_for_the_candidate_route() -> None:
    owner = next(declared for declared in archive_verification_domain_adapters(Path(".")) if declared.name == CHECK)
    assert owner.semantic_owner == "reasoning-materialization"
    assert "reindex-cross-tier-candidate" in owner.applicable_routes
    assert "blob/" in owner.population


# ---------------------------------------------------------------------------
# The production route conserves the mixed cohort
# ---------------------------------------------------------------------------


def test_production_route_conserves_every_variant_of_both_origins(tmp_path: Path) -> None:
    _seed_mixed_cohort(tmp_path)

    check = _run(tmp_path)

    assert check.status is OutcomeStatus.OK, check.summary
    origins = check.evidence["origins"]
    assert origins[ORIGIN_CLAUDE_CODE]["witnesses_by_variant"] == {
        VARIANT_TEXT: 1,
        VARIANT_SIGNATURE_ONLY: 1,
        VARIANT_EMPTY: 1,
    }
    assert origins[ORIGIN_CODEX]["witnesses_by_variant"] == {
        VARIANT_SUMMARY: 1,
        VARIANT_CONTENT: 0,
        VARIANT_OPAQUE: 1,
    }
    assert check.evidence["terms"]["reasoning_materialized"]["count"] == 5
    assert check.evidence["blocking_count"] == 0


# ---------------------------------------------------------------------------
# Negative twins
# ---------------------------------------------------------------------------


def test_missing_origin_cannot_hide_behind_the_other(tmp_path: Path) -> None:
    """Dropping Codex entirely leaves Claude's witnesses conserved.

    A check reporting one aggregate would still be green: every witness it
    could see still materializes. The per-origin denominator is what makes
    the absence visible.
    """
    _seed_mixed_cohort(tmp_path)
    _index(tmp_path, "DELETE FROM sessions WHERE origin = ?", (ORIGIN_CODEX,))

    check = _run(tmp_path)

    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["terms"]["origin_evidence_absent"]["breakdown"] == {ORIGIN_CODEX: 1}
    assert check.evidence["origins"][ORIGIN_CLAUDE_CODE]["outcomes_by_term"]["reasoning_materialized"] == 3


def test_collapsed_block_kind_trips_reasoning_conservation(tmp_path: Path) -> None:
    """Thinking material retyped as ordinary text is not conservation.

    The words survive and every count matches; only the kind is wrong, which
    is exactly what makes reasoning unreadable as reasoning on every route.
    """
    _seed_mixed_cohort(tmp_path)
    _index(
        tmp_path, "UPDATE blocks SET block_type = 'text' WHERE block_type = 'thinking' AND text = ?", (THINKING_TEXT,)
    )

    check = _run(tmp_path)

    assert check.status is OutcomeStatus.ERROR
    collapsed = check.evidence["terms"]["reasoning_kind_collapsed"]
    assert collapsed["count"] == 1
    assert collapsed["breakdown"] == {f"{ORIGIN_CLAUDE_CODE}:{VARIANT_TEXT}": 1}


def test_signature_only_thinking_lost_trips_reasoning_conservation(tmp_path: Path) -> None:
    """polylogue-vf9x's exact shape: reasoning happened, its text never shipped.

    Losing the signature loses the only evidence the model reasoned here, and
    a count of thinking blocks cannot see it -- the block is still there.
    """
    _seed_mixed_cohort(tmp_path)
    _index(tmp_path, "UPDATE blocks SET signature = NULL WHERE signature = ?", (SIGNATURE_ONLY_VALUE,))

    check = _run(tmp_path)

    assert check.status is OutcomeStatus.ERROR
    lost = check.evidence["terms"]["reasoning_material_lost"]
    assert lost["count"] == 1
    assert lost["breakdown"] == {f"{ORIGIN_CLAUDE_CODE}:{VARIANT_SIGNATURE_ONLY}": 1}


def test_offsetting_losses_do_not_pass_as_an_aggregate(tmp_path: Path) -> None:
    """Codex reasoning deleted, Claude thinking duplicated: the total is right.

    The archive-wide thinking-block count is unchanged, so any check comparing
    one grand total with another passes. Per-witness tracing does not.
    """
    _seed_mixed_cohort(tmp_path)
    _index(
        tmp_path,
        "DELETE FROM blocks WHERE block_type = 'thinking' AND session_id LIKE ?",
        (f"{ORIGIN_CODEX}:%",),
    )
    _index(
        tmp_path,
        """
        INSERT INTO blocks(message_id, session_id, position, block_type, text)
        SELECT message_id, session_id, position + 100, 'thinking', text
        FROM blocks WHERE block_type = 'thinking' AND session_id LIKE ?
        ORDER BY message_id LIMIT 2
        """,
        (f"{ORIGIN_CLAUDE_CODE}:%",),
    )

    conn = _connect(tmp_path / "index.db")
    try:
        total = conn.execute("SELECT COUNT(*) FROM blocks WHERE block_type = 'thinking'").fetchone()[0]
    finally:
        conn.close()
    assert total == 5

    check = _run(tmp_path)

    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["origins"][ORIGIN_CODEX]["outcomes_by_term"]["reasoning_material_lost"] == 2
    assert check.evidence["origins"][ORIGIN_CODEX]["outcomes_by_term"].get("reasoning_materialized", 0) == 0
    assert check.evidence["origins"][ORIGIN_CLAUDE_CODE]["outcomes_by_term"]["reasoning_materialized"] == 3


def test_unreadable_evidence_is_named_not_counted_as_conserved(tmp_path: Path) -> None:
    """A pruned blob makes its witnesses unknown, never conserved."""
    _seed_mixed_cohort(tmp_path)
    conn = _connect(tmp_path / "source.db")
    try:
        row = conn.execute("SELECT lower(hex(blob_hash)) FROM raw_sessions WHERE raw_id = 'raw-codex'").fetchone()
    finally:
        conn.close()
    BlobStore(tmp_path / "blob").blob_path(str(row[0])).unlink()

    check = _run(tmp_path)

    assert check.evidence["terms"]["reasoning_evidence_unreadable"]["breakdown"] == {ORIGIN_CODEX: 1}
    assert check.evidence["origins"][ORIGIN_CODEX]["witnesses"] == 0

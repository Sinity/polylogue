-- Rebuild ``assertions`` so ``status`` is NOT NULL.
--
-- ``status TEXT DEFAULT 'active'`` was nullable. No production writer could
-- ever put a NULL there: both write sites route through
-- ``_normalize_assertion_status`` (``user_write.py:upsert_assertion`` and
-- ``user_write.py:mark_assertion_status``), which resolves an absent status to
-- ``ASSERTION_DEFAULT_STATUS`` and rejects anything outside ``AssertionStatus``
-- before it reaches SQL. The nullability was therefore unreachable by the
-- writer and load-bearing only for readers, which compensated for it at
-- sixteen SQL sites across eight modules -- ``user_write.py`` (1115, 1814,
-- 2008, 2582, 2677, 2742), ``archive_query_reads.py:1733``,
-- ``archive.py:7780``, ``user_overlay.py:46``, ``user_audit.py`` (121, 123),
-- ``session_suppression.py:220``, ``storage/derived/feedback/__init__.py``
-- (175, 240, 260) and ``operations/user_state_resolution.py:57`` -- plus the
-- Python-side default at ``archive_query_reads.py:4514``. That is the
-- symptom, not the cause.
--
-- user.db is durable and irreplaceable: it is never rebuilt from source
-- evidence, so fresh DDL alone would leave every existing archive nullable.
-- SQLite cannot add NOT NULL to an existing column, so this is the ordinary
-- 12-step table rebuild and is NOT ``additive-no-backup`` -- it runs behind the
-- verified backup manifest gate.
--
-- Two row dispositions, both deliberate and neither silent:
--
--   * ``status IS NULL`` becomes ``'active'``. This is not an adjudication of
--     an unknown value; it materializes the default the column already
--     declares and that every reader above already applies, so no reader
--     observes a different answer before and after. Collapsing the two
--     spellings of "active" is the whole point of the column being NOT NULL.
--   * A non-null status outside ``AssertionStatus`` is carried forward
--     verbatim. The rebuilt table deliberately carries no membership CHECK:
--     durable-tier vocabulary is validated at the write boundary, never pinned
--     into durable DDL (docs/internals.md, ``devtools gate
--     durable-enum-checks``). An unsupported legacy value therefore survives
--     for adjudication instead of being coerced or dropped.
--
-- Row counts are unchanged, so the train declares no row-change allowance.
-- The four indexes and three epoch triggers are dropped and recreated verbatim
-- from canonical DDL; the rebuild itself must not bump the query-unit frame
-- epoch, which is why the triggers are absent while the rows are copied.

DROP TRIGGER IF EXISTS query_unit_frame_assertions_insert;
DROP TRIGGER IF EXISTS query_unit_frame_assertions_update;
DROP TRIGGER IF EXISTS query_unit_frame_assertions_delete;

DROP INDEX IF EXISTS idx_assertions_target_kind;
DROP INDEX IF EXISTS idx_assertions_kind_status_updated;
DROP INDEX IF EXISTS idx_assertions_target_kind_status_visibility;
DROP INDEX IF EXISTS idx_assertions_scope_kind_status;

CREATE TABLE IF NOT EXISTS assertions_status_not_null (
    assertion_id        TEXT PRIMARY KEY,
    scope_ref           TEXT,
    target_ref          TEXT NOT NULL,
    key                 TEXT,
    kind                TEXT NOT NULL,
    value_json          TEXT,
    body_text           TEXT,
    author_ref          TEXT DEFAULT 'user:local',
    author_kind         TEXT DEFAULT 'user',
    evidence_refs_json  TEXT DEFAULT '[]',
    status              TEXT NOT NULL DEFAULT 'active',
    visibility          TEXT DEFAULT 'private',
    confidence          REAL,
    staleness_json      TEXT,
    context_policy_json TEXT DEFAULT '{"inject":false}',
    supersedes_json     TEXT DEFAULT '[]',
    created_at_ms       INTEGER NOT NULL,
    updated_at_ms       INTEGER NOT NULL
) STRICT;

INSERT INTO assertions_status_not_null (
    assertion_id, scope_ref, target_ref, key, kind, value_json, body_text,
    author_ref, author_kind, evidence_refs_json, status, visibility,
    confidence, staleness_json, context_policy_json, supersedes_json,
    created_at_ms, updated_at_ms
)
SELECT
    assertion_id, scope_ref, target_ref, key, kind, value_json, body_text,
    author_ref, author_kind, evidence_refs_json,
    COALESCE(status, 'active'),
    visibility, confidence, staleness_json, context_policy_json,
    supersedes_json, created_at_ms, updated_at_ms
FROM assertions;

DROP TABLE assertions;

ALTER TABLE assertions_status_not_null RENAME TO assertions;

CREATE INDEX IF NOT EXISTS idx_assertions_target_kind
ON assertions(target_ref, kind);

CREATE INDEX IF NOT EXISTS idx_assertions_kind_status_updated
ON assertions(kind, status, updated_at_ms);

CREATE INDEX IF NOT EXISTS idx_assertions_target_kind_status_visibility
ON assertions(target_ref, kind, status, visibility);

CREATE INDEX IF NOT EXISTS idx_assertions_scope_kind_status
ON assertions(scope_ref, kind, status);

CREATE TRIGGER IF NOT EXISTS query_unit_frame_assertions_insert
AFTER INSERT ON assertions BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_assertions_update
AFTER UPDATE ON assertions BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_assertions_delete
AFTER DELETE ON assertions BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;

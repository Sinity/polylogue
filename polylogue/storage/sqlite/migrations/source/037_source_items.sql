-- migration-safety: additive-no-backup
-- Durable acquisition authority.  The current row is the source-generation
-- fact; retry history and worker scheduling remain in ops.db.
CREATE TABLE IF NOT EXISTS source_generations (
    source_generation_id TEXT PRIMARY KEY,
    manifest_digest TEXT NOT NULL CHECK(length(manifest_digest) = 64),
    addressing_mode TEXT NOT NULL CHECK(length(trim(addressing_mode)) > 0),
    item_count INTEGER NOT NULL CHECK(item_count >= 0),
    sealed_at_ms INTEGER,
    created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;

CREATE TABLE IF NOT EXISTS source_items (
    source_generation_id TEXT NOT NULL REFERENCES source_generations(source_generation_id) ON DELETE CASCADE,
    source_item_id       TEXT NOT NULL,
    logical_coordinate   TEXT NOT NULL CHECK(length(trim(logical_coordinate)) > 0),
    addressing_mode      TEXT NOT NULL CHECK(length(trim(addressing_mode)) > 0),
    origin               TEXT CHECK (origin IN ('claude-code-session', 'codex-session', 'gemini-cli-session', 'hermes-session', 'antigravity-session', 'beads-issue', 'grok-export', 'chatgpt-export', 'claude-ai-export', 'claude-design-session', 'aistudio-drive', 'unknown-export') OR origin IS NULL),
    source_path          TEXT,
    source_index         INTEGER CHECK(source_index >= 0),
    disposition          TEXT NOT NULL CHECK(disposition IN (
        'pending', 'admitted', 'non_session', 'empty', 'unsupported', 'corrupt', 'unknown_blocking'
    )),
    outcome_code         TEXT NOT NULL CHECK(outcome_code IN (
        'success', 'validation_rejected', 'unsupported_shape', 'corrupt_input',
        'transient_error', 'parser_defect', 'downstream_failure', 'canceled',
        'interrupted', 'legacy_unknown'
    )),
    stage                TEXT NOT NULL CHECK(length(trim(stage)) > 0),
    retryable            INTEGER CHECK(retryable IN (0, 1)),
    diagnostic           TEXT CHECK(diagnostic IS NULL OR length(diagnostic) <= 4096),
    evidence_ref         TEXT,
    content_fingerprint  TEXT,
    source_fingerprint   TEXT,
    parser_fingerprint   TEXT,
    policy_fingerprint   TEXT,
    raw_id               TEXT REFERENCES raw_sessions(raw_id) ON DELETE SET NULL,
    blob_hash            BLOB CHECK(blob_hash IS NULL OR length(blob_hash) = 32),
    revision             INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
    request_id           TEXT,
    observed_at_ms       INTEGER NOT NULL CHECK(observed_at_ms >= 0),
    updated_at_ms        INTEGER NOT NULL CHECK(updated_at_ms >= 0),
    PRIMARY KEY(source_generation_id, source_item_id),
    UNIQUE(source_generation_id, logical_coordinate, addressing_mode)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_source_items_disposition ON source_items(source_generation_id, disposition);
CREATE INDEX IF NOT EXISTS idx_source_items_raw_id ON source_items(raw_id);

CREATE VIEW IF NOT EXISTS source_item_reconciliation AS
WITH item_counts AS (
    SELECT source_generation_id,
           COUNT(*) AS manifested,
           SUM(disposition = 'pending') AS pending,
           SUM(disposition = 'admitted') AS admitted,
           SUM(disposition IN ('non_session','empty','unsupported','corrupt')) AS deliberate,
           SUM(disposition = 'unknown_blocking') AS unknown_blocking,
           SUM(raw_id IS NULL AND disposition = 'admitted') AS admitted_without_raw,
           COUNT(DISTINCT source_item_id) AS distinct_items
      FROM source_items GROUP BY source_generation_id
), raw_counts AS (
    SELECT si.source_generation_id,
           COUNT(si.raw_id) AS linked_raw,
           COUNT(DISTINCT si.raw_id) AS distinct_raw
      FROM source_items si WHERE si.raw_id IS NOT NULL
     GROUP BY si.source_generation_id
)
SELECT g.source_generation_id, g.item_count AS manifest_items,
       COALESCE(i.manifested, 0) AS manifested,
       COALESCE(i.pending, 0) AS pending,
       COALESCE(i.admitted, 0) AS admitted,
       COALESCE(i.deliberate, 0) AS deliberate,
       COALESCE(i.unknown_blocking, 0) AS unknown_blocking,
       COALESCE(i.admitted_without_raw, 0) AS admitted_without_raw,
       COALESCE(i.distinct_items, 0) AS distinct_items,
       COALESCE(r.linked_raw, 0) AS linked_raw,
       COALESCE(r.distinct_raw, 0) AS distinct_raw,
       (g.item_count - COALESCE(i.manifested, 0)) AS missing,
       (COALESCE(i.manifested, 0) - COALESCE(i.distinct_items, 0)) AS duplicate,
       (g.item_count = COALESCE(i.manifested, 0)
        AND COALESCE(i.manifested, 0) = COALESCE(i.distinct_items, 0)
        AND COALESCE(i.pending, 0) = 0
        AND COALESCE(i.unknown_blocking, 0) = 0
        AND COALESCE(i.admitted_without_raw, 0) = 0) AS sealable
  FROM source_generations g
  LEFT JOIN item_counts i USING(source_generation_id)
  LEFT JOIN raw_counts r USING(source_generation_id);

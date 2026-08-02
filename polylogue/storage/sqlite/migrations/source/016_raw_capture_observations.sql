-- polylogue-buns: durable multimap of every acquisition mode ever observed
-- for one raw_id. raw_sessions.capture_mode (v8) caches only the FIRST known
-- mode -- write_source_raw_session/write_source_raw_session_blob_ref/
-- save_raw_session all UPDATE it only `WHERE capture_mode IS NULL`, so any
-- later, different capture_mode observed for the same raw_id was silently
-- dropped. Because raw_id is content-derived, a GEMINI export and a live
-- DRIVE acquisition of byte-identical content collide on the same raw_id
-- even though their acquisition mechanism genuinely differs -- that
-- collision is exactly the case this table exists to stop losing.
--
-- One row per distinct (raw_id, capture_mode) ever observed. The existing
-- raw_sessions.capture_mode column is unchanged and keeps its first-known
-- convenience-cache behavior; this table is the durable source of truth a
-- caller reads to tell "only one mode was ever observed" (unambiguous) apart
-- from "more than one was, and the cache only remembers the first"
-- (ambiguous). See read_capture_mode_resolution (source_write.py) and
-- get_capture_mode_resolution (queries/raw_reads.py).
CREATE TABLE IF NOT EXISTS raw_capture_observations (
    raw_id               TEXT NOT NULL REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    capture_mode         TEXT NOT NULL CHECK(capture_mode IN (
                             'chatgpt', 'claude-ai', 'claude-design', 'claude-code', 'codex',
                             'gemini', 'gemini-cli', 'hermes', 'antigravity', 'beads', 'grok',
                             'drive', 'unknown'
                         )),
    first_observed_at_ms INTEGER NOT NULL CHECK(first_observed_at_ms >= 0),
    PRIMARY KEY (raw_id, capture_mode)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_capture_observations_raw_id
ON raw_capture_observations(raw_id);

-- Backfill: every pre-existing raw_sessions row with a known capture_mode
-- observed that mode at least once, at its acquisition time -- the earliest
-- evidence this migration can reconstruct for rows written before this table
-- existed. This does not, and cannot, recover a second mode a pre-migration
-- write already discarded; it only carries forward what the cache still has.
INSERT OR IGNORE INTO raw_capture_observations (raw_id, capture_mode, first_observed_at_ms)
SELECT raw_id, capture_mode, acquired_at_ms
FROM raw_sessions
WHERE capture_mode IS NOT NULL;

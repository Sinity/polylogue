-- polylogue-w6hql (Phase 2) / polylogue-tw4ar: persisted cache table for the
-- closed RawAuthorityVerdict vocabulary. Not a new source of truth -- every
-- row is a read-through cache of what
-- polylogue.storage.raw_authority_verdict_projection.project_raw_authority_verdicts
-- would recompute from raw_sessions + blob storage; it exists only so a
-- repeated read (e.g. blob-GC invariant checks at scale) doesn't re-run the
-- full byte-proof classifier every time.
--
-- Invalidated by content, not by elapsed time: cohort_fingerprint is a
-- SHA-256 over the cohort's own (raw_id, revision_kind, blob_hash) rows
-- (polylogue.storage.raw_authority_verdict_cache._cohort_fingerprint), so any
-- membership or content change to the logical_source_key cohort changes the
-- fingerprint and the previously-cached rows read as stale on the next
-- lookup rather than being trusted past their true validity window.
--
-- The daemon's bounded raw_authority_verdict_cache convergence stage also
-- warms eligible full/unknown cohorts ahead of reads. The read-through
-- get_or_compute_raw_authority_verdicts() path remains the fallback for a
-- cold or invalidated cohort. Cohorts containing append revisions are
-- explicitly skipped because their authority has a separate proof shape;
-- absence from this cache is not an assertion that the raw evidence is absent.
-- This cache never controls raw retention or index rebuild selection; those
-- decisions continue to use direct source-tier evidence and their own proofs.
CREATE TABLE IF NOT EXISTS raw_authority_verdicts (
    raw_id                TEXT PRIMARY KEY REFERENCES raw_sessions(raw_id) ON DELETE CASCADE,
    logical_source_key    TEXT NOT NULL,
    verdict                TEXT NOT NULL CHECK(verdict IN ('verified', 'superseded', 'sole-copy', 'diverged', 'unchecked')),
    cohort_member_count   INTEGER NOT NULL CHECK(cohort_member_count >= 1),
    cohort_fingerprint    BLOB NOT NULL CHECK(length(cohort_fingerprint) = 32),
    computed_at_ms        INTEGER NOT NULL CHECK(computed_at_ms >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_authority_verdicts_logical_source
ON raw_authority_verdicts(logical_source_key);

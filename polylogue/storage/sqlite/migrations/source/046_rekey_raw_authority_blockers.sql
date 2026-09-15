-- migration-safety: additive-backup-required
-- polylogue-5dzj9, executing the AMENDED 2026-09-15 ruling recorded on
-- polylogue-6kur: raw_authority_blockers is durable frontier AUTHORIZATION and
-- survives the fresh start, but it currently foreign-keys into two tables the
-- same ruling retires (raw_authority_plans, raw_authority_censuses).  Those
-- three facts are mutually unsatisfiable, so the blocker is re-keyed first --
-- step M1/M2 of docs/design/raw-decision-authority.md, which already recorded
-- this table as "RETAIN, re-keyed" and gated the census DROP on it.
--
-- Nothing is lost, and nothing is reconstructed from a rebuildable tier:
--
--   * plan_id -> plan_input_digest.  plan_id is a surrogate over the plan's
--     own content address (build_raw_replay_plan mints
--     f"raw-replay:{input_digest}"); the digest is carried here directly from
--     raw_authority_plans.input_digest, which the NOT NULL foreign key
--     guarantees exists for every pre-v46 row.  expected_json -- the full
--     RawReplayPlan.to_dict() snapshot every blocker writer records -- is the
--     fallback and the authority every reader now uses for the plan itself.
--   * census_id -> observed_pass_id, the same value with the foreign key
--     removed.  It is a breadcrumb naming the pass that first observed the
--     blocked state, not authority: no resolution path reads it.
--
-- SQLite cannot drop a column's REFERENCES clause in place, so this is a table
-- rebuild in the 041 house style.  The open-blocker uniqueness invariant (at
-- most one unresolved blocker per plan) is preserved, re-expressed on the
-- digest.
CREATE TABLE raw_authority_blockers__046 (
    blocker_id          TEXT PRIMARY KEY,
    plan_input_digest   TEXT NOT NULL CHECK(length(plan_input_digest) = 64),
    observed_pass_id    TEXT,
    reason              TEXT NOT NULL,
    expected_json       TEXT NOT NULL CHECK(json_valid(expected_json)),
    observed_json       TEXT NOT NULL CHECK(json_valid(observed_json)),
    created_at_ms       INTEGER NOT NULL CHECK(created_at_ms >= 0),
    resolved_at_ms      INTEGER CHECK(resolved_at_ms IS NULL OR resolved_at_ms >= created_at_ms),
    resolution          TEXT,
    CHECK((resolved_at_ms IS NULL) = (resolution IS NULL))
) STRICT;

INSERT INTO raw_authority_blockers__046 (
    blocker_id, plan_input_digest, observed_pass_id, reason, expected_json,
    observed_json, created_at_ms, resolved_at_ms, resolution
)
SELECT b.blocker_id,
       COALESCE(p.input_digest, json_extract(b.expected_json, '$.input_digest')),
       b.census_id,
       b.reason,
       b.expected_json,
       b.observed_json,
       b.created_at_ms,
       b.resolved_at_ms,
       b.resolution
FROM raw_authority_blockers AS b
LEFT JOIN raw_authority_plans AS p ON p.plan_id = b.plan_id;

DROP INDEX IF EXISTS idx_raw_authority_blockers_open_plan;
DROP TABLE raw_authority_blockers;
ALTER TABLE raw_authority_blockers__046 RENAME TO raw_authority_blockers;

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_authority_blockers_open_plan
ON raw_authority_blockers(plan_input_digest)
WHERE resolved_at_ms IS NULL;

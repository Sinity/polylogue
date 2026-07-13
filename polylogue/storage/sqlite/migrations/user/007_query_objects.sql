-- Durable analysis-provenance substrate. Existing saved-query assertions are
-- repointed by migration_runner after this additive DDL has committed inside
-- the same verified-backup transaction.
CREATE TABLE IF NOT EXISTS queries (
    query_hash          TEXT PRIMARY KEY NOT NULL CHECK(length(query_hash) = 64 AND query_hash NOT GLOB '*[^0-9a-f]*'),
    canonical_plan_json TEXT NOT NULL CHECK(json_valid(canonical_plan_json) AND json_type(canonical_plan_json) = 'object'),
    grain               TEXT NOT NULL CHECK(length(trim(grain)) > 0),
    lane                TEXT NOT NULL CHECK(length(trim(lane)) > 0),
    rank_policy         TEXT NOT NULL CHECK(length(trim(rank_policy)) > 0),
    created_at_ms       INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;
CREATE TABLE IF NOT EXISTS query_names (
    name                  TEXT PRIMARY KEY NOT NULL CHECK(length(trim(name)) > 0),
    query_hash            TEXT NOT NULL REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    supersedes_query_hash TEXT REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    updated_at_ms         INTEGER NOT NULL CHECK(updated_at_ms >= 0)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_query_names_query_hash ON query_names(query_hash, updated_at_ms DESC);
CREATE TABLE IF NOT EXISTS result_sets (
    result_set_id          TEXT PRIMARY KEY NOT NULL CHECK(length(trim(result_set_id)) > 0),
    query_hash             TEXT NOT NULL REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    grain                  TEXT NOT NULL CHECK(length(trim(grain)) > 0),
    corpus_epoch           TEXT NOT NULL CHECK(length(trim(corpus_epoch)) > 0),
    member_count           INTEGER NOT NULL CHECK(member_count >= 0),
    membership_merkle_root TEXT NOT NULL CHECK(length(membership_merkle_root) = 64 AND membership_merkle_root NOT GLOB '*[^0-9a-f]*'),
    ordered_rank_hash      TEXT NOT NULL CHECK(length(ordered_rank_hash) = 64 AND ordered_rank_hash NOT GLOB '*[^0-9a-f]*'),
    exactness              TEXT NOT NULL CHECK(exactness IN ('exact', 'capped', 'sampled', 'estimate')),
    persistence_class      TEXT NOT NULL CHECK(persistence_class IN ('routine', 'watch', 'pinned', 'finding', 'cohort')),
    created_at_ms          INTEGER NOT NULL CHECK(created_at_ms >= 0)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_result_sets_query_epoch ON result_sets(query_hash, corpus_epoch, created_at_ms DESC);
CREATE TABLE IF NOT EXISTS result_set_members (
    result_set_id TEXT NOT NULL REFERENCES result_sets(result_set_id) ON DELETE CASCADE,
    rank          INTEGER NOT NULL CHECK(rank >= 0),
    member_ref    TEXT NOT NULL CHECK(length(trim(member_ref)) > 0),
    PRIMARY KEY (result_set_id, rank),
    UNIQUE (result_set_id, member_ref)
) STRICT;
CREATE TABLE IF NOT EXISTS query_edges (
    src_query_hash TEXT NOT NULL REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    dst_query_hash TEXT NOT NULL REFERENCES queries(query_hash) ON UPDATE RESTRICT ON DELETE RESTRICT,
    edge_kind      TEXT NOT NULL CHECK(edge_kind IN ('operand-of', 'refines', 'supersedes', 'derived-from', 'same-as')),
    created_at_ms  INTEGER NOT NULL CHECK(created_at_ms >= 0),
    PRIMARY KEY (src_query_hash, dst_query_hash, edge_kind)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_query_edges_dst_kind ON query_edges(dst_query_hash, edge_kind, created_at_ms DESC);

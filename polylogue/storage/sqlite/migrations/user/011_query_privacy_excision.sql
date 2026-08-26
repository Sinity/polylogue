-- migration-safety: additive-backup-required
-- Promotion policy is explicit and the tombstone ledger is durable authority.
ALTER TABLE queries ADD COLUMN privacy_class TEXT;
ALTER TABLE queries ADD COLUMN retention_policy_json TEXT;
ALTER TABLE queries ADD COLUMN excision_link TEXT;
ALTER TABLE queries ADD COLUMN promoted_at_ms INTEGER;

ALTER TABLE result_sets ADD COLUMN privacy_class TEXT;
ALTER TABLE result_sets ADD COLUMN retention_policy_json TEXT;
ALTER TABLE result_sets ADD COLUMN excision_link TEXT;
ALTER TABLE result_sets ADD COLUMN promoted_at_ms INTEGER;

CREATE TABLE query_excision_ledger (
    ledger_id           TEXT PRIMARY KEY NOT NULL,
    query_hash          TEXT,
    result_set_id       TEXT,
    excision_link       TEXT NOT NULL CHECK(length(trim(excision_link)) > 0),
    reason_digest       TEXT NOT NULL CHECK(length(reason_digest) = 64 AND reason_digest NOT GLOB '*[^0-9a-f]*'),
    actor_ref           TEXT NOT NULL CHECK(length(trim(actor_ref)) > 0),
    prior_revision      INTEGER NOT NULL CHECK(prior_revision >= 0),
    excised_at_ms       INTEGER NOT NULL CHECK(excised_at_ms >= 0),
    CHECK ((query_hash IS NOT NULL AND result_set_id IS NULL) OR (query_hash IS NULL AND result_set_id IS NOT NULL)),
    UNIQUE(query_hash),
    UNIQUE(result_set_id)
) STRICT;
CREATE INDEX idx_query_excision_ledger_link ON query_excision_ledger(excision_link);

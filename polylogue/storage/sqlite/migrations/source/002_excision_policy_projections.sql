-- migration-safety: additive-no-backup
-- Register excision_policy_projections in the durable source tier.
--
-- The table existed before this slot, but only in archives an ordinary
-- manifest write happened to reach: publish_source_generation created it at
-- write time whenever a policy snapshot was supplied, and the reader probed
-- sqlite_schema to find out whether the shape was there at all. Canonical DDL
-- now owns it, so a source tier this runtime admits always has the table and
-- an absent row means "this generation recorded no policy binding".
--
-- Additive and idempotent in both directions an existing v1 archive can be
-- in: IF NOT EXISTS adopts the byte-identical write-path table where one was
-- created, and creates it where no policy-bearing generation was ever
-- published. No row is written, read, or moved.
CREATE TABLE IF NOT EXISTS excision_policy_projections (
    source_generation_id TEXT PRIMARY KEY REFERENCES source_generations(source_generation_id) ON DELETE CASCADE,
    policy_digest TEXT NOT NULL CHECK(length(policy_digest) = 64),
    user_generation INTEGER NOT NULL CHECK(user_generation >= 0),
    audit_generation INTEGER NOT NULL CHECK(audit_generation >= 0),
    audit_head TEXT NOT NULL CHECK(length(audit_head) = 64),
    assertion_refs_json TEXT NOT NULL DEFAULT '[]',
    generated_at_ms INTEGER NOT NULL CHECK(generated_at_ms >= 0)
) STRICT;

-- migration-safety: additive-backup-required
-- Retained ZIP enumeration records the physical central-directory denominator
-- separately from source-43 decoded raw records.  Existing raw membership rows
-- remain the authoritative admitted-record ledger.
ALTER TABLE source_items ADD COLUMN enumerated_member_count INTEGER
    CHECK(enumerated_member_count IS NULL OR enumerated_member_count >= 0);
ALTER TABLE source_items ADD COLUMN enumeration_member_digest TEXT
    CHECK(enumeration_member_digest IS NULL OR length(enumeration_member_digest) = 64);

CREATE TABLE source_item_member_dispositions (
    source_generation_id TEXT NOT NULL,
    source_item_id TEXT NOT NULL,
    entry_ordinal INTEGER NOT NULL CHECK(entry_ordinal >= 0),
    member_name TEXT NOT NULL CHECK(length(trim(member_name)) > 0),
    disposition TEXT NOT NULL CHECK(disposition IN ('admitted', 'refused', 'unselected')),
    diagnostic TEXT NOT NULL DEFAULT '' CHECK(length(diagnostic) <= 4096),
    observed_at_ms INTEGER NOT NULL CHECK(observed_at_ms >= 0),
    PRIMARY KEY(source_generation_id, source_item_id, entry_ordinal),
    FOREIGN KEY(source_generation_id, source_item_id)
        REFERENCES source_items(source_generation_id, source_item_id) ON DELETE CASCADE
) STRICT;
CREATE INDEX idx_source_item_member_dispositions_item
ON source_item_member_dispositions(source_generation_id, source_item_id);

DROP VIEW source_item_reconciliation;
CREATE VIEW source_item_reconciliation AS
WITH member_counts AS (
    SELECT source_generation_id, source_item_id,
           COUNT(*) AS records, SUM(raw_id IS NOT NULL) AS present
      FROM source_item_raw_members GROUP BY source_generation_id, source_item_id
), item_counts AS (
    SELECT si.source_generation_id,
           COUNT(*) AS manifested,
           SUM(disposition = 'pending') AS pending,
           SUM(disposition = 'admitted') AS admitted,
           SUM(disposition IN ('non_session','empty','unsupported','corrupt')) AS deliberate,
           SUM(disposition = 'unknown_blocking') AS unknown_blocking,
           SUM(si.raw_id IS NULL AND COALESCE(mc.present, 0) = 0 AND disposition = 'admitted') AS admitted_without_raw,
           SUM(si.enumeration_fingerprint IS NOT NULL AND (
               si.enumerated_record_count IS NULL OR si.enumeration_digest IS NULL OR si.enumerated_at_ms IS NULL
               OR si.enumerated_record_count != COALESCE(mc.records, 0)
               OR si.enumerated_member_count IS NULL OR si.enumeration_member_digest IS NULL
           )) AS enumeration_pending,
           SUM(COALESCE(mc.records, 0) - COALESCE(mc.present, 0)) AS retired_raw_members,
           SUM(COALESCE(md.refused, 0)) AS member_refused,
           SUM(COALESCE(md.unselected, 0)) AS member_unselected,
           SUM(COALESCE(md.total, 0)) AS member_dispositions,
           COUNT(DISTINCT si.source_item_id) AS distinct_items
      FROM source_items si LEFT JOIN member_counts mc
        ON mc.source_generation_id = si.source_generation_id AND mc.source_item_id = si.source_item_id
      LEFT JOIN (
        SELECT source_generation_id, source_item_id,
               SUM(disposition = 'refused') AS refused,
               SUM(disposition = 'unselected') AS unselected,
               SUM(disposition IN ('refused','unselected')) AS total
          FROM source_item_member_dispositions
         GROUP BY source_generation_id, source_item_id
      ) md ON md.source_generation_id = si.source_generation_id AND md.source_item_id = si.source_item_id
     GROUP BY si.source_generation_id
), raw_links AS (
    SELECT source_generation_id, raw_id FROM source_item_raw_members WHERE raw_id IS NOT NULL
    UNION ALL
    SELECT si.source_generation_id, si.raw_id FROM source_items si
     WHERE si.raw_id IS NOT NULL AND NOT EXISTS (
         SELECT 1 FROM source_item_raw_members m WHERE m.source_generation_id = si.source_generation_id
           AND m.source_item_id = si.source_item_id
     )
), raw_counts AS (
    SELECT source_generation_id, COUNT(*) AS linked_raw, COUNT(DISTINCT raw_id) AS distinct_raw
      FROM raw_links GROUP BY source_generation_id
)
SELECT g.source_generation_id, g.item_count AS manifest_items,
       COALESCE(i.manifested, 0) AS manifested,
       COALESCE(i.pending, 0) AS pending,
       COALESCE(i.admitted, 0) AS admitted,
       COALESCE(i.deliberate, 0) AS deliberate,
       COALESCE(i.unknown_blocking, 0) AS unknown_blocking,
       COALESCE(i.admitted_without_raw, 0) AS admitted_without_raw,
       COALESCE(i.enumeration_pending, 0) AS enumeration_pending,
       COALESCE(i.retired_raw_members, 0) AS retired_raw_members,
       COALESCE(i.member_refused, 0) AS member_refused,
       COALESCE(i.member_unselected, 0) AS member_unselected,
       COALESCE(i.member_dispositions, 0) AS member_dispositions,
       COALESCE(i.distinct_items, 0) AS distinct_items,
       COALESCE(r.linked_raw, 0) AS linked_raw,
       COALESCE(r.distinct_raw, 0) AS distinct_raw,
       (g.item_count - COALESCE(i.manifested, 0)) AS missing,
       (COALESCE(i.manifested, 0) - COALESCE(i.distinct_items, 0)) AS duplicate,
       (g.item_count = COALESCE(i.manifested, 0)
        AND COALESCE(i.manifested, 0) = COALESCE(i.distinct_items, 0)
        AND COALESCE(i.pending, 0) = 0
        AND COALESCE(i.unknown_blocking, 0) = 0
        AND COALESCE(i.enumeration_pending, 0) = 0
        AND COALESCE(i.member_dispositions, 0) = 0
        AND COALESCE(i.retired_raw_members, 0) = 0
        AND COALESCE(i.admitted_without_raw, 0) = 0) AS sealable
  FROM source_generations g
  LEFT JOIN item_counts i USING(source_generation_id)
  LEFT JOIN raw_counts r USING(source_generation_id);

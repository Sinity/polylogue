-- migration-safety: additive-no-backup
-- A pending physical deletion may resume only in the namespace it observed.
ALTER TABLE gc_generations ADD COLUMN blob_namespace_marker TEXT;

-- v33 could not create exact member intents. A pre-v35 incomplete row with
-- no members has no physical deletion to recover, so terminalize it without
-- inventing a deletion outcome.
UPDATE gc_generations
SET completed_at_ms = started_at_ms
WHERE completed_at_ms IS NULL
  AND NOT EXISTS (
      SELECT 1 FROM gc_generation_members AS member
      WHERE member.generation_id = gc_generations.generation_id
  );

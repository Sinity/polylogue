-- Lifecycle invalidation uses the existing committed fact positions.
CREATE INDEX IF NOT EXISTS primary_fact_changed ON primary_fact(kind,changed)
WHERE changed!=admitted;

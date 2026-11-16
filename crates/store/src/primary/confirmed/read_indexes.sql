-- Lifecycle invalidation uses the existing committed event positions.
CREATE INDEX IF NOT EXISTS confirmed_event_changed ON confirmed_event(kind,changed)
WHERE changed!=committed;

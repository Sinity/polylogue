CREATE INDEX IF NOT EXISTS saved_rows_name
  ON saved_rows(name, ran_at, id) WHERE name IS NOT NULL;

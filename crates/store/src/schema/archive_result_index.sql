CREATE INDEX IF NOT EXISTS kept_result_name
  ON kept_result(name, ran_at, id) WHERE name IS NOT NULL;

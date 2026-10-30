ALTER TABLE @source.observation ADD COLUMN path_bytes BLOB;
CREATE INDEX @source.observation_path_bytes ON @source.observation(machine,path_bytes,commit_seq);
ALTER TABLE @source.extent ADD COLUMN acquisition BLOB;

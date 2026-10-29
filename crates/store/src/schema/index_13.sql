ALTER TABLE primary_pending ADD COLUMN native_context BLOB;
ALTER TABLE primary_fact ADD COLUMN native_context BLOB;
ALTER TABLE primary_complete ADD COLUMN evidence BLOB;

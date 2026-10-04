-- This replacement requires authenticated backup for populated Source tiers.
CREATE TABLE blob_refs_coordinate_identity (
    blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
    ref_id TEXT NOT NULL,
    ref_type TEXT NOT NULL,
    source_path TEXT,
    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
    acquired_at_ms INTEGER NOT NULL
) STRICT;
INSERT INTO blob_refs_coordinate_identity(rowid, blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
SELECT rowid, blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms FROM blob_refs;
DROP TABLE blob_refs;
ALTER TABLE blob_refs_coordinate_identity RENAME TO blob_refs;
CREATE INDEX idx_blob_refs_ref_id ON blob_refs(ref_id);
CREATE UNIQUE INDEX idx_blob_refs_owner_identity
ON blob_refs(blob_hash, ref_type, ref_id) WHERE ref_type != 'attachment';
CREATE UNIQUE INDEX idx_blob_refs_attachment_identity
ON blob_refs(blob_hash, ref_type, ref_id, coalesce(source_path, '')) WHERE ref_type = 'attachment';

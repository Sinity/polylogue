-- migration-safety: additive-no-backup
CREATE TABLE IF NOT EXISTS sinex_publication_originals (
    object_id TEXT NOT NULL,
    protocol_version TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    manifest_digest TEXT NOT NULL,
    position INTEGER NOT NULL CHECK(position >= 0),
    role TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
    media_type TEXT NOT NULL,
    normalizer_version TEXT NOT NULL,
    acquisition_json TEXT NOT NULL,
    PRIMARY KEY(object_id, protocol_version, revision_id, manifest_digest, position),
    FOREIGN KEY(object_id, protocol_version, revision_id, manifest_digest)
        REFERENCES sinex_publication_payloads(object_id, protocol_version, revision_id, manifest_digest)
        ON DELETE CASCADE
) STRICT;

-- migration-safety: additive-no-backup
CREATE TABLE IF NOT EXISTS material_observations (
    material_id              TEXT PRIMARY KEY,
    referrer_ref             TEXT NOT NULL CHECK(length(trim(referrer_ref)) > 0),
    source_uri               TEXT NOT NULL CHECK(length(trim(source_uri)) > 0),
    acquisition_state        TEXT NOT NULL CHECK(acquisition_state IN (
        'claimed', 'acquired', 'duplicate', 'partial', 'unavailable',
        'expired', 'access_denied', 'malformed', 'superseded'
    )),
    diagnostic               TEXT NOT NULL DEFAULT '' CHECK(length(diagnostic) <= 4096),
    retryable                INTEGER NOT NULL CHECK(retryable IN (0, 1)),
    supersedes_material_id   TEXT REFERENCES material_observations(material_id),
    blob_hash                BLOB CHECK(blob_hash IS NULL OR length(blob_hash) = 32),
    byte_size                INTEGER CHECK(byte_size IS NULL OR byte_size >= 0),
    media_type               TEXT,
    media_charset            TEXT,
    filename                 TEXT,
    extraction_manifest_json TEXT NOT NULL DEFAULT '{}',
    custody                  TEXT NOT NULL CHECK(custody IN ('claimed', 'retained', 'verified', 'released')),
    privacy_classification   TEXT NOT NULL CHECK(privacy_classification IN (
        'private', 'restricted', 'public', 'synthetic'
    )),
    acquired_at_ms           INTEGER NOT NULL CHECK(acquired_at_ms >= 0),
    created_at_ms            INTEGER NOT NULL CHECK(created_at_ms >= 0),
    FOREIGN KEY (supersedes_material_id) REFERENCES material_observations(material_id)
) STRICT;
CREATE INDEX IF NOT EXISTS idx_material_observations_state
ON material_observations(acquisition_state, created_at_ms);
CREATE INDEX IF NOT EXISTS idx_material_observations_blob
ON material_observations(blob_hash);
CREATE TABLE IF NOT EXISTS material_evidence_links (
    material_id       TEXT NOT NULL REFERENCES material_observations(material_id) ON DELETE CASCADE,
    evidence_ref      TEXT NOT NULL CHECK(length(trim(evidence_ref)) > 0),
    relation          TEXT NOT NULL CHECK(relation IN ('refers_to', 'acquired_from', 'supports', 'affected')),
    authority         TEXT NOT NULL CHECK(authority IN ('provider', 'operator', 'repository', 'inferred', 'unknown')),
    confidence        REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    observed_at_ms    INTEGER NOT NULL CHECK(observed_at_ms >= 0),
    source_diagnostic TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(material_id, evidence_ref, relation),
    FOREIGN KEY (material_id) REFERENCES material_observations(material_id) ON DELETE CASCADE
) STRICT;
CREATE INDEX IF NOT EXISTS idx_material_evidence_links_ref
ON material_evidence_links(evidence_ref, relation);

"""Auxiliary schema DDL fragments."""

from __future__ import annotations

VEC0_DDL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS message_embeddings USING vec0(
        message_id TEXT PRIMARY KEY,
        embedding float[1024],
        +provider_name TEXT,
        +conversation_id TEXT
    )
"""


ARTIFACT_OBSERVATION_DDL = """
        CREATE TABLE IF NOT EXISTS artifact_observations (
            observation_id TEXT PRIMARY KEY,
            raw_id TEXT NOT NULL REFERENCES raw_conversations(raw_id) ON DELETE CASCADE,
            provider_name TEXT NOT NULL,
            payload_provider TEXT,
            source_name TEXT,
            source_path TEXT NOT NULL,
            source_index INTEGER,
            file_mtime TEXT,
            wire_format TEXT,
            artifact_kind TEXT NOT NULL,
            classification_reason TEXT NOT NULL,
            parse_as_conversation INTEGER NOT NULL DEFAULT 0,
            schema_eligible INTEGER NOT NULL DEFAULT 0,
            support_status TEXT NOT NULL
                CHECK (support_status IN (
                    'supported_parseable',
                    'recognized_unparsed',
                    'unsupported_parseable',
                    'decode_failed',
                    'unknown'
                )),
            malformed_jsonl_lines INTEGER NOT NULL DEFAULT 0,
            decode_error TEXT,
            bundle_scope TEXT,
            cohort_id TEXT,
            resolved_package_version TEXT,
            resolved_element_kind TEXT,
            resolution_reason TEXT,
            link_group_key TEXT,
            sidecar_agent_type TEXT,
            first_observed_at TEXT NOT NULL,
            last_observed_at TEXT NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_obs_source_identity
        ON artifact_observations(
            COALESCE(source_name, ''),
            source_path,
            COALESCE(source_index, -1)
        );

        CREATE INDEX IF NOT EXISTS idx_artifact_obs_raw_id
        ON artifact_observations(raw_id);

        CREATE INDEX IF NOT EXISTS idx_artifact_obs_provider_status
        ON artifact_observations(payload_provider, support_status);

        CREATE INDEX IF NOT EXISTS idx_artifact_obs_kind_status
        ON artifact_observations(artifact_kind, support_status);

        CREATE INDEX IF NOT EXISTS idx_artifact_obs_cohort
        ON artifact_observations(cohort_id)
        WHERE cohort_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_artifact_obs_link_group
        ON artifact_observations(link_group_key)
        WHERE link_group_key IS NOT NULL;
"""


PUBLICATION_DDL = """
        CREATE TABLE IF NOT EXISTS publications (
            publication_id TEXT PRIMARY KEY,
            publication_kind TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            duration_ms INTEGER,
            manifest_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_publications_kind_generated
        ON publications(publication_kind, generated_at DESC);
"""


__all__ = [
    "ARTIFACT_OBSERVATION_DDL",
    "PUBLICATION_DDL",
    "VEC0_DDL",
]

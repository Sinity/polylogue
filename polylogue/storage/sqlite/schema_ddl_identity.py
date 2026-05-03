"""Identity ledger and user metadata schema for the archive database.

These tables track the identity of imported conversations across reset cycles,
enabling identity-preserving soft-delete and user-metadata retention.

``identity_ledger`` records every raw→conversation mapping seen during ingest,
keyed by (provider, source, source_path, provider_conversation_id, raw_hash).
This allows the system to recognize a re-imported conversation as the same
logical entity even after a reset.

``conversation_user_metadata`` stores user-attached metadata (tags, summaries,
notes) keyed by the logical identity key, so it survives soft-delete and
re-import of the same conversation.
"""

from __future__ import annotations

IDENTITY_LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS identity_ledger (
    provider TEXT NOT NULL,
    source TEXT NOT NULL,
    source_path TEXT NOT NULL,
    provider_conversation_id TEXT NOT NULL,
    raw_hash TEXT NOT NULL,
    current_conversation_id TEXT NOT NULL,
    PRIMARY KEY (provider, source, source_path, provider_conversation_id, raw_hash)
);

CREATE INDEX IF NOT EXISTS idx_identity_ledger_conv_id
    ON identity_ledger(current_conversation_id);
"""

CONVERSATION_USER_METADATA_DDL = """
CREATE TABLE IF NOT EXISTS conversation_user_metadata (
    identity_key TEXT NOT NULL,
    metadata_kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (identity_key, metadata_kind)
);
"""

# Combined DDL for inclusion in schema bootstrap.
IDENTITY_DDL = IDENTITY_LEDGER_DDL + "\n\n" + CONVERSATION_USER_METADATA_DDL

__all__ = [
    "CONVERSATION_USER_METADATA_DDL",
    "IDENTITY_DDL",
    "IDENTITY_LEDGER_DDL",
]

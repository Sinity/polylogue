"""Schema DDL declarations grouped by storage domain bands."""

from __future__ import annotations

from polylogue.storage.sqlite.schema_ddl_actions import (
    ACTION_EVENT_DDL as _ACTION_EVENT_DDL,
)
from polylogue.storage.sqlite.schema_ddl_actions import (
    ACTION_FTS_DDL as _ACTION_FTS_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    ARCHIVE_STORAGE_DDL as _ARCHIVE_STORAGE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    BLACKBOARD_NOTES_DDL as _BLACKBOARD_NOTES_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    BLOB_LEASE_DDL as _BLOB_LEASE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    MESSAGE_FTS_DDL as _MESSAGE_FTS_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    OTLP_SPANS_DDL as _OTLP_SPANS_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    RAW_ARCHIVE_DDL as _RAW_ARCHIVE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    READER_WORKSPACES_DDL as _READER_WORKSPACES_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    RECALL_PACKS_DDL as _RECALL_PACKS_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    SAVED_VIEWS_DDL as _SAVED_VIEWS_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    TAGS_M2M_DDL as _TAGS_M2M_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    TOPOLOGY_EDGES_DDL as _TOPOLOGY_EDGES_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    USER_ANNOTATIONS_DDL as _USER_ANNOTATIONS_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    USER_CORRECTIONS_DDL as _USER_CORRECTIONS_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    USER_MARKS_DDL as _USER_MARKS_DDL,
)
from polylogue.storage.sqlite.schema_ddl_aux import (
    ARTIFACT_OBSERVATION_DDL as _ARTIFACT_OBSERVATION_DDL,
)
from polylogue.storage.sqlite.schema_ddl_aux import (
    VEC0_DDL as _VEC0_DDL,
)
from polylogue.storage.sqlite.schema_ddl_cursor import (
    SOURCE_FILE_CURSOR_DDL as _SOURCE_FILE_CURSOR_DDL,
)
from polylogue.storage.sqlite.schema_ddl_identity import (
    IDENTITY_DDL as _IDENTITY_DDL,
)
from polylogue.storage.sqlite.schema_ddl_insight_aggregates import (
    SESSION_INSIGHT_AGGREGATE_DDL as _SESSION_INSIGHT_AGGREGATE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_insight_latency import (
    SESSION_INSIGHT_LATENCY_DDL as _SESSION_INSIGHT_LATENCY_DDL,
)
from polylogue.storage.sqlite.schema_ddl_insight_profiles import (
    SESSION_INSIGHT_PROFILE_DDL as _SESSION_INSIGHT_PROFILE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_insight_timelines import (
    SESSION_INSIGHT_TIMELINE_DDL as _SESSION_INSIGHT_TIMELINE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_provider_events import (
    PROVIDER_EVENT_DDL as _PROVIDER_EVENT_DDL,
)
from polylogue.storage.sqlite.schema_ddl_repo_identity import (
    REPO_IDENTITY_DDL as _REPO_IDENTITY_DDL,
)

SCHEMA_VERSION = 22  # Canonical schema. No in-place upgrade chain: mismatch -> re-ingest from source. Added per-role message counts to session_stats (user/assistant/system/tool) to eliminate the messages-table scan from provider metrics.


# Complete target schema applied to fresh databases.
SCHEMA_DDL = (
    _RAW_ARCHIVE_DDL + "\n\n" + _ARTIFACT_OBSERVATION_DDL + "\n\n" + _ARCHIVE_STORAGE_DDL + "\n\n" + _MESSAGE_FTS_DDL
)

SCHEMA_DDL += "\n\n" + _SOURCE_FILE_CURSOR_DDL
SCHEMA_DDL += "\n\n" + _TAGS_M2M_DDL
SCHEMA_DDL += "\n\n" + _IDENTITY_DDL
SCHEMA_DDL += "\n\n" + _BLOB_LEASE_DDL
SCHEMA_DDL += "\n\n" + _PROVIDER_EVENT_DDL
SCHEMA_DDL += "\n\n" + _ACTION_EVENT_DDL
SCHEMA_DDL += _ACTION_FTS_DDL
SCHEMA_DDL += "\n\n" + _USER_MARKS_DDL
SCHEMA_DDL += "\n\n" + _USER_ANNOTATIONS_DDL
SCHEMA_DDL += "\n\n" + _SAVED_VIEWS_DDL
SCHEMA_DDL += "\n\n" + _RECALL_PACKS_DDL
SCHEMA_DDL += "\n\n" + _READER_WORKSPACES_DDL
SCHEMA_DDL += "\n\n" + _USER_CORRECTIONS_DDL
SCHEMA_DDL += "\n\n" + _TOPOLOGY_EDGES_DDL
SCHEMA_DDL += "\n\n" + _BLACKBOARD_NOTES_DDL
SCHEMA_DDL += "\n\n" + _OTLP_SPANS_DDL
SCHEMA_DDL += "\n\n" + _REPO_IDENTITY_DDL

_SESSION_INSIGHT_DDL = (
    _SESSION_INSIGHT_PROFILE_DDL
    + _SESSION_INSIGHT_TIMELINE_DDL
    + _SESSION_INSIGHT_LATENCY_DDL
    + _SESSION_INSIGHT_AGGREGATE_DDL
)

SCHEMA_DDL += _SESSION_INSIGHT_DDL

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_DDL",
    "_ACTION_EVENT_DDL",
    "_ACTION_FTS_DDL",
    "_ARTIFACT_OBSERVATION_DDL",
    "_ARCHIVE_STORAGE_DDL",
    "_BLOB_LEASE_DDL",
    "_IDENTITY_DDL",
    "_MESSAGE_FTS_DDL",
    "_PROVIDER_EVENT_DDL",
    "_RAW_ARCHIVE_DDL",
    "_REPO_IDENTITY_DDL",
    "_READER_WORKSPACES_DDL",
    "_SESSION_INSIGHT_DDL",
    "_SESSION_INSIGHT_LATENCY_DDL",
    "_SOURCE_FILE_CURSOR_DDL",
    "_RECALL_PACKS_DDL",
    "_SAVED_VIEWS_DDL",
    "_TAGS_M2M_DDL",
    "_TOPOLOGY_EDGES_DDL",
    "_USER_ANNOTATIONS_DDL",
    "_USER_CORRECTIONS_DDL",
    "_USER_MARKS_DDL",
    "_VEC0_DDL",
]

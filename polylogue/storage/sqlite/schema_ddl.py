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
    BLOB_LEASE_DDL as _BLOB_LEASE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    MESSAGE_FTS_DDL as _MESSAGE_FTS_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    RAW_ARCHIVE_DDL as _RAW_ARCHIVE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_archive import (
    TAGS_M2M_DDL as _TAGS_M2M_DDL,
)
from polylogue.storage.sqlite.schema_ddl_aux import (
    ARTIFACT_OBSERVATION_DDL as _ARTIFACT_OBSERVATION_DDL,
)
from polylogue.storage.sqlite.schema_ddl_aux import (
    PUBLICATION_DDL as _PUBLICATION_DDL,
)
from polylogue.storage.sqlite.schema_ddl_aux import (
    VEC0_DDL as _VEC0_DDL,
)
from polylogue.storage.sqlite.schema_ddl_insight_aggregates import (
    SESSION_INSIGHT_AGGREGATE_DDL as _SESSION_INSIGHT_AGGREGATE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_insight_profiles import (
    SESSION_INSIGHT_PROFILE_DDL as _SESSION_INSIGHT_PROFILE_DDL,
)
from polylogue.storage.sqlite.schema_ddl_insight_timelines import (
    SESSION_INSIGHT_TIMELINE_DDL as _SESSION_INSIGHT_TIMELINE_DDL,
)

SCHEMA_VERSION = 5


# Complete target schema applied to fresh databases.
SCHEMA_DDL = (
    _RAW_ARCHIVE_DDL
    + "\n\n"
    + _ARTIFACT_OBSERVATION_DDL
    + "\n\n"
    + _PUBLICATION_DDL
    + "\n\n"
    + _ARCHIVE_STORAGE_DDL
    + "\n\n"
    + _MESSAGE_FTS_DDL
)

SCHEMA_DDL += "\n\n" + _TAGS_M2M_DDL
SCHEMA_DDL += "\n\n" + _BLOB_LEASE_DDL
SCHEMA_DDL += "\n\n" + _ACTION_EVENT_DDL
SCHEMA_DDL += _ACTION_FTS_DDL

_SESSION_INSIGHT_DDL = _SESSION_INSIGHT_PROFILE_DDL + _SESSION_INSIGHT_TIMELINE_DDL + _SESSION_INSIGHT_AGGREGATE_DDL

SCHEMA_DDL += _SESSION_INSIGHT_DDL

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_DDL",
    "_ACTION_EVENT_DDL",
    "_ACTION_FTS_DDL",
    "_ARTIFACT_OBSERVATION_DDL",
    "_ARCHIVE_STORAGE_DDL",
    "_BLOB_LEASE_DDL",
    "_MESSAGE_FTS_DDL",
    "_PUBLICATION_DDL",
    "_RAW_ARCHIVE_DDL",
    "_SESSION_INSIGHT_DDL",
    "_TAGS_M2M_DDL",
    "_VEC0_DDL",
]

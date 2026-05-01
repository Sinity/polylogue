"""Schema DDL declarations grouped by storage domain bands."""

from __future__ import annotations

from polylogue.storage.backends.schema_ddl_actions import (
    ACTION_EVENT_DDL as _ACTION_EVENT_DDL,
)
from polylogue.storage.backends.schema_ddl_actions import (
    ACTION_FTS_DDL as _ACTION_FTS_DDL,
)
from polylogue.storage.backends.schema_ddl_archive import (
    ARCHIVE_STORAGE_DDL as _ARCHIVE_STORAGE_DDL,
)
from polylogue.storage.backends.schema_ddl_archive import (
    MESSAGE_FTS_DDL as _MESSAGE_FTS_DDL,
)
from polylogue.storage.backends.schema_ddl_archive import (
    RAW_ARCHIVE_DDL as _RAW_ARCHIVE_DDL,
)
from polylogue.storage.backends.schema_ddl_aux import (
    ARTIFACT_OBSERVATION_DDL as _ARTIFACT_OBSERVATION_DDL,
)
from polylogue.storage.backends.schema_ddl_aux import (
    PUBLICATION_DDL as _PUBLICATION_DDL,
)
from polylogue.storage.backends.schema_ddl_aux import (
    VEC0_DDL as _VEC0_DDL,
)
from polylogue.storage.backends.schema_ddl_product_aggregates import (
    SESSION_PRODUCT_AGGREGATE_DDL as _SESSION_PRODUCT_AGGREGATE_DDL,
)
from polylogue.storage.backends.schema_ddl_product_profiles import (
    SESSION_PRODUCT_PROFILE_DDL as _SESSION_PRODUCT_PROFILE_DDL,
)
from polylogue.storage.backends.schema_ddl_product_timelines import (
    SESSION_PRODUCT_TIMELINE_DDL as _SESSION_PRODUCT_TIMELINE_DDL,
)

SCHEMA_VERSION = 3


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

SCHEMA_DDL += _ACTION_EVENT_DDL
SCHEMA_DDL += _ACTION_FTS_DDL

_SESSION_PRODUCT_DDL = _SESSION_PRODUCT_PROFILE_DDL + _SESSION_PRODUCT_TIMELINE_DDL + _SESSION_PRODUCT_AGGREGATE_DDL

SCHEMA_DDL += _SESSION_PRODUCT_DDL

__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_DDL",
    "_ACTION_EVENT_DDL",
    "_ACTION_FTS_DDL",
    "_ARTIFACT_OBSERVATION_DDL",
    "_ARCHIVE_STORAGE_DDL",
    "_MESSAGE_FTS_DDL",
    "_PUBLICATION_DDL",
    "_RAW_ARCHIVE_DDL",
    "_SESSION_PRODUCT_DDL",
    "_VEC0_DDL",
]

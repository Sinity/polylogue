"""Session-product schema DDL fragments."""

from __future__ import annotations

from polylogue.storage.backends.schema_ddl_product_aggregates import (
    SESSION_PRODUCT_AGGREGATE_DDL,
)
from polylogue.storage.backends.schema_ddl_product_profiles import (
    SESSION_PRODUCT_PROFILE_DDL,
)
from polylogue.storage.backends.schema_ddl_product_timelines import (
    SESSION_PRODUCT_TIMELINE_DDL,
)

SESSION_PRODUCT_DDL = (
    SESSION_PRODUCT_PROFILE_DDL
    + SESSION_PRODUCT_TIMELINE_DDL
    + SESSION_PRODUCT_AGGREGATE_DDL
)


__all__ = ["SESSION_PRODUCT_DDL"]

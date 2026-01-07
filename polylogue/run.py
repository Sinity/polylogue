"""Pipeline orchestration (deprecated facade, see polylogue/pipeline/)."""

from __future__ import annotations

# Re-export pipeline components for backward compatibility
from polylogue.pipeline.ids import (
    attachment_content_id as _attachment_content_id,
)
from polylogue.pipeline.ids import (
    attachment_seed as _attachment_seed,
)
from polylogue.pipeline.ids import (
    conversation_content_hash as _conversation_content_hash,
)
from polylogue.pipeline.ids import (
    conversation_id as _conversation_id,
)
from polylogue.pipeline.ids import (
    message_content_hash as _message_content_hash,
)
from polylogue.pipeline.ids import (
    message_id as _message_id,
)
from polylogue.pipeline.ingest import prepare_ingest as _prepare_ingest
from polylogue.pipeline.models import (
    ExistingConversation,
    PlanResult,
    RunResult,
)
from polylogue.pipeline.runner import (
    latest_run,
    plan_sources,
    run_sources,
)

__all__ = [
    "PlanResult",
    "RunResult",
    "ExistingConversation",
    "plan_sources",
    "run_sources",
    "latest_run",
    "_prepare_ingest",
    "_conversation_id",
    "_message_id",
    "_attachment_seed",
    "_attachment_content_id",
    "_conversation_content_hash",
    "_message_content_hash",
]

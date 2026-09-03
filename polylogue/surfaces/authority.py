"""One typed authority snapshot shared by read-surface adapters."""

from __future__ import annotations

import uuid
from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AuthorityEnvelope(BaseModel):
    """Attribution for one operation result.

    The operation boundary creates this value once. Adapters only serialize it;
    they must not reconstruct archive or daemon facts independently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    archive_epoch: str
    generation_id: str
    tier_schema_versions: dict[str, int]
    server_identity: Literal["daemon", "direct"]
    elapsed_ms: int = Field(ge=0)
    degraded: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


# The descriptive name is retained for callers that refer to the metadata as
# a block rather than an envelope. It is the same type, not a second schema.
AuthorityBlock = AuthorityEnvelope


def build_authority_envelope(
    *,
    archive_epoch: str,
    generation_id: str,
    tier_schema_versions: dict[str, int],
    server_identity: Literal["daemon", "direct"],
    started_at: float | None = None,
    run_id: str | None = None,
    degraded: tuple[str, ...] = (),
) -> AuthorityEnvelope:
    """Assemble the envelope from facts the operation boundary already resolved.

    Archive resolution belongs to the operation boundary
    (:mod:`polylogue.operations.authority`), which knows which generation the
    reader opened. A surface that re-resolved the archive here would attribute
    a result to whichever generation happened to be active at serialization
    time.
    """

    reasons = list(degraded)
    if server_identity == "direct" and "daemon_unavailable" not in reasons:
        reasons.append("daemon_unavailable")
    return AuthorityEnvelope(
        run_id=run_id or str(uuid.uuid4()),
        archive_epoch=archive_epoch,
        generation_id=generation_id,
        tier_schema_versions=tier_schema_versions,
        server_identity=server_identity,
        elapsed_ms=max(0, round((monotonic() - started_at) * 1000)) if started_at is not None else 0,
        degraded=tuple(dict.fromkeys(reasons)),
    )


def serialize_authority(envelope: AuthorityEnvelope) -> dict[str, object]:
    """Serialize the canonical value for CLI, MCP, HTTP, and Python adapters."""

    return envelope.to_dict()


__all__ = ["AuthorityBlock", "AuthorityEnvelope", "build_authority_envelope", "serialize_authority"]

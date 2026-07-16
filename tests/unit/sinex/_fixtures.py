from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from polylogue.sinex.models import PublicationPayload


def publication_payload(
    object_id: str = "claude-code-session:s1",
    revision_id: str = "rev-1",
    marker: str = "one",
) -> PublicationPayload:
    manifest = (
        json.dumps(
            {"marker": marker, "object_id": object_id, "revision_id": revision_id},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    return PublicationPayload(
        object_id=object_id,
        protocol_version="polylogue.material-protocol/v1",
        revision_id=revision_id,
        manifest_digest=hashlib.sha256(manifest).hexdigest(),
        manifest_bytes=manifest,
        segments=(
            ("head.ndjson", b'{"kind":"head"}\n'),
            ("seg-00000.ndjson", marker.encode("utf-8")),
        ),
    )


@dataclass
class MutableClock:
    now_ms: int = 1_000

    def __call__(self) -> int:
        return self.now_ms

    def advance(self, milliseconds: int) -> None:
        self.now_ms += milliseconds

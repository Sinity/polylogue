"""Representative corpus manifest and generation for provider schemas.

Representative corpora are small, reviewed, deterministic provider exports
that live adjacent to schema packages and serve as demo/test data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from polylogue.types import Provider


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Metadata for one representative corpus sample set."""

    provider: str
    schema_version: str
    generator_command: str
    generator_version: str
    seed: int
    source_mode: str  # "schema-only", "curated-theme", "llm-authored"
    sample_count: int
    privacy_status: str  # "reviewed", "auto-generated-safe"
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_path(cls, path: Path) -> CorpusManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def write(self, path: Path) -> None:
        data = {
            "provider": self.provider,
            "schema_version": self.schema_version,
            "generator_command": self.generator_command,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "source_mode": self.source_mode,
            "sample_count": self.sample_count,
            "privacy_status": self.privacy_status,
            "generated_at": self.generated_at,
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def representatives_dir(provider: Provider | str) -> Path:
    provider_name = getattr(provider, "value", str(provider))
    return Path(__file__).resolve().parents[1] / "schemas" / "providers" / provider_name / "representatives"


__all__ = ["CorpusManifest", "representatives_dir"]

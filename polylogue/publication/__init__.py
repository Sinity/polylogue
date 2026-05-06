"""Typed publication manifest models and output artifact scanning."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class PublishedArtifact(BaseModel):
    """One materialized artifact written to an output directory."""

    relative_path: str
    size_bytes: int
    sha256: str | None = None

    @field_validator("relative_path")
    @classmethod
    def non_empty_relative_path(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("relative_path cannot be empty")
        return value


class OutputManifest(BaseModel):
    """Stable manifest for a directory of materialized output artifacts."""

    schema_version: int = 1
    entry_count: int = 0
    entries: list[PublishedArtifact] = Field(default_factory=list)

    @classmethod
    def scan(
        cls,
        output_dir: Path,
        *,
        include_hashes: bool = True,
        exclude_paths: set[str] | None = None,
    ) -> OutputManifest:
        """Scan ``output_dir`` into a stable manifest."""
        entries: list[PublishedArtifact] = []
        excluded = {path.replace("\\", "/") for path in (exclude_paths or set())}
        if output_dir.exists():
            for path in sorted(output_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative_path = str(path.relative_to(output_dir)).replace("\\", "/")
                if relative_path in excluded:
                    continue
                sha256: str | None = None
                if include_hashes:
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    sha256 = digest.hexdigest()
                entries.append(
                    PublishedArtifact(
                        relative_path=relative_path,
                        size_bytes=path.stat().st_size,
                        sha256=sha256,
                    )
                )
        return cls(entry_count=len(entries), entries=entries)

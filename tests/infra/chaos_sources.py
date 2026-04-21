"""Chaos source builders for hostile ingestion testing.

Builds on large_batches.py to produce complete inbox directories
with controlled corruption patterns for integration testing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

from tests.infra.large_batches import (
    corrupt_line_bad_utf8,
    corrupt_line_malformed_json,
    corrupt_line_truncated,
    corrupt_line_wrong_envelope,
    generate_large_jsonl,
    write_jsonl_file,
    write_jsonl_with_bad_utf8,
)

ChaosFileSpec: TypeAlias = dict[str, object]


def _spec_str(spec: ChaosFileSpec, key: str, default: str = "") -> str:
    value = spec.get(key, default)
    return value if isinstance(value, str) else default


def _spec_int(spec: ChaosFileSpec, key: str, default: int = 0) -> int:
    value = spec.get(key, default)
    return value if isinstance(value, int) else default


def _spec_indices(spec: ChaosFileSpec) -> list[int]:
    value = spec.get("corrupt_indices", [])
    return [item for item in value if isinstance(item, int)] if isinstance(value, list) else []


class ChaosInboxBuilder:
    """Builder for inbox directories with controlled corruption patterns.

    Creates realistic failure scenarios for resilience testing:
    - Valid provider exports alongside corrupted files
    - Mixed corruption types (malformed JSON, truncation, UTF-8, etc.)
    - Empty files, binary garbage, zero-byte files
    - BOM markers and nested subdirectories
    """

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, ChaosFileSpec] = {}

    def add_valid_jsonl(
        self,
        filename: str,
        provider: str = "codex",
        count: int = 10,
    ) -> ChaosInboxBuilder:
        """Add a valid JSONL file with N records."""
        self._files[filename] = {
            "type": "valid_jsonl",
            "provider": provider,
            "count": count,
        }
        return self

    def add_corrupted_jsonl(
        self,
        filename: str,
        provider: str = "codex",
        count: int = 100,
        corrupt_indices: list[int] | None = None,
        corruption_type: str = "malformed",
    ) -> ChaosInboxBuilder:
        """Add a JSONL file with controlled corruption at specific line indices.

        Args:
            filename: Target filename
            provider: Provider name (codex, claude-code)
            count: Total number of records
            corrupt_indices: Line indices to corrupt (0-based)
            corruption_type: Type of corruption (malformed, truncated, bad_utf8, wrong_envelope)
        """
        self._files[filename] = {
            "type": "corrupted_jsonl",
            "provider": provider,
            "count": count,
            "corrupt_indices": corrupt_indices or [],
            "corruption_type": corruption_type,
        }
        return self

    def add_empty_file(self, filename: str) -> ChaosInboxBuilder:
        """Add an empty zero-byte file."""
        self._files[filename] = {"type": "empty"}
        return self

    def add_binary_garbage(self, filename: str, size: int = 256) -> ChaosInboxBuilder:
        """Add a file with random binary garbage."""
        self._files[filename] = {"type": "binary_garbage", "size": size}
        return self

    def add_file_with_bom(self, filename: str) -> ChaosInboxBuilder:
        """Add a JSONL file with UTF-8 BOM marker."""
        self._files[filename] = {"type": "jsonl_with_bom"}
        return self

    def add_mixed_provider_dir(
        self,
        dirname: str,
        providers: list[str] | None = None,
        count_per_provider: int = 10,
    ) -> ChaosInboxBuilder:
        """Add a subdirectory with multiple providers' JSONL files."""
        providers = providers or ["codex", "claude-code"]
        for provider in providers:
            filename = f"{dirname}/{provider}_data.jsonl"
            self.add_valid_jsonl(filename, provider=provider, count=count_per_provider)
        return self

    def build(self) -> Path:
        """Materialize all files to the inbox directory."""
        for filename, spec in self._files.items():
            path = self.base_path / filename
            path.parent.mkdir(parents=True, exist_ok=True)

            spec_type = _spec_str(spec, "type")

            if spec_type == "empty":
                path.write_bytes(b"")

            elif spec_type == "binary_garbage":
                path.write_bytes(b"\x00\xff\xde\xad\xbe\xef" * (_spec_int(spec, "size") // 6))

            elif spec_type == "valid_jsonl":
                lines = generate_large_jsonl(
                    _spec_int(spec, "count"),
                    provider=_spec_str(spec, "provider", "codex"),
                )
                write_jsonl_file(path, lines)

            elif spec_type == "jsonl_with_bom":
                # Valid codex JSONL but with UTF-8 BOM marker
                lines = generate_large_jsonl(5, provider="codex")
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    # Write UTF-8 BOM
                    f.write(b"\xef\xbb\xbf")
                    # Write JSONL content
                    for line in lines:
                        f.write(line.encode("utf-8") + b"\n")

            elif spec_type == "corrupted_jsonl":
                lines = generate_large_jsonl(
                    _spec_int(spec, "count"),
                    provider=_spec_str(spec, "provider", "codex"),
                )

                # Apply corruptions in reverse order to preserve indices
                corruption_type = _spec_str(spec, "corruption_type", "malformed")
                corruption_fn = self._get_corruption_fn(corruption_type)
                for idx in sorted(_spec_indices(spec), reverse=True):
                    if 0 <= idx < len(lines):
                        lines = corruption_fn(lines, idx)

                # Special handling for bad UTF-8
                if corruption_type == "bad_utf8":
                    write_jsonl_with_bad_utf8(path, lines)
                else:
                    write_jsonl_file(path, lines)

        return self.base_path

    @staticmethod
    def _get_corruption_fn(corruption_type: str) -> Callable[[list[str], int], list[str]]:
        """Map corruption type to corruption function."""
        mapping = {
            "malformed": corrupt_line_malformed_json,
            "truncated": corrupt_line_truncated,
            "bad_utf8": corrupt_line_bad_utf8,
            "wrong_envelope": corrupt_line_wrong_envelope,
        }
        return mapping.get(corruption_type, corrupt_line_malformed_json)


def build_corrupted_codex_inbox(
    tmp_path: Path,
    *,
    num_records: int = 100,
    corrupt_indices: list[int] | None = None,
) -> Path:
    """Convenience function: codex JSONL with specific line corruptions.

    Args:
        tmp_path: Directory to create inbox in
        num_records: Total number of records
        corrupt_indices: Which lines to corrupt (0-based, default: [5, 10, 15])

    Returns:
        Path to the created inbox directory
    """
    if corrupt_indices is None:
        corrupt_indices = [5, 10, 15]

    inbox = tmp_path / "inbox"
    builder = ChaosInboxBuilder(inbox)
    builder.add_corrupted_jsonl(
        "corrupted.jsonl",
        provider="codex",
        count=num_records,
        corrupt_indices=corrupt_indices,
        corruption_type="malformed",
    )
    return builder.build()


def build_mixed_provider_inbox(
    tmp_path: Path,
    *,
    providers: list[str] | None = None,
    count_per_provider: int = 10,
) -> Path:
    """Convenience function: inbox with multiple providers in separate files.

    Args:
        tmp_path: Directory to create inbox in
        providers: List of provider names (default: ["codex", "claude-code"])
        count_per_provider: Records per provider file

    Returns:
        Path to the created inbox directory
    """
    providers = providers or ["codex", "claude-code"]
    inbox = tmp_path / "inbox"
    builder = ChaosInboxBuilder(inbox)

    for provider in providers:
        filename = f"{provider}_export.jsonl"
        builder.add_valid_jsonl(filename, provider=provider, count=count_per_provider)

    return builder.build()

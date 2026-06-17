"""Import-source preflight classification for truthful scheduling.

This module intentionally does not parse full sessions.  It answers the
admission-time question: does the staged artifact contain at least one
payload shape that Polylogue knows how to parse, and are there caveats the
operator should see before the daemon claims the import is pending?
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Any

from polylogue.core.enums import Provider
from polylogue.sources.decoders import _decode_json_bytes, _iter_json_stream
from polylogue.sources.dispatch import detect_provider

_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_ZIP_JSON_SUFFIXES = (".json", ".jsonl", ".ndjson", ".jsonl.txt")
_MAX_DIRECTORY_CANDIDATES = 256
_MAX_STREAM_RECORDS = 32


class ImportPreflightStatus(str, Enum):
    """Admission-time classification for an import source."""

    SUPPORTED = "supported"
    DEGRADED = "degraded"
    UNSUPPORTED = "unsupported"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class ImportPreflightResult:
    """Bounded preflight envelope shared by daemon and CLI tests."""

    status: ImportPreflightStatus
    source_path: str
    candidate_count: int = 0
    supported_count: int = 0
    unsupported_count: int = 0
    malformed_count: int = 0
    ignored_count: int = 0
    providers: tuple[Provider, ...] = ()
    caveats: tuple[str, ...] = ()
    samples: tuple[str, ...] = ()

    @property
    def admissible(self) -> bool:
        return self.status in {ImportPreflightStatus.SUPPORTED, ImportPreflightStatus.DEGRADED}

    @property
    def error_code(self) -> str:
        if self.status is ImportPreflightStatus.MALFORMED:
            return "malformed_import_source"
        if self.status is ImportPreflightStatus.UNSUPPORTED:
            return "unsupported_import_source"
        return ""

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "source_path": self.source_path,
            "candidate_count": self.candidate_count,
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "malformed_count": self.malformed_count,
            "ignored_count": self.ignored_count,
            "providers": [provider.value for provider in self.providers],
            "caveats": list(self.caveats),
            "samples": list(self.samples),
        }

    def summary(self) -> str:
        if self.status is ImportPreflightStatus.SUPPORTED:
            provider_names = ", ".join(provider.value for provider in self.providers) or "supported provider"
            return f"Import source preflight passed: {self.supported_count} supported candidate(s) ({provider_names})."
        if self.status is ImportPreflightStatus.DEGRADED:
            return (
                "Import source preflight is degraded: "
                f"{self.supported_count} supported, {self.unsupported_count} unsupported, "
                f"{self.malformed_count} malformed candidate(s)."
            )
        if self.status is ImportPreflightStatus.MALFORMED:
            return f"Import source is malformed: {self.malformed_count} candidate(s) could not be decoded."
        return "Import source is unsupported: no parseable Polylogue export shape was detected."


@dataclass(slots=True)
class _PreflightAccumulator:
    source_path: str
    candidate_count: int = 0
    supported_count: int = 0
    unsupported_count: int = 0
    malformed_count: int = 0
    ignored_count: int = 0
    providers: set[Provider] = field(default_factory=set)
    caveats: list[str] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)

    def supported(self, label: str, provider: Provider) -> None:
        self.candidate_count += 1
        self.supported_count += 1
        self.providers.add(provider)
        self._sample(f"{label}: {provider.value}")

    def unsupported(self, label: str, reason: str) -> None:
        self.candidate_count += 1
        self.unsupported_count += 1
        self._caveat(f"{label}: {reason}")

    def malformed(self, label: str, reason: str) -> None:
        self.candidate_count += 1
        self.malformed_count += 1
        self._caveat(f"{label}: {reason}")

    def ignored(self) -> None:
        self.ignored_count += 1

    def _sample(self, value: str) -> None:
        if len(self.samples) < 5:
            self.samples.append(value)

    def _caveat(self, value: str) -> None:
        if len(self.caveats) < 5:
            self.caveats.append(value)

    def result(self) -> ImportPreflightResult:
        status = self._status()
        return ImportPreflightResult(
            status=status,
            source_path=self.source_path,
            candidate_count=self.candidate_count,
            supported_count=self.supported_count,
            unsupported_count=self.unsupported_count,
            malformed_count=self.malformed_count,
            ignored_count=self.ignored_count,
            providers=tuple(sorted(self.providers, key=lambda provider: provider.value)),
            caveats=tuple(self.caveats),
            samples=tuple(self.samples),
        )

    def _status(self) -> ImportPreflightStatus:
        if self.supported_count > 0 and (self.unsupported_count > 0 or self.malformed_count > 0):
            return ImportPreflightStatus.DEGRADED
        if self.supported_count > 0:
            return ImportPreflightStatus.SUPPORTED
        if self.malformed_count > 0:
            return ImportPreflightStatus.MALFORMED
        return ImportPreflightStatus.UNSUPPORTED


def preflight_import_source(path: Path) -> ImportPreflightResult:
    """Classify a staged import source before the daemon claims acceptance."""
    resolved = path.resolve()
    acc = _PreflightAccumulator(source_path=str(resolved))
    if resolved.is_dir():
        _preflight_directory(resolved, acc)
    else:
        _preflight_file(resolved, acc, label=resolved.name)
    return acc.result()


def _preflight_directory(path: Path, acc: _PreflightAccumulator) -> None:
    candidates_seen = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        if not _is_candidate_path(child):
            acc.ignored()
            continue
        candidates_seen += 1
        if candidates_seen > _MAX_DIRECTORY_CANDIDATES:
            acc._caveat(f"{path}: stopped after {_MAX_DIRECTORY_CANDIDATES} candidate files")
            break
        _preflight_file(child, acc, label=str(child.relative_to(path)))
    if candidates_seen == 0:
        acc.unsupported(str(path), "directory contains no JSON, JSONL, or ZIP import candidates")


def _preflight_file(path: Path, acc: _PreflightAccumulator, *, label: str) -> None:
    lower_name = path.name.lower()
    if lower_name.endswith(".zip"):
        _preflight_zip(path, acc, label=label)
        return
    if not _is_json_candidate_name(lower_name):
        acc.unsupported(label, "file extension is not a supported import candidate")
        return
    try:
        raw = path.read_bytes()
    except OSError as exc:
        acc.malformed(label, f"could not read file: {exc}")
        return
    _preflight_json_bytes(raw, acc, label=label)


def _preflight_zip(path: Path, acc: _PreflightAccumulator, *, label: str) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            json_entries = [
                info
                for info in zf.infolist()
                if not info.is_dir() and info.filename.lower().endswith(_ZIP_JSON_SUFFIXES)
            ]
            if not json_entries:
                acc.unsupported(label, "ZIP contains no JSON or JSONL import candidates")
                return
            for info in json_entries:
                entry_label = f"{label}:{info.filename}"
                try:
                    raw = zf.read(info)
                except (OSError, zipfile.BadZipFile) as exc:
                    acc.malformed(entry_label, f"could not read ZIP entry: {exc}")
                    continue
                _preflight_json_bytes(raw, acc, label=entry_label)
    except zipfile.BadZipFile as exc:
        acc.malformed(label, f"invalid ZIP archive: {exc}")
    except OSError as exc:
        acc.malformed(label, f"could not read ZIP archive: {exc}")


def _preflight_json_bytes(raw: bytes, acc: _PreflightAccumulator, *, label: str) -> None:
    text = _decode_json_bytes(raw)
    if text is None:
        acc.malformed(label, "unsupported text encoding")
        return
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        _preflight_json_stream(raw, acc, label=label)
        return
    provider = detect_provider(payload)
    if provider is None:
        acc.unsupported(label, "JSON shape is not a supported export")
        return
    acc.supported(label, provider)


def _preflight_json_stream(raw: bytes, acc: _PreflightAccumulator, *, label: str) -> None:
    try:
        payloads = list(islice(_iter_json_stream(BytesIO(raw), label), _MAX_STREAM_RECORDS))
    except Exception as exc:
        acc.malformed(label, f"could not decode JSON stream: {type(exc).__name__}")
        return
    provider = detect_provider(payloads)
    if provider is None:
        acc.unsupported(label, "JSONL shape is not a supported export")
        return
    acc.supported(label, provider)


def _is_candidate_path(path: Path) -> bool:
    return path.name.lower().endswith(".zip") or _is_json_candidate_name(path.name.lower())


def _is_json_candidate_name(name: str) -> bool:
    return name.endswith(".jsonl.txt") or Path(name).suffix.lower() in _JSON_SUFFIXES


__all__ = [
    "ImportPreflightResult",
    "ImportPreflightStatus",
    "preflight_import_source",
]

"""Import-source preflight classification for truthful scheduling.

This module answers the
admission-time question: does the staged artifact contain at least one
payload shape that Polylogue knows how to parse, and are there caveats the
operator should see before the daemon claims the import is pending?
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from polylogue.core.enums import Provider
from polylogue.sources.decoder_zip import (
    ZIP_JSON_SUFFIXES,
    ZipEntryValidator,
    open_zip_entry,
)
from polylogue.sources.dispatch import detect_provider_from_stream_evidence
from polylogue.sources.sqlite_inspection import inspect_sqlite_source

if TYPE_CHECKING:
    from polylogue.sources.source_staging import SourceInputBinding

_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})


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
        if self.supported_count > 0 and (self.unsupported_count > 0 or self.malformed_count > 0 or self.caveats):
            return ImportPreflightStatus.DEGRADED
        if self.supported_count > 0:
            return ImportPreflightStatus.SUPPORTED
        if self.malformed_count > 0:
            return ImportPreflightStatus.MALFORMED
        return ImportPreflightStatus.UNSUPPORTED


def _preflight_sqlite(
    path: Path,
    acc: _PreflightAccumulator,
    *,
    label: str,
    source_binding: SourceInputBinding | None = None,
    check_stop: Callable[[], None] | None = None,
) -> None:
    """Classify a SQLite import by its provider schema, never by its suffix."""
    callback_failed = False

    def heartbeat() -> None:
        nonlocal callback_failed
        from polylogue.core.compute_cancel import check_compute_cancelled

        callback_failed = True
        check_compute_cancelled()
        if check_stop is not None:
            check_stop()
        callback_failed = False

    try:
        inspection = inspect_sqlite_source(path, preflight=True, source_binding=source_binding, check_stop=heartbeat)
        if inspection.domain == "antigravity_trajectory_db":
            if inspection.admitted:
                acc.supported(label, Provider.ANTIGRAVITY)
                if inspection.degraded:
                    acc._caveat(f"{label}: trajectory contains unsupported, degraded or empty steps")
            else:
                acc.unsupported(label, "Antigravity trajectory schema contains no materialized messages")
            return
    except Exception as exc:
        from polylogue.core.compute import DaemonOperationCancelled

        if callback_failed or isinstance(exc, DaemonOperationCancelled):
            raise
        acc.malformed(label, f"could not inspect SQLite trajectory: {type(exc).__name__}: {exc}")
        return
    acc.unsupported(label, "SQLite schema is not a supported Antigravity trajectory store")


def _preflight_zip(
    path: Path,
    acc: _PreflightAccumulator,
    *,
    label: str,
    handle: IO[bytes] | None = None,
    check_stop: Callable[[], None] | None = None,
) -> None:
    try:
        with zipfile.ZipFile(path if handle is None else handle) as zf:
            validator = ZipEntryValidator("unknown", cursor_state=None, zip_path=path)
            admitted = False
            for info in validator.filter_entries(
                zf.infolist(),
                allowed_suffixes=ZIP_JSON_SUFFIXES,
            ):
                admitted = True
                entry_label = f"{label}:{info.filename}"
                try:
                    with open_zip_entry(zf, info) as handle:
                        _preflight_json_handle(handle, acc, label=entry_label, check_stop=check_stop)
                except (OSError, KeyError, zipfile.BadZipFile) as exc:
                    acc.malformed(entry_label, f"could not read ZIP entry: {exc}")
                    continue

            if not admitted:
                acc.unsupported(label, "ZIP contains no JSON or JSONL import candidates")
    except zipfile.BadZipFile as exc:
        acc.malformed(label, f"invalid ZIP archive: {exc}")
    except OSError as exc:
        acc.malformed(label, f"could not read ZIP archive: {exc}")


def _preflight_json_handle(
    handle: IO[bytes], acc: _PreflightAccumulator, *, label: str, check_stop: Callable[[], None] | None = None
) -> None:
    import ijson

    try:
        provider, _evidence = detect_provider_from_stream_evidence(handle, check_stop=check_stop)
    except (ijson.JSONError, UnicodeError, ValueError) as exc:
        acc.malformed(label, f"could not decode complete JSON input: {type(exc).__name__}")
        return
    if provider is None:
        acc.unsupported(label, "JSON shape is not a supported export")
        return
    acc.supported(label, provider)


def _is_candidate_path(path: Path) -> bool:
    lower_name = path.name.lower()
    return (
        lower_name.endswith(".zip")
        or _is_json_candidate_name(lower_name)
        or Path(lower_name).suffix in {".db", ".sqlite", ".sqlite3"}
    )


def _is_json_candidate_name(name: str) -> bool:
    return name.endswith(".jsonl.txt") or Path(name).suffix.lower() in _JSON_SUFFIXES


__all__ = [
    "ImportPreflightResult",
    "ImportPreflightStatus",
]


def _preflight_handle(
    handle: IO[bytes], semantic_path: Path, *, check_stop: Callable[[], None] | None = None
) -> ImportPreflightResult:
    """Inspect the actual accepted byte descriptor in its fresh reader process."""
    acc = _PreflightAccumulator(str(semantic_path))
    if semantic_path.suffix.lower() == ".zip":
        _preflight_zip(semantic_path, acc, label=semantic_path.name, handle=handle, check_stop=check_stop)
    elif _is_json_candidate_name(semantic_path.name.lower()):
        _preflight_json_handle(handle, acc, label=semantic_path.name, check_stop=check_stop)
    else:
        acc.unsupported(semantic_path.name, "file extension is not a supported import candidate")
    return acc.result()


def _decode_bound_preflight(value: dict[str, Any], semantic_path: Path) -> ImportPreflightResult:
    names = {
        "status",
        "source_path",
        "candidate_count",
        "supported_count",
        "unsupported_count",
        "malformed_count",
        "ignored_count",
        "providers",
        "caveats",
        "samples",
    }
    if set(value) != names or value["source_path"] != str(semantic_path):
        raise ValueError("source preflight protocol differs from its accepted coordinate")
    counts = {name: value[name] for name in names if name.endswith("_count")}
    if any(type(count) is not int or count < 0 for count in counts.values()):
        raise ValueError("invalid source preflight counts")
    if counts["candidate_count"] != sum(
        counts[name] for name in ("supported_count", "unsupported_count", "malformed_count")
    ):
        raise ValueError("inconsistent source preflight counts")
    if any(
        not isinstance(value[name], list) or any(not isinstance(item, str) for item in value[name])
        for name in ("providers", "caveats", "samples")
    ):
        raise ValueError("invalid source preflight evidence")
    acc = _PreflightAccumulator(
        str(semantic_path),
        **counts,
        providers={Provider(provider) for provider in value["providers"]},
        caveats=value["caveats"],
        samples=value["samples"],
    )
    result = acc.result()
    if result.status.value != value["status"]:
        raise ValueError("inconsistent source preflight status")
    return result


def preflight_import_bindings(
    members: Iterable[tuple[SourceInputBinding, str]],
    *,
    source_path: str,
    single_file: bool,
    check_stop: Callable[[], None],
) -> ImportPreflightResult:
    """Classify every captured member before scheduling, preserving original labels."""
    from polylogue.sources.source_staging import preflight_bound_bytes
    from polylogue.sources.sqlite_snapshot import is_sqlite_path

    acc = _PreflightAccumulator(source_path)
    seen = False
    for binding, label in members:
        seen = True
        if not _is_candidate_path(binding.source_path):
            if single_file:
                acc.unsupported(label, "file extension is not a supported import candidate")
            else:
                acc.ignored()
            continue
        if is_sqlite_path(binding.source_path):
            _preflight_sqlite(binding.source, acc, label=label, source_binding=binding, check_stop=check_stop)
            continue
        result = _decode_bound_preflight(preflight_bound_bytes(binding, check_stop=check_stop), binding.source_path)
        for name in ("candidate_count", "supported_count", "unsupported_count", "malformed_count", "ignored_count"):
            setattr(acc, name, getattr(acc, name) + getattr(result, name))
        acc.providers.update(result.providers)
        # The bound preflight already labels its evidence with the member's
        # semantic source path; prefixing the binding label again doubles it.
        for caveat in result.caveats:
            acc._caveat(caveat)
        for sample in result.samples:
            acc._sample(sample)
    if not seen or not acc.candidate_count:
        acc.unsupported(source_path, "directory contains no JSON, JSONL, or ZIP import candidates")
    return acc.result()

"""Production-route differential support for source normalization.

This is deliberately a test adapter, not a second source registry.  The
``OriginSpec`` inventory decides which routes exist; this module only supplies
the common execution and semantic projection used by tests.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.sources.assembly import get_assembly_spec
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import STREAM_RECORD_PROVIDERS, parse_payload, parse_stream_payload
from polylogue.sources.origin_specs import OriginSpec, origin_specs
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.source_parsing import iter_source_sessions

# These are transport/runtime fields, not semantic values.  They are typed and
# path-local so adding a new ignored field requires naming its exact location.
# In particular, timestamps, outcomes, topology, events, titles and content are
# intentionally absent and therefore remain part of the comparison.
OPERATIONAL_PATHS: Mapping[str, frozenset[str]] = {
    "session": frozenset(),
    "message": frozenset({"parent_message_position", "owner_coordinate"}),
    "attachment": frozenset(
        {"message_position", "message_variant_index", "owner_coordinate", "inline_bytes", "precomputed_blob"}
    ),
}


@dataclass(frozen=True, slots=True)
class SourceSpecimen:
    """Provider bytes and sidecars shared by every isolated route."""

    provider: Provider
    raw_bytes: bytes
    filename: str = "specimen.jsonl"
    sidecars: Mapping[str, bytes] = field(default_factory=dict)
    fallback_id: str = "differential-specimen"


@dataclass(frozen=True, slots=True)
class AdapterDeclaration:
    identity: str
    origin: str
    provider: Provider
    kind: str
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RouteResult:
    adapter: AdapterDeclaration
    input_hash: str
    sidecar_hash: str
    sessions: tuple[dict[str, object], ...]
    semantic_hash: str


@dataclass(frozen=True, slots=True)
class DifferentialReport:
    routes: tuple[RouteResult, ...]

    @property
    def adapters(self) -> tuple[str, ...]:
        return tuple(result.adapter.identity for result in self.routes)

    @property
    def canonical_hash(self) -> str:
        hashes = {result.semantic_hash for result in self.routes}
        if len(hashes) != 1:
            raise AssertionError(f"semantic route drift: {self._hashes()}")
        return next(iter(hashes))

    def _hashes(self) -> dict[str, str]:
        return {result.adapter.identity: result.semantic_hash for result in self.routes}

    def assert_complete(self) -> None:
        identities = self.adapters
        if len(identities) != len(set(identities)):
            raise AssertionError(f"duplicate adapter execution: {identities}")
        if not identities:
            raise AssertionError("no declared adapters executed")
        if len(set(self._hashes().values())) != 1:
            raise AssertionError(f"semantic route drift: {self._hashes()}")


def declared_adapters(specs: Sequence[OriginSpec] | None = None) -> tuple[AdapterDeclaration, ...]:
    """Derive retained normalization routes directly from current declarations."""
    result: list[AdapterDeclaration] = []
    for spec in origin_specs() if specs is None else specs:
        if spec.lifecycle != "executable" or not spec.provider_wires:
            continue
        provider = spec.provider_wires[0]
        evidence = (*spec.parser_paths, *spec.assembly_paths)
        result.append(AdapterDeclaration(f"{spec.origin.value}:eager", spec.origin.value, provider, "eager", evidence))
        if spec.stream_parser_path is not None and provider in STREAM_RECORD_PROVIDERS:
            result.append(
                AdapterDeclaration(
                    f"{spec.origin.value}:streaming",
                    spec.origin.value,
                    provider,
                    "streaming",
                    (spec.stream_parser_path,),
                )
            )
        result.append(
            AdapterDeclaration(
                f"{spec.origin.value}:replay",
                spec.origin.value,
                provider,
                "replay",
                ("polylogue.sources.source_parsing.iter_source_sessions",),
            )
        )
        if spec.assembly_spec_path is not None:
            result.append(
                AdapterDeclaration(
                    f"{spec.origin.value}:assembly", spec.origin.value, provider, "assembly", (spec.assembly_spec_path,)
                )
            )
    return tuple(result)


def _without_operational(value: object, *, path: str) -> object:
    if isinstance(value, dict):
        ignored = OPERATIONAL_PATHS.get(path, frozenset())
        return {key: _without_operational(item, path=path) for key, item in value.items() if key not in ignored}
    if isinstance(value, list):
        child = "message" if path == "session" else "attachment" if path == "attachments" else path
        return [_without_operational(item, path=child) for item in value]
    return value


def project_sessions(sessions: Sequence[ParsedSession]) -> tuple[dict[str, object], ...]:
    """Project all semantic axes of parsed sessions into stable JSON values."""
    values = [
        cast(dict[str, object], _without_operational(session.model_dump(mode="json"), path="session"))
        for session in sessions
    ]
    return tuple(sorted(values, key=lambda item: (str(item.get("source_name")), str(item.get("provider_session_id")))))


def semantic_hash(sessions: Sequence[ParsedSession]) -> str:
    payload = json.dumps(project_sessions(sessions), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _decode(raw: bytes) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def run_differential(specimen: SourceSpecimen, *, spec: OriginSpec | None = None) -> DifferentialReport:
    """Run every declared route for one specimen in isolated filesystem state."""
    current = spec or next(item for item in origin_specs() if specimen.provider in item.provider_wires)
    declarations = tuple(item for item in declared_adapters((current,)))
    if not declarations:
        raise AssertionError(f"no executable declaration for {specimen.provider.value}")
    input_hash = hashlib.sha256(specimen.raw_bytes).hexdigest()
    sidecar_payload = json.dumps(dict(specimen.sidecars), sort_keys=True, default=str).encode()
    sidecar_hash = hashlib.sha256(sidecar_payload).hexdigest()
    results: list[RouteResult] = []
    with TemporaryDirectory(prefix="polylogue-source-differential-") as temporary:
        root = Path(temporary)
        source_path = root / specimen.filename
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(specimen.raw_bytes)
        for name, data in (specimen.sidecars or {}).items():
            sidecar_path = root / name
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_bytes(data)
        for adapter in declarations:
            if adapter.kind == "eager":
                sessions = parse_payload(
                    specimen.provider, _decode(specimen.raw_bytes), specimen.fallback_id, source_path=str(source_path)
                )
            elif adapter.kind == "streaming":
                sessions = parse_stream_payload(
                    specimen.provider,
                    _iter_json_stream(io.BytesIO(specimen.raw_bytes), specimen.filename),
                    specimen.fallback_id,
                    source_path=str(source_path),
                )
            else:
                sessions = list(iter_source_sessions(Source(name=specimen.provider.value, path=source_path)))
            if adapter.kind == "assembly":
                assembly = get_assembly_spec(specimen.provider)
                if assembly is None:
                    raise AssertionError(f"declared assembly has no production spec: {adapter.identity}")
                sidecar_data = assembly.discover_sidecars([source_path])
                sessions = [assembly.enrich_session(session, sidecar_data) for session in sessions]
            projected = project_sessions(sessions)
            rendered = json.dumps(projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            results.append(
                RouteResult(adapter, input_hash, sidecar_hash, projected, hashlib.sha256(rendered).hexdigest())
            )
    report = DifferentialReport(tuple(results))
    report.assert_complete()
    return report


__all__ = [
    "AdapterDeclaration",
    "DifferentialReport",
    "OPERATIONAL_PATHS",
    "RouteResult",
    "SourceSpecimen",
    "declared_adapters",
    "project_sessions",
    "run_differential",
    "semantic_hash",
]

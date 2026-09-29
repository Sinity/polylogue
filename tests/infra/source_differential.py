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
from polylogue.sources.dispatch import is_stream_record_provider, parse_payload, parse_stream_payload
from polylogue.sources.origin_specs import OriginSpec, origin_specs
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.source_parsing import iter_source_sessions


@dataclass(frozen=True, slots=True)
class SourceSpecimen:
    """Provider bytes and sidecars shared by every isolated route."""

    provider: Provider
    raw_bytes: bytes
    filename: str = "specimen.jsonl"
    sidecars: Mapping[str, bytes] = field(default_factory=dict)

    @property
    def fallback_id(self) -> str:
        """Use the same file identity as the production source-walk route."""
        return Path(self.filename).stem


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
    declarations: tuple[AdapterDeclaration, ...]
    routes: tuple[RouteResult, ...]

    @property
    def adapters(self) -> tuple[str, ...]:
        return tuple(result.adapter.identity for result in self.routes)

    @property
    def canonical_hash(self) -> str:
        self.assert_complete()
        return self.routes[0].semantic_hash

    def _hashes(self) -> dict[str, str]:
        return {result.adapter.identity: result.semantic_hash for result in self.routes}

    def assert_complete(self) -> None:
        expected = {adapter.identity: adapter for adapter in self.declarations}
        identities = self.adapters
        if not expected or len(expected) != len(self.declarations):
            raise AssertionError("adapter declarations must be nonempty and unique")
        if len(identities) != len(set(identities)):
            raise AssertionError(f"duplicate adapter execution: {identities}")
        if set(identities) != set(expected):
            raise AssertionError(f"incomplete adapter execution: expected={tuple(expected)}, actual={identities}")
        for result in self.routes:
            if result.adapter != expected[result.adapter.identity]:
                raise AssertionError(f"adapter does not match its declaration: {result.adapter.identity}")
            if not result.sessions:
                raise AssertionError(f"adapter produced no sessions: {result.adapter.identity}")
        if len({(result.input_hash, result.sidecar_hash) for result in self.routes}) != 1:
            raise AssertionError("adapters did not compare the same specimen and sidecars")
        if len(set(self._hashes().values())) != 1:
            raise AssertionError(f"semantic route drift: {self._hashes()}")


def declared_adapters(
    specimen: SourceSpecimen,
    specs: Sequence[OriginSpec] | None = None,
) -> tuple[AdapterDeclaration, ...]:
    """Derive retained normalization routes directly from current declarations."""
    result: list[AdapterDeclaration] = []
    for spec in origin_specs() if specs is None else specs:
        if spec.lifecycle != "executable" or specimen.provider not in spec.provider_wires:
            continue
        provider = specimen.provider
        evidence = (*spec.parser_paths, *spec.assembly_paths)
        result.append(AdapterDeclaration(f"{spec.origin.value}:eager", spec.origin.value, provider, "eager", evidence))
        if spec.stream_parser_path is not None and is_stream_record_provider(specimen.filename, provider):
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
        if spec.assembly_spec_path is not None and get_assembly_spec(provider) is not None:
            result.append(
                AdapterDeclaration(
                    f"{spec.origin.value}:assembly", spec.origin.value, provider, "assembly", (spec.assembly_spec_path,)
                )
            )
    return tuple(result)


def project_sessions(sessions: Sequence[ParsedSession]) -> tuple[dict[str, object], ...]:
    """Project all semantic axes of parsed sessions into stable JSON values."""
    # Parser models own the field partition, including private attachment
    # coordinates. A second recursive classifier can only disagree with it.
    values = [cast(dict[str, object], session.model_dump(mode="json")) for session in sessions]
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
    declarations = declared_adapters(specimen, (current,))
    if not declarations:
        raise AssertionError(f"no executable declaration for {specimen.provider.value}")
    input_hash = hashlib.sha256(specimen.raw_bytes).hexdigest()
    sidecar_payload = json.dumps(dict(specimen.sidecars), sort_keys=True, default=str).encode()
    sidecar_hash = hashlib.sha256(sidecar_payload).hexdigest()
    results: list[RouteResult] = []
    with TemporaryDirectory(prefix="polylogue-source-differential-") as temporary:
        # Parent-relative sidecar discovery must stay inside this specimen.
        root = Path(temporary) / "inputs" / "source"
        source_path = root / specimen.filename
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(specimen.raw_bytes)
        for name, data in (specimen.sidecars or {}).items():
            sidecar_path = root / name
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_bytes(data)
        for adapter in declarations:
            if adapter.kind == "replay":
                # Source walking already discovers sidecars and enriches once.
                sessions = list(iter_source_sessions(Source(name=specimen.provider.value, path=source_path)))
            else:
                if adapter.kind == "streaming":
                    sessions = parse_stream_payload(
                        specimen.provider,
                        _iter_json_stream(io.BytesIO(specimen.raw_bytes), specimen.filename),
                        specimen.fallback_id,
                        source_path=str(source_path),
                    )
                else:
                    sessions = parse_payload(
                        specimen.provider,
                        _decode(specimen.raw_bytes),
                        specimen.fallback_id,
                        source_path=str(source_path),
                    )
                # These parser-entry routes stop before source-walk enrichment;
                # supply the same production assembly boundary exactly once.
                assembly = get_assembly_spec(specimen.provider)
                if assembly is not None:
                    sidecar_data = assembly.discover_sidecars([source_path])
                    sessions = [assembly.enrich_session(session, sidecar_data) for session in sessions]
            projected = project_sessions(sessions)
            rendered = json.dumps(projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            results.append(
                RouteResult(adapter, input_hash, sidecar_hash, projected, hashlib.sha256(rendered).hexdigest())
            )
    report = DifferentialReport(declarations, tuple(results))
    report.assert_complete()
    return report


__all__ = [
    "AdapterDeclaration",
    "DifferentialReport",
    "RouteResult",
    "SourceSpecimen",
    "declared_adapters",
    "project_sessions",
    "run_differential",
    "semantic_hash",
]

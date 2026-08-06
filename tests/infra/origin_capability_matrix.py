"""Typed loader and production-path helpers for the origin capability matrix."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, cast

from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import origin_from_provider
from polylogue.sources.origin_specs import ORIGIN_SPECS, OriginSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests" / "data" / "origin_capability_matrix.json"

UnsupportedReason = Literal["compatibility-only", "no-parser"]
WitnessRoute = Literal["detected", "source-hint"]
_LiteralValue = TypeVar("_LiteralValue", bound=str)


@dataclass(frozen=True, slots=True)
class ParserClaim:
    provider: Provider


@dataclass(frozen=True, slots=True)
class CapabilityWitness:
    origin: Origin
    parser_claims: tuple[ParserClaim, ...]
    fixture_path: str
    fixture_format: Literal["json", "jsonl"]
    fallback_id: str
    route: WitnessRoute


@dataclass(frozen=True, slots=True)
class UnsupportedReceipt:
    status: Literal["unsupported"]
    reason: UnsupportedReason
    detail: str


@dataclass(frozen=True, slots=True)
class CapabilityEntry:
    origin: Origin
    witnesses: tuple[CapabilityWitness, ...]
    unsupported: UnsupportedReceipt | None


@dataclass(frozen=True, slots=True)
class NegativeCase:
    name: str
    provider: Provider
    payload: object


@dataclass(frozen=True, slots=True)
class CollisionCase:
    name: str
    expected_provider: Provider
    payload: object
    fallback_id: str


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    entries: tuple[CapabilityEntry, ...]
    empty: tuple[NegativeCase, ...]
    partial: tuple[NegativeCase, ...]
    malformed: tuple[NegativeCase, ...]
    collisions: tuple[CollisionCase, ...]


def load_manifest() -> CapabilityManifest:
    """Load the committed matrix and validate it against live OriginSpec declarations."""
    return load_manifest_payload(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


def load_manifest_payload(payload: object) -> CapabilityManifest:
    """Validate one manifest payload, including parser-claim cardinality."""
    root = _mapping(payload, "manifest")
    if root.get("schema_version") != 1:
        raise ValueError("origin capability manifest schema_version must be 1")

    raw_entries = _list(root.get("entries"), "manifest.entries")
    entries = tuple(_entry(item) for item in raw_entries)
    origins = tuple(entry.origin for entry in entries)
    if len(set(origins)) != len(origins):
        raise ValueError("origin capability manifest contains duplicate origins")

    declared_origins = {spec.origin for spec in ORIGIN_SPECS}
    if set(origins) != set(Origin) or declared_origins != set(Origin):
        raise ValueError("origin capability manifest must cover every OriginSpec and public Origin")

    claimed_wires = tuple(
        witness.parser_claims[0].provider
        for entry in entries
        if entry.unsupported is None
        for witness in entry.witnesses
    )
    executable_wires = executable_provider_wires()
    if len(claimed_wires) != len(set(claimed_wires)):
        raise ValueError("origin capability manifest contains duplicate executable provider claims")
    if set(claimed_wires) != set(executable_wires):
        raise ValueError(
            "origin capability manifest provider claims must cover executable OriginSpec.provider_wires: "
            f"claimed={sorted(provider.value for provider in claimed_wires)!r}, "
            f"declared={sorted(provider.value for provider in executable_wires)!r}"
        )

    empty = tuple(_negative_case(item) for item in _list(root.get("empty"), "manifest.empty"))
    partial = tuple(_negative_case(item) for item in _list(root.get("partial"), "manifest.partial"))
    malformed = tuple(_negative_case(item) for item in _list(root.get("malformed"), "manifest.malformed"))
    collisions = tuple(_collision_case(item) for item in _list(root.get("collisions"), "manifest.collisions"))
    for family_name, cases in (("empty", empty), ("partial", partial), ("malformed", malformed)):
        if not cases:
            raise ValueError(f"origin capability manifest requires {family_name} negatives")
        for case in cases:
            if not _provider_has_executable_spec(case.provider):
                raise ValueError(f"{family_name} case {case.name!r} names an undeclared parser provider")
    if not malformed:
        raise ValueError("origin capability manifest requires malformed negatives")
    if not collisions:
        raise ValueError("origin capability manifest requires collision negatives")
    for collision in collisions:
        if not _provider_has_executable_spec(collision.expected_provider):
            raise ValueError(f"collision case {collision.name!r} names an undeclared parser provider")
    return CapabilityManifest(
        entries=entries,
        empty=empty,
        partial=partial,
        malformed=malformed,
        collisions=collisions,
    )


def load_witness_fixture(witness: CapabilityWitness) -> object:
    """Read one committed witness fixture without invoking a parser helper."""
    path = _repo_path(witness.fixture_path)
    if witness.fixture_format == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _entry(payload: object) -> CapabilityEntry:
    raw = _mapping(payload, "manifest entry")
    origin = _origin(raw.get("origin"))
    status = raw.get("status")
    raw_witnesses = _list(raw.get("witnesses"), f"{origin.value}.witnesses")
    witnesses = tuple(_witness(item, origin) for item in raw_witnesses)

    if status == "supported":
        if not witnesses:
            raise ValueError(f"{origin.value}: supported entry requires at least one witness")
        spec = _spec_for(origin)
        if spec.lifecycle != "executable":
            raise ValueError(f"{origin.value}: supported entry is not executable in OriginSpec")
        for witness in witnesses:
            if len(witness.parser_claims) != 1:
                raise ValueError(f"{origin.value}: each supported fixture requires exactly one parser claim")
            claim = witness.parser_claims[0]
            if claim.provider not in spec.provider_wires:
                raise ValueError(f"{origin.value}: parser claim is not declared by OriginSpec")
            if origin_from_provider(claim.provider) is not origin:
                raise ValueError(f"{origin.value}: parser claim maps to a different public origin")
        if raw.get("unsupported") is not None:
            raise ValueError(f"{origin.value}: supported entry cannot carry an unsupported receipt")
        return CapabilityEntry(origin, witnesses, None)

    if status != "unsupported":
        raise ValueError(f"{origin.value}: status must be supported or unsupported")
    if witnesses:
        raise ValueError(f"{origin.value}: unsupported entry cannot declare witnesses")
    spec = _spec_for(origin)
    if spec.lifecycle == "executable":
        raise ValueError(f"{origin.value}: executable OriginSpec cannot be silently unsupported")
    receipt = _unsupported_receipt(raw.get("unsupported"), origin)
    return CapabilityEntry(origin, (), receipt)


def _witness(payload: object, origin: Origin) -> CapabilityWitness:
    raw = _mapping(payload, f"{origin.value}.witness")
    raw_claims = _list(raw.get("parser_claims"), f"{origin.value}.witness.parser_claims")
    claims = tuple(
        ParserClaim(provider=_provider(_mapping(item, "parser claim").get("provider"))) for item in raw_claims
    )
    return CapabilityWitness(
        origin=origin,
        parser_claims=claims,
        fixture_path=_string(raw.get("fixture"), f"{origin.value}.witness.fixture"),
        fixture_format=cast(
            Literal["json", "jsonl"],
            _literal(raw.get("format"), ("json", "jsonl"), f"{origin.value}.witness.format"),
        ),
        fallback_id=_string(raw.get("fallback_id"), f"{origin.value}.witness.fallback_id"),
        route=cast(
            WitnessRoute,
            _literal(raw.get("route", "detected"), ("detected", "source-hint"), f"{origin.value}.witness.route"),
        ),
    )


def _unsupported_receipt(payload: object, origin: Origin) -> UnsupportedReceipt:
    raw = _mapping(payload, f"{origin.value}.unsupported")
    reason = cast(
        UnsupportedReason,
        _literal(raw.get("reason"), ("compatibility-only", "no-parser"), f"{origin.value}.unsupported.reason"),
    )
    detail = _string(raw.get("detail"), f"{origin.value}.unsupported.detail")
    return UnsupportedReceipt(status="unsupported", reason=reason, detail=detail)


def _negative_case(payload: object) -> NegativeCase:
    raw = _mapping(payload, "malformed case")
    return NegativeCase(
        name=_string(raw.get("name"), "malformed.name"),
        provider=_provider(raw.get("provider")),
        payload=raw.get("payload"),
    )


def _collision_case(payload: object) -> CollisionCase:
    raw = _mapping(payload, "collision case")
    return CollisionCase(
        name=_string(raw.get("name"), "collision.name"),
        expected_provider=_provider(raw.get("expected_provider")),
        payload=raw.get("payload"),
        fallback_id=_string(raw.get("fallback_id"), "collision.fallback_id"),
    )


def _spec_for(origin: Origin) -> OriginSpec:
    for spec in ORIGIN_SPECS:
        if spec.origin is origin:
            return spec
    raise ValueError(f"{origin.value}: missing OriginSpec")


def _provider_has_executable_spec(provider: Provider) -> bool:
    return any(spec.lifecycle == "executable" and provider in spec.provider_wires for spec in ORIGIN_SPECS)


def executable_provider_wires() -> tuple[Provider, ...]:
    """Return every provider wire declared by an executable OriginSpec."""
    return tuple(
        provider for spec in ORIGIN_SPECS if spec.lifecycle == "executable" for provider in spec.provider_wires
    )


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"fixture path escapes repository: {value!r}") from exc
    if not path.is_file():
        raise ValueError(f"fixture path does not exist: {value!r}")
    return path


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _origin(value: object) -> Origin:
    if not isinstance(value, str):
        raise ValueError("origin must be a string")
    try:
        return Origin(value)
    except ValueError as exc:
        raise ValueError(f"unknown origin: {value!r}") from exc


def _provider(value: object) -> Provider:
    if not isinstance(value, str):
        raise ValueError("provider must be a string")
    try:
        return Provider(value)
    except ValueError as exc:
        raise ValueError(f"unknown provider: {value!r}") from exc


def _literal(value: object, choices: tuple[_LiteralValue, ...], label: str) -> _LiteralValue:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{label} must be one of {choices!r}")
    return value


__all__ = [
    "CapabilityEntry",
    "CapabilityManifest",
    "CapabilityWitness",
    "CollisionCase",
    "MANIFEST_PATH",
    "NegativeCase",
    "ParserClaim",
    "UnsupportedReceipt",
    "WitnessRoute",
    "load_manifest",
    "load_manifest_payload",
    "load_witness_fixture",
    "executable_provider_wires",
]

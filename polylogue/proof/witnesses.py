"""Privacy-aware witness metadata for proof and regression artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from polylogue.lib.json import JSONDocument, loads, require_json_document

WITNESS_SCHEMA_VERSION = 1
LOCAL_WITNESS_INBOX = Path(".local/witnesses/new")
COMMITTED_WITNESS_DIR = Path("tests/witnesses")

WitnessOrigin = Literal["golden-surface-snapshot", "live-derived", "synthetic", "external", "regression"]
MinimizationStatus = Literal["raw", "minimized", "rejected", "not_applicable"]
PrivateMaterialState = Literal["not_observed", "observed"]
WitnessState = Literal["discovered", "minimized", "committed", "exercised", "retired"]
PrivacyClassification = Literal["synthetic", "redacted", "public"]


def _string_tuple(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(str(value) for value in values if str(value).strip())


@dataclass(frozen=True, slots=True)
class PrivacyRecord:
    """How private material was handled while creating a witness."""

    private_material: PrivateMaterialState
    transformed: bool = False
    redacted: bool = False
    discarded: bool = False
    retained: bool = False
    notes: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: JSONDocument) -> PrivacyRecord:
        private_material_raw = str(payload.get("private_material", "not_observed"))
        private_material: PrivateMaterialState = "observed" if private_material_raw == "observed" else "not_observed"
        return cls(
            private_material=private_material,
            transformed=bool(payload.get("transformed", False)),
            redacted=bool(payload.get("redacted", False)),
            discarded=bool(payload.get("discarded", False)),
            retained=bool(payload.get("retained", False)),
            notes=_string_tuple(payload.get("notes", [])),
        )

    def to_payload(self) -> JSONDocument:
        return {
            "private_material": self.private_material,
            "transformed": self.transformed,
            "redacted": self.redacted,
            "discarded": self.discarded,
            "retained": self.retained,
            "notes": list(self.notes),
        }

    def live_commit_errors(self) -> tuple[str, ...]:
        if self.private_material == "not_observed":
            return ()
        errors: list[str] = []
        if self.retained:
            errors.append("live-derived private material cannot be retained in committed witnesses")
        if not (self.transformed or self.redacted or self.discarded):
            errors.append("live-derived private material must be transformed, redacted, or discarded")
        return tuple(errors)


@dataclass(frozen=True, slots=True)
class WitnessLifecycle:
    """State-machine tracking for a witness from discovery to retirement."""

    state: WitnessState = "discovered"
    discovered_at: str = ""
    minimized_at: str | None = None
    committed_at: str | None = None
    last_exercised_at: str | None = None
    retired_at: str | None = None
    retirement_reason: str | None = None

    @classmethod
    def new(cls) -> WitnessLifecycle:
        return cls(state="discovered", discovered_at=datetime.now(tz=timezone.utc).isoformat())

    def transition(self, state: WitnessState) -> WitnessLifecycle:
        now = datetime.now(tz=timezone.utc).isoformat()
        kwargs: dict[str, object] = {
            "state": state,
            "discovered_at": self.discovered_at,
            "minimized_at": self.minimized_at,
            "committed_at": self.committed_at,
            "last_exercised_at": self.last_exercised_at,
            "retired_at": self.retired_at,
            "retirement_reason": self.retirement_reason,
        }
        if state == "minimized":
            kwargs["minimized_at"] = now
        elif state == "committed":
            kwargs["committed_at"] = now
        elif state == "exercised":
            kwargs["last_exercised_at"] = now
        elif state == "retired":
            kwargs["retired_at"] = now
        return WitnessLifecycle(
            state=state,
            discovered_at=str(kwargs["discovered_at"]),
            minimized_at=kwargs.get("minimized_at"),  # type: ignore[arg-type]
            committed_at=kwargs.get("committed_at"),  # type: ignore[arg-type]
            last_exercised_at=kwargs.get("last_exercised_at"),  # type: ignore[arg-type]
            retired_at=kwargs.get("retired_at"),  # type: ignore[arg-type]
            retirement_reason=kwargs.get("retirement_reason"),  # type: ignore[arg-type]
        )

    @classmethod
    def from_payload(cls, payload: JSONDocument) -> WitnessLifecycle:
        return cls(
            state=_witness_state(payload.get("state", "discovered")),
            discovered_at=str(payload.get("discovered_at", "")),
            minimized_at=str(p) if (p := payload.get("minimized_at")) else None,
            committed_at=str(p) if (p := payload.get("committed_at")) else None,
            last_exercised_at=str(p) if (p := payload.get("last_exercised_at")) else None,
            retired_at=str(p) if (p := payload.get("retired_at")) else None,
            retirement_reason=str(r) if (r := payload.get("retirement_reason")) else None,
        )

    def to_payload(self) -> JSONDocument:
        return {
            "state": self.state,
            "discovered_at": self.discovered_at,
            "minimized_at": self.minimized_at,
            "committed_at": self.committed_at,
            "last_exercised_at": self.last_exercised_at,
            "retired_at": self.retired_at,
            "retirement_reason": self.retirement_reason,
        }


@dataclass(frozen=True, slots=True)
class WitnessMetadata:
    """A replayable witness lifecycle record."""

    witness_id: str
    path: str
    origin: WitnessOrigin
    provenance: JSONDocument
    preserved_semantic_facts: tuple[str, ...]
    minimization_status: MinimizationStatus
    privacy: PrivacyRecord | None = None
    privacy_classification: PrivacyClassification | None = None
    lifecycle: WitnessLifecycle | None = None
    committed: bool = True
    known_failing: bool = False
    xfail_strict: bool = False
    rejection_reason: str | None = None
    notes: tuple[str, ...] = ()
    schema_version: int = WITNESS_SCHEMA_VERSION

    @classmethod
    def from_payload(cls, payload: JSONDocument) -> WitnessMetadata:
        schema_version_raw = payload.get("schema_version")
        schema_version = schema_version_raw if isinstance(schema_version_raw, int) else 0
        if schema_version != WITNESS_SCHEMA_VERSION:
            raise ValueError(f"unsupported witness schema_version: {schema_version}")
        privacy_payload = payload.get("privacy")
        privacy = (
            PrivacyRecord.from_payload(require_json_document(privacy_payload, context="witness privacy"))
            if privacy_payload is not None
            else None
        )
        lifecycle_payload = payload.get("lifecycle")
        lifecycle = (
            WitnessLifecycle.from_payload(require_json_document(lifecycle_payload, context="witness lifecycle"))
            if lifecycle_payload is not None
            else None
        )
        privacy_class_raw = payload.get("privacy_classification")
        privacy_class: PrivacyClassification | None = None
        if privacy_class_raw is not None:
            pc = str(privacy_class_raw)
            if pc in ("synthetic", "redacted", "public"):
                privacy_class = cast(PrivacyClassification, pc)
        return cls(
            witness_id=str(payload["witness_id"]),
            path=str(payload["path"]),
            origin=_origin(payload.get("origin")),
            provenance=require_json_document(payload.get("provenance", {}), context="witness provenance"),
            preserved_semantic_facts=_string_tuple(payload.get("preserved_semantic_facts", [])),
            minimization_status=_minimization_status(payload.get("minimization_status")),
            privacy=privacy,
            privacy_classification=privacy_class,
            lifecycle=lifecycle,
            committed=bool(payload.get("committed", True)),
            known_failing=bool(payload.get("known_failing", False)),
            xfail_strict=bool(payload.get("xfail_strict", False)),
            rejection_reason=(
                str(payload["rejection_reason"]) if payload.get("rejection_reason") is not None else None
            ),
            notes=_string_tuple(payload.get("notes", [])),
            schema_version=schema_version,
        )

    @classmethod
    def read(cls, path: Path) -> WitnessMetadata:
        return cls.from_payload(require_json_document(loads(path.read_text(encoding="utf-8"))))

    def to_payload(self) -> JSONDocument:
        return {
            "schema_version": self.schema_version,
            "witness_id": self.witness_id,
            "path": self.path,
            "origin": self.origin,
            "committed": self.committed,
            "provenance": self.provenance,
            "preserved_semantic_facts": list(self.preserved_semantic_facts),
            "minimization_status": self.minimization_status,
            "privacy": self.privacy.to_payload() if self.privacy is not None else None,
            "privacy_classification": self.privacy_classification,
            "lifecycle": self.lifecycle.to_payload() if self.lifecycle is not None else None,
            "known_failing": self.known_failing,
            "xfail_strict": self.xfail_strict,
            "rejection_reason": self.rejection_reason,
            "notes": list(self.notes),
        }

    def validation_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.committed and self.origin == "live-derived":
            if self.privacy is None:
                errors.append("committed live-derived witnesses require privacy metadata")
            else:
                errors.extend(self.privacy.live_commit_errors())
        if self.committed and self.privacy_classification is None:
            errors.append("committed witnesses require a privacy_classification (synthetic, redacted, or public)")
        if not self.preserved_semantic_facts:
            errors.append("witness must declare preserved semantic facts")
        if self.committed and self.minimization_status == "raw":
            errors.append("committed witnesses must be minimized, rejected, or explicitly not applicable")
        if self.known_failing and not self.rejection_reason:
            errors.append("known failing witnesses require an explicit rejection_reason")
        return tuple(errors)


def _origin(value: object) -> WitnessOrigin:
    allowed: tuple[WitnessOrigin, ...] = (
        "golden-surface-snapshot",
        "live-derived",
        "synthetic",
        "external",
        "regression",
    )
    text = str(value)
    if text in allowed:
        return cast(WitnessOrigin, text)
    raise ValueError(f"unsupported witness origin: {text}")


def _witness_state(value: object) -> WitnessState:
    allowed: tuple[WitnessState, ...] = ("discovered", "minimized", "committed", "exercised", "retired")
    text = str(value)
    if text in allowed:
        return cast(WitnessState, text)
    raise ValueError(f"unsupported witness state: {text}")


def _minimization_status(value: object) -> MinimizationStatus:
    allowed: tuple[MinimizationStatus, ...] = ("raw", "minimized", "rejected", "not_applicable")
    text = str(value)
    if text in allowed:
        return cast(MinimizationStatus, text)
    raise ValueError(f"unsupported witness minimization_status: {text}")


def load_witnesses(paths: Iterable[Path]) -> tuple[WitnessMetadata, ...]:
    return tuple(WitnessMetadata.read(path) for path in sorted(paths))


def load_committed_witnesses(root: Path = COMMITTED_WITNESS_DIR) -> tuple[WitnessMetadata, ...]:
    if not root.exists():
        return ()
    return load_witnesses(root.glob("*.witness.json"))


def committed_witness_errors(witnesses: Iterable[WitnessMetadata]) -> JSONDocument:
    errors: JSONDocument = {}
    for witness in witnesses:
        witness_errors = witness.validation_errors()
        if witness_errors:
            errors[witness.witness_id] = list(witness_errors)
    return errors


__all__ = [
    "COMMITTED_WITNESS_DIR",
    "LOCAL_WITNESS_INBOX",
    "MinimizationStatus",
    "PrivacyClassification",
    "PrivacyRecord",
    "PrivateMaterialState",
    "WITNESS_SCHEMA_VERSION",
    "WitnessLifecycle",
    "WitnessMetadata",
    "WitnessOrigin",
    "WitnessState",
    "committed_witness_errors",
    "load_committed_witnesses",
    "load_witnesses",
]

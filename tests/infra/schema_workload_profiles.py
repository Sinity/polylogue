"""Neutral package evidence for controls of the actual shared workload driver."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.generation.models import _PackageAccumulator, _UnitMembership
from polylogue.schemas.generation.workload_profiles import build_package_workload_profile
from polylogue.schemas.observation import SchemaUnit
from polylogue.schemas.workload_profile_rows import WorkloadProfile


def inferred_record_profile(
    provider: str, schema: JSONDocument, samples: list[JSONDocument], *, element: str
) -> WorkloadProfile:
    """Run the real journal aggregation over a declared neutral physical scope."""
    unit = SchemaUnit(cluster_payload={}, schema_samples=samples, artifact_kind=element, bundle_scope="neutral-fixture")
    return build_package_workload_profile(
        provider=provider,
        version="v1",
        package=_PackageAccumulator(
            provider=provider,
            anchor_family_id="neutral-family",
            anchor_kind=element,
            memberships=[_UnitMembership(unit, "neutral-family")],
        ),
        element_schemas={element: schema},
        privacy_policy="standard",
    )


def inferred_physical_profile(
    provider: str, schemas: dict[str, JSONDocument], units: list[SchemaUnit]
) -> WorkloadProfile:
    """Retain declared neutral files through the original package journal."""
    return build_package_workload_profile(
        provider=provider,
        version="v1",
        package=_PackageAccumulator(
            provider=provider,
            anchor_family_id="neutral-family",
            anchor_kind="session_record_stream",
            memberships=[_UnitMembership(unit, "neutral-family") for unit in units],
        ),
        element_schemas=schemas,
        privacy_policy="standard",
    )


def inferred_codex_session_profile(
    schema: JSONDocument, samples: list[JSONDocument], *, native_id: str
) -> tuple[JSONDocument, WorkloadProfile]:
    """Declare neutral wire branches before drawing original constrained fields.

    These operands are authored synthetic evidence. Header and public wire
    tokens come from the supplied neutral records; field constraints remain
    on the supplied schema, and the same journal infers the physical profile.
    """
    records: list[JSONDocument] = (
        samples
        if any(row.get("type") == "session_meta" for row in samples)
        else [{"type": "session_meta", "payload": {"id": native_id}}, *samples]
    )
    branches: list[JSONValue] = []
    for row in records:
        branch = deepcopy(schema)
        fields = cast(JSONDocument, branch.setdefault("properties", {}))
        fields["type"] = {"type": "string", "x-polylogue-values": [row["type"]]}
        branch["required"] = list(row)
        payload = cast(JSONDocument, row["payload"])
        payload_schema = cast(JSONDocument, fields.setdefault("payload", {"type": "object"}))
        payload_fields = cast(JSONDocument, payload_schema.setdefault("properties", {}))
        for name in ("type", "role"):
            if name in payload:
                payload_fields[name] = {"type": "string", "x-polylogue-values": [payload[name]]}
        if row["type"] == "session_meta":
            payload_fields["id"] = {"type": "string", "x-polylogue-values": [payload["id"]]}
        payload_schema["required"] = list(payload)
        if branch not in branches:
            branches.append(branch)
    selected: JSONDocument = {"oneOf": branches}
    return selected, inferred_record_profile("codex", selected, records, element="session_record_stream")

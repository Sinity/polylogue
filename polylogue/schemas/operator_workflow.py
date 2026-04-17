"""Small public root for typed schema/evidence operator workflows."""

from __future__ import annotations

from polylogue.schemas.operator_annotations import collect_annotation_summary
from polylogue.schemas.operator_inference import (
    audit_schemas as _audit_schemas,
)
from polylogue.schemas.operator_inference import (
    compare_schema_versions as _compare_schema_versions,
)
from polylogue.schemas.operator_inference import (
    infer_schema as _infer_schema,
)
from polylogue.schemas.operator_inference import (
    list_schemas as _list_schemas,
)
from polylogue.schemas.operator_inference import (
    promote_schema_cluster as _promote_schema_cluster,
)
from polylogue.schemas.operator_models import SchemaAnnotationSummary
from polylogue.schemas.operator_resolution import (
    explain_schema as _explain_schema,
)
from polylogue.schemas.operator_resolution import (
    resolve_schema_payload as _resolve_schema_payload,
)
from polylogue.schemas.operator_verification import (
    list_artifact_cohorts as _list_artifact_cohorts,
)
from polylogue.schemas.operator_verification import (
    list_artifact_observations as _list_artifact_observations,
)
from polylogue.schemas.operator_verification import (
    run_artifact_proof as _run_artifact_proof,
)
from polylogue.schemas.operator_verification import (
    run_schema_verification as _run_schema_verification,
)

infer_schema = _infer_schema
list_schemas = _list_schemas
compare_schema_versions = _compare_schema_versions
promote_schema_cluster = _promote_schema_cluster
explain_schema = _explain_schema
resolve_schema_payload = _resolve_schema_payload
audit_schemas = _audit_schemas
run_schema_verification = _run_schema_verification
run_artifact_proof = _run_artifact_proof
list_artifact_observations = _list_artifact_observations
list_artifact_cohorts = _list_artifact_cohorts


def _collect_annotation_summary(schema: dict[str, object]) -> SchemaAnnotationSummary:
    return collect_annotation_summary(schema)

"""Exact canonical batch evidence and bounded authorization operands."""

from __future__ import annotations

import hashlib
from typing import Any

from polylogue.annotations.batch import AnnotationBatch
from polylogue.annotations.import_spill import AnnotationImportSpill
from polylogue.core.json import JSONDocument
from polylogue.storage.sqlite.connection_profile import scratch_connection_context


def test_spilled_batch_matches_original_canonical_provenance_and_binds_row_values() -> None:
    fields: dict[str, Any] = {
        "batch_id": "neutral",
        "schema_id": "test.import",
        "schema_version": 1,
        "target_ref": "session:neutral",
        "source_result_ref": "result-set:neutral",
        "actor_ref": "agent:neutral",
        "model_ref": "agent:model",
        "prompt_ref": "block:prompt:0",
    }
    failure: JSONDocument = {"line": 2, "row_key": None, "errors": ["e\u0301"]}
    header = AnnotationBatch(**fields, total_count=0, valid_count=0, invalid_count=0, abstained_count=0)
    with scratch_connection_context(prefix="annotation-spill-law-", filename="scratch.sqlite") as connection:
        batch = AnnotationImportSpill(connection, header)
        assert batch.ref_resolution("session:missing") is None
        batch.record_ref_resolution("session:missing", False)
        assert batch.ref_resolution("session:missing") is False
        batch.record_ref_resolution("session:present", True)
        assert batch.ref_resolution("session:present") is True
        batch.append_row(1, "one", '{"value":"first"}', "assertion:nfd-e\u0301", None, False)
        batch.append_failure(2, failure)
        batch.seal()
        original = AnnotationBatch(
            **fields,
            total_count=2,
            valid_count=1,
            invalid_count=1,
            abstained_count=0,
            assertion_refs=("assertion:nfd-e\u0301",),
            validation_failures=(failure,),
        )
        assert batch.provenance_digest() == hashlib.sha256(original.canonical_provenance_bytes()).hexdigest()
        row_digest = batch.rows_digest()
        connection.execute("UPDATE annotation_import_rows SET row_json=?", ('{"value":"changed"}',))
        assert batch.rows_digest() != row_digest
        # Values changing under identical row refs cannot pass the authorization
        # binding merely because the complete provenance roster still agrees.
        assert batch.provenance_digest() == hashlib.sha256(original.canonical_provenance_bytes()).hexdigest()

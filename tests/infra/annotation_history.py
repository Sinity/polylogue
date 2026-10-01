"""Synthetic post-floor user rows with public, exact historical definitions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Literal

from polylogue.core.enums import AssertionKind
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion

AnnotationHistoryVariant = Literal["pre_5314", "post_5314"]


def historical_seed_definitions(variant: AnnotationHistoryVariant) -> list[dict[str, str]]:
    path = Path(__file__).parents[1] / "fixtures/annotations/post_floor_seed_v1.json"
    return json.loads(path.read_text())[variant]["definitions"]


def seed_annotation_history(root: Path, variant: AnnotationHistoryVariant) -> None:
    """Install historical evidence, without passing it through new-write admission.

    Definitions were extracted from the public constructor declarations in
    217e9e9827aff2321352fca7ec8b81fc145b9d69 and dafcfc61296413a8bd69e5be4e09606c3aad8c31.
    Rows and batch identities are neutral synthetic historical observations.
    """
    initialize_active_archive_root(root)
    with sqlite3.connect(root / "user.db") as conn:
        conn.execute("DELETE FROM annotation_schemas WHERE schema_id LIKE 'seed.%'")
        for item in historical_seed_definitions(variant):
            definition = json.loads(item["definition_json"])
            schema_id = definition["schema_id"]
            target_kind = next(
                (kind for kind in definition["target_ref_kinds"] if kind in {"phase", "work_event"}),
                definition["target_ref_kinds"][0],
            )
            target = f"{target_kind}:historical-evidence"
            batch_id = f"historical-{schema_id}"
            assertion_id = f"label-{schema_id}"
            conn.execute(
                "INSERT INTO annotation_schemas VALUES (?, 1, ?, ?, 123)",
                (schema_id, item["definition_json"], item["definition_sha256"]),
            )
            # The former live target may no longer be writable; fixture setup
            # retains its original target text after the common assertion shape.
            upsert_assertion(
                conn,
                assertion_id=assertion_id,
                scope_ref=f"annotation-batch:{batch_id}",
                target_ref="session:historical-evidence",
                kind=AssertionKind.ANNOTATION,
                key="old-label",
                value={"_schema": f"{schema_id}@v1", "_batch": f"annotation-batch:{batch_id}", "abstain": True},
                author_ref="agent:historical-labeler",
                author_kind="agent",
                evidence_refs=["session:historical-evidence"],
                now_ms=456,
            )
            conn.execute("UPDATE assertions SET target_ref = ? WHERE assertion_id = ?", (target, assertion_id))
            conn.execute(
                """INSERT INTO annotation_batches (
                    batch_id, schema_id, schema_version, target_ref, source_result_ref,
                    actor_ref, model_ref, prompt_ref, total_count, valid_count, invalid_count,
                    abstained_count, assertion_refs_json, validation_failures_json, metadata_json, created_at_ms
                ) VALUES (?, ?, 1, ?, 'result-set:historical-evidence', 'agent:historical-labeler',
                    'agent:historical-model', 'block:historical-prompt:0', 1, 1, 0, 1, ?, '[]', '{}', 456)""",
                (batch_id, schema_id, target, json.dumps([f"assertion:{assertion_id}"])),
            )


def historical_user_rows(root: Path) -> dict[str, list[tuple[object, ...]]]:
    with sqlite3.connect(root / "user.db") as conn:
        return {
            "schemas": conn.execute(
                "SELECT * FROM annotation_schemas WHERE schema_version = 1 ORDER BY schema_id"
            ).fetchall(),
            "batches": conn.execute("SELECT * FROM annotation_batches ORDER BY batch_id").fetchall(),
            "assertions": conn.execute("SELECT * FROM assertions ORDER BY assertion_id").fetchall(),
        }

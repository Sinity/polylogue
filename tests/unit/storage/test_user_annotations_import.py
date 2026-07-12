"""Fresh-interpreter import contract for the durable annotation writer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_user_annotations_imports_before_annotations_write_in_fresh_interpreter() -> None:
    """The storage module must not depend on package import order."""

    repo_root = Path(__file__).parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from polylogue.storage.sqlite.archive_tiers.user_annotations "
                "import persist_annotation_batch, persist_annotation_schema; "
                "from polylogue.annotations.write import upsert_annotation_assertion; "
                "assert callable(persist_annotation_batch); "
                "assert callable(persist_annotation_schema); "
                "assert callable(upsert_annotation_assertion)"
            ),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr

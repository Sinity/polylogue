"""Meta-validation for packaged provider schemas.

Ensures all committed provider schemas are valid Draft 2020-12 schemas.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


def _provider_schema_paths() -> list[Path]:
    schema_dir = Path(__file__).resolve().parents[3] / "polylogue" / "schemas" / "providers"
    return sorted(schema_dir.glob("*.schema.json.gz"))


@pytest.mark.parametrize("schema_path", _provider_schema_paths(), ids=lambda p: p.name)
def test_packaged_provider_schema_is_valid_draft202012(schema_path: Path) -> None:
    schema = json.loads(gzip.decompress(schema_path.read_bytes()).decode("utf-8"))
    Draft202012Validator.check_schema(schema)

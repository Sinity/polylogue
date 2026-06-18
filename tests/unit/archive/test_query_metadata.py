from __future__ import annotations

import json
import subprocess
import sys


def test_query_completions_do_not_import_expression_parser() -> None:
    script = """
import json
import sys
from polylogue.archive.query.completions import query_field_candidates

candidates = query_field_candidates("re")
print(json.dumps({
    "values": [candidate.value for candidate in candidates],
    "expression_loaded": "polylogue.archive.query.expression" in sys.modules,
}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert "repo" in payload["values"]
    assert payload["expression_loaded"] is False

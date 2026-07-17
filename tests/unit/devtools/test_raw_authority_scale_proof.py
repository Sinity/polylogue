from __future__ import annotations

from pathlib import Path
from typing import cast

from devtools.raw_authority_scale_proof import run_raw_authority_scale_proof


def test_raw_authority_scale_proof_reaches_two_matching_quiescent_censuses(tmp_path: Path) -> None:
    payload = run_raw_authority_scale_proof(tmp_path, components=3, raws=5, pass_limit=2)

    assert payload["requested_shape"] == {"components": 3, "raws": 5, "pass_limit": 2}
    digests = cast(list[str], payload["fixed_point_digests"])
    passes = cast(list[dict[str, object]], payload["passes"])
    receipt = cast(dict[str, object], payload["receipt"])
    assert len(digests) == 2
    assert digests[0] == digests[1]
    assert passes[-1]["candidate_count"] == 0
    assert receipt["status"] == "succeeded"

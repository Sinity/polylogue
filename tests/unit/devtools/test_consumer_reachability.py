"""Regression coverage for polylogue-gb4e consumer-reachability V3/R2."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devtools.consumer_reachability import ConsumerReachabilityError, _authority, _waivers, check


def test_polylogue_gb4e_v3_r2_production_route_has_current_authority() -> None:
    """The owning command executes against the current production checkout."""
    root = Path(__file__).parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    report = check(root, base=head, head=head)
    assert report.ok
    assert report.base == report.head


@pytest.mark.parametrize("value", ["HEAD", "not-a-sha", "0" * 39])
def test_polylogue_gb4e_rejects_malformed_authority(value: str) -> None:
    with pytest.raises(ConsumerReachabilityError, match="malformed authority"):
        _authority(Path(__file__).parents[3], value, value)


def test_polylogue_gb4e_rejects_stale_head() -> None:
    root = Path(__file__).parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=root, text=True).strip()
    with pytest.raises(ConsumerReachabilityError, match="stale authority head"):
        _authority(root, head, parent)


def test_polylogue_gb4e_rejects_duplicate_waiver(tmp_path: Path) -> None:
    body = tmp_path / "body.txt"
    body.write_text(
        "consumer-reachability-waiver: polylogue/x.py first reason is sufficiently explicit\n"
        "consumer-reachability-waiver: polylogue/x.py second reason is sufficiently explicit\n",
        encoding="utf-8",
    )
    with pytest.raises(ConsumerReachabilityError, match="duplicate waiver"):
        _waivers(body)

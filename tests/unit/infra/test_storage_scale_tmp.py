"""The storage_scale marker moves a test's private tree off the tmpfs lane."""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools import verify_runs

_NVME_ROOT = verify_runs.DEFAULT_PYTEST_BASETEMP_ROOT / "storage-scale"


@pytest.mark.storage_scale
def test_storage_scale_tmp_path_is_redirected_to_scratch(tmp_path: Path) -> None:
    if not verify_runs.DEFAULT_PYTEST_BASETEMP_ROOT.parent.is_dir():
        pytest.skip("NVMe scratch root unavailable (cloud sandbox); fallback path covered below")
    assert tmp_path.is_dir()
    assert tmp_path.is_relative_to(_NVME_ROOT)


def test_unmarked_tmp_path_stays_on_the_run_basetemp(tmp_path: Path) -> None:
    assert tmp_path.is_dir()
    assert not tmp_path.is_relative_to(_NVME_ROOT)

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.pipeline import parsed_tree_size
from polylogue.sources.parsers.base_models import ParsedContentBlock, ParsedMessage, ParsedSession


def test_cgroup_v2_limit_paths_include_nested_service_ancestry(tmp_path: Path) -> None:
    cgroup_root = tmp_path / "cgroup"
    service = cgroup_root / "user.slice" / "user-1000.slice" / "worker.service"
    service.mkdir(parents=True)
    membership = tmp_path / "self-cgroup"
    membership.write_text("0::/user.slice/user-1000.slice/worker.service\n")

    paths = parsed_tree_size._cgroup_memory_limit_paths(
        cgroup_v2_root=cgroup_root,
        cgroup_v1_root=tmp_path / "cgroup-v1-memory",
        proc_cgroup_path=membership,
    )

    assert service / "memory.max" in paths
    assert cgroup_root / "user.slice" / "memory.max" in paths
    assert cgroup_root / "memory.max" in paths


def test_cgroup_v1_paths_use_combined_memory_controller_mount(tmp_path: Path) -> None:
    mount = tmp_path / "cgroup-cpu-memory"
    worker = mount / "worker.service"
    worker.mkdir(parents=True)
    membership = tmp_path / "self-cgroup"
    membership.write_text("5:cpu,memory:/slice/worker.service\n")
    mountinfo = tmp_path / "self-mountinfo"
    mountinfo.write_text(f"42 25 0:42 /slice {mount} rw - cgroup cgroup rw,cpu,memory\n")

    paths = parsed_tree_size._cgroup_memory_limit_paths(
        cgroup_v2_root=tmp_path / "cgroup-v2",
        cgroup_v1_root=tmp_path / "canonical-memory",
        proc_cgroup_path=membership,
        proc_mountinfo_path=mountinfo,
    )

    assert worker / "memory.limit_in_bytes" in paths
    assert mount / "memory.limit_in_bytes" in paths


def test_effective_memory_caps_host_ram_to_nested_cgroup_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    limit_path = tmp_path / "worker-memory.max"
    limit_path.write_text(str(10 * 1024**3))
    monkeypatch.setattr(os, "sysconf", lambda key: 32 * 1024**3 if key == "SC_PHYS_PAGES" else 1)
    monkeypatch.setattr(parsed_tree_size, "_cgroup_memory_limit_paths", lambda: (limit_path,))

    assert parsed_tree_size.effective_physical_memory_bytes() == 10 * 1024**3


def _session_with_text(native_id: str, text_len: int) -> ParsedSession:
    """One session, one message, one block, all carrying ``text_len`` chars.

    Deliberately constructed directly (not run through a provider parser) so
    tree size is exactly controlled for size-comparison assertions below.
    """
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        messages=[
            ParsedMessage(
                provider_message_id=f"{native_id}-m0",
                role=Role.USER,
                text="x" * text_len,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="x" * text_len)],
            )
        ],
    )


def _deep_size_bytes(obj: object, seen: set[int] | None = None) -> int:
    """Manual recursive object-graph size walk -- a from-scratch equivalent
    of ``pympler.asizeof.asizeof`` (not a repo dependency) used ONLY here,
    to calibrate/verify ``estimate_parsed_tree_bytes``'s cheap structural
    formula against ground truth. Never used on the production hot path --
    that is exactly the cost ``estimate_parsed_tree_bytes`` exists to avoid."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += _deep_size_bytes(key, seen)
            size += _deep_size_bytes(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += _deep_size_bytes(item, seen)
    elif hasattr(obj, "__dict__"):
        size += _deep_size_bytes(obj.__dict__, seen)
    return size


def test_estimate_parsed_tree_bytes_is_monotone_in_content_size() -> None:
    """Production dependency: ``estimate_parsed_tree_bytes`` itself. A
    mutation that stopped reading ``message.text``/``block.text`` lengths
    (e.g. hardcoding a fixed per-session constant) would make these three
    estimates equal instead of strictly increasing, and this would fail."""
    small = parsed_tree_size.estimate_parsed_tree_bytes([_session_with_text("s", 10)])
    medium = parsed_tree_size.estimate_parsed_tree_bytes([_session_with_text("s", 1_000)])
    large = parsed_tree_size.estimate_parsed_tree_bytes([_session_with_text("s", 100_000)])

    assert small < medium < large

    # Also monotone in session COUNT at fixed per-session size.
    one_session = parsed_tree_size.estimate_parsed_tree_bytes([_session_with_text("s", 500)])
    two_sessions = parsed_tree_size.estimate_parsed_tree_bytes(
        [_session_with_text("a", 500), _session_with_text("b", 500)]
    )
    assert two_sessions > one_session


def test_estimate_parsed_tree_bytes_within_3x_of_deep_measurement() -> None:
    """Production dependency: the calibrated constants
    ``_ESTIMATOR_BYTES_PER_CHAR``/``_ESTIMATOR_OBJECT_OVERHEAD_BYTES`` inside
    ``estimate_parsed_tree_bytes``. If either constant were set to a wildly
    wrong value (e.g. 0, or 1000x too small), the estimate would fall
    outside a 3x band of ``_deep_size_bytes``'s independent ground-truth
    measurement on the same synthetic session and this would fail -- this is
    the test that keeps the calibration comment in parsed_tree_size.py honest
    against actual measured memory, not just internally self-consistent."""
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="calibration",
        messages=[
            ParsedMessage(
                provider_message_id=f"m{i}",
                role=Role.USER,
                text="x" * 300,
                blocks=[
                    ParsedContentBlock(type=BlockType.TEXT, text="y" * 200),
                    ParsedContentBlock(type=BlockType.TEXT, text="z" * 200),
                ],
            )
            for i in range(50)
        ],
    )

    estimate = parsed_tree_size.estimate_parsed_tree_bytes([session])
    deep = _deep_size_bytes(session)

    assert deep / 3 <= estimate <= deep * 3

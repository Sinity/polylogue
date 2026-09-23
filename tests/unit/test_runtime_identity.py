from __future__ import annotations

import sys

from devtools.measurement_receipts import host_fingerprint
from polylogue.runtime import runtime_identity


def test_runtime_identity_records_context_inheritance_flag() -> None:
    identity = runtime_identity()
    expected = getattr(sys.flags, "thread_inherit_context", None)

    assert identity.thread_inherit_context == (None if expected is None else bool(expected))
    assert identity.to_dict()["thread_inherit_context"] == identity.thread_inherit_context


def test_benchmark_host_fingerprint_records_context_inheritance_flag() -> None:
    fingerprint = host_fingerprint()
    expected = getattr(sys.flags, "thread_inherit_context", None)

    assert fingerprint["thread_inherit_context"] == (None if expected is None else bool(expected))

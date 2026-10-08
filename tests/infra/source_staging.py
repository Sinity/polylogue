"""Small production staging receipts for source-binding regressions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from polylogue.sources.source_staging import SourceInputBinding, bind_staged_member, read_staging_receipt


def staged_members(slot: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    members: list[dict[str, Any]] = []
    receipt = read_staging_receipt(slot, on_member=members.append, check_stop=lambda: None)
    assert receipt is not None
    return receipt, members


@contextmanager
def single_staged_binding(slot: Path) -> Iterator[SourceInputBinding]:
    receipt, members = staged_members(slot)
    assert len(members) == 1
    with bind_staged_member(slot, members[0], receipt) as binding:
        yield binding

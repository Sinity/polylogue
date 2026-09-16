"""No connection profile is defined twice.

``ISOLATED_TIER_WRITE_PROFILE`` was bound twice at module scope with different
timeout and autocheckpoint policy. Python kept the second, so the first --
together with the comment justifying it -- was dead code that read as the live
policy. The same shadowing is possible for every other profile constant in this
module, so pin the whole module rather than the one name.

Anti-vacuity: re-adding any second top-level assignment to a name this module
already binds (or a duplicate ``__all__`` entry) makes this red.
"""

from __future__ import annotations

import ast
import inspect
from collections import Counter

from polylogue.storage.sqlite import connection_profile


def _module_source() -> ast.Module:
    return ast.parse(inspect.getsource(connection_profile))


def test_no_module_level_name_is_assigned_twice() -> None:
    assigned: Counter[str] = Counter()
    for node in _module_source().body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assigned[target.id] += 1
    assert [name for name, count in assigned.items() if count > 1] == []


def test_dunder_all_has_no_duplicate_entries() -> None:
    duplicates = [name for name, count in Counter(connection_profile.__all__).items() if count > 1]
    assert duplicates == []


def test_isolated_tier_write_profile_is_the_publication_timeout_policy() -> None:
    """The surviving definition is the one-tier excision/backup policy."""
    profile = connection_profile.ISOLATED_TIER_WRITE_PROFILE
    assert profile.role == "write"
    assert profile.timeout_seconds == connection_profile.TIMEOUT_CLASS_PUBLICATION_S
    assert profile.wal_autocheckpoint_pages == connection_profile.WAL_AUTOCHECKPOINT_PAGES

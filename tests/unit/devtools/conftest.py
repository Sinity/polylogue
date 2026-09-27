"""Keep runner tests independent of the branch the tested checkout is on.

``devtools test`` and ``devtools verify`` refuse on the default branch. Tests
that drive those runners exercise other behaviour, and must pass the same way
in a feature worktree and in a deliberate run on the default branch, so they
see a feature branch unless a test asks for the real identity.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from devtools import checkout_identity as identity_module
from devtools import run_tests, verify


@pytest.fixture(autouse=True)
def _runners_see_a_feature_branch(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if request.node.get_closest_marker("real_checkout_identity") is None:
        real_identity = identity_module.checkout_identity

        def feature_branch(root: Path) -> identity_module.CheckoutIdentity:
            return replace(real_identity(root), branch="test/feature")

        # The runners bind the name at import; the queued slot re-reads it
        # from the module when the run starts.
        monkeypatch.setattr(identity_module, "checkout_identity", feature_branch)
        monkeypatch.setattr(verify, "checkout_identity", feature_branch)
        monkeypatch.setattr(run_tests, "checkout_identity", feature_branch)
    yield


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "real_checkout_identity: let the runners read the checkout's real branch")

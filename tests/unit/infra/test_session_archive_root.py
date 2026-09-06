"""A pytest process must not be able to resolve the operator's archive.

Four synthetic fixtures from this suite reached the operator's live blob
store, carrying no ``blob_refs`` or ``raw_sessions`` row: written straight to
a ``BlobStore`` opened at the resolved root, not admitted through acquisition.
``XDG_DATA_HOME`` does not isolate that root -- ``[archive] root`` in the user
config outranks the XDG default -- so any code running outside the per-test
environment fixture (collection, plugin hooks, session-scoped fixtures)
resolves the archive that config names.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from polylogue.paths import archive_root, blob_store_root
from tests.infra.session_archive_root import (
    ARCHIVE_ROOT_ENV,
    discard_session_archive_root,
    pin_session_archive_root,
)


@pytest.fixture
def operator_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A user config naming a live archive, with no archive root in the environment."""
    config_root = tmp_path / "config"
    (config_root / "polylogue").mkdir(parents=True)
    live = tmp_path / "live-archive"
    (config_root / "polylogue" / "polylogue.toml").write_text(f'[archive]\nroot = "{live}"\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_root))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "isolated-xdg-data"))
    monkeypatch.delenv(ARCHIVE_ROOT_ENV, raising=False)
    return live


def test_xdg_redirection_alone_resolves_the_configured_archive(operator_config: Path) -> None:
    """The route: a process that isolates only XDG still resolves the live archive.

    Mutation: make ``resolve_archive_root`` prefer the XDG data home over the
    user config's ``[archive] root``. This goes red, and so does the reason
    the pin below exists.
    """
    assert archive_root() == operator_config


def test_pinning_overrides_the_configured_archive(
    monkeypatch: pytest.MonkeyPatch, operator_config: Path, tmp_path: Path
) -> None:
    """Mutation: drop the ``environ[ARCHIVE_ROOT_ENV] = ...`` assignment from
    ``pin_session_archive_root``. Resolution falls back to the configured live
    archive and both assertions below name it."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    pinned = pin_session_archive_root(os.environ, base=tmp_path)
    try:
        assert archive_root() == pinned
        assert blob_store_root() == pinned / "blob"
    finally:
        discard_session_archive_root(pinned)


def test_the_session_pin_is_installed_by_the_harness() -> None:
    """The configure hook minted a scratch root that outlives every fixture."""
    import tests.conftest as root_conftest

    pinned = root_conftest._SESSION_ARCHIVE_ROOT
    assert pinned is not None
    assert pinned.is_dir()


def test_discard_refuses_a_root_it_did_not_mint(tmp_path: Path) -> None:
    """Teardown never removes a directory outside its own naming."""
    foreign = tmp_path / "not-a-session-root"
    foreign.mkdir()

    discard_session_archive_root(foreign)

    assert foreign.is_dir()

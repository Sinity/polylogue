"""CLI entrypoint exports.

``cli``/``main`` are exported lazily (PEP 562 module ``__getattr__``) rather
than imported eagerly at package-import time. ``polylogue.cli.click_app``
pulls in the full archive/storage stack (``polylogue.storage``,
``polylogue.archive.query.*``); an eager ``from .click_app import cli, main``
here would defeat the whole point of lightweight submodules such as
``polylogue.cli.daemon_client`` (pinned by
``tests/unit/cli/test_daemon_client.py::test_daemon_client_import_does_not_load_storage``)
-- Python always executes a package's ``__init__.py`` before any of its
submodules, so ``import polylogue.cli.daemon_client`` would otherwise import
``click_app`` as a side effect regardless of what ``daemon_client`` itself
imports. Console-script entry points (``polylogue = "polylogue.cli:main"``)
resolve attributes via ``getattr``, which ``__getattr__`` intercepts
transparently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .click_app import cli, main

__all__ = ["cli", "main"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from . import click_app

        return getattr(click_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

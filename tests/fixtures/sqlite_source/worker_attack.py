"""Synthetic filesystem races injected at the isolated SQLite reader boundary."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

from polylogue.sources import sqlite_export

source = Path(os.environ["POLYLOGUE_TEST_SQLITE_SOURCE"])
external = Path(os.environ["POLYLOGUE_TEST_SQLITE_EXTERNAL"])
attack = os.environ["POLYLOGUE_TEST_SQLITE_ATTACK"]
real_connect = sqlite3.connect


def swap_main(connect: Callable[..., sqlite3.Connection], *args: Any, **kwargs: Any) -> sqlite3.Connection:
    held = source.with_name("held-main.sqlite")
    source.rename(held)
    if attack == "main-symlink":
        source.symlink_to(external)
    else:
        external.rename(source)
    try:
        return connect(*args, **kwargs)
    finally:
        if attack == "main-symlink":
            source.unlink()
        else:
            source.rename(external)
        held.rename(source)


class RacingConnection(sqlite3.Connection):
    def execute(self, statement: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        if attack.startswith("main-"):
            Path(os.environ["POLYLOGUE_TEST_SQLITE_SQL_MARKER"]).write_text("SQL ran", encoding="utf-8")
        if attack == "sidecar" and "FROM sqlite_master" in statement:
            held = []
            for suffix in ("-wal", "-shm"):
                path = Path(str(source) + suffix)
                original = path.with_name(path.name + ".held")
                foreign = Path(str(external) + suffix)
                path.rename(original)
                foreign.rename(path)
                held.append((path, original, foreign))
            try:
                return super().execute(statement, *args, **kwargs)
            finally:
                for path, original, foreign in held:
                    path.rename(foreign)
                    original.rename(path)
        return super().execute(statement, *args, **kwargs)


def connect(database: str, *args: Any, **kwargs: Any) -> sqlite3.Connection:
    if source.name not in str(database):
        return real_connect(database, *args, **kwargs)
    if attack.startswith("main-"):
        return swap_main(real_connect, database, *args, factory=RacingConnection, **kwargs)
    conn = real_connect(database, *args, factory=RacingConnection, **kwargs)
    if attack in {"unknown", "unlinked"}:
        os.open(external, os.O_RDONLY)
        if attack == "unlinked":
            external.unlink()
    return conn


with patch("sqlite3.connect", connect):
    sqlite_export._source_worker_main()

"""Neutral live processes for admitted-run custody controls."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import BinaryIO

DETACHED_PROGRAM = r"""
import json, os, pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
child_code = 'import pathlib,sys,time; payload=bytearray(8*1024*1024); root=pathlib.Path(sys.argv[1]); (root/"ready").touch();\nwhile not (root/"stop").exists(): time.sleep(0.01)'
child = subprocess.Popen([sys.executable, '-c', child_code, str(root)], start_new_session=True)
(root/'identities.json').write_text(json.dumps({'controller':os.getpid(), 'detached':child.pid, 'custody':os.environ['POLYLOGUE_PYTEST_CUSTODY'], 'receipt':os.environ['POLYLOGUE_PYTEST_RUN_ID']}))
while not (root/'ready').exists(): time.sleep(0.01)
time.sleep(1.1)
"""


class EnvironmentReader:
    """Record the actual comparison's requested read sizes on an owned file."""

    def __init__(self, handle: BinaryIO, sizes: list[int]) -> None:
        self.handle = handle
        self.sizes = sizes

    def __enter__(self) -> EnvironmentReader:
        return self

    def __exit__(self, *_args: object) -> None:
        self.handle.close()

    def read(self, size: int = -1) -> bytes:
        self.sizes.append(size)
        return self.handle.read(size)


def wait_until_detached_child_is_reparented(root: Path) -> tuple[int, int]:
    """Observe the real child's parent transition before acquiring its sampler."""
    identities = root / "identities.json"
    while not identities.exists():
        time.sleep(0.01)
    values = json.loads(identities.read_text())
    controller, detached = values["controller"], values["detached"]
    while True:
        fields = (Path("/proc") / str(detached) / "stat").read_text().rpartition(")")[2].split()
        if int(fields[1]) != controller:
            return controller, detached
        time.sleep(0.01)

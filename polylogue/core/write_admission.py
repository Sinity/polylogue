"""The exact execution frame admitted by the archive writer coordinator."""

from __future__ import annotations

import asyncio
import contextvars
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WriteAdmission:
    coordinator: object
    task: asyncio.Task[Any]
    archive_root: Path
    pid: int = field(default_factory=os.getpid)
    thread: threading.Thread = field(default_factory=threading.current_thread)

    def admits(self, coordinator: object, archive_root: str | Path) -> bool:
        return (
            self.coordinator is coordinator
            and self.pid == os.getpid()
            and self.thread is threading.current_thread()
            and self.task is asyncio.current_task()
            and self.archive_root.resolve() == Path(archive_root).resolve()
        )


active_write_admission: contextvars.ContextVar[WriteAdmission | None] = contextvars.ContextVar(
    "polylogue_active_write_admission", default=None
)

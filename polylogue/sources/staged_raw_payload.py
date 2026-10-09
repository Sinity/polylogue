"""Private raw files retained from parser preparation to creator pickup."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import IO
from uuid import uuid4

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.prepared_file import PreparedFileSeal


class StagedRawPayload:
    """One sealed file; its preparation directory lives through physical pickup."""

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = (
            tempfile.TemporaryDirectory(prefix="polylogue-raw-preparation-") if directory is None else None
        )
        root = Path(self._directory.name) if self._directory is not None else directory
        assert root is not None
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._path = root / f"raw-{uuid4().hex}.json"
        self._seal: PreparedFileSeal | None = None
        self.retained = False
        # Shared parser outputs can borrow the same file. The creator's first
        # physical copy settles it for all of them before deleting that file.
        self.adopted: tuple[Path, str | None, str, int, str | None] | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def seal(self) -> PreparedFileSeal:
        if self._seal is None:
            raise ValueError("raw capture is not sealed")
        return self._seal

    @classmethod
    def prepare(cls, writer: Callable[[Path], None], *, directory: Path | None = None) -> StagedRawPayload:
        staged = cls(directory)
        try:
            writer(staged.path)
            staged._seal = PreparedFileSeal.capture(staged.path)
            return staged
        except BaseException:
            staged.discard()
            raise

    @classmethod
    def from_value(cls, value: object, *, directory: Path | None = None) -> StagedRawPayload:
        from polylogue.sources.streamed_json_output import write_streamed_json

        return cls.prepare(lambda path: write_streamed_json(value, path), directory=directory)

    @classmethod
    def from_stream(cls, source: IO[bytes], *, directory: Path | None = None) -> StagedRawPayload:
        def write(path: Path) -> None:
            with path.open("xb") as target:
                while True:
                    check_compute_cancelled()
                    chunk = source.read(64 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)

        return cls.prepare(write, directory=directory)

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)
        if self._directory is not None:
            self._directory.cleanup()

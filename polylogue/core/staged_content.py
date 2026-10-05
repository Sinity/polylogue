"""Private seekable content owned by one completed delivery lifetime."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import BinaryIO, TextIO


@contextmanager
def staged_binary_content() -> Iterator[BinaryIO]:
    """Remove the private document on failure, cancellation, or completed delivery."""
    with tempfile.TemporaryFile(mode="w+b") as content:
        os.fchmod(content.fileno(), 0o600)
        yield content


@contextmanager
def staged_text_content() -> Iterator[TextIO]:
    """Own the UTF-8 rendering document through its final publication."""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as content:
        os.fchmod(content.fileno(), 0o600)
        yield content

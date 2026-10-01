"""A module-local closer with deliberately ambiguous descriptor settlement."""

import errno
import os
from collections.abc import Callable


class DescriptorCloseFault:
    def __init__(self, rejected: Callable[[int], bool]) -> None:
        self.rejected = rejected
        self.attempts: list[int] = []

    def close(self, descriptor: int) -> None:
        self.attempts.append(descriptor)
        if self.rejected(descriptor):
            raise OSError(errno.EINTR, "synthetic ambiguous descriptor close")
        os.close(descriptor)

    def __getattr__(self, name: str) -> object:
        return getattr(os, name)

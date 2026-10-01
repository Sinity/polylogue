"""Observe physical SQLite file descriptors in synthetic Linux fixtures."""

import os


def selected_file_descriptors(identity: tuple[int, int]) -> tuple[int, ...]:
    selected = []
    for name in os.listdir("/proc/self/fd"):
        try:
            metadata = os.fstat(int(name))
        except OSError:
            continue
        if (metadata.st_dev, metadata.st_ino) == identity:
            selected.append(int(name))
    return tuple(selected)

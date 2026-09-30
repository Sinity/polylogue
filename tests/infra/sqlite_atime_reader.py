"""Run the production custodian with observable read-induced timestamp change."""

from __future__ import annotations

import os

from polylogue.storage.sqlite import identity_custodian


def main() -> None:
    read = os.read
    advanced = False

    def read_and_advance_atime(descriptor: int, size: int) -> bytes:
        nonlocal advanced
        before = os.fstat(descriptor)
        chunk = read(descriptor, size)
        if chunk and not advanced:
            # The host may mount noatime. Make the first streamed read's
            # timestamp effect observable without changing its bytes or mtime.
            os.utime(descriptor, ns=(before.st_atime_ns + 1_000_000_000, before.st_mtime_ns))
            advanced = True
        return chunk

    os.read = read_and_advance_atime
    identity_custodian.main()


if __name__ == "__main__":
    main()

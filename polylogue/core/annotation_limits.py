"""Bounds on an annotation batch import, owned in one leaf module.

These are protocol constants: the request contract in
``operations/daemon_protocol.py`` bounds its ``jsonl`` field by them, the
importer enforces them, and the CLI uses them to avoid buffering more than
the operation would ever accept.

They live in ``polylogue.core`` rather than in ``polylogue.annotations``
because both of those consumers must be able to read them without importing
the annotation substrate:

- ``operations/daemon_protocol.py`` is imported by ``polylogue.daemon_client``,
  whose whole point is to be import-light -- it previously reached
  ``annotations.importer`` for one integer and pulled 42 ``polylogue.storage``
  modules in behind it, through ``annotations/__init__`` -> ``annotations.write``.
- ``cli/commands/annotations.py`` may not import the annotation substrate at
  all under the mutation-authority layering rule, and so carried its own
  duplicate literal of the byte bound.

A module here has no substrate imports, so both can name the same constant
instead of one leaking and the other copying it.
"""

from __future__ import annotations

MAX_ANNOTATION_IMPORT_BYTES = 1_048_576
MAX_ANNOTATION_IMPORT_ROWS = 10_000
MAX_ANNOTATION_IMPORT_LINE_BYTES = 65_536
MAX_ANNOTATION_IMPORT_REF_BYTES = 4_096

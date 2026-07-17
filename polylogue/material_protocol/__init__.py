"""Polylogue public normalized-session material protocol.

This package defines the wire format Polylogue uses to publish a
Sinex-independent, deterministic, byte-stable representation of a session
(messages, blocks, tool calls/results, lineage, compactions, attachments,
usage) as bounded immutable NDJSON segments plus a revision manifest.

See ``polylogue.material_protocol.v1`` for the current protocol version and
``docs/material-protocol-v1.md`` for the wire-format reference. This package
provides encode/decode/verify/segmentation/anchor-resolution helpers only;
transport, durable publication, and Sinex settlement are out of scope here
(polylogue-303r.2).
"""

from __future__ import annotations

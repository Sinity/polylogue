"""Excision and secret-hygiene primitives (polylogue-27m).

"The archive can forget on purpose." This package owns three related, but
distinct, mechanisms:

- :mod:`polylogue.security.secret_scan` -- a candidate-only secret detector.
  It finds credential-shaped spans in captured content and records them as
  non-injectable ``AssertionKind.SECRET_CANDIDATE`` assertions. It never
  returns, stores, or logs a matched literal.
- :mod:`polylogue.security.excision` -- standalone/off-mode local excision.
  Authoritative: it removes a session's rows from source.db, index.db
  (cascading to messages/blocks/FTS/session_links), embeddings.db, and
  ``blob_refs``, records a durable removed-hash marker so ordinary re-ingest
  cannot resurrect the content, and writes a durable audit receipt.
- :mod:`polylogue.security.lifecycle` -- the mirror/primary durable
  lifecycle-request/outbox mechanics, exercised against a fault-injecting
  versioned contract fake. This bead owns the *mechanism*; binding it to a
  real Sinex confirmation, purge, residual, rebuild, and backup proof is
  polylogue-303r.6's scope, not this package's.

See ``docs/security.md`` ("Excision and secret hygiene") for the threat
model and explicit non-goals.
"""

from __future__ import annotations

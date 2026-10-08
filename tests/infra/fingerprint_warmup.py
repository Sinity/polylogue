"""Resolve the source-closure fingerprints once per test process.

The parser, lowering, replay-routing and materializer fingerprints are AST
closures over Polylogue's own source. The disk memo that makes them cheap
lives under the user cache home, and most fixtures repoint ``HOME`` at an
empty directory, so the first test in a process to reach a fingerprint used
to recompute it cold (about 90 s under load) inside its own time budget.

Resolving them at configure time, before any fixture isolates the home, puts
the cost (normally a memo read) outside every test. The in-process memo is
keyed by the observed source bytes, never by the cache location, so later
isolated lookups reuse it. The controller runs first and publishes the disk
memo, so xdist workers started after it read rather than recompute.
"""

from __future__ import annotations


def warm_source_fingerprints() -> None:
    from polylogue.archive.revision_authority import raw_authority_parser_fingerprint
    from polylogue.storage.sqlite.archive_tiers.schema_identity import DerivedTier, derived_schema_identity

    raw_authority_parser_fingerprint()
    for tier in DerivedTier:
        derived_schema_identity(tier)

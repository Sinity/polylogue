"""The provider denominator must not be derived from what it measures.

Anti-vacuity for every test here: re-derive the subject list from the committed
package tree, the schema registry, the frontier, or the run receipts -- any of
the artefacts a generation pass produces -- and these go red, because each one
asserts a subject the denominator holds *and* those artefacts do not.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.core.schema_subjects import SCHEMA_SUBJECTS
from polylogue.schemas.provider_denominator import derive_provider_denominator
from polylogue.schemas.registry import SCHEMA_DIR
from polylogue.sources.origin_specs import origin_specs


def test_every_declared_subject_is_counted_exactly_once() -> None:
    denominator = derive_provider_denominator()
    tokens = [item.subject for item in denominator.subjects]

    assert tokens == sorted(tokens), "the denominator must be ordered so two runs bind the same digest"
    assert set(tokens) == {spec.token for spec in SCHEMA_SUBJECTS}
    assert len(tokens) == len(set(tokens))


def test_a_subject_with_no_committed_package_is_still_in_the_denominator() -> None:
    """The proof that the denominator does not read its own measurement.

    ``grok``, ``beads`` and ``browser-capture`` are declared subjects with no
    directory under the committed package root. A denominator derived from that
    root -- the defect this module exists to prevent -- could not name them.
    """

    denominator = derive_provider_denominator()
    packaged = {path.name for path in Path(SCHEMA_DIR).iterdir() if path.is_dir()}

    unpackaged = {"beads", "browser-capture", "grok"}
    assert not (unpackaged & packaged), "this proof needs subjects the package tree does not hold"
    assert unpackaged <= {item.subject for item in denominator.subjects}


def test_a_subject_absent_from_the_frontier_is_still_in_the_denominator() -> None:
    """A subject with no declared source root cannot fall out of the count.

    ``beads`` and ``browser-capture`` carry no frontier roots at all, so a
    denominator read off the declared frontier would silently shrink by two.
    """

    denominator = derive_provider_denominator()
    assert {"beads", "browser-capture"} <= {item.subject for item in denominator.subjects}


def test_the_two_declarations_agree_at_head() -> None:
    denominator = derive_provider_denominator()
    assert denominator.findings == ()


def test_every_executable_origin_is_claimed_by_exactly_one_subject() -> None:
    """The cross-check that makes the denominator falsifiable.

    Anti-vacuity: the assertion is not that the two declarations were compared,
    but that every executable origin in the detector registry has a subject. A
    new executable origin with no schema subject leaves this set non-empty.
    """

    executable = {str(spec.origin) for spec in origin_specs() if spec.lifecycle == "executable"}
    claimed: dict[str, int] = {}
    for spec in SCHEMA_SUBJECTS:
        for origin in spec.origins:
            claimed[origin] = claimed.get(origin, 0) + 1

    assert executable, "the executable origin registry must not be empty"
    assert executable <= set(claimed)
    assert all(claimed[origin] == 1 for origin in executable)


def test_disposition_separates_declared_exclusion_from_a_missing_route() -> None:
    denominator = derive_provider_denominator()

    browser_capture = denominator.subject("browser-capture")
    assert browser_capture is not None
    assert browser_capture.disposition == "declared_non_applicable"
    assert browser_capture.reason

    beads = denominator.subject("beads")
    assert beads is not None
    assert beads.disposition == "no_executable_route"
    assert beads.executable_origins == ()

    codex = denominator.subject("codex")
    assert codex is not None
    assert codex.disposition == "generation_required"
    assert codex.executable_origins == ("codex-session",)


def test_the_declaration_digest_moves_with_the_declaration() -> None:
    """A receipt bound to the digest must stop matching when a subject changes."""

    denominator = derive_provider_denominator()
    baseline = denominator.declaration_digest
    assert derive_provider_denominator().declaration_digest == baseline

    mutated = type(denominator)(subjects=denominator.subjects[:-1], findings=denominator.findings)
    assert mutated.declaration_digest != baseline

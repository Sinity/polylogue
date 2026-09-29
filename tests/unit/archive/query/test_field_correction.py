"""An unknown query field gets one local diagnostic and a safe candidate correction.

polylogue-z9gh.3.3 AC4: a malformed request receives a precise diagnostic and
only a semantics-preserving candidate correction. The correction renames the
misspelt field and nothing else, and is withheld whenever the guess is
ambiguous, would reshape a scoped field, or would not compile.
"""

from __future__ import annotations

import pytest

from polylogue.archive.query.expression import (
    UnknownQueryFieldError,
    compile_expression,
    parse_unit_source_expression,
    propose_field_correction,
)


def _unknown_field(expression: str) -> UnknownQueryFieldError:
    with pytest.raises(UnknownQueryFieldError) as raised:
        if parse_unit_source_expression(expression) is None:
            compile_expression(expression)
    return raised.value


@pytest.mark.parametrize(
    ("expression", "corrected"),
    [
        ("messages where rol:user", "messages where role:user"),
        ("orign:codex-session", "origin:codex-session"),
        ("messages where rol:user and text:rol:x", "messages where role:user and text:rol:x"),
        ("messages where rol:user and text:(rol:x)", "messages where role:user and text:(rol:x)"),
        ("messages where -rol:user", "messages where -role:user"),
        # Quoted literals are values, not clause keys: only the key is renamed.
        ('messages where rol:user and text:"rol:x"', 'messages where role:user and text:"rol:x"'),
    ],
)
def test_an_unambiguous_misspelling_gets_the_field_renamed_and_nothing_else(expression: str, corrected: str) -> None:
    """Anti-vacuity: rename inside quotes, or drop the candidate check, and a row goes red."""
    error = _unknown_field(expression)

    assert len(error.candidates) == 1
    assert propose_field_correction(expression, error) == corrected


@pytest.mark.parametrize(
    "expression",
    [
        # Three declared names are equally near: guessing one would decide for the caller.
        "messages where ro:user",
        # Nearest by spelling is bare ``session``; renaming the scoped field to it
        # would change what the clause selects.
        "sesion.tag:x",
    ],
)
def test_an_ambiguous_or_reshaping_guess_is_no_correction(expression: str) -> None:
    """Anti-vacuity: take the first candidate unconditionally and both rows offer a correction."""
    error = _unknown_field(expression)

    assert error.candidates
    assert propose_field_correction(expression, error) is None


def test_a_shared_field_is_not_its_own_ambiguous_candidate() -> None:
    """Duplicating names from the two registries makes an exact candidate ambiguous."""
    error = _unknown_field("messages where duraton_ms:>=1")

    assert error.candidates == ("duration_ms",)
    assert propose_field_correction("messages where duraton_ms:>=1", error) == "messages where duration_ms:>=1"

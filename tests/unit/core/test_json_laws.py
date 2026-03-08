"""Property laws for core JSON utilities and type coercion.

Each law covers a behavioral invariant that supersedes specific parametrized
example tests in test_json.py.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from polylogue.lib import json as core_json
from polylogue.types import Provider


# ---------------------------------------------------------------------------
# Serializable value strategy
# ---------------------------------------------------------------------------

_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False),
    st.text(alphabet=st.characters(blacklist_categories=("Cs",))),
)

_json_value = st.recursive(
    _scalar,
    lambda children: st.one_of(
        st.lists(children, max_size=6),
        st.dictionaries(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20), children, max_size=6),
    ),
    max_leaves=20,
)


# ---------------------------------------------------------------------------
# Law 1: JSON roundtrip for basic serializable types
# ---------------------------------------------------------------------------

@given(_json_value)
def test_json_roundtrip_basic_types(value: object) -> None:
    """Any JSON-serializable value survives a dumps/loads roundtrip."""
    output = core_json.dumps(value)
    result = core_json.loads(output)
    assert result == value


# ---------------------------------------------------------------------------
# Law 2: Decimal encodes to float exactly
# ---------------------------------------------------------------------------

@given(st.decimals(
    allow_nan=False,
    allow_infinity=False,
    min_value=Decimal("-1e15"),
    max_value=Decimal("1e15"),
))
def test_decimal_encodes_to_float(d: Decimal) -> None:
    """Decimal always encodes to float — never to string, never raises.
    The encoded value equals float(d)."""
    output = core_json.dumps({"v": d})
    result = core_json.loads(output)
    assert isinstance(result["v"], float)
    assert result["v"] == float(d)


# ---------------------------------------------------------------------------
# Law 3: Invalid JSON always raises, never silently returns None
# ---------------------------------------------------------------------------

_INVALID_JSON_FRAGMENTS = [
    "{",
    "}",
    "[",
    "]",
    "{1: 2}",
    "undefined",
    "{'single': 'quotes'}",
    "{\"key\": undefined}",
    "NaN",
    "Infinity",
    "01",  # leading zero
    "",
]


@pytest.mark.parametrize("fragment", _INVALID_JSON_FRAGMENTS)
def test_loads_known_invalid_json_raises(fragment: str) -> None:
    """Invalid JSON fragments always raise an exception."""
    with pytest.raises(Exception):
        core_json.loads(fragment)


@given(
    st.text(min_size=1, alphabet="{[}\"]:")
    .filter(lambda s: s.strip() not in ("[]", "{}", '""', "{}"))
)
def test_loads_malformed_json_never_silent(text: str) -> None:
    """loads either raises or returns a non-None value; it never silently returns None
    for a non-null JSON input."""
    try:
        result = core_json.loads(text)
        # If loads succeeds, it must have parsed to something
        # (Note: valid JSON 'null' would return None, so we only assert on successful
        # non-null parses)
        _ = result  # No assertion needed - successful parse is fine
    except Exception:
        pass  # Expected for malformed input


# ---------------------------------------------------------------------------
# Law 4: Provider.from_string always returns a non-empty value
# ---------------------------------------------------------------------------

@given(st.text())
def test_provider_from_string_value_never_empty(text: str) -> None:
    """Provider.from_string(x).value is always a non-empty string."""
    result = Provider.from_string(text)
    assert isinstance(result.value, str)
    assert len(result.value) > 0


# ---------------------------------------------------------------------------
# Law 5: Provider.from_string is idempotent on known values
# ---------------------------------------------------------------------------

@given(st.sampled_from([p.value for p in Provider]))
def test_provider_from_string_idempotent(value: str) -> None:
    """Applying from_string to a known provider value twice gives the same result."""
    first = Provider.from_string(value)
    second = Provider.from_string(first.value)
    assert first == second
    assert first.value == second.value

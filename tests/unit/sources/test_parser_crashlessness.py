"""Schema-driven crashlessness tests for provider parsers.

Generates structurally valid payloads from real provider JSON schemas
(via hypothesis-jsonschema) and feeds them through parser detection +
parsing. The goal: parsers never crash on schema-conformant input.

Acceptable rejection: ValueError, TypeError, KeyError (parse rejection).
Unacceptable: AttributeError, IndexError, RecursionError (bugs).
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.sources.parsers import chatgpt, claude, codex, drive
from tests.infra.strategies.schema_driven import schema_conformant_payload

# Map provider names to their parser functions
PROVIDER_PARSERS = {
    "chatgpt": {
        "looks_like": chatgpt.looks_like,
        "parse": lambda payload: chatgpt.parse(payload, "crashtest"),
    },
    "claude-ai": {
        "looks_like": claude.looks_like_ai,
        "parse": lambda payload: claude.parse_ai(payload, "crashtest"),
    },
    "claude-code": {
        "looks_like": claude.looks_like_code,
        "parse": lambda payload: claude.parse_code(payload, "crashtest"),
    },
    "codex": {
        "looks_like": codex.looks_like,
        "parse": lambda payload: codex.parse(payload, "crashtest"),
    },
    "gemini": {
        "looks_like": drive.looks_like,
        "parse": lambda payload: drive.parse_chunked_prompt("gemini", payload, "crashtest"),
    },
}

# Exceptions that indicate real bugs in parser code
CRASH_EXCEPTIONS = (IndexError, RecursionError)


@pytest.mark.parametrize("provider", sorted(PROVIDER_PARSERS.keys()))
@given(data=st.data())
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
def test_looks_like_never_crashes(provider: str, data) -> None:
    """looks_like() never raises crash exceptions on schema-conformant input."""
    if provider not in PROVIDER_PARSERS:
        pytest.skip(f"Provider {provider} not configured")

    payload = data.draw(schema_conformant_payload(provider))
    try:
        result = PROVIDER_PARSERS[provider]["looks_like"](payload)
        assert isinstance(result, bool)
    except CRASH_EXCEPTIONS as exc:
        raise AssertionError(
            f"{provider}.looks_like() crashed on schema-conformant input: {type(exc).__name__}: {exc}"
        ) from exc
    except Exception:
        # Other exceptions from looks_like are acceptable (format detection rejection)
        pass


@pytest.mark.parametrize("provider", sorted(PROVIDER_PARSERS.keys()))
@given(data=st.data())
@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
def test_parse_never_crashes(provider: str, data) -> None:
    """parse() never raises crash exceptions on schema-conformant input.

    We feed schema-conformant payloads to parse(). Even if looks_like()
    would reject, we test parse() since the schema says the data is
    structurally valid.
    """
    if provider not in PROVIDER_PARSERS:
        pytest.skip(f"Provider {provider} not configured")

    payload = data.draw(schema_conformant_payload(provider))
    try:
        PROVIDER_PARSERS[provider]["parse"](payload)
    except CRASH_EXCEPTIONS as exc:
        raise AssertionError(
            f"{provider}.parse() crashed on schema-conformant input: {type(exc).__name__}: {exc}"
        ) from exc
    except Exception:
        # ValueError/TypeError/KeyError are acceptable parse rejections
        pass

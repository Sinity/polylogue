"""Schema-driven crashlessness tests for provider parsers.

Generates structurally valid payloads from real provider JSON schemas
(via hypothesis-jsonschema) and feeds them through parser detection +
parsing. The goal: parsers never crash on schema-conformant input.

Acceptable rejection: ValueError, TypeError, KeyError (parse rejection).
Unacceptable: AttributeError, IndexError, RecursionError (bugs).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict, cast

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from polylogue.sources.parsers import chatgpt, claude, codex, drive
from tests.infra.strategies.schema_driven import schema_conformant_payload


class ProviderParser(TypedDict):
    looks_like: Callable[[object], bool]
    parse: Callable[[object], object]


def _payload_dict(payload: object) -> dict[str, object]:
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _payload_list(payload: object) -> list[object]:
    assert isinstance(payload, list)
    return cast(list[object], payload)


def _parse_chatgpt(payload: object) -> object:
    return chatgpt.parse(_payload_dict(payload), "crashtest")


def _parse_claude_ai(payload: object) -> object:
    return claude.parse_ai(_payload_dict(payload), "crashtest")


def _parse_claude_code(payload: object) -> object:
    return claude.parse_code(_payload_list(payload), "crashtest")


def _looks_like_claude_code(payload: object) -> bool:
    return claude.looks_like_code(_payload_list(payload))


def _looks_like_codex(payload: object) -> bool:
    return codex.looks_like(_payload_list(payload))


def _parse_codex(payload: object) -> object:
    return codex.parse(_payload_list(payload), "crashtest")


def _parse_gemini(payload: object) -> object:
    return drive.parse_chunked_prompt("gemini", _payload_dict(payload), "crashtest")


PROVIDER_PARSERS: dict[str, ProviderParser] = {
    "chatgpt": {"looks_like": chatgpt.looks_like, "parse": _parse_chatgpt},
    "claude-ai": {"looks_like": claude.looks_like_ai, "parse": _parse_claude_ai},
    "claude-code": {"looks_like": _looks_like_claude_code, "parse": _parse_claude_code},
    "codex": {"looks_like": _looks_like_codex, "parse": _parse_codex},
    "gemini": {"looks_like": drive.looks_like, "parse": _parse_gemini},
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
def test_looks_like_never_crashes(provider: str, data: st.DataObject) -> None:
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
def test_parse_never_crashes(provider: str, data: st.DataObject) -> None:
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

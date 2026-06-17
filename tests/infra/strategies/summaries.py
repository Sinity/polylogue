"""Hypothesis strategies and helpers for SessionSummary contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from hypothesis import strategies as st

from polylogue.archive.models import SessionSummary
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.types import SessionId

_PROVIDERS = ("claude-ai", "chatgpt", "gemini", "codex", "claude-code")
_SLUG_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"
_TEXT_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;?!-_/"


@dataclass(frozen=True)
class SessionSummarySpec:
    """Generated summary fixture used across CLI and site contract tests."""

    session_id: str
    provider: str
    title: str | None
    summary: str | None
    tags: tuple[str, ...]
    created_at: datetime | None
    updated_at: datetime | None
    message_count: int


@st.composite
def session_summary_spec_strategy(draw: st.DrawFn) -> SessionSummarySpec:
    """Generate one lightweight summary with stable display semantics."""
    slug = draw(st.text(alphabet=_SLUG_ALPHABET, min_size=6, max_size=14))
    created_at = draw(
        st.one_of(
            st.none(),
            st.datetimes(
                min_value=datetime(2023, 1, 1),
                max_value=datetime(2026, 12, 31),
                timezones=st.just(timezone.utc),
            ),
        )
    )
    updated_at = created_at
    if created_at is not None:
        updated_at = draw(
            st.one_of(
                st.just(created_at),
                st.integers(min_value=0, max_value=3600).map(lambda seconds: created_at + timedelta(seconds=seconds)),
            )
        )

    title = draw(
        st.one_of(
            st.none(),
            st.text(alphabet=_TEXT_ALPHABET, min_size=1, max_size=48).filter(lambda value: value.strip() != ""),
        )
    )
    summary = draw(
        st.one_of(
            st.none(),
            st.text(alphabet=_TEXT_ALPHABET + "\n", min_size=1, max_size=120).filter(lambda value: value.strip() != ""),
        )
    )
    tags = tuple(
        draw(
            st.lists(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_", min_size=1, max_size=12),
                min_size=0,
                max_size=3,
                unique=True,
            )
        )
    )
    return SessionSummarySpec(
        session_id=f"conv-{slug}",
        provider=draw(st.sampled_from(_PROVIDERS)),
        title=title,
        summary=summary,
        tags=tags,
        created_at=created_at,
        updated_at=updated_at,
        message_count=draw(st.integers(min_value=0, max_value=200)),
    )


@st.composite
def session_summary_batch_strategy(
    draw: st.DrawFn,
    *,
    min_size: int = 0,
    max_size: int = 6,
) -> tuple[SessionSummarySpec, ...]:
    """Generate a unique batch of summaries for list/index contract tests."""
    return tuple(
        draw(
            st.lists(
                session_summary_spec_strategy(),
                min_size=min_size,
                max_size=max_size,
                unique_by=lambda spec: spec.session_id,
            )
        )
    )


def build_session_summary(spec: SessionSummarySpec) -> SessionSummary:
    """Materialize a generated summary spec into a real model."""
    return SessionSummary(
        id=SessionId(spec.session_id),
        origin=origin_from_provider(Provider.from_string(spec.provider)),
        title=spec.title,
        created_at=spec.created_at,
        updated_at=spec.updated_at,
        metadata={
            "tags": list(spec.tags),
            "summary": spec.summary,
        },
    )


def build_message_counts(
    specs: tuple[SessionSummarySpec, ...] | list[SessionSummarySpec],
) -> dict[str, int]:
    """Build the message-count mapping used by summary and site contracts."""
    return {spec.session_id: spec.message_count for spec in specs}

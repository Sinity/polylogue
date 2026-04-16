"""Hypothesis strategies and helpers for static-site contract tests."""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from tests.infra.strategies.summaries import (
    ConversationSummarySpec,
    conversation_summary_batch_strategy,
)

_TITLE_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_/"


@dataclass(frozen=True)
class SiteArchiveSpec:
    """Generated site-build case for archive/index/search contracts."""

    summaries: tuple[ConversationSummarySpec, ...]
    include_dashboard: bool
    enable_search: bool
    search_provider: str
    custom_title: str | None
    precreate_output_dir: bool


@st.composite
def site_archive_spec_strategy(draw: st.DrawFn) -> SiteArchiveSpec:
    """Generate a site build with bounded provider/archive shape."""
    return SiteArchiveSpec(
        summaries=draw(conversation_summary_batch_strategy(min_size=0, max_size=6)),
        include_dashboard=draw(st.booleans()),
        enable_search=draw(st.booleans()),
        search_provider=draw(st.sampled_from(("pagefind", "lunr"))),
        custom_title=draw(
            st.one_of(
                st.none(),
                st.text(alphabet=_TITLE_ALPHABET, min_size=1, max_size=36).filter(lambda value: value.strip() != ""),
            )
        ),
        precreate_output_dir=draw(st.booleans()),
    )


def expected_index_pages(spec: SiteArchiveSpec) -> int:
    """Return the expected index/dashboard page count for a generated build."""
    provider_count = len({summary.provider for summary in spec.summaries})
    return 1 + provider_count + (1 if spec.include_dashboard else 0)

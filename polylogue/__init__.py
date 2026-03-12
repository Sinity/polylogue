"""Polylogue - AI Conversation Archive Library.

This library provides tools for parsing, storing, and querying AI conversation
exports from ChatGPT, Claude, Codex, Gemini, and other AI assistants.

Example::

    from polylogue import Polylogue

    async with Polylogue() as archive:
        # Statistics
        stats = await archive.stats()
        print(f"{stats.conversation_count} conversations")

        # Query conversations
        convs = await archive.filter().provider("claude-ai").since("2024-01-01").list()
        for conv in convs:
            print(f"{conv.display_title}: {conv.message_count} messages")

        # Search
        results = await archive.search("python error handling")
        for hit in results.hits:
            print(hit.title)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.errors import PolylogueError
    from polylogue.facade import ArchiveStats, Polylogue
    from polylogue.lib.attribution import ConversationAttribution
    from polylogue.lib.coverage import ArchiveCoverage
    from polylogue.lib.filters import ConversationFilter
    from polylogue.lib.models import Conversation, Message
    from polylogue.lib.phases import SessionPhase
    from polylogue.lib.pricing import ModelPricing, estimate_cost, harmonize_session_cost
    from polylogue.lib.session_profile import SessionProfile
    from polylogue.lib.summaries import DaySessionSummary, WeekSessionSummary
    from polylogue.lib.tagging import infer_tags
    from polylogue.lib.threads import WorkThread, build_session_threads
    from polylogue.lib.work_events import WorkEvent, WorkEventKind
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.search import SearchResult
    from polylogue.sync import SyncPolylogue


def __getattr__(name: str) -> object:
    if name == "ArchiveStats":
        from polylogue.facade import ArchiveStats

        return ArchiveStats
    if name == "Conversation":
        from polylogue.lib.models import Conversation

        return Conversation
    if name == "ConversationAttribution":
        from polylogue.lib.attribution import ConversationAttribution

        return ConversationAttribution
    if name == "Decision":
        from polylogue.lib.decisions import Decision

        return Decision
    if name == "ConversationFilter":
        from polylogue.lib.filters import ConversationFilter

        return ConversationFilter
    if name == "ConversationRepository":
        from polylogue.storage.repository import ConversationRepository

        return ConversationRepository
    if name == "ArchiveCoverage":
        from polylogue.lib.coverage import ArchiveCoverage

        return ArchiveCoverage
    if name == "Message":
        from polylogue.lib.models import Message

        return Message
    if name == "Polylogue":
        from polylogue.facade import Polylogue

        return Polylogue
    if name == "PolylogueError":
        from polylogue.errors import PolylogueError

        return PolylogueError
    if name == "SearchResult":
        from polylogue.storage.search import SearchResult

        return SearchResult
    if name == "SessionPhase":
        from polylogue.lib.phases import SessionPhase

        return SessionPhase
    if name == "SessionProfile":
        from polylogue.lib.session_profile import SessionProfile

        return SessionProfile
    if name == "DaySessionSummary":
        from polylogue.lib.summaries import DaySessionSummary

        return DaySessionSummary
    if name == "WeekSessionSummary":
        from polylogue.lib.summaries import WeekSessionSummary

        return WeekSessionSummary
    if name == "SyncPolylogue":
        from polylogue.sync import SyncPolylogue

        return SyncPolylogue
    if name == "WorkEvent":
        from polylogue.lib.work_events import WorkEvent

        return WorkEvent
    if name == "WorkEventKind":
        from polylogue.lib.work_events import WorkEventKind

        return WorkEventKind
    if name == "ModelPricing":
        from polylogue.lib.pricing import ModelPricing

        return ModelPricing
    if name == "estimate_cost":
        from polylogue.lib.pricing import estimate_cost

        return estimate_cost
    if name == "harmonize_session_cost":
        from polylogue.lib.pricing import harmonize_session_cost

        return harmonize_session_cost
    if name == "infer_tags":
        from polylogue.lib.tagging import infer_tags

        return infer_tags
    if name == "WorkThread":
        from polylogue.lib.threads import WorkThread

        return WorkThread
    if name == "build_session_threads":
        from polylogue.lib.threads import build_session_threads

        return build_session_threads
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ArchiveCoverage",
    "ArchiveStats",
    "Conversation",
    "ConversationAttribution",
    "ConversationFilter",
    "ConversationRepository",
    "DaySessionSummary",
    "Decision",
    "Message",
    "ModelPricing",
    "Polylogue",
    "PolylogueError",
    "SearchResult",
    "SessionPhase",
    "SessionProfile",
    "SyncPolylogue",
    "WeekSessionSummary",
    "WorkEvent",
    "WorkEventKind",
    "WorkThread",
    "build_session_threads",
    "estimate_cost",
    "harmonize_session_cost",
    "infer_tags",
]

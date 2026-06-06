"""Session filtering helpers for immutable session query plans."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, TypeVar

from polylogue.archive.message.types import MessageType, validate_message_type_filter
from polylogue.archive.query.path_prefix import path_matches_prefix
from polylogue.archive.query.runtime_matching import (
    matches_action_sequence,
    matches_action_terms,
    matches_action_text_terms,
    matches_referenced_path,
    matches_tool_terms,
)
from polylogue.archive.query.support import session_has_branches

if TYPE_CHECKING:
    from polylogue.archive.models import Session
    from polylogue.archive.query.plan import SessionQueryPlan


class FilterableSessionLike(Protocol):
    @property
    def origin(self) -> object: ...

    @property
    def updated_at(self) -> datetime | None: ...

    @property
    def display_title(self) -> str: ...

    @property
    def parent_id(self) -> str | None: ...

    @property
    def tags(self) -> Sequence[str]: ...

    @property
    def id(self) -> str: ...

    @property
    def summary(self) -> str | None: ...

    @property
    def is_continuation(self) -> bool: ...

    @property
    def is_sidechain(self) -> bool: ...

    @property
    def is_root(self) -> bool: ...


_T = TypeVar("_T", bound=FilterableSessionLike)


def apply_common_filters(
    plan: SessionQueryPlan,
    items: list[_T],
    *,
    sql_pushed: bool,
) -> list[_T]:
    results = list(items)

    if not sql_pushed:
        origin_set = set(plan.origins)
        if origin_set:
            results = [item for item in results if str(item.origin) in origin_set]
        if plan.since:
            results = [item for item in results if item.updated_at and item.updated_at >= plan.since]
        if plan.until:
            results = [item for item in results if item.updated_at and item.updated_at <= plan.until]
        if plan.title:
            lowered = plan.title.lower()
            results = [item for item in results if item.display_title and lowered in item.display_title.lower()]
        if plan.parent_id:
            results = [item for item in results if str(item.parent_id or "") == plan.parent_id]

    if plan.excluded_origins:
        excluded = set(plan.excluded_origins)
        results = [item for item in results if str(item.origin) not in excluded]
    if plan.tags:
        tag_set = set(plan.tags)
        results = [item for item in results if tag_set.intersection(item.tags)]
    if plan.excluded_tags:
        excluded_tags = set(plan.excluded_tags)
        results = [item for item in results if not excluded_tags.intersection(item.tags)]
    if plan.session_id:
        results = [item for item in results if str(item.id).startswith(plan.session_id)]
    if "summary" in plan.has_types:
        results = [item for item in results if item.summary]
    if plan.continuation is True:
        results = [item for item in results if item.is_continuation]
    if plan.continuation is False:
        results = [item for item in results if not item.is_continuation]
    if plan.sidechain is True:
        results = [item for item in results if item.is_sidechain]
    if plan.sidechain is False:
        results = [item for item in results if not item.is_sidechain]
    if plan.root is True:
        results = [item for item in results if item.is_root]
    if plan.root is False:
        results = [item for item in results if not item.is_root]

    return results


def _has_negative_term(session: Session, negative_terms: list[str]) -> bool:
    for message in session.messages:
        if not message.text:
            continue
        lowered = message.text.lower()
        for term in negative_terms:
            if term in lowered:
                return True
    return False


def apply_full_filters(
    plan: SessionQueryPlan,
    sessions: list[Session],
    *,
    sql_pushed: bool,
) -> list[Session]:
    results = apply_common_filters(plan, sessions, sql_pushed=sql_pushed)

    if plan.has_types:
        for content_type in plan.has_types:
            if content_type == "thinking":
                results = [c for c in results if any(m.is_thinking for m in c.messages)]
            elif content_type == "tools":
                results = [c for c in results if any(m.is_tool_use for m in c.messages)]
            elif content_type == "attachments":
                results = [c for c in results if any(m.attachments for m in c.messages)]

    if plan.filter_has_tool_use:
        results = [c for c in results if any(m.is_tool_use for m in c.messages)]
    if plan.filter_has_thinking:
        results = [c for c in results if any(m.is_thinking for m in c.messages)]
    if plan.min_messages is not None:
        results = [c for c in results if len(c.messages) >= plan.min_messages]
    if plan.max_messages is not None:
        results = [c for c in results if len(c.messages) <= plan.max_messages]
    if plan.min_words is not None:
        results = [c for c in results if sum(len((m.text or "").split()) for m in c.messages) >= plan.min_words]
    if plan.since_session_id:
        scoped_ids = {str(session.id) for session in _apply_since_session(sessions, plan.since_session_id)}
        results = [session for session in results if str(session.id) in scoped_ids]

    if plan.message_type is not None:
        wanted_type = validate_message_type_filter(plan.message_type)
        results = [
            c
            for c in results
            if any(
                MessageType.normalize(getattr(m, "message_type", MessageType.MESSAGE)) == wanted_type
                for m in c.messages
            )
        ]

    if plan.cwd_prefix:
        results = [session for session in results if _session_matches_cwd_prefix(session, plan.cwd_prefix)]

    if plan.negative_terms:
        negative_terms = [term.lower() for term in plan.negative_terms]
        results = [session for session in results if not _has_negative_term(session, negative_terms)]

    if plan.has_branches is True:
        results = [item for item in results if session_has_branches(item)]
    if plan.has_branches is False:
        results = [item for item in results if not session_has_branches(item)]

    for predicate in plan.predicates:
        results = [session for session in results if predicate(session)]

    if plan.referenced_path:
        results = [session for session in results if matches_referenced_path(plan, session)]
    if plan.action_terms or plan.excluded_action_terms:
        results = [session for session in results if matches_action_terms(plan, session)]
    if plan.action_sequence:
        results = [session for session in results if matches_action_sequence(plan, session)]
    if plan.action_text_terms:
        results = [session for session in results if matches_action_text_terms(plan, session)]
    if plan.tool_terms or plan.excluded_tool_terms:
        results = [session for session in results if matches_tool_terms(plan, session)]

    return results


def _apply_since_session(
    sessions: list[Session],
    reference_id: str,
) -> list[Session]:
    """Filter to sessions in same cwd after the reference session's last message."""
    reference_conv = None
    for c in sessions:
        if str(c.id) == reference_id or str(c.id).startswith(reference_id):
            reference_conv = c
            break

    if reference_conv is None:
        return []

    ref_cwds: list[str] = []
    ref_meta = getattr(reference_conv, "provider_meta", None) or {}
    if isinstance(ref_meta, dict):
        wds = ref_meta.get("working_directories") or []
        ref_cwds = [str(wd) for wd in wds if isinstance(wd, str) and wd]

    last_ts = reference_conv.updated_at
    if reference_conv.messages:
        last_msg_ts = max(
            (m.timestamp for m in reference_conv.messages if m.timestamp),
            default=None,
        )
        if last_msg_ts:
            last_ts = last_msg_ts

    results: list[Session] = []
    for c in sessions:
        if str(c.id) == str(reference_conv.id):
            continue
        if last_ts and c.updated_at and c.updated_at <= last_ts:
            continue
        if ref_cwds:
            c_meta = getattr(c, "provider_meta", None) or {}
            if isinstance(c_meta, dict):
                c_wds = c_meta.get("working_directories") or []
                c_wd_strs = [str(wd) for wd in c_wds if isinstance(wd, str) and wd]
                if not c_wd_strs:
                    continue
                if not any(path_matches_prefix(cwd, ref_cwd) for cwd in c_wd_strs for ref_cwd in ref_cwds):
                    continue
        results.append(c)

    return results


def _session_matches_cwd_prefix(session: Session, cwd_prefix: str) -> bool:
    meta = getattr(session, "provider_meta", None) or {}
    if not isinstance(meta, dict):
        return False
    cwd_values = meta.get("working_directories") or []
    return any(path_matches_prefix(cwd, cwd_prefix) for cwd in cwd_values if isinstance(cwd, str) and cwd)


__all__ = ["FilterableSessionLike", "apply_common_filters", "apply_full_filters"]

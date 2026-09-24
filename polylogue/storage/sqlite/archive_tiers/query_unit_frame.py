"""Relation-scoped continuation frames for query-unit paging.

A continuation must be invalidated when the archive moves *under the rows it
served* -- and only then. A single archive-wide counter cannot express that:
every committed write advances it, so an unrelated derived-relation rewrite
(the session-profile sweep the daemon runs on its own schedule) turns the
next page of an ongoing read into a ``query_continuation_stale`` 409.

The frame is therefore a vector: one epoch per tracked relation. A lowered
request declares the relations it reads, the continuation carries only those
components, and validation compares only what the continuation carries.

Declaring the vocabulary here keeps three things from drifting apart: the
index-tier DDL that seeds the rows, the triggers that advance them, and the
reader that parses a continuation's frame.
"""

from __future__ import annotations

from typing import Literal, get_args

# Index-tier relations whose contents can change which terminal rows a
# query-unit page returns, or in what order.
IndexFrameRelation = Literal[
    "action_pairs",
    "blocks",
    "delegation_facts",
    "messages",
    "repos",
    "session_links",
    "session_profiles",
    "session_repos",
    "session_tags",
    "sessions",
    "session_working_dirs",
]

# User-tier relations with the same property. The user tier is durable and
# keeps its single-relation singleton row (reshaping it would be a durable
# migration for no gain: one tracked relation is already its own scope), so
# this name exists only in the composed frame, not in a user-tier column.
UserFrameRelation = Literal["assertions"]

FrameRelation = IndexFrameRelation | UserFrameRelation

INDEX_FRAME_RELATIONS: tuple[str, ...] = tuple(sorted(get_args(IndexFrameRelation)))
USER_FRAME_RELATIONS: tuple[str, ...] = tuple(sorted(get_args(UserFrameRelation)))
ALL_FRAME_RELATIONS: tuple[str, ...] = tuple(sorted((*INDEX_FRAME_RELATIONS, *USER_FRAME_RELATIONS)))


def index_frame_seed_sql() -> str:
    """Return the INSERT seeding one frame row per tracked index relation."""
    values = ", ".join(f"('{relation}', 0)" for relation in INDEX_FRAME_RELATIONS)
    return f"INSERT OR IGNORE INTO query_unit_frame_state(relation, epoch) VALUES {values};"


def index_frame_bump_sql(relation: str) -> str:
    """Return the trigger body advancing exactly one relation's epoch.

    Every epoch-bumping trigger in the index tier routes through this helper,
    so a new tracked relation cannot be added with a body that still points
    at some other relation's row.
    """
    if relation not in INDEX_FRAME_RELATIONS:
        raise ValueError(f"unknown query-unit frame relation {relation!r}; expected one of {INDEX_FRAME_RELATIONS}")
    return f"UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE relation = '{relation}';"


__all__ = [
    "ALL_FRAME_RELATIONS",
    "INDEX_FRAME_RELATIONS",
    "USER_FRAME_RELATIONS",
    "FrameRelation",
    "IndexFrameRelation",
    "UserFrameRelation",
    "index_frame_bump_sql",
    "index_frame_seed_sql",
]

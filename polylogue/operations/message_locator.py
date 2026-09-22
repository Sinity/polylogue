"""Resolve one message reference to the transcript window that contains it.

A deep link (``/s/<session>#msg-<message>``) names a *message*, not a page.
Without this module the only way a surface could honour that link was to walk
pages from the top until the message showed up, which has two observable
costs: a target past the walker's page ceiling makes a **valid, resolvable**
reference report as unlocatable, and a target inside the ceiling still fetches
and retains every message before it.

The locator answers the one question that removes the walk -- what is this
message's ordinal index in the session's composed transcript? -- so the caller
can ask for exactly the window that holds it.

Composition order here is the storage layer's own ``(position, variant_index)``
order, which is the order ``read_archive_session_page`` windows with. The index
this module returns and the window a page read delivers therefore cannot
disagree; an index derived from any other ordering would silently land the
reader on the wrong page.

This module owns no window arithmetic beyond aligning an index onto the
caller's declared window: ``operations/transcript_window.py`` stays the one
owner of ``[offset, offset + limit)``, ``next_offset`` and continuations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "MessageLocation",
    "MessageNotInSessionError",
    "locate_message_in_archive",
    "window_offset_around",
    "window_offset_for_index",
]


class MessageNotInSessionError(LookupError, ValueError):
    """The named message is not part of the named session's transcript.

    This is a refusal, not an empty answer: the caller asked for the window
    around a message this session does not contain, and there is no window
    that honestly satisfies that request. Answering page zero instead would
    hand back a *different* message's window under the caller's reference.

    It is both a ``LookupError`` -- what the HTTP route already catches to map
    it onto ``404`` -- and a ``ValueError``, which is the declared read
    operation's vocabulary for "the request was refused". One class with one
    ``code`` therefore reaches every surface, instead of the operation route
    escaping the kernel's error map and reaching the operator as a traceback
    (polylogue-idrej).
    """

    code = "message_not_found"

    def __init__(self, session_id: str, message_id: str) -> None:
        super().__init__(f"message {message_id!r} is not part of session {session_id!r}")
        self.session_id = session_id
        self.message_id = message_id


@dataclass(frozen=True, slots=True)
class MessageLocation:
    """Where one message sits in its session's composed transcript."""

    session_id: str
    message_id: str
    index: int


def window_offset_for_index(index: int, limit: int) -> int:
    """Return the offset of the ``limit``-sized window holding ``index``.

    Windows are aligned to the caller's declared page size rather than
    centred on the target, so a deep-linked window and the window the same
    reader would reach by paging are the *same* window -- otherwise "load
    more" from a deep link would re-deliver rows the reader already has.
    """

    if limit <= 0:
        raise ValueError("a window offset needs a positive window size")
    if index < 0:
        raise ValueError("a message index is never negative")
    return index - (index % limit)


def window_offset_around(archive: Any, session_id: str, message_id: str, limit: int) -> int:
    """Return the offset of the ``limit``-sized window holding ``message_id``.

    The locate and the alignment are one step because separating them is what
    lets two surfaces align the same index differently -- and an ``around``
    window that is not the window the same reader reaches by paging would make
    "load more" re-deliver rows it already has. Every surface that accepts an
    anchor calls this.
    """

    return window_offset_for_index(locate_message_in_archive(archive, session_id, message_id).index, limit)


def locate_message_in_archive(archive: Any, session_id: str, message_id: str) -> MessageLocation:
    """Locate ``message_id`` in ``session_id`` against an open archive reader.

    The storage layer owns the answer (``ArchiveStore.locate_composed_message``)
    because the composed transcript is its composition: a plain session is
    numbered by two indexed counts, and a prefix-sharing lineage child -- whose
    composed transcript is the ancestral prefix plus its own divergent tail --
    is numbered from the same segment plan ``read_archive_session_page``
    windows with, one indexed count per ancestor (polylogue-2go3o). Neither
    shape composes a transcript to number one message.
    """

    index = archive.locate_composed_message(session_id, message_id)
    if index is None:
        raise MessageNotInSessionError(session_id, message_id)
    return MessageLocation(session_id=session_id, message_id=message_id, index=int(index))

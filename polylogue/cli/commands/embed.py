"""Embedding generation helpers.

Business logic for the ``run embed`` stage subcommand.  The Click
command decorator was removed when embed moved from a standalone root
command to ``polylogue run embed``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.cli.embed_runtime import embed_batch, embed_single
from polylogue.cli.embed_stats import EmbeddingStatusPayload, embedding_status_payload, render_embedding_stats

if TYPE_CHECKING:
    from polylogue.cli.types import AppEnv

_embed_single = embed_single
_embed_batch = embed_batch


def _embedding_status_payload(env: AppEnv) -> EmbeddingStatusPayload:
    return embedding_status_payload(env)


def _show_embedding_stats(env: AppEnv, *, json_output: bool = False) -> None:
    render_embedding_stats(_embedding_status_payload(env), json_output=json_output)


__all__ = [
    "_embed_batch",
    "_embed_single",
    "_embedding_status_payload",
    "_show_embedding_stats",
]

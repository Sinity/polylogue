"""Shared helpers for derived-model status assembly."""

from __future__ import annotations


def pending_rows(source_rows: int, materialized_rows: int) -> int:
    return max(0, source_rows - materialized_rows)


def pending_docs(source_docs: int, materialized_docs: int) -> int:
    return max(0, source_docs - materialized_docs)


__all__ = ["pending_docs", "pending_rows"]

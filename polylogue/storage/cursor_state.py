"""Cursor payload contracts for acquisition and source traversal."""

from __future__ import annotations

from typing_extensions import TypedDict


class CursorFailurePayload(TypedDict):
    path: str
    error: str


class CursorStatePayload(TypedDict, total=False):
    file_count: int
    error_count: int
    latest_mtime: float
    latest_file_name: str
    latest_path: str
    latest_file_id: str
    latest_error: str
    latest_error_file: str
    failed_count: int
    failed_files: list[CursorFailurePayload]


__all__ = ["CursorFailurePayload", "CursorStatePayload"]

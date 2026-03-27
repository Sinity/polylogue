"""Governance-oriented derived product storage models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

from .store_core import MAINTENANCE_RUN_SCHEMA_VERSION


class MaintenanceRunRecord(BaseModel):
    maintenance_run_id: str
    schema_version: int = MAINTENANCE_RUN_SCHEMA_VERSION
    executed_at: str
    mode: str
    preview: bool = False
    repair_selected: bool = False
    cleanup_selected: bool = False
    vacuum_requested: bool = False
    target_names: tuple[str, ...] = ()
    success: bool = True
    manifest: dict[str, Any]

    @field_validator("maintenance_run_id", "executed_at", "mode")
    @classmethod
    def maintenance_non_empty_string(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value


__all__ = ["MaintenanceRunRecord"]

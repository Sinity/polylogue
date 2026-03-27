"""Public validation result models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of schema validation with drift detection."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    drift_warnings: list[str] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return len(self.drift_warnings) > 0

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            raise ValueError(f"Schema validation failed: {'; '.join(self.errors)}")

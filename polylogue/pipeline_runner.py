from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

class PipelineStage(Protocol):
    """Stage interface for pipeline execution."""

    def run(self, context: "PipelineContext") -> None:
        ...


@dataclass
class PipelineContext:
    """Shared context passed between pipeline stages."""

    env: Any
    options: Any
    data: Dict[str, Any] = field(default_factory=dict)
    aborted: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.data:
            raise KeyError(f"Pipeline context missing required key '{key}'")
        return self.data[key]

    def abort(self) -> None:
        self.aborted = True

    def record_stage(
        self,
        *,
        name: str,
        status: str,
        duration: float,
        error: Optional[BaseException] = None,
    ) -> None:
        entry = {
            "name": name,
            "status": status,
            "duration": duration,
        }
        if error is not None:
            entry["error"] = repr(error)
            self.errors.append(f"{name}: {error}")
        self.history.append(entry)


class Pipeline:
    """Simple synchronous pipeline runner."""

    def __init__(self, stages: Iterable[PipelineStage]):
        self._stages: List[PipelineStage] = list(stages)

    def run(self, context: PipelineContext) -> PipelineContext:
        for stage in self._stages:
            if context.aborted:
                break
            stage_name = getattr(stage, "name", stage.__class__.__name__)
            started = time.perf_counter()
            try:
                stage.run(context)
            except Exception as exc:  # pragma: no cover - bubbled to caller with trace recorded
                duration = time.perf_counter() - started
                context.record_stage(name=stage_name, status="error", duration=duration, error=exc)
                context.abort()
                raise
            else:
                duration = time.perf_counter() - started
                context.record_stage(name=stage_name, status="ok", duration=duration)
        return context

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class StateStore:
    path: Path

    def load(self) -> dict:
        try:
            if self.path.exists():
                return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def save(self, state: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass

    def mutate(self, mutator: Callable[[dict], None]) -> dict:
        state = self.load()
        mutator(state)
        self.save(state)
        return state


__all__ = ["StateStore"]

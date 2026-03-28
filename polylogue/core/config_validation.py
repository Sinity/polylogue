from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class QdrantModel(BaseModel):
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection: Optional[str] = None
    vector_size: Optional[int] = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")


class IndexModel(BaseModel):
    backend: Optional[str] = None
    qdrant: Optional[QdrantModel] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def ensure_backend(cls, values: "IndexModel") -> "IndexModel":
        backend = (values.backend or "sqlite").strip().lower()
        if backend not in {"sqlite", "qdrant", "none"}:
            raise ValueError("index.backend must be sqlite, qdrant, or none")
        values.backend = backend
        return values


class ExportsModel(BaseModel):
    chatgpt: Optional[str] = None
    claude: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class UiModel(BaseModel):
    collapse_threshold: Optional[int] = Field(default=None, ge=1)
    html: Optional[bool] = None
    theme: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class PathsModel(BaseModel):
    input_root: Optional[str] = None
    output_root: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    def expand(self) -> Dict[str, Path]:
        data: Dict[str, Path] = {}
        for key, value in self.model_dump(exclude_none=True).items():
            data[key] = Path(value).expanduser()
        return data


class ConfigModel(BaseModel):
    paths: Optional[PathsModel] = None
    ui: Optional[UiModel] = None
    defaults: Optional[UiModel] = None
    index: Optional[IndexModel] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def ensure_theme(cls, values: "ConfigModel") -> "ConfigModel":
        ui = values.ui or values.defaults
        theme = ui.theme if ui else None
        if theme and theme not in {"light", "dark"}:
            raise ValueError("ui.theme must be 'light' or 'dark'")
        return values


def validate_config_payload(payload: Dict[str, Any]) -> None:
    try:
        ConfigModel.model_validate(payload)
    except ValidationError as exc:
        raise SystemExit(f"Invalid polylogue config: {exc}") from exc

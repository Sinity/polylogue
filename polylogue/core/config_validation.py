from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class OutputDirsModel(BaseModel):
    render: Optional[str] = None
    sync_drive: Optional[str] = None
    sync_codex: Optional[str] = None
    sync_claude_code: Optional[str] = None
    import_chatgpt: Optional[str] = None
    import_claude: Optional[str] = None

    model_config = ConfigDict(extra="forbid")

    def expand(self) -> Dict[str, Path]:
        data: Dict[str, Path] = {}
        for key, value in self.model_dump(exclude_none=True).items():
            data[key] = Path(value).expanduser()
        return data


class DriveModel(BaseModel):
    credentials_path: Optional[str] = None
    token_path: Optional[str] = None
    retries: Optional[int] = Field(default=None, ge=1)
    retry_base: Optional[float] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


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


class DefaultsModel(BaseModel):
    collapse_threshold: Optional[int] = Field(default=None, ge=1)
    html_previews: Optional[bool] = None
    html_theme: Optional[str] = None
    output_dirs: Optional[OutputDirsModel] = None

    model_config = ConfigDict(extra="forbid")


class ConfigModel(BaseModel):
    drive: Optional[DriveModel] = None
    index: Optional[IndexModel] = None
    exports: Optional[ExportsModel] = None
    defaults: Optional[DefaultsModel] = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def ensure_theme(cls, values: "ConfigModel") -> "ConfigModel":
        theme = values.defaults.html_theme if values.defaults else None
        if theme and theme not in {"light", "dark"}:
            raise ValueError("defaults.html_theme must be 'light' or 'dark'")
        return values


def validate_config_payload(payload: Dict[str, Any]) -> None:
    try:
        ConfigModel.model_validate(payload)
    except ValidationError as exc:
        raise SystemExit(f"Invalid polylogue config: {exc}") from exc

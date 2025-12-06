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


class DefaultsModel(BaseModel):
    collapse_threshold: Optional[int] = Field(default=None, ge=1)
    html_previews: Optional[bool] = None
    html_theme: Optional[str] = None
    output_dirs: Optional[OutputDirsModel] = None

    model_config = ConfigDict(extra="forbid")


class ConfigModel(BaseModel):
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

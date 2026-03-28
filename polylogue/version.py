from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version as metadata_version
from pathlib import Path


def _resolve_version() -> str:
    """Resolve the Polylogue version from package metadata or pyproject.toml.

    This keeps `--version` aligned with releases even from source checkouts.
    """
    try:
        return metadata_version("polylogue")
    except PackageNotFoundError:
        pass
    except Exception:
        pass

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        try:
            text = pyproject_path.read_text(encoding="utf-8")
            match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if match:
                return match.group(1)
        except Exception:
            pass

    return "unknown"


POLYLOGUE_VERSION = _resolve_version()

__all__ = ["POLYLOGUE_VERSION", "_resolve_version"]

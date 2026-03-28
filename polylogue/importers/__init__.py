"""Importer package for third-party conversation formats."""

from .base import ImportResult

__all__ = [
    "ImportResult",
    "import_chatgpt_export",
    "import_claude_export",
    "import_claude_code_session",
    "import_codex_session",
]


def __getattr__(name):
    if name == "import_chatgpt_export":
        from .chatgpt import import_chatgpt_export as loader

        return loader
    if name == "import_claude_export":
        from .claude_ai import import_claude_export as loader

        return loader
    if name == "import_claude_code_session":
        from .claude_code import import_claude_code_session as loader

        return loader
    if name == "import_codex_session":
        from .codex import import_codex_session as loader

        return loader
    raise AttributeError(name)


def __dir__():
    return sorted(__all__)

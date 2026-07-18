"""Shared color palette and theme constants.

Single source of truth for provider brand colors, role colors,
status colors, and UI theme tokens used across CLI, HTML rendering,
and daemon web surfaces.

Theme mode resolution (#1274):

The active theme — ``"dark"``, ``"light"``, or auto-detected — is resolved
via :func:`resolve_theme_mode`. Precedence (highest first):

1. ``POLYLOGUE_THEME`` environment variable (``dark``/``light``/``auto``).
2. ``[ui] theme`` in ``polylogue.toml`` (same values).
3. ``COLORFGBG`` / ``TERM_BACKGROUND`` hints when set to ``auto``.
4. Default: ``"dark"``.

``POLYLOGUE_FORCE_PLAIN`` and ``NO_COLOR`` are orthogonal — plain mode strips
Rich layout entirely, independent of the resolved theme.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Origin

ThemeMode = Literal["dark", "light"]

# =============================================================================
# Provider brand colors
# =============================================================================


@dataclass(frozen=True)
class ProviderColor:
    """A provider's brand color in multiple formats."""

    hex: str
    """Hex color value (e.g. '#d97757')."""

    @property
    def rgb(self) -> tuple[int, int, int]:
        """RGB tuple from hex."""
        h = self.hex.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def css_bg(self, alpha: float = 0.1) -> str:
        """CSS rgba background value."""
        r, g, b = self.rgb
        return f"rgba({r}, {g}, {b}, {alpha})"

    def css_border(self, alpha: float = 0.3) -> str:
        """CSS rgba border value."""
        return self.css_bg(alpha)


PROVIDER_COLORS: dict[str, ProviderColor] = {
    "claude-ai": ProviderColor("#d97757"),
    "claude-code": ProviderColor("#d97757"),
    "chatgpt": ProviderColor("#10a37f"),
    "gemini": ProviderColor("#4285f4"),
    "codex": ProviderColor("#00bcd4"),
    "google-ai-studio": ProviderColor("#4285f4"),
}

DEFAULT_PROVIDER_COLOR = ProviderColor("#e5e7eb")


def provider_color(name: str) -> ProviderColor:
    """Look up a provider color by name, with fuzzy matching.

    Matches on substring so 'claude-code' matches 'claude-ai',
    'openai-codex' matches 'codex', etc.
    """
    # Exact match first
    if name in PROVIDER_COLORS:
        return PROVIDER_COLORS[name]
    # Substring match
    for key, color in PROVIDER_COLORS.items():
        if key in name:
            return color
    return DEFAULT_PROVIDER_COLOR


# =============================================================================
# Role colors
# =============================================================================


@dataclass(frozen=True)
class RoleColor:
    """A role's color in multiple formats."""

    hex: str
    """Base hex color."""

    label: str
    """Rich markup style for role label (e.g. 'bold #6366f1')."""

    @property
    def rgb(self) -> tuple[int, int, int]:
        h = self.hex.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def css_bg(self, alpha: float = 0.1) -> str:
        r, g, b = self.rgb
        return f"rgba({r}, {g}, {b}, {alpha})"

    def css_border(self, alpha: float = 0.3) -> str:
        return self.css_bg(alpha)


ROLE_COLORS: dict[str, RoleColor] = {
    "user": RoleColor(hex="#6366f1", label="bold #6366f1"),
    "assistant": RoleColor(hex="#10b981", label="bold #10b981"),
    "system": RoleColor(hex="#f59e0b", label="bold #f59e0b"),
    "tool": RoleColor(hex="#8b5cf6", label="bold #8b5cf6"),
}

DEFAULT_ROLE_COLOR = RoleColor(hex="#94a3b8", label="#94a3b8")


def role_color(role: str | Role) -> RoleColor:
    """Look up a role color."""
    if isinstance(role, Role):
        normalized = role
    else:
        raw = str(role).strip()
        normalized = Role.normalize(raw) if raw else Role.UNKNOWN
    return ROLE_COLORS.get(str(normalized), DEFAULT_ROLE_COLOR)


# =============================================================================
# Status colors
# =============================================================================

STATUS_COLORS = {
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "info": "#38bdf8",
}


# =============================================================================
# UI theme tokens (dark theme)
# =============================================================================

DARK_THEME = {
    # Backgrounds
    "bg_primary": "#0a0a0c",
    "bg_secondary": "#16161a",
    "bg_elevated": "#1e1e24",
    "bg_code": "#282c34",
    # Text
    "text_primary": "#f8f9fa",
    "text_secondary": "#94a3b8",
    "text_muted": "#6b7280",
    # Accent
    "accent": "#6366f1",
    "accent_glow": "rgba(99, 102, 241, 0.4)",
    # Borders
    "border": "#2d2d35",
    "border_subtle": "#1f1f23",
    # Special
    "glass": "rgba(255, 255, 255, 0.03)",
    "glass_border": "rgba(255, 255, 255, 0.1)",
}

LIGHT_THEME = {
    "bg_primary": "#ffffff",
    "bg_secondary": "#f9fafb",
    "bg_elevated": "#ffffff",
    "bg_code": "#f6f8fa",
    "text_primary": "#111827",
    "text_secondary": "#4b5563",
    "text_muted": "#9ca3af",
    "accent": "#6366f1",
    "accent_glow": "rgba(99, 102, 241, 0.2)",
    "border": "#e5e7eb",
    "border_subtle": "#f3f4f6",
    "glass": "rgba(0, 0, 0, 0.02)",
    "glass_border": "rgba(0, 0, 0, 0.08)",
}


# =============================================================================
# WebUI v2 generated design-system tokens
# =============================================================================

# The browser design system is generated from these Python-owned values by
# ``devtools render webui-design-system``.  Keeping the public Origin list and
# palette here prevents the TypeScript client from growing a second provider
# vocabulary while still leaving archive semantics on the server.
PUBLIC_ORIGIN_TOKENS: tuple[Origin, ...] = tuple(origin for origin in Origin if origin is not Origin.UNKNOWN_EXPORT)

WEBUI_SHARED_TOKENS: dict[str, str] = {
    "--pl-font-sans": 'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    "--pl-font-mono": 'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace',
    "--pl-font-size-xs": "0.75rem",
    "--pl-font-size-sm": "0.8125rem",
    "--pl-font-size-md": "0.9375rem",
    "--pl-font-size-lg": "1.125rem",
    "--pl-font-size-xl": "1.5rem",
    "--pl-line-height-tight": "1.25",
    "--pl-line-height-body": "1.55",
    "--pl-space-0": "0",
    "--pl-space-1": "0.25rem",
    "--pl-space-2": "0.5rem",
    "--pl-space-3": "0.75rem",
    "--pl-space-4": "1rem",
    "--pl-space-5": "1.5rem",
    "--pl-space-6": "2rem",
    "--pl-space-7": "3rem",
    "--pl-space-8": "4rem",
    "--pl-radius-sm": "0.25rem",
    "--pl-radius-md": "0.5rem",
    "--pl-radius-lg": "0.75rem",
    "--pl-border-width": "1px",
    "--pl-focus-width": "3px",
    "--pl-density-row-comfortable": "2.75rem",
    "--pl-density-row-compact": "2.125rem",
    "--pl-content-measure": "76rem",
    "--pl-transcript-measure": "72ch",
    "--pl-motion-fast": "120ms",
    "--pl-motion-normal": "180ms",
}

WEBUI_THEME_TOKENS: dict[ThemeMode, dict[str, str]] = {
    "dark": {
        "--pl-color-bg": "#0b1015",
        "--pl-color-surface": "#111820",
        "--pl-color-surface-raised": "#18222d",
        "--pl-color-surface-inset": "#0d141b",
        "--pl-color-text": "#eef4f8",
        "--pl-color-text-muted": "#b3c0cb",
        "--pl-color-text-subtle": "#8998a6",
        "--pl-color-border": "#344453",
        "--pl-color-border-strong": "#526578",
        "--pl-color-accent": "#7dd3fc",
        "--pl-color-accent-strong": "#38bdf8",
        "--pl-color-focus": "#fbbf24",
        "--pl-color-selection": "#14354a",
        "--pl-color-code-bg": "#090e13",
        "--pl-color-shadow": "rgba(0, 0, 0, 0.42)",
    },
    "light": {
        "--pl-color-bg": "#f7f9fb",
        "--pl-color-surface": "#ffffff",
        "--pl-color-surface-raised": "#eef2f6",
        "--pl-color-surface-inset": "#f3f6f8",
        "--pl-color-text": "#18212b",
        "--pl-color-text-muted": "#465564",
        "--pl-color-text-subtle": "#607080",
        "--pl-color-border": "#b8c3ce",
        "--pl-color-border-strong": "#7b8a99",
        "--pl-color-accent": "#075985",
        "--pl-color-accent-strong": "#0369a1",
        "--pl-color-focus": "#92400e",
        "--pl-color-selection": "#d9edf7",
        "--pl-color-code-bg": "#f1f5f9",
        "--pl-color-shadow": "rgba(15, 23, 42, 0.16)",
    },
}

WEBUI_EVIDENCE_BADGE_TOKENS: dict[ThemeMode, dict[str, tuple[str, str]]] = {
    "dark": {
        "exact": ("#9ae6b4", "#123522"),
        "qualified": ("#93c5fd", "#102a43"),
        "stale": ("#fde68a", "#3d2e0c"),
        "unknown": ("#d1d5db", "#29313a"),
        "degraded": ("#fda4af", "#3b121a"),
    },
    "light": {
        "exact": ("#166534", "#dcfce7"),
        "qualified": ("#1e40af", "#dbeafe"),
        "stale": ("#854d0e", "#fef3c7"),
        "unknown": ("#374151", "#e5e7eb"),
        "degraded": ("#9f1239", "#ffe4e6"),
    },
}

WEBUI_ORIGIN_BADGE_TOKENS: dict[ThemeMode, dict[Origin, tuple[str, str]]] = {
    "dark": {
        Origin.CLAUDE_CODE_SESSION: ("#fdba74", "#3b2410"),
        Origin.CODEX_SESSION: ("#67e8f9", "#12313a"),
        Origin.GEMINI_CLI_SESSION: ("#93c5fd", "#142b4d"),
        Origin.HERMES_SESSION: ("#c4b5fd", "#2b1f4a"),
        Origin.ANTIGRAVITY_SESSION: ("#f9a8d4", "#421b34"),
        Origin.BEADS_ISSUE: ("#fde68a", "#3d2e0c"),
        Origin.GROK_EXPORT: ("#d1d5db", "#29313a"),
        Origin.CHATGPT_EXPORT: ("#86efac", "#123522"),
        Origin.CLAUDE_AI_EXPORT: ("#fed7aa", "#3b2410"),
        Origin.AISTUDIO_DRIVE: ("#a5b4fc", "#24264b"),
    },
    "light": {
        Origin.CLAUDE_CODE_SESSION: ("#9a3412", "#ffedd5"),
        Origin.CODEX_SESSION: ("#155e75", "#cffafe"),
        Origin.GEMINI_CLI_SESSION: ("#1e40af", "#dbeafe"),
        Origin.HERMES_SESSION: ("#5b21b6", "#ede9fe"),
        Origin.ANTIGRAVITY_SESSION: ("#9d174d", "#fce7f3"),
        Origin.BEADS_ISSUE: ("#854d0e", "#fef3c7"),
        Origin.GROK_EXPORT: ("#374151", "#e5e7eb"),
        Origin.CHATGPT_EXPORT: ("#166534", "#dcfce7"),
        Origin.CLAUDE_AI_EXPORT: ("#9a3412", "#ffedd5"),
        Origin.AISTUDIO_DRIVE: ("#3730a3", "#e0e7ff"),
    },
}


def webui_theme_tokens(mode: ThemeMode) -> dict[str, str]:
    """Return one complete WebUI theme token mapping for generation."""

    return {**WEBUI_SHARED_TOKENS, **WEBUI_THEME_TOKENS[mode]}


# =============================================================================
# Thinking / reasoning block styling
# =============================================================================

THINKING_STYLE = {
    "rich_style": "dim italic",
    "border_color": "#475569",
    "label": "dim italic #94a3b8",
    "icon": "💭",
}


# =============================================================================
# Rich theme dict (for ConsoleFacade)
# =============================================================================


# =============================================================================
# Semantic style tokens (#1274)
#
# These are the canonical names every CLI surface should reach for when
# rendering severity-coded output. They map to Rich Theme style keys via
# ``semantic_style(token, mode)`` so light/dark resolutions stay consistent.
# =============================================================================

SEMANTIC_TOKENS: tuple[str, ...] = ("error", "warning", "ok", "dim", "info")
SyntaxSurface = Literal["terminal_code", "terminal_diff", "html"]

_SYNTAX_THEME_BY_MODE: dict[ThemeMode, dict[SyntaxSurface, str]] = {
    "dark": {
        "terminal_code": "monokai",
        "terminal_diff": "monokai",
        "html": "monokai",
    },
    "light": {
        "terminal_code": "default",
        "terminal_diff": "default",
        "html": "default",
    },
}


def _semantic_style_map(mode: ThemeMode) -> dict[str, str]:
    if mode == "light":
        # On light terminals, "dim" needs more contrast than the dark default.
        return {
            "error": f"bold {STATUS_COLORS['error']}",
            "warning": f"bold {STATUS_COLORS['warning']}",
            "ok": f"bold {STATUS_COLORS['success']}",
            "info": f"bold {STATUS_COLORS['info']}",
            "dim": "#6b7280",
        }
    return {
        "error": f"bold {STATUS_COLORS['error']}",
        "warning": f"bold {STATUS_COLORS['warning']}",
        "ok": f"bold {STATUS_COLORS['success']}",
        "info": f"bold {STATUS_COLORS['info']}",
        "dim": "dim #94a3b8",
    }


def semantic_style(token: str, mode: ThemeMode | None = None) -> str:
    """Return the Rich style string for a semantic token under ``mode``.

    ``token`` must be one of :data:`SEMANTIC_TOKENS`. ``mode`` defaults to
    the active theme resolved via :func:`resolve_theme_mode`.
    """
    if token not in SEMANTIC_TOKENS:
        raise KeyError(f"unknown semantic token: {token!r}")
    active = mode if mode is not None else resolve_theme_mode()
    return _semantic_style_map(active)[token]


def syntax_theme(surface: SyntaxSurface, mode: ThemeMode | None = None) -> str:
    """Return the Pygments/Rich syntax style for a semantic render surface.

    Code and diff renderers should use this resolver instead of hard-coded
    call-site styles so terminal and HTML surfaces stay aligned with the active
    theme. Plain/no-color callers still bypass syntax color before this helper
    is reached.
    """
    active = mode if mode is not None else resolve_theme_mode()
    return _SYNTAX_THEME_BY_MODE[active][surface]


def resolve_theme_mode() -> ThemeMode:
    """Resolve the active theme mode honoring env, config, and auto-detection.

    See module docstring for precedence rules. Returns ``"dark"`` on
    indeterminate auto-detection so the historical default is preserved.
    """
    env_value = os.environ.get("POLYLOGUE_THEME", "").strip().lower()
    if env_value == "dark":
        return "dark"
    if env_value == "light":
        return "light"
    if env_value == "" or env_value == "auto":
        # Try config layer; importing lazily keeps the theme module
        # importable from anywhere without circular dependency risk.
        try:
            from polylogue.config import load_polylogue_config

            configured = (load_polylogue_config().theme or "").strip().lower()
            if configured == "dark":
                return "dark"
            if configured == "light":
                return "light"
            if configured and configured != "auto":
                return "dark"
        except Exception:
            pass
        # Auto-detection: COLORFGBG="<fg>;<bg>" — bg index <8 → dark, else light.
        fgbg = os.environ.get("COLORFGBG", "")
        if fgbg:
            parts = fgbg.split(";")
            if len(parts) >= 2:
                try:
                    bg = int(parts[-1])
                    return "light" if bg >= 8 else "dark"
                except ValueError:
                    pass
    return "dark"


def rich_theme_styles(mode: ThemeMode | None = None) -> dict[str, str]:
    """Build Rich Theme style dict for ``mode`` (defaults to resolved mode)."""
    active = mode if mode is not None else resolve_theme_mode()
    sem = _semantic_style_map(active)
    if active == "light":
        banner_title = "bold #134e4a"
        banner_subtitle = "#0f766e"
        banner_icon = "bold #0d9488"
        panel_text = "#111827"
        summary_title = "bold #1e3a8a"
        summary_text = "#1f2937"
        status_message = "#111827"
    else:
        banner_title = "bold #e0f2f1"
        banner_subtitle = "#cdecef"
        banner_icon = "bold #7fdbca"
        panel_text = "#e5e7eb"
        summary_title = "bold #c4e0ff"
        summary_text = "#d6dee8"
        status_message = "#e5e7eb"

    styles: dict[str, str] = {
        # Banner
        "banner.icon": banner_icon,
        "banner.title": banner_title,
        "banner.subtitle": banner_subtitle,
        "banner.border": "#14b8a6",
        # Panels
        "panel.border": "#3b82f6",
        "panel.text": panel_text,
        # Summary
        "summary.title": summary_title,
        "summary.bullet": "bold #34d399",
        "summary.text": summary_text,
        # Status
        "status.icon.error": sem["error"],
        "status.icon.warning": sem["warning"],
        "status.icon.success": sem["ok"],
        "status.icon.info": sem["info"],
        "status.message": status_message,
        # Code
        "code.border": "#4c1d95",
        "markdown.border": "#475569",
        # Thinking
        "thinking.text": THINKING_STYLE["rich_style"],
        "thinking.border": THINKING_STYLE["border_color"],
        "thinking.label": THINKING_STYLE["label"],
        # Semantic tokens (canonical names — prefer these in new code).
        "semantic.error": sem["error"],
        "semantic.warning": sem["warning"],
        "semantic.ok": sem["ok"],
        "semantic.info": sem["info"],
        "semantic.dim": sem["dim"],
    }

    # Role styles
    for role_name, rc in ROLE_COLORS.items():
        styles[f"role.{role_name}"] = rc.label
        styles[f"role.{role_name}.dim"] = f"dim {rc.hex}"

    # Provider styles
    for prov_name, pc in PROVIDER_COLORS.items():
        styles[f"provider.{prov_name}"] = pc.hex

    return styles


def css_variables(theme: str = "dark") -> dict[str, str]:
    """Export theme tokens as CSS variable name→value mapping.

    Useful for injecting into HTML templates.
    """
    tokens = DARK_THEME if theme == "dark" else LIGHT_THEME
    variables: dict[str, str] = {}
    for key, value in tokens.items():
        css_name = f"--{key.replace('_', '-')}"
        variables[css_name] = value

    # Add role colors as CSS variables
    for role_name, rc in ROLE_COLORS.items():
        variables[f"--{role_name}-bg"] = rc.css_bg(0.1)
        variables[f"--{role_name}-border"] = rc.css_border(0.3)

    return variables


def css_variable_declarations(theme: str = "dark") -> str:
    """Return theme CSS variables as indented ``name: value;`` declarations."""
    return "\n".join(f"            {name}: {value};" for name, value in css_variables(theme).items())

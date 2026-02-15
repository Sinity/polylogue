"""Shared color palette and theme constants.

Single source of truth for provider brand colors, role colors,
status colors, and UI theme tokens used across CLI, HTML rendering,
and static site generation.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "claude": ProviderColor("#d97757"),
    "claude-code": ProviderColor("#d97757"),
    "chatgpt": ProviderColor("#10a37f"),
    "gemini": ProviderColor("#4285f4"),
    "codex": ProviderColor("#00bcd4"),
    "google-ai-studio": ProviderColor("#4285f4"),
}

DEFAULT_PROVIDER_COLOR = ProviderColor("#e5e7eb")


def provider_color(name: str) -> ProviderColor:
    """Look up a provider color by name, with fuzzy matching.

    Matches on substring so 'claude-code' matches 'claude',
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


def role_color(role: str) -> RoleColor:
    """Look up a role color."""
    return ROLE_COLORS.get(role, DEFAULT_ROLE_COLOR)


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
    # Text
    "text_primary": "#f8f9fa",
    "text_secondary": "#94a3b8",
    "text_muted": "#6b7280",
    # Accent
    "accent": "#6366f1",
    "accent_glow": "rgba(99, 102, 241, 0.4)",
    # Borders
    "border": "#2d2d35",
    # Special
    "glass": "rgba(255, 255, 255, 0.03)",
    "glass_border": "rgba(255, 255, 255, 0.1)",
}

LIGHT_THEME = {
    "bg_primary": "#ffffff",
    "bg_secondary": "#f9fafb",
    "bg_elevated": "#ffffff",
    "text_primary": "#111827",
    "text_secondary": "#4b5563",
    "text_muted": "#9ca3af",
    "accent": "#6366f1",
    "accent_glow": "rgba(99, 102, 241, 0.2)",
    "border": "#e5e7eb",
    "glass": "rgba(0, 0, 0, 0.02)",
    "glass_border": "rgba(0, 0, 0, 0.08)",
}


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

def rich_theme_styles() -> dict[str, str]:
    """Build Rich Theme style dict incorporating role and provider colors."""
    styles: dict[str, str] = {
        # Banner
        "banner.icon": "bold #7fdbca",
        "banner.title": "bold #e0f2f1",
        "banner.subtitle": "#cdecef",
        "banner.border": "#14b8a6",
        # Panels
        "panel.border": "#3b82f6",
        "panel.text": "#e5e7eb",
        # Summary
        "summary.title": "bold #c4e0ff",
        "summary.bullet": "bold #34d399",
        "summary.text": "#d6dee8",
        # Status
        "status.icon.error": f"bold {STATUS_COLORS['error']}",
        "status.icon.warning": f"bold {STATUS_COLORS['warning']}",
        "status.icon.success": f"bold {STATUS_COLORS['success']}",
        "status.icon.info": f"bold {STATUS_COLORS['info']}",
        "status.message": "#e5e7eb",
        # Code
        "code.border": "#4c1d95",
        "markdown.border": "#475569",
        # Thinking
        "thinking.text": THINKING_STYLE["rich_style"],
        "thinking.border": THINKING_STYLE["border_color"],
        "thinking.label": THINKING_STYLE["label"],
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

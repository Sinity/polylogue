"""Generate the WebUI v2 public badge contracts and CSS design tokens."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Literal, TypedDict, cast

from devtools.command_catalog import control_plane_command
from devtools.render_support import write_if_changed
from polylogue.core.enums import Origin

# Loading ``polylogue.ui.theme`` through the package would execute the broad UI
# facade import graph.  The renderer only needs this leaf authority module, so
# execute that file directly; its own imports (Role and Origin) are leaf-safe.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_THEME_AUTHORITY = runpy.run_path(str(_REPO_ROOT / "polylogue/ui/theme.py"))
ThemeMode = Literal["dark", "light"]
THEME_MODES: tuple[ThemeMode, ...] = ("light", "dark")


class ContrastPair(TypedDict):
    mode: ThemeMode
    pair: str
    foreground: str
    background: str
    ratio: float
    threshold: float
    passes: bool


class ContrastReport(TypedDict):
    schema_version: int
    generated_by: str
    method: str
    minimum_text_ratio: float
    minimum_focus_ratio: float
    pairs: list[ContrastPair]


PUBLIC_ORIGIN_TOKENS = cast(tuple[Origin, ...], _THEME_AUTHORITY["PUBLIC_ORIGIN_TOKENS"])
WEBUI_EVIDENCE_BADGE_TOKENS = cast(
    dict[ThemeMode, dict[str, tuple[str, str]]], _THEME_AUTHORITY["WEBUI_EVIDENCE_BADGE_TOKENS"]
)
WEBUI_ORIGIN_BADGE_TOKENS = cast(
    dict[ThemeMode, dict[Origin, tuple[str, str]]], _THEME_AUTHORITY["WEBUI_ORIGIN_BADGE_TOKENS"]
)
WEBUI_SHARED_TOKENS = cast(dict[str, str], _THEME_AUTHORITY["WEBUI_SHARED_TOKENS"])
WEBUI_THEME_TOKENS = cast(dict[ThemeMode, dict[str, str]], _THEME_AUTHORITY["WEBUI_THEME_TOKENS"])

DEFAULT_OUTPUT_DIR = Path("webui/src/generated")
EVIDENCE_STATES: tuple[str, ...] = ("exact", "qualified", "stale", "unknown", "degraded")
ORIGIN_LABELS: dict[Origin, str] = {
    Origin.CLAUDE_CODE_SESSION: "Claude Code",
    Origin.CODEX_SESSION: "Codex",
    Origin.GEMINI_CLI_SESSION: "Gemini CLI",
    Origin.HERMES_SESSION: "Hermes",
    Origin.ANTIGRAVITY_SESSION: "Antigravity",
    Origin.BEADS_ISSUE: "Beads",
    Origin.GROK_EXPORT: "Grok export",
    Origin.CHATGPT_EXPORT: "ChatGPT export",
    Origin.CLAUDE_AI_EXPORT: "Claude.ai export",
    Origin.AISTUDIO_DRIVE: "AI Studio Drive",
}
EVIDENCE_LABELS: dict[str, str] = {
    "exact": "Exact",
    "qualified": "Qualified",
    "stale": "Stale",
    "unknown": "Unknown",
    "degraded": "Degraded",
}


def _linear_channel(channel: int) -> float:
    value = channel / 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """Return WCAG relative luminance for a six-digit hex color."""

    value = hex_color.removeprefix("#")
    if len(value) != 6:
        raise ValueError(f"contrast checks require six-digit hex colors, got {hex_color!r}")
    red, green, blue = (int(value[offset : offset + 2], 16) for offset in (0, 2, 4))
    return 0.2126 * _linear_channel(red) + 0.7152 * _linear_channel(green) + 0.0722 * _linear_channel(blue)


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio for two opaque hex colors."""

    lighter, darker = sorted((relative_luminance(foreground), relative_luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _contract_validation() -> None:
    public_origins = tuple(PUBLIC_ORIGIN_TOKENS)
    if len(public_origins) != 10:
        raise ValueError(f"expected 10 public Origin tokens, found {len(public_origins)}")
    if Origin.UNKNOWN_EXPORT in public_origins:
        raise ValueError("unknown-export is a fallback state, not a public Origin badge token")
    if set(ORIGIN_LABELS) != set(public_origins):
        raise ValueError("origin label map does not exactly cover the public Origin contract")
    for mode in THEME_MODES:
        if set(WEBUI_ORIGIN_BADGE_TOKENS[mode]) != set(public_origins):
            raise ValueError(f"{mode} origin badge token map does not exactly cover the public Origin contract")
        if set(WEBUI_EVIDENCE_BADGE_TOKENS[mode]) != set(EVIDENCE_STATES):
            raise ValueError(f"{mode} evidence badge token map does not match the public state vocabulary")


def _theme_declarations(mode: ThemeMode) -> list[str]:
    declarations = [f"  {name}: {value};" for name, value in WEBUI_THEME_TOKENS[mode].items()]
    for state in EVIDENCE_STATES:
        foreground, background = WEBUI_EVIDENCE_BADGE_TOKENS[mode][state]
        declarations.extend(
            (
                f"  --pl-evidence-{state}-fg: {foreground};",
                f"  --pl-evidence-{state}-bg: {background};",
                f"  --pl-evidence-{state}-border: {foreground};",
            )
        )
    for origin in PUBLIC_ORIGIN_TOKENS:
        foreground, background = WEBUI_ORIGIN_BADGE_TOKENS[mode][origin]
        declarations.extend(
            (
                f"  --pl-origin-{origin.value}-fg: {foreground};",
                f"  --pl-origin-{origin.value}-bg: {background};",
                f"  --pl-origin-{origin.value}-border: {foreground};",
            )
        )
    return declarations


def _badge_rules() -> str:
    rules: list[str] = []
    for state in EVIDENCE_STATES:
        rules.append(
            "\n".join(
                (
                    f".pl-evidence-badge[data-evidence-state='{state}'] {{",
                    f"  color: var(--pl-evidence-{state}-fg);",
                    f"  background: var(--pl-evidence-{state}-bg);",
                    "}",
                )
            )
        )
    for origin in PUBLIC_ORIGIN_TOKENS:
        token = origin.value
        rules.append(
            "\n".join(
                (
                    f".pl-origin-badge[data-origin='{token}'] {{",
                    f"  color: var(--pl-origin-{token}-fg);",
                    f"  background: var(--pl-origin-{token}-bg);",
                    "}",
                )
            )
        )
    return "\n\n".join(rules)


def render_tokens_css() -> str:
    """Render the generated light/dark CSS custom-property surface."""

    generated_by = control_plane_command("render webui-design-system")
    shared = "\n".join(f"  {name}: {value};" for name, value in WEBUI_SHARED_TOKENS.items())
    light = "\n".join(_theme_declarations("light"))
    dark = "\n".join(_theme_declarations("dark"))
    badge_rules = _badge_rules()
    source_note = (
        "/* Generated by "
        f"`{generated_by}`. Edit polylogue/ui/theme.py and "
        "devtools/render_webui_design_system.py instead. */"
    )
    return f"""{source_note}
:root {{
{shared}
  color-scheme: light dark;
}}

:root,
:root[data-theme="light"] {{
{light}
  color-scheme: light;
}}

@media (prefers-color-scheme: dark) {{
  :root:not([data-theme]) {{
{dark}
    color-scheme: dark;
  }}
}}

:root[data-theme="dark"] {{
{dark}
  color-scheme: dark;
}}

{badge_rules}

@media (prefers-reduced-motion: reduce) {{
  :root {{
    --pl-motion-fast: 0ms;
    --pl-motion-normal: 0ms;
  }}
}}
"""


def render_contracts_ts() -> str:
    """Render public TypeScript token unions without duplicating Python enums."""

    generated_by = control_plane_command("render webui-design-system")
    origin_values = [origin.value for origin in PUBLIC_ORIGIN_TOKENS]
    origin_labels = {origin.value: ORIGIN_LABELS[origin] for origin in PUBLIC_ORIGIN_TOKENS}
    rendered_origin_labels = json.dumps(origin_labels, indent=2, ensure_ascii=False)
    rendered_evidence_labels = json.dumps(EVIDENCE_LABELS, indent=2)
    return "\n".join(
        (
            f"// Generated by `{generated_by}`. Edit Python authority instead.",
            "export const DESIGN_SYSTEM_CONTRACT_VERSION = 1 as const;",
            f"export const PUBLIC_ORIGINS = {json.dumps(origin_values, indent=2)} as const;",
            "export type OriginToken = (typeof PUBLIC_ORIGINS)[number];",
            f"export const ORIGIN_LABELS: Readonly<Record<OriginToken, string>> = {rendered_origin_labels};",
            f"export const EVIDENCE_STATES = {json.dumps(list(EVIDENCE_STATES), indent=2)} as const;",
            "export type EvidenceState = (typeof EVIDENCE_STATES)[number];",
            "export const EVIDENCE_STATE_LABELS: Readonly<Record<EvidenceState, string>> = "
            f"{rendered_evidence_labels};",
            "",
        )
    )


def build_contrast_report() -> ContrastReport:
    """Build and validate the contrast evidence shipped beside the token file."""

    pairs: list[ContrastPair] = []
    for mode in THEME_MODES:
        theme = WEBUI_THEME_TOKENS[mode]
        ordinary_pairs = {
            "text-on-background": (theme["--pl-color-text"], theme["--pl-color-bg"], 4.5),
            "text-on-surface": (theme["--pl-color-text"], theme["--pl-color-surface"], 4.5),
            "muted-text-on-background": (theme["--pl-color-text-muted"], theme["--pl-color-bg"], 4.5),
            "subtle-text-on-background": (theme["--pl-color-text-subtle"], theme["--pl-color-bg"], 4.5),
            "accent-on-background": (theme["--pl-color-accent"], theme["--pl-color-bg"], 4.5),
            "focus-on-surface": (theme["--pl-color-focus"], theme["--pl-color-surface"], 3.0),
        }
        for name, (foreground, background, threshold) in ordinary_pairs.items():
            ratio = contrast_ratio(foreground, background)
            pairs.append(
                {
                    "mode": mode,
                    "pair": name,
                    "foreground": foreground,
                    "background": background,
                    "ratio": round(ratio, 2),
                    "threshold": threshold,
                    "passes": ratio >= threshold,
                }
            )
        for state, (foreground, background) in WEBUI_EVIDENCE_BADGE_TOKENS[mode].items():
            ratio = contrast_ratio(foreground, background)
            pairs.append(
                {
                    "mode": mode,
                    "pair": f"evidence-{state}",
                    "foreground": foreground,
                    "background": background,
                    "ratio": round(ratio, 2),
                    "threshold": 4.5,
                    "passes": ratio >= 4.5,
                }
            )
        for origin, (foreground, background) in WEBUI_ORIGIN_BADGE_TOKENS[mode].items():
            ratio = contrast_ratio(foreground, background)
            pairs.append(
                {
                    "mode": mode,
                    "pair": f"origin-{origin.value}",
                    "foreground": foreground,
                    "background": background,
                    "ratio": round(ratio, 2),
                    "threshold": 4.5,
                    "passes": ratio >= 4.5,
                }
            )
    failures = [pair for pair in pairs if not pair["passes"]]
    if failures:
        rendered = ", ".join(f"{item['mode']}:{item['pair']}={item['ratio']}" for item in failures)
        raise ValueError(f"WebUI contrast contract failed: {rendered}")
    text_ratios = [pair["ratio"] for pair in pairs if pair["threshold"] == 4.5]
    focus_ratios = [pair["ratio"] for pair in pairs if pair["threshold"] == 3.0]
    return {
        "schema_version": 1,
        "generated_by": control_plane_command("render webui-design-system"),
        "method": "WCAG 2.x relative luminance and contrast ratio",
        "minimum_text_ratio": min(text_ratios),
        "minimum_focus_ratio": min(focus_ratios),
        "pairs": pairs,
    }


def render_contrast_report() -> str:
    return json.dumps(build_contrast_report(), indent=2, sort_keys=True) + "\n"


def _outputs(output_dir: Path) -> dict[Path, str]:
    _contract_validation()
    return {
        output_dir / "tokens.css": render_tokens_css(),
        output_dir / "contracts.ts": render_contracts_ts(),
        output_dir / "contrast-report.json": render_contrast_report(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render generated WebUI v2 design-system contracts.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    outputs = _outputs(Path(args.output_dir))

    if args.check:
        drift = [path for path, rendered in outputs.items() if not path.exists() or path.read_text() != rendered]
        if drift:
            print("render webui-design-system: out of sync:", file=sys.stderr)
            for path in drift:
                print(f"  - {path}", file=sys.stderr)
            print(
                f"render webui-design-system: run: {control_plane_command('render webui-design-system')}",
                file=sys.stderr,
            )
            return 1
        print("render webui-design-system: sync OK")
        return 0

    for path, rendered in outputs.items():
        write_if_changed(path, rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

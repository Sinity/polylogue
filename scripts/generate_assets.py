#!/usr/bin/env python3
"""Generate visual assets for README and documentation.

Creates:
- docs/assets/hero-banner.svg — Project header banner

Usage:
    python scripts/generate_assets.py
"""

from __future__ import annotations

from pathlib import Path


def _generate_hero_banner(output_path: Path) -> None:
    """Generate a hero banner SVG with the polylogue wordmark and visual motif."""
    # Color palette: Catppuccin Mocha-inspired
    bg = "#1e1e2e"
    surface = "#313244"
    text_primary = "#cdd6f4"
    text_muted = "#a6adc8"
    accent_blue = "#89b4fa"
    accent_green = "#a6e3a1"
    accent_peach = "#fab387"
    accent_mauve = "#cba6f7"
    accent_red = "#f38ba8"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 200" fill="none">
  <defs>
    <linearGradient id="bg-grad" x1="0" y1="0" x2="700" y2="200" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{bg}"/>
      <stop offset="100%" stop-color="#181825"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <!-- Background -->
  <rect width="700" height="200" rx="16" fill="url(#bg-grad)"/>

  <!-- Conversation bubble motif (left side) -->
  <!-- Bubble 1: ChatGPT (green) -->
  <rect x="40" y="50" width="120" height="32" rx="8" fill="{accent_green}" opacity="0.15"/>
  <rect x="40" y="50" width="120" height="32" rx="8" stroke="{accent_green}" stroke-width="1" opacity="0.4" fill="none"/>
  <circle cx="54" cy="66" r="4" fill="{accent_green}" opacity="0.6"/>
  <rect x="64" y="62" width="80" height="3" rx="1.5" fill="{accent_green}" opacity="0.3"/>
  <rect x="64" y="68" width="50" height="3" rx="1.5" fill="{accent_green}" opacity="0.2"/>

  <!-- Bubble 2: Claude (mauve) -->
  <rect x="60" y="90" width="130" height="32" rx="8" fill="{accent_mauve}" opacity="0.15"/>
  <rect x="60" y="90" width="130" height="32" rx="8" stroke="{accent_mauve}" stroke-width="1" opacity="0.4" fill="none"/>
  <circle cx="74" cy="106" r="4" fill="{accent_mauve}" opacity="0.6"/>
  <rect x="84" y="102" width="90" height="3" rx="1.5" fill="{accent_mauve}" opacity="0.3"/>
  <rect x="84" y="108" width="60" height="3" rx="1.5" fill="{accent_mauve}" opacity="0.2"/>

  <!-- Bubble 3: Gemini (peach) -->
  <rect x="45" y="130" width="110" height="28" rx="8" fill="{accent_peach}" opacity="0.15"/>
  <rect x="45" y="130" width="110" height="28" rx="8" stroke="{accent_peach}" stroke-width="1" opacity="0.4" fill="none"/>
  <circle cx="59" cy="144" r="4" fill="{accent_peach}" opacity="0.6"/>
  <rect x="69" y="140" width="70" height="3" rx="1.5" fill="{accent_peach}" opacity="0.3"/>
  <rect x="69" y="146" width="45" height="3" rx="1.5" fill="{accent_peach}" opacity="0.2"/>

  <!-- Connection lines (archive concept) -->
  <line x1="165" y1="66" x2="210" y2="100" stroke="{surface}" stroke-width="1" opacity="0.5" stroke-dasharray="4 4"/>
  <line x1="195" y1="106" x2="210" y2="100" stroke="{surface}" stroke-width="1" opacity="0.5" stroke-dasharray="4 4"/>
  <line x1="160" y1="144" x2="210" y2="100" stroke="{surface}" stroke-width="1" opacity="0.5" stroke-dasharray="4 4"/>

  <!-- Central archive node -->
  <circle cx="210" cy="100" r="6" fill="{accent_blue}" opacity="0.3"/>
  <circle cx="210" cy="100" r="3" fill="{accent_blue}" opacity="0.7"/>

  <!-- Arrow to wordmark -->
  <line x1="220" y1="100" x2="250" y2="100" stroke="{accent_blue}" stroke-width="1" opacity="0.3"/>

  <!-- Wordmark -->
  <text x="270" y="92" font-family="'JetBrains Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace" font-size="48" font-weight="700" fill="{text_primary}" filter="url(#glow)">polylogue</text>

  <!-- Tagline -->
  <text x="270" y="118" font-family="'Inter', 'SF Pro', -apple-system, system-ui, sans-serif" font-size="14" fill="{text_muted}">preserve &middot; index &middot; expose</text>

  <!-- Provider dots (right side, subtle) -->
  <g transform="translate(620, 50)" opacity="0.5">
    <circle cx="0" cy="0" r="5" fill="{accent_green}"/>
    <text x="12" y="4" font-family="monospace" font-size="9" fill="{text_muted}">chatgpt</text>
  </g>
  <g transform="translate(620, 72)" opacity="0.5">
    <circle cx="0" cy="0" r="5" fill="{accent_mauve}"/>
    <text x="12" y="4" font-family="monospace" font-size="9" fill="{text_muted}">claude</text>
  </g>
  <g transform="translate(620, 94)" opacity="0.5">
    <circle cx="0" cy="0" r="5" fill="{accent_peach}"/>
    <text x="12" y="4" font-family="monospace" font-size="9" fill="{text_muted}">gemini</text>
  </g>
  <g transform="translate(620, 116)" opacity="0.5">
    <circle cx="0" cy="0" r="5" fill="{accent_blue}"/>
    <text x="12" y="4" font-family="monospace" font-size="9" fill="{text_muted}">codex</text>
  </g>
  <g transform="translate(620, 138)" opacity="0.5">
    <circle cx="0" cy="0" r="5" fill="{accent_red}"/>
    <text x="12" y="4" font-family="monospace" font-size="9" fill="{text_muted}">cc</text>
  </g>

  <!-- Version badge -->
  <rect x="270" y="135" width="55" height="20" rx="4" fill="{surface}"/>
  <text x="298" y="149" font-family="monospace" font-size="10" fill="{text_muted}" text-anchor="middle">v0.1.0</text>
</svg>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    assets_dir = Path("docs/assets")

    print("Generating hero banner...")
    _generate_hero_banner(assets_dir / "hero-banner.svg")
    print(f"  Created: {assets_dir / 'hero-banner.svg'}")

    print("Done!")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 720
BG = "#0b0f14"
PANEL = "#111821"
BORDER = "#263241"
TEXT = "#e6edf3"
MUTED = "#8b9cad"
ACCENT = "#7ee787"
WARN = "#ff7b72"
CYAN = "#79c0ff"
PURPLE = "#d2a8ff"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_SANS = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
FONT_SANS_BOLD = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
mono = ImageFont.truetype(FONT_MONO, 22)
mono_small = ImageFont.truetype(FONT_MONO, 18)
sans = ImageFont.truetype(FONT_SANS, 19)
sans_small = ImageFont.truetype(FONT_SANS, 16)
bold = ImageFont.truetype(FONT_SANS_BOLD, 30)
label = ImageFont.truetype(FONT_SANS_BOLD, 14)


def text(draw: ImageDraw.ImageDraw, xy, value, font=mono, fill=TEXT, spacing=7):
    draw.multiline_text(xy, value, font=font, fill=fill, spacing=spacing)


def frame(kicker: str, title: str, command: str, body_lines: list[tuple[str, str]], footer: str) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    # terminal chrome
    d.rounded_rectangle((22, 20, W - 22, H - 20), radius=16, fill=PANEL, outline=BORDER, width=2)
    d.ellipse((45, 43, 59, 57), fill="#ff5f56")
    d.ellipse((68, 43, 82, 57), fill="#ffbd2e")
    d.ellipse((91, 43, 105, 57), fill="#27c93f")
    text(d, (126, 40), "polylogue demo tour", font=mono_small, fill=MUTED)
    # story header
    text(d, (54, 94), kicker.upper(), font=label, fill=CYAN)
    text(d, (54, 118), title, font=bold, fill=TEXT)
    d.rounded_rectangle((52, 174, W - 52, 230), radius=8, fill="#080c10", outline=BORDER)
    text(d, (72, 190), "$ " + command, font=mono_small, fill=ACCENT)
    y = 268
    for value, color in body_lines:
        text(d, (64, y), value, font=mono, fill=color)
        y += 44 if "\n" not in value else 74
    d.line((52, H - 83, W - 52, H - 83), fill=BORDER, width=1)
    text(d, (54, H - 63), footer, font=sans_small, fill=MUTED)
    text(d, (W - 94, H - 65), "▮", font=mono_small, fill=ACCENT)
    return im


frames = [
    frame(
        "1 · evidence first",
        "The receipt, before the dashboard",
        "polylogue --id codex-session:demo-terminal-error read --view messages",
        [
            ("Run the command and stop if it fails.", TEXT),
            ("exit_code = 4", WARN),
            ("ERROR: file or directory not found:\ntests/missing_test.py", WARN),
            ("Assistant: I hit an error and need the missing path corrected.", CYAN),
        ],
        "Direct provider-normalized tool evidence · exact session and message refs remain resolvable",
    ),
    frame(
        "2 · anti-grep",
        "Count structural failures, not scary words",
        "polylogue 'actions where is_error:true | group by tool | count'",
        [
            ("tool=Bash          count=4", PURPLE),
            ("tool=exec_command  count=1", PURPLE),
            ("Oracle: normalized is_error / exit fields", ACCENT),
            ("Not the occurrence of the word ‘error’ in prose", MUTED),
        ],
        "One typed query across provider-specific tool representations",
    ),
    frame(
        "3 · lineage",
        "Preserve every artifact. Count copied work once.",
        "polylogue --id codex-session:demo-lineage-fork read --view chronicle",
        [
            ("parent:parent-u0  Map the demo lineage base context.", CYAN),
            ("parent:parent-a1  I have the base context…", CYAN),
            ("fork:fork-u2      Audit construct validity.", PURPLE),
            ("fork:fork-a3      The fork diverges…", PURPLE),
        ],
        "Inherited messages retain parent refs; the fork contributes its unique tail",
    ),
    frame(
        "4 · scope",
        "Only then zoom out",
        "polylogue analyze --facets",
        [
            ("11 sessions · 43 messages · 5 origins", ACCENT),
            ("codex-session       5", TEXT),
            ("claude-code-session 3", TEXT),
            ("aistudio · chatgpt · claude-ai  1 each", TEXT),
            ("Deferred families are labeled, not silently guessed.", MUTED),
        ],
        "The archive overview follows inspectable evidence rather than replacing it",
    ),
    frame(
        "proof boundary",
        "A compact story with a complete audit behind it",
        "cat report.md",
        [
            ("30 / 30 declared fixture constructs satisfied", ACCENT),
            ("0 absolute-path leaks · 0 semantic problems", ACCENT),
            ("Does not prove provider completeness, memory uplift,", MUTED),
            ("private-archive scale, deletion, or the Sinex backend.", MUTED),
        ],
        "Human transcript: compact · report.json: complete · every non-claim stated explicitly",
    ),
]

parser = argparse.ArgumentParser(description="Render the Polylogue evidence-first demo GIF.")
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "polylogue-demo-tour" / "demo-tour.gif",
)
parser.add_argument(
    "--repo-copy",
    type=Path,
    help="Optional second destination, for example a patched repository's docs/examples/demo-tour/demo-tour.gif.",
)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
frames[0].save(
    args.output,
    save_all=True,
    append_images=frames[1:],
    duration=[2200, 2200, 2400, 2200, 2600],
    loop=0,
    optimize=True,
)
if args.repo_copy:
    args.repo_copy.parent.mkdir(parents=True, exist_ok=True)
    args.repo_copy.write_bytes(args.output.read_bytes())
print(args.output, args.output.stat().st_size)

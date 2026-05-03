"""Generate README media assets (architecture diagrams, flowcharts).

Produces Mermaid (.mmd) source files and renders them to SVG under
``docs/media/`` using the system ``mmdc`` (Mermaid CLI) tool.

Diagrams:

- ``architecture-overview`` — four-ring Polylogue architecture
- ``data-flow`` — pipeline data flow (sources → parse → store → index → query)
- ``provider-detection`` — provider detection decision tree
- ``repo-structure`` — repository layout overview
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

_DOCS_MEDIA = Path(__file__).resolve().parent.parent / "docs" / "media"

# ---------------------------------------------------------------------------
# Diagram definitions
# ---------------------------------------------------------------------------

_DIAGRAMS: dict[str, tuple[str, str]] = {
    "architecture-overview": (
        "Polylogue Architecture",
        """graph TB
    subgraph Ring1["1. Archive Substrate"]
        A1["Source Acquisition"]
        A2["Provider Parsing"]
        A3["SQLite Persistence"]
        A4["Search Indexes"]
    end

    subgraph Ring2["2. Derived Read Models"]
        B1["Session Profiles"]
        B2["Work Events / Phases"]
        B3["Day / Week Summaries"]
        B4["Provider Analytics"]
    end

    subgraph Ring3["3. Surfaces"]
        C1["CLI"]
        C2["Python API"]
        C3["MCP Server"]
        C4["Static Site"]
        C5["Dashboard / TUI"]
        C6["Renderers"]
    end

    subgraph Ring4["4. Verification"]
        D1["Schema Inference"]
        D2["Synthetic Corpora"]
        D3["Showcase / QA"]
        D4["DevTools"]
        D5["Validation Lanes"]
    end

    Ring1 -->|"feeds"| Ring2
    Ring2 -->|"exposes"| Ring3
    Ring1 -->|"verified by"| Ring4
    Ring2 -->|"verified by"| Ring4
    Ring3 -->|"verified by"| Ring4""",
    ),
    "data-flow": (
        "Polylogue Data Flow",
        """flowchart LR
    SRC["Source Files<br/>(JSON / JSONL / ZIP)"] --> PROV["Provider Detection<br/>dispatch.detect_provider()"]
    PROV --> PARSE["Parse<br/>per-provider parsers"]
    PARSE --> HASH["Content Hash<br/>SHA-256 (NFC)"]
    HASH --> STORE["Store<br/>idempotent upsert"]
    STORE --> INDEX["Index<br/>FTS5 (unicode61)"]
    INDEX --> QUERY["Query<br/>CLI / MCP / API"]

    PARSE --> INSIGHTS["Session Insights<br/>profiles, events, phases"]
    INSIGHTS --> QUERY""",
    ),
    "provider-detection": (
        "Provider Detection Flow",
        """flowchart TD
    INPUT["Input File"] --> CHECK{"looks_like()"}
    CHECK -->|"ChatGPT"| CHATGPT["parsers/chatgpt.py<br/>mapping dict with message graph"]
    CHECK -->|"Claude Web"| CLAUDE_WEB["parsers/claude.py<br/>chat_messages list"]
    CHECK -->|"Claude Code"| CLAUDE_CODE["parsers/claude.py<br/>parentUuid / sessionId"]
    CHECK -->|"Codex"| CODEX["parsers/codex.py<br/>session envelope"]
    CHECK -->|"Gemini"| GEMINI["parsers/drive.py<br/>chunkedPrompt.chunks"]
    CHECK -->|"Unknown"| UNKNOWN["Provider.UNKNOWN"]""",
    ),
    "repo-structure": (
        "Repository Structure",
        """flowchart TB
    ROOT["polylogue/"] --> CORE["core/<br/>domain types, hashing, JSON"]
    ROOT --> SOURCES["sources/<br/>provider detection, parsers"]
    ROOT --> STORAGE["storage/<br/>SQLite backends, FTS, repos"]
    ROOT --> PIPELINE["pipeline/<br/>ingestion, validation, rendering"]
    ROOT --> SCHEMAS["schemas/<br/>synthetic corpus, inference"]
    ROOT --> INSIGHTS["insights/<br/>session profiles, analytics"]
    ROOT --> CLI["cli/<br/>Click commands, output"]
    ROOT --> MCP["mcp/<br/>MCP server tools"]
    ROOT --> API["api/<br/>async library API"]
    ROOT --> SITE["site/<br/>static site generation"]
    ROOT --> RENDERING["rendering/<br/>markdown / HTML"]
    ROOT --> UI["ui/<br/>TUI / dashboard"]
    ROOT --> SCENARIOS["scenarios/<br/>corpus specs, scenarios"]
    ROOT --> PROOF["proof/<br/>obligations, claims, witnesses"]
    ROOT --> SHOWCASE["showcase/<br/>QA exercises"]
    ROOT --> DEVTOOLS["devtools/<br/>operator tooling"]
    ROOT --> TESTS["tests/<br/>pytest suite"]""",
    ),
}


def _mmd_content(title: str, body: str) -> str:
    return f"""---
title: {title}
---
{body}
"""


def _write_mmd(name: str, title: str, body: str, output_dir: Path) -> Path:
    path = output_dir / f"{name}.mmd"
    path.write_text(_mmd_content(title, body), encoding="utf-8")
    return path


def _render_svg(mmd_path: Path, output_dir: Path, *, check: bool = False) -> Path | None:
    """Render an .mmd file to SVG using mmdc."""
    svg_path = output_dir / f"{mmd_path.stem}.svg"
    mmdc = shutil.which("mmdc")
    if mmdc is None:
        if check:
            print(f"  [SKIP] mmdc not found, cannot render {mmd_path.name}")
        return None

    result = subprocess.run(
        [mmdc, "-i", str(mmd_path), "-o", str(svg_path), "--quiet"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"  [ERROR] mmdc failed on {mmd_path.name}: {result.stderr.strip()}")
        return None
    return svg_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate README media assets (Mermaid diagrams → SVG).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DOCS_MEDIA,
        help=f"Output directory for generated files (default: {_DOCS_MEDIA}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that mmdc is available without rendering.",
    )
    parser.add_argument(
        "--name",
        help="Generate only one diagram by name.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_names",
        help="List available diagram names and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_names:
        for name, (title, _body) in _DIAGRAMS.items():
            print(f"{name:<30} {title}")
        return 0

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    diagram_names = [args.name] if args.name else list(_DIAGRAMS.keys())

    rendered: list[str] = []
    for name in diagram_names:
        if name not in _DIAGRAMS:
            print(f"[ERROR] unknown diagram: {name!r}")
            return 1
        title, body = _DIAGRAMS[name]
        mmd_path = _write_mmd(name, title, body, output_dir)
        print(f"  [WRITE] {mmd_path.relative_to(output_dir.parent)}")

        if args.check:
            continue

        svg_path = _render_svg(mmd_path, output_dir)
        if svg_path is not None:
            rel = svg_path.relative_to(output_dir.parent)
            print(f"  [RENDER] {rel}")
            rendered.append(name)
        else:
            print(f"  [MMD ONLY] {mmd_path.name} (mmdc not available)")

    print()
    if rendered:
        print(f"Generated {len(rendered)} diagram(s) in {output_dir}")
    else:
        print(f"Wrote {len(diagram_names)} .mmd source(s) in {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

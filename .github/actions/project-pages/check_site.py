"""Validate local links and required project-site artifacts."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"] or "")
        attribute = "href" if tag in {"a", "link"} else "src" if tag == "script" else None
        if attribute and values.get(attribute):
            self.values.append(values[attribute] or "")


def target_for(page: Path, root: Path, value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        return None
    if not parsed.path and parsed.fragment:
        return page
    if not parsed.path:
        return None
    target = root / parsed.path.lstrip("/") if parsed.path.startswith("/") else page.parent / parsed.path
    target = target.resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    if parsed.path.endswith("/") or target.is_dir():
        target /= "index.html"
    return target


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    required = (
        root / "index.html",
        root / "project.json",
        root / "beads" / "index.html",
        root / "beads" / "issues.jsonl",
    )
    missing = [path.relative_to(root).as_posix() for path in required if not path.is_file()]
    broken: list[str] = []
    parsed_pages: dict[Path, Links] = {}
    for page in root.rglob("*.html"):
        parser = Links()
        parser.feed(page.read_text(encoding="utf-8"))
        parsed_pages[page.resolve()] = parser
    for page, parser in parsed_pages.items():
        for value in parser.values:
            target = target_for(page, root, value)
            if target is not None and not target.exists():
                broken.append(f"{page.relative_to(root)}: {value} -> {target.relative_to(root)}")
                continue
            fragment = urlparse(value).fragment
            if target is not None and fragment:
                target_parser = parsed_pages.get(target.resolve())
                if target_parser is not None and fragment not in target_parser.ids:
                    broken.append(f"{page.relative_to(root)}: {value} -> missing #{fragment}")
    if missing or broken:
        for value in missing:
            print(f"missing: {value}", file=sys.stderr)
        for value in broken:
            print(f"broken: {value}", file=sys.stderr)
        return 1
    print(f"site: {sum(1 for _ in root.rglob('*.html'))} HTML pages; local links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate the private-data-free Codex whale fixture pack on disk."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tests.infra.whale_fixtures import write_codex_whale_fixture_pack


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="directory for codex-whale.jsonl and manifest.json")
    args = parser.parse_args(argv)
    source_path, manifest_path = write_codex_whale_fixture_pack(args.output_dir)
    print(source_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

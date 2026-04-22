"""Verification-lab synthetic corpus and demo workspace command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polylogue.scenarios import CorpusSourceKind
from polylogue.showcase.lab_corpus import generate_lab_corpus, seed_lab_demo


def _add_corpus_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        "-p",
        dest="providers",
        action="append",
        default=[],
        help="Provider to include. May be repeated. Defaults to all providers for the default corpus source.",
    )
    parser.add_argument("--count", "-n", type=int, default=3, help="Conversations per provider.")
    parser.add_argument(
        "--corpus-source",
        choices=[kind.value for kind in CorpusSourceKind],
        default=CorpusSourceKind.DEFAULT.value,
        help="Corpus spec source to execute.",
    )
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="Output directory.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate verification-lab synthetic corpus fixtures.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate raw provider wire-format files.")
    _add_corpus_options(generate_parser)

    seed_parser = subparsers.add_parser("seed", help="Seed a complete demo archive workspace.")
    _add_corpus_options(seed_parser)
    seed_parser.add_argument("--env-only", action="store_true", help="Print shell exports for the seeded workspace.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    providers = tuple(str(provider) for provider in args.providers)
    corpus_source = CorpusSourceKind(str(args.corpus_source))
    try:
        if args.action == "generate":
            generated = generate_lab_corpus(
                providers=providers,
                count=int(args.count),
                corpus_source=corpus_source,
                output_dir=args.output_dir,
            )
            for batch in generated.batches:
                print(
                    f"  {batch.provider}:{batch.scope_label}: {batch.generated_count} files "
                    f"({batch.element_kind}) -> {batch.provider_dir}"
                )
            print(f"\nCorpus written to: {generated.output_dir}")
            return 0

        if args.action == "seed":
            seeded = seed_lab_demo(
                providers=providers,
                count=int(args.count),
                corpus_source=corpus_source,
                output_dir=args.output_dir,
            )
            if args.env_only:
                for key, value in seeded.env_vars.items():
                    print(f'export {key}="{value}"')
            else:
                print(
                    f"Seeded {seeded.counts.get('conversations', 0)} conversations, "
                    f"{seeded.counts.get('messages', 0)} messages"
                )
                print(f"\nDemo environment: {seeded.output_dir}")
                print("\nTo use:")
                for key, value in seeded.env_vars.items():
                    print(f'  export {key}="{value}"')
            return 0
    except ValueError as exc:
        print(f"lab-corpus: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

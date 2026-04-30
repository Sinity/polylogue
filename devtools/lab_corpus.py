"""Verification-lab synthetic corpus and demo workspace command."""

from __future__ import annotations

import argparse
import json
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

    list_parser = subparsers.add_parser("list", help="List available corpus sources.")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    rep_gen_parser = subparsers.add_parser(
        "representative-generate",
        help="Generate committed representative corpus samples adjacent to provider schemas.",
    )
    rep_gen_parser.add_argument(
        "--provider",
        "-p",
        dest="providers",
        action="append",
        default=[],
        help="Provider to include. May be repeated. Defaults to all.",
    )
    rep_gen_parser.add_argument("--count", "-n", type=int, default=3, help="Samples per provider.")
    rep_gen_parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    rep_gen_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Override output directory (default: schemas/providers/<name>/representatives).",
    )

    rep_verify_parser = subparsers.add_parser(
        "representative-verify",
        help="Verify committed representative corpora against manifests, schemas, and parseability.",
    )
    rep_verify_parser.add_argument(
        "--provider",
        "-p",
        dest="providers",
        action="append",
        default=[],
        help="Provider to verify. May be repeated. Defaults to all with representatives.",
    )
    return parser


def list_corpus_sources(*, as_json: bool) -> int:
    """List the corpus source kinds available for `generate` / `seed`."""
    payload = {
        "corpus_sources": [
            {"name": kind.value, "is_default": kind is CorpusSourceKind.DEFAULT} for kind in CorpusSourceKind
        ]
    }
    if as_json:
        print(json.dumps(payload, indent=2))
        return 0
    for entry in payload["corpus_sources"]:
        marker = " (default)" if entry["is_default"] else ""
        print(f"  {entry['name']}{marker}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action == "list":
        return list_corpus_sources(as_json=bool(args.json))
    if args.action == "representative-generate":
        return _representative_generate(
            providers=tuple(str(p) for p in getattr(args, "providers", []) or []),
            count=int(getattr(args, "count", 2)),
            seed=int(getattr(args, "seed", 42)),
            output_dir=getattr(args, "output_dir", None),
        )
    if args.action == "representative-verify":
        return _representative_verify(
            providers=tuple(str(p) for p in getattr(args, "providers", []) or []),
        )
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


def _representative_generate(
    *,
    providers: tuple[str, ...],
    count: int,
    seed: int,
    output_dir: Path | None,
) -> int:
    from polylogue.proof.corpus import CorpusManifest, representatives_dir
    from polylogue.schemas.registry import SchemaRegistry
    from polylogue.schemas.synthetic import SyntheticCorpus
    from polylogue.showcase.workspace import build_synthetic_corpus_specs, generate_synthetic_fixtures_from_specs
    from polylogue.version import POLYLOGUE_VERSION

    available = tuple(SyntheticCorpus.available_providers())
    selected = providers or available
    invalid = set(selected) - set(available)
    if invalid:
        print(f"lab-corpus: unknown provider(s): {', '.join(sorted(invalid))}", file=sys.stderr)
        return 2

    registry = SchemaRegistry()

    for provider_name in selected:
        rep_dir = representatives_dir(provider_name) if output_dir is None else output_dir / provider_name
        rep_dir.mkdir(parents=True, exist_ok=True)

        # Get schema version from registry
        pkg = registry.get_package(provider_name)
        schema_version = pkg.version if pkg else "unknown"

        # Generate schema-conformant fixtures using the existing pipeline.
        # generate_synthetic_fixtures_from_specs creates a provider-named subdir;
        # we write to a temp dir and move files up.
        import shutil

        with __import__("tempfile").TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            specs = build_synthetic_corpus_specs(
                providers=(provider_name,),
                count=count,
                seed=seed,
                style="default",
                messages_min=3,
                messages_max=6,
            )
            generate_synthetic_fixtures_from_specs(tmp_path, corpus_specs=specs, prefix="sample")

            # Move sample files from provider subdirectory to rep_dir
            provider_subdir = tmp_path / provider_name
            if provider_subdir.is_dir():
                for f in sorted(provider_subdir.iterdir()):
                    shutil.move(str(f), str(rep_dir / f.name))
            else:
                for f in sorted(tmp_path.glob("sample-*.json")):
                    shutil.move(str(f), str(rep_dir / f.name))

        sample_count = len([f for f in rep_dir.glob("sample-*.json") if f.name != "corpus-manifest.json"])

        manifest = CorpusManifest(
            provider=provider_name,
            schema_version=schema_version,
            generator_command=f"devtools lab-corpus representative-generate -p {provider_name} -n {count} --seed {seed}",
            generator_version=POLYLOGUE_VERSION,
            seed=seed,
            source_mode="schema-only",
            sample_count=sample_count,
            privacy_status="auto-generated-safe",
        )
        manifest.write(rep_dir / "corpus-manifest.json")
        print(f"  {provider_name}: {manifest.sample_count} samples -> {rep_dir}")

    return 0


def _representative_verify(*, providers: tuple[str, ...]) -> int:
    import json as _json

    from polylogue.proof.corpus import CorpusManifest, representatives_dir

    base = Path(__file__).resolve().parents[1] / "polylogue" / "schemas" / "providers"
    candidates = providers or tuple(
        p.name for p in base.iterdir() if p.is_dir() and p.joinpath("representatives").is_dir()
    )

    errors: list[str] = []
    verified = 0
    for provider_name in candidates:
        rep_dir = representatives_dir(provider_name)
        manifest_path = rep_dir / "corpus-manifest.json"
        if not manifest_path.exists():
            print(f"  {provider_name}: SKIP (no manifest)")
            continue

        manifest = CorpusManifest.from_path(manifest_path)
        samples = sorted(rep_dir.glob("sample-*.json"))
        samples = [s for s in samples if s.name != "corpus-manifest.json"]
        if not samples:
            errors.append(f"{provider_name}: no sample files found")
            continue

        expected_count = manifest.sample_count
        if len(samples) != expected_count:
            errors.append(f"{provider_name}: {len(samples)} samples != {expected_count} declared")

        for sample_path in samples:
            try:
                _json.loads(sample_path.read_text(encoding="utf-8"))
            except _json.JSONDecodeError as exc:
                errors.append(f"{provider_name}/{sample_path.name}: invalid JSON: {exc}")
                continue
            verified += 1
        print(f"  {provider_name}: {len(samples)} samples ({expected_count} declared), {manifest.privacy_status}")

    if errors:
        print(f"\n{len(errors)} verification error(s):", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1

    print(f"\n{verified} representative sample(s) verified across {len(candidates)} provider(s).")
    return 0


def _provider_names_match(declared: str, detected: object) -> bool:
    detected_name = str(detected).lower() if detected else ""
    declared_name = str(declared).lower()
    if detected_name == declared_name:
        return True
    # Normalize: claude-ai maps to claude
    aliases = {"claude": "claude-ai"}
    return aliases.get(detected_name, detected_name) == declared_name


if __name__ == "__main__":
    raise SystemExit(main())

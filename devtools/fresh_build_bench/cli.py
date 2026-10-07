"""``devtools bench fresh-build``: build, measure and compare fresh archives.

Subcommands::

    corpus sample --out DIR --seed N --fraction F   private stratified sample
    corpus files --out DIR FILE...                  private corpus of named files (a whale)
    run --corpus DIR --work DIR [--profile]         one measured daemon build
        [--max-rss-mib N] [--max-promotion-s N]     asserted budgets; exit 1 unless qualified
    report RECEIPT [--refresh]                      render one receipt
    components blob ...                     time one production stage
    compare BEFORE AFTER                            deltas and output equivalence
    profile SAMPLES [--thread PREFIX]               stack-sample tables

``run`` is heavy: start it through the declared ``fresh_build_bench``
AgentCTL operation, not in an interactive shell.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from devtools.fresh_build_bench.corpus import refuse_inside_checkout


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite number of seconds above zero")
    return seconds


def _refuse_repo_path(path: Path, what: str) -> None:
    try:
        refuse_inside_checkout(path, what)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools bench fresh-build", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    corpus = commands.add_parser("corpus", help="create a sealed benchmark corpus")
    corpus_kinds = corpus.add_subparsers(dest="kind", required=True)
    sample = corpus_kinds.add_parser("sample", help="copy a seeded stratified sample of real sources (private)")
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--seed", type=int, default=1)
    sample.add_argument("--fraction", type=float, required=True)
    sample.add_argument("--home", type=Path, default=Path.home())

    explicit = corpus_kinds.add_parser("files", help="copy the named real files (e.g. one whale) into a corpus")
    explicit.add_argument("--out", type=Path, required=True)
    explicit.add_argument("--home", type=Path, default=Path.home())
    explicit.add_argument(
        "--export",
        action="append",
        default=[],
        metavar="ORIGIN=PATH",
        help="stage an export file under exports/ORIGIN/ (chatgpt, claude-ai)",
    )
    explicit.add_argument(
        "--hooks",
        type=Path,
        default=None,
        help="copy a complete legacy hook spool tree into the sealed corpus and build archive",
    )
    explicit.add_argument(
        "--hooks-fraction",
        type=float,
        default=1.0,
        help="deterministic fraction of hook files to stage (default: 1.0)",
    )
    explicit.add_argument("files", nargs="*", type=Path)

    run = commands.add_parser("run", help="run one measured fresh build through polylogued run")
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True, help="empty directory for the archive, logs and receipt")
    run.add_argument("--label", default="run")
    run.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[2])
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--profile", action="store_true", help="run the in-daemon stack sampler")
    run.add_argument("--profile-interval", type=_positive_seconds, default=0.01)
    run.add_argument(
        "--stall-timeout", type=_positive_seconds, default=900.0, help="stop when nothing observable moves"
    )
    run.add_argument("--no-fingerprint", action="store_true")
    run.add_argument("--max-rss-mib", type=float, default=None, help="assert a whole-process-tree peak RSS budget")
    run.add_argument("--max-promotion-s", type=float, default=None, help="assert a time-to-promotion budget")
    run.add_argument("--max-terminal-s", type=float, default=None, help="assert a time-to-terminal budget")
    run.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra daemon environment (recorded in the receipt)",
    )

    report = commands.add_parser("report", help="render one receipt")
    report.add_argument("receipt", type=Path)
    report.add_argument("--json", action="store_true")
    report.add_argument("--refresh", action="store_true", help="recompute derived sections from the run directory")

    compare = commands.add_parser("compare", help="compare two receipts")
    compare.add_argument("before", type=Path)
    compare.add_argument("after", type=Path)
    compare.add_argument(
        "--allow-unqualified",
        action="store_true",
        help="compare runs that did not qualify (e.g. promoted but derived phase unsettled)",
    )

    components = commands.add_parser("components", help="time one production stage over a corpus", add_help=False)
    components.add_argument("rest", nargs=argparse.REMAINDER)

    profile = commands.add_parser("profile", help="summarise stack samples from a --profile run")
    profile.add_argument("samples", type=Path)
    profile.add_argument("--thread", default=None)
    profile.add_argument("--top", type=int, default=40)
    profile.add_argument("--collapsed", type=Path, default=None)
    return parser


def _refuse_source_root_path(out: Path, home: Path) -> None:
    """Refuse a corpus output inside a watched source root.

    Sampling copies transcripts into ``--out``; beneath a live source root
    those copies would be ingested by the production daemon as duplicate
    sessions. Both the ``--home`` roots and the invoking user's own roots are
    watched. Checked before anything is created.
    """
    from devtools.fresh_build_bench.corpus import default_sample_sources

    resolved = out.expanduser().resolve()
    for base in {home.expanduser().resolve(), Path.home().resolve()}:
        for source in default_sample_sources(base):
            root = source.root.resolve()
            if resolved == root or root in resolved.parents:
                raise SystemExit(f"--out must lie outside the watched source root {source.root}: {out}")


def _refuse_work_inside_sources(work: Path, corpus: Path) -> None:
    """Refuse a run's work directory inside the corpus or a watched source root.

    The daemon watches the corpus's source roots and the manifest seals every
    file under it: output written there would be read back as input and break
    the final manifest check.
    """
    resolved, corpus_path = work.resolve(), corpus.resolve()
    if resolved == corpus_path or corpus_path in resolved.parents:
        raise SystemExit(f"--work must live outside the corpus ({corpus_path})")
    _refuse_source_root_path(work, Path.home())


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "corpus":
        _refuse_repo_path(args.out, "--out")
        _refuse_source_root_path(args.out, args.home)
        if args.kind == "sample":
            from devtools.fresh_build_bench.corpus import default_sample_sources, sample_real

            manifest = sample_real(
                args.out,
                seed=args.seed,
                fraction=args.fraction,
                sources=default_sample_sources(args.home),
            )
        else:
            from devtools.fresh_build_bench.corpus import corpus_from_files

            exports = []
            for item in args.export:
                origin, _, path = item.partition("=")
                exports.append((origin, Path(path)))
            if not args.files and not exports and args.hooks is None:
                raise SystemExit("name at least one transcript, --export, or --hooks")
            manifest = corpus_from_files(
                args.out,
                args.files,
                home=args.home,
                exports=exports,
                hooks=args.hooks,
                hooks_fraction=args.hooks_fraction,
            )
        print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))
        return 0
    if args.command == "run":
        from devtools.fresh_build_bench.corpus import load_manifest, verify_manifest
        from devtools.fresh_build_bench.report import render
        from devtools.fresh_build_bench.run import RunConfig, run_build

        _refuse_repo_path(args.work, "--work")
        _refuse_repo_path(args.corpus, "--corpus")
        candidate_root = args.candidate.resolve()
        work = args.work.resolve()
        if work == candidate_root or candidate_root in work.parents:
            raise SystemExit(f"--work must live outside the candidate checkout ({candidate_root})")
        corpus_path = args.corpus.resolve()
        if corpus_path == candidate_root or candidate_root in corpus_path.parents:
            # A corpus copied or hand-built under the candidate checkout
            # would otherwise reach load_manifest below without the
            # checkout-path refusal --out/corpus-create apply: its transcript
            # copies and manifest are untracked repository content that can
            # be staged accidentally.
            raise SystemExit(f"--corpus must live outside the candidate checkout ({candidate_root})")
        _refuse_work_inside_sources(args.work, args.corpus)
        manifest = load_manifest(args.corpus)
        verify_manifest(args.corpus, manifest)
        if not manifest["file_count"]:
            raise SystemExit("the corpus holds no files; there is no build to measure")
        extra: list[tuple[str, str]] = []
        for item in args.env:
            key, _, value = item.partition("=")
            if not key.startswith("POLYLOGUE_"):
                raise SystemExit(f"--env accepts POLYLOGUE_* settings only: {item}")
            if key.startswith("POLYLOGUE_LOG_") or key.endswith(
                ("_ROOT", "_ROOTS", "_CONFIG", "_PATH", "_DIR", "_HOME", "_FILE")
            ):
                # A path override can add an unsealed source (or move the
                # log the receipt reads); the corpus is the only input.
                raise SystemExit(f"--env may not set source, path or log settings: {key}")
            if any(existing == key for existing, _value in extra):
                raise SystemExit(f"--env names {key} more than once")
            extra.append((key, value))
        config = RunConfig(
            corpus=args.corpus.resolve(),
            work=args.work,
            candidate=args.candidate.resolve(),
            python=args.python,
            label=args.label,
            profile=args.profile,
            profile_interval_s=args.profile_interval,
            stall_timeout_s=args.stall_timeout,
            fingerprint=not args.no_fingerprint,
            extra_env=tuple(extra),
            budgets=tuple(
                (name, float(limit))
                for name, limit in (
                    ("rss_peak_mib", args.max_rss_mib),
                    ("promotion_s", args.max_promotion_s),
                    ("terminal_s", args.max_terminal_s),
                )
                if limit is not None
            ),
        )
        receipt = run_build(config, progress=lambda line: print(line, flush=True))
        print(render(receipt))
        return 0 if receipt["qualified"] else 1
    if args.command == "report":
        from devtools.fresh_build_bench.report import render

        if args.refresh:
            from devtools.fresh_build_bench.report import refresh

            receipt = refresh(args.receipt)
        else:
            receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        print(json.dumps(receipt, indent=1) if args.json else render(receipt))
        return 0
    if args.command == "compare":
        from devtools.fresh_build_bench.report import compare

        before = json.loads(args.before.read_text(encoding="utf-8"))
        after = json.loads(args.after.read_text(encoding="utf-8"))
        admissible, text = compare(before, after, allow_unqualified=args.allow_unqualified)
        print(text)
        return 0 if admissible else 1
    if args.command == "components":
        from devtools.fresh_build_bench.components import main as components_main

        return components_main(args.rest)
    from devtools.fresh_build_bench.profile_report import main as profile_main

    forwarded = [str(args.samples), "--top", str(args.top)]
    if args.thread:
        forwarded += ["--thread", args.thread]
    if args.collapsed:
        forwarded += ["--collapsed", str(args.collapsed)]
    return profile_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())

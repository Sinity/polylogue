"""Render verification checks affected by changed paths or refs."""

from __future__ import annotations

import argparse
import json
import sys

from polylogue.proof.diffing import (
    build_affected_obligation_report,
    changed_paths_between_refs,
    obligation_ids_for_ref,
    render_affected_obligations,
    render_affected_obligations_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref", default="origin/master", help="Base git ref for changed-path and verification check diffing."
    )
    parser.add_argument(
        "--head-ref", default="HEAD", help="Head git ref for changed-path and verification check diffing."
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Changed repo-relative path. Repeat to bypass git diff path discovery.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=[],
        metavar="PATH",
        help=(
            "Speculative repo-relative path set (nargs+). Equivalent to repeating --path, "
            "intended for agents asking 'if I touch these files, what gates trip?' before editing. "
            "Combines with --path entries; presence of either bypasses git diff discovery."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable affected-check report.")
    parser.add_argument("--markdown", action="store_true", help="Emit a markdown-formatted affected-check report.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Emit the full verification-impact report with domain grouping, artifact taxonomy, and gate classification.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when blocking policy fails (only with --full).",
    )
    args = parser.parse_args(argv)
    combined_paths: list[str] = []
    seen_paths: set[str] = set()
    for value in (*args.path, *args.paths):
        if value and value not in seen_paths:
            seen_paths.add(value)
            combined_paths.append(value)

    if args.full:
        from devtools import repo_root as _get_root
        from devtools.verification_impact import (
            _print_human_report,
            build_verification_impact_report,
            evaluate_check_policy,
            render_markdown,
        )

        root = _get_root()
        full_report = build_verification_impact_report(
            root,
            base_ref=args.base_ref,
            head_ref=args.head_ref,
            changed_paths=combined_paths or None,
        )
        check_result = evaluate_check_policy(full_report)
        if args.check:
            full_report["check"] = check_result
        if args.json:
            json.dump(full_report, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        elif args.markdown:
            sys.stdout.write(render_markdown(full_report))
            sys.stdout.write("\n")
        else:
            _print_human_report(full_report)
        return 1 if args.check and check_result["status"] != "ok" else 0

    explicit_paths = tuple(combined_paths)
    base_ids: tuple[str, ...] | None
    head_ids: tuple[str, ...] | None
    if explicit_paths:
        paths = explicit_paths
        base_ids = ()
        head_ids = None
    else:
        paths = changed_paths_between_refs(args.base_ref, args.head_ref)
        base_ids = obligation_ids_for_ref(args.base_ref)
        head_ids = obligation_ids_for_ref(args.head_ref)
    report = build_affected_obligation_report(
        paths,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        base_obligation_ids=base_ids,
        head_obligation_ids=head_ids,
    )
    if args.json:
        json.dump(report.to_payload(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if args.markdown:
        sys.stdout.write(render_affected_obligations_markdown(report))
        sys.stdout.write("\n")
        return 0
    sys.stdout.write(render_affected_obligations(report))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]

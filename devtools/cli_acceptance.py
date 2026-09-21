"""CLI usability, accessibility and latency acceptance (``devtools verify cli-acceptance``).

One entry point that aggregates the four acceptance lanes for the public CLI
front door. The underlying per-behavior coverage already lives in
``tests/unit/cli``; what did not exist before this module is a single command
that renders the user-visible surface, lints it as *output* rather than as
assertions, and lands the latency evidence next to it.

Lanes
-----

**gallery** -- invokes a declared set of public, archive-free invocations
(help, the unsignalled-query hint, a rejected root filter, ``--version``) at
four terminal widths and three color lanes (default, ``NO_COLOR=1``,
``POLYLOGUE_FORCE_PLAIN=1``) and captures the exact text a user sees. The
gallery is the reviewable artifact; ``--gallery-out`` writes it as Markdown.

**lint** -- accessibility and legibility properties over that gallery, listed
in :data:`LINT_PROPERTIES`. These are output properties, not renderer unit
tests: they hold for whatever the CLI actually printed.

**scenarios** -- every declared invocation must terminate inside the CLI's own
exit vocabulary with no Python traceback, at every width and color lane.

**benchmarks** -- delegates to :mod:`devtools.verify_slos` so the declared
latency budgets in ``docs/plans/slo-catalog.yaml`` are measured by the same
command that proves the rendering.

Anti-vacuity
------------

Each of these makes the command exit non-zero: emitting an ANSI escape off a
TTY; letting ``NO_COLOR`` or ``POLYLOGUE_FORCE_PLAIN`` change the text of an
already-plain render (which would mean color carried meaning); truncating or
mangling the user's own token out of the error that rejects it; raising an
unhandled exception out of any declared invocation; returning an exit code
outside the declared table; dropping the "what to run next" line out of an
error; leaking a raw control character into rendered output; or growing the
count of lines that overflow the terminal past
:data:`REFLOW_OVERFLOW_BASELINE`. Deleting the sample table does not pass
vacuously -- an empty gallery is reported as ``empty_gallery``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from wcwidth import wcswidth

#: Widths the gallery renders at. 40 is the narrowest breakpoint the CLI
#: layout helpers declare (``_LayoutBreakpoints.NARROW_MIN``); 120 is a wide
#: terminal.
GALLERY_WIDTHS: tuple[int, ...] = (40, 60, 80, 120)

#: Color lanes. Off a TTY all three must render identical text -- that is the
#: ``color_is_not_meaning`` property, not an implementation detail.
COLOR_LANES: tuple[tuple[str, Mapping[str, str]], ...] = (
    ("default", {}),
    ("no-color", {"NO_COLOR": "1"}),
    ("force-plain", {"POLYLOGUE_FORCE_PLAIN": "1"}),
)

#: Exit codes a public CLI invocation may terminate with. 0/1/2 are the
#: ``OUTCOME_EXIT_CODES`` table (ok / degraded+error / empty); 2 is also
#: Click's usage-error code. Anything else (notably 3 for an unhandled
#: exception, or a signal-derived code) is a finding.
DECLARED_EXIT_CODES: frozenset[int] = frozenset({0, 1, 2})

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
#: C0/C1 and bidi-override characters that must never reach a rendered line.
_CONTROL_RE = re.compile("[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f\\u061c\\u200e\\u200f\\u202a-\\u202e\\u2066-\\u2069]")
_TRACEBACK = "Traceback (most recent call last)"


@dataclass(frozen=True, slots=True)
class GallerySample:
    """One public invocation rendered into the gallery."""

    name: str
    argv: tuple[str, ...]
    #: ``help`` / ``error`` / ``meta``. ``error`` samples additionally owe the
    #: reader a next action.
    kind: str
    #: A token the user typed that the rendered output must reproduce verbatim.
    echo: str | None = None
    #: Human note carried into the rendered gallery.
    note: str = ""


#: The declared acceptance surface. Every sample is archive-free and
#: daemon-free so the gallery is reproducible on any checkout.
GALLERY_SAMPLES: tuple[GallerySample, ...] = (
    GallerySample(
        "root-help",
        ("--help",),
        "help",
        note="The front door: dispatch model, product roles, query mode, verbs.",
    ),
    GallerySample(
        "find-help",
        ("find", "--help"),
        "help",
        note="The query verb's own help, including root-filter/verb-option ordering.",
    ),
    GallerySample(
        "read-help",
        ("read", "--help"),
        "help",
        note="An action verb's help.",
    ),
    GallerySample(
        "unsignalled-query",
        ("migration",),
        "error",
        echo="migration",
        note="A bare unquoted word is not query mode; the hint must name the signalled form.",
    ),
    GallerySample(
        "rejected-root-filter",
        ("--origin", "bogus-origin", "find", "x"),
        "error",
        echo="bogus-origin",
        note="A rejected root filter must echo the rejected value and list the valid origins.",
    ),
    GallerySample(
        "version",
        ("--version",),
        "meta",
        note="Version identity, used to attribute any finding to a build.",
    ),
)

#: Ratchet: the number of rendered lines wider than the terminal, per
#: ``sample@width``. The CLI's hand-written help and error prose does not
#: reflow (Click only wraps what it owns), so narrow terminals overflow today.
#: This is a recorded defect, not an accepted design: the numbers may fall,
#: never rise. Adding an unwrapped paragraph to any help text turns the lane
#: red at the width where it first does not fit.
#:
#: RAISED ONCE, 2026-09-21 (polylogue-1khzy), five rows. The raise is stated
#: here because a falling-only ratchet is worth nothing if a red row can be
#: re-recorded without a reason. The whole difference is f8a63f0b6 (#5249),
#: "Output dialect normalization (--format/--json + --to/--out)", measured by
#: capturing the same gallery at 44c5f868d -- the commit that recorded the
#: original numbers -- and diffing the rendered lines. Nothing else widened:
#:
#: * ``root-help``: #5249 added three option lines that did not exist before
#:   (``--to [terminal|stdout|browser|clipboard|file]`` at 47, its wrapped
#:   help continuation at 78, ``--out PATH ... File path for --to file.`` at
#:   58), which is +3 at width 40 and +1 at width 60; and it widened
#:   ``-f, --format [...]`` from 74 to 89 by publishing the ``md`` and
#:   ``jsonl`` short spellings, which is +1 at width 80.
#: * ``read-help``: the same ``-f, --format`` line went 79 -> 88 (+1 at width
#:   80) and ``--json  Alias for --format json.`` at 58 is new (+1 at 40).
#:
#: Why this is not narrowed back: every one of those lines is surface the CLI
#: now genuinely accepts. ``--to``, ``--out`` and ``--json`` are options a
#: reader needs to see, and ``md``/``jsonl`` are spellings the parser takes;
#: hiding them from ``--help`` would trade a legible help screen for a green
#: number. The help text also does not reflow at all -- every width renders
#: the identical 80-column layout -- so narrowing means deleting words, not
#: wrapping them. Fixing *that* is the ratchet's actual target and is not
#: what this raise buys.
REFLOW_OVERFLOW_BASELINE: Mapping[str, int] = {
    "root-help@40": 159,
    "root-help@60": 101,
    "root-help@80": 10,
    "root-help@120": 2,
    "find-help@40": 14,
    "find-help@60": 11,
    "find-help@80": 0,
    "find-help@120": 0,
    "read-help@40": 55,
    "read-help@60": 37,
    "read-help@80": 3,
    "read-help@120": 0,
    "unsignalled-query@40": 3,
    "unsignalled-query@60": 3,
    "unsignalled-query@80": 1,
    "unsignalled-query@120": 0,
    "rejected-root-filter@40": 2,
    "rejected-root-filter@60": 1,
    "rejected-root-filter@80": 1,
    "rejected-root-filter@120": 1,
    "version@40": 0,
    "version@60": 0,
    "version@80": 0,
    "version@120": 0,
}

#: Documented lint properties, in report order.
LINT_PROPERTIES: tuple[tuple[str, str], ...] = (
    ("no_ansi_off_tty", "Rendered output off a TTY carries no ANSI escape sequence."),
    ("color_is_not_meaning", "NO_COLOR and POLYLOGUE_FORCE_PLAIN do not change already-plain text."),
    ("control_safety", "No raw control or bidi-override character reaches a rendered line."),
    ("echo_preserved", "A rejected user token survives verbatim at every width."),
    ("failure_is_textual", "A non-zero exit is announced in words, not by color alone."),
    ("repair_named", "An error names a command the reader can run next."),
    ("no_traceback", "No public invocation prints a Python traceback."),
    ("declared_exit_code", "Every invocation exits inside the declared exit table."),
    ("reflow_overflow_ratchet", "Lines wider than the terminal do not exceed the recorded baseline."),
)


@dataclass(frozen=True, slots=True)
class GalleryFrame:
    """One captured render."""

    sample: str
    width: int
    lane: str
    exit_code: int
    text: str

    @property
    def key(self) -> str:
        return f"{self.sample}@{self.width}/{self.lane}"

    def overflow_lines(self) -> tuple[str, ...]:
        return tuple(line for line in self.text.splitlines() if max(0, wcswidth(line)) > self.width)


@dataclass(frozen=True, slots=True)
class Finding:
    """A concrete, blocking accessibility or semantics violation."""

    code: str
    subject: str
    message: str


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    frames: tuple[GalleryFrame, ...]
    findings: tuple[Finding, ...]
    overflow: Mapping[str, int] = field(default_factory=dict)
    benchmark_exit: int | None = None

    @property
    def ok(self) -> bool:
        return not self.findings and (self.benchmark_exit in (None, 0))


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------


def capture_frame(sample: GallerySample, width: int, lane: str, lane_env: Mapping[str, str]) -> GalleryFrame:
    """Invoke one sample under one width and color lane."""
    # Imported lazily: the gallery is the only devtools lane that loads the CLI.
    from click.testing import CliRunner

    from polylogue.cli.click_app import cli

    env = {"COLUMNS": str(width), "LINES": "40", "TERM": "dumb", **lane_env}
    # Neither override is set unless the lane asks for it.
    for name in ("NO_COLOR", "POLYLOGUE_FORCE_PLAIN"):
        env.setdefault(name, "")
    # ``prog_name`` matters: the usage line and the "try --help" repair hint are
    # rendered from it, so a default of "cli" would make the gallery lie about
    # what an operator sees from the installed entry point.
    result = CliRunner(env=env).invoke(cli, list(sample.argv), prog_name="polylogue", catch_exceptions=True)
    text = result.output or ""
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        import traceback as _traceback

        text += "".join(_traceback.format_exception(result.exception))
    return GalleryFrame(sample.name, width, lane, int(result.exit_code), text)


def build_gallery(
    samples: Sequence[GallerySample] = GALLERY_SAMPLES,
    widths: Sequence[int] = GALLERY_WIDTHS,
    lanes: Sequence[tuple[str, Mapping[str, str]]] = COLOR_LANES,
) -> tuple[GalleryFrame, ...]:
    return tuple(
        capture_frame(sample, width, lane, lane_env)
        for sample in samples
        for width in widths
        for lane, lane_env in lanes
    )


def render_gallery_markdown(frames: Sequence[GalleryFrame], samples: Sequence[GallerySample] = GALLERY_SAMPLES) -> str:
    """Render the reviewable gallery: one fenced block per sample and width."""
    notes = {sample.name: sample for sample in samples}
    lines = [
        "# CLI acceptance gallery",
        "",
        "Generated by `devtools verify cli-acceptance --gallery-out`. Every block is the",
        "exact text a user sees. Color lanes render identically off a TTY, so one block",
        "per width is the whole truth.",
        "",
    ]
    for name in dict.fromkeys(frame.sample for frame in frames):
        sample = notes.get(name)
        lines.append(f"## {name}")
        lines.append("")
        if sample is not None:
            lines.append(f"`polylogue {' '.join(sample.argv)}`")
            lines.append("")
            if sample.note:
                lines.append(sample.note)
                lines.append("")
        for frame in frames:
            if frame.sample != name or frame.lane != "default":
                continue
            lines.append(f"### {frame.width} columns (exit {frame.exit_code})")
            lines.append("")
            lines.append("```")
            lines.extend(frame.text.rstrip("\n").splitlines() or [""])
            lines.append("```")
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------


def lint_findings(
    frames: Sequence[GalleryFrame],
    samples: Sequence[GallerySample] = GALLERY_SAMPLES,
    baseline: Mapping[str, int] = REFLOW_OVERFLOW_BASELINE,
) -> tuple[Finding, ...]:
    """Apply :data:`LINT_PROPERTIES` to a captured gallery."""
    findings: list[Finding] = []
    if not frames:
        return (Finding("empty_gallery", "gallery", "No frames were captured; the acceptance lane proves nothing."),)

    by_name = {sample.name: sample for sample in samples}
    for frame in frames:
        sample = by_name.get(frame.sample)
        if sample is None:
            findings.append(Finding("undeclared_sample", frame.key, "Frame has no declared sample."))
            continue
        if _ANSI_RE.search(frame.text):
            findings.append(Finding("no_ansi_off_tty", frame.key, "ANSI escape sequence rendered off a TTY."))
        controls = sorted({f"U+{ord(char):04X}" for char in _CONTROL_RE.findall(frame.text)})
        if controls:
            findings.append(
                Finding("control_safety", frame.key, f"Raw control characters rendered: {', '.join(controls)}")
            )
        if _TRACEBACK in frame.text:
            findings.append(Finding("no_traceback", frame.key, "A Python traceback reached the terminal."))
        if frame.exit_code not in DECLARED_EXIT_CODES:
            findings.append(
                Finding(
                    "declared_exit_code", frame.key, f"Exit {frame.exit_code} is outside {sorted(DECLARED_EXIT_CODES)}."
                )
            )
        if sample.echo is not None and sample.echo not in frame.text:
            findings.append(
                Finding("echo_preserved", frame.key, f"The rejected token {sample.echo!r} did not survive rendering.")
            )
        if frame.exit_code != 0:
            lowered = frame.text.lower()
            if "error" not in lowered and "usage" not in lowered:
                findings.append(
                    Finding("failure_is_textual", frame.key, "Non-zero exit carries no textual failure marker.")
                )
        if sample.kind == "error" and "polylogue " not in frame.text:
            findings.append(Finding("repair_named", frame.key, "Error output names no command to run next."))

    findings.extend(_color_lane_findings(frames))
    findings.extend(_overflow_findings(frames, baseline))
    return tuple(findings)


def _color_lane_findings(frames: Sequence[GalleryFrame]) -> tuple[Finding, ...]:
    """Off a TTY, the three color lanes must be byte-identical."""
    findings: list[Finding] = []
    by_render: dict[tuple[str, int], dict[str, GalleryFrame]] = {}
    for frame in frames:
        by_render.setdefault((frame.sample, frame.width), {})[frame.lane] = frame
    for (sample, width), lanes in sorted(by_render.items()):
        default = lanes.get("default")
        if default is None:
            continue
        for lane, frame in sorted(lanes.items()):
            if lane == "default":
                continue
            if frame.text != default.text or frame.exit_code != default.exit_code:
                findings.append(
                    Finding(
                        "color_is_not_meaning",
                        f"{sample}@{width}/{lane}",
                        f"{lane} output differs from the default render off a TTY.",
                    )
                )
    return tuple(findings)


def measure_overflow(frames: Sequence[GalleryFrame]) -> dict[str, int]:
    """Count lines wider than the terminal, per ``sample@width``."""
    measured: dict[str, int] = {}
    for frame in frames:
        if frame.lane != "default":
            continue
        measured[f"{frame.sample}@{frame.width}"] = len(frame.overflow_lines())
    return measured


def _overflow_findings(frames: Sequence[GalleryFrame], baseline: Mapping[str, int]) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    for key, count in sorted(measure_overflow(frames).items()):
        allowed = baseline.get(key)
        if allowed is None:
            findings.append(
                Finding("reflow_overflow_ratchet", key, f"{count} overflowing lines with no recorded baseline.")
            )
        elif count > allowed:
            findings.append(
                Finding(
                    "reflow_overflow_ratchet",
                    key,
                    f"{count} lines exceed the terminal width, above the recorded baseline of {allowed}.",
                )
            )
    return tuple(findings)


def tightenable_overflow(
    frames: Sequence[GalleryFrame], baseline: Mapping[str, int] = REFLOW_OVERFLOW_BASELINE
) -> dict[str, tuple[int, int]]:
    """Baselines that now over-allow: ``key -> (measured, recorded)``."""
    return {
        key: (count, baseline[key])
        for key, count in sorted(measure_overflow(frames).items())
        if key in baseline and count < baseline[key]
    }


# ---------------------------------------------------------------------------
# Scenarios and benchmarks
# ---------------------------------------------------------------------------


def scenario_findings(frames: Sequence[GalleryFrame]) -> tuple[Finding, ...]:
    """Every declared invocation terminated typed, at every width and lane."""
    return tuple(
        finding
        for finding in lint_findings(frames)
        if finding.code in {"no_traceback", "declared_exit_code", "empty_gallery"}
    )


def run(*, baseline: Mapping[str, int] = REFLOW_OVERFLOW_BASELINE) -> AcceptanceReport:
    frames = build_gallery()
    return AcceptanceReport(
        frames=frames,
        findings=lint_findings(frames, baseline=baseline),
        overflow=measure_overflow(frames),
    )


def _report_payload(report: AcceptanceReport) -> dict[str, object]:
    return {
        "frames": len(report.frames),
        "samples": [sample.name for sample in GALLERY_SAMPLES],
        "widths": list(GALLERY_WIDTHS),
        "lanes": [lane for lane, _ in COLOR_LANES],
        "properties": [{"code": code, "statement": statement} for code, statement in LINT_PROPERTIES],
        "overflow": dict(report.overflow),
        "tightenable": {key: list(value) for key, value in tightenable_overflow(report.frames).items()},
        "findings": [
            {"code": finding.code, "subject": finding.subject, "message": finding.message}
            for finding in report.findings
        ],
        "benchmark_exit": report.benchmark_exit,
        "ok": report.ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render, lint and measure the public CLI acceptance surface.")
    parser.add_argument("--json", action="store_true", help="Emit the acceptance report as JSON.")
    parser.add_argument("--gallery-out", metavar="PATH", help="Write the reviewable Markdown gallery to PATH.")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip the latency lane (devtools bench slo).")
    parser.add_argument(
        "--include-lab",
        action="store_true",
        help="Include the lab-tier latency rows (cold start, warm status, concurrent reads).",
    )
    args = parser.parse_args(argv)

    frames = build_gallery()
    findings = lint_findings(frames)
    overflow = measure_overflow(frames)

    benchmark_exit: int | None = None
    if not args.skip_benchmarks:
        from devtools import verify_slos

        slo_argv = ["--include-lab"] if args.include_lab else []
        benchmark_exit = verify_slos.main(slo_argv)

    report = AcceptanceReport(frames=frames, findings=findings, overflow=overflow, benchmark_exit=benchmark_exit)

    if args.gallery_out:
        path = Path(args.gallery_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_gallery_markdown(frames), encoding="utf-8")

    if args.json:
        print(json.dumps(_report_payload(report), indent=2, sort_keys=True))
    else:
        print(
            f"cli-acceptance: {len(frames)} frames "
            f"({len(GALLERY_SAMPLES)} samples x {len(GALLERY_WIDTHS)} widths x {len(COLOR_LANES)} color lanes)"
        )
        for code, statement in LINT_PROPERTIES:
            failed = sum(1 for finding in findings if finding.code == code)
            print(f"cli-acceptance: {'FAIL' if failed else 'ok  '} {code}: {statement}")
        for key, (measured, recorded) in tightenable_overflow(frames).items():
            print(f"cli-acceptance: baseline can tighten: {key} measured {measured} < recorded {recorded}")
        for finding in findings:
            print(f"cli-acceptance: {finding.code}: {finding.subject}: {finding.message}", file=sys.stderr)
        if benchmark_exit is not None:
            print(f"cli-acceptance: latency lane exited {benchmark_exit}")
        if args.gallery_out:
            print(f"cli-acceptance: gallery written to {args.gallery_out}")

    return 0 if report.ok else 1


__all__ = [
    "COLOR_LANES",
    "DECLARED_EXIT_CODES",
    "GALLERY_SAMPLES",
    "GALLERY_WIDTHS",
    "LINT_PROPERTIES",
    "REFLOW_OVERFLOW_BASELINE",
    "AcceptanceReport",
    "Finding",
    "GalleryFrame",
    "GallerySample",
    "build_gallery",
    "capture_frame",
    "lint_findings",
    "main",
    "measure_overflow",
    "render_gallery_markdown",
    "run",
    "scenario_findings",
    "tightenable_overflow",
]

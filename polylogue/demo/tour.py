"""One-command public demo tour for a deterministic archive."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from polylogue.scenarios import (
    DEMO_CODEX_LINEAGE_FORK_SESSION_ID,
    DEMO_CODEX_TERMINAL_ERROR_SESSION_ID,
)

from .models import DemoTourResult, DemoTourStep
from .seed import seed_demo_archive
from .verify import verify_demo_archive

FIRST_RESULT_BUDGET_S = 30.0
FULL_TOUR_BUDGET_S = 420.0


def run_demo_tour(
    *,
    output_dir: Path,
    archive_root: Path | None = None,
    force: bool = True,
) -> DemoTourResult:
    """Run the deterministic public tour and write report artifacts."""

    resolved_output = output_dir.expanduser().resolve()
    if force and resolved_output.exists():
        shutil.rmtree(resolved_output)
    resolved_output.mkdir(parents=True, exist_ok=True)

    resolved_archive = (archive_root or resolved_output / "archive").expanduser().resolve()
    resolved_archive.mkdir(parents=True, exist_ok=True)

    transcript_path = resolved_output / "transcript.txt"
    report_json_path = resolved_output / "report.json"
    report_markdown_path = resolved_output / "report.md"
    recording_tape_path = resolved_output / "recording.tape"
    command_output_dir = resolved_output / "command-output"
    command_output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    transcript_parts: list[str] = []

    seed_start = time.perf_counter()
    seed = asyncio.run(seed_demo_archive(resolved_archive, force=True, with_overlays=True))
    seed_duration = time.perf_counter() - seed_start
    seed_passed = sum(row.ok for row in seed.construct_coverage)
    transcript_parts.append(
        _format_internal_step(
            "prepare deterministic proof archive",
            (
                f"Seeded {seed.session_count} sessions, {seed.message_count} messages, "
                f"and {seed.assertion_count} user-state assertions in {seed_duration:.3f}s. "
                f"Fixture audit: {seed_passed}/{len(seed.construct_coverage)} declared constructs satisfied."
            ),
        )
    )

    verify_start = time.perf_counter()
    verify = verify_demo_archive(resolved_archive, require_overlays=True)
    verify_duration = time.perf_counter() - verify_start
    transcript_parts.append(
        _format_internal_step(
            "verify evidence before presenting it",
            (
                f"Verification {'passed' if verify.ok else 'failed'} in {verify_duration:.3f}s; "
                f"{len(verify.absolute_path_leaks)} path leaks and {len(verify.problems)} semantic problems. "
                "The complete fixture and verification audit remains in report.json."
            ),
        )
    )

    steps: list[DemoTourStep] = []
    first_result_s = 0.0
    env = _tour_env(resolved_archive)
    command_specs = (
        (
            "structural failure receipt",
            ("--id", DEMO_CODEX_TERMINAL_ERROR_SESSION_ID, "read", "--view", "messages", "--limit", "10"),
            (
                "Start with the receipt, not a dashboard: the assistant acknowledges a failed test run, "
                "and the tool result retains exit_code=4 plus the exact error output."
            ),
        ),
        (
            "failed actions aggregate",
            ("actions where is_error:true | group by tool | count",),
            (
                "Now aggregate the same structural field across providers. This query counts normalized "
                "failed actions; it does not search prose for the word 'error'."
            ),
        ),
        (
            "composed lineage",
            ("--id", DEMO_CODEX_LINEAGE_FORK_SESSION_ID, "read", "--view", "chronicle"),
            (
                "Read a fork as one logical chronicle: inherited parent messages remain attributable to "
                "their origin while the fork contributes only its divergent tail."
            ),
        ),
        (
            "archive facets",
            ("analyze", "--facets"),
            (
                "Only after inspecting evidence, zoom out to the archive: eleven sessions across five "
                "origins, with deferred families labeled rather than silently guessed."
            ),
        ),
    )
    for index, (name, args, explanation) in enumerate(command_specs, start=1):
        step, rendered = _run_cli_step(
            name=name,
            args=args,
            explanation=explanation,
            env=env,
            output_path=command_output_dir / f"{index:02d}-{_slug(name)}.txt",
        )
        steps.append(step)
        transcript_parts.append(rendered)
        if index == 1:
            first_result_s = time.perf_counter() - start

    total_duration_s = time.perf_counter() - start
    problems = _tour_problems(
        verify_ok=verify.ok, steps=tuple(steps), first_result_s=first_result_s, total_s=total_duration_s
    )
    ok = not problems

    transcript_path.write_text("\n".join(transcript_parts), encoding="utf-8")
    _write_recording_tape(recording_tape_path, out_dir_name=resolved_output.name)

    result = DemoTourResult(
        archive_root=resolved_archive,
        output_dir=resolved_output,
        ok=ok,
        first_result_s=first_result_s,
        total_duration_s=total_duration_s,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        transcript_path=transcript_path,
        recording_tape_path=recording_tape_path,
        seed=seed,
        verify=verify,
        steps=tuple(steps),
        problems=problems,
    )
    report_json_path.write_text(
        json.dumps(_public_payload(result.to_payload(), base_dir=resolved_output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path.write_text(_render_report_markdown(result), encoding="utf-8")
    return result


def _tour_env(archive_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    env["POLYLOGUE_FORCE_PLAIN"] = "1"
    return env


def _run_cli_step(
    *,
    name: str,
    args: tuple[str, ...],
    explanation: str,
    env: dict[str, str],
    output_path: Path,
) -> tuple[DemoTourStep, str]:
    command = (sys.executable, "-m", "polylogue", *args)
    started = time.perf_counter()
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    duration_s = time.perf_counter() - started
    output = completed.stdout
    if completed.stderr:
        output = output + ("\n" if output else "") + completed.stderr
    output_path.write_text(output, encoding="utf-8")
    step = DemoTourStep(
        name=name,
        command=("polylogue", *args),
        exit_code=completed.returncode,
        duration_s=duration_s,
        output_path=output_path,
        bytes_written=len(output.encode("utf-8")),
    )
    rendered = "\n".join(
        [
            f"$ {shlex.join(step.command)}",
            explanation,
            f"exit={step.exit_code} duration={step.duration_s:.3f}s bytes={step.bytes_written}",
            output.strip(),
            "",
        ]
    )
    return step, rendered


def _format_internal_step(name: str, summary: str) -> str:
    return "\n".join([f"# {name}", summary, ""])


def _tour_problems(
    *,
    verify_ok: bool,
    steps: tuple[DemoTourStep, ...],
    first_result_s: float,
    total_s: float,
) -> tuple[str, ...]:
    problems: list[str] = []
    if not verify_ok:
        problems.append("demo archive verification failed")
    failed = [step.name for step in steps if step.exit_code != 0]
    if failed:
        problems.append(f"tour commands failed: {', '.join(failed)}")
    if first_result_s > FIRST_RESULT_BUDGET_S:
        problems.append(f"first result exceeded {FIRST_RESULT_BUDGET_S:.0f}s budget: {first_result_s:.3f}s")
    if total_s > FULL_TOUR_BUDGET_S:
        problems.append(f"full tour exceeded {FULL_TOUR_BUDGET_S:.0f}s budget: {total_s:.3f}s")
    return tuple(problems)


def _render_report_markdown(result: DemoTourResult) -> str:
    status = "passed" if result.ok else "failed"
    construct_count = len(result.verify.construct_coverage)
    constructs_passed = sum(row.ok for row in result.verify.construct_coverage)
    lines = [
        "# Polylogue Demo Tour Report",
        "",
        f"Status: **{status}**",
        "",
        "This report was produced by `polylogue demo tour` against the deterministic",
        "private-data-free demo archive. The transcript is ordered as an evidence story:",
        "one receipt, a structural aggregate, copied-lineage composition, then archive scope.",
        "",
        "## What this tour proves",
        "",
        "- A provider-normalized tool result can retain structural failure evidence, including an exit code and output.",
        "- The query surface can aggregate failed actions from structured fields rather than keyword matching assistant prose.",
        "- A fork can be read as a logical chronicle while inherited messages keep their original refs.",
        "- One deterministic archive can expose multiple provider origins through the same read and analysis surfaces.",
        f"- The fixture verifier found {constructs_passed}/{construct_count} declared constructs and no absolute-path leaks.",
        "",
        "## What this tour does not prove",
        "",
        "- It is not a scale or latency claim for a private multi-million-message archive.",
        "- It does not prove complete capture fidelity for every supported provider or every historical export variant.",
        "- It does not prove that semantic retrieval, reviewed memory, or context injection improves agent outcomes.",
        "- It does not prove the proposed Sinex-backed storage architecture or selective physical deletion.",
        "- It is a deterministic product-contract demonstration, not a provider invoice or general model-behavior study.",
        "",
        "## Timings",
        "",
        f"- First evidence result: {result.first_result_s:.3f}s (budget {FIRST_RESULT_BUDGET_S:.0f}s)",
        f"- Full tour: {result.total_duration_s:.3f}s (budget {FULL_TOUR_BUDGET_S:.0f}s)",
        "",
        "## Archive",
        "",
        "- Archive root: `archive`",
        f"- Sessions: {result.seed.session_count}",
        f"- Messages: {result.seed.message_count}",
        f"- User overlays: {'present' if result.verify.overlays_present else 'missing'}",
        f"- Declared fixture constructs: {constructs_passed}/{construct_count} satisfied",
        "",
        "## Steps",
        "",
        "| Step | Exit | Duration | Bytes | Output |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {step.name} | {step.exit_code} | {step.duration_s:.3f}s | {step.bytes_written} | `command-output/{step.output_path.name}` |"
        for step in result.steps
    )
    lines.extend(["", "## Problems", ""])
    if result.problems:
        lines.extend(f"- {problem}" for problem in result.problems)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Human-readable evidence story: `transcript.txt`",
            "- Complete machine-readable fixture, verification, timing, and command record: `report.json`",
            "- This bounded scope statement: `report.md`",
            "- VHS recording recipe: `recording.tape`",
            "- Raw command outputs: `command-output/`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_recording_tape(path: Path, *, out_dir_name: str) -> None:
    # Reference the out-dir by basename only: the tape must reproduce the run
    # it sits next to without baking a machine-local absolute path into a
    # potentially committed artifact.
    out_ref = shlex.quote(out_dir_name)
    path.write_text(
        textwrap.dedent(
            f"""\
            Output "demo-tour.gif"
            Set FontSize 18
            Set Width 1080
            Set Height 720
            Set Padding 18
            Set TypingSpeed 0.04
            Type "polylogue demo tour --out-dir {out_ref}"
            Enter
            Sleep 15s
            Type "cat {out_ref}/transcript.txt"
            Enter
            Sleep 4s
            Type "cat {out_ref}/report.md"
            Enter
            Sleep 2s
            """
        ),
        encoding="utf-8",
    )


def _public_payload(value: object, *, base_dir: Path) -> object:
    """Replace local absolute output paths with stable artifact-relative paths."""

    if isinstance(value, dict):
        return {key: _public_payload(item, base_dir=base_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_payload(item, base_dir=base_dir) for item in value]
    if isinstance(value, str):
        try:
            path = Path(value)
            if path.is_absolute() and path.is_relative_to(base_dir):
                return str(path.relative_to(base_dir)) or "."
        except ValueError:
            pass
    return value


def _slug(text: str) -> str:
    return "-".join(part for part in text.lower().replace("/", " ").split() if part)


__all__ = ["FULL_TOUR_BUDGET_S", "FIRST_RESULT_BUDGET_S", "run_demo_tour"]

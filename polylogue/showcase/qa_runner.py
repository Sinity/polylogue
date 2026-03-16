"""QA orchestration: composable audit → exercises → invariants pipeline.

Supports both synthetic (fresh workspace) and live (existing DB) modes,
with optional real-data ingestion into isolated workspaces.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polylogue.showcase.invariants import (
    InvariantResult,
    check_invariants,
    format_invariant_summary,
)
from polylogue.showcase.runner import ShowcaseResult, ShowcaseRunner


@dataclass
class QAResult:
    """Complete QA session result."""

    audit_passed: bool = False
    audit_report: dict[str, Any] | None = None
    audit_skipped: bool = False
    showcase_result: ShowcaseResult | None = None
    exercises_skipped: bool = False
    invariant_results: list[InvariantResult] = field(default_factory=list)
    invariants_skipped: bool = False
    report_dir: Path | None = None

    @property
    def all_passed(self) -> bool:
        """True if all executed stages passed."""
        if not self.audit_skipped and not self.audit_passed:
            return False
        if self.showcase_result and self.showcase_result.failed > 0:
            return False
        if any(r.status == "fail" for r in self.invariant_results):
            return False
        return True


def _generate_extra_exercises() -> list:
    """Generate dynamic exercises from CLI introspection and schema catalog."""
    from polylogue.showcase.generators import (
        generate_format_exercises,
        generate_schema_exercises,
    )

    exercises = []
    exercises.extend(generate_schema_exercises())
    exercises.extend(generate_format_exercises())
    return exercises


def _create_workspace(workspace_dir: Path | None = None) -> tuple[Path, dict[str, str]]:
    """Create an isolated QA workspace with env vars.

    Returns (workspace_path, env_vars_dict).
    """
    if workspace_dir is None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="polylogue-qa-"))

    data_home = workspace_dir / "data"
    state_home = workspace_dir / "state"
    archive_root = workspace_dir / "archive"
    render_root = archive_root / "render"
    fake_home = workspace_dir / "home"

    for d in [data_home, state_home, archive_root, render_root, fake_home]:
        d.mkdir(parents=True, exist_ok=True)

    env_vars = {
        "HOME": str(fake_home),
        "XDG_DATA_HOME": str(data_home),
        "XDG_STATE_HOME": str(state_home),
        "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
        "POLYLOGUE_RENDER_ROOT": str(render_root),
        "POLYLOGUE_FORCE_PLAIN": "1",
    }

    return workspace_dir, env_vars


def _run_pipeline_in_workspace(
    env_vars: dict[str, str],
    workspace_dir: Path,
    *,
    source_names: list[str] | None = None,
    regenerate_schemas: bool = False,
) -> None:
    """Run the ingestion pipeline inside a workspace with env overrides."""
    from polylogue.config import Config, Source, get_config
    from polylogue.pipeline.runner import run_sources

    old_env: dict[str, str | None] = {}
    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        # When source_names are provided, resolve from user config
        if source_names:
            user_config = get_config()
            sources = [s for s in user_config.sources if s.name in source_names]
        else:
            # Synthetic: fixture dir is the source
            fixture_dir = workspace_dir / "fixtures"
            sources = []
            if fixture_dir.exists():
                for provider_dir in sorted(fixture_dir.iterdir()):
                    if provider_dir.is_dir():
                        sources.append(Source(name=provider_dir.name, path=provider_dir))

        archive_root = Path(env_vars["POLYLOGUE_ARCHIVE_ROOT"])
        render_root = Path(env_vars["POLYLOGUE_RENDER_ROOT"])

        config = Config(
            archive_root=archive_root,
            render_root=render_root,
            sources=sources,
        )

        stage = "all"
        if regenerate_schemas:
            stage = "all"  # generate-schemas is part of "all"

        asyncio.run(run_sources(
            config=config,
            stage=stage,
            plan=None,
            ui=None,
            source_names=None,
        ))
    finally:
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _generate_synthetic_fixtures(workspace_dir: Path, *, count: int = 3) -> None:
    """Generate schema-driven synthetic fixtures into the workspace."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    fixture_dir = workspace_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    data_home = workspace_dir / "data"
    inbox_dir = data_home / "polylogue" / "inbox"
    inbox_dir.mkdir(parents=True, exist_ok=True)

    for provider in SyntheticCorpus.available_providers():
        corpus = SyntheticCorpus.for_provider(provider)
        provider_dir = fixture_dir / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        ext = ".json" if corpus.wire_format.encoding == "json" else ".jsonl"
        raw_items = corpus.generate(
            count=count,
            messages_per_conversation=range(6, 20),
            seed=42,
            style="showcase",
        )
        for idx, raw_bytes in enumerate(raw_items):
            (provider_dir / f"showcase-{idx:02d}{ext}").write_bytes(raw_bytes)

        # Copy into inbox so get_sources() finds them
        dest = inbox_dir / provider
        dest.mkdir(parents=True, exist_ok=True)
        for f in provider_dir.iterdir():
            if f.is_file():
                (dest / f.name).write_bytes(f.read_bytes())


def run_qa_session(
    *,
    live: bool = False,
    fresh: bool = True,
    ingest: bool = False,
    source_names: list[str] | None = None,
    regenerate_schemas: bool = False,
    skip_audit: bool = False,
    skip_exercises: bool = False,
    skip_invariants: bool = False,
    workspace_dir: Path | None = None,
    workspace_env: dict[str, str] | None = None,
    report_dir: Path | None = None,
    provider: str | None = None,
    verbose: bool = False,
    fail_fast: bool = False,
    tier_filter: int | None = None,
    synthetic_count: int = 3,
) -> QAResult:
    """Execute a composable QA session.

    Stages (in order, each skippable):
      1. Schema audit
      2. Exercises (showcase)
      3. Invariant checks
      4. Report generation
    """
    result = QAResult(report_dir=report_dir)
    workspace_env_for_runner: dict[str, str] | None = workspace_env

    # Skip workspace setup if only running audit (no data needed)
    needs_workspace = not skip_exercises

    # --- Workspace setup ---
    if needs_workspace and fresh and not live:
        # Synthetic mode: create workspace, generate fixtures, ingest
        ws_dir, ws_env = _create_workspace(workspace_dir)
        _generate_synthetic_fixtures(ws_dir, count=synthetic_count)
        _run_pipeline_in_workspace(
            ws_env, ws_dir,
            regenerate_schemas=regenerate_schemas,
        )
        workspace_env_for_runner = ws_env
        if report_dir is None:
            report_dir = ws_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            result.report_dir = report_dir

    elif needs_workspace and fresh and live and source_names:
        # Real data in fresh workspace
        ws_dir, ws_env = _create_workspace(workspace_dir)
        _run_pipeline_in_workspace(
            ws_env, ws_dir,
            source_names=source_names,
            regenerate_schemas=regenerate_schemas,
        )
        workspace_env_for_runner = ws_env
        if report_dir is None:
            report_dir = ws_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            result.report_dir = report_dir

    elif needs_workspace and fresh and live:
        # Fresh workspace with all real sources
        ws_dir, ws_env = _create_workspace(workspace_dir)
        _run_pipeline_in_workspace(
            ws_env, ws_dir,
            source_names=None,
            regenerate_schemas=regenerate_schemas,
        )
        workspace_env_for_runner = ws_env
        if report_dir is None:
            report_dir = ws_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            result.report_dir = report_dir

    elif needs_workspace and live and ingest:
        # Live mode with ingestion on existing DB
        from polylogue.config import get_config
        from polylogue.pipeline.runner import run_sources

        config = get_config()
        names = source_names if source_names else None
        asyncio.run(run_sources(
            config=config,
            stage="all",
            plan=None,
            ui=None,
            source_names=names,
        ))

    # --- Step 1: Schema audit ---
    if skip_audit:
        result.audit_skipped = True
        result.audit_passed = True  # Don't block exercises
    else:
        try:
            from polylogue.schemas.audit import audit_all_providers, audit_provider

            if provider:
                audit_report = audit_provider(provider)
            else:
                audit_report = audit_all_providers()

            result.audit_report = audit_report.to_json()
            result.audit_passed = audit_report.all_passed

            if not audit_report.all_passed:
                if verbose:
                    print(audit_report.format_text(), file=sys.stderr)
                return result
        except Exception as e:
            result.audit_report = {"error": str(e)}
            result.audit_passed = False
            return result

    # --- Step 2: Exercises ---
    if skip_exercises:
        result.exercises_skipped = True
    else:
        extra = _generate_extra_exercises()
        runner = ShowcaseRunner(
            live=live and not fresh,
            output_dir=report_dir,
            fail_fast=fail_fast,
            verbose=verbose,
            tier_filter=tier_filter,
            extra_exercises=extra,
            workspace_env=workspace_env_for_runner,
        )
        showcase_result = runner.run()
        result.showcase_result = showcase_result

    # --- Step 3: Invariant checks ---
    if skip_invariants:
        result.invariants_skipped = True
    elif result.showcase_result:
        result.invariant_results = check_invariants(result.showcase_result.results)

    # --- Step 4: Save reports ---
    if report_dir:
        result.report_dir = report_dir
        _save_qa_reports(result, report_dir)

    return result


def _save_qa_reports(result: QAResult, report_dir: Path) -> None:
    """Save all QA artifacts to the report directory."""
    import json

    report_dir.mkdir(parents=True, exist_ok=True)

    if result.audit_report:
        (report_dir / "schema-audit.json").write_text(
            json.dumps(result.audit_report, indent=2)
        )

    invariant_data = [
        {
            "invariant": r.invariant_name,
            "exercise": r.exercise_name,
            "status": r.status,
            "error": r.error,
        }
        for r in result.invariant_results
    ]
    if invariant_data:
        (report_dir / "invariant-checks.json").write_text(
            json.dumps(invariant_data, indent=2)
        )

    if result.showcase_result:
        from polylogue.showcase.report import (
            generate_qa_markdown,
            save_reports,
        )

        save_reports(result.showcase_result)
        qa_md = generate_qa_markdown(result.showcase_result)

        if result.invariant_results:
            invariant_summary = format_invariant_summary(result.invariant_results)
            qa_md += f"\n\n## Invariant Checks\n\n```\n{invariant_summary}\n```\n"

        (report_dir / "qa-session.md").write_text(qa_md)


def format_qa_summary(result: QAResult) -> str:
    """Format a human-readable QA session summary."""
    lines: list[str] = []

    # Audit status
    if result.audit_skipped:
        lines.append("Schema Audit: SKIPPED")
    elif result.audit_passed:
        lines.append("Schema Audit: PASS")
    else:
        lines.append("Schema Audit: FAIL — halting QA")
        return "\n".join(lines)

    # Showcase status
    if result.exercises_skipped:
        lines.append("Exercises: SKIPPED")
    elif result.showcase_result:
        sr = result.showcase_result
        total = len(sr.results)
        lines.append(
            f"Exercises: {sr.passed}/{total} passed, "
            f"{sr.failed} failed, {sr.skipped} skipped "
            f"({sr.total_duration_ms/1000:.1f}s)"
        )

    # Invariant status
    if result.invariants_skipped:
        lines.append("Invariants: SKIPPED")
    elif result.invariant_results:
        inv_passed = sum(1 for r in result.invariant_results if r.status == "pass")
        inv_failed = sum(1 for r in result.invariant_results if r.status == "fail")
        inv_skipped = sum(1 for r in result.invariant_results if r.status == "skip")
        lines.append(
            f"Invariants: {inv_passed} pass, {inv_failed} fail, {inv_skipped} skip"
        )

    # Overall
    status = "PASS" if result.all_passed else "FAIL"
    lines.append(f"\nOverall: {status}")

    if result.report_dir:
        lines.append(f"Reports: {result.report_dir}")

    return "\n".join(lines)


__all__ = [
    "QAResult",
    "format_qa_summary",
    "run_qa_session",
]

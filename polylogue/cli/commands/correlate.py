"""Session-to-git correlation commands (#1690 phase 1-3)."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import click

from polylogue.cli.shared.types import AppEnv

if TYPE_CHECKING:
    from polylogue.insights.session_commit import SessionCorrelationResult


@click.command("correlate")
@click.argument("session_id")
@click.option("--repo-path", "-r", default=None, help="Path to git repository. Defaults to current directory.")
@click.option("--since-hours", "-w", type=int, default=2, help="Hours before/after session to scan for commits.")
@click.option("--format", "-f", "output_format", type=click.Choice(["json"]), default=None)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.3,
    help="Minimum confidence for file-overlap commit detection.",
)
@click.option(
    "--github-api/--no-github-api",
    default=True,
    help="Cross-reference issue/PR references with GitHub API if gh CLI is available.",
)
@click.option(
    "--otlp",
    is_flag=True,
    default=False,
    help="Add OTLP span evidence to the correlation output.",
)
@click.pass_obj
def correlate_command(
    env: AppEnv,
    session_id: str,
    repo_path: str | None,
    since_hours: int,
    output_format: str | None,
    confidence_threshold: float,
    github_api: bool,
    otlp: bool,
) -> None:
    """Show git commits and GitHub refs within the session's time window (#1690)."""
    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.insights.session_commit import build_correlation_result

    conv = run_coroutine_sync(env.polylogue.get_session(session_id))
    if conv is None:
        env.ui.error(f"Session not found: {session_id}")
        raise SystemExit(1)

    # Determine time window from session timestamps.
    start = conv.created_at
    end = conv.updated_at
    if start is None or end is None:
        env.ui.error("Session has no timestamp data.")
        raise SystemExit(1)

    # Determine repo path.
    repo: str = repo_path or "."
    meta: dict[str, object] = conv.provider_meta or {} if isinstance(conv.provider_meta, dict) else {}
    if not repo_path:
        repo_url = meta.get("git_repository_url")
        if repo_url and isinstance(repo_url, str):
            repo = repo_url
        else:
            cwd = meta.get("cwd")
            repo = str(cwd) if cwd and isinstance(cwd, str) else "."

    # Build messages as dicts for the detection logic.
    messages: list[dict[str, object]] = []
    for msg in conv.messages:
        msg_dict: dict[str, object] = {
            "id": msg.id,
            "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
            "text": msg.text,
        }
        content_blocks = msg.content_blocks if hasattr(msg, "content_blocks") else []
        msg_dict["content_blocks"] = list(content_blocks) if content_blocks else []
        messages.append(msg_dict)

    # Use session messages directly — they carry full text and
    # content_blocks from the archive read path already.

    result = build_correlation_result(
        session_id=session_id,
        messages=messages,
        session_created_at=start,
        session_updated_at=end,
        repo_path=repo,
        before_hours=since_hours,
        after_hours=since_hours,
        confidence_threshold=confidence_threshold,
    )

    # Phase 3: Optionally cross-reference with GitHub API
    if github_api and (result.issue_refs or result.pr_refs):
        result = _enrich_with_github_api(result)

    if output_format == "json":
        from polylogue.insights.session_commit import correlation_result_to_payload

        env.ui.console.print(json.dumps(correlation_result_to_payload(result), indent=2))
        return

    _print_correlation_result(env, result)

    # OTLP span evidence
    if otlp:
        _print_otlp_evidence(env, session_id, output_format)


def _print_otlp_evidence(env: AppEnv, session_id: str, output_format: str | None) -> None:
    """Print OTLP span evidence for a session, if available."""
    from polylogue.insights.otlp_correlation import get_session_tool_timing
    from polylogue.paths import active_index_db_path

    try:
        timing = get_session_tool_timing(str(active_index_db_path()), session_id)
    except Exception:
        env.ui.console.print("\n[dim]No OTLP data available.[/dim]")
        return

    if output_format == "json":
        env.ui.console.print(json.dumps({"otlp_tool_timing": timing.as_dict()}, indent=2))
        return

    if not timing.evidence_available or not timing.tool_timings:
        env.ui.console.print("\n[dim]No OTLP data for this session.[/dim]")
        return

    env.ui.console.print(f"\n[bold]OTLP Tool Timing[/bold] ({timing.total_tools_with_otlp} tools)")
    env.ui.console.print("  Evidence source: otlp_span")

    for t in timing.tool_timings:
        status_color = "green" if t.status == "ok" else "red"
        env.ui.console.print(
            f"  - [bold]{t.tool_name}[/bold] [{status_color}]{t.status}[/{status_color}] [dim]{t.duration_ms}ms[/dim]"
        )
        if t.start_time:
            env.ui.console.print(f"    {t.start_time} → {t.end_time}")


def _enrich_with_github_api(result: SessionCorrelationResult) -> SessionCorrelationResult:
    """Cross-reference issue/PR refs against the GitHub API via gh CLI."""

    from polylogue.insights.session_commit import GitHubRef

    all_refs: list[tuple[GitHubRef, str]] = []
    for ref in result.issue_refs:
        all_refs.append((ref, "issue"))
    for ref in result.pr_refs:
        all_refs.append((ref, "pr"))

    enriched_issues: list[GitHubRef] = []
    enriched_prs: list[GitHubRef] = []

    for ref, default_kind in all_refs:
        if ref.owner and ref.repo:
            try:
                gh_result = subprocess.run(
                    [
                        "gh",
                        "issue",
                        "view",
                        str(ref.number),
                        "--repo",
                        f"{ref.owner}/{ref.repo}",
                        "--json",
                        "title,state,url",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if gh_result.returncode == 0:
                    import json

                    data = json.loads(gh_result.stdout)
                    enriched = GitHubRef(
                        owner=ref.owner,
                        repo=ref.repo,
                        number=ref.number,
                        kind=default_kind,
                        url=data.get("url") or ref.url,
                        raw_match=ref.raw_match,
                        message_id=ref.message_id,
                    )
                    if default_kind == "pr":
                        enriched_prs.append(enriched)
                    else:
                        enriched_issues.append(enriched)
                    continue
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass
        # Keep original if gh fails
        if default_kind == "pr":
            enriched_prs.append(ref)
        else:
            enriched_issues.append(ref)

    return SessionCorrelationResult(
        session_id=result.session_id,
        window_start=result.window_start,
        window_end=result.window_end,
        repo=result.repo,
        commits=result.commits,
        issue_refs=enriched_issues,
        pr_refs=enriched_prs,
        file_paths=result.file_paths,
    )


def _print_correlation_result(env: AppEnv, result: SessionCorrelationResult) -> None:
    """Pretty-print a correlation result to the terminal."""
    from datetime import datetime

    env.ui.console.print(f"\n[bold]Session:[/bold] {result.session_id}")
    if result.window_start:
        try:
            ws = datetime.fromisoformat(result.window_start)
            we = datetime.fromisoformat(result.window_end)
            env.ui.console.print(
                f"[bold]Window:[/bold] {ws.strftime('%Y-%m-%d %H:%M')} → {we.strftime('%Y-%m-%d %H:%M')}"
            )
        except (ValueError, TypeError):
            env.ui.console.print(f"[bold]Window:[/bold] {result.window_start} → {result.window_end}")
    if result.repo:
        env.ui.console.print(f"[bold]Repo:[/bold] {result.repo}")

    files = result.file_paths or []
    env.ui.console.print(f"[bold]Touched files:[/bold] {len(files)}")

    # Commits
    commits = result.commits or []
    if commits:
        env.ui.console.print(f"\n[bold]Commits:[/bold] {len(commits)}")
        for c in commits:
            method_label = {
                "explicit_ref": "●",
                "file_overlap": "○",
                "time_window": "·",
            }.get(c.detection_method, " ")
            env.ui.console.print(
                f"  {method_label} [bold]{c.commit_sha[:8]}[/bold] "
                f"(confidence: {c.confidence:.2f}, files: {c.file_overlap_count}) "
                f"[dim]{c.detection_method}[/dim]"
            )
    else:
        env.ui.console.print("\n[dim]No commits found in session window.[/dim]")

    # Issue refs
    issue_refs = result.issue_refs or []
    if issue_refs:
        env.ui.console.print(f"\n[bold]Issue references:[/bold] {len(issue_refs)}")
        for ref in issue_refs:
            label = f"{ref.owner}/{ref.repo}#{ref.number}" if ref.owner else f"#{ref.number}"
            env.ui.console.print(f"  - {label} [dim]{ref.raw_match}[/dim]")

    # PR refs
    pr_refs = result.pr_refs or []
    if pr_refs:
        env.ui.console.print(f"\n[bold]PR references:[/bold] {len(pr_refs)}")
        for ref in pr_refs:
            label = f"{ref.owner}/{ref.repo}#{ref.number}" if ref.owner else f"#{ref.number}"
            env.ui.console.print(f"  - {label} [dim]{ref.raw_match}[/dim]")


def _parse_git_log(output: str, session_files: set[str]) -> list[dict[str, object]]:
    """Parse git log output into commit dicts with file-overlap detection."""
    commits: list[dict[str, object]] = []
    blocks = output.split("\n---\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        commit_hash = lines[0].strip()
        date_str = lines[1].strip()
        subject = lines[2].strip()
        changed_files = {f.strip() for f in lines[3:] if f.strip()}
        overlaps = bool(session_files & changed_files) if session_files else False
        commits.append(
            {
                "hash": commit_hash,
                "short_hash": commit_hash[:8],
                "date": date_str,
                "subject": subject,
                "changed_files": sorted(changed_files)[:20],
                "overlaps_files": overlaps,
            }
        )
    return commits

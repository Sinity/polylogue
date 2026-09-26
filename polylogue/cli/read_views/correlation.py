"""Correlation read-view handler."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from polylogue.cli.operation_kernel import OperationKernelError, OperationRequest
from polylogue.cli.read_dispatch import daemon_route_disabled, dispatch_read
from polylogue.cli.read_view_registry import CORRELATION_READ_VIEW_OPTION_NAMES
from polylogue.cli.read_views.base import (
    ReadViewCorrelationOptions,
    ReadViewInvocation,
    ReadViewOptionValues,
    deliver_content,
)
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


def build_correlation_options(values: ReadViewOptionValues) -> ReadViewCorrelationOptions:
    """Build options owned by the correlation read view."""

    return ReadViewCorrelationOptions(
        repo_path=cast(str | None, values.get("repo_path")),
        since_hours=cast(int, values.get("since_hours", 2)),
        confidence_threshold=cast(float, values.get("confidence_threshold", 0.3)),
        github_api=cast(bool, values.get("github_api", True)),
    )


def run_read_correlation(env: AppEnv, request: RootModeRequest, invocation: ReadViewInvocation) -> None:
    """Render GitHub/Git correlation evidence around one session."""

    assert invocation.session_id is not None
    options = cast(ReadViewCorrelationOptions, invocation.options or ReadViewCorrelationOptions())
    projection = invocation.projection_spec.projection if invocation.projection_spec is not None else None
    github_api = (
        projection.correlation_github_api
        if projection and projection.correlation_github_api is not None
        else options.github_api
    )
    try:
        result, _ = dispatch_read(
            env.config,
            OperationRequest(
                "read.correlation",
                {
                    "session_id": invocation.session_id,
                    "repo_path": projection.correlation_repo_path
                    if projection and projection.correlation_repo_path is not None
                    else options.repo_path,
                    "since_hours": projection.correlation_since_hours
                    if projection and projection.correlation_since_hours is not None
                    else options.since_hours,
                    "confidence_threshold": projection.correlation_confidence_threshold
                    if projection and projection.correlation_confidence_threshold is not None
                    else options.confidence_threshold,
                },
            ),
            daemon_disabled=daemon_route_disabled(flag=bool(request.params.get("no_daemon"))),
        )
    except OperationKernelError as exc:
        from polylogue.cli.render.outcome import exit_for_read_failure

        exit_for_read_failure(exc)
    payload = result.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("read.correlation returned no payload")
    document = dict(payload)
    if github_api:
        _enrich_github_refs(document)
    fmt = invocation.output_format or "plaintext"
    content = json.dumps(document, indent=2) + "\n" if fmt == "json" else _render_correlation_plain(document)
    deliver_content(env, content, destination=invocation.destination, out_path=invocation.out_path, output_format=fmt)


def _enrich_github_refs(document: dict[str, object]) -> None:
    """Keep optional CLI GitHub lookups outside the daemon's pinned read."""

    import subprocess

    for key in ("issue_refs", "pr_refs"):
        refs = document.get(key)
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            owner, repo, number = ref.get("owner"), ref.get("repo"), ref.get("number")
            if not isinstance(owner, str) or not isinstance(repo, str) or not isinstance(number, int):
                continue
            try:
                completed = subprocess.run(
                    ["gh", "issue", "view", str(number), "--repo", f"{owner}/{repo}", "--json", "title,state,url"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if completed.returncode == 0:
                    fetched = json.loads(completed.stdout)
                    if isinstance(fetched, dict) and isinstance(fetched.get("url"), str):
                        ref["url"] = fetched["url"]
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError, json.JSONDecodeError):
                pass


def _render_correlation_plain(payload: Mapping[str, object]) -> str:
    from datetime import datetime

    from polylogue.core.localtime import format_local_datetime

    lines = [f"Session: {payload.get('session_id')}"]
    start, end = payload.get("window_start"), payload.get("window_end")
    if isinstance(start, str) and isinstance(end, str):
        try:
            lines.append(
                f"Window: {format_local_datetime(datetime.fromisoformat(start))} -> {format_local_datetime(datetime.fromisoformat(end))}"
            )
        except ValueError:
            lines.append(f"Window: {start} -> {end}")
    if payload.get("repo"):
        lines.append(f"Repo: {payload['repo']}")
    files = payload.get("file_paths") or []
    lines.append(f"Touched files: {len(files) if isinstance(files, list) else 0}")
    commits = payload.get("commits") or []
    if isinstance(commits, list) and commits:
        lines.append(f"Commits: {len(commits)}")
        for commit in commits:
            if not isinstance(commit, Mapping):
                continue
            method = str(commit.get("detection_method", ""))
            label = {"origin_reported": "=", "explicit_ref": "*", "file_overlap": "o", "time_window": "."}.get(
                method, " "
            )
            lines.append(
                f"  {label} {str(commit.get('commit_sha', ''))[:8]} "
                f"(confidence: {float(commit.get('confidence', 0)):.2f}, files: {commit.get('file_overlap_count', 0)}) {method}"
            )
            if commit.get("disagreement_note"):
                lines.append(f"    ! {commit['disagreement_note']}")
    else:
        lines.append("No commits found in session window.")
    for key, title in (("issue_refs", "Issue references"), ("pr_refs", "PR references")):
        refs = payload.get(key) or []
        if isinstance(refs, list) and refs:
            lines.append(f"{title}: {len(refs)}")
            for ref in refs:
                if isinstance(ref, Mapping):
                    label = (
                        f"{ref['owner']}/{ref['repo']}#{ref['number']}" if ref.get("owner") else f"#{ref.get('number')}"
                    )
                    lines.append(f"  - {label} {ref.get('raw_match', '')} ({ref.get('source', '')})")
    disagreements = payload.get("disagreements") or []
    if isinstance(disagreements, list) and disagreements:
        lines.append(f"Disagreements: {len(disagreements)}")
        for item in disagreements:
            if isinstance(item, Mapping):
                lines.append(f"  - {item.get('kind')}: {item.get('detail')}")
    return "\n".join(lines) + "\n"


__all__ = ["CORRELATION_READ_VIEW_OPTION_NAMES", "build_correlation_options", "run_read_correlation"]

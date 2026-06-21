"""Probe the deployed Polylogue CLI, daemon API, and browser-capture receiver."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SYSTEMWIDE_PATH = (
    "/run/current-system/sw/bin:"
    "/etc/profiles/per-user/sinity/bin:"
    "/home/sinity/.nix-profile/bin:"
    "/home/sinity/.local/bin:"
    "/usr/bin:"
    "/bin"
)

REQUIRED_DAEMON_ROUTES = (
    "/api/status",
    "/api/read-view-profiles",
)
REQUIRED_WEB_ROUTES = (
    "/api/sessions?limit=1&offset=0",
    "/api/facets",
)
REQUIRED_RECEIVER_ROUTES = ("/v1/status",)
COMPLETION_PROBES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("query-then-connector", "polylogue find id:abc t", ("then",)),
    ("query-action", "polylogue find id:abc then s", ("select",)),
    (
        "read-views",
        "polylogue find id:abc then read --view ",
        ("summary", "messages", "raw", "context-pack"),
    ),
    (
        "message-formats",
        "polylogue find id:abc then read --view messages --format ",
        ("json", "ndjson", "text"),
    ),
)


@dataclass(frozen=True, slots=True)
class CommandProbe:
    name: str
    path: str | None
    exit_code: int | None
    stdout: str
    stderr: str
    ok: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RouteProbe:
    url: str
    status: int | None
    ok: bool
    payload: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CompletionProbe:
    name: str
    comp_words: str
    exit_code: int | None
    candidates: list[str]
    expected: list[str]
    missing: list[str]
    ok: bool
    stderr: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BrowserCaptureArchiveProbe:
    spool_path: str
    source_db_path: str
    spooled_count: int
    latest_spooled_path: str | None
    latest_spooled_mtime: float | None
    raw_rows: int | None
    latest_raw_file_mtime_ms: int | None
    ok: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DeploymentSmokeReport:
    ok: bool
    path: str
    repo_head: str | None
    daemon_base_url: str
    receiver_base_url: str
    commands: list[CommandProbe]
    routes: list[RouteProbe]
    completions: list[CompletionProbe]
    browser_capture_archive: BrowserCaptureArchiveProbe
    diagnostics: dict[str, Any]
    failures: list[str]


def _command_env(path: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = path
    return env


def _resolve_command(name: str, *, path: str) -> str | None:
    return shutil.which(name, path=path)


def _run_command(command: list[str], *, path: str, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=_command_env(path),
    )


def _run_completion_command(
    *,
    path: str,
    timeout_s: float,
    comp_words: str,
) -> subprocess.CompletedProcess[str]:
    env = _command_env(path)
    words = comp_words.split()
    env["COMP_WORDS"] = comp_words
    env["COMP_CWORD"] = str(len(words) if comp_words.endswith(" ") else max(0, len(words) - 1))
    env["_POLYLOGUE_COMPLETE"] = "bash_complete"
    return subprocess.run(
        ["polylogue"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )


def _open_url(url: str, *, timeout_s: float) -> Any:
    return urllib.request.urlopen(url, timeout=timeout_s)


def _timeout_stream(value: bytes | None) -> str:
    return value.decode("utf-8", "replace") if value else ""


def _repo_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _version_commit(stdout: str) -> str | None:
    match = re.search(r"\+([0-9a-f]{7,40})(?:[-+].*)?$", stdout.strip())
    return match.group(1) if match else None


def _probe_command(command: list[str], *, path: str, timeout_s: float) -> CommandProbe:
    name = command[0]
    resolved = _resolve_command(name, path=path)
    if resolved is None:
        return CommandProbe(
            name=" ".join(command),
            path=None,
            exit_code=None,
            stdout="",
            stderr="",
            ok=False,
            error="not_found_on_path",
        )
    try:
        result = _run_command(command, path=path, timeout_s=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return CommandProbe(
            name=" ".join(command),
            path=resolved,
            exit_code=None,
            stdout=_timeout_stream(exc.stdout),
            stderr=_timeout_stream(exc.stderr),
            ok=False,
            error="timeout",
        )
    except OSError as exc:
        return CommandProbe(
            name=" ".join(command),
            path=resolved,
            exit_code=None,
            stdout="",
            stderr="",
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return CommandProbe(
        name=" ".join(command),
        path=resolved,
        exit_code=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
        ok=result.returncode == 0,
    )


def _completion_candidates(stdout: str) -> list[str]:
    candidates: list[str] = []
    for line in stdout.splitlines():
        kind, sep, value = line.partition(",")
        if kind == "plain" and sep and value:
            candidates.append(value)
    return candidates


def _probe_completion(
    *,
    name: str,
    comp_words: str,
    expected: tuple[str, ...],
    path: str,
    timeout_s: float,
) -> CompletionProbe:
    resolved = _resolve_command("polylogue", path=path)
    if resolved is None:
        return CompletionProbe(
            name=name,
            comp_words=comp_words,
            exit_code=None,
            candidates=[],
            expected=list(expected),
            missing=list(expected),
            ok=False,
            stderr="",
            error="polylogue_not_found_on_path",
        )
    try:
        result = _run_completion_command(path=path, timeout_s=timeout_s, comp_words=comp_words)
    except subprocess.TimeoutExpired as exc:
        return CompletionProbe(
            name=name,
            comp_words=comp_words,
            exit_code=None,
            candidates=_completion_candidates(_timeout_stream(exc.stdout)),
            expected=list(expected),
            missing=list(expected),
            ok=False,
            stderr=_timeout_stream(exc.stderr),
            error="timeout",
        )
    except OSError as exc:
        return CompletionProbe(
            name=name,
            comp_words=comp_words,
            exit_code=None,
            candidates=[],
            expected=list(expected),
            missing=list(expected),
            ok=False,
            stderr="",
            error=f"{type(exc).__name__}: {exc}",
        )
    candidates = _completion_candidates(result.stdout)
    missing = [candidate for candidate in expected if candidate not in candidates]
    return CompletionProbe(
        name=name,
        comp_words=comp_words,
        exit_code=result.returncode,
        candidates=candidates,
        expected=list(expected),
        missing=missing,
        ok=result.returncode == 0 and not missing,
        stderr=result.stderr.strip(),
        error=None,
    )


def _probe_browser_capture_archive(*, archive_root: Path) -> BrowserCaptureArchiveProbe:
    spool_path = archive_root / "browser-capture"
    source_db_path = archive_root / "source.db"
    files = (
        sorted(
            (path for path in spool_path.rglob("*.json") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if spool_path.exists()
        else []
    )
    latest = files[0] if files else None
    latest_mtime = latest.stat().st_mtime if latest is not None else None
    raw_rows: int | None = None
    latest_raw_file_mtime_ms: int | None = None
    error: str | None = None
    if source_db_path.exists():
        try:
            with sqlite3.connect(source_db_path) as conn:
                row = conn.execute(
                    """
                    SELECT count(*), max(file_mtime_ms)
                    FROM raw_sessions
                    WHERE source_path LIKE '%browser-capture%'
                    """
                ).fetchone()
            raw_rows = int(row[0]) if row is not None and row[0] is not None else 0
            latest_raw_file_mtime_ms = int(row[1]) if row is not None and row[1] is not None else None
        except sqlite3.Error as exc:
            error = f"{type(exc).__name__}: {exc}"
    elif files:
        error = "source_db_missing"
    ok = error is None and (not files or (raw_rows is not None and raw_rows > 0))
    return BrowserCaptureArchiveProbe(
        spool_path=str(spool_path),
        source_db_path=str(source_db_path),
        spooled_count=len(files),
        latest_spooled_path=str(latest) if latest is not None else None,
        latest_spooled_mtime=latest_mtime,
        raw_rows=raw_rows,
        latest_raw_file_mtime_ms=latest_raw_file_mtime_ms,
        ok=ok,
        error=error,
    )


def _probe_route(url: str, *, timeout_s: float) -> RouteProbe:
    try:
        with _open_url(url, timeout_s=timeout_s) as response:
            raw = response.read().decode("utf-8", "replace")
            payload: dict[str, Any] | None = None
            if raw:
                decoded = json.loads(raw)
                payload = decoded if isinstance(decoded, dict) else {"value": decoded}
            return RouteProbe(url=url, status=response.status, ok=200 <= response.status < 300, payload=payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        payload = None
        if body:
            try:
                decoded = json.loads(body)
                payload = decoded if isinstance(decoded, dict) else {"value": decoded}
            except json.JSONDecodeError:
                payload = {"body": body[:500]}
        return RouteProbe(url=url, status=exc.code, ok=False, payload=payload, error="http_error")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return RouteProbe(url=url, status=None, ok=False, error=f"{type(exc).__name__}: {exc}")


def _diagnose(
    commands: list[CommandProbe],
    routes: list[RouteProbe],
    repo_head: str | None,
    browser_capture_archive: BrowserCaptureArchiveProbe,
) -> dict[str, Any]:
    command_by_name = {probe.name: probe for probe in commands}
    route_by_path = {probe.url.rsplit("/", 1)[-1]: probe for probe in routes}
    polylogue = command_by_name.get("polylogue --version")
    polylogued = command_by_name.get("polylogued --version")
    deployed_commit = _version_commit(polylogue.stdout) if polylogue is not None else None
    read_view_route = route_by_path.get("read-view-profiles")
    facets_route = next((probe for probe in routes if probe.url.endswith("/api/facets")), None)
    daemon_status = next((probe for probe in routes if probe.url.endswith("/api/status")), None)
    browser_capture_state = None
    if daemon_status is not None and daemon_status.payload is not None:
        browser_capture_state = {
            "component_state": daemon_status.payload.get("component_state"),
            "browser_capture_active": daemon_status.payload.get("browser_capture_active"),
            "browser_capture": daemon_status.payload.get("browser_capture"),
        }
    likely_causes: list[str] = []
    next_actions: list[str] = []
    if polylogued is not None and not polylogued.ok:
        likely_causes.append("deployed polylogued predates the --version option")
        next_actions.append("rebuild/restart the systemwide Polylogue package or check PATH ordering")
    if read_view_route is not None and not read_view_route.ok:
        likely_causes.append("deployed daemon API predates /api/read-view-profiles or is not restarted")
        next_actions.append("restart polylogued after deploying a build that contains the read-view profiles route")
    if facets_route is not None and not facets_route.ok:
        likely_causes.append("web-shell facets route exceeds the deployed smoke timeout")
        next_actions.append("profile /api/facets and move expensive facet aggregation behind caching or bounded reads")
    if not browser_capture_archive.ok and browser_capture_archive.spooled_count > 0:
        likely_causes.append("browser-capture artifacts are spooled but absent from raw archive rows")
        next_actions.append(
            "verify daemon watch/catch-up includes the browser-capture spool and restart the deployed daemon"
        )
    if repo_head is not None and deployed_commit is not None and not repo_head.startswith(deployed_commit):
        likely_causes.append("systemwide polylogue commit differs from the current checkout")
        next_actions.append("compare the deployed package input with the checkout before trusting live UI probes")
    return {
        "repo_head": repo_head,
        "polylogue_version": polylogue.stdout if polylogue is not None and polylogue.stdout else None,
        "polylogue_commit": deployed_commit,
        "polylogued_version_ok": polylogued.ok if polylogued is not None else None,
        "browser_capture_state": browser_capture_state,
        "browser_capture_archive": asdict(browser_capture_archive),
        "likely_causes": likely_causes,
        "next_actions": next_actions,
    }


def build_report(
    *,
    path: str,
    daemon_base_url: str,
    receiver_base_url: str,
    archive_root: Path,
    timeout_s: float,
) -> DeploymentSmokeReport:
    commands = [
        _probe_command(["polylogue", "--version"], path=path, timeout_s=timeout_s),
        _probe_command(["polylogued", "--version"], path=path, timeout_s=timeout_s),
    ]
    completions = [
        _probe_completion(name=name, comp_words=comp_words, expected=expected, path=path, timeout_s=timeout_s)
        for name, comp_words, expected in COMPLETION_PROBES
    ]
    routes = [
        *(
            _probe_route(f"{daemon_base_url.rstrip('/')}{route}", timeout_s=timeout_s)
            for route in (*REQUIRED_DAEMON_ROUTES, *REQUIRED_WEB_ROUTES)
        ),
        *(
            _probe_route(f"{receiver_base_url.rstrip('/')}{route}", timeout_s=timeout_s)
            for route in REQUIRED_RECEIVER_ROUTES
        ),
    ]
    failures: list[str] = []
    failures.extend(f"command:{probe.name}:{probe.error or probe.exit_code}" for probe in commands if not probe.ok)
    failures.extend(f"route:{probe.url}:{probe.error or probe.status}" for probe in routes if not probe.ok)
    failures.extend(
        f"completion:{probe.name}:{probe.error or 'missing=' + ','.join(probe.missing)}"
        for probe in completions
        if not probe.ok
    )
    browser_capture_archive = _probe_browser_capture_archive(archive_root=archive_root)
    if not browser_capture_archive.ok:
        failures.append(f"browser-capture-archive:{browser_capture_archive.error or 'spooled_without_raw_rows'}")
    repo_head = _repo_head()
    return DeploymentSmokeReport(
        ok=not failures,
        path=path,
        repo_head=repo_head,
        daemon_base_url=daemon_base_url,
        receiver_base_url=receiver_base_url,
        commands=commands,
        routes=routes,
        completions=completions,
        browser_capture_archive=browser_capture_archive,
        diagnostics=_diagnose(commands, routes, repo_head, browser_capture_archive),
        failures=failures,
    )


def _print_human(report: DeploymentSmokeReport) -> None:
    print("Polylogue deployment smoke")
    print(f"PATH: {report.path}")
    print(f"repo HEAD: {report.repo_head or 'unknown'}")
    print(f"daemon: {report.daemon_base_url}")
    print(f"receiver: {report.receiver_base_url}")
    print(f"status: {'ok' if report.ok else 'failed'}")
    print("")
    print("Commands:")
    for probe in report.commands:
        marker = "ok" if probe.ok else "FAIL"
        detail = probe.stdout or probe.stderr or probe.error or ""
        print(f"  {marker} {probe.name} [{probe.path or 'not found'}] {detail}")
    print("")
    print("Routes:")
    for route_probe in report.routes:
        marker = "ok" if route_probe.ok else "FAIL"
        print(f"  {marker} {route_probe.url} status={route_probe.status} error={route_probe.error or ''}")
    print("")
    print("Completions:")
    for completion_probe in report.completions:
        marker = "ok" if completion_probe.ok else "FAIL"
        detail = ", ".join(completion_probe.candidates[:8])
        if completion_probe.missing:
            detail = f"missing={','.join(completion_probe.missing)} candidates={detail}"
        print(f"  {marker} {completion_probe.name}: {detail}")
    capture_probe = report.browser_capture_archive
    print("")
    print("Browser capture archive:")
    marker = "ok" if capture_probe.ok else "FAIL"
    print(
        f"  {marker} spooled={capture_probe.spooled_count} raw_rows={capture_probe.raw_rows} "
        f"latest={capture_probe.latest_spooled_path or ''} error={capture_probe.error or ''}"
    )
    if report.failures:
        print("")
        print("Failures:")
        for failure in report.failures:
            print(f"  - {failure}")
    next_actions = report.diagnostics.get("next_actions")
    if isinstance(next_actions, list) and next_actions:
        print("")
        print("Next actions:")
        for action in next_actions:
            print(f"  - {action}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    parser.add_argument("--path", default=SYSTEMWIDE_PATH, help="PATH used to resolve deployed binaries.")
    parser.add_argument("--daemon-url", default="http://127.0.0.1:8766", help="Daemon API base URL.")
    parser.add_argument("--receiver-url", default="http://127.0.0.1:8765", help="Browser-capture receiver base URL.")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path.home() / ".local" / "share" / "polylogue",
        help="Archive root used for local browser-capture/archive consistency probes.",
    )
    parser.add_argument("--timeout-s", type=float, default=5.0, help="Per-probe timeout in seconds.")
    args = parser.parse_args(argv)

    report = build_report(
        path=str(args.path),
        daemon_base_url=str(args.daemon_url),
        receiver_base_url=str(args.receiver_url),
        archive_root=Path(args.archive_root),
        timeout_s=float(args.timeout_s),
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

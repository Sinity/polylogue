"""The installed capture command must stay exec-cheap (polylogue-jv94g).

A harness fires PreToolUse and PostToolUse on every tool call, so this
command's cost is paid twice per tool call per concurrent agent. The wired
command used to be a console script that imported the polylogue package and
ran the free-threaded runtime gate: 473 ms of CPU per invocation against 39 ms
for the same work under ``python -I -S``.

The tests here pin the two properties that keep it cheap: the rendered command
runs the producer script directly under isolated/no-site flags, and the
producer's import graph stays inside cheap stdlib.
"""

from __future__ import annotations

import ast
import json
import shlex
import subprocess
from pathlib import Path

import pytest

from polylogue.hooks import (
    PRODUCER_SCRIPT_NAME,
    hook_command_available,
    installed_hook_commands,
    plan_hook_change,
    producer_script_path,
    resolve_events,
    settings_path,
)
from polylogue.sources.hook_producer import append_event, day_shard
from polylogue.sources.hooks import hook_carrier_dir, hook_carrier_provider_dir

# Every module the producer may import, module scope or nested. ``uuid``,
# ``re``, ``contextlib`` and ``tempfile`` are absent deliberately -- each
# costs more to import than it saves, and each has a cheaper equivalent
# already in use. ``tempfile`` left when the producer stopped publishing by
# temp-file-and-rename: a carrier line is one ``O_APPEND`` write, so nothing
# on this path needs it any more.
_ALLOWED_IMPORTS = frozenset({"__future__", "datetime", "json", "os", "pathlib", "sys", "types"})

_PAYLOAD = '{"session_id":"producer-session","permission_mode":"bypassPermissions","tool_name":"Bash"}'


def _carriers(root: Path, provider: str = "claude-code") -> list[Path]:
    return sorted(hook_carrier_provider_dir(provider, root).rglob("*.ndjson"))


def _carrier_records(carrier: Path) -> list[dict[str, object]]:
    """Decode a carrier the way acquisition does: whole newline-terminated lines."""
    payload = carrier.read_bytes()
    assert payload.endswith(b"\n"), payload
    return [json.loads(line) for line in payload.splitlines()]


@pytest.fixture
def isolated_hook_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))
    return home


def _install_session_start(harness: str = "claude-code") -> str:
    plan_hook_change(
        "install",
        harness,  # type: ignore[arg-type]
        resolve_events(harness, "SessionStart"),  # type: ignore[arg-type]
        dry_run=False,
    )
    commands = installed_hook_commands(harness)  # type: ignore[arg-type]
    assert len(commands) == 1, commands
    return commands[0]


def test_producer_imports_only_cheap_stdlib() -> None:
    """No import anywhere in the producer reaches outside the cheap stdlib set.

    Anti-vacuity: reintroducing ``from polylogue.runtime import
    require_free_threaded_runtime`` (the 183 ms native-extension probe this
    change removed from the capture path) or a convenience ``import pathlib``
    makes this fail. Dynamic package resolution is deliberately not an import
    statement: it goes through ``_import_optional``, which returns ``None``
    when the package is unreachable under ``-S``.
    """

    tree = ast.parse(producer_script_path().read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            imported.add(node.module.split(".")[0])

    assert imported <= _ALLOWED_IMPORTS, sorted(imported - _ALLOWED_IMPORTS)


def test_installed_command_records_an_event_without_importing_polylogue(isolated_hook_home: Path) -> None:
    """The wired command spools its event out of stdlib alone.

    Anti-vacuity: render the handler as the ``polylogue-hook`` console script
    again -- or drop ``-S`` from the rendered flags -- and the interpreter
    loads the polylogue package, which this assertion names directly.
    """

    command = _install_session_start()
    argv = shlex.split(command)
    assert argv[1:3] == ["-I", "-S"], argv
    assert Path(argv[3]).name == PRODUCER_SCRIPT_NAME, argv

    # ``-X importtime`` observes the same script under the same flags; it adds
    # no import of its own, so the module list it prints is what the wired
    # command really loads.
    result = subprocess.run(
        [argv[0], "-X", "importtime", *argv[1:]],
        input=_PAYLOAD,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    loaded = [
        name
        for line in result.stderr.splitlines()
        if line.startswith("import time:") and (name := line.rsplit("|", 1)[-1].strip()) != "imported package"
    ]
    assert "json" in loaded, loaded
    assert not [name for name in loaded if name.split(".")[0] == "polylogue"], loaded
    for expensive in ("uuid", "site", "polylogue"):
        assert expensive not in loaded, f"{expensive} back on the capture path: {loaded}"

    sidecar = Path(argv[argv.index("--sidecar-dir") + 1])
    (carrier,) = _carriers(sidecar)
    # One carrier per producer process per UTC day per harness: the provider
    # and the day are directory segments and the pid is the file name, so
    # acquisition never opens a carrier to learn which provider wrote it.
    assert carrier.relative_to(hook_carrier_dir(sidecar)).parts[:2] == ("claude-code", day_shard())
    assert carrier.stem.isdigit(), carrier.name
    (record,) = _carrier_records(carrier)
    assert record["event_type"] == "SessionStart"
    assert record["provider"] == "claude-code"
    assert record["session_id"] == "producer-session"


def test_installed_command_and_library_append_write_the_same_envelope(isolated_hook_home: Path, tmp_path: Path) -> None:
    """One implementation: the exec-cheap command and the in-process append
    agree on every envelope field materialization reads back."""

    command = _install_session_start()
    argv = shlex.split(command)
    result = subprocess.run(argv, input=_PAYLOAD, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr

    sidecar = Path(argv[argv.index("--sidecar-dir") + 1])
    (spooled,) = _carrier_records(_carriers(sidecar)[0])

    library_root = tmp_path / "library-spool"
    library_path = append_event(
        event_type="SessionStart",
        session_id="producer-session",
        provider="claude-code",
        timestamp=str(spooled["timestamp"]),
        payload=json.loads(_PAYLOAD),
        root=str(library_root),
        event_id=str(spooled["event_id"]),
    )
    (library_record,) = _carrier_records(Path(library_path))
    assert library_record == spooled


def test_console_script_commands_installed_earlier_are_still_owned(isolated_hook_home: Path) -> None:
    """Settings written before the exec-cheap command must stay removable.

    An operator who never re-runs ``hooks install`` keeps a
    ``polylogue-hook <event> ...`` handler; ownership detection has to
    recognize it or uninstall would strand it in their settings forever.
    """

    target = settings_path("claude-code")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "polylogue-hook SessionStart --provider claude-code "
                                    "--sidecar-dir /tmp/legacy-spool",
                                }
                            ]
                        }
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert installed_hook_commands("claude-code") != ()

    plan = plan_hook_change("uninstall", "claude-code", ("SessionStart",), dry_run=False)
    assert plan.changed_events == ("SessionStart",)
    assert installed_hook_commands("claude-code") == ()


def test_hook_command_available_reports_the_wired_commands_own_target(isolated_hook_home: Path) -> None:
    """Availability resolves what the wired command names, not this process's PATH.

    Anti-vacuity: go back to probing ``shutil.which("polylogue-hook")`` and a
    wired command pointing at a deleted producer script reports healthy, which
    is exactly the silent-capture-gap the daemon health check exists to catch.
    """

    _install_session_start()
    assert hook_command_available("claude-code") is True

    target = settings_path("claude-code")
    document = json.loads(target.read_text(encoding="utf-8"))
    handler = document["hooks"]["SessionStart"][0]["hooks"][0]
    argv = shlex.split(str(handler["command"]))
    argv[3] = str(Path(argv[3]).parent / "does_not_exist" / PRODUCER_SCRIPT_NAME)
    handler["command"] = " ".join(shlex.quote(part) for part in argv)
    target.write_text(json.dumps(document) + "\n", encoding="utf-8")

    assert hook_command_available("claude-code") is False


def test_producer_refuses_a_payload_that_duplicates_transcript_content(isolated_hook_home: Path) -> None:
    """The bundled producer enforces the same no-duplicated-transcript rule as
    the standalone adapters, and reports it as an exit status rather than a
    traceback the harness would surface as hook noise."""

    command = _install_session_start()
    argv = shlex.split(command)
    result = subprocess.run(
        argv,
        input=json.dumps({"session_id": "producer-session", "text": "x" * 5000}),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "duplicated transcript" in result.stderr
    assert "Traceback" not in result.stderr
    sidecar = Path(argv[argv.index("--sidecar-dir") + 1])
    assert list(hook_carrier_dir(sidecar).rglob("*.ndjson")) == []


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"permissionMode": "auto", "toolName": "Bash"}, id="camel-permission-mode"),
        pytest.param({"permission_mode": "auto", "tool_name": "Bash"}, id="snake-permission-mode"),
    ],
)
def test_provider_detection_reads_both_payload_generations(
    payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bd polylogue-cp806: a camelCase Claude Code payload must not fall through
    to the Codex branch or to "could not detect provider".

    Anti-vacuity: keyed on ``permission_mode`` alone, the camel case resolves
    ``None`` and the manually-invoked command refuses the event outright.
    """
    from polylogue.sources import hook_producer

    # Detection from payload shape only: an operator-forced provider would
    # answer before the shape is ever inspected.
    monkeypatch.setattr(hook_producer, "_configured_provider", lambda: None)

    assert hook_producer.detect_provider(payload) == "claude-code"


def test_nested_transcript_payloads_are_rejected() -> None:
    """polylogue-54a31 (3): a transcript hidden in a list or dict is still a transcript."""
    import pytest

    from polylogue.sources.hook_producer import (
        MAX_TRANSCRIPT_LIKE_FIELD_CHARS,
        HookSpoolRecordError,
        reject_duplicated_transcript,
    )

    body = "x" * (MAX_TRANSCRIPT_LIKE_FIELD_CHARS + 1)
    with pytest.raises(HookSpoolRecordError):
        reject_duplicated_transcript({"messages": [body]})
    with pytest.raises(HookSpoolRecordError):
        reject_duplicated_transcript({"content": {"parts": [{"text": body}]}})
    # Two small strings that only sum past the cap are still one copy of a transcript.
    half = "y" * (MAX_TRANSCRIPT_LIKE_FIELD_CHARS // 2 + 1)
    with pytest.raises(HookSpoolRecordError):
        reject_duplicated_transcript({"transcript": [half, half]})
    reject_duplicated_transcript({"messages": ["short"], "tool_name": "Bash"})

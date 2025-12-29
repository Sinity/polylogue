from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.cli.watch import _run_watch_sessions, run_watch_cli
from polylogue.commands import CommandEnv
from polylogue.local_sync import LocalSyncResult


class DummyConsole:
    def print(self, *_args, **_kwargs):
        # Capture output for assertions
        line = " ".join(str(arg) for arg in _args)
        self.lines.append(line)

    def __init__(self):
        self.lines: list[str] = []


class DummyUI:
    plain = True
    console = DummyConsole()

    def summary(self, *_args, **_kwargs):
        pass

    def banner(self, *_args, **_kwargs):
        pass


class RecordingProvider:
    def __init__(self, base_dir: Path, watch_suffixes: tuple[str, ...]):
        self.name = "recording"
        self.title = "Recording Provider"
        self.default_base = base_dir
        self.default_output = base_dir / "out"
        self.list_sessions = lambda _base: []
        self.watch_banner = "watching"
        self.watch_log_title = "log"
        self.watch_suffixes = watch_suffixes
        self.supports_watch = True
        self.calls: list[list[Path] | None] = []
        self.call_kwargs: list[dict] = []

    def sync_fn(self, *, sessions=None, **kwargs):  # noqa: ANN001
        self.calls.append(list(sessions) if sessions else None)
        self.call_kwargs.append(kwargs)
        return LocalSyncResult(
            written=[],
            skipped=0,
            pruned=0,
            output_dir=kwargs["output_dir"],
        )


def _watch_args(base_dir: Path, out_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        provider="recording",
        base_dir=str(base_dir),
        out=str(out_dir),
        collapse_threshold=None,
        html_mode="off",
        debounce=0,
        stall_seconds=60.0,
        fail_on_stall=False,
        once=False,
        force=False,
        prune=False,
        diff=False,
    )


def test_watch_requires_existing_base_dir(tmp_path):
    missing = tmp_path / "missing"
    provider = RecordingProvider(missing, (".jsonl",))
    args = _watch_args(missing, provider.default_output)
    env = CommandEnv(ui=DummyUI())

    with pytest.raises(SystemExit):
        _run_watch_sessions(args, env, provider)

    assert not missing.exists()


def test_watch_once_creates_missing_base_dir(tmp_path):
    missing = tmp_path / "missing"
    provider = RecordingProvider(missing, (".jsonl",))
    args = _watch_args(missing, provider.default_output)
    args.once = True
    env = CommandEnv(ui=DummyUI())

    _run_watch_sessions(args, env, provider)

    assert missing.exists()
    # once mode runs the initial sync only
    assert len(provider.calls) == 1


def test_watch_provider_can_opt_in_to_base_dir_creation(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    provider = RecordingProvider(missing, (".jsonl",))
    provider.create_base_dir = True
    args = _watch_args(missing, provider.default_output)
    env = CommandEnv(ui=DummyUI())

    def no_events(_base, recursive=True):  # noqa: ARG001
        return iter(())

    monkeypatch.setattr("polylogue.cli.watch._watch_directory", no_events)

    _run_watch_sessions(args, env, provider)

    assert missing.exists()
    assert len(provider.calls) == 1


def test_watch_filters_suffixes(monkeypatch, tmp_path):
    provider = RecordingProvider(tmp_path, (".jsonl",))
    args = _watch_args(tmp_path, provider.default_output)
    env = CommandEnv(ui=DummyUI())

    def fake_watch_directory(_base, recursive=True):  # noqa: ARG001
        yield {(1, str(tmp_path / "ignored.tmp"))}
        yield {(1, str(tmp_path / "session.jsonl"))}

    ticks = iter([0.0, 1.0, 2.0])

    monkeypatch.setattr("polylogue.cli.watch._watch_directory", fake_watch_directory)
    monkeypatch.setattr("polylogue.cli.watch.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("polylogue.cli.watch.time.sleep", lambda _t: None)

    _run_watch_sessions(args, env, provider)

    assert len(provider.calls) == 2  # initial sync + matching change
    assert provider.calls[0] is None
    assert provider.calls[1] == [tmp_path / "session.jsonl"]


def test_watch_accepts_bundle_extensions(monkeypatch, tmp_path):
    provider = RecordingProvider(tmp_path, (".zip", ".json"))
    args = _watch_args(tmp_path, provider.default_output)
    env = CommandEnv(ui=DummyUI())

    def fake_watch_directory(_base, recursive=True):  # noqa: ARG001
        yield {(1, str(tmp_path / "export.zip"))}

    ticks = iter([0.0, 1.0])

    monkeypatch.setattr("polylogue.cli.watch._watch_directory", fake_watch_directory)
    monkeypatch.setattr("polylogue.cli.watch.time.monotonic", lambda: next(ticks))

    _run_watch_sessions(args, env, provider)

    assert len(provider.calls) == 2
    assert provider.calls[1] == [tmp_path / "export.zip"]


def test_run_watch_cli_surfaces_watch_errors(monkeypatch, tmp_path):
    args = _watch_args(tmp_path, tmp_path / "out")
    env = CommandEnv(ui=DummyUI())
    provider = RecordingProvider(tmp_path, (".jsonl",))

    def failing_watch(*_args, **_kwargs):
        raise RuntimeError("watchfiles unavailable")

    monkeypatch.setattr("polylogue.cli.watch.get_local_provider", lambda _name: provider)
    monkeypatch.setattr("polylogue.cli.watch._watch_directory", failing_watch)

    with pytest.raises(RuntimeError, match="watchfiles unavailable"):
        run_watch_cli(args, env)


def test_watch_passes_flags(monkeypatch, tmp_path):
    provider = RecordingProvider(tmp_path, (".jsonl",))
    provider.supports_diff = True  # explicitly allow diff
    args = _watch_args(tmp_path, provider.default_output)
    args.once = True
    args.force = True
    args.prune = True
    args.diff = True
    env = CommandEnv(ui=DummyUI())

    _run_watch_sessions(args, env, provider)

    assert len(provider.calls) == 1
    # The provider records the sessions list; ensure it saw the initial call.
    assert provider.calls[0] is None
    assert provider.call_kwargs[0]["force"] is True
    assert provider.call_kwargs[0]["prune"] is True
    assert provider.call_kwargs[0]["diff"] is True


def test_watch_logs_skipped_events(monkeypatch, tmp_path, capsys):
    provider = RecordingProvider(tmp_path, (".jsonl",))
    args = _watch_args(tmp_path, provider.default_output)
    env = CommandEnv(ui=DummyUI())

    def fake_watch_directory(_base, recursive=True):  # noqa: ARG001
        yield {(1, str(tmp_path / "early.jsonl"))}
        yield {(1, str(tmp_path / "debounced.jsonl"))}
        yield {(1, str(tmp_path / "late.jsonl"))}

    ticks = iter([0.0, 0.5, 0.6, 2.0])

    monkeypatch.setattr("polylogue.cli.watch._watch_directory", fake_watch_directory)
    monkeypatch.setattr("polylogue.cli.watch.time.monotonic", lambda: next(ticks))

    _run_watch_sessions(args, env, provider)

    joined = "\n".join(env.ui.console.lines)
    assert "Skipped" in joined


def test_watch_fail_on_stall_triggers_during_debounce(monkeypatch, tmp_path):
    provider = RecordingProvider(tmp_path, (".jsonl",))
    args = _watch_args(tmp_path, provider.default_output)
    args.debounce = 10.0
    args.stall_seconds = 1.0
    args.fail_on_stall = True
    env = CommandEnv(ui=DummyUI())

    def fake_watch_directory(_base, recursive=True):  # noqa: ARG001
        yield {(1, str(tmp_path / "a.jsonl"))}
        yield {(1, str(tmp_path / "b.jsonl"))}
        yield {(1, str(tmp_path / "c.jsonl"))}

    ticks = iter([0.0, 0.2, 0.4, 2.0])

    monkeypatch.setattr("polylogue.cli.watch._watch_directory", fake_watch_directory)
    monkeypatch.setattr("polylogue.cli.watch.time.monotonic", lambda: next(ticks))

    with pytest.raises(SystemExit) as excinfo:
        _run_watch_sessions(args, env, provider)

    assert excinfo.value.code == 2
    joined = "\n".join(env.ui.console.lines)
    assert "No sync progress" in joined

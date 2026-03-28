from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from polylogue.cli.watch import _run_watch_sessions, run_watch_cli
from polylogue.commands import CommandEnv
from polylogue.local_sync import LocalSyncResult


class DummyConsole:
    def print(self, *_args, **_kwargs):
        pass


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

    def sync_fn(self, *, sessions=None, **kwargs):  # noqa: ANN001
        self.calls.append(list(sessions) if sessions else None)
        return LocalSyncResult(
            written=[],
            skipped=0,
            pruned=0,
            output_dir=kwargs["output_dir"],
        )


def _watch_args(base_dir: Path, out_dir: Path) -> Namespace:
    return Namespace(
        provider="recording",
        base_dir=str(base_dir),
        out=str(out_dir),
        collapse_threshold=None,
        html_mode="off",
        debounce=0,
        once=False,
    )


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

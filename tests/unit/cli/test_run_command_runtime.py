# mypy: disable-error-code="no-untyped-def,call-arg,arg-type,attr-defined"

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import click
import pytest

from polylogue.cli.commands import run as run_command_module
from polylogue.storage.run_state import RunCounts, RunDrift, RunResult


def _run_result() -> RunResult:
    return RunResult(
        run_id="run-1",
        counts=RunCounts(conversations=1),
        drift=RunDrift(),
        indexed=False,
        index_error=None,
        duration_ms=12,
        render_failures=[],
    )


def _env(*, plain: bool = True, confirm: bool = True) -> object:
    ui = SimpleNamespace(
        plain=plain,
        console=SimpleNamespace(print=MagicMock()),
        confirm=MagicMock(return_value=confirm),
    )
    return SimpleNamespace(
        ui=ui,
        config=SimpleNamespace(sources=[], render_root=Path("/render")),
        backend=SimpleNamespace(),
        repository=SimpleNamespace(reset_parse_status=lambda source_names=None: object()),
    )


def _ctx(env: object) -> click.Context:
    ctx = click.Context(run_command_module.run_command)
    ctx.obj = env
    return ctx


def _raw_callback() -> Callable[..., object]:
    callback = getattr(run_command_module._run_result_callback, "__wrapped__", run_command_module._run_result_callback)
    assert callback is not None
    return cast(Callable[..., object], callback)


def _run_stage_request(name: str, *, render_format: str | None = None, site_options: dict[str, object] | None = None):
    return run_command_module._make_stage_request(name, render_format=render_format, site_options=site_options)


def _embed_request(
    *,
    conversation: str | None = None,
    model: str = "voyage-4",
    rebuild: bool = False,
    stats: bool = False,
    json_output: bool = False,
    limit: int | None = None,
) -> run_command_module.RunStageRequest:
    return run_command_module.RunStageRequest(
        name="embed",
        stage_sequence=run_command_module.expand_requested_stage("embed"),
        embed_options=run_command_module.EmbedOptions(
            conversation=conversation,
            model=model,
            rebuild=rebuild,
            stats=stats,
            json_output=json_output,
            limit=limit,
        ),
    )


def test_run_stage_request_helpers_cover_resolution_and_stage_commands() -> None:
    acquire = _run_stage_request("acquire")
    render = _run_stage_request("render", render_format="markdown")
    site = _run_stage_request(
        "site",
        site_options={
            "title": "Docs",
            "search": False,
            "search_provider": "lunr",
            "dashboard": False,
            "output": Path("/tmp/site"),
        },
    )
    embed = _embed_request(conversation="conv-1", model="voyage-4-large", rebuild=True, limit=5)

    assert acquire.name == "acquire"
    assert _run_stage_request("schema").name == "schema"
    assert _run_stage_request("parse").name == "parse"
    assert _run_stage_request("materialize").name == "materialize"
    assert _run_stage_request("index").name == "index"
    assert _run_stage_request("reprocess").name == "reprocess"
    assert _run_stage_request("all").name == "all"
    assert render.render_format == "markdown"
    assert site.site_options == {
        "title": "Docs",
        "search": False,
        "search_provider": "lunr",
        "dashboard": False,
        "output": Path("/tmp/site"),
    }
    assert embed.embed_options is not None
    assert embed.embed_options.model == "voyage-4-large"
    assert run_command_module._flatten_stage_requests([acquire, render]) == ("acquire", "render")
    assert run_command_module._resolve_canonical_stage([]) == "all"
    assert run_command_module._resolve_canonical_stage([run_command_module._make_stage_request("all")]) == "all"
    assert run_command_module._display_stage_label([acquire, render], "all") == "acquire -> render"
    assert run_command_module._resolve_render_format([acquire]) == "html"
    assert run_command_module._resolve_embed_options([acquire]) is None
    assert run_command_module._resolve_site_options([acquire]) is None


def test_run_stage_request_helper_conflicts_fail_fast() -> None:
    with pytest.raises(SystemExit, match="run: Conflicting render formats requested: html, markdown"):
        run_command_module._resolve_render_format(
            [
                _run_stage_request("render", render_format="html"),
                _run_stage_request("render", render_format="markdown"),
            ]
        )

    with pytest.raises(SystemExit, match="run: Multiple embed stage requests with different options"):
        run_command_module._resolve_embed_options(
            [
                _embed_request(stats=True),
                _embed_request(stats=True, json_output=True),
            ]
        )

    with pytest.raises(SystemExit, match="run: Multiple site stage requests with different options"):
        run_command_module._resolve_site_options(
            [
                _run_stage_request(
                    "site",
                    site_options={"title": "Docs", "search": True, "search_provider": "pagefind", "dashboard": True},
                ),
                _run_stage_request(
                    "site",
                    site_options={
                        "title": "Docs",
                        "search": True,
                        "search_provider": "pagefind",
                        "dashboard": True,
                        "output": Path("/tmp/site"),
                    },
                ),
            ]
        )


def test_run_result_callback_rejects_watch_only_flags_without_watch_mode() -> None:
    with pytest.raises(SystemExit, match="run: --notify, --exec, and --webhook require --watch mode"):
        _raw_callback()(
            _ctx(_env()),
            [],
            False,
            (),
            (),
            False,
            True,
            None,
            None,
            False,
        )


def test_run_result_callback_surfaces_stage_normalization_errors() -> None:
    with patch("polylogue.cli.commands.run.normalize_stage_sequence", side_effect=ValueError("bad stage")):
        with pytest.raises(SystemExit, match="run: bad stage"):
            _raw_callback()(
                _ctx(_env()),
                [run_command_module._make_stage_request("parse")],
                False,
                (),
                (),
                False,
                False,
                None,
                None,
                False,
            )


def test_run_result_callback_embed_only_returns_before_source_resolution() -> None:
    ctx = _ctx(_env())
    with patch("polylogue.cli.commands.run._run_embed_standalone") as run_embed:
        with patch("polylogue.cli.commands.run.resolve_sources") as resolve_sources:
            _raw_callback()(
                ctx,
                [_embed_request(stats=True)],
                False,
                (),
                (),
                False,
                False,
                None,
                None,
                False,
            )

    run_embed.assert_called_once()
    resolve_sources.assert_not_called()


def test_run_result_callback_embed_then_parse_strips_embed_from_stage_sequence() -> None:
    env = _env(plain=True)
    with patch("polylogue.cli.commands.run._run_embed_standalone") as run_embed:
        with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
            with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
                with patch("polylogue.cli.commands.run._run_sync_once", return_value=_run_result()) as run_sync_once:
                    with patch("polylogue.cli.commands.run._display_result") as display_result:
                        _raw_callback()(
                            _ctx(env),
                            [
                                _embed_request(stats=True),
                                _run_stage_request("parse"),
                            ],
                            False,
                            (),
                            (),
                            False,
                            False,
                            None,
                            None,
                            False,
                        )

    run_embed.assert_called_once()
    assert run_sync_once.call_args.args[3] == ("parse",)
    display_result.assert_called_once()


def test_run_result_callback_forwards_reparse_as_force_write() -> None:
    env = _env(plain=True)
    with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
        with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
            with patch("polylogue.cli.commands.run.run_coroutine_sync", return_value=0):
                with patch("polylogue.cli.commands.run._run_sync_once", return_value=_run_result()) as run_sync_once:
                    with patch("polylogue.cli.commands.run._display_result"):
                        _raw_callback()(
                            _ctx(env),
                            [_run_stage_request("parse")],
                            False,
                            (),
                            (),
                            False,
                            False,
                            None,
                            None,
                            True,
                        )

    assert run_sync_once.call_args.kwargs["force_write"] is True


def test_run_result_callback_watch_mode_builds_observers_and_executes_sync_once() -> None:
    env = _env(plain=True)
    runner_state: dict[str, object] = {}

    class FakeWatchRunner:
        def __init__(self, *, sync_fn: object, observer: object, interval: int) -> None:
            runner_state["observer"] = observer
            runner_state["interval"] = interval
            runner_state["sync_fn"] = sync_fn

        def run(self) -> None:
            sync_fn = runner_state["sync_fn"]
            assert callable(sync_fn)
            sync_fn()

    with patch("polylogue.cli.commands.run.resolve_sources", return_value=["drive"]):
        with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=["drive"]):
            with patch("polylogue.cli.commands.run._WatchDisplayObserver", return_value="display"):
                with patch("polylogue.cli.commands.run._WatchStatusObserver", return_value="status"):
                    with patch("polylogue.cli.commands.run.NotificationObserver", return_value="notify"):
                        with patch("polylogue.cli.commands.run.ExecObserver", return_value="exec"):
                            with patch("polylogue.cli.commands.run.WebhookObserver", return_value="webhook"):
                                with patch(
                                    "polylogue.cli.commands.run.CompositeObserver",
                                    side_effect=lambda observers: tuple(observers),
                                ):
                                    with patch(
                                        "polylogue.cli.commands.run._run_with_progress", return_value=_run_result()
                                    ) as run_with_progress:
                                        with patch("polylogue.pipeline.watch.WatchRunner", FakeWatchRunner):
                                            _raw_callback()(
                                                _ctx(env),
                                                [_run_stage_request("render", render_format="html")],
                                                False,
                                                (),
                                                (),
                                                True,
                                                True,
                                                "echo hi",
                                                "https://example.test",
                                                False,
                                            )

    assert runner_state["interval"] == 60
    assert runner_state["observer"] == ("display", "status", "notify", "exec", "webhook")
    run_with_progress.assert_called_once()
    printed = [call.args[0] for call in env.ui.console.print.call_args_list]
    assert "Watch mode: syncing every 60 seconds. Press Ctrl+C to stop." in printed
    assert printed[-1] == "\nWatch mode stopped."


def test_run_result_callback_can_cancel_nonwatch_execution_before_running() -> None:
    env = _env(plain=False, confirm=False)
    with patch("polylogue.cli.commands.run.resolve_sources", return_value=None):
        with patch("polylogue.cli.commands.run.maybe_prompt_sources", return_value=None):
            with patch("polylogue.cli.commands.run._run_sync_once") as run_sync_once:
                _raw_callback()(
                    _ctx(env),
                    [_run_stage_request("parse")],
                    False,
                    (),
                    (),
                    False,
                    False,
                    None,
                    None,
                    False,
                )

    run_sync_once.assert_not_called()
    env.ui.console.print.assert_called_with("Cancelled.")


def test_run_embed_standalone_covers_error_stats_single_and_batch_paths() -> None:
    env = _env()
    env.repository = "repo"

    with pytest.raises(click.Abort):
        run_command_module._run_embed_standalone(
            env,
            run_command_module.EmbedOptions(json_output=True, stats=False),
        )

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(click.Abort):
            run_command_module._run_embed_standalone(env, run_command_module.EmbedOptions())

    with patch("polylogue.cli.shared.embed_stats.show_embedding_stats") as show_stats:
        run_command_module._run_embed_standalone(
            env,
            run_command_module.EmbedOptions(stats=True, json_output=True),
        )
    show_stats.assert_called_once_with(env, json_output=True)

    with patch.dict("os.environ", {"VOYAGE_API_KEY": "key"}, clear=True):
        with patch("polylogue.storage.search_providers.create_vector_provider", return_value=None):
            with pytest.raises(click.Abort):
                run_command_module._run_embed_standalone(env, run_command_module.EmbedOptions())

    provider = SimpleNamespace(model="voyage-4")
    with patch.dict("os.environ", {"VOYAGE_API_KEY": "key"}, clear=True):
        with patch("polylogue.storage.search_providers.create_vector_provider", return_value=provider):
            with patch("polylogue.cli.shared.embed_runtime.embed_single") as embed_single:
                run_command_module._run_embed_standalone(
                    env,
                    run_command_module.EmbedOptions(conversation="conv-1", model="voyage-4-large"),
                )
    embed_single.assert_called_once_with(env, "repo", provider, "conv-1")
    assert provider.model == "voyage-4-large"

    provider = SimpleNamespace(model="voyage-4")
    with patch.dict("os.environ", {"VOYAGE_API_KEY": "key"}, clear=True):
        with patch("polylogue.storage.search_providers.create_vector_provider", return_value=provider):
            with patch("polylogue.cli.shared.embed_runtime.embed_batch") as embed_batch:
                run_command_module._run_embed_standalone(
                    env,
                    run_command_module.EmbedOptions(rebuild=True, limit=9),
                )
    embed_batch.assert_called_once_with(env, "repo", provider, rebuild=True, limit=9)

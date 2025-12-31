from __future__ import annotations

import json

from polylogue.config import Source, default_config, write_config
from polylogue.run import plan_sources, run_sources


def test_plan_and_run_sources(workspace_env, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv1",
        "title": "Test",
        "messages": [
            {"id": "m1", "role": "user", "content": "hello"},
            {"id": "m2", "role": "assistant", "content": "world"},
        ],
    }
    source_file = inbox / "conversation.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    config = default_config()
    config.sources = [Source(name="codex", path=source_file)]
    write_config(config)

    plan = plan_sources(config)
    assert plan.counts["conversations"] == 1

    result = run_sources(config=config, stage="all", plan=plan)
    assert result.counts["conversations"] == 1
    run_dir = workspace_env["archive_root"] / "runs"
    assert any(run_dir.iterdir())


def test_run_sources_filtered(workspace_env, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload_a = {"id": "conv-a", "messages": [{"id": "m1", "role": "user", "content": "hello"}]}
    payload_b = {"id": "conv-b", "messages": [{"id": "m1", "role": "user", "content": "world"}]}
    source_a = inbox / "a.json"
    source_b = inbox / "b.json"
    source_a.write_text(json.dumps(payload_a), encoding="utf-8")
    source_b.write_text(json.dumps(payload_b), encoding="utf-8")

    config = default_config()
    config.sources = [
        Source(name="source-a", path=source_a),
        Source(name="source-b", path=source_b),
    ]
    write_config(config)

    result = run_sources(config=config, stage="ingest", source_names=["source-a"])
    assert result.counts["conversations"] == 1


def test_render_filtered_by_source_meta(workspace_env, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv-chatgpt",
        "mapping": {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "user"},
                    "content": {"parts": ["hello"]},
                    "create_time": 1,
                },
            }
        },
    }
    source_file = inbox / "conversation.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    config = default_config()
    config.sources = [Source(name="inbox", path=source_file)]
    write_config(config)

    run_sources(config=config, stage="ingest", source_names=["inbox"])
    result = run_sources(config=config, stage="render", source_names=["inbox"])
    assert result.counts["conversations"] == 0
    render_root = workspace_env["archive_root"] / "render"
    assert any(render_root.rglob("conversation.md"))


def test_run_all_skips_render_when_unchanged(workspace_env, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv1",
        "messages": [
            {"id": "m1", "role": "user", "content": "hello"},
            {"id": "m2", "role": "assistant", "content": "world"},
        ],
    }
    source_file = inbox / "conversation.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    config = default_config()
    config.sources = [Source(name="inbox", path=source_file)]
    write_config(config)

    run_sources(config=config, stage="all")

    render_root = workspace_env["archive_root"] / "render"
    convo_path = next(render_root.rglob("conversation.md"))
    first_mtime = convo_path.stat().st_mtime

    run_sources(config=config, stage="all")
    second_mtime = convo_path.stat().st_mtime
    assert first_mtime == second_mtime


def test_run_rerenders_when_content_changes(workspace_env, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv1",
        "messages": [
            {"id": "m1", "role": "user", "content": "hello"},
        ],
    }
    source_file = inbox / "conversation.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    config = default_config()
    config.sources = [Source(name="inbox", path=source_file)]
    write_config(config)

    run_sources(config=config, stage="all")

    render_root = workspace_env["archive_root"] / "render"
    convo_path = next(render_root.rglob("conversation.md"))
    first_mtime = convo_path.stat().st_mtime

    payload["messages"][0]["content"] = "hello world"
    source_file.write_text(json.dumps(payload), encoding="utf-8")
    run_sources(config=config, stage="all")

    second_mtime = convo_path.stat().st_mtime
    assert second_mtime > first_mtime


def test_incremental_index_updates(workspace_env, tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload_a = {
        "id": "conv-a",
        "messages": [
            {"id": "m1", "role": "user", "content": "alpha"},
        ],
    }
    payload_b = {
        "id": "conv-b",
        "messages": [
            {"id": "m1", "role": "user", "content": "beta"},
        ],
    }
    source_a = inbox / "a.json"
    source_b = inbox / "b.json"
    source_a.write_text(json.dumps(payload_a), encoding="utf-8")
    source_b.write_text(json.dumps(payload_b), encoding="utf-8")

    config = default_config()
    config.sources = [Source(name="inbox", path=inbox)]
    write_config(config)

    run_sources(config=config, stage="all")


def test_index_failure_is_nonfatal(workspace_env, monkeypatch):
    config = default_config()
    write_config(config)

    import polylogue.run as run_mod

    def boom():
        raise RuntimeError("index failed")

    monkeypatch.setattr(run_mod, "rebuild_index", boom)
    result = run_sources(config=config, stage="index")
    assert result.indexed is False
    assert result.index_error is not None
    assert "index failed" in result.index_error


def test_run_writes_unique_report_files(workspace_env, tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv1",
        "messages": [
            {"id": "m1", "role": "user", "content": "hello"},
        ],
    }
    source_file = inbox / "conversation.json"
    source_file.write_text(json.dumps(payload), encoding="utf-8")

    config = default_config()
    config.sources = [Source(name="inbox", path=source_file)]
    write_config(config)

    import polylogue.run as run_mod

    fixed_time = 1_700_000_000
    monkeypatch.setattr(run_mod.time, "time", lambda: fixed_time)
    monkeypatch.setattr(run_mod.time, "perf_counter", lambda: 0.0)

    run_sources(config=config, stage="all")
    run_sources(config=config, stage="all")

    run_dir = workspace_env["archive_root"] / "runs"
    runs = list(run_dir.glob(f"run-{fixed_time}-*.json"))
    assert len(runs) == 2

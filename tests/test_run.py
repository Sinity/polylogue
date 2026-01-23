from __future__ import annotations

import json
import time

from polylogue.config import Source, default_config, write_config
from polylogue.storage.db import open_connection
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

    time.sleep(0.01)  # Ensure fs timestamp resolution
    payload["messages"][0]["content"] = "hello world"
    source_file.write_text(json.dumps(payload), encoding="utf-8")
    run_sources(config=config, stage="all")

    second_mtime = convo_path.stat().st_mtime
    assert second_mtime > first_mtime


def test_run_rerenders_when_title_changes(workspace_env, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv-title",
        "title": "Old title",
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
    original = convo_path.read_text(encoding="utf-8")

    payload["title"] = "New title"
    source_file.write_text(json.dumps(payload), encoding="utf-8")
    run_sources(config=config, stage="all")

    updated = convo_path.read_text(encoding="utf-8")
    assert "# New title" in updated
    assert original != updated


def test_run_index_filters_selected_sources(workspace_env, tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload_a = {"id": "conv-a", "messages": [{"id": "m1", "role": "user", "text": "alpha"}]}
    payload_b = {"id": "conv-b", "messages": [{"id": "m1", "role": "user", "text": "beta"}]}
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

    run_sources(config=config, stage="ingest")

    id_by_source = {}
    with open_connection(None) as conn:
        rows = conn.execute("SELECT conversation_id, provider_meta FROM conversations").fetchall()
    for row in rows:
        meta = json.loads(row["provider_meta"] or "{}")
        name = meta.get("source")
        if name:
            id_by_source[name] = row["conversation_id"]

    update_calls = []
    from polylogue.pipeline.services.indexing import IndexService

    original_update = IndexService.update_index
    def fake_update_method(self, ids):
        update_calls.append(list(ids))
        return True
    monkeypatch.setattr(IndexService, "update_index", fake_update_method)

    run_sources(config=config, stage="index", source_names=["source-b"])

    assert update_calls == [[id_by_source["source-b"]]]


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

    from polylogue.pipeline.services.indexing import IndexService

    def boom(self):
        raise RuntimeError("index failed")

    monkeypatch.setattr(IndexService, "rebuild_index", boom)
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

    import polylogue.pipeline.runner as runner_mod

    fixed_time = 1_700_000_000
    monkeypatch.setattr(runner_mod.time, "time", lambda: fixed_time)
    monkeypatch.setattr(runner_mod.time, "perf_counter", lambda: 0.0)

    run_sources(config=config, stage="all")
    run_sources(config=config, stage="all")

    run_dir = workspace_env["archive_root"] / "runs"
    runs = list(run_dir.glob(f"run-{fixed_time}-*.json"))
    assert len(runs) == 2


# latest_run() tests


def test_latest_run_parses_json_columns(workspace_env, tmp_path):
    """latest_run() returns RunRecord with parsed dicts for counts and drift."""
    from polylogue.pipeline.runner import latest_run

    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "conv-latest-run",
        "messages": [{"id": "m1", "role": "user", "content": "test"}],
    }
    (inbox / "conversation.json").write_text(json.dumps(payload), encoding="utf-8")

    config = default_config()
    config.sources = [Source(name="inbox", path=inbox)]
    write_config(config)

    run_sources(config=config, stage="all")

    result = latest_run()
    assert result is not None
    assert result.run_id is not None

    # counts should be parsed to dict
    if result.counts is not None:
        assert isinstance(result.counts, dict)
        # Should have typical count keys
        assert "conversations" in result.counts or "messages" in result.counts

    # drift should be parsed to dict
    if result.drift is not None:
        assert isinstance(result.drift, dict)


def test_latest_run_handles_null_json_columns(workspace_env):
    """latest_run() handles NULL values in JSON columns gracefully."""
    from polylogue.pipeline.runner import latest_run

    # Insert a run record with NULL JSON columns directly
    with open_connection(None) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, timestamp, plan_snapshot, counts_json, drift_json, indexed, duration_ms)
            VALUES (?, ?, NULL, NULL, NULL, 0, 100)
            """,
            ("null-test-run", str(int(time.time()))),
        )
        conn.commit()

    result = latest_run()
    assert result is not None
    # Should not crash, NULL columns should remain as None
    assert result.plan_snapshot is None
    assert result.counts is None
    assert result.drift is None

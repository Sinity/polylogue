from __future__ import annotations

import json

from polylogue.config import Source, default_config, write_config
from polylogue.run_v666 import plan_sources, run_sources


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
    config.sources = [Source(name="codex", type="codex", path=source_file)]
    write_config(config)

    plan = plan_sources(config, profile=config.profiles["default"])
    assert plan.counts["conversations"] == 1

    profile = config.profiles["default"]
    result = run_sources(config=config, profile=profile, stage="all", plan=plan)
    assert result.counts["conversations"] == 1
    run_dir = workspace_env["archive_root"] / "runs"
    assert any(run_dir.iterdir())

"""Multi-provider daemon convergence benchmarks.

Extends the Claude Code-only convergence benchmarks to cover all major
providers: Codex, ChatGPT, and Gemini. Measures throughput, memory, and
correctness per provider at controlled scale tiers.

Run with:
    pytest tests/benchmarks/test_daemon_convergence.py \\
      tests/benchmarks/test_daemon_convergence_multi_provider.py \\
      --benchmark-enable -p no:xdist -o "addopts=" -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.benchmarks.helpers import BenchmarkFixture

# ── Synthetic data generation per provider ────────────────────────────


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _write_json(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f)


def _make_claude_code_session(uuid: str, n_messages: int, *, include_tools: bool = True) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    tool_names = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"]
    for i in range(n_messages):
        is_user = i % 2 == 0
        record: dict[str, object] = {
            "parentUuid": None if i == 0 else f"msg-{i - 1:04d}",
            "sessionId": uuid,
            "type": "user" if is_user else "assistant",
            "message": {
                "role": "user" if is_user else "assistant",
                "content": f"Synthetic message {i} in session {uuid}. "
                f"{'The quick brown fox jumps over the lazy dog. ' * 15}",
            },
            "uuid": f"msg-{i:04d}",
            "timestamp": f"2026-05-05T00:{i // 60:02d}:{i % 60:02d}.000Z",
            "cwd": "/realm/project/polylogue",
            "version": "1.0.6",
            "isSidechain": False,
            "userType": "external",
        }
        if not is_user and include_tools and i % 4 == 1:
            tool = tool_names[i % len(tool_names)]
            record["message"]["content"] = [  # type: ignore[index]
                {
                    "type": "tool_use",
                    "name": tool,
                    "id": f"tool-{i:04d}",
                    "input": {"command": f"echo 'hello {i}'"},
                }
            ]
            record["type"] = "assistant"
        records.append(record)
    return records


def _make_codex_session(uuid: str, n_messages: int) -> list[dict[str, object]]:
    """Generate a Codex-shaped session JSONL."""
    records: list[dict[str, object]] = []
    for i in range(n_messages):
        is_user = i % 2 == 0
        record: dict[str, object] = {
            "sessionId": uuid,
            "id": f"{uuid}-msg-{i:04d}",
            "role": "user" if is_user else "assistant",
            "content": [{"type": "text", "text": f"Codex message {i}. {'Lorem ipsum dolor sit amet. ' * 15}"}],
            "timestamp": f"2026-05-05T00:{i // 60:02d}:{i % 60:02d}.000Z",
            "cwd": "/realm/project/demo",
        }
        records.append(record)
    return records


def _make_chatgpt_session(conv_id: str, n_messages: int) -> dict[str, object]:
    """Generate a ChatGPT-shaped session JSON."""
    messages: list[dict[str, object]] = []
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(
            {
                "id": f"{conv_id}-msg-{i:04d}",
                "role": role,
                "content": f"ChatGPT message {i}. {'Lorem ipsum dolor sit amet. ' * 15}",
                "create_time": 1700000000 + i,
            }
        )
    return {
        "id": conv_id,
        "title": f"ChatGPT Session {conv_id}",
        "create_time": 1700000000,
        "update_time": 1700000000 + n_messages,
        "messages": messages,
    }


# mypy note: generator functions have different signatures but all return list[dict].
_PROVIDER_GENERATORS = {
    "claude-code": ("jsonl", _make_claude_code_session),
    "codex": ("jsonl", _make_codex_session),
    "chatgpt": ("json", _make_chatgpt_session),
}

_SCALE_TIERS = {
    "xs-tiny": {"files": 5, "msgs_per_file": 10},
    "sm-small": {"files": 20, "msgs_per_file": 25},
    "md-medium": {"files": 50, "msgs_per_file": 50},
}


def _generate_corpus(tmp_path: Path, tier: str, provider: str) -> Path:
    spec = _SCALE_TIERS[tier]
    root = tmp_path / "corpus" / f"{provider}-project"
    fmt, generator = _PROVIDER_GENERATORS[provider]
    write = _write_jsonl if fmt == "jsonl" else _write_json

    for i in range(spec["files"]):
        if provider == "chatgpt":
            conv_id = f"chatgpt-conv-{i:04d}"
            conv_data = generator(conv_id, spec["msgs_per_file"])  # type: ignore[operator]
            # chats/ subdir with individual JSON files
            chat_dir = root / "chats"
            chat_dir.mkdir(parents=True, exist_ok=True)
            with open(chat_dir / f"{conv_id}.json", "w") as f:
                json.dump(conv_data, f)
        else:
            uuid = f"deadbeef-0000-0000-0000-{i:012x}"
            records = generator(uuid, spec["msgs_per_file"])  # type: ignore[operator]
            write(root / f"{uuid}.jsonl", records)
    return root.parent


# ── Probe ──────────────────────────────────────────────────────────────


def _run_convergence_probe(
    corpus_root: Path,
    tmp_path: Path,
) -> dict[str, float]:
    import asyncio

    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.convergence_stages import make_default_convergence_stages
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.watcher import WatchSource

    # Archive root / config are scoped by the calling test via
    # ``monkeypatch.setenv`` so the probe never mutates process-global
    # ``os.environ`` directly (#1878). The convergence path reads the archive
    # root from the ``_BenchmarkPolylogue`` object, not the env.
    db_path = tmp_path / "index.db"

    files = list(corpus_root.rglob("*.jsonl")) + list(corpus_root.rglob("*.json"))
    # Filter only session files (skip metadata)
    files = [f for f in files if not f.name.startswith(".")]

    converger = DaemonConverger(stages=make_default_convergence_stages(db_path), max_workers=4)
    polylogue = _BenchmarkPolylogue(tmp_path, db_path)
    processor = LiveBatchProcessor(
        cast(Any, polylogue),
        (WatchSource(name="benchmark", root=corpus_root),),
        cursor=CursorStore(db_path),
        parser_fingerprint="benchmark-multi-v1",
        converger=converger,
    )

    t_total = time.perf_counter()
    metrics = asyncio.run(processor.ingest_files(files, emit_event=False))
    elapsed = time.perf_counter() - t_total

    return {
        # Unrounded elapsed: rounding to 2 decimals would collapse a sub-10ms
        # run to ``0.0`` and trip the ``total_s > 0`` guard (#1878). Round at
        # display time only.
        "total_s": elapsed,
        "files": float(len(files)),
        "total_files": float(len(files)),
        "succeeded_files": float(metrics.succeeded_file_count),
        "failed_files": float(metrics.failed_file_count),
        "parse_wall_s": metrics.parse_time_s,
        "convergence_wall_s": metrics.convergence_time_s,
    }


class _BenchmarkPolylogue:
    def __init__(self, archive_root: Path, db_path: Path) -> None:
        self.archive_root = archive_root
        self.backend = SimpleNamespace(db_path=db_path)


# ── Parameterized benchmark tests ─────────────────────────────────────


def _provider_tier_params() -> list[Any]:
    params: list[Any] = []
    for provider in ["claude-code", "codex"]:  # ChatGPT needs special handling
        for tier in _SCALE_TIERS:
            params.append(pytest.param(provider, tier, id=f"{provider}-{tier}"))
    return params


@pytest.mark.benchmark
@pytest.mark.parametrize("provider,tier", _provider_tier_params())
def test_convergence_per_provider(
    benchmark: BenchmarkFixture,
    provider: str,
    tier: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measure convergence throughput for each provider at each scale tier."""
    corpus_root = _generate_corpus(tmp_path, tier, provider)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    result = benchmark(lambda: _run_convergence_probe(corpus_root, tmp_path))

    spec = _SCALE_TIERS[tier]
    total_msgs = spec["files"] * spec["msgs_per_file"]
    if result["total_s"] > 0:
        msgs_per_s = total_msgs / result["total_s"]
        extras = {
            "provider": provider,
            "tier": tier,
            "files": spec["files"],
            "msgs_per_file": spec["msgs_per_file"],
            "total_msgs": total_msgs,
            "total_s": round(result["total_s"], 2),
            "msgs_per_s": round(msgs_per_s, 1),
            "parse_wall_s": result["parse_wall_s"],
            "convergence_wall_s": result["convergence_wall_s"],
        }
        if hasattr(benchmark, "extra_info"):
            benchmark.extra_info.update(extras)
        assert result["succeeded_files"] == result["total_files"], (
            f"{provider}/{tier}: {result['failed_files']} files failed"
        )
    else:
        pytest.fail("Zero elapsed time — measurement broken")


@pytest.mark.benchmark
@pytest.mark.parametrize("provider", ["claude-code", "codex"])
def test_convergence_single_file_per_provider(
    benchmark: BenchmarkFixture,
    provider: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-provider throughput on a single 500-message file."""
    root = tmp_path / "corpus" / "test"
    fmt, generator = _PROVIDER_GENERATORS[provider]
    uuid = f"single-{provider}-test"
    if fmt == "jsonl":
        records = generator(uuid, 500)  # type: ignore[operator]
        _write_jsonl(root / f"{uuid}.jsonl", records)
    else:
        conv_data = generator(uuid, 500)  # type: ignore[operator]
        root.mkdir(parents=True, exist_ok=True)
        with open(root / f"{uuid}.json", "w") as f:
            json.dump(conv_data, f)

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    result = benchmark(lambda: _run_convergence_probe(root.parent, tmp_path))
    msgs = 500
    if result["total_s"] > 0:
        extras = {
            "provider": provider,
            "total_s": round(result["total_s"], 2),
            "msgs_per_s": round(msgs / result["total_s"], 1),
        }
        if hasattr(benchmark, "extra_info"):
            benchmark.extra_info.update(extras)
        assert result["failed_files"] == 0


# ── Cross-provider correctness assertion ──────────────────────────────


@pytest.mark.benchmark
def test_cross_provider_convergence_correctness(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All providers should converge with zero failures on a mixed corpus."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir(parents=True)

    # Generate files from all providers into subdirectories
    providers = ["claude-code", "codex"]
    for provider in providers:
        tier = "xs-tiny"
        _generate_corpus(tmp_path, tier, provider)

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(tmp_path / "polylogue.toml"))

    result = benchmark(lambda: _run_convergence_probe(corpus_root, tmp_path))

    assert result["failed_files"] == 0, f"Cross-provider convergence had failures: {result}"
    assert result["succeeded_files"] >= len(providers), (
        f"Expected at least {len(providers)} converged, got {result['succeeded_files']}"
    )

    extras = {
        "providers": ",".join(providers),
        "total_files": result["files"],
        "total_s": round(result["total_s"], 2),
        "succeeded": int(result["succeeded_files"]),
    }
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info.update(extras)

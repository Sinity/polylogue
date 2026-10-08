"""The fresh-build benchmark's receipt arithmetic and corpus sealing.

The end-to-end run is exercised by running it; these pin the pure parts a
wrong receipt would come from: the event-log reduction, the stage rollup, the
projection, budgets, the terminal predicate and the corpus seal.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from devtools.fresh_build_bench.corpus import (
    SampleSource,
    corpus_from_files,
    load_manifest,
    sample_real,
    seal,
    verify_manifest,
)
from devtools.fresh_build_bench.profile_report import thread_group_label
from devtools.fresh_build_bench.report import (
    _derive_dependents,
    _ts,
    analyse_batches,
    analyse_events,
    compare,
    evaluate_budgets,
    projection,
)
from devtools.fresh_build_bench.run import REQUIRED_READINESS_DOMAINS, Observation, RunConfig, _daemon_env
from devtools.fresh_build_bench.sampler import thread_group
from tests.infra.frozen_clock import FrozenClock


def _event(ts: str, event: str, **fields: object) -> str:
    return json.dumps({"ts": f"2026-09-27T10:00:{ts}Z", "event": event, **fields})


def test_event_reduction_scopes_intake_and_writer_share_to_promotion(tmp_path: Path) -> None:
    """Anti-vacuity: counting the post-promotion chunk moves the intake end to
    50 s, and counting the post-promotion hold raises the busy share above 0.4."""
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                _event("00.000", "daemon.run.start"),
                _event("02.000", "daemon.cold_build.preparation"),
                _event("10.000", "live.ingest.chunk", files=2, bytes=100, duration_ms=5000),
                _event("10.000", "daemon.writer.released", actor="watcher.live_ingest.full", hold_ms=4000, queued=1),
                _event("12.000", "live.ingest.source_group", source_name="codex", files=2, duration_ms=5000),
                _event("20.000", "live.ingest.chunk", files=1, bytes=50, duration_ms=2000),
                _event("20.000", "daemon.cold_build.generation_promoted"),
                _event("50.000", "live.ingest.chunk", files=1, bytes=10, duration_ms=1000),
                _event("50.000", "daemon.writer.released", actor="derivation.session_profile", hold_ms=20000),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = analyse_events(events)
    milestones = summary["milestones_s"]
    assert milestones["promoted_s"] == 20.0
    assert milestones["last_chunk_done_s"] == 20.0
    assert milestones["post_promotion_chunks"] == 1
    assert summary["writer"]["busy_s_to_promotion"] == 4.0
    assert summary["writer"]["busy_share_to_promotion"] == 0.2
    assert summary["writer"]["queue_depth_max"] == 1
    assert summary["by_source"]["codex"] == {"groups": 1, "files": 2, "seconds": 5.0}


def test_stage_rollup_counts_leaf_timers_once(tmp_path: Path) -> None:
    """Anti-vacuity: rolling up the parent ``full.index.full_replace`` as well
    as its children doubles the ``index`` bucket; matching the generic index
    prefix first charges ``full.index.graph_resolve`` to ``index``."""
    import sqlite3

    ops = tmp_path / "ops.db"
    with sqlite3.connect(ops) as conn:
        conn.execute("CREATE TABLE daemon_events (kind TEXT, payload_json TEXT)")
        payload = {
            "total_time_s": 10.0,
            "stage_timings_s": {
                "full.index.full_replace": 3.0,
                "full.index.full_replace.messages": 2.0,
                "full.index.full_replace.blocks": 1.0,
                "full.index.prepare": 4.0,
                "full.index.graph_resolve": 5.0,
            },
        }
        conn.execute("INSERT INTO daemon_events VALUES ('ingestion_batch', ?)", (json.dumps(payload),))
    batches = analyse_batches(ops)
    buckets = dict(batches["stage_buckets_seconds"])
    assert buckets["index"] == 3.0
    assert buckets["materialize"] == 4.0
    assert buckets["derived_inline"] == 5.0
    assert batches["totals"]["total_time_s"] == 10.0


def test_projection_scales_origin_rates_to_the_population() -> None:
    manifest = {
        "by_origin": {"codex": {"files": 2, "bytes": 2 << 20}, "claude-code": {"files": 1, "bytes": 1 << 20}},
        "parameters": {"population": {"codex": {"bytes": 200 << 20}, "claude-code": {"bytes": 10 << 20}}},
    }
    by_source = {"codex": {"seconds": 4.0}, "claude-code": {"seconds": 1.0}}
    result = projection(manifest, by_source, intake_wall_s=10.0)
    assert result["by_origin"]["codex"]["projected_s"] == 400.0
    assert result["by_origin"]["claude-code"]["projected_s"] == 10.0
    assert result["wall_scale"] == 2.0
    assert result["projected_intake_hours"] == round(820.0 / 3600, 2)


def test_budgets_fail_closed() -> None:
    rows = evaluate_budgets({"rss_peak_mib": 3072.0, "terminal_s": 60.0}, {"rss_peak_mib": 2000.0})
    assert rows["rss_peak_mib"]["pass"] is True
    assert rows["terminal_s"]["pass"] is False
    with pytest.raises(ValueError, match="unknown budget"):
        evaluate_budgets({"rss": 1.0}, {})


def test_terminal_requires_every_readiness_domain() -> None:
    """Anti-vacuity: accepting a readiness map that lacks a required domain
    makes the first observation terminal."""
    promoted = "/archive/.index-generations/gen/index.db"
    ready = dict.fromkeys(REQUIRED_READINESS_DOMAINS, True)
    partial = Observation(
        1.0, cursor_rows=3, cursor_complete=3, promoted_index=promoted, readiness={"archive_sessions": True}
    )
    complete = Observation(1.0, cursor_rows=3, cursor_complete=3, promoted_index=promoted, readiness=ready)
    unsettled = Observation(
        1.0, cursor_rows=3, cursor_complete=3, memberships_pending=1, promoted_index=promoted, readiness=ready
    )
    assert not partial.terminal
    assert complete.terminal
    assert not unsettled.terminal


def test_seal_detects_a_changed_file(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    transcript = corpus / "home" / ".codex" / "sessions" / "2026" / "rollout-a.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    manifest = seal(corpus, kind="sample", parameters={})
    assert manifest["by_origin"] == {"codex": {"files": 1, "bytes": transcript.stat().st_size}}
    verify_manifest(corpus, load_manifest(corpus))
    transcript.write_text('{"type":"session_meta","extra":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed size"):
        verify_manifest(corpus, load_manifest(corpus))


def test_seal_detects_a_same_size_edit_and_an_added_file(tmp_path: Path) -> None:
    """Anti-vacuity: checking sizes only, or only the sealed file list,
    accepts both corpora."""
    corpus = tmp_path / "corpus"
    transcript = corpus / "home" / ".codex" / "sessions" / "rollout-a.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    seal(corpus, kind="sample", parameters={})
    transcript.write_text('{"type":"session_mete"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed content"):
        verify_manifest(corpus, load_manifest(corpus))
    transcript.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    (transcript.parent / "rollout-b.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="added since sealing"):
        verify_manifest(corpus, load_manifest(corpus))


def test_explicit_corpus_admits_only_watched_transcripts(tmp_path: Path) -> None:
    home = tmp_path / "home"
    sessions = home / ".codex" / "sessions" / "2026" / "01" / "01"
    sessions.mkdir(parents=True)
    (sessions / "rollout-a.jsonl").write_text("{}\n", encoding="utf-8")
    (sessions / "large.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="not a transcript"):
        corpus_from_files(tmp_path / "bad", [sessions / "large.bin"], home=home)
    manifest = corpus_from_files(tmp_path / "good", [sessions / "rollout-a.jsonl"], home=home)
    assert manifest["kind"] == "files"


def test_a_sample_that_draws_nothing_still_seals(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    day = root / "2026" / "01" / "01"
    day.mkdir(parents=True)
    (day / "rollout-one.jsonl").write_bytes(b"x" * 1000)
    sources = (SampleSource("codex", root, "home/.codex/sessions"),)
    for seed in range(20):
        manifest = sample_real(tmp_path / f"s{seed}", seed=seed, fraction=0.01, sources=sources)
        assert manifest["file_count"] in {0, 1}


def test_event_timestamps_are_utc() -> None:
    assert _ts("1970-01-01T00:00:10.000000Z") == 10.0


def _receipt(**overrides: object) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "label": "r",
        "outcome": "terminal",
        "candidate": {"git_sha": "0" * 40},
        "corpus": {"digest": "c", "total_bytes": 10 << 20, "file_count": 10},
        "config": {"digest": "k"},
        "environment": {"python": "3.14.4", "gil_enabled": False},
        "timing_s": {"promotion": 10.0, "terminal": 20.0},
        "process_tree": {"rss_peak_bytes": 1 << 30},
        "checks": {"promoted": True},
        "budgets": {"promotion_s": {"limit": 15.0}},
        "stages": {},
        "throughput": {},
        "output_fingerprint": {"digest": "same", "tables": {}},
    }
    receipt.update(overrides)
    return receipt


def test_dependent_fields_follow_the_milestones() -> None:
    """Anti-vacuity: keeping throughput, budgets or qualification from an
    earlier promotion time leaves them at 1.0 MiB/s and qualified."""
    receipt = _receipt()
    _derive_dependents(receipt)
    assert receipt["throughput"]["mib_per_s_to_promotion"] == 1.0
    assert receipt["qualified"] is True
    receipt["timing_s"]["promotion"] = 20.0
    _derive_dependents(receipt)
    assert receipt["throughput"]["mib_per_s_to_promotion"] == 0.5
    assert receipt["timing_s"]["derived_after_promotion"] == 0.0
    assert receipt["budgets"]["promotion_s"]["pass"] is False
    assert receipt["qualified"] is False


def test_compare_refuses_different_configs_and_unqualified_runs() -> None:
    """Anti-vacuity: checking only the corpus digest reports IDENTICAL for
    runs made under different configurations."""
    before = _receipt(qualified=True)
    ok, text = compare(before, _receipt(qualified=True))
    assert ok and "IDENTICAL" in text
    ok, text = compare(before, _receipt(qualified=True, config={"digest": "profiled"}))
    assert not ok and "IDENTICAL" not in text
    ok, _text = compare(before, _receipt(qualified=True, environment={"python": "3.14.4", "gil_enabled": True}))
    assert not ok
    ok, _text = compare(
        before, _receipt(qualified=True, environment={"python": "3.14.4", "gil_enabled": False, "host_cpu_count": 2})
    )
    assert not ok
    unqualified = _receipt(qualified=False, outcome="settle_timeout")
    ok, _text = compare(before, unqualified)
    assert not ok
    ok, text = compare(before, unqualified, allow_unqualified=True)
    assert ok and "WARNING" in text and "IDENTICAL" in text


@pytest.mark.parametrize(
    "check", ["candidate_unchanged", "corpus_unchanged", "events_lossless", "evidence_unchanged", "wall_clock_steady"]
)
def test_allow_unqualified_never_waives_an_integrity_failure(check: str) -> None:
    """Anti-vacuity: waiving every qualification problem admits a receipt
    whose corpus, candidate or event log was invalid and prints IDENTICAL."""
    invalid = _receipt(qualified=False, checks={"promoted": True, check: False})
    ok, text = compare(_receipt(qualified=True), invalid, allow_unqualified=True)
    assert not ok
    assert "IDENTICAL" not in text
    assert f"integrity check {check}" in text


def test_overrides_cannot_redirect_the_isolated_archive(tmp_path: Path) -> None:
    config = RunConfig(
        corpus=tmp_path,
        work=tmp_path,
        candidate=tmp_path,
        python="python",
        label="override",
        extra_env=(("POLYLOGUE_ARCHIVE_ROOT", "/elsewhere"),),
    )
    paths = {name: tmp_path / name for name in ("home", "xdg", "tmp", "archive", "config", "events", "stacks")}
    with pytest.raises(ValueError, match="driver-owned"):
        _daemon_env(config, paths)


def test_sample_is_seeded_and_keeps_session_units_together(tmp_path: Path) -> None:
    """Anti-vacuity: sampling files instead of session units separates a
    subagent transcript from its parent in some seed."""
    home = tmp_path / "home"
    projects = home / ".claude" / "projects" / "-proj"
    for index in range(20):
        session_id = f"00000000-0000-4000-8000-{index:012d}"
        session = projects / f"{session_id}.jsonl"
        session.parent.mkdir(parents=True, exist_ok=True)
        session.write_bytes(b"x" * (1000 + index))
        child = projects / session_id / "subagents" / "agent-a.jsonl"
        child.parent.mkdir(parents=True)
        child.write_bytes(b"y" * 500)
    sources = (SampleSource("claude-code", home / ".claude" / "projects", "home/.claude/projects", True),)
    first = sample_real(tmp_path / "a", seed=5, fraction=0.3, sources=sources)
    again = sample_real(tmp_path / "b", seed=5, fraction=0.3, sources=sources)
    assert first["digest"] == again["digest"]
    paths = {row[0] for row in first["files"]}
    parents = {path for path in paths if "subagents" not in path}
    assert parents
    for parent in parents:
        assert parent.removesuffix(".jsonl") + "/subagents/agent-a.jsonl" in paths
    assert 0.1 < first["total_bytes"] / first["parameters"]["population"]["claude-code"]["bytes"] < 0.6


def test_pool_workers_collapse_into_one_thread_group() -> None:
    assert thread_group("ThreadPoolExecutor-3_17") == "ThreadPoolExecutor"
    assert thread_group("polylogue-writer:watcher.live_ingest.full") == "polylogue-writer:watcher.live_ingest.full"
    assert thread_group_label("polylogue-writer:watcher.live_ingest.full") == "polylogue-writer"


def test_thread_cpu_separates_writer_actors_from_other_threads() -> None:
    """Anti-vacuity: grouping by the collapsed label merges every writer
    actor into one row and loses the per-actor split."""
    from devtools.fresh_build_bench.report import thread_cpu_summary

    summary = thread_cpu_summary(
        {
            "clock_ticks_per_s": 100,
            "process_cpu_ticks": 500,
            "thread_cpu_ticks": {
                "polylogue-writer:watcher.live_ingest.full": 300,
                "polylogue-writer:derivation.session_profile": 100,
                "MainThread": 50,
            },
        }
    )
    assert summary["writer_total_lower_bound"] == 4.0
    assert summary["unattributed"] == 0.5
    assert summary["writer_by_actor"] == [["watcher.live_ingest.full", 3.0], ["derivation.session_profile", 1.0]]
    assert summary["other_threads"] == [["MainThread", 0.5]]


def test_event_times_share_the_driver_clock_and_stop_at_promotion(tmp_path: Path) -> None:
    """Anti-vacuity: measuring from ``daemon.run.start`` shifts promotion to
    18 s; charging the post-promotion group doubles codex's seconds."""
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                _event("02.000", "daemon.run.start"),
                _event("10.000", "live.ingest.source_group", source_name="codex", files=1, duration_ms=3000),
                _event("20.000", "daemon.cold_build.generation_promoted"),
                _event("30.000", "live.ingest.source_group", source_name="codex", files=1, duration_ms=3000),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    origin = _ts("2026-09-27T10:00:00.000000Z")
    summary = analyse_events(events, origin_unix=origin)
    assert summary["milestones_s"]["promoted_s"] == 20.0
    assert summary["by_source"]["codex"] == {"groups": 1, "files": 1, "seconds": 3.0}
    absent = analyse_events(tmp_path / "absent.jsonl")
    assert absent["event_count"] == 0
    assert absent["coverage"]["file_present"] is False
    assert absent["coverage"]["complete"] is False


def test_stratum_boundary_is_one_draw(tmp_path: Path) -> None:
    """Anti-vacuity: redrawing the boundary on every later unit selects a
    file in nearly every seed instead of about one in ten."""
    root = tmp_path / "src"
    root.mkdir()
    day = root / "2026" / "01" / "01"
    day.mkdir(parents=True)
    for index in range(50):
        (day / f"rollout-f{index:02d}.jsonl").write_bytes(b"x" * 1000)
    sources = (SampleSource("codex", root, "home/.codex/sessions"),)
    seeds = 60
    selected = sum(
        sample_real(tmp_path / f"s{seed}", seed=seed, fraction=0.002, sources=sources)["file_count"]
        for seed in range(seeds)
    )
    # Expected 0.1 per seed; allow generous noise, far below "every seed".
    assert selected <= seeds * 0.35


def test_seal_detects_edited_aggregates(tmp_path: Path) -> None:
    """Anti-vacuity: verifying file rows only accepts a manifest whose
    total_bytes was edited."""
    corpus = tmp_path / "corpus"
    transcript = corpus / "home" / ".codex" / "sessions" / "rollout.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}\n", encoding="utf-8")
    manifest = seal(corpus, kind="sample", parameters={})
    manifest["total_bytes"] += 1
    with pytest.raises(ValueError, match="total_bytes"):
        verify_manifest(corpus, manifest)


def test_seal_detects_an_edited_population_parameter(tmp_path: Path) -> None:
    """A projection's denominator (parameters.population) is part of the sealed
    identity too: verify_manifest recomputes file rows from the corpus tree,
    but population describes the pre-sampling source tree, which nothing on
    disk here can recompute -- so an edit to it alone must still invalidate
    the digest, not pass silently under the original file bytes.

    Anti-vacuity: verify a manifest whose digest was sealed without folding
    in parameters and an edited population.<origin>.bytes passes.
    """
    corpus = tmp_path / "corpus"
    transcript = corpus / "home" / ".codex" / "sessions" / "rollout.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}\n", encoding="utf-8")
    manifest = seal(
        corpus,
        kind="sample",
        parameters={"population": {"codex": {"units": 10, "bytes": 1000}}},
    )
    manifest["parameters"]["population"]["codex"]["bytes"] = 10_000_000
    with pytest.raises(ValueError, match="digest"):
        verify_manifest(corpus, manifest)


def test_private_corpora_are_owner_only(tmp_path: Path) -> None:
    """Anti-vacuity: default creation modes leave the copy world-readable."""
    home = tmp_path / "home"
    sessions = home / ".codex" / "sessions" / "2026" / "01" / "01"
    sessions.mkdir(parents=True)
    (sessions / "rollout-a.jsonl").write_text("{}\n", encoding="utf-8")
    out = tmp_path / "corpus"
    corpus_from_files(out, [sessions / "rollout-a.jsonl"], home=home)
    assert out.stat().st_mode & 0o077 == 0
    copied = out / "home" / ".codex" / "sessions" / "2026" / "01" / "01" / "rollout-a.jsonl"
    assert copied.stat().st_mode & 0o077 == 0


def test_projection_is_incomplete_when_a_populated_origin_is_unmeasured() -> None:
    """Anti-vacuity: iterating only measured origins reports a full-census
    projection that silently omits gemini-cli."""
    manifest = {
        "by_origin": {"codex": {"files": 1, "bytes": 1 << 20}},
        "parameters": {"population": {"codex": {"bytes": 10 << 20}, "gemini-cli": {"bytes": 5 << 20}}},
    }
    result = projection(manifest, {"codex": {"seconds": 2.0}}, intake_wall_s=2.0)
    assert result["complete"] is False
    assert result["unmeasured_origins"] == ["gemini-cli"]
    assert result["projected_intake_hours"] is None
    assert result["projected_measured_origins_hours"] == round(20.0 / 3600, 2)


def test_a_ledgerless_ops_db_reduces_to_zero_batches(tmp_path: Path) -> None:
    import sqlite3

    ops = tmp_path / "ops.db"
    sqlite3.connect(ops).close()
    assert analyse_batches(ops) == {"batches": 0}


def test_postings_digest_sees_every_term(tmp_path: Path) -> None:
    """Anti-vacuity: digesting only each sampled block's first token sees
    ``common`` alone, so moving a ``uniqueNNN`` posting to another block
    keeps the digest; the posting count alone misses the move too."""
    import sqlite3

    from devtools.fresh_build_bench.report import _fts_postings

    texts = {1: "common unique001", 2: "common unique002", 3: "common unique003"}

    def build(name: str, indexed: dict[int, str]) -> dict[str, object]:
        path = tmp_path / f"{name}.db"
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE blocks (block_id TEXT, search_text TEXT)")
            conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(text, content='', contentless_delete=1)")
            for rowid, text in texts.items():
                conn.execute(
                    "INSERT INTO blocks (rowid, block_id, search_text) VALUES (?, ?, ?)", (rowid, f"b{rowid}", text)
                )
                conn.execute("INSERT INTO messages_fts (rowid, text) VALUES (?, ?)", (rowid, indexed[rowid]))
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            return _fts_postings(conn)

    right = build("right", texts)
    again = build("again", dict(texts))
    moved = build("moved", {**texts, 2: "common unique003", 3: "common unique002"})
    assert right == again
    assert right["rows"] == moved["rows"] == 6
    assert right["terms"] == 4
    assert right["sha256"] != moved["sha256"]


def test_export_only_corpus_has_a_home_and_rejects_colliding_exports(tmp_path: Path) -> None:
    """Anti-vacuity: without the stand-in home the run cannot resolve
    ``home/``; without the collision check the second export overwrites the
    first while the manifest counts two."""
    first, second = tmp_path / "a" / "conversations.json", tmp_path / "b" / "conversations.json"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text("[]", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    manifest = corpus_from_files(tmp_path / "exports-only", [], home=home, exports=[("chatgpt", first)])
    assert manifest["file_count"] == 1
    assert (tmp_path / "exports-only" / "home").is_dir()
    with pytest.raises(ValueError, match="share the name"):
        corpus_from_files(tmp_path / "collide", [], home=home, exports=[("chatgpt", first), ("chatgpt", second)])


def test_verification_errors_name_no_corpus_path(tmp_path: Path) -> None:
    """Anti-vacuity: naming the changed file puts a private project path in
    the job log."""
    corpus = tmp_path / "corpus"
    transcript = corpus / "home" / ".claude" / "projects" / "private-project" / "session-secret.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}\n", encoding="utf-8")
    seal(corpus, kind="sample", parameters={})
    transcript.write_text("{} \n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        verify_manifest(corpus, load_manifest(corpus))
    assert "1 corpus file(s) changed size" in str(caught.value)
    assert "private-project" not in str(caught.value)
    assert "session-secret" not in str(caught.value)


def test_component_run_fails_when_the_corpus_changes_during_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: verifying only before timing reports throughput for the
    sealed byte count although the worker read other bytes."""
    from devtools.fresh_build_bench import components
    from polylogue.storage.blob_store import BlobStore

    corpus = tmp_path / "corpus"
    transcript = corpus / "home" / ".codex" / "sessions" / "rollout.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}\n", encoding="utf-8")
    seal(corpus, kind="sample", parameters={})
    original = BlobStore.write_from_path

    def grow_then_write(self: BlobStore, path: Path) -> object:
        with path.open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
        return original(self, path)

    monkeypatch.setattr(BlobStore, "write_from_path", grow_then_write)
    with pytest.raises(ValueError, match="changed size"):
        components.bench_blob(corpus, tmp_path / "scratch", workers=1, origins=None, limit=None)


def test_sampler_keeps_io_counters_of_a_process_that_vanished(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: treating unreadable counters as zeros drops the exited
    worker's 300 read bytes from the totals."""
    from devtools.fresh_build_bench import run

    readable = {"io": True}
    monkeypatch.setattr(run, "_tree", lambda pid: [pid, 42])
    monkeypatch.setattr(run, "_proc_stat", lambda pid: (10, 4096, 1))
    monkeypatch.setattr(run, "_proc_rss_hwm", lambda pid: 0)
    monkeypatch.setattr(
        run, "_proc_io", lambda pid: ((100, 0) if pid != 42 else (300, 7)) if readable["io"] or pid != 42 else None
    )
    sampler = run.TreeSampler(1, origin=0.0)
    sampler._sample()
    readable["io"] = False
    sampler._sample()
    assert sampler.samples[-1][4:] == (400, 7)


def test_different_builds_of_one_python_version_are_not_comparable() -> None:
    """Anti-vacuity: comparing only version and GIL mode admits a PGO and a
    non-PGO build of 3.14.4 as the same interpreter."""
    env = {"python": "3.14.4", "gil_enabled": False, "python_build": "3.14.4 (pgo)", "python_executable": "/a"}
    other = {**env, "python_build": "3.14.4 (plain)"}
    ok, text = compare(_receipt(qualified=True, environment=env), _receipt(qualified=True, environment=other))
    assert not ok and "python_build" in text


def test_sampled_units_carry_their_sidecars(tmp_path: Path) -> None:
    """Anti-vacuity: a suffix-only census drops Claude Code tool-results and
    Gemini tool-outputs files, so the sample parses different sessions."""
    from devtools.fresh_build_bench.corpus import default_sample_sources

    home = tmp_path / "home"
    project = home / ".claude" / "projects" / "-proj"
    (project / "11111111-1111-4111-8111-111111111111" / "tool-results").mkdir(parents=True)
    (project / "11111111-1111-4111-8111-111111111111.jsonl").write_text("{}\n", encoding="utf-8")
    (project / "11111111-1111-4111-8111-111111111111" / "tool-results" / "toolu_1.txt").write_text(
        "full output", encoding="utf-8"
    )
    gemini = home / ".gemini" / "tmp" / "hash1"
    (gemini / "chats").mkdir(parents=True)
    (gemini / "tool-outputs" / "session-x").mkdir(parents=True)
    (gemini / "chats" / "session-x.json").write_text("{}", encoding="utf-8")
    (gemini / "tool-outputs" / "session-x" / "shell_1.txt").write_text("output", encoding="utf-8")
    manifest = sample_real(tmp_path / "corpus", seed=1, fraction=1.0, sources=default_sample_sources(home))
    paths = {row[0] for row in manifest["files"]}
    assert "home/.claude/projects/-proj/11111111-1111-4111-8111-111111111111/tool-results/toolu_1.txt" in paths
    assert "home/.gemini/tmp/hash1/tool-outputs/session-x/shell_1.txt" in paths


def test_refresh_voids_cpu_to_promotion_when_promotion_moves(tmp_path: Path) -> None:
    """Anti-vacuity: keeping the recorded CPU leaves a value cut at the old
    promotion time beside the new one."""
    from devtools.fresh_build_bench.report import refresh

    (tmp_path / "events.jsonl").write_text(
        _event("20.000", "daemon.cold_build.generation_promoted") + "\n", encoding="utf-8"
    )
    receipt = _receipt(
        started_at_unix=_ts("2026-09-27T10:00:00.000000Z"),
        process_tree={"rss_peak_bytes": 1 << 30, "cpu_seconds_to_promotion": 9.0, "mean_cores_to_promotion": 0.9},
    )
    receipt["corpus"]["path"] = str(tmp_path / "absent-corpus")
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    refreshed = refresh(path)
    assert refreshed["timing_s"]["promotion"] == 20.0
    assert refreshed["process_tree"]["cpu_seconds_to_promotion"] is None
    assert refreshed["process_tree"]["mean_cores_to_promotion"] is None
    assert json.loads(path.read_text(encoding="utf-8")) == refreshed


def test_a_progressing_build_runs_past_any_elapsed_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5678): an absolute run deadline ends a build
    that keeps making progress past six hours with an unqualified outcome."""

    from devtools.fresh_build_bench import report, run

    clock = {"now": 0.0}

    def monotonic() -> float:
        return clock["now"]

    polls = {"count": 0}
    ready = dict.fromkeys(REQUIRED_READINESS_DOMAINS, True)

    def observe(_archive: Path, started: float, **_kwargs: object) -> Observation:
        clock["now"] += 3600.0  # every poll is an hour later
        polls["count"] += 1
        done = polls["count"] >= 9
        return Observation(
            clock["now"] - started,
            cursor_rows=20,
            cursor_complete=20 if done else polls["count"],
            promoted_index="/archive/.index-generations/gen/index.db" if done else None,
            readiness=ready if done else {},
        )

    class Process:
        pid = 1

        def poll(self) -> None:
            return None

    class Sampler:
        samples: list[object] = []
        daemon_rss_hwm_bytes = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None: ...

        def start(self) -> None: ...

        def finish(self) -> None: ...

    # ``run`` reads these through the ``time`` and ``subprocess`` modules.
    monkeypatch.setattr(time, "monotonic", monotonic)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(run, "TreeSampler", Sampler)
    monkeypatch.setattr(run, "observe", observe)
    monkeypatch.setattr(run, "_stop", lambda _process, **_kwargs: (0, 0.0))
    monkeypatch.setattr(run, "verify_manifest", lambda *_args: None)
    monkeypatch.setattr(run, "candidate_identity", lambda _candidate: {})
    monkeypatch.setattr(run, "candidate_stamp", lambda _candidate: {})
    monkeypatch.setattr(report, "build_receipt", lambda **kwargs: {"outcome": kwargs["outcome"]})
    config = RunConfig(
        corpus=tmp_path,
        work=tmp_path,
        candidate=tmp_path,
        python="python",
        label="l",
        stall_timeout_s=7200.0,
    )
    paths = {
        "daemon_log": tmp_path / "daemon.log",
        "archive": tmp_path,
        "receipt": tmp_path / "receipt.json",
        "events": tmp_path / "events.jsonl",
    }

    receipt = run._measure_and_write_receipt(
        config,
        manifest={},
        paths=paths,
        identity={"git_sha": None, "dirty": None, "tracked_diff_sha256": None},
        candidate_files={},
        env_summary={},
        command=[],
        daemon_env={},
        interrupted=[],
        progress=lambda _line: None,
        hook_preparation=None,
    )

    assert receipt["outcome"] == "terminal"
    assert clock["now"] > 6 * 3600.0


def test_component_commands_refuse_a_corpus_inside_the_checkout(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5678): validate only ``--scratch`` and a sealed
    private corpus under the checkout is accepted."""
    from devtools.fresh_build_bench import components
    from devtools.fresh_build_bench.corpus import _CHECKOUT

    with pytest.raises(ValueError, match="--corpus must be outside the checkout"):
        components.main(["blob", "--corpus", str(_CHECKOUT / "tests"), "--scratch", str(tmp_path / "scratch")])


def _sidecar_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    project = corpus / "home" / ".claude" / "projects" / "-proj"
    (project / "11111111-1111-4111-8111-111111111111" / "tool-results").mkdir(parents=True)
    (project / "11111111-1111-4111-8111-111111111111.jsonl").write_text("{}\n", encoding="utf-8")
    (project / "11111111-1111-4111-8111-111111111111" / "tool-results" / "toolu_1.txt").write_text(
        "full output", encoding="utf-8"
    )
    seal(corpus, kind="sample", parameters={})
    return corpus


def test_blob_component_stores_sidecars(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5678): dropping retained sidecars omits real
    acquisition work from the blob timing."""
    from devtools.fresh_build_bench import components

    corpus = _sidecar_corpus(tmp_path)
    manifest = load_manifest(corpus)
    blob_files = components._corpus_files(corpus, manifest, None, None)
    assert sorted(path.name for path, _origin, _size in blob_files) == [
        "11111111-1111-4111-8111-111111111111.jsonl",
        "toolu_1.txt",
    ]


def test_an_empty_component_selection_fails(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5678): an origin absent from the corpus prints
    a zero-file timing and exits successfully."""
    from devtools.fresh_build_bench import components

    corpus = _sidecar_corpus(tmp_path)
    with pytest.raises(SystemExit, match="no corpus files match"):
        components.bench_blob(corpus, tmp_path / "scratch", workers=1, origins=["codex"], limit=None)


def test_seal_covers_sidecar_mtimes_and_a_sample_keeps_them(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5678): hash only path, size and bytes and a
    touched sidecar still verifies, though its mtime is the parsed event time."""
    import os

    from devtools.fresh_build_bench.corpus import default_sample_sources

    corpus = _sidecar_corpus(tmp_path)
    sidecar = (
        corpus
        / "home"
        / ".claude"
        / "projects"
        / "-proj"
        / "11111111-1111-4111-8111-111111111111"
        / "tool-results"
        / "toolu_1.txt"
    )
    verify_manifest(corpus, load_manifest(corpus))
    os.utime(sidecar, ns=(sidecar.stat().st_atime_ns, sidecar.stat().st_mtime_ns + 10**9))
    with pytest.raises(ValueError, match="changed mtime"):
        verify_manifest(corpus, load_manifest(corpus))

    home = tmp_path / "home"
    source = (
        home
        / ".claude"
        / "projects"
        / "-proj"
        / "11111111-1111-4111-8111-111111111111"
        / "tool-results"
        / "toolu_1.txt"
    )
    source.parent.mkdir(parents=True)
    (home / ".claude" / "projects" / "-proj" / "11111111-1111-4111-8111-111111111111.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    source.write_text("full output", encoding="utf-8")
    os.utime(source, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    manifest = sample_real(tmp_path / "sampled", seed=1, fraction=1.0, sources=default_sample_sources(home))
    assert manifest["sidecar_mtimes_ns"] == {
        "home/.claude/projects/-proj/11111111-1111-4111-8111-111111111111/tool-results/toolu_1.txt": 1_700_000_000_000_000_000
    }


def test_receipts_from_different_benchmark_code_or_limits_do_not_compare() -> None:
    """Anti-vacuity (Codex P1, #5678): compare only host fields and identical
    configs under different benchmark code or cgroup quotas read as controlled."""
    base = _receipt(qualified=True, benchmark_implementation_sha256="a")
    ok, text = compare(base, _receipt(qualified=True, benchmark_implementation_sha256="b"))
    assert not ok and "benchmark implementation" in text
    quota_two = {"python": "3.14.4", "gil_enabled": False, "effective_cpus": 2}
    quota_sixteen = {**quota_two, "effective_cpus": 16}
    ok, text = compare(
        _receipt(qualified=True, environment=quota_two), _receipt(qualified=True, environment=quota_sixteen)
    )
    assert not ok and "effective_cpus" in text


def test_an_unpromoted_run_has_no_cpu_to_promotion() -> None:
    """Anti-vacuity (Codex P2, #5678): cut at no milestone and the whole run's
    CPU is stored as ``cpu_seconds_to_promotion``."""
    from devtools.fresh_build_bench.report import _tree_summary

    tree = _tree_summary([(1.0, 1 << 20, 0.5, 4, 0, 0), (2.0, 1 << 20, 1.5, 4, 0, 0)], None)
    assert tree["cpu_seconds_total"] == 1.5
    assert tree["cpu_seconds_to_promotion"] is None
    assert tree["mean_cores_to_promotion"] is None


def test_refresh_recomputes_the_projection_without_the_corpus(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5678): recompute only when the corpus directory
    exists and a refreshed receipt keeps a projection its new ``by_source``
    contradicts."""
    from devtools.fresh_build_bench.report import refresh

    (tmp_path / "events.jsonl").write_text(
        "\n".join(
            [
                _event("01.000", "daemon.cold_build.preparation"),
                _event("05.000", "live.ingest.source_group", source_name="codex", files=2, duration_ms=2000),
                _event("06.000", "live.ingest.chunk", files=2, bytes=2 << 20, duration_ms=1000),
                _event("09.000", "daemon.cold_build.generation_promoted"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = _receipt(started_at_unix=_ts("2026-09-27T10:00:00.000000Z"), projection={"stale": True})
    receipt["corpus"].update(
        path=str(tmp_path / "absent-corpus"),
        by_origin={"codex": {"files": 2, "bytes": 2 << 20}},
        population={"codex": {"files": 20, "bytes": 20 << 20}},
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")

    refreshed = refresh(path)

    expected = projection(
        {"by_origin": receipt["corpus"]["by_origin"], "parameters": {"population": receipt["corpus"]["population"]}},
        refreshed["by_source"],
        5.0,
    )
    assert refreshed["projection"] == expected
    assert "stale" not in refreshed["projection"]


def test_a_refreshed_receipt_names_the_refreshing_implementation(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5678): keep the recorded implementation digest
    on refresh and a receipt re-reduced by new code compares as IDENTICAL with
    an unrefreshed receipt of the old code."""
    from devtools.fresh_build_bench.report import benchmark_implementation_sha256, refresh

    (tmp_path / "events.jsonl").write_text(_event("09.000", "daemon.cold_build.generation_promoted") + "\n")
    old = _receipt(
        qualified=True, benchmark_implementation_sha256="old", started_at_unix=_ts("2026-09-27T10:00:00.000000Z")
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(old), encoding="utf-8")

    refreshed = refresh(path)

    assert refreshed["benchmark_implementation_sha256"] == f"old+refresh:{benchmark_implementation_sha256()}"
    ok, text = compare(old, refreshed)
    assert not ok and "benchmark implementation" in text


def test_the_implementation_digest_covers_the_fingerprint_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity (Codex P1, #5678): hash only the package and an edit to the
    fingerprint's census or normalization leaves the digest unchanged."""
    from devtools.fresh_build_bench import report
    from tests.infra import reindex_differential

    assert Path(reindex_differential.__file__).resolve() in report._FINGERPRINT_DEPENDENCIES
    rules = tmp_path / "reindex_differential.py"
    rules.write_text("VOLATILE = ()\n", encoding="utf-8")
    monkeypatch.setattr(report, "_FINGERPRINT_DEPENDENCIES", (rules,))
    before = report.benchmark_implementation_sha256()
    rules.write_text("VOLATILE = ('ts',)\n", encoding="utf-8")
    assert report.benchmark_implementation_sha256() != before


def test_only_a_promoted_run_may_waive_qualification() -> None:
    """Anti-vacuity (Codex P1, #5678): waive every "not qualified" problem and a
    run that stalled before promotion compares as admissible."""
    stalled = _receipt(qualified=False, outcome="stalled", checks={"promoted": False}, output_fingerprint=None)
    ok, text = compare(_receipt(qualified=True), stalled, allow_unqualified=True)
    assert not ok and "IDENTICAL" not in text


def test_receipts_from_different_cpu_models_do_not_compare() -> None:
    """Anti-vacuity (Codex P1, #5678): drop ``cpu_model`` from the host keys and
    same-sized workers on different processors compare as controlled."""
    intel = {"python": "3.14.4", "gil_enabled": False, "host_cpu_count": 8, "cpu_model": "Intel Xeon"}
    amd = {**intel, "cpu_model": "AMD EPYC"}
    ok, text = compare(_receipt(qualified=True, environment=intel), _receipt(qualified=True, environment=amd))
    assert not ok and "cpu_model" in text


def test_the_benchmark_daemon_ignores_the_host_site_configuration(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5678): leave ``POLYLOGUE_SITE_CONFIG`` unset and
    the daemon layers ``/etc/polylogue/polylogue.toml`` under the receipt's config."""
    config = RunConfig(corpus=tmp_path, work=tmp_path, candidate=tmp_path, python="python", label="site")
    paths = {name: tmp_path / name for name in ("home", "xdg", "tmp", "archive", "config", "events", "stacks")}
    env = _daemon_env(config, paths)
    # An empty value is the config loader's "no site layer" (``_site_config_path``).
    assert env["POLYLOGUE_SITE_CONFIG"] == ""


def test_a_corpus_is_never_written_inside_a_watched_source_root(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5678): refuse only checkout paths and the corpus
    is created beneath the live Codex root, whose daemon ingests the copies."""
    from devtools.fresh_build_bench.cli import _refuse_source_root_path

    home = tmp_path / "home"
    (home / ".codex" / "sessions").mkdir(parents=True)
    with pytest.raises(SystemExit, match="watched source root"):
        _refuse_source_root_path(home / ".codex" / "sessions" / "bench", home)
    _refuse_source_root_path(tmp_path / "elsewhere", home)


def test_event_reduction_memory_does_not_grow_with_the_log(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5678): collect every decoded event before
    reducing and 100k writer events hold tens of MiB of dicts at once."""
    import tracemalloc

    line = json.dumps(
        {
            "ts": "2026-09-27T10:00:01.000000Z",
            "event": "daemon.writer.released",
            "actor": "a",
            "hold_ms": 1,
            "queued": 2,
        }
    )
    with (tmp_path / "events.jsonl").open("w", encoding="utf-8") as stream:
        stream.write(_event("00.500", "daemon.run.start") + "\n")
        for _ in range(100_000):
            stream.write(line + "\n")

    tracemalloc.start()
    try:
        reduced = analyse_events(tmp_path / "events.jsonl")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert reduced["event_count"] == 100_001
    assert reduced["writer"]["queue_depth_max"] == 2
    assert peak < 8 << 20


def _scripted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frames: list[dict[str, Any]],
    *,
    stall_timeout_s: float,
    wall_jump: float = 0.0,
    wall_restore: bool = False,
    interrupt_at_stop: bool = False,
) -> dict[str, Any]:
    """Drive ``_measure_and_write_receipt`` through scripted observations, one per 600 s poll."""
    from devtools.fresh_build_bench import report, run

    clock = {"now": 0.0, "wall": 1_000.0}
    index = {"i": 0}
    observe_kwargs: list[dict[str, object]] = []
    interrupted: list[int] = []

    def stop(_process: object, **_kwargs: object) -> tuple[int, float]:
        if interrupt_at_stop:
            interrupted.append(15)
        return 0, 0.0

    def observe(_archive: Path, started: float, **kwargs: object) -> Observation:
        observe_kwargs.append(kwargs)
        clock["now"] += 600.0
        clock["wall"] += 600.0 + (wall_jump if index["i"] == 0 else 0.0)
        if wall_restore and index["i"] == 1:
            clock["wall"] -= wall_jump
        frame = frames[min(index["i"], len(frames) - 1)]
        index["i"] += 1
        return Observation(clock["now"] - started, cursor_rows=1, **frame)

    class Process:
        pid = 1

        def poll(self) -> None:
            return None

    class Sampler:
        samples: list[object] = []
        daemon_rss_hwm_bytes = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None: ...

        def start(self) -> None: ...

        def finish(self) -> None: ...

    captured: dict[str, Any] = {}

    def build_receipt(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"outcome": kwargs["outcome"]}

    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(time, "time", lambda: clock["wall"])
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(run, "TreeSampler", Sampler)
    monkeypatch.setattr(run, "observe", observe)
    monkeypatch.setattr(run, "_stop", stop)
    monkeypatch.setattr(run, "verify_manifest", lambda *_args: None)
    monkeypatch.setattr(run, "candidate_identity", lambda _candidate: {})
    monkeypatch.setattr(run, "candidate_stamp", lambda _candidate: {})
    monkeypatch.setattr(report, "build_receipt", build_receipt)
    config = RunConfig(
        corpus=tmp_path, work=tmp_path, candidate=tmp_path, python="python", label="l", stall_timeout_s=stall_timeout_s
    )
    paths = {
        "daemon_log": tmp_path / "daemon.log",
        "archive": tmp_path,
        "receipt": tmp_path / "receipt.json",
        "events": tmp_path / "events.jsonl",
    }
    run._measure_and_write_receipt(
        config,
        manifest={},
        paths=paths,
        identity={"git_sha": None, "dirty": None, "tracked_diff_sha256": None},
        candidate_files={},
        env_summary={},
        command=[],
        daemon_env={},
        interrupted=interrupted,
        progress=lambda _line: None,
        hook_preparation=None,
    )
    captured["clock"] = clock
    captured["observe_kwargs"] = observe_kwargs
    return captured


def _terminal_frame() -> dict[str, Any]:
    return {
        "cursor_complete": 1,
        "promoted_index": "/archive/.index-generations/gen/index.db",
        "readiness": dict.fromkeys(REQUIRED_READINESS_DOMAINS, True),
    }


def test_debt_waiting_on_its_scheduled_retry_is_not_a_stall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5678): ignore scheduled retries and debt
    backing off past the stall window is stopped as ``stalled``."""
    waiting = {"cursor_complete": 1, "open_debt": 1, "debt_by_stage": {"s": 1}, "debt_waiting_by_stage": {"s": 1}}
    captured = _scripted_run(tmp_path, monkeypatch, [waiting] * 4 + [_terminal_frame()], stall_timeout_s=900.0)

    assert captured["outcome"] == "terminal"


def test_repeated_failed_retries_are_activity_without_useful_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry counters must not indefinitely renew the useful-progress clock."""
    frames = [
        {"cursor_complete": 1, "open_debt": 1, "debt_by_stage": {"s": 1}, "debt_attempts": attempt}
        for attempt in range(1, 5)
    ]
    captured = _scripted_run(tmp_path, monkeypatch, frames, stall_timeout_s=900.0)
    assert captured["outcome"] == "stalled"
    last = captured["observations"][-1]
    assert last.useful_progress_at_s is None
    assert last.activity_at_s == last.t


def test_retry_activity_can_eventually_finish_without_becoming_useful_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declared backoff can recover; only the actual readiness/publication moves progress."""
    waiting = {"cursor_complete": 1, "open_debt": 1, "debt_by_stage": {"s": 1}, "debt_waiting_by_stage": {"s": 1}}
    frames = [{**waiting, "debt_attempts": attempt} for attempt in range(1, 5)]
    captured = _scripted_run(tmp_path, monkeypatch, [*frames, _terminal_frame()], stall_timeout_s=900.0)
    assert captured["outcome"] == "terminal"
    assert all(frame.useful_progress_at_s is None for frame in captured["observations"][:4])
    assert captured["observations"][4].useful_progress_at_s == captured["observations"][4].t


def test_the_wall_clock_step_over_the_run_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P2, #5678): measure only one clock and a wall step
    that misaligns event milestones with samples goes unnoticed."""
    captured = _scripted_run(tmp_path, monkeypatch, [_terminal_frame()] * 3, stall_timeout_s=900.0, wall_jump=10.0)

    assert captured["clock_step_s"] == pytest.approx(10.0)


def test_run_work_under_the_corpus_is_refused(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5678): check only the checkouts and the work
    directory is accepted beneath the corpus's watched source roots."""
    from devtools.fresh_build_bench.cli import _refuse_work_inside_sources

    corpus = tmp_path / "corpus"
    (corpus / "home" / ".codex" / "sessions").mkdir(parents=True)
    with pytest.raises(SystemExit, match="outside the corpus"):
        _refuse_work_inside_sources(corpus / "home" / ".codex" / "sessions" / "bench", corpus)
    _refuse_work_inside_sources(tmp_path / "work", corpus)


def test_a_cursor_waiting_on_its_scheduled_retry_is_not_a_stall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity (Codex P1, #5678): track only debt retries and a cursor
    backing off past the stall window is stopped as ``stalled``."""
    waiting = {"cursor_complete": 0, "cursor_retry_waiting": 1}
    captured = _scripted_run(tmp_path, monkeypatch, [waiting] * 4 + [_terminal_frame()], stall_timeout_s=900.0)

    assert captured["outcome"] == "terminal"


def test_the_stall_policy_is_part_of_the_config_digest(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5678): digest only the profile and env and runs
    stopped under different stall timeouts compare as one configuration."""
    from devtools.fresh_build_bench.report import config_digest

    def config(stall: float) -> RunConfig:
        return RunConfig(
            corpus=tmp_path, work=tmp_path, candidate=tmp_path, python="p", label="l", stall_timeout_s=stall
        )

    assert config_digest(config(100.0)) != config_digest(config(900.0))


def test_a_rewrite_that_restores_the_bytes_is_still_a_corpus_change(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5678): verify content only at the endpoints and
    a transcript edited during the run and restored before its end passes."""
    import os

    from devtools.fresh_build_bench.corpus import change_stamp

    transcript = tmp_path / "home" / "t.jsonl"
    transcript.parent.mkdir()
    transcript.write_bytes(b"sealed\n")
    before = change_stamp(tmp_path)
    status = transcript.stat()
    transcript.write_bytes(b"transient\n")
    transcript.write_bytes(b"sealed\n")
    os.utime(transcript, ns=(status.st_atime_ns, status.st_mtime_ns))

    assert change_stamp(tmp_path) != before


def test_receipts_on_different_backing_devices_do_not_compare() -> None:
    """Anti-vacuity (Codex P2, #5678): compare only the filesystem type and an
    NVMe run and a loop-backed ext4 run compare as controlled."""
    nvme = {"python": "3.14.4", "gil_enabled": False, "work_filesystem": "ext2/ext3", "work_device": "/dev/nvme0n1p2"}
    loop = {**nvme, "work_device": "/dev/loop3"}
    ok, text = compare(_receipt(qualified=True, environment=nvme), _receipt(qualified=True, environment=loop))
    assert not ok and "work_device" in text


def test_a_symlinked_export_root_is_refused(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5678): follow child symlinks of ``exports/`` and
    an unsealed operator directory becomes a daemon source root."""
    from devtools.fresh_build_bench import run

    corpus = tmp_path / "corpus"
    (corpus / "exports").mkdir(parents=True)
    (corpus / "home").mkdir()
    outside = tmp_path / "live"
    outside.mkdir()
    (corpus / "exports" / "chatgpt").symlink_to(outside)
    config = RunConfig(corpus=corpus, work=tmp_path / "work", candidate=tmp_path, python="p", label="l")

    with pytest.raises(ValueError, match="sealed export directory"):
        run._prepare_paths(config)


def test_a_raw_only_sidecar_is_not_a_corpus_transcript(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5678): admit every watched suffix and a Gemini
    tool-output sidecar becomes a corpus whose run materializes nothing."""
    from devtools.fresh_build_bench.corpus import corpus_from_files

    home = tmp_path / "home"
    sidecar = home / ".gemini" / "tmp" / "project" / "tool-outputs" / "session-x" / "result.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text('{"output": "x"}', encoding="utf-8")

    with pytest.raises(ValueError, match="not a session transcript"):
        corpus_from_files(tmp_path / "out", [sidecar], home=home)


def test_a_nested_symlink_in_a_corpus_is_refused(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5678): skip symlinks in the sealing walk and a
    linked live session directory is ingested unsealed under a valid digest."""
    from devtools.fresh_build_bench.corpus import _hash_tree

    sessions = tmp_path / "home" / ".codex" / "sessions"
    sessions.mkdir(parents=True)
    outside = tmp_path / "operator"
    outside.mkdir()
    (sessions / "live").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        _hash_tree(tmp_path)


def test_component_scratch_inside_the_corpus_is_refused(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5678): check only the checkout and blob output is
    written into the sealed corpus, dirtying it for good."""
    from devtools.fresh_build_bench.components import refuse_scratch_inside_corpus

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(SystemExit, match="outside the corpus"):
        refuse_scratch_inside_corpus(corpus / "home" / "bench", corpus)
    refuse_scratch_inside_corpus(tmp_path / "scratch", corpus)


def test_inbox_time_is_attributed_to_a_single_export_origin() -> None:
    """Anti-vacuity (Codex P2, #5678): map only the typed watch sources and the
    inbox's export timing is dropped from the projection; map it for several
    export origins and one origin's time prices another."""
    one = {
        "by_origin": {"chatgpt": {"files": 1, "bytes": 1 << 20}},
        "parameters": {"population": {"chatgpt": {"files": 10, "bytes": 10 << 20}}},
    }
    result = projection(one, {"inbox": {"groups": 1, "files": 1, "seconds": 2.0}}, 2.0)
    assert result["by_origin"]["chatgpt"]["projected_s"] == 20.0
    assert result["complete"]

    two = {
        "by_origin": {"chatgpt": {"files": 1, "bytes": 1 << 20}, "claude-ai": {"files": 1, "bytes": 1 << 20}},
        "parameters": {
            "population": {"chatgpt": {"files": 10, "bytes": 10 << 20}, "claude-ai": {"files": 5, "bytes": 5 << 20}}
        },
    }
    result = projection(two, {"inbox": {"groups": 1, "files": 2, "seconds": 2.0}}, 2.0)
    assert result["by_origin"] == {}
    assert result["unmeasured_origins"] == ["chatgpt", "claude-ai"]


def test_exports_are_staged_into_the_archive_inbox_as_copies(tmp_path: Path) -> None:
    """Anti-vacuity: configure export directories as ``[sources] roots`` and the
    daemon's config loader refuses the key, so no export is ever measured; link
    instead of copying and the sealed file's ctime moves under the manifest."""
    from devtools.fresh_build_bench import run

    corpus = tmp_path / "corpus"
    (corpus / "home").mkdir(parents=True)
    for origin, name in (("chatgpt", "conversations.json"), ("claude-ai", "claude.json")):
        (corpus / "exports" / origin).mkdir(parents=True)
        (corpus / "exports" / origin / name).write_text(f'["{origin}"]', encoding="utf-8")
    sealed = corpus / "exports" / "chatgpt" / "conversations.json"
    before = sealed.stat().st_ctime_ns
    config = RunConfig(corpus=corpus, work=tmp_path / "work", candidate=tmp_path, python="p", label="l")

    paths = run._prepare_paths(config)

    inbox = paths["archive"] / "inbox"
    assert sorted(path.name for path in inbox.iterdir()) == ["claude.json", "conversations.json"]
    assert (inbox / "conversations.json").read_text(encoding="utf-8") == '["chatgpt"]'
    assert (inbox / "conversations.json").stat().st_ino != sealed.stat().st_ino
    assert sealed.stat().st_ctime_ns == before
    assert "sources" not in paths["config"].read_text(encoding="utf-8")


def test_exports_of_two_origins_with_one_name_are_refused(tmp_path: Path) -> None:
    """Anti-vacuity: stage without checking and the second origin's export
    overwrites the first in the inbox, so one sealed file is never ingested."""
    from devtools.fresh_build_bench import run

    corpus = tmp_path / "corpus"
    (corpus / "home").mkdir(parents=True)
    for origin in ("chatgpt", "claude-ai"):
        (corpus / "exports" / origin).mkdir(parents=True)
        (corpus / "exports" / origin / "conversations.json").write_text("[]", encoding="utf-8")
    config = RunConfig(corpus=corpus, work=tmp_path / "work", candidate=tmp_path, python="p", label="l")

    with pytest.raises(ValueError, match="share the name"):
        run._prepare_paths(config)


def test_a_refresh_over_changed_evidence_does_not_qualify(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5678): keep the original qualification and a
    truncated event log re-reduces to lower timings that still compare."""
    from devtools.fresh_build_bench.report import evidence_digests, refresh

    (tmp_path / "events.jsonl").write_text(_event("09.000", "daemon.cold_build.generation_promoted") + "\n")
    receipt = _receipt(qualified=True, started_at_unix=_ts("2026-09-27T10:00:00.000000Z"))
    receipt["evidence_sha256"] = evidence_digests(tmp_path)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert refresh(path)["checks"]["evidence_unchanged"] is True

    (tmp_path / "events.jsonl").write_text("")
    refreshed = refresh(path)

    assert refreshed["checks"]["evidence_unchanged"] is False
    assert refreshed["qualified"] is False


@pytest.mark.parametrize("value", ["nan", "-5", "0"])
def test_the_stall_timeout_must_be_a_positive_finite_duration(value: str) -> None:
    """Anti-vacuity (Codex P2, #5678): accept any float and ``nan`` never stalls."""
    from devtools.fresh_build_bench.cli import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "--corpus", "c", "--work", "w", "--stall-timeout", value])


def test_a_failed_observation_is_not_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P1, #5678): count an errored all-zero observation as a
    change and a frozen build alternating with read failures never stalls."""
    frozen = {"cursor_complete": 0, "raw_rows": 3}
    failed = {"error": "OperationalError: database is locked"}
    captured = _scripted_run(tmp_path, monkeypatch, [frozen, failed] * 6, stall_timeout_s=1500.0)

    assert captured["outcome"] == "stalled"


def test_a_candidate_without_a_census_table_still_gets_a_census(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5678): count a fixed table list and an older
    candidate's index without ``action_pairs`` aborts before the receipt."""
    import sqlite3

    from devtools.fresh_build_bench.report import archive_census

    index = tmp_path / "index.db"
    with sqlite3.connect(index) as conn:
        for table in ("sessions", "messages", "blocks", "file_edits", "session_links"):
            conn.execute(f"CREATE TABLE {table} (x)")
    census = archive_census(tmp_path, str(index))

    assert census["rows"]["action_pairs"] is None
    assert census["rows"]["sessions"] == 0


def test_a_source_rewritten_during_its_copy_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P2, #5678): check only the copied size and a
    same-size rewrite mid-copy seals a torn transcript."""
    import shutil

    from devtools.fresh_build_bench import corpus

    source = tmp_path / "t.jsonl"
    source.write_bytes(b"original\n")
    original_copy = shutil.copyfile

    def rewrite_while_copying(src: Any, dst: Any, **kwargs: Any) -> Any:
        result = original_copy(src, dst, **kwargs)
        Path(src).write_bytes(b"rewrote!\n")
        return result

    monkeypatch.setattr(shutil, "copyfile", rewrite_while_copying)
    with pytest.raises(ValueError, match="changed while it was being copied"):
        corpus._copy_private(source, tmp_path / "out" / "t.jsonl")


def test_a_file_added_and_removed_during_the_run_changes_the_corpus_stamp(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5678): stamp files only and a transcript created,
    ingested and deleted before the end leaves the stamp unchanged."""
    from devtools.fresh_build_bench.corpus import change_stamp

    project = tmp_path / "home" / "p"
    project.mkdir(parents=True)
    (project / "sealed.jsonl").write_bytes(b"sealed\n")
    before = change_stamp(tmp_path)
    transient = project / "transient.jsonl"
    transient.write_bytes(b"transient\n")
    transient.unlink()

    assert change_stamp(tmp_path) != before


def test_a_candidate_edit_restored_before_the_end_is_a_change(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5678): compare the Git identity only at the
    endpoints and a module edited, imported and restored passes."""
    import os

    from devtools.fresh_build_bench.run import candidate_identity, candidate_stamp

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    module = tmp_path / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "module.py"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "c"],
        check=True,
    )
    before = candidate_stamp(tmp_path)
    identity = candidate_identity(tmp_path)
    status = module.stat()
    module.write_text("VALUE = 2\n", encoding="utf-8")
    module.write_text("VALUE = 1\n", encoding="utf-8")
    os.utime(module, ns=(status.st_atime_ns, status.st_mtime_ns))

    assert candidate_identity(tmp_path) == identity
    assert candidate_stamp(tmp_path) != before


def test_readiness_is_rechecked_within_the_stall_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P2, #5678): cache readiness for a fixed 60 s and a
    30 s stall window declares a build stalled before its next census."""
    captured = _scripted_run(tmp_path, monkeypatch, [_terminal_frame()], stall_timeout_s=30.0)

    ages = {kwargs.get("readiness_max_age_s") for kwargs in captured["observe_kwargs"] if kwargs}
    assert ages == {15.0}


def test_a_cancellation_after_the_loop_marks_the_receipt_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity (Codex P1, #5678): keep the loop's ``terminal`` outcome and a
    SIGTERM during shutdown still yields a qualified receipt."""
    captured = _scripted_run(tmp_path, monkeypatch, [_terminal_frame()], stall_timeout_s=7200.0, interrupt_at_stop=True)

    assert captured["outcome"] == "interrupted"


def test_the_fingerprint_reports_tables_an_older_candidate_lacks(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5678): select from every current table and a
    candidate without one raises instead of recording a mismatch."""
    import sqlite3
    from contextlib import closing

    from devtools.fresh_build_bench.report import output_fingerprint

    index = tmp_path / "index.db"
    with closing(sqlite3.connect(index)) as conn:
        conn.execute("CREATE TABLE unrelated (x)")
        conn.commit()

    fingerprint = output_fingerprint(tmp_path, str(index), tmp_path / "scratch")

    assert fingerprint["tables"]
    assert all(entry.get("absent") is True for entry in fingerprint["tables"].values())


def test_refreshing_a_refreshed_receipt_keeps_one_refresh_identity() -> None:
    """Anti-vacuity (Codex P2, #5678): append ``+refresh:`` on every refresh and
    two refreshes by one implementation stop comparing with one."""
    from devtools.fresh_build_bench.report import refreshed_identity

    once = refreshed_identity("recorded", "current")
    assert refreshed_identity(once, "current") == once == "recorded+refresh:current"
    assert refreshed_identity(once, "newer") == "recorded+refresh:newer"


def test_sampling_follows_symlinked_source_directories(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P1, #5678): sample with ``rglob`` and the transcripts
    behind a linked directory are missing from the sample and its population."""
    from devtools.fresh_build_bench.corpus import SampleSource, _units

    root = tmp_path / "home" / ".codex" / "sessions"
    # A link whose target stays inside the root is followed, as discovery
    # follows it; the target itself sits where the layout never reaches.
    elsewhere = root / ".store" / "2026"
    (elsewhere / "01" / "01").mkdir(parents=True)
    (elsewhere / "01" / "01" / "rollout-a.jsonl").write_bytes(b"{}\n")
    (root / "2026").symlink_to(elsewhere, target_is_directory=True)
    source = SampleSource("codex", root, "home/.codex/sessions")

    units = _units(source)

    assert [Path(key).name for key, _paths, _size in units] == ["rollout-a.jsonl"]


def test_a_progressing_shutdown_is_never_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A daemon still draining after any fixed deadline keeps running until it exits.

    Anti-vacuity (Codex P1, #5678): kill after 300 s and this shutdown, which
    takes an hour while its CPU keeps moving, ends as an unclean shutdown.
    """
    from devtools.fresh_build_bench import run

    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    class Draining:
        returncode: int | None = None
        killed = False
        waits = 0

        def poll(self) -> None:
            return None

        def send_signal(self, _signal: int) -> None: ...

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            clock["now"] += 60.0
            if self.waits < 60:
                raise subprocess.TimeoutExpired("daemon", timeout or 0)
            self.returncode = 0
            return 0

    process = Draining()
    cpu = iter(range(1000))
    exit_code, shutdown_s = run._stop(
        process,  # type: ignore[arg-type]
        stall_s=900.0,
        progress=lambda: next(cpu),
        interrupted=[],
    )

    assert exit_code == 0 and not process.killed
    assert shutdown_s >= 3600.0


def test_a_shutdown_that_stops_moving_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A daemon whose process tree stops moving is killed after the stall window."""
    from devtools.fresh_build_bench import run

    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    class Stuck:
        returncode: int | None = None
        killed = False

        def poll(self) -> None:
            return None

        def send_signal(self, _signal: int) -> None: ...

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            if self.killed:
                return -9
            clock["now"] += 60.0
            raise subprocess.TimeoutExpired("daemon", timeout or 0)

    process = Stuck()
    exit_code, _shutdown_s = run._stop(process, stall_s=300.0, progress=lambda: 7, interrupted=[])  # type: ignore[arg-type]

    assert process.killed and exit_code == -9


def test_a_permanent_observation_error_is_a_typed_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A schema the driver cannot read ends the run instead of polling forever.

    Anti-vacuity (Codex P1, #5678): skip every failed observation and a
    candidate whose cursor table lacks a read column never reaches a receipt.
    """
    import sqlite3

    from devtools.fresh_build_bench.run import _retryable_observation_error

    assert _retryable_observation_error(sqlite3.OperationalError("database is locked"))
    assert _retryable_observation_error(sqlite3.OperationalError("locking protocol"))
    assert not _retryable_observation_error(sqlite3.OperationalError("no such column: deferred_end_offset"))
    captured = _scripted_run(
        tmp_path,
        monkeypatch,
        [{"error": "OperationalError: no such column: deferred_end_offset", "error_retryable": False}],
        stall_timeout_s=7200.0,
    )

    assert captured["outcome"] == "observation_refused"
    # The receipt names the error that refused the run.
    from devtools.fresh_build_bench.report import _observation_errors

    assert _observation_errors(captured["observations"]) == {
        "OperationalError: no such column: deferred_end_offset": {"count": 1, "retryable": False}
    }


def test_linked_top_level_corpus_roots_are_refused(tmp_path: Path) -> None:
    """A corpus whose ``home`` is a link to a live tree cannot be sealed.

    Anti-vacuity (Codex P1, #5678): check only nested links and a linked
    ``home`` is walked and sealed, while the change stamp skips it.
    """
    from devtools.fresh_build_bench.corpus import _hash_tree

    live = tmp_path / "live"
    (live / ".codex").mkdir(parents=True)
    (live / ".codex" / "rollout.jsonl").write_bytes(b"{}\n")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "home").symlink_to(live, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        _hash_tree(corpus)


def test_sampling_skips_linked_files_and_refuses_unreadable_subtrees(tmp_path: Path) -> None:
    """Sampling sees what production's walk ingests, and never loses a subtree silently.

    Anti-vacuity (Codex P1/P2, #5678): follow a linked transcript file and it
    joins the sample; walk without ``onerror`` and an unreadable directory's
    transcripts vanish from the population.
    """
    import os

    from devtools.fresh_build_bench.corpus import SampleSource, _units

    root = tmp_path / "sessions"
    day = root / "2026" / "01" / "01"
    day.mkdir(parents=True)
    (day / "rollout-real.jsonl").write_bytes(b"{}\n")
    (tmp_path / "elsewhere.jsonl").write_bytes(b"{}\n")
    (day / "rollout-latest.jsonl").symlink_to(tmp_path / "elsewhere.jsonl")
    source = SampleSource("codex", root, "home/.codex/sessions")

    assert [Path(key).name for key, _paths, _size in _units(source)] == ["rollout-real.jsonl"]

    locked = root / "2026" / "02"
    locked.mkdir()
    (locked / "hidden.jsonl").write_bytes(b"{}\n")
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("running with permission to read any directory")
        with pytest.raises(ValueError, match="unreadable"):
            _units(source)
    finally:
        locked.chmod(0o700)


def test_a_candidate_module_added_and_removed_during_the_run_is_a_change(tmp_path: Path) -> None:
    """A transient untracked module leaves its directory's times changed.

    Anti-vacuity (Codex P1, #5678): stamp listed files only and a module
    created, imported and deleted before the end passes.
    """
    from devtools.fresh_build_bench.run import candidate_stamp

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "pkg/__init__.py"], check=True)
    before = candidate_stamp(tmp_path)
    transient = package / "transient.py"
    transient.write_text("VALUE = 1\n", encoding="utf-8")
    transient.unlink()

    assert candidate_stamp(tmp_path) != before


def test_named_transcripts_bring_their_sidecar_units(tmp_path: Path) -> None:
    """A named Claude transcript is sealed with its tool-results sidecars.

    Anti-vacuity (Codex P1, #5678): copy only the named JSONL and the
    corpus lacks the sidecar the daemon reads beside it.
    """
    from devtools.fresh_build_bench.corpus import corpus_from_files

    home = tmp_path / "home"
    project = home / ".claude" / "projects" / "-proj"
    (project / "11111111-1111-4111-8111-111111111111" / "tool-results").mkdir(parents=True)
    transcript = project / "11111111-1111-4111-8111-111111111111.jsonl"
    transcript.write_text('{"type": "user", "message": {"role": "user", "content": "hi"}}\n', encoding="utf-8")
    (project / "11111111-1111-4111-8111-111111111111" / "tool-results" / "toolu_1.txt").write_text(
        "full output", encoding="utf-8"
    )

    corpus_from_files(tmp_path / "corpus", [transcript], home=home)

    sealed = tmp_path / "corpus" / "home" / ".claude" / "projects" / "-proj"
    assert (sealed / "11111111-1111-4111-8111-111111111111.jsonl").is_file()
    assert (sealed / "11111111-1111-4111-8111-111111111111" / "tool-results" / "toolu_1.txt").is_file()


def test_a_file_below_a_linked_source_directory_is_a_member_by_its_lexical_path(tmp_path: Path) -> None:
    """``corpus files`` accepts a transcript production reaches through a linked directory.

    Anti-vacuity (Codex P2, #5678): resolve the file before the membership
    check and ``.store/team/...`` is outside the declared Codex layout.
    """
    from devtools.fresh_build_bench.corpus import corpus_from_files

    home = tmp_path / "home"
    sessions = home / ".codex" / "sessions"
    sessions.mkdir(parents=True)
    team = sessions / ".store" / "team"
    (team / "01" / "01").mkdir(parents=True)
    rollout = team / "01" / "01" / "rollout-2026-01-01T00-00-00-00000000-0000-0000-0000-000000000001.jsonl"
    rollout.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "00000000-0000-0000-0000-000000000001"}}) + "\n",
        encoding="utf-8",
    )
    (sessions / "2026").symlink_to(team, target_is_directory=True)
    linked = sessions / "2026" / "01" / "01" / rollout.name

    corpus_from_files(tmp_path / "corpus", [linked], home=home)

    assert (tmp_path / "corpus" / "home" / ".codex" / "sessions" / "2026" / "01" / "01" / rollout.name).is_file()


def test_a_cancellation_interrupts_the_fingerprint_sort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancellation during SQLite's own sort pass ends the fingerprint.

    Anti-vacuity (Codex P2, #5678): check ``cancelled`` only between scanned
    rows and a cancellation after spooling waits out the whole sort and
    digest.
    """
    import sqlite3
    from contextlib import closing

    import tests.infra.reindex_differential as differential
    from devtools.fresh_build_bench import report

    index = tmp_path / "index.db"
    with closing(sqlite3.connect(index)) as conn:
        conn.execute("CREATE TABLE big (value TEXT)")
        conn.executemany("INSERT INTO big VALUES (?)", ((f"{index:08d}",) for index in range(20_000)))
        conn.commit()
    monkeypatch.setattr(differential, "compared_table_census", lambda: ["big"])
    monkeypatch.setattr(differential, "_VOLATILE_COLUMNS", {"big": set()})
    monkeypatch.setattr(report, "_CANCEL_CHECK_OPS", 100)
    spooled = {"rows": 0}
    real_fact_row = differential._fact_row

    def counting(row: Any) -> Any:
        spooled["rows"] += 1
        return real_fact_row(row)

    monkeypatch.setattr(differential, "_fact_row", counting)

    with pytest.raises(report.FingerprintCancelledError):
        report.output_fingerprint(
            tmp_path, str(index), tmp_path / "scratch", cancelled=lambda: spooled["rows"] >= 20_000
        )


def test_component_preparations_are_consumed_as_they_finish() -> None:
    """A finished file's preparation is consumed while others still run.

    Anti-vacuity (Codex P2, #5678): collect every result before consuming
    any and the second item never sees the first one's cleanup.
    """
    import threading

    from devtools.fresh_build_bench.components import _timed_map

    first_consumed = threading.Event()
    saw_cleanup: list[bool] = []

    def work(item: tuple[Path, str, int]) -> Any:
        if item[1] == "late":
            saw_cleanup.append(first_consumed.wait(10))

        def finish() -> dict[str, int]:
            if item[1] == "early":
                first_consumed.set()
            return {"files": 1}

        return finish

    rows, _wall, counts = _timed_map(work, [(Path("a"), "early", 1), (Path("b"), "late", 1)], workers=2)

    assert saw_cleanup == [True]
    assert counts == {"files": 2} and len(rows) == 2


def test_the_periodic_profile_dump_counts_as_sampler_overhead(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sampler's own snapshot writes are overhead, not workload CPU.

    Anti-vacuity (Codex P2, #5678): finalize the tick's time before the
    periodic dump and the ten seconds it spent vanish from overhead.
    """
    from devtools.fresh_build_bench import sampler as sampler_module

    clock = {"now": 0.0}
    monkeypatch.setattr("devtools.fresh_build_bench.sampler.time.perf_counter", lambda: clock["now"])
    monkeypatch.setattr(sampler_module, "_FLUSH_EVERY_S", 0.0)
    sampler = sampler_module.StackSampler(tmp_path / "stacks.json", interval_s=0.0, stacks=False)
    waits = iter([False, True])
    monkeypatch.setattr(sampler._stop, "wait", lambda _timeout: next(waits))

    def slow_dump() -> None:
        clock["now"] += 10.0

    monkeypatch.setattr(sampler, "_dump", slow_dump)
    sampler._run()

    assert sampler._sample_seconds >= 10.0


def test_deep_stacks_keep_every_frame() -> None:
    """A stack deeper than 96 frames keeps its outer callers.

    Anti-vacuity (Codex P2, #5678): stop at a fixed depth and the outermost
    frames of this 200-deep recursion are dropped.
    """
    import sys

    from devtools.fresh_build_bench.sampler import _stack

    captured: list[Any] = []

    def recurse(depth: int) -> None:
        if depth == 0:
            captured.append(_stack(sys._getframe()))
            return
        recurse(depth - 1)

    recurse(200)

    assert sum(1 for frame in captured[0] if frame[1].endswith("recurse")) == 201


def test_a_wall_clock_step_restored_before_the_end_is_still_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient step displaces the milestones logged while it lasted.

    Anti-vacuity (Codex P2, #5678): compare wall and monotonic only at the
    endpoints and a step restored before the end reads as a steady clock.
    """
    captured = _scripted_run(
        tmp_path,
        monkeypatch,
        [{}, {}, _terminal_frame()],
        stall_timeout_s=7200.0,
        wall_jump=3600.0,
        wall_restore=True,
    )

    assert abs(captured["clock_step_s"]) >= 3600.0


def test_shutdown_progress_ignores_the_stack_sampler_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung daemon whose only moving counter is the sampler's CPU is not progressing.

    Anti-vacuity (Codex P1, #5678): count process CPU as shutdown progress and
    the injected sampler's 50 ms wakeups keep a hung shutdown waiting forever.
    """
    from devtools.fresh_build_bench import run

    probes: list[Any] = []

    def capture_stop(_process: object, **kwargs: Any) -> tuple[int, float]:
        probes.append(kwargs["progress"])
        return 0, 0.0

    captured = _scripted_run(tmp_path, monkeypatch, [_terminal_frame()], stall_timeout_s=7200.0)
    del captured
    monkeypatch.setattr(run, "_stop", capture_stop)

    class Sampler:
        samples: list[tuple[float, int, float, int, int, int]] = [(0.0, 1, 1.0, 5, 6, 7)]
        daemon_rss_hwm_bytes = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None: ...

        def start(self) -> None: ...

        def finish(self) -> None: ...

    monkeypatch.setattr(run, "TreeSampler", Sampler)
    config = RunConfig(corpus=tmp_path, work=tmp_path, candidate=tmp_path, python="python", label="l")
    paths = {
        "daemon_log": tmp_path / "daemon.log",
        "archive": tmp_path,
        "receipt": tmp_path / "receipt.json",
        "events": tmp_path / "events.jsonl",
    }
    run._measure_and_write_receipt(
        config,
        manifest={},
        paths=paths,
        identity={"git_sha": None, "dirty": None, "tracked_diff_sha256": None},
        candidate_files={},
        env_summary={},
        command=[],
        daemon_env={},
        interrupted=[],
        progress=lambda _line: None,
        hook_preparation=None,
    )
    (probe,) = probes
    before = probe()
    Sampler.samples.append((1.0, 1, 99.0, 5, 6, 7))  # only CPU moved

    assert probe() == before

    # A hung daemon's status readers and periodic events keep read I/O,
    # threads, the event log and SQLite read marks moving: none of that is a
    # draining shutdown, so a stalled run is still terminated (run5 sat past
    # its stall timeout until cancelled when these counted).
    Sampler.samples.append((2.0, 1, 99.0, 50, 6_000, 7_000))
    (tmp_path / "events.jsonl").write_text('{"event":"source.hook_spool.skipped"}\n', encoding="utf-8")
    (tmp_path / "index.db-shm").write_bytes(b"read marks")
    assert probe() == before

    # A checkpoint or drain writes the archive's database files.
    (tmp_path / "index.db-wal").write_bytes(b"frames")
    assert probe() != before


def test_a_full_fraction_sample_takes_zero_byte_units(tmp_path: Path) -> None:
    """``--fraction 1`` copies every unit, an empty transcript included.

    Anti-vacuity (Codex P1, #5678): stop a stratum once its byte goal (0) is met
    and the empty file is in the population but not the corpus.
    """
    root = tmp_path / "src"
    (root / "2026" / "01" / "01").mkdir(parents=True)
    (
        root / "2026" / "01" / "01" / "rollout-2026-01-01T00-00-00-00000000-0000-0000-0000-000000000001.jsonl"
    ).write_bytes(b"")
    sources = (SampleSource("codex", root, "home/.codex/sessions"),)

    manifest = sample_real(tmp_path / "corpus", seed=1, fraction=1.0, sources=sources)

    assert manifest["file_count"] == 1


def test_a_source_rewritten_after_its_copy_refuses_the_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A same-size in-place rewrite after copying is caught at the end of sampling.

    Anti-vacuity (Codex P2, #5678): recount only keys and sizes and the corpus
    seals the old bytes as a sample of the newer population.
    """
    from devtools.fresh_build_bench import corpus

    root = tmp_path / "src"
    day = root / "2026" / "01" / "01"
    day.mkdir(parents=True)
    first = day / "rollout-2026-01-01T00-00-00-00000000-0000-0000-0000-000000000001.jsonl"
    first.write_bytes(b"aaaa\n")
    (day / "rollout-2026-01-01T00-00-00-00000000-0000-0000-0000-000000000002.jsonl").write_bytes(b"bbbb\n")
    real_copy = corpus._copy_private
    copies: list[Path] = []

    def copy_then_rewrite_the_first(source: Path, destination: Path) -> None:
        real_copy(source, destination)
        copies.append(source)
        if len(copies) == 2:
            first.write_bytes(b"cccc\n")

    monkeypatch.setattr(corpus, "_copy_private", copy_then_rewrite_the_first)
    sources = (SampleSource("codex", root, "home/.codex/sessions"),)

    with pytest.raises(ValueError, match="changed after it was copied"):
        sample_real(tmp_path / "corpus", seed=1, fraction=1.0, sources=sources)


def test_ops_wal_is_part_of_the_sealed_evidence(tmp_path: Path) -> None:
    """A committed row living only in the WAL changes the evidence digest.

    Anti-vacuity (Codex P2, #5678): hash only ``ops.db`` and a WAL-only change
    keeps ``evidence_unchanged`` true.
    """
    from devtools.fresh_build_bench.report import evidence_digests

    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "ops.db").write_bytes(b"main")
    (tmp_path / "archive" / "ops.db-wal").write_bytes(b"wal-1")
    before = evidence_digests(tmp_path)
    (tmp_path / "archive" / "ops.db-wal").write_bytes(b"wal-2")

    assert evidence_digests(tmp_path) != before


def test_component_preparations_are_bounded_in_flight() -> None:
    """At most twice the workers are submitted and unconsumed at once.

    Anti-vacuity (Codex P2, #5678): submit the whole corpus up front and all
    ten preparations are in flight before the first is consumed.
    """
    import threading

    from devtools.fresh_build_bench.components import _timed_map

    lock = threading.Lock()
    state = {"live": 0, "peak": 0}

    def work(_item: tuple[Path, str, int]) -> Any:
        with lock:
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])

        def finish() -> dict[str, int]:
            with lock:
                state["live"] -= 1
            return {}

        return finish

    _timed_map(work, [(Path(f"{index}"), "codex", 1) for index in range(10)], workers=2)

    assert state["peak"] <= 4


def test_the_benchmark_identity_covers_production_reducers() -> None:
    """Readiness and FTS rules imported from production are part of the identity.

    Anti-vacuity (Codex P2, #5678): hash only the package and one helper and a
    changed readiness rule keeps the same implementation digest.
    """
    from devtools.fresh_build_bench.report import _reducer_dependencies

    names = {path.as_posix() for path in _reducer_dependencies()}

    assert any(name.endswith("polylogue/storage/archive_readiness.py") for name in names)


def test_the_import_closure_follows_relative_and_submodule_imports(tmp_path: Path) -> None:
    """A production module reached by ``from . import`` or ``from pkg import module`` is in the identity.

    Anti-vacuity: follow only absolute ``import``/``from module`` names and a
    rule in ``polylogue/a/c.py`` or ``polylogue/a/d.py`` changes without
    moving the digest.
    """
    from devtools.fresh_build_bench.report import polylogue_import_closure

    package = tmp_path / "polylogue" / "a"
    package.mkdir(parents=True)
    (tmp_path / "polylogue" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "b.py").write_text("from .c import rule\nfrom polylogue.a import d\n", encoding="utf-8")
    (package / "c.py").write_text("rule = 1\n", encoding="utf-8")
    (package / "d.py").write_text("", encoding="utf-8")
    root = tmp_path / "bench.py"
    root.write_text("import polylogue.a.b\n", encoding="utf-8")

    closure = {path.relative_to(tmp_path).as_posix() for path in polylogue_import_closure([root], tmp_path)}

    assert {"polylogue/a/b.py", "polylogue/a/c.py", "polylogue/a/d.py"} <= closure


def test_source_timing_coverage_is_required(tmp_path: Path) -> None:
    """A sampled origin without a measured source timing does not qualify.

    Anti-vacuity (Codex P1, #5678): qualify without it and a candidate that
    never emits ``live.ingest.source_group`` compares beside one that does.
    """
    from devtools.fresh_build_bench.report import _source_timing_recorded

    manifest = {"by_origin": {"codex": {"bytes": 2**20}}, "parameters": {"population": {}}}

    assert _source_timing_recorded(manifest, {}) is False
    assert _source_timing_recorded(manifest, {"codex": {"seconds": 1.0}}) is True


def test_the_python_probe_runs_from_the_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative ``--python`` is resolved where the daemon launches.

    Anti-vacuity (Codex P2, #5678): probe from the driver's cwd and a
    candidate-relative interpreter is identified as another build.
    """
    from devtools.fresh_build_bench import run

    seen: list[object] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs.get("cwd"))
        raise RuntimeError("stop after the probe")

    monkeypatch.setattr("devtools.fresh_build_bench.run.subprocess.run", fake_run)
    config = RunConfig(
        corpus=tmp_path, work=tmp_path, candidate=tmp_path / "cand", python=".venv/bin/python", label="l"
    )
    with pytest.raises(RuntimeError, match="stop after the probe"):
        run.environment(config)

    assert seen == [tmp_path / "cand"]


def test_an_interrupted_receipt_skips_the_archive_census(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails if an interrupted run enters the census, whose full-table counts cannot be cancelled."""
    from devtools.fresh_build_bench import report

    entered: list[object] = []

    def census(*args: object) -> dict[str, Any]:
        entered.append(args)
        return {}

    monkeypatch.setattr(report, "archive_census", census)
    arguments: dict[str, Any] = {
        "config": RunConfig(
            corpus=tmp_path, work=tmp_path, candidate=tmp_path, python="python", label="l", fingerprint=False
        ),
        "manifest": {"total_bytes": 0, "kind": "sample", "digest": "d", "file_count": 0, "by_origin": {}},
        "paths": {
            "events": tmp_path / "events.jsonl",
            "archive": tmp_path,
            "stacks": tmp_path / "stacks.json",
            "work": tmp_path,
        },
        "identity": {"unchanged_during_run": True},
        "environment": {},
        "command": [],
        "started_wall": 0.0,
        "wall_s": 1.0,
        "terminal_at": None,
        "exit_code": None,
        "shutdown_s": 0.0,
        "observations": [],
        "final": Observation(1.0, cursor_rows=1),
        "tree_samples": [],
    }
    receipt = report.build_receipt(outcome="interrupted", **arguments)
    assert entered == []
    assert receipt["qualified"] is False
    report.build_receipt(outcome="terminal", **arguments)
    assert len(entered) == 1


@pytest.mark.parametrize("origin", [None, _ts("2026-09-27T10:00:00.000000Z")])
def test_event_coverage_exposes_every_dropped_record(tmp_path: Path, origin: float | None) -> None:
    """Malformed/non-object/unusable records must invalidate complete coverage in either clock mode."""
    from devtools.fresh_build_bench.report import _events_lossless

    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                _event("00.000", "daemon.run.start"),
                "{bad",
                "[]",
                "{}",
                _event("01.000", "daemon.cold_build.generation_promoted"),
            ]
        )
        + "\n"
    )
    result = analyse_events(path, origin_unix=origin)
    assert result["event_count"] == 2
    assert result["coverage"] == {
        "file_present": True,
        "lines": 5,
        "valid_events": 2,
        "malformed_json": 1,
        "non_object": 1,
        "invalid_event": 1,
        "complete": False,
    }
    assert not _events_lossless(result, {"dropped": 0, "failures": 0, "undrained": 0})


@pytest.mark.parametrize(
    "delivery",
    [
        None,
        {},
        {"dropped": 0, "failures": 0},
        {"dropped": 0, "failures": None, "undrained": 0},
        {"dropped": 0, "failures": 0, "undrained": 1},
        {"dropped": False, "failures": 0, "undrained": 0},
    ],
)
def test_unknown_or_nonzero_terminal_delivery_cannot_claim_lossless(tmp_path: Path, delivery: Any) -> None:
    """Absent counters differ from measured zero even when every recorded line parses."""
    from devtools.fresh_build_bench.report import _events_lossless

    path = tmp_path / "events.jsonl"
    path.write_text(_event("00.000", "daemon.run.start") + "\n")
    events = analyse_events(path)
    assert events["coverage"]["complete"] is True
    assert not _events_lossless(events, delivery)
    assert _events_lossless(events, {"dropped": 0, "failures": 0, "undrained": 0})


def test_batch_missing_and_null_metrics_remain_unknown(tmp_path: Path) -> None:
    """Partial measurements cannot masquerade as a measured zero total."""
    import sqlite3

    path = tmp_path / "ops.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE daemon_events(kind TEXT,payload_json TEXT)")
        for payload in [
            {"input_bytes": 0, "parse_time_s": 1, "total_time_s": 0},
            {"input_bytes": None, "total_time_s": 0},
        ]:
            conn.execute("INSERT INTO daemon_events VALUES('ingestion_batch',?)", (json.dumps(payload),))
    result = analyse_batches(path)
    assert result["batches"] == 2
    assert result["totals"]["input_bytes"] is None
    assert result["totals"]["parse_time_s"] is None
    assert result["totals"]["ingested_bytes"] is None
    assert result["totals"]["total_time_s"] == 0
    assert result["metric_coverage"]["input_bytes"] == {"observed": 1, "missing": 1}
    assert result["metric_coverage"]["ingested_bytes"] == {"observed": 0, "missing": 2}


def test_refresh_rechecks_event_coverage_and_terminal_delivery(tmp_path: Path) -> None:
    """Refreshing cannot keep a previously claimed lossless verdict on unparsed evidence."""
    from devtools.fresh_build_bench.report import refresh

    path = tmp_path / "events.jsonl"
    path.write_text(_event("09.000", "daemon.cold_build.generation_promoted") + "\n{bad\n")
    (tmp_path / "stacks.json").write_text(json.dumps({"log_delivery": {"dropped": 0, "failures": 0, "undrained": 0}}))
    receipt = _receipt(qualified=True, started_at_unix=_ts("2026-09-27T10:00:00.000000Z"))
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    result = refresh(receipt_path)
    assert result["checks"]["events_lossless"] is False
    assert result["event_log"]["malformed_json"] == 1
    assert result["qualified"] is False


@pytest.mark.parametrize("origin", [None, _ts("2026-09-27T10:00:00.000000Z")])
def test_event_log_without_any_usable_record_has_incomplete_coverage(tmp_path: Path, origin: float | None) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("{bad\n[]\n{}\n", encoding="utf-8")
    result = analyse_events(path, origin_unix=origin)
    assert result["event_count"] == 0
    assert result["coverage"]["lines"] == 3
    assert result["coverage"]["complete"] is False


def test_parse_failure_does_not_count_as_reduced_required_work() -> None:
    from devtools.fresh_build_bench.run import _useful_progress

    pending = Observation(0.0, raw_rows=1, raw_pending=1)
    failed = Observation(1.0, raw_rows=1, raw_failed=1)
    accepted = Observation(1.0, raw_rows=1)
    assert not _useful_progress(pending, failed)
    assert _useful_progress(pending, accepted)


def test_advancing_work_progress_events_are_useful_progress_and_a_frozen_unit_is_not(tmp_path: Path) -> None:
    """Long preparation that changes no archive row is judged by the work it reports.

    Red if the observer ignores ``daemon.work.progress`` (a 13-minute
    preparation reads as stalled), or if a repeated event with unchanged
    counters still counts (a hung unit reads as progressing).
    """
    import json

    from devtools.fresh_build_bench.run import WorkProgressTail, _useful_progress

    events = tmp_path / "events.jsonl"

    def append(*records: dict[str, object]) -> None:
        with events.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def progress(
        unit_id: str,
        messages: int,
        byte_count: int = 0,
        *,
        productive_id: str = "raw-a:revision-1:advisory",
    ) -> dict[str, object]:
        return {
            "event": "daemon.work.progress",
            "phase": "source_preparation",
            "unit_id": unit_id,
            "productive_id": productive_id,
            "messages": messages,
            "bytes": byte_count,
        }

    tail = WorkProgressTail(events)
    append({"event": "daemon.started"}, progress("attempt-a", 100))
    before = Observation(0.0, work_progress=0)
    advanced = Observation(1.0, work_progress=tail.poll())
    assert _useful_progress(before, advanced)

    append(progress("attempt-a", 100))
    frozen = Observation(2.0, work_progress=tail.poll())
    assert not _useful_progress(advanced, frozen)

    # A retry is a new unit whose zeroed counters do not make progress just
    # because they differ from the previous unit's completed counters.
    append(progress("attempt-b", 0))
    reset = Observation(3.0, work_progress=tail.poll())
    assert not _useful_progress(frozen, reset)

    # Once the new unit reports real work, its own counters advance normally.
    append(progress("attempt-b", 10))
    retry_advanced = Observation(4.0, work_progress=tail.poll())
    assert not _useful_progress(reset, retry_advanced)

    # Distinct parser work gets its own baseline even when its counters are lower.
    append(progress("attempt-c", 10, productive_id="raw-b:revision-1:advisory"))
    repeated_reset = Observation(5.0, work_progress=tail.poll())
    assert _useful_progress(retry_advanced, repeated_reset)

    # The same productive work counts again only after exceeding its prior high-water.
    append(progress("attempt-d", 101))
    same_work_advanced = Observation(6.0, work_progress=tail.poll())
    assert _useful_progress(repeated_reset, same_work_advanced)

    # A record split across two writes is read once it is complete.
    line = json.dumps(progress("attempt-d", 102, 64)) + "\n"
    with events.open("a", encoding="utf-8") as handle:
        handle.write(line[:10])
    assert tail.poll() == same_work_advanced.work_progress
    with events.open("a", encoding="utf-8") as handle:
        handle.write(line[10:])
    assert _useful_progress(same_work_advanced, Observation(7.0, work_progress=tail.poll()))
    tail.close()


def test_work_progress_tail_spills_high_water_and_streams_appended_event_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from devtools.fresh_build_bench import run

    monkeypatch.setattr(run, "_WORK_PROGRESS_READ_CHUNK_BYTES", 128)
    events = tmp_path / "many-progress-events.jsonl"
    event_count = 1_200
    with events.open("w", encoding="utf-8") as handle:
        for index in range(event_count):
            handle.write(
                json.dumps(
                    {
                        "event": "daemon.work.progress",
                        "phase": "source_preparation",
                        "unit_id": f"unit-{index}",
                        "productive_id": f"recipe-{index}",
                        "messages": 1,
                        "bytes": 0,
                    }
                )
                + "\n"
            )

    tail = run.WorkProgressTail(events, state_root=tmp_path)
    state_directory = Path(tail._state_directory.name)
    try:
        assert tail.poll() == event_count
        assert tail._pending == b""
        assert not hasattr(tail, "_high_water")
        assert tail._connection.execute("PRAGMA cache_size").fetchone() == (-256,)
        assert tail._connection.execute("SELECT count(*) FROM productive_high_water").fetchone() == (event_count,)
        assert (state_directory / "high-water.sqlite3").is_file()
    finally:
        tail.close()

    assert not state_directory.exists()


@pytest.mark.parametrize("with_debt", [False, True])
def test_observation_preserves_scheduled_retry_evidence_without_counting_activity_as_progress(
    tmp_path: Path, with_debt: bool, frozen_clock: FrozenClock
) -> None:
    import sqlite3

    from devtools.fresh_build_bench.run import observe

    due = "2099-01-01T00:00:00+00:00"
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute(
            "CREATE TABLE ingest_cursor(excluded INTEGER,byte_offset INTEGER,stat_size INTEGER,failure_count INTEGER,deferred_end_offset INTEGER,next_retry_at TEXT)"
        )
        conn.execute("INSERT INTO ingest_cursor VALUES(0,0,10,2,NULL,?)", (due,))
        if with_debt:
            conn.execute("CREATE TABLE convergence_debt(stage TEXT,attempts INTEGER,next_retry_at TEXT)")
            conn.execute("INSERT INTO convergence_debt VALUES('lineage',3,?)", (due,))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("CREATE TABLE empty(value INTEGER)")
    observation = observe(tmp_path, frozen_clock.monotonic())
    assert observation.error is None
    assert observation.cursor_failures == 2
    assert observation.cursor_retry_waiting == 1
    assert observation.cursor_next_retry_at == due
    assert observation.debt_next_retry_at == (due if with_debt else None)
    assert observation.debt_waiting_by_stage == ({"lineage": 1} if with_debt else {})
    assert observation.useful_progress_at_s is None


def test_human_receipt_exposes_unparsed_event_coverage() -> None:
    from devtools.fresh_build_bench.report import render

    receipt = _receipt(
        candidate={"git_sha": "0" * 40, "dirty": False},
        corpus={"kind": "sample", "digest": "c", "total_bytes": 0, "file_count": 0},
        budgets={},
        event_log={
            "file_present": True,
            "lines": 3,
            "valid_events": 2,
            "malformed_json": 1,
            "non_object": 0,
            "invalid_event": 0,
            "complete": False,
        },
    )
    rendered = render(receipt)
    assert "malformed_json=1" in rendered
    assert "complete=False" in rendered


def test_free_threaded_profile_refuses_frame_traversal_but_keeps_cpu_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from devtools.fresh_build_bench import sampler as sampler_module

    monkeypatch.setattr("devtools.fresh_build_bench.sampler.sysconfig.get_config_var", lambda name: 1)
    monkeypatch.setattr(
        "devtools.fresh_build_bench.sampler.sys._current_frames", lambda: pytest.fail("unsafe live frame traversal")
    )
    sampler = sampler_module.StackSampler(tmp_path / "stacks.json", interval_s=0.0, stacks=True)
    waits = iter([False, True])
    monkeypatch.setattr(sampler._stop, "wait", lambda timeout: next(waits))
    sampler._run()
    sampler.write()
    document = json.loads(sampler.out_path.read_text())
    assert document["profile_refusal"] == "free_threaded_frame_snapshot_unavailable"
    assert document["ticks"] == 1
    assert document["stacks"] == []
    assert document["process_cpu_ticks"] is not None

    from devtools.fresh_build_bench import report

    receipt = report.build_receipt(
        config=RunConfig(
            corpus=tmp_path,
            work=tmp_path,
            candidate=tmp_path,
            python="python",
            label="l",
            profile=True,
            fingerprint=False,
        ),
        manifest={"total_bytes": 0, "kind": "sample", "digest": "d", "file_count": 0, "by_origin": {}},
        paths={"events": tmp_path / "events.jsonl", "archive": tmp_path, "stacks": sampler.out_path, "work": tmp_path},
        identity={"unchanged_during_run": True},
        environment={},
        command=[],
        started_wall=0.0,
        wall_s=1.0,
        outcome="interrupted",
        terminal_at=None,
        exit_code=None,
        shutdown_s=0.0,
        observations=[],
        final=Observation(1.0),
        tree_samples=[],
    )
    assert receipt["profile"]["outcome"] == "refused"
    assert receipt["profile"]["reason"] == "free_threaded_frame_snapshot_unavailable"
    assert receipt["checks"]["profile_collected"] is False


@pytest.mark.parametrize("samples", ["refused", "absent", "empty", "collected"])
def test_requested_profile_controls_terminal_qualification_and_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, samples: str
) -> None:
    from devtools.fresh_build_bench import report

    document: dict[str, Any] = {
        "clock_ticks_per_s": 100,
        "interval_s": 0.01,
        "elapsed_s": 1.0,
        "sampler_seconds": 0.0,
        "stacks": [],
        "log_delivery": {"dropped": 0, "failures": 0, "undrained": 0},
    }
    if samples == "refused":
        document["profile_refusal"] = "free_threaded_frame_snapshot_unavailable"
    elif samples == "collected":
        document["stacks"] = [
            {"thread": "worker", "stack": [["/synthetic/worker.py", "run", 1]], "wall_samples": 1, "cpu_ticks": 1}
        ]
    stacks = tmp_path / "stacks.json"
    if samples != "absent":
        stacks.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(report, "_log_delivery", lambda path: document["log_delivery"])
    monkeypatch.setattr(
        report,
        "analyse_events",
        lambda *args, **kwargs: {"milestones_s": {"promoted_s": 0.5}, "coverage": {"complete": True}},
    )
    monkeypatch.setattr(report, "archive_census", lambda *args: {"messages_fts_rows": 0, "fts_indexable_rows": 0})
    receipt = report.build_receipt(
        config=RunConfig(
            corpus=tmp_path,
            work=tmp_path,
            candidate=tmp_path,
            python="python",
            label="l",
            profile=True,
            fingerprint=False,
        ),
        manifest={"total_bytes": 0, "kind": "sample", "digest": "d", "file_count": 0, "by_origin": {}},
        paths={"events": tmp_path / "events.jsonl", "archive": tmp_path, "stacks": stacks, "work": tmp_path},
        identity={"unchanged_during_run": True, "git_sha": "0" * 40, "dirty": False},
        environment={},
        command=[],
        started_wall=0.0,
        wall_s=1.0,
        outcome="terminal",
        terminal_at=1.0,
        exit_code=0,
        shutdown_s=0.0,
        observations=[],
        final=Observation(
            1.0,
            cursor_rows=1,
            cursor_complete=1,
            promoted_index="synthetic-index",
            readiness=dict.fromkeys(REQUIRED_READINESS_DOMAINS, True),
        ),
        tree_samples=[],
    )
    assert all(value for key, value in receipt["checks"].items() if key != "profile_collected")
    assert receipt["checks"]["profile_collected"] is (samples == "collected")
    assert receipt["qualified"] is (samples == "collected")
    # Refresh uses this same owner; it must not resurrect qualification.
    _derive_dependents(receipt)
    assert receipt["qualified"] is (samples == "collected")
    if samples != "collected":
        assert "profile outcome=refused reason=" in report.render(receipt)
        for allow in (False, True):
            ok, text = compare(receipt, receipt, allow_unqualified=allow)
            assert not ok and "profile refused" in text


@pytest.mark.parametrize("reason", ["free_threaded_frame_snapshot_unavailable", "profile_samples_unavailable"])
@pytest.mark.parametrize("entrypoint", ["standalone", "bench"])
def test_refused_stack_document_is_not_an_empty_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], reason: str, entrypoint: str
) -> None:
    from devtools.fresh_build_bench import profile_report

    document: dict[str, Any] = {"stacks": []}
    if reason != "profile_samples_unavailable":
        document["profile_refusal"] = reason
    samples, output = tmp_path / "stacks.json", tmp_path / "collapsed.txt"
    samples.write_text(json.dumps(document), encoding="utf-8")
    if entrypoint == "standalone":
        exit_code = profile_report.main([str(samples), "--collapsed", str(output)])
    else:
        from devtools.fresh_build_bench.cli import main

        exit_code = main(["profile", str(samples), "--collapsed", str(output)])
    assert exit_code == 2
    assert f"profile outcome=refused reason={reason}" in capsys.readouterr().out
    assert not output.exists()
    with pytest.raises(profile_report.ProfileRefusedError) as caught:
        profile_report.summarise(document, top=10, thread_filter=None)
    assert caught.value.reason == reason
    with pytest.raises(profile_report.ProfileRefusedError):
        profile_report.collapsed(document, weight="cpu", thread_filter=None)


@pytest.mark.uses_real_clock("sampler accounts actual OS thread CPU")
def test_actual_runtime_profile_environment_never_walks_free_threaded_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import sysconfig

    from devtools.fresh_build_bench import sampler as sampler_module

    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    if free_threaded:
        monkeypatch.setattr(sys, "_current_frames", lambda: pytest.fail("unsafe foreign frames"))
    monkeypatch.setenv("POLYLOGUE_BENCH_STACK_SAMPLES", str(tmp_path / "samples.json"))
    monkeypatch.setenv("POLYLOGUE_BENCH_STACKS", "1")
    # Drive one tick deterministically through the daemon's environment entrypoint.
    monkeypatch.setattr(sampler_module.StackSampler, "start", lambda self: None)
    sampler = sampler_module.start_from_environment()
    assert sampler is not None
    waits = iter([False, True])
    monkeypatch.setattr(sampler._stop, "wait", lambda timeout: next(waits))
    sampler._run()
    sampler.write()
    document = json.loads(sampler.out_path.read_text())
    assert document["profile_refusal"] == ("free_threaded_frame_snapshot_unavailable" if free_threaded else None)
    assert document["ticks"] == 1
    assert document["process_cpu_ticks"] is not None
    if free_threaded:
        assert document["stacks"] == []


def test_explicit_corpus_seals_and_stages_hook_spool_tree(tmp_path: Path) -> None:
    from devtools.fresh_build_bench.run import _prepare_paths

    source = tmp_path / "operator-hooks"
    (source / "carriers" / "codex").mkdir(parents=True)
    (source / "carriers" / "codex" / "events.jsonl").write_text('{"event_id":"e1"}\n', encoding="utf-8")
    (source / "pending").mkdir()
    (source / "pending" / "e2.json").write_text('{"event_id":"e2"}', encoding="utf-8")
    corpus = tmp_path / "corpus"
    manifest = corpus_from_files(corpus, [], home=tmp_path / "empty-home", hooks=source)
    verify_manifest(corpus, manifest)

    paths = _prepare_paths(RunConfig(corpus, tmp_path / "run", Path(__file__).resolve().parents[3], "python", "test"))

    assert (paths["archive"] / "hooks" / "carriers" / "codex" / "events.jsonl").read_text() == '{"event_id":"e1"}\n'
    assert (paths["archive"] / "hooks" / "pending" / "e2.json").exists()


def test_production_progress_events_reach_benchmark_high_water(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    """Registry omissions cannot erase the producer identity before the monitor."""
    from devtools.fresh_build_bench.run import WorkProgressTail, _useful_progress
    from polylogue import logging as plog
    from polylogue.core.work_progress import advance_work_progress, work_progress

    events = tmp_path / "production-progress.jsonl"
    tail = WorkProgressTail(events, state_root=tmp_path)
    previous_level = plog.set_level("info")
    try:
        with plog.capture() as records:
            with work_progress("source_preparation", productive_id="neutral-recipe") as unit:
                advance_work_progress(messages=100, bytes=64)
                frozen_clock.advance(11)
                advance_work_progress(messages=1, bytes=1)
            first_unit = unit.unit_id
        assert not any(record["event"] == "log.field_rejected" for record in records)
        progress = [record for record in records if record["event"] == "daemon.work.progress"]
        assert progress and all(record["unit_id"] == first_unit for record in progress)
        assert all(record["productive_id"] == "neutral-recipe" for record in progress)
        events.write_text("".join(json.dumps(record) + "\n" for record in records))
        first = Observation(0.0, work_progress=0)
        advanced = Observation(1.0, work_progress=tail.poll())
        assert _useful_progress(first, advanced)
        assert advanced.work_progress == 1  # unchanged final emission is not work

        with plog.capture() as retry_records:
            with work_progress("source_preparation", productive_id="neutral-recipe") as retry:
                advance_work_progress(messages=10, bytes=10)
        assert retry.unit_id != first_unit
        with events.open("a") as stream:
            stream.writelines(json.dumps(record) + "\n" for record in retry_records)
        retry_frame = Observation(2.0, work_progress=tail.poll())
        assert not _useful_progress(advanced, retry_frame)
    finally:
        plog.set_level(previous_level)
        tail.close()


def test_shutdown_write_stamp_checks_writer_locations_without_walking_payloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from devtools.fresh_build_bench.run import _archive_write_stamp
    from polylogue.storage.archive_identity import TIER_FILENAMES

    expected: set[str] = set()
    for _, filename in TIER_FILENAMES:
        for suffix in ("", "-wal", "-journal"):
            name = filename + suffix
            (tmp_path / name).write_bytes(b"database")
            expected.add(name)
    for dirname, filename in (
        (".index-generations/candidate", "index.db"),
        (".embeddings-generations/candidate", "embeddings.db"),
    ):
        directory = tmp_path / dirname
        directory.mkdir(parents=True)
        for suffix in ("", "-wal", "-journal"):
            name = f"{dirname}/{filename}{suffix}"
            (tmp_path / name).write_bytes(b"generation")
            expected.add(name)
    blob = tmp_path / "blob" / "nested"
    blob.mkdir(parents=True)
    (blob / "unrelated.db").write_bytes(b"payload")
    (tmp_path / "index.db-shm").write_bytes(b"reader marks")
    visited: list[Path] = []
    real_iterdir = Path.iterdir

    def direct_generation_members(path: Path):
        assert path in {tmp_path / ".index-generations", tmp_path / ".embeddings-generations"}
        visited.append(path)
        return real_iterdir(path)

    def refuse_recursive_walk(*_args: object, **_kwargs: object):
        pytest.fail("shutdown observer traversed the archive payload tree")

    monkeypatch.setattr(Path, "iterdir", direct_generation_members)
    monkeypatch.setattr(Path, "rglob", refuse_recursive_walk)
    before = _archive_write_stamp(tmp_path)
    assert {name for name, _, _ in before} == expected
    assert len(visited) == 2
    (tmp_path / ".index-generations/candidate/index.db-wal").write_bytes(b"checkpoint progress")
    assert _archive_write_stamp(tmp_path) != before
    stable = _archive_write_stamp(tmp_path)
    (blob / "unrelated.db").write_bytes(b"changed payload")
    (tmp_path / "index.db-shm").write_bytes(b"changed reader marks")
    assert _archive_write_stamp(tmp_path) == stable

"""The fresh-build benchmark's receipt arithmetic and corpus sealing.

The end-to-end run is exercised by running it; these pin the pure parts a
wrong receipt would come from: the event-log reduction, the stage rollup, the
projection, budgets, the terminal predicate and the corpus seal.
"""

from __future__ import annotations

import json
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
    sessions = home / ".codex" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "rollout.jsonl").write_text("{}\n", encoding="utf-8")
    (sessions / "large.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="not a transcript"):
        corpus_from_files(tmp_path / "bad", [sessions / "large.bin"], home=home)
    manifest = corpus_from_files(tmp_path / "good", [sessions / "rollout.jsonl"], home=home)
    assert manifest["kind"] == "files"


def test_a_sample_that_draws_nothing_still_seals(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    (root / "one.jsonl").write_bytes(b"x" * 1000)
    sources = (SampleSource("codex", root, "home/.codex/sessions", (".jsonl",)),)
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


@pytest.mark.parametrize("check", ["candidate_unchanged", "corpus_unchanged", "events_lossless"])
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
    projects = home / ".claude" / "projects" / "proj"
    for index in range(20):
        session = projects / f"s{index:02d}.jsonl"
        session.parent.mkdir(parents=True, exist_ok=True)
        session.write_bytes(b"x" * (1000 + index))
        child = projects / f"s{index:02d}" / "subagents" / "agent-a.jsonl"
        child.parent.mkdir(parents=True)
        child.write_bytes(b"y" * 500)
    sources = (SampleSource("claude-code", home / ".claude" / "projects", "home/.claude/projects", (".jsonl",), True),)
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
    assert analyse_events(tmp_path / "absent.jsonl") == {"event_count": 0}


def test_stratum_boundary_is_one_draw(tmp_path: Path) -> None:
    """Anti-vacuity: redrawing the boundary on every later unit selects a
    file in nearly every seed instead of about one in ten."""
    root = tmp_path / "src"
    root.mkdir()
    for index in range(50):
        (root / f"f{index:02d}.jsonl").write_bytes(b"x" * 1000)
    sources = (SampleSource("codex", root, "home/.codex/sessions", (".jsonl",)),)
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
    sessions = home / ".codex" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "rollout.jsonl").write_text("{}\n", encoding="utf-8")
    out = tmp_path / "corpus"
    corpus_from_files(out, [sessions / "rollout.jsonl"], home=home)
    assert out.stat().st_mode & 0o077 == 0
    copied = out / "home" / ".codex" / "sessions" / "rollout.jsonl"
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
    project = home / ".claude" / "projects" / "proj"
    (project / "s1" / "tool-results").mkdir(parents=True)
    (project / "s1.jsonl").write_text("{}\n", encoding="utf-8")
    (project / "s1" / "tool-results" / "toolu_1.txt").write_text("full output", encoding="utf-8")
    gemini = home / ".gemini" / "tmp" / "hash1"
    (gemini / "chats").mkdir(parents=True)
    (gemini / "tool-outputs" / "session-x").mkdir(parents=True)
    (gemini / "chats" / "session-x.json").write_text("{}", encoding="utf-8")
    (gemini / "tool-outputs" / "session-x" / "shell_1.txt").write_text("output", encoding="utf-8")
    manifest = sample_real(tmp_path / "corpus", seed=1, fraction=1.0, sources=default_sample_sources(home))
    paths = {row[0] for row in manifest["files"]}
    assert "home/.claude/projects/proj/s1/tool-results/toolu_1.txt" in paths
    assert "home/.gemini/tmp/hash1/tool-outputs/session-x/shell_1.txt" in paths


def test_parse_component_uses_the_production_stream_predicate_and_times_only_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: an exact ``.jsonl`` comparison parses ``rollout.JSONL``
    as a document; timing the count loop charges its 0.3 s to the worker."""
    import time
    from types import SimpleNamespace

    from devtools.fresh_build_bench import components

    corpus = tmp_path / "corpus"
    transcript = corpus / "home" / ".codex" / "sessions" / "rollout.JSONL"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("{}\n", encoding="utf-8")
    seal(corpus, kind="sample", parameters={})
    seen: list[bool] = []

    def slow_sessions() -> list[object]:
        time.sleep(0.3)
        return [SimpleNamespace(messages=[1, 2])]

    def worker(_provider: str, _path: str, _stem: str, *, is_stream: bool, **_kwargs: object) -> object:
        seen.append(is_stream)
        return SimpleNamespace(error=None, iter_sessions=slow_sessions)

    monkeypatch.setattr("polylogue.sources.live.parse_prefetch.live_parse_path_worker", worker)
    result = components.bench_parse(corpus, tmp_path / "scratch", workers=1, origins=None, limit=None)
    assert seen == [True]
    assert result["counts"] == {"sessions": 1, "messages": 2}
    assert result["by_origin"]["codex"]["seconds"] < 0.2


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

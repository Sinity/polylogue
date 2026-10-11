"""Synthetic controls for complete named and sampled production inputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from devtools.fresh_build_bench import corpus
from devtools.fresh_build_bench.report import projection, render
from polylogue.sources.live import discovery
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.retained_assembly import claude_code_sidecar_coordinates, codex_sidecar_coordinates

SID = "11111111-1111-4111-8111-111111111111"


def put(path: Path, text: str = "{}\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def claude(home: Path) -> Path:
    project = home / ".claude/projects/-project"
    main = put(project / f"{SID}.jsonl", '{"type":"user","message":{"role":"user","content":"neutral"}}\n')
    put(project / SID / "subagents/agent-child.jsonl", main.read_text())
    put(project / SID / "subagents/agent-child.meta.json")
    sidecar = put(project / SID / "tool-results/toolu_1.txt", "neutral tool output")
    os.utime(sidecar, ns=(123000000000, 123000000000))
    put(project / "sessions-index.json", '{"entries":[{"sessionId":"neutral"}]}')
    put(home / ".claude/history.jsonl", '{"sessionId":"unselected","display":"full history"}\n')
    return main


@pytest.mark.parametrize("named_child", [False, True])
def test_named_claude_native_unit_and_full_metadata_without_unrelated_descent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named_child: bool
) -> None:
    home = tmp_path / "home"
    main = claude(home)
    unrelated = put(main.parent / "22222222-2222-4222-8222-222222222222/subagents/agent-other.jsonl")
    put(home / ".claude/projects/-other/33333333-3333-4333-8333-333333333333.jsonl")
    visited = []
    original = discovery._ordered_children

    def observe(source: WatchSource, directory: Path, *args: Any, **kwargs: Any) -> list[tuple[str, Path, bool]]:
        visited.append(directory)
        assert directory != unrelated.parent and "-other" not in directory.parts
        return original(source, directory, *args, **kwargs)

    monkeypatch.setattr(discovery, "_ordered_children", observe)
    named = main.parent / SID / "subagents/agent-child.jsonl" if named_child else main
    out = tmp_path / "corpus"
    manifest = corpus.corpus_from_files(out, [named], home=home)
    copied = out / "home/.claude/projects/-project" / main.name
    assert copied.is_file()
    assert (copied.parent / SID / "subagents/agent-child.meta.json").is_file()
    assert (copied.parent / SID / "tool-results/toolu_1.txt").stat().st_mtime_ns == 123000000000
    coordinates = claude_code_sidecar_coordinates(str(copied))
    assert coordinates is not None
    assert all(Path(path).is_file() for path in coordinates)
    assert manifest["file_count"] == 6
    assert set(visited) == {
        main.parent,
        main.parent / SID,
        main.parent / SID / "subagents",
        main.parent / SID / "tool-results",
    }


@pytest.mark.parametrize("origin", ["claude-code", "codex"])
def test_named_alias_coordinates_follow_production_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    home = tmp_path / "home"
    if origin == "claude-code":
        physical = home / ".config/claude/projects"
        physical.mkdir(parents=True)
        (home / ".claude").mkdir()
        (home / ".claude/projects").symlink_to(physical, target_is_directory=True)
        main = claude(home)
        # Production history resolves across the watched-root link before '..'.
        put(home / ".config/claude/history.jsonl", "actual install history\n")
        resolver = claude_code_sidecar_coordinates
    else:
        sessions = home / ".codex/sessions"
        main = put(
            sessions / ".store/team/01/01/rollout-neutral.jsonl", '{"type":"session_meta","payload":{"id":"neutral"}}\n'
        )
        (sessions / "2026").symlink_to(sessions / ".store/team", target_is_directory=True)
        main = sessions / "2026/01/01" / main.name
        put(home / ".codex/session_index.jsonl", "full index\n")
        put(home / ".codex/history.jsonl", "full history\n")
        resolver = codex_sidecar_coordinates
    out = tmp_path / "corpus"
    corpus.corpus_from_files(out, [main], home=home)
    copied = out / "home" / main.relative_to(home)
    assert copied.is_file()
    original_coordinates = resolver(str(main))
    copied_coordinates = resolver(str(copied))
    assert original_coordinates is not None and copied_coordinates is not None
    if origin == "claude-code":
        from polylogue.sources import assembly_claude_code

        read_coordinates = []

        def observe(path: Path) -> dict[str, Any]:
            read_coordinates.append(path)
            return {}

        monkeypatch.setattr(assembly_claude_code, "parse_sessions_index", observe)
        monkeypatch.setattr(assembly_claude_code, "build_session_paste_index", observe)
        spec = assembly_claude_code.ClaudeCodeAssemblySpec()
        spec.discover_sidecars([main])
        spec.discover_sidecars([copied])
        assert read_coordinates == [Path(path) for path in (*original_coordinates, *copied_coordinates)]
    for source, target in zip(original_coordinates, copied_coordinates, strict=True):
        assert Path(target).read_bytes() == Path(source).read_bytes()
    assert not (out / "home/.config").exists()
    assert not (out / "home/.codex/sessions/.store").exists()


def test_named_codex_no_population_walk_and_optional_absence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    main = put(
        home / ".codex/sessions/2026/01/01/rollout-neutral.jsonl",
        '{"type":"session_meta","payload":{"id":"neutral"}}\n',
    )
    monkeypatch.setattr(corpus, "_units", lambda *args: pytest.fail("named route enumerated population"))
    manifest = corpus.corpus_from_files(tmp_path / "corpus", [main], home=home)
    assert manifest["file_count"] == 1


def test_named_gemini_keeps_complete_project(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = home / ".gemini/tmp/project"
    main = put(project / "chats/session-neutral.json", json.dumps({"sessionId": "neutral", "messages": []}))
    put(project / "logs.json")
    put(project / "tool-outputs/session-neutral/result.txt", "full output")
    put(home / ".gemini/tmp/unselected/chats/session-other.json")
    manifest = corpus.corpus_from_files(tmp_path / "corpus", [main], home=home)
    assert manifest["file_count"] == 3
    assert not (tmp_path / "corpus/home/.gemini/tmp/unselected").exists()


@pytest.mark.parametrize("change", ["appears", "rewritten", "linked"])
def test_selected_dependency_absence_currentness_and_link_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    home = tmp_path / "home"
    main = put(
        home / ".codex/sessions/2026/01/01/rollout-neutral.jsonl",
        '{"type":"session_meta","payload":{"id":"neutral"}}\n',
    )
    history = home / ".codex/history.jsonl"
    if change != "appears":
        put(history, "before\n")
    if change == "linked":
        history.unlink()
        history.symlink_to(put(tmp_path / "linked-history"))
    original = corpus._copy_private

    def copy(path: Path, destination: Path) -> None:
        original(path, destination)
        if path == main:
            put(history, "after\n")

    monkeypatch.setattr(corpus, "_copy_private", copy)
    with pytest.raises(ValueError, match="dependency|closure|unlinked"):
        corpus.corpus_from_files(tmp_path / "corpus", [main], home=home)


def test_named_escaping_selected_ancestor_refused(tmp_path: Path) -> None:
    home = tmp_path / "home"
    sessions = home / ".codex/sessions"
    sessions.mkdir(parents=True)
    external = put(
        tmp_path / "external/01/01/rollout-neutral.jsonl", '{"type":"session_meta","payload":{"id":"neutral"}}\n'
    )
    (sessions / "2026").symlink_to(external.parents[2], target_is_directory=True)
    with pytest.raises(ValueError, match="refused"):
        corpus.corpus_from_files(tmp_path / "corpus", [sessions / "2026/01/01" / external.name], home=home)


def test_sample_original_draws_shared_metadata_once_and_projection_unknown(tmp_path: Path) -> None:
    home = tmp_path / "home"
    main = claude(home)
    put(main.parent / "22222222-2222-4222-8222-222222222222.jsonl", main.read_text())
    sources = corpus.default_sample_sources(home)[:1]
    without = tmp_path / "without"
    history = home / ".claude/history.jsonl"
    content = history.read_bytes()
    history.unlink()
    first = corpus.sample_real(without, seed=7, fraction=0.5, sources=sources)
    history.write_bytes(content)
    with_metadata = corpus.sample_real(tmp_path / "with", seed=7, fraction=0.5, sources=sources)
    drawn_first = first["parameters"]["drawn_units"]["claude-code"]
    drawn_second = with_metadata["parameters"]["drawn_units"]["claude-code"]
    assert (drawn_first["units"], drawn_first["bytes"]) == (drawn_second["units"], drawn_second["bytes"])
    assert first["parameters"]["population"] == with_metadata["parameters"]["population"]
    assert drawn_second["shared_dependency_bytes"] >= len(content)
    paths = [row[0] for row in with_metadata["files"]]
    assert paths.count("home/.claude/history.jsonl") == 1
    measured = projection(with_metadata, {"claude-code": {"seconds": 2}}, 2)
    assert measured["projected_intake_hours"] is None
    assert measured["unknown_denominator_origins"] == ["claude-code"]
    assert measured["by_origin"]["claude-code"]["seconds"] == 2
    rendered = render(
        {
            "label": "synthetic projection control",
            "outcome": "passed",
            "candidate": {"git_sha": "0" * 40, "dirty": False},
            "corpus": with_metadata,
            "timing_s": {},
            "throughput": {},
            "projection": measured,
        }
    )
    assert "projection unavailable for claude-code" in rendered
    assert "full shared assembly metadata has no sampling denominator" in rendered


def test_selected_real_and_full_sample_share_named_alias_closure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    physical = home / ".config/claude/projects"
    physical.mkdir(parents=True)
    (home / ".claude").mkdir()
    (home / ".claude/projects").symlink_to(physical, target_is_directory=True)
    main = claude(home)
    put(home / ".config/claude/history.jsonl", "correct shared history\n")
    source = corpus.default_sample_sources(home)[:1]
    named = corpus.corpus_from_files(tmp_path / "named", [main], home=home)
    selected = corpus.select_real(
        tmp_path / "selected", seed=2, max_bytes=10000, max_unit_bytes=10000, max_units=1, sources=source
    )
    sampled = corpus.sample_real(tmp_path / "sampled", seed=2, fraction=1, sources=source)
    assert named["files"] == selected["files"] == sampled["files"]
    assert sampled["parameters"]["drawn_units"]["claude-code"]["shared_dependency_bytes"] == len(
        "correct shared history\n"
    )


def test_selected_native_fault_refuses_but_unrelated_link_is_not_handled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    main = claude(home)
    (main.parent / "22222222-2222-4222-8222-222222222222").symlink_to(tmp_path / "absent", target_is_directory=True)
    corpus.corpus_from_files(tmp_path / "good", [main], home=home)
    original = discovery._ordered_children

    def unavailable(source: WatchSource, directory: Path, *args: Any, **kwargs: Any) -> list[tuple[str, Path, bool]]:
        if directory == main.parent / SID / "subagents":
            from polylogue.sources.walk_faults import WalkFault, WalkRefusedError

            raise WalkRefusedError("synthetic selected refusal", [WalkFault(directory, "unreadable")])
        return original(source, directory, *args, **kwargs)

    monkeypatch.setattr(discovery, "_ordered_children", unavailable)
    with pytest.raises(ValueError, match="refused"):
        corpus.corpus_from_files(tmp_path / "bad", [main], home=home)


def test_named_gemini_same_key_projects_match_original_unit_without_other_descent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    root = home / ".gemini/tmp"
    main = put(root / "project/chats/session-neutral.json", json.dumps({"sessionId": "neutral", "messages": []}))
    sibling = put(root / "project.extra/chats/session-extra.json", json.dumps({"sessionId": "extra", "messages": []}))
    put(root / "unrelated/chats/session-unrelated.json")
    source = corpus.default_sample_sources(home)[2]
    original_unit = next(paths for key, paths, _size in corpus._units(source) if main in paths)
    assert set(original_unit) == {main, sibling}
    original = discovery._ordered_children
    visited: list[Path] = []

    def observe(source: WatchSource, directory: Path, *args: Any, **kwargs: Any) -> list[tuple[str, Path, bool]]:
        visited.append(directory)
        assert "unrelated" not in directory.parts
        return original(source, directory, *args, **kwargs)

    monkeypatch.setattr(discovery, "_ordered_children", observe)
    out = tmp_path / "corpus"
    manifest = corpus.corpus_from_files(out, [main], home=home)
    assert {out / row[0] for row in manifest["files"]} == {
        out / "home" / path.relative_to(home) for path in original_unit
    }
    assert root in visited and root / "project.extra" in visited

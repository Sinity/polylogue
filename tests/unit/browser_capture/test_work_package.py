from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from polylogue.browser_capture.work_package import build_sol_pro_work_package


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "CLAUDE.md").write_text("# Instructions\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "ignored.txt").write_text("not selected\n", encoding="utf-8")
    (repo / ".beads").mkdir()
    (repo / ".beads" / "issues.jsonl").write_text(
        json.dumps({"id": "polylogue-test", "title": "Test", "notes": "full note"}) + "\n",
        encoding="utf-8",
    )
    return repo


def test_targeted_work_package_is_deterministic_and_manifest_checked(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = build_sol_pro_work_package(
        repo_root=repo,
        scope_prompt="Implement the selected worker safely.",
        bead_ids=("polylogue-test",),
        source_paths=(Path("src/worker.py"),),
    )
    second = build_sol_pro_work_package(
        repo_root=repo,
        scope_prompt="Implement the selected worker safely.",
        bead_ids=("polylogue-test",),
        source_paths=(Path("src/worker.py"),),
    )

    assert first.content == second.content
    assert first.sha256 == hashlib.sha256(first.content).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(first.content), mode="r:gz") as archive:
        names = set(archive.getnames())
        assert "REPO/src/worker.py" in names
        assert "ignored.txt" not in names
        assert "PROMPT.md" in names
        assert "BEADS/selected.json" in names
        manifest = json.load(archive.extractfile("MANIFEST.json"))  # type: ignore[arg-type]
        records = {item["path"]: item for item in manifest["files"]}
        for name in names - {"MANIFEST.json"}:
            content = archive.extractfile(name).read()  # type: ignore[union-attr]
            assert records[name]["size_bytes"] == len(content)
            assert records[name]["sha256"] == hashlib.sha256(content).hexdigest()


def test_unknown_bead_and_repository_escape_fail_closed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown Beads"):
        build_sol_pro_work_package(repo_root=repo, scope_prompt="work", bead_ids=("missing",))
    with pytest.raises(ValueError, match="escapes repository"):
        build_sol_pro_work_package(repo_root=repo, scope_prompt="work", source_paths=(outside,))


def test_full_worktree_is_only_included_by_explicit_fallback(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    targeted = build_sol_pro_work_package(repo_root=repo, scope_prompt="work")
    full = build_sol_pro_work_package(repo_root=repo, scope_prompt="work", full_worktree_fallback=True)
    with tarfile.open(fileobj=io.BytesIO(targeted.content), mode="r:gz") as archive:
        assert not any(name.startswith("REPO/") for name in archive.getnames())
    with tarfile.open(fileobj=io.BytesIO(full.content), mode="r:gz") as archive:
        assert "REPO/src/worker.py" in archive.getnames()

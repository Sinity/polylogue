from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock
from urllib import request

import pytest

from devtools import pr_scope

HEAD_SHA = "a" * 40
ASSIGNED = "polylogue-assigned"
OTHER_ASSIGNED = "polylogue-other-assigned"
OPEN_SUCCESSOR = "polylogue-open-successor"
CLOSED_SUCCESSOR = "polylogue-closed-successor"


def _record(bead_id: str, status: str = "open", *, title: str | None = None) -> dict[str, object]:
    return {
        "_type": "issue",
        "id": bead_id,
        "title": title or bead_id,
        "description": "test record",
        "acceptance_criteria": "This prose is deliberately opaque to pr_scope.",
        "status": status,
        "updated_at": "2026-08-06T00:00:00Z",
    }


@pytest.fixture
def beads_path(tmp_path: Path) -> Path:
    path = tmp_path / "issues.jsonl"
    records = [_record(ASSIGNED), _record(OTHER_ASSIGNED), _record(OPEN_SUCCESSOR), _record(CLOSED_SUCCESSOR, "closed")]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repository), *args], capture_output=True, text=True, check=True)


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Polylogue test",
        "-c",
        "user.email=polylogue-test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=",
        "commit",
        "-qm",
        message,
    )
    return _git(repository, "rev-parse", "HEAD").stdout.strip()


def _input(disposition: str = "satisfied", successors: list[str] | None = None) -> dict[str, object]:
    return {
        "assigned_beads": [ASSIGNED],
        "dispositions": [
            {
                "bead_id": ASSIGNED,
                "disposition": disposition,
                "evidence": [{"kind": "test", "ref": "tests/unit/devtools/test_pr_scope.py"}],
                "successors": successors or [],
            }
        ],
    }


def _v2_input() -> dict[str, object]:
    return {
        "scope_kind": "bead",
        "assigned_beads": [ASSIGNED],
        "mutated_beads": [],
        "dispositions": [
            {
                "bead_id": ASSIGNED,
                "disposition": "satisfied",
                "evidence": [{"kind": "test", "ref": "tests/unit/devtools/test_pr_scope.py"}],
                "successors": [],
            }
        ],
    }


def _body(input_payload: dict[str, object], beads_path: Path, *, head_sha: str = HEAD_SHA) -> str:
    carrier = dict(input_payload)
    carrier["version"] = 1
    carrier["head_sha"] = head_sha
    assigned = carrier["assigned_beads"]
    assert isinstance(assigned, list)
    carrier["beads_digest"] = pr_scope.canonical_beads_digest(pr_scope.load_bead_records(beads_path), assigned)
    carrier["scope_digest"] = pr_scope.carrier_digest(carrier)
    return f"## Summary\n\nStructured scope test.\n\n{pr_scope.render_carrier(carrier)}\n"


def _check(
    body: str, beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], *, head_sha: str = HEAD_SHA
) -> str:
    body_path = tmp_path / "pr-body.md"
    body_path.write_text(body)
    exit_code = pr_scope.main(
        ["check", "--body-file", str(body_path), "--head-sha", head_sha, "--beads-path", str(beads_path)]
    )
    output = capsys.readouterr().out
    return f"{exit_code}\n{output}"


def _validate_rendered_without_mutation_scope(body: str, beads_path: Path, *, head_sha: str) -> pr_scope.ScopeVerdict:
    carrier, reasons = pr_scope.extract_carrier(body)
    assert not reasons
    assert carrier is not None
    return pr_scope.validate_carrier(carrier, head_sha=head_sha, is_draft=False, beads_path=beads_path)


class _FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_rendered_carrier_passes_the_production_check_command(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scope_input = tmp_path / "scope.json"
    scope_input.write_text(json.dumps(_input()))

    assert (
        pr_scope.main(["render", "--input", str(scope_input), "--head-sha", HEAD_SHA, "--beads-path", str(beads_path)])
        == 0
    )
    rendered = capsys.readouterr().out

    assert _check(rendered, beads_path, tmp_path, capsys).startswith("0\npr-scope OK")


def test_v2_rendered_carrier_stays_valid_when_the_head_changes(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scope_input = tmp_path / "scope.json"
    scope_input.write_text(json.dumps(_v2_input()))

    assert (
        pr_scope.main(["render", "--input", str(scope_input), "--head-sha", HEAD_SHA, "--beads-path", str(beads_path)])
        == 0
    )
    rendered = capsys.readouterr().out

    assert "head_sha" not in rendered
    assert "beads_digest" not in rendered
    assert _validate_rendered_without_mutation_scope(rendered, beads_path, head_sha="b" * 40).ok


def test_v2_self_contained_scope_needs_no_invented_bead(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scope_input = tmp_path / "scope.json"
    scope_input.write_text(
        json.dumps({"scope_kind": "self_contained", "assigned_beads": [], "mutated_beads": [], "dispositions": []})
    )

    assert (
        pr_scope.main(["render", "--input", str(scope_input), "--head-sha", HEAD_SHA, "--beads-path", str(beads_path)])
        == 0
    )
    rendered = capsys.readouterr().out

    assert _validate_rendered_without_mutation_scope(rendered, beads_path, head_sha="b" * 40).ok


def test_v2_body_file_check_requires_base_revision(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scope_input = tmp_path / "scope.json"
    scope_input.write_text(json.dumps(_v2_input()))
    assert (
        pr_scope.main(["render", "--input", str(scope_input), "--head-sha", HEAD_SHA, "--beads-path", str(beads_path)])
        == 0
    )
    body_path = tmp_path / "body.md"
    body_path.write_text(capsys.readouterr().out)

    assert (
        pr_scope.main(
            [
                "check",
                "--body-file",
                str(body_path),
                "--head-sha",
                HEAD_SHA,
                "--beads-path",
                str(beads_path),
            ]
        )
        == 2
    )
    assert "--base-sha is required" in capsys.readouterr().err


def test_v2_body_file_check_refuses_a_checkout_other_than_the_supplied_head(
    monkeypatch: pytest.MonkeyPatch, beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scope_input = tmp_path / "scope.json"
    scope_input.write_text(json.dumps(_v2_input()))
    assert (
        pr_scope.main(["render", "--input", str(scope_input), "--head-sha", HEAD_SHA, "--beads-path", str(beads_path)])
        == 0
    )
    body_path = tmp_path / "body.md"
    body_path.write_text(capsys.readouterr().out)
    monkeypatch.setattr(pr_scope, "_git_head_sha", lambda: "c" * 40)

    assert (
        pr_scope.main(
            [
                "check",
                "--body-file",
                str(body_path),
                "--head-sha",
                HEAD_SHA,
                "--base-sha",
                "b" * 40,
                "--beads-path",
                str(beads_path),
            ]
        )
        == 2
    )
    assert "current checkout HEAD does not match" in capsys.readouterr().err


def test_base_revision_fetch_has_a_finite_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    responses = [
        MagicMock(returncode=1, stdout=b"", stderr=b""),
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout=b"", stderr=b""),
    ]

    def _run(command: list[str], **kwargs: object) -> MagicMock:
        calls.append((command, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(subprocess, "run", _run)

    pr_scope._ensure_local_commit("a" * 40)

    assert calls[1][0][:2] == ["git", "fetch"]
    assert calls[1][1]["timeout"] == 120


@pytest.mark.parametrize("version", [True, 1.0])
def test_render_rejects_non_integer_carrier_versions(
    version: object, beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scope_input = tmp_path / "scope.json"
    payload = _v2_input()
    payload["version"] = version
    scope_input.write_text(json.dumps(payload))

    assert (
        pr_scope.main(["render", "--input", str(scope_input), "--head-sha", HEAD_SHA, "--beads-path", str(beads_path)])
        == 2
    )
    assert "input version must be 1 or 2" in capsys.readouterr().err


def test_v2_check_rejects_an_unlisted_bead_mutation_via_the_production_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pr_scope, "_require_checkout_authority", lambda **_kwargs: None)
    repository = tmp_path / "repository"
    beads_path = repository / ".beads" / "issues.jsonl"
    beads_path.parent.mkdir(parents=True)
    beads_path.write_text(json.dumps(_record(ASSIGNED)) + "\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", ".beads/issues.jsonl"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Polylogue test",
            "-c",
            "user.email=polylogue-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=",
            "commit",
            "-qm",
            "base Bead state",
        ],
        check=True,
    )
    base_sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    changed = _record(ASSIGNED, title="changed tracker record")
    beads_path.write_text(json.dumps(changed) + "\n")
    subprocess.run(["git", "-C", str(repository), "add", ".beads/issues.jsonl"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Polylogue test",
            "-c",
            "user.email=polylogue-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=",
            "commit",
            "-qm",
            "candidate Bead change",
        ],
        check=True,
    )
    head_sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    scope_input = repository / "scope.json"
    scope_input.write_text(json.dumps(_v2_input()))
    monkeypatch.chdir(repository)

    assert (
        pr_scope.main(["render", "--input", str(scope_input), "--head-sha", head_sha, "--beads-path", str(beads_path)])
        == 0
    )
    rendered = capsys.readouterr().out
    body_path = repository / "body.md"
    body_path.write_text(rendered)

    assert (
        pr_scope.main(
            [
                "check",
                "--body-file",
                str(body_path),
                "--head-sha",
                head_sha,
                "--base-sha",
                base_sha,
                "--beads-path",
                str(beads_path),
            ]
        )
        == 1
    )
    assert "mutated_beads does not match the complete Bead mutation set" in capsys.readouterr().out

    legacy_body_path = repository / "legacy-body.md"
    legacy_body_path.write_text(_body(_input(), beads_path, head_sha=head_sha))
    assert (
        pr_scope.main(
            [
                "check",
                "--body-file",
                str(legacy_body_path),
                "--head-sha",
                head_sha,
                "--base-sha",
                base_sha,
                "--beads-path",
                str(beads_path),
            ]
        )
        == 1
    )
    assert "legacy v1 carrier cannot omit Bead mutations" in capsys.readouterr().out


def test_validation_uses_the_immutable_head_beads_when_the_worktree_diverges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repository"
    beads_path = repository / ".beads" / "issues.jsonl"
    beads_path.parent.mkdir(parents=True)
    beads_path.write_text(json.dumps(_record(ASSIGNED)) + "\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    base_sha = _commit(repository, "base Bead state")

    beads_path.write_text("")
    head_sha = _commit(repository, "candidate deletes assigned Bead")

    carrier = _v2_input()
    carrier["version"] = 2
    carrier["mutated_beads"] = [ASSIGNED]
    carrier["scope_digest"] = pr_scope.carrier_digest(carrier)

    # Simulate a concurrent/live Beads checkout that no longer matches the
    # candidate commit. Revision-bound validation must follow the Git object,
    # not allow this mutable path to resurrect the deleted record.
    beads_path.write_text(json.dumps(_record(ASSIGNED)) + "\n")
    monkeypatch.chdir(repository)

    verdict = pr_scope.validate_carrier(
        carrier,
        head_sha=head_sha,
        is_draft=False,
        beads_path=beads_path,
        base_sha=base_sha,
    )

    assert not verdict.ok
    assert f"assigned Bead record(s) missing: {ASSIGNED}" in verdict.reasons


def test_v2_self_contained_scope_rejects_a_real_bead_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repository = tmp_path / "repository"
    beads_path = repository / ".beads" / "issues.jsonl"
    beads_path.parent.mkdir(parents=True)
    beads_path.write_text(json.dumps(_record(ASSIGNED)) + "\n")
    beads_path.write_text(json.dumps(_record(ASSIGNED, title="changed by a self-contained PR")) + "\n")
    scope_input = repository / "scope.json"
    scope_input.write_text(
        json.dumps(
            {"scope_kind": "self_contained", "assigned_beads": [], "mutated_beads": [ASSIGNED], "dispositions": []}
        )
    )
    assert (
        pr_scope.main(
            [
                "render",
                "--input",
                str(scope_input),
                "--head-sha",
                HEAD_SHA,
                "--beads-path",
                str(beads_path),
            ]
        )
        == 2
    )
    assert "self_contained scope cannot declare or mutate Beads" in capsys.readouterr().err


def test_v2_deleted_mutated_bead_stays_in_bead_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pr_scope, "_require_checkout_authority", lambda **_kwargs: None)
    repository = tmp_path / "repository"
    beads_path = repository / ".beads" / "issues.jsonl"
    beads_path.parent.mkdir(parents=True)
    beads_path.write_text(
        "\n".join(json.dumps(record) for record in [_record(ASSIGNED), _record(OTHER_ASSIGNED)]) + "\n"
    )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", ".beads/issues.jsonl"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Polylogue test",
            "-c",
            "user.email=polylogue-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=",
            "commit",
            "-qm",
            "base Bead state",
        ],
        check=True,
    )
    base_sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    beads_path.write_text(json.dumps(_record(ASSIGNED)) + "\n")
    subprocess.run(["git", "-C", str(repository), "add", ".beads/issues.jsonl"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Polylogue test",
            "-c",
            "user.email=polylogue-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=",
            "commit",
            "-qm",
            "candidate Bead deletion",
        ],
        check=True,
    )
    head_sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    scope_input = repository / "scope.json"
    payload = _v2_input()
    payload["mutated_beads"] = [OTHER_ASSIGNED]
    scope_input.write_text(json.dumps(payload))
    monkeypatch.chdir(repository)

    assert (
        pr_scope.main(["render", "--input", str(scope_input), "--head-sha", head_sha, "--beads-path", str(beads_path)])
        == 0
    )
    body_path = repository / "body.md"
    body_path.write_text(capsys.readouterr().out)

    assert (
        pr_scope.main(
            [
                "check",
                "--body-file",
                str(body_path),
                "--head-sha",
                head_sha,
                "--base-sha",
                base_sha,
                "--beads-path",
                str(beads_path),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.startswith("pr-scope OK")


def test_v2_mutation_scope_uses_the_pr_merge_base_not_the_moved_target_tip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pr_scope, "_require_checkout_authority", lambda **_kwargs: None)
    repository = tmp_path / "repository"
    beads_path = repository / ".beads" / "issues.jsonl"
    beads_path.parent.mkdir(parents=True)
    beads_path.write_text(
        "\n".join(json.dumps(record) for record in [_record(ASSIGNED), _record(OTHER_ASSIGNED)]) + "\n"
    )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", ".beads/issues.jsonl"], check=True)
    commit = [
        "git",
        "-C",
        str(repository),
        "-c",
        "user.name=Polylogue test",
        "-c",
        "user.email=polylogue-test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=",
        "commit",
        "-qm",
    ]
    subprocess.run([*commit, "shared base"], check=True)
    shared_base = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    beads_path.write_text(
        "\n".join(
            json.dumps(record) for record in [_record(ASSIGNED), _record(OTHER_ASSIGNED, title="upstream change")]
        )
        + "\n"
    )
    subprocess.run(["git", "-C", str(repository), "add", ".beads/issues.jsonl"], check=True)
    subprocess.run([*commit, "target branch changed Bead"], check=True)
    moved_target = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    subprocess.run(["git", "-C", str(repository), "checkout", "-q", "-b", "feature", shared_base], check=True)
    beads_path.write_text(
        "\n".join(json.dumps(record) for record in [_record(ASSIGNED, title="feature change"), _record(OTHER_ASSIGNED)])
        + "\n"
    )
    subprocess.run(["git", "-C", str(repository), "add", ".beads/issues.jsonl"], check=True)
    subprocess.run([*commit, "feature Bead change"], check=True)
    head_sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    scope_input = repository / "scope.json"
    scope_input.write_text(json.dumps(_v2_input() | {"mutated_beads": [ASSIGNED]}))
    monkeypatch.chdir(repository)

    assert (
        pr_scope.main(["render", "--input", str(scope_input), "--head-sha", head_sha, "--beads-path", str(beads_path)])
        == 0
    )
    body_path = repository / "body.md"
    body_path.write_text(capsys.readouterr().out)

    assert (
        pr_scope.main(
            [
                "check",
                "--body-file",
                str(body_path),
                "--head-sha",
                head_sha,
                "--base-sha",
                moved_target,
                "--beads-path",
                str(beads_path),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.startswith("pr-scope OK")


def test_sync_reports_a_head_bound_attestation_without_rewriting_the_body(
    monkeypatch: pytest.MonkeyPatch, beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scope_input = tmp_path / "scope.json"
    scope_input.write_text(json.dumps(_v2_input()))
    assert (
        pr_scope.main(["render", "--input", str(scope_input), "--head-sha", HEAD_SHA, "--beads-path", str(beads_path)])
        == 0
    )
    body = capsys.readouterr().out
    metadata = pr_scope.PullRequestMetadata(body=body, head_sha=HEAD_SHA, base_sha="b" * 40, is_draft=False)
    monkeypatch.setattr(pr_scope, "fetch_pr_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(pr_scope, "resolve_repository", lambda _repo: "Sinity/polylogue")
    monkeypatch.setattr(pr_scope, "_git_head_sha", lambda: HEAD_SHA)
    monkeypatch.setattr(pr_scope, "changed_bead_ids", lambda **_kwargs: [])
    monkeypatch.setattr(pr_scope, "_bead_records_at", lambda _revision: pr_scope.load_bead_records(beads_path))
    monkeypatch.setattr(pr_scope, "_beads_snapshot_matches_head", lambda _path: True)

    assert pr_scope.main(["sync", "--pr", "42", "--repo", "Sinity/polylogue", "--beads-path", str(beads_path)]) == 0
    attestation = json.loads(capsys.readouterr().out)

    assert attestation["head_sha"] == HEAD_SHA
    assert attestation["base_sha"] == "b" * 40
    assert attestation["beads_digest"] == pr_scope.canonical_beads_digest(
        pr_scope.load_bead_records(beads_path), [ASSIGNED], carrier_version=2
    )
    assert attestation["attestation_digest"]


def test_sync_refuses_uncommitted_bead_contents(
    monkeypatch: pytest.MonkeyPatch, beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scope_input = tmp_path / "scope.json"
    scope_input.write_text(json.dumps(_v2_input()))
    assert (
        pr_scope.main(["render", "--input", str(scope_input), "--head-sha", HEAD_SHA, "--beads-path", str(beads_path)])
        == 0
    )
    metadata = pr_scope.PullRequestMetadata(
        body=capsys.readouterr().out, head_sha=HEAD_SHA, base_sha="b" * 40, is_draft=False
    )
    monkeypatch.setattr(pr_scope, "fetch_pr_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(pr_scope, "resolve_repository", lambda _repo: "Sinity/polylogue")
    monkeypatch.setattr(pr_scope, "_git_head_sha", lambda: HEAD_SHA)
    monkeypatch.setattr(pr_scope, "_beads_snapshot_matches_head", lambda _path: False)

    assert pr_scope.main(["sync", "--pr", "42", "--repo", "Sinity/polylogue", "--beads-path", str(beads_path)]) == 2
    assert "Beads snapshot does not match the committed PR head" in capsys.readouterr().err


def test_pr_check_uses_public_github_rest_without_cli_auth(
    monkeypatch: pytest.MonkeyPatch,
    beads_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = _body(_input(), beads_path)
    requests: list[request.Request] = []

    def _urlopen(api_request: request.Request, *, timeout: int) -> _FakeHttpResponse:
        assert timeout == 30
        requests.append(api_request)
        payload = {
            "number": 42,
            "state": "open",
            "body": body,
            "draft": False,
            "head": {"sha": HEAD_SHA},
            "base": {"sha": "b" * 40, "ref": "master"},
        }
        return _FakeHttpResponse(json.dumps(payload).encode())

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(request, "urlopen", _urlopen)
    monkeypatch.setattr(pr_scope, "_git_head_sha", lambda: HEAD_SHA)
    monkeypatch.setattr(pr_scope, "_beads_snapshot_matches_head", lambda _path: True)
    monkeypatch.setattr(pr_scope, "changed_bead_ids", lambda **_kwargs: [])
    monkeypatch.setattr(pr_scope, "_bead_records_at", lambda _revision: pr_scope.load_bead_records(beads_path))

    exit_code = pr_scope.main(["check", "--pr", "42", "--repo", "Sinity/polylogue", "--beads-path", str(beads_path)])

    assert exit_code == 0
    assert capsys.readouterr().out.startswith("pr-scope OK")
    assert len(requests) == 1
    assert requests[0].full_url == "https://api.github.com/repos/Sinity/polylogue/pulls/42"
    assert requests[0].get_header("Authorization") is None


def test_pr_check_refuses_a_checkout_other_than_the_fetched_head(
    monkeypatch: pytest.MonkeyPatch, beads_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metadata = pr_scope.PullRequestMetadata(
        body=_body(_input(), beads_path), head_sha=HEAD_SHA, base_sha="b" * 40, is_draft=False
    )
    monkeypatch.setattr(pr_scope, "resolve_repository", lambda _repo: "Sinity/polylogue")
    monkeypatch.setattr(pr_scope, "fetch_pr_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(pr_scope, "_git_head_sha", lambda: "c" * 40)

    assert pr_scope.main(["check", "--pr", "42", "--repo", "Sinity/polylogue", "--beads-path", str(beads_path)]) == 2
    assert "current checkout HEAD does not match the fetched PR head" in capsys.readouterr().err


def test_pr_check_refuses_uncommitted_bead_contents(
    monkeypatch: pytest.MonkeyPatch, beads_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metadata = pr_scope.PullRequestMetadata(
        body=_body(_input(), beads_path), head_sha=HEAD_SHA, base_sha="b" * 40, is_draft=False
    )
    monkeypatch.setattr(pr_scope, "resolve_repository", lambda _repo: "Sinity/polylogue")
    monkeypatch.setattr(pr_scope, "fetch_pr_metadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(pr_scope, "_git_head_sha", lambda: HEAD_SHA)
    monkeypatch.setattr(pr_scope, "_beads_snapshot_matches_head", lambda _path: False)

    assert pr_scope.main(["check", "--pr", "42", "--repo", "Sinity/polylogue", "--beads-path", str(beads_path)]) == 2
    assert "Beads snapshot does not match the committed PR head" in capsys.readouterr().err


def test_fetch_pr_metadata_fetches_files_for_authoritative_dependabot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[str] = []

    def _urlopen(api_request: request.Request, *, timeout: int) -> _FakeHttpResponse:
        assert timeout == 30
        requests.append(api_request.full_url)
        payload: object
        if api_request.full_url.endswith("/pulls/42"):
            payload = {
                "number": 42,
                "state": "open",
                "body": "",
                "draft": False,
                "head": {"sha": HEAD_SHA},
                "base": {"sha": "b" * 40, "ref": "master"},
                "user": {"login": "dependabot[bot]", "type": "Bot"},
            }
        else:
            payload = [{"filename": "pyproject.toml"}, {"filename": "uv.lock"}]
        return _FakeHttpResponse(json.dumps(payload).encode())

    monkeypatch.setattr(request, "urlopen", _urlopen)

    metadata = pr_scope.fetch_pr_metadata(42, repository="Sinity/polylogue")

    assert metadata.author_login == "dependabot[bot]"
    assert metadata.author_type == "Bot"
    assert metadata.changed_files == ("pyproject.toml", "uv.lock")
    assert requests == [
        "https://api.github.com/repos/Sinity/polylogue/pulls/42",
        "https://api.github.com/repos/Sinity/polylogue/pulls/42/files?per_page=100",
    ]


def test_fetch_pr_files_rejects_truncated_automated_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [{"filename": f"dependency-{index}.toml"} for index in range(100)]
    monkeypatch.setattr(
        pr_scope,
        "_github_request_bytes",
        lambda _path: json.dumps(payload).encode(),
    )

    with pytest.raises(ValueError, match="100 or more changed files"):
        pr_scope.fetch_pr_files(42, repository="Sinity/polylogue")


@pytest.mark.parametrize(
    ("checkout_head_sha", "expected_head_sha"),
    [("c" * 40, HEAD_SHA), (HEAD_SHA, "d" * 40)],
)
def test_ci_check_refuses_head_mismatch_before_fetching_base(
    checkout_head_sha: str,
    expected_head_sha: str,
    monkeypatch: pytest.MonkeyPatch,
    beads_path: Path,
) -> None:
    metadata = pr_scope.PullRequestMetadata(
        body=_body(_input(), beads_path),
        head_sha=HEAD_SHA,
        base_sha="b" * 40,
        is_draft=False,
    )
    fetch_base = MagicMock()
    fetch_base.return_value = b"not a validator"
    monkeypatch.setattr(pr_scope, "fetch_base_validator_source", fetch_base)

    assert (
        pr_scope.check_ci_metadata(
            metadata,
            repository="Sinity/polylogue",
            beads_path=beads_path,
            checkout_head_sha=checkout_head_sha,
            expected_head_sha=expected_head_sha,
        )
        == 2
    )
    fetch_base.assert_not_called()


def test_ci_check_refuses_draft_before_fetching_base(
    monkeypatch: pytest.MonkeyPatch,
    beads_path: Path,
) -> None:
    metadata = pr_scope.PullRequestMetadata(
        body=_body(_input(), beads_path),
        head_sha=HEAD_SHA,
        base_sha="b" * 40,
        is_draft=True,
    )
    fetch_base = MagicMock()
    monkeypatch.setattr(pr_scope, "fetch_base_validator_source", fetch_base)

    assert (
        pr_scope.check_ci_metadata(
            metadata,
            repository="Sinity/polylogue",
            beads_path=beads_path,
            checkout_head_sha=HEAD_SHA,
            expected_head_sha=HEAD_SHA,
        )
        == 2
    )
    fetch_base.assert_not_called()


def test_ci_check_refuses_uncommitted_bead_contents_before_fetching_base(
    monkeypatch: pytest.MonkeyPatch,
    beads_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = pr_scope.PullRequestMetadata(
        body=_body(_input(), beads_path),
        head_sha=HEAD_SHA,
        base_sha="b" * 40,
        is_draft=False,
    )
    fetch_base = MagicMock()
    monkeypatch.setattr(pr_scope, "fetch_base_validator_source", fetch_base)
    monkeypatch.setattr(pr_scope, "_beads_snapshot_matches_head", lambda _path: False)

    assert (
        pr_scope.check_ci_metadata(
            metadata,
            repository="Sinity/polylogue",
            beads_path=beads_path,
            checkout_head_sha=HEAD_SHA,
            expected_head_sha=HEAD_SHA,
        )
        == 2
    )
    assert "Beads snapshot does not match the committed PR head" in capsys.readouterr().err
    fetch_base.assert_not_called()


def test_ci_accepts_authoritative_dependabot_dependency_only_pr(
    monkeypatch: pytest.MonkeyPatch,
    beads_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = pr_scope.PullRequestMetadata(
        body="",
        head_sha=HEAD_SHA,
        base_sha="b" * 40,
        is_draft=False,
        author_login="dependabot[bot]",
        author_type="Bot",
        changed_files=("pyproject.toml", "uv.lock"),
    )
    fetch_base = MagicMock()
    monkeypatch.setattr(pr_scope, "fetch_base_validator_source", fetch_base)
    monkeypatch.setattr(pr_scope, "_beads_snapshot_matches_head", lambda _path: True)

    assert (
        pr_scope.check_ci_metadata(
            metadata,
            repository="Sinity/polylogue",
            beads_path=beads_path,
            checkout_head_sha=HEAD_SHA,
            expected_head_sha=HEAD_SHA,
        )
        == 0
    )
    assert "typed automated-dependency disposition" in capsys.readouterr().out
    assert pr_scope.automated_dependency_scope_allowed(
        author_login="app/dependabot",
        author_type=None,
        author_is_bot=True,
        changed_files=("pyproject.toml", "uv.lock"),
    )
    fetch_base.assert_not_called()


@pytest.mark.parametrize(
    "metadata",
    [
        pr_scope.PullRequestMetadata(
            body="",
            head_sha=HEAD_SHA,
            base_sha="b" * 40,
            is_draft=False,
            author_login="dependabot[bot]",
            author_type="User",
            changed_files=("pyproject.toml",),
        ),
        pr_scope.PullRequestMetadata(
            body="",
            head_sha=HEAD_SHA,
            base_sha="b" * 40,
            is_draft=False,
            author_login="dependabot[bot]",
            author_type="Bot",
            changed_files=("pyproject.toml", "polylogue/storage/write.py"),
        ),
    ],
)
def test_ci_rejects_spoofed_or_extra_file_automated_scope(
    metadata: pr_scope.PullRequestMetadata,
    monkeypatch: pytest.MonkeyPatch,
    beads_path: Path,
) -> None:
    fetch_base = MagicMock()
    fetch_base.return_value = b"not a validator"
    monkeypatch.setattr(pr_scope, "fetch_base_validator_source", fetch_base)
    monkeypatch.setattr(pr_scope, "_run_validator_source", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(pr_scope, "_beads_snapshot_matches_head", lambda _path: True)

    assert (
        pr_scope.check_ci_metadata(
            metadata,
            repository="Sinity/polylogue",
            beads_path=beads_path,
            checkout_head_sha=HEAD_SHA,
            expected_head_sha=HEAD_SHA,
        )
        == 1
    )
    fetch_base.assert_called_once()


def test_ci_check_executes_base_revision_validator(
    monkeypatch: pytest.MonkeyPatch,
    beads_path: Path,
) -> None:
    metadata = pr_scope.PullRequestMetadata(
        body="## Summary\n\nA PR-modified validator would accept this body.",
        head_sha=HEAD_SHA,
        base_sha="b" * 40,
        is_draft=False,
    )
    base_source = Path(pr_scope.__file__).read_bytes()
    current_validator = MagicMock(return_value=pr_scope.ScopeVerdict(ok=True))
    monkeypatch.setattr(pr_scope, "fetch_base_validator_source", lambda **_kwargs: base_source)
    monkeypatch.setattr(pr_scope, "validate_pr_body", current_validator)
    monkeypatch.setattr(pr_scope, "_beads_snapshot_matches_head", lambda _path: True)

    exit_code = pr_scope.check_ci_metadata(
        metadata,
        repository="Sinity/polylogue",
        beads_path=beads_path,
        checkout_head_sha=HEAD_SHA,
        expected_head_sha=HEAD_SHA,
    )

    assert exit_code == 1
    current_validator.assert_not_called()


def test_ci_check_bootstraps_once_when_base_has_no_validator(
    monkeypatch: pytest.MonkeyPatch,
    beads_path: Path,
) -> None:
    metadata = pr_scope.PullRequestMetadata(
        body=_body(_input(), beads_path),
        head_sha=HEAD_SHA,
        base_sha="b" * 40,
        is_draft=False,
    )
    monkeypatch.setattr(pr_scope, "fetch_base_validator_source", lambda **_kwargs: None)
    monkeypatch.setattr(pr_scope, "changed_bead_ids", lambda **_kwargs: [])
    monkeypatch.setattr(pr_scope, "_bead_records_at", lambda _revision: pr_scope.load_bead_records(beads_path))
    monkeypatch.setattr(pr_scope, "_beads_snapshot_matches_head", lambda _path: True)

    assert (
        pr_scope.check_ci_metadata(
            metadata,
            repository="Sinity/polylogue",
            beads_path=beads_path,
            checkout_head_sha=HEAD_SHA,
            expected_head_sha=HEAD_SHA,
        )
        == 0
    )


def test_fetch_base_validator_prefers_local_base_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github_fetch = MagicMock()
    monkeypatch.setattr(pr_scope, "_github_request_bytes", github_fetch)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git", "show"], returncode=0, stdout=b"base validator", stderr=b""
        ),
    )

    assert pr_scope.fetch_base_validator_source(repository="Sinity/polylogue", base_sha="b" * 40) == b"base validator"
    github_fetch.assert_not_called()


def test_changed_bead_ids_fetches_missing_target_commit_before_merge_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_sha = "b" * 40
    calls: list[list[str]] = []
    cat_file_attempts = 0

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal cat_file_attempts
        calls.append(argv)
        if argv[:3] == ["git", "cat-file", "-e"]:
            cat_file_attempts += 1
            return subprocess.CompletedProcess(argv, 0 if cat_file_attempts in {2, 4} else 1, "", "")
        if argv[:2] == ["git", "fetch"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:2] == ["git", "merge-base"]:
            return subprocess.CompletedProcess(argv, 0, "a" * 40 + "\n", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(pr_scope, "_bead_records_at", lambda _sha: {})
    assert pr_scope.changed_bead_ids(base_sha=base_sha, head_sha=HEAD_SHA) == []
    assert ["git", "fetch", "--no-tags", "--quiet", "origin", base_sha] in calls
    assert calls.index(["git", "fetch", "--no-tags", "--quiet", "origin", base_sha]) < calls.index(
        ["git", "merge-base", base_sha, HEAD_SHA]
    )


def test_changed_bead_ids_unshallows_before_retrying_the_merge_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_sha = "b" * 40
    calls: list[tuple[list[str], dict[str, object]]] = []
    merge_base_attempts = 0

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal merge_base_attempts
        calls.append((argv, kwargs))
        if argv[:3] == ["git", "cat-file", "-e"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:2] == ["git", "merge-base"]:
            merge_base_attempts += 1
            output = "a" * 40 + "\n" if merge_base_attempts == 2 else ""
            return subprocess.CompletedProcess(argv, 0 if output else 1, output, "")
        if argv == ["git", "rev-parse", "--is-shallow-repository"]:
            return subprocess.CompletedProcess(argv, 0, "true\n", "")
        if argv[:2] == ["git", "fetch"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(pr_scope, "_bead_records_at", lambda _sha: {})
    assert pr_scope.changed_bead_ids(base_sha=base_sha, head_sha=HEAD_SHA) == []
    unshallow = next(item for item in calls if "--unshallow" in item[0])
    assert unshallow[1]["timeout"] == 120
    assert merge_base_attempts == 2


def test_check_rejects_carrier_bound_to_a_different_head_sha(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _check(_body(_input(), beads_path, head_sha="b" * 40), beads_path, tmp_path, capsys)

    assert result.startswith("1\n")
    assert "head_sha does not match" in result


def test_check_rejects_missing_assigned_bead_disposition(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _input()
    payload["assigned_beads"] = [ASSIGNED, OTHER_ASSIGNED]
    result = _check(_body(payload, beads_path), beads_path, tmp_path, capsys)

    assert result.startswith("1\n")
    assert f"missing whole-Bead disposition(s): {OTHER_ASSIGNED}" in result


@pytest.mark.parametrize(
    ("successor", "reason"),
    [(CLOSED_SUCCESSOR, "is closed"), ("polylogue-does-not-exist", "is unknown")],
)
def test_check_rejects_closed_or_unknown_residual_successor(
    successor: str, reason: str, beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _check(_body(_input("partial", [successor]), beads_path), beads_path, tmp_path, capsys)

    assert result.startswith("1\n")
    assert reason in result


def test_check_rejects_unlinked_residual_successor(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [_record(ASSIGNED), _record(OPEN_SUCCESSOR)]
    records[1]["dependencies"] = [
        {"issue_id": OPEN_SUCCESSOR, "depends_on_id": "polylogue-unrelated", "type": "relates-to"}
    ]
    beads_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    result = _check(_body(_input("partial", [OPEN_SUCCESSOR]), beads_path), beads_path, tmp_path, capsys)

    assert result.startswith("1\n")
    assert "has no durable Beads relationship" in result


def test_check_accepts_linked_residual_successor(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [_record(ASSIGNED), _record(OPEN_SUCCESSOR)]
    records[1]["dependencies"] = [{"issue_id": OPEN_SUCCESSOR, "depends_on_id": ASSIGNED, "type": "discovered-from"}]
    beads_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    result = _check(_body(_input("partial", [OPEN_SUCCESSOR]), beads_path), beads_path, tmp_path, capsys)

    assert result.startswith("0\n")


def test_check_uses_prospective_merge_state_for_residual_successor(
    monkeypatch: pytest.MonkeyPatch, beads_path: Path
) -> None:
    head_records = [_record(ASSIGNED), _record(OPEN_SUCCESSOR)]
    head_records[1]["dependencies"] = [
        {"issue_id": OPEN_SUCCESSOR, "depends_on_id": ASSIGNED, "type": "discovered-from"}
    ]
    beads_path.write_text("\n".join(json.dumps(record) for record in head_records) + "\n")
    carrier = pr_scope.build_carrier(
        {
            "scope_kind": "bead",
            "assigned_beads": [ASSIGNED],
            "mutated_beads": [],
            "dispositions": [
                {
                    "bead_id": ASSIGNED,
                    "disposition": "partial",
                    "evidence": [{"kind": "test", "ref": "prospective successor regression"}],
                    "successors": [OPEN_SUCCESSOR],
                }
            ],
        },
        head_sha=HEAD_SHA,
        beads_path=beads_path,
    )
    target_records = {record["id"]: record for record in head_records}
    target_records[OPEN_SUCCESSOR] = _record(OPEN_SUCCESSOR, "closed")
    target_records[OPEN_SUCCESSOR]["dependencies"] = head_records[1]["dependencies"]
    monkeypatch.setattr(pr_scope, "changed_bead_ids", lambda **_kwargs: [])
    monkeypatch.setattr(pr_scope, "_bead_records_at", lambda _revision: target_records)

    verdict = pr_scope.validate_carrier(
        carrier,
        head_sha=HEAD_SHA,
        is_draft=False,
        beads_path=beads_path,
        base_sha="b" * 40,
    )

    assert not verdict.ok
    assert any(f"successor {OPEN_SUCCESSOR} is closed" in reason for reason in verdict.reasons)


def test_check_uses_prospective_merge_state_for_assigned_bead(
    monkeypatch: pytest.MonkeyPatch, beads_path: Path
) -> None:
    head_records = [_record(ASSIGNED)]
    beads_path.write_text(json.dumps(head_records[0]) + "\n")
    carrier = pr_scope.build_carrier(_input(), head_sha=HEAD_SHA, beads_path=beads_path)
    monkeypatch.setattr(pr_scope, "changed_bead_ids", lambda **_kwargs: [])
    monkeypatch.setattr(pr_scope, "_bead_records_at", lambda _revision: {})

    verdict = pr_scope.validate_carrier(
        carrier,
        head_sha=HEAD_SHA,
        is_draft=False,
        beads_path=beads_path,
        base_sha="b" * 40,
    )

    assert not verdict.ok
    assert f"assigned Bead record(s) missing: {ASSIGNED}" in verdict.reasons


def test_check_rejects_stale_canonical_beads_digest(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = _body(_input(), beads_path)
    records = [json.loads(line) for line in beads_path.read_text().splitlines()]
    records[0]["title"] = "changed after carrier render"
    beads_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    result = _check(body, beads_path, tmp_path, capsys)

    assert result.startswith("1\n")
    assert "beads_digest is stale" in result


def test_check_rejects_partial_disposition_without_successor(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _check(_body(_input("partial"), beads_path), beads_path, tmp_path, capsys)

    assert result.startswith("1\n")
    assert "partial disposition requires a named successor" in result


def test_check_rejects_unknown_schema_fields(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    carrier = pr_scope.build_carrier(_input(), head_sha=HEAD_SHA, beads_path=beads_path)
    carrier["acceptance_summary"] = "silently extending v1 would make the schema ambiguous"
    carrier["scope_digest"] = pr_scope.carrier_digest(carrier)

    result = _check(pr_scope.render_carrier(carrier), beads_path, tmp_path, capsys)

    assert result.startswith("1\n")
    assert "unknown field(s): acceptance_summary" in result


def test_render_refuses_invalid_partial_scope_input(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scope_input = tmp_path / "scope.json"
    scope_input.write_text(json.dumps(_input("partial")))

    exit_code = pr_scope.main(
        ["render", "--input", str(scope_input), "--head-sha", HEAD_SHA, "--beads-path", str(beads_path)]
    )

    assert exit_code == 2
    assert "partial disposition requires a named successor" in capsys.readouterr().err


def test_check_rejects_pr_body_without_carrier(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _check("## Summary\n\nNo structured carrier.\n", beads_path, tmp_path, capsys)

    assert result.startswith("1\n")
    assert "missing the structured pr-scope carrier" in result

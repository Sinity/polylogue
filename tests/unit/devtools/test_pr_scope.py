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
            "body": body,
            "draft": False,
            "head": {"sha": HEAD_SHA},
            "base": {"sha": "b" * 40},
        }
        return _FakeHttpResponse(json.dumps(payload).encode())

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(request, "urlopen", _urlopen)

    exit_code = pr_scope.main(["check", "--pr", "42", "--repo", "Sinity/polylogue", "--beads-path", str(beads_path)])

    assert exit_code == 0
    assert capsys.readouterr().out.startswith("pr-scope OK")
    assert len(requests) == 1
    assert requests[0].full_url == "https://api.github.com/repos/Sinity/polylogue/pulls/42"
    assert requests[0].get_header("Authorization") is None


def test_ci_resolves_pr_from_exact_head_when_circle_pr_url_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[request.Request] = []

    def _urlopen(api_request: request.Request, *, timeout: int) -> _FakeHttpResponse:
        assert timeout == 30
        requests.append(api_request)
        payload = [
            {
                "number": 3845,
                "state": "open",
                "body": "carrier",
                "draft": False,
                "head": {"sha": HEAD_SHA},
                "base": {"sha": "b" * 40},
            },
            {
                "number": 3800,
                "state": "closed",
                "body": "old carrier",
                "draft": False,
                "head": {"sha": HEAD_SHA},
                "base": {"sha": "c" * 40},
            },
        ]
        return _FakeHttpResponse(json.dumps(payload).encode())

    monkeypatch.setattr(request, "urlopen", _urlopen)

    pr_number, metadata = pr_scope.fetch_pr_for_head(repository="Sinity/polylogue", head_sha=HEAD_SHA)

    assert pr_number == 3845
    assert metadata.head_sha == HEAD_SHA
    assert requests[0].full_url == f"https://api.github.com/repos/Sinity/polylogue/commits/{HEAD_SHA}/pulls"


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
                "body": "",
                "draft": False,
                "head": {"sha": HEAD_SHA},
                "base": {"sha": "b" * 40},
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


def test_fetch_pr_for_head_reports_when_no_open_pr_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(request, "urlopen", lambda *_args, **_kwargs: _FakeHttpResponse(b"[]"))

    with pytest.raises(pr_scope.NoOpenPullRequestError, match="no open PR"):
        pr_scope.fetch_pr_for_head(repository="Sinity/polylogue", head_sha=HEAD_SHA)


def test_ci_skips_when_commit_has_no_open_pr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(pr_scope, "resolve_repository", lambda _repo: "Sinity/polylogue")
    monkeypatch.setattr(pr_scope, "_git_head_sha", lambda: HEAD_SHA)

    def _no_pr(**_kwargs: object) -> tuple[int, pr_scope.PullRequestMetadata]:
        raise pr_scope.NoOpenPullRequestError("no open PR found for head aaaaaaaa")

    monkeypatch.setattr(pr_scope, "fetch_pr_for_head", _no_pr)

    assert pr_scope.main(["check-ci", "--repo", "Sinity/polylogue", "--expected-head-sha", HEAD_SHA]) == 0
    assert "pr-scope CI skip" in capsys.readouterr().err


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

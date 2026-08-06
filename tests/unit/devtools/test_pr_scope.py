from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

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
    carrier = pr_scope.build_carrier(input_payload, head_sha=head_sha, beads_path=beads_path)
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


def test_check_rejects_pr_body_without_carrier(
    beads_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _check("## Summary\n\nNo structured carrier.\n", beads_path, tmp_path, capsys)

    assert result.startswith("1\n")
    assert "missing the structured pr-scope carrier" in result


def test_pr_lookup_falls_back_to_github_api_when_gh_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response(BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def missing_gh(*args: object, **kwargs: object) -> object:
        command = args[0]
        if isinstance(command, list) and command[:1] == ["gh"]:
            raise FileNotFoundError("gh")
        return type("Completed", (), {"stdout": "git@github.com:Sinity/polylogue.git\n"})()

    seen_urls: list[str] = []

    def github_response(api_request: object, *, timeout: int) -> Response:
        assert isinstance(api_request, pr_scope.request.Request)
        seen_urls.append(api_request.full_url)
        return Response(json.dumps({"body": "carrier", "draft": False, "head": {"sha": HEAD_SHA}}).encode())

    monkeypatch.setattr(pr_scope.subprocess, "run", missing_gh)
    monkeypatch.setattr(pr_scope.request, "urlopen", github_response)

    assert pr_scope._pr_body(3845) == ("carrier", HEAD_SHA, False)
    assert seen_urls == ["https://api.github.com/repos/Sinity/polylogue/pulls/3845"]

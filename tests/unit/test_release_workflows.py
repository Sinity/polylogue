"""Exercise release workflow shell contracts without contacting publishers."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
import tomllib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def workflow(name: str) -> dict[str, Any]:
    # BaseLoader preserves GitHub's `on` key instead of YAML 1.1's boolean.
    return cast(dict[str, Any], yaml.load((REPO_ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader))


@pytest.mark.parametrize("audit_exit", [0, 1, 2])
def test_nightly_audit_propagates_findings_and_collection_errors(tmp_path: Path, audit_exit: int) -> None:
    """Restoring || true hides both findings and errors; dropping --strict hides collection errors."""
    config = yaml.safe_load((REPO_ROOT / ".circleci/config.yml").read_text())
    assert config["workflows"]["nightly"]["jobs"] == ["dependency-audit"]
    audit = next(step["run"] for step in config["jobs"]["dependency-audit"]["steps"] if "run" in step)
    uv = tmp_path / ".local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'test "$1 $2 $3" = "run pip-audit --strict" || exit 99\n'
        'test "$4" = "--skip-editable" || exit 99\n'
        'echo "synthetic audit status ${AUDIT_EXIT}" >&2\n'
        'exit "${AUDIT_EXIT}"\n'
    )
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", audit["command"]],
        env={**os.environ, "HOME": str(tmp_path), "AUDIT_EXIT": str(audit_exit)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == audit_exit, result.stderr


@pytest.mark.parametrize("dispatch_exit", [0, 1])
def test_release_please_dispatches_exact_tag_using_existing_authority(tmp_path: Path, dispatch_exit: int) -> None:
    """Execute the production dispatch command; a suppressed tag push cannot replace it."""
    producer = workflow("release-please.yml")
    steps = producer["jobs"]["release-please"]["steps"]
    action = next(step for step in steps if step.get("uses", "").startswith("googleapis/release-please-action@"))
    dispatch = next(step for step in steps if "run" in step)
    assert re.fullmatch(r"googleapis/release-please-action@[0-9a-f]{40}", action["uses"])
    assert action["id"] == "release"
    assert action["with"]["token"] == "${{ secrets.GITHUB_TOKEN }}"
    assert dispatch["if"] == "steps.release.outputs.release_created == 'true'"
    assert dispatch["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "GH_REPO": "${{ github.repository }}",
        "RELEASE_TAG": "${{ steps.release.outputs.tag_name }}",
    }
    assert producer["permissions"]["actions"] == "write"
    gh = tmp_path / "gh"
    calls = tmp_path / "calls.jsonl"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(int(os.environ['DISPATCH_EXIT']))\n"
    )
    gh.chmod(0o755)
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", dispatch["run"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "RELEASE_TAG": "v1.2.3",
            "CALLS": str(calls),
            "DISPATCH_EXIT": str(dispatch_exit),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == dispatch_exit, result.stderr
    actual = [json.loads(line) for line in calls.read_text().splitlines()]
    expected = [
        ["workflow", "run", "release.yml", "--ref", "master", "-f", "release_tag=v1.2.3", "-f", "publish=true"],
        ["workflow", "run", "container.yml", "--ref", "master", "-f", "release_tag=v1.2.3", "-f", "push=true"],
        ["workflow", "run", "extension-release.yml", "--ref", "master", "-f", "release_tag=v1.2.3"],
        ["workflow", "run", "homebrew-bump.yml", "--ref", "master", "-f", "release_tag=v1.2.3"],
        ["workflow", "run", "flakehub.yml", "--ref", "v1.2.3"],
        ["workflow", "run", "cachix.yml", "--ref", "v1.2.3"],
    ]
    assert actual == (expected if dispatch_exit == 0 else expected[:1])
    for call in expected:
        consumer = workflow(call[2])
        assert "workflow_dispatch" in consumer["on"]
        if call[4] == "master":
            assert "release_tag" in consumer["on"]["workflow_dispatch"]["inputs"]
        if call[-1] in {"publish=true", "push=true"}:
            assert call[-1].split("=")[0] in consumer["on"]["workflow_dispatch"]["inputs"]


def test_exact_main_package_dependents_wait_for_successful_main_upload() -> None:
    """GitHub implicitly applies success() to needs; removing the edge permits an orphan MCP upload."""
    release = workflow("release.yml")
    dependents = []
    for package in sorted((REPO_ROOT / "packaging").glob("*/pyproject.toml")):
        project = tomllib.loads(package.read_text())["project"]
        if not any(dep.startswith("polylogue==") for dep in project.get("dependencies", [])):
            continue
        dependents.append(project["name"])
        job = release["jobs"][f"publish-pypi-{project['name'].removeprefix('polylogue-')}"]
        assert "publish-pypi" in job["needs"]
        # Status-check overrides could defeat the default success dependency.
        assert not any(token in job["if"] for token in ("always(", "failure(", "cancelled("))
        assert job["if"] == release["jobs"]["publish-pypi"]["if"]
    assert dependents == ["polylogue-mcp"]


def test_flakehub_tag_dispatch_publishes_the_selected_tag_instead_of_rolling() -> None:
    """The producer dispatches a tag ref; restoring event-name-only checks publishes a rolling build."""
    flakehub = workflow("flakehub.yml")
    push = next(step for step in flakehub["jobs"]["publish"]["steps"] if "with" in step)["with"]
    assert push["rolling"] == "${{ github.event_name == 'workflow_dispatch' && github.ref_type != 'tag' }}"
    assert (
        push["rolling-minor"]
        == "${{ github.event_name == 'workflow_dispatch' && github.ref_type != 'tag' && inputs.tag || '' }}"
    )
    assert push["tag"] == "${{ github.ref_type == 'tag' && github.ref_name || '' }}"

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
@pytest.mark.parametrize(
    "consumer", ["release.yml", "container.yml", "extension-release.yml", "flakehub.yml", "cachix.yml"]
)
def test_release_please_dispatches_exact_tag_using_existing_authority(
    tmp_path: Path, dispatch_exit: int, consumer: str
) -> None:
    """Execute each production lane, including failures, without contacting GitHub."""
    producer = workflow("release-please.yml")
    action = producer["jobs"]["release-please"]["steps"][0]
    assert re.fullmatch(r"googleapis/release-please-action@[0-9a-f]{40}", action["uses"])
    assert action["id"] == "release"
    assert action["with"]["token"] == "${{ secrets.GITHUB_TOKEN }}"
    assert producer["jobs"]["release-please"]["outputs"]["release_tag"] == (
        "${{ steps.release.outputs.release_created == 'true' && steps.release.outputs.tag_name || '' }}"
    )
    lane = producer["jobs"]["dispatch"]
    assert lane["needs"] == "release-please"
    assert lane["if"] == "needs.release-please.outputs.release_tag != ''"
    assert lane["strategy"]["fail-fast"] == "false"
    assert lane["strategy"]["matrix"]["workflow"] == [
        "release.yml",
        "container.yml",
        "extension-release.yml",
        "flakehub.yml",
        "cachix.yml",
    ]
    dispatch = lane["steps"][0]
    assert dispatch["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "GH_REPO": "${{ github.repository }}",
        "RELEASE_TAG": "${{ needs.release-please.outputs.release_tag }}",
        "WORKFLOW": "${{ matrix.workflow }}",
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
            "WORKFLOW": consumer,
            "CALLS": str(calls),
            "DISPATCH_EXIT": str(dispatch_exit),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == dispatch_exit, result.stderr
    expected = ["workflow", "run", consumer, "--ref", "v1.2.3"]
    if consumer == "release.yml":
        expected += ["-f", "release_tag=v1.2.3", "-f", "publish=true"]
    elif consumer == "container.yml":
        expected += ["-f", "push=true"]
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [expected]
    assert "workflow_dispatch" in workflow(consumer)["on"]


def test_homebrew_waits_for_main_publication_and_cdn_propagation(tmp_path: Path) -> None:
    """Removing the success edge races publication; a fixed polling cap rejects slow CDN propagation."""
    lane = workflow("release.yml")["jobs"]["dispatch-homebrew"]
    assert lane["needs"] == ["build", "publish-pypi"]
    assert "if" not in lane
    assert lane["permissions"]["actions"] == "write"
    dispatch = lane["steps"][0]
    assert dispatch["env"]["RELEASE_TAG"] == "v${{ needs.build.outputs.version }}"
    assert '--ref "${RELEASE_TAG}"' in dispatch["run"]
    bump = workflow("homebrew-bump.yml")["jobs"]["bump"]
    assert "timeout-minutes" not in bump
    poll = next(step["run"] for step in bump["steps"] if step.get("id") == "pypi")
    assert 'until curl -fsSLo "${sdist}" "${url}"; do' in poll
    assert "attempts=" not in poll
    assert "push" not in workflow("homebrew-bump.yml")["on"]
    curl = tmp_path / "curl"
    curl.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "count = pathlib.Path('count')\n"
        "n = int(count.read_text()) + 1 if count.exists() else 1\n"
        "count.write_text(str(n))\n"
        "if n <= 21: sys.exit(22)\n"
        "pathlib.Path(sys.argv[2]).write_text('synthetic sdist')\n"
    )
    curl.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 0\n")
    sleep.chmod(0o755)
    output = tmp_path / "outputs"
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", poll],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "VERSION": "1.2.3",
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "count").read_text() == "22"
    assert "sha256=" in output.read_text()


def test_every_action_in_the_release_fan_out_has_an_immutable_commit() -> None:
    """Moving a privileged action's tag must not change executed workflow code."""
    for name in ["release-please", "release", "container", "extension-release", "homebrew-bump", "flakehub", "cachix"]:
        for job in workflow(name + ".yml")["jobs"].values():
            for step in job["steps"]:
                if "uses" in step:
                    assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"]), (name, step["uses"])
    container = workflow("container.yml")["jobs"]["build-and-push"]["steps"]
    for step in container:
        if step.get("uses", "").startswith("docker/setup-qemu-action@"):
            assert re.fullmatch(r"docker.io/tonistiigi/binfmt@sha256:[0-9a-f]{64}", step["with"]["image"])
        if step.get("uses", "").startswith("docker/setup-buildx-action@"):
            assert re.fullmatch(r"image=moby/buildkit@sha256:[0-9a-f]{64}", step["with"]["driver-opts"])
            assert step["with"]["version"] == "v0.37.2"
    for name, job in workflow("release.yml")["jobs"].items():
        if not name.startswith("publish-pypi"):
            continue
        bootstrap = next(step for step in job["steps"] if step.get("name") == "Pin Sigstore bootstrap installer")
        assert "uv==0.12.21" in bootstrap["run"]
        assert "GITHUB_ENV" not in bootstrap["run"]
        signer = next(step for step in job["steps"] if step.get("uses", "").startswith("sigstore/"))
        assert signer["env"] == {
            "PIP_CONSTRAINT": "${{ runner.temp }}/sigstore-bootstrap-constraints.txt",
            "UV_CONSTRAINT": "${{ runner.temp }}/sigstore-bootstrap-constraints.txt",
        }
        publisher = next(step for step in job["steps"] if step.get("uses", "").startswith("pypa/"))
        assert "env" not in publisher
    cachix = workflow("cachix.yml")["jobs"]["push"]
    assert cachix["if"] == "${{ vars.CACHIX_CACHE_ENABLED == 'true' }}"
    action = next(step for step in cachix["steps"] if step.get("uses", "").startswith("cachix/cachix-action@"))
    assert action["with"]["installCommand"] == "nix profile install --inputs-from . nixpkgs#cachix"


@pytest.mark.parametrize("ref_type", ["tag", "branch"])
def test_homebrew_dispatch_preserves_tag_fallback_and_branch_recovery(tmp_path: Path, ref_type: str) -> None:
    """The real tag-push fallback has no GitHub Release; a recovery input wrongly requires one."""
    dispatch = workflow("release.yml")["jobs"]["dispatch-homebrew"]["steps"][0]
    calls = tmp_path / "calls.jsonl"
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    gh.chmod(0o755)
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", dispatch["run"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CALLS": str(calls),
            "RELEASE_TAG": "v1.2.3",
            "RECOVERY_TAG": "v1.2.3",
            "GITHUB_REF_TYPE": ref_type,
            "GITHUB_REF_NAME": "master" if ref_type == "branch" else "v1.2.3",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    expected = ["workflow", "run", "homebrew-bump.yml", "--ref", "master" if ref_type == "branch" else "v1.2.3"]
    if ref_type == "branch":
        expected += ["-f", "release_tag=v1.2.3"]
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [expected]


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
    push = next(
        step
        for step in flakehub["jobs"]["publish"]["steps"]
        if step.get("uses", "").startswith("DeterminateSystems/flakehub-push@")
    )["with"]
    assert push["rolling"] == "${{ github.event_name == 'workflow_dispatch' && github.ref_type != 'tag' }}"
    assert (
        push["rolling-minor"]
        == "${{ github.event_name == 'workflow_dispatch' && github.ref_type != 'tag' && inputs.tag || '' }}"
    )
    assert push["tag"] == "${{ github.ref_type == 'tag' && github.ref_name || '' }}"
    assert push["source-revision"] == "e001ee821cdb763ef120c01f1048bfb2f938bb9c"


@pytest.mark.parametrize(
    "job,kind", [("publish-pypi", "main"), ("publish-pypi-mcp", "mcp"), ("publish-pypi-hooks", "hooks")]
)
@pytest.mark.parametrize("mutation", ["unchanged", "tampered", "missing", "extra"])
def test_publishers_verify_the_build_jobs_staged_bytes(tmp_path: Path, job: str, kind: str, mutation: str) -> None:
    """Removing the production hash check permits changed bytes to be signed and uploaded."""
    import hashlib

    publisher = workflow("release.yml")["jobs"][job]
    verify = next(step for step in publisher["steps"] if step.get("name") == "Verify staged distribution hashes")
    assert verify["env"]["EXPECTED_HASHES"] == f"${{{{ needs.build.outputs.{kind}_hashes }}}}"
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "synthetic-1.0-py3-none-any.whl"
    wheel.write_bytes(b"synthetic wheel")
    expected = {wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest()}
    if mutation == "tampered":
        wheel.write_bytes(b"changed wheel")
    elif mutation == "missing":
        wheel.unlink()
    elif mutation == "extra":
        (dist / "unexpected.whl").write_bytes(b"extra")
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", verify["run"]],
        cwd=tmp_path,
        env={**os.environ, "EXPECTED_HASHES": json.dumps(expected)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is (mutation == "unchanged"), result.stderr


def test_extension_release_executes_only_lockfile_tools() -> None:
    """A mutable npx range or npm install fallback bypasses the committed executable lock."""
    package = json.loads((REPO_ROOT / "browser-extension/package.json").read_text())
    lock = json.loads((REPO_ROOT / "browser-extension/package-lock.json").read_text())
    for name in ("playwright", "addons-linter"):
        version = package["devDependencies"][name]
        assert re.fullmatch(r"\d+\.\d+\.\d+", version)
        assert lock["packages"][""]["devDependencies"][name] == version
        locked = lock["packages"][f"node_modules/{name}"]
        assert locked["version"] == version
        assert locked["integrity"].startswith("sha512-")
    build_steps = workflow("extension-release.yml")["jobs"]["build"]["steps"]
    install = next(step for step in build_steps if step.get("name") == "Install dependencies")
    assert install["run"] == "npm ci"
    assert (
        next(step for step in build_steps if step.get("name") == "Install Playwright Chromium")["run"]
        == 'npm run --prefix "${RELEASE_TOOLS}" install:screenshot-browser'
    )
    lint = next(step for step in build_steps if step.get("name") == "Lint Firefox xpi")["run"]
    assert '"${RELEASE_TOOLS}/node_modules/.bin/addons-linter" dist/firefox-unpacked' in lint
    assert "npx" not in "\n".join(step.get("run", "") for step in build_steps)
    assert package["scripts"]["install:screenshot-browser"] == "playwright install --with-deps chromium"


def test_build_hash_outputs_cover_each_staged_distribution(tmp_path: Path) -> None:
    import hashlib

    build = workflow("release.yml")["jobs"]["build"]
    hashes = next(step for step in build["steps"] if step.get("id") == "hashes")
    expected = {}
    for kind, directory in [("main", "staged-dist"), ("mcp", "staged-dist-mcp"), ("hooks", "staged-dist-hooks")]:
        stage = tmp_path / directory
        stage.mkdir()
        wheel = stage / f"synthetic-{kind}.whl"
        wheel.write_bytes(kind.encode())
        expected[kind] = {wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest()}
        assert build["outputs"][f"{kind}_hashes"] == f"${{{{ steps.hashes.outputs.{kind} }}}}"
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", hashes["run"]],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert {
        key: json.loads(value) for key, value in (line.split("=", 1) for line in output.read_text().splitlines())
    } == expected


def test_recovery_toolchains_belong_to_the_workflow_revision() -> None:
    """Using the old product tag for new tool files breaks recovery before it can build."""
    for name in ("release.yml", "extension-release.yml"):
        jobs = workflow(name)["jobs"]
        toolchain = jobs["release-toolchain"]
        assert toolchain["steps"][0]["with"]["ref"] == "${{ github.workflow_sha }}"
        assert jobs["build"]["needs"] == "release-toolchain"
        product = jobs["build"]["steps"][0]["with"]["ref"]
        assert product == "${{ inputs.release_tag && format('refs/tags/{0}', inputs.release_tag) || github.ref }}"
    source = (REPO_ROOT / "packaging/Containerfile").read_text()
    external_images = re.findall(r"(?:^FROM |COPY --from=)([^\s]+)", source, re.MULTILINE)
    assert len(external_images) == 11
    assert all("@sha256:" in image for image in external_images if image not in {"builder", "runtime"})
    downloads = json.loads((REPO_ROOT / "packaging/python-downloads.json").read_text())
    assert set(downloads) == {
        "cpython-3.14.7+freethreaded-linux-x86_64-gnu",
        "cpython-3.14.7+freethreaded-linux-aarch64-gnu",
    }
    assert all(
        item["variant"] == "freethreaded" and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        for item in downloads.values()
    )

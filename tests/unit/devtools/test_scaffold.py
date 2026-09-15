"""Contract for ``devtools schema new``: the interview, the refusals, and the writes.

Anti-vacuity: each test names the mutation that turns it red -- an unanswered
compatibility dimension that is accepted anyway, a duplicate family created
without justification, a new durable object created silently, an overwritten
bundle, a partial bundle surviving a failed write, or generation that is not
byte-for-byte reproducible.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from devtools import scaffold
from devtools.command_catalog import COMMANDS
from polylogue.declarations import CompatibilityKey

ROOT = Path(__file__).resolve().parents[3]

_MCP_ANSWERS = CompatibilityKey(
    identity="mcp-tool:sample",
    lifecycle="registered-handler-retained",
    authority="mcp-capability:read",
    access_result_shape="query:exhaustive_page:envelope:items",
    durability="transport-adapter; domain-owner-controls-durability",
)
_ROUTE_ANSWERS = CompatibilityKey(
    identity="daemon-route",
    lifecycle="stable",
    authority="daemon-read",
    access_result_shape="json-envelope",
    durability="read-only",
)


def _plan(tmp_path: Path, family: str = "mcp-tool", **kwargs: object) -> scaffold.ScaffoldPlan:
    answers = kwargs.pop("compatibility", _MCP_ANSWERS if family == "mcp-tool" else _ROUTE_ANSWERS)
    return scaffold.build_plan(
        family,
        str(kwargs.pop("name", "sample_tool")),
        compatibility=answers,  # type: ignore[arg-type]
        output_root=tmp_path,
        **kwargs,  # type: ignore[arg-type]
    )


def test_command_is_registered() -> None:
    assert COMMANDS["schema new"].module == "devtools.scaffold"


def test_incomplete_interview_names_the_missing_dimension(tmp_path: Path) -> None:
    incomplete = CompatibilityKey("id", "", "read", "shape", "durability-token")
    with pytest.raises(scaffold.ScaffoldError) as excinfo:
        _plan(tmp_path, compatibility=incomplete)
    assert "--lifecycle" in str(excinfo.value)


def test_existing_compatible_family_is_refused_without_justification(tmp_path: Path) -> None:
    """Re-declaring an existing family's exact key must point at reuse."""

    from polylogue.mcp.declarations.registry import MCP_KERNEL_REGISTRY

    existing = MCP_KERNEL_REGISTRY.families()[0]
    with pytest.raises(scaffold.ScaffoldError) as excinfo:
        _plan(tmp_path, compatibility=existing.compatibility)
    message = str(excinfo.value)
    assert "--reuse-family" in message and existing.family_id in message


def test_reuse_family_must_be_declared_and_compatible(tmp_path: Path) -> None:
    from polylogue.mcp.declarations.registry import MCP_KERNEL_REGISTRY

    existing = MCP_KERNEL_REGISTRY.families()[0]
    with pytest.raises(scaffold.ScaffoldError) as excinfo:
        _plan(tmp_path, reuse_family="family.that.does.not.exist")
    assert "not a declared family" in str(excinfo.value)

    with pytest.raises(scaffold.ScaffoldError) as excinfo:
        _plan(tmp_path, reuse_family=existing.family_id)
    assert "incompatible" in str(excinfo.value)


def test_new_durable_object_is_refused_without_justification(tmp_path: Path) -> None:
    durable = CompatibilityKey("new-thing", "retained", "write", "single-object", "durable")
    with pytest.raises(scaffold.ScaffoldError) as excinfo:
        _plan(tmp_path, compatibility=durable)
    assert "--justification" in str(excinfo.value)

    plan = _plan(tmp_path, compatibility=durable, justification="new durable annotation registry, approved")
    assert plan.answers.justification


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = _plan(tmp_path)
    second = _plan(tmp_path)
    assert [(item.relative_path, item.content) for item in first.files] == [
        (item.relative_path, item.content) for item in second.files
    ]


def test_apply_writes_the_bundle_and_refuses_collisions(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    written = scaffold.apply_plan(plan, root=tmp_path)
    assert {path.name for path in written} == {
        "declaration.py",
        "adapter.py",
        "test_sample_tool_route_contract.py",
        "PLAN.md",
    }
    with pytest.raises(scaffold.ScaffoldError) as excinfo:
        scaffold.apply_plan(plan, root=tmp_path)
    assert "refusing to overwrite" in str(excinfo.value)


def test_partial_write_failure_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure mid-bundle leaves no half-written scaffold behind."""

    plan = _plan(tmp_path)
    original = Path.write_text
    calls = {"count": 0}

    def flaky(self: Path, *args: object, **kwargs: object) -> int:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("disk full")
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", flaky)
    with pytest.raises(OSError):
        scaffold.apply_plan(plan, root=tmp_path)
    monkeypatch.undo()
    target = tmp_path / plan.output_root
    assert not any(target.glob("*.py")) and not (target / "PLAN.md").exists()


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    exit_code = scaffold.main(
        [
            "mcp-tool",
            "sample_tool",
            "--identity",
            _MCP_ANSWERS.identity,
            "--lifecycle",
            _MCP_ANSWERS.lifecycle,
            "--authority",
            _MCP_ANSWERS.authority,
            "--access-result-shape",
            _MCP_ANSWERS.access_result_shape,
            "--durability",
            _MCP_ANSWERS.durability,
            "--dry-run",
            "--output-root",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert not list(tmp_path.rglob("*.py"))


@pytest.mark.parametrize("family", ["mcp-tool", "daemon-route"])
def test_generated_bundle_compiles(tmp_path: Path, family: str) -> None:
    """Both families generate compiling stubs from the same protocol."""

    plan = _plan(tmp_path, family=family, name="sample_tool" if family == "mcp-tool" else "sample_route")
    scaffold.apply_plan(plan, root=tmp_path)
    target = tmp_path / plan.output_root
    sources = sorted(target.glob("*.py"))
    assert sources
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", *(str(path) for path in sources)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_generated_declaration_passes_production_diagnostics(tmp_path: Path) -> None:
    """The generated record resolves through the production diagnostics.

    Anti-vacuity: drop the handler binding or the owner path from the template
    and this reports ``unresolved_handler_path``.
    """

    import importlib.util

    from polylogue.declarations import DeclarationRegistry
    from polylogue.declarations.diagnostics import diagnose_registry

    plan = _plan(tmp_path, family="daemon-route", name="sample_route")
    scaffold.apply_plan(plan, root=tmp_path)
    module_path = tmp_path / plan.output_root / "declaration.py"
    spec = importlib.util.spec_from_file_location("scaffolded_declaration", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    registry = DeclarationRegistry()
    registry.register(module.DECLARATION)
    diagnostics = diagnose_registry(registry, root=ROOT)
    assert not diagnostics, [item.message for item in diagnostics]


def test_validate_reports_the_live_owning_registry() -> None:
    assert scaffold.validate_owner_registry("mcp-tool", root=ROOT) == ()
    assert scaffold.validate_owner_registry("daemon-route", root=ROOT) == ()


def test_generated_mcp_contract_executes_through_the_production_adapter(tmp_path: Path) -> None:
    """The generated skeleton's assertions run against production code.

    Anti-vacuity: point the template at a local validator instead of
    ``DeclaredToolRegistrar`` and the refusal assertion stops holding.
    """

    import importlib.util

    plan = _plan(tmp_path)
    scaffold.apply_plan(plan, root=tmp_path)
    module_path = tmp_path / plan.output_root / "test_sample_tool_route_contract.py"
    spec = importlib.util.spec_from_file_location("scaffolded_contract", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from polylogue.mcp.declarations.adapter import DeclaredToolRegistrar

    assert module.DeclaredToolRegistrar is DeclaredToolRegistrar
    # The invalid invocation must be refused by the production adapter.
    module.test_invalid_invocation_is_refused_by_the_production_adapter()
    # The minimal-valid invocation is gated on landing the declaration; before
    # that, the production adapter refuses it and the skeleton skips with the
    # landing step named.
    with pytest.raises(BaseException) as excinfo:
        module.test_minimal_valid_invocation_through_the_production_adapter()
    assert "not declared yet" in str(excinfo.value)

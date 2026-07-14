"""Import-resolution drift gate for operation/artifact catalog ``code_refs``.

``OperationSpec.code_refs`` (``polylogue/operations/specs.py``) and
``ArtifactNode.code_refs`` (``polylogue/artifacts/runtime.py``) are dotted-path
strings that attribute verification-catalog coverage to concrete source. Prior
to this test, no runtime or test code ever *import-resolved* those strings:
``tests/unit/operations/test_specs.py`` only asserted non-emptiness,
``tests/unit/daemon/test_daemon_http_contracts.py`` only regex-shape-checked
them, and ``tests/unit/mcp/test_per_tool_contracts.py`` cross-checks only the
``server_mutation_tools.*`` slice by last path segment. A rename or move of a
referenced symbol (e.g. renaming
``storage.insights.session.rebuild.rebuild_session_insights_sync``) silently
strands the catalog ref — coverage attribution decays invisibly, the same
failure family as the api/contracts shadow-adapter drift (polylogue-a7xr.13).

This test closes that gap: for every non-empty ``code_ref`` across
``RUNTIME_OPERATION_SPECS``, ``DECLARED_CONTROL_PLANE_OPERATION_SPECS``, and
the artifact graph nodes, split on dots, ``importlib.import_module`` the
longest importable module prefix, then ``getattr``-chain the remainder
(handles ``Class.method`` refs like
``SqliteVecRuntimeMixin._ensure_tables``). No allowlist is needed today:
every currently-declared ref is a real, resolvable symbol (there are no
deliberately aspirational refs as of writing) — if one is later added
on purpose, it belongs behind an explicit ``aspirational_refs`` field on the
spec/node, not a skip in this test (per polylogue-a7xr.17's design note).

Ref polylogue-a7xr.17.
"""

from __future__ import annotations

import importlib
import inspect
import re

import pytest

from polylogue.artifacts.runtime import RUNTIME_ARTIFACT_NODES
from polylogue.operations.specs import (
    DECLARED_CONTROL_PLANE_OPERATION_SPECS,
    RUNTIME_OPERATION_SPECS,
)


class _NestedFunctionRef:
    """Marker sentinel: an attr resolved via the nested-def source-scan fallback."""

    def __init__(self, name: str) -> None:
        self.name = name


def _resolve_attr(obj: object, attr: str) -> object:
    """Resolve one ``.attr`` step, with two fallbacks getattr alone can't see.

    - **Pydantic model fields with no default** (e.g. ``forensic_index:
      ForensicIndex`` on a ``BaseModel`` subclass): pydantic v2 does not set
      these as class attributes, so ``getattr(Model, "forensic_index")``
      raises ``AttributeError`` even though the field is real. Fall back to
      ``model_fields`` (or legacy ``__fields__``).
    - **Closure-scoped functions**: this codebase defines MCP tools as nested
      functions inside ``register_*`` factories (``register_mutation_tools``,
      ``register_personal_state_tools``, ...) rather than as module globals —
      by design, not by accident (see CLAUDE.md's control-center notes). They
      are real, live, callable symbols but unreachable via getattr. Fall back
      to a textual scan of the owning module's source for a top-level
      ``def {attr}(``/``async def {attr}(``. This still catches a rename (the
      old name stops appearing anywhere in the module) even though — unlike
      getattr — it cannot verify the def is reachable at runtime.
    """

    try:
        return getattr(obj, attr)
    except AttributeError as exc:
        model_fields = getattr(obj, "model_fields", None) or getattr(obj, "__fields__", None)
        if isinstance(model_fields, dict) and attr in model_fields:
            return model_fields[attr]
        if inspect.ismodule(obj):
            source = inspect.getsource(obj)
            if re.search(rf"^\s*(async def|def)\s+{re.escape(attr)}\s*\(", source, re.MULTILINE):
                return _NestedFunctionRef(attr)
        raise exc


def _resolve_code_ref(ref: str) -> object:
    """Import-resolve a dotted ``module.sub.Class.attr`` code_ref.

    Some refs name a module in its own right (``pkg.mod.submodule``, or a test
    module such as ``tests.unit.cli.test_demo_command``) rather than an
    attribute within one — ``importlib.import_module`` on the whole ref
    handles that case directly (submodules are not guaranteed to be bound as
    attributes of their parent package purely from importing the parent).
    Otherwise, try the longest importable module prefix first (so
    ``pkg.mod.Class.method`` resolves ``pkg.mod`` as the module and walks
    ``Class`` then ``method`` via getattr), falling back to shorter prefixes.
    """

    parts = ref.split(".")
    if len(parts) < 2:
        raise ValueError(f"code_ref has no dotted module component: {ref!r}")

    try:
        return importlib.import_module(ref)
    except ImportError:
        pass

    last_error: Exception | None = None
    for split_index in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:split_index])
        attr_path = parts[split_index:]
        try:
            obj: object = importlib.import_module(module_path)
        except ImportError as exc:
            last_error = exc
            continue
        try:
            for attr in attr_path:
                obj = _resolve_attr(obj, attr)
        except AttributeError as exc:
            raise AttributeError(f"{ref!r}: module {module_path!r} imported but {attr_path!r} not found") from exc
        return obj

    raise ImportError(f"{ref!r}: no importable module prefix found") from last_error


def _all_declared_code_refs() -> list[tuple[str, str]]:
    """Return (owner_label, code_ref) pairs for every non-empty declared ref."""

    pairs: list[tuple[str, str]] = []
    for spec in RUNTIME_OPERATION_SPECS:
        for ref in spec.code_refs:
            pairs.append((f"RUNTIME_OPERATION_SPECS[{spec.name!r}]", ref))
    for spec in DECLARED_CONTROL_PLANE_OPERATION_SPECS:
        for ref in spec.code_refs:
            pairs.append((f"DECLARED_CONTROL_PLANE_OPERATION_SPECS[{spec.name!r}]", ref))
    for node in RUNTIME_ARTIFACT_NODES:
        for ref in node.code_refs:
            pairs.append((f"RUNTIME_ARTIFACT_NODES[{node.name!r}]", ref))
    return pairs


_DECLARED_CODE_REFS = _all_declared_code_refs()


def test_declared_code_refs_are_non_empty() -> None:
    """Fixture sanity: the catalogs must actually declare refs, or this test is vacuous."""

    assert _DECLARED_CODE_REFS, "no code_refs found across operation/artifact catalogs — catalogs changed shape?"
    assert len(_DECLARED_CODE_REFS) > 50, (
        f"only {len(_DECLARED_CODE_REFS)} code_refs found; expected the catalogs' full breadth "
        "(RUNTIME_OPERATION_SPECS + DECLARED_CONTROL_PLANE_OPERATION_SPECS + RUNTIME_ARTIFACT_NODES)"
    )


@pytest.mark.parametrize(
    "owner,ref",
    _DECLARED_CODE_REFS,
    ids=[f"{owner}::{ref}" for owner, ref in _DECLARED_CODE_REFS],
)
def test_code_ref_resolves_to_a_real_symbol(owner: str, ref: str) -> None:
    """Every declared code_ref must import-resolve to a live symbol.

    Renaming or moving any referenced symbol without updating its catalog
    entry fails this test, naming both the owning spec/node and the stale ref.
    """

    try:
        _resolve_code_ref(ref)
    except (ImportError, AttributeError, ValueError) as exc:
        pytest.fail(f"{owner} declares unresolvable code_ref {ref!r}: {exc}")

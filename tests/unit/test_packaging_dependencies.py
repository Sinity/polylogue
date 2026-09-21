"""Packaging declarations that decide content-hash bytes, not just speed.

``polylogue.core.digest.IDENTITY`` -- the profile behind material-protocol v1
canonical record and manifest bytes -- declares ``encoder="core-json"``, so its
bytes come from whichever backend ``polylogue.core.json`` selected at import
time. The two backends do not agree: stdlib ``json`` switches to exponent
notation at a different magnitude than msgspec (and than the orjson formatter
every pre-existing archive hash was computed under), so the same payload gets a
different SHA-256 depending only on whether msgspec happened to be installed.

That makes msgspec a *correctness* dependency. While it sat in an optional
``speed`` extra, ``pip install polylogue`` produced an install that silently
hashed differently from every other one. These tests pin the declaration that
closes that hole, and the divergence that makes the declaration load-bearing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import tomllib

from polylogue.core import json as core_json
from polylogue.core.digest import IDENTITY, canonical_bytes, digest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: A payload whose canonical bytes differ between the two backends. msgspec
#: writes the decimal expansion (``0.00001``); stdlib json crosses to exponent
#: notation here and writes ``1e-05``. Both are valid JSON for the same double,
#: which is exactly why the divergence is silent.
_WITNESS: dict[str, float] = {"x": 0.00001}

#: The canonical IDENTITY bytes of :data:`_WITNESS` under msgspec. Pinned as a
#: literal so a future backend or formatter change that moves them has to say
#: so here rather than silently re-hash the archive.
_WITNESS_MSGSPEC_BYTES = b'{"x":0.00001}'


def _project_metadata() -> dict[str, Any]:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return dict(payload["project"])


def _requirement_names(specifiers: list[str]) -> set[str]:
    return {re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0].strip().lower() for spec in specifiers}


def test_msgspec_is_declared_a_base_dependency() -> None:
    """``pip install polylogue`` must install the backend the digest assumes.

    Anti-vacuity: move ``msgspec`` back out of ``[project] dependencies`` into
    any extra (its previous ``speed`` home) and this goes red. It reads the
    real pyproject.toml, so it cannot pass against a stale copy of the
    declaration.
    """
    metadata = _project_metadata()
    assert "msgspec" in _requirement_names(list(metadata["dependencies"])), (
        "msgspec decides IDENTITY digest bytes; a base install without it hashes differently"
    )


def test_no_extra_makes_msgspec_optional() -> None:
    """No optional group may re-offer msgspec as a selectable accelerator.

    A base dependency that is *also* an extra invites the reading that the
    extra is what installs it, which is the shape this change removed.

    Anti-vacuity: re-add ``speed = ["msgspec>=0.20.0"]`` (or name msgspec in
    any other extra) and this goes red.
    """
    extras: dict[str, list[str]] = _project_metadata().get("optional-dependencies", {})
    offering = sorted(name for name, specs in extras.items() if "msgspec" in _requirement_names(list(specs)))
    assert offering == [], f"extras still present msgspec as optional: {offering}"


def test_the_nix_package_carries_msgspec() -> None:
    """The packaged (non-pip) build installs the same backend.

    flake.nix builds the shipped program from its own dependency list rather
    than from pyproject, so the two can drift apart. A packaged daemon that
    lost msgspec would hash unlike every pip install of the same commit.

    Anti-vacuity: delete ``msgspec`` from the program's ``dependencies = with
    pythonPackages; [...]`` list in flake.nix and this goes red.
    """
    flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
    # flake.nix defines several Python packages; anchor on the program's own
    # derivation rather than the first dependency list in the file.
    program = flake.index('polylogue = pythonPackages.buildPythonPackage {\n        pname = "polylogue";')
    start = flake.index("dependencies = with pythonPackages; [", program)
    end = flake.index("];", start)
    entries = {line.strip() for line in flake[start:end].splitlines()}
    assert "msgspec" in entries


def test_identity_bytes_come_from_the_pluggable_backend() -> None:
    """Pin the coupling the dependency declaration exists to protect.

    Anti-vacuity: change ``IDENTITY.encoder`` to ``"stdlib"`` (which would make
    the profile backend-independent and every existing material-protocol hash
    wrong) and this goes red.
    """
    assert IDENTITY.encoder == "core-json"


def test_the_active_backend_is_msgspec() -> None:
    """A supported install never runs the digest on the stdlib formatter.

    Anti-vacuity: run this in an interpreter without msgspec -- precisely the
    ``pip install polylogue`` that the old ``speed`` extra produced -- and it
    goes red instead of silently hashing differently. It asserts the observed
    import-time selection, not a re-read of the declaration, so a declaration
    that failed to reach the environment does not satisfy it.
    """
    assert core_json.backend() == "msgspec"
    assert canonical_bytes(_WITNESS, IDENTITY) == _WITNESS_MSGSPEC_BYTES


@pytest.mark.skipif(
    "stdlib" not in core_json.available_backends(),
    reason="the stdlib backend module is unavailable in this interpreter",
)
def test_the_stdlib_backend_would_produce_a_different_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Show that losing msgspec changes the digest rather than degrading gracefully.

    This is the defect the dependency promotion closes, kept executable so the
    claim is not just a comment. Forcing the backend is the same monkeypatch
    idiom ``tests/unit/core/test_json.py`` uses for cross-backend parity.

    Anti-vacuity: if the two formatters were ever reconciled so canonical bytes
    no longer depend on the backend, this goes red -- and then the base
    dependency's stated justification, here and in pyproject.toml, needs
    rewriting rather than the test relaxing.
    """
    msgspec_bytes = canonical_bytes(_WITNESS, IDENTITY)
    msgspec_digest = digest(_WITNESS, IDENTITY)

    monkeypatch.setattr(core_json, "_BACKEND", "stdlib")
    stdlib_bytes = canonical_bytes(_WITNESS, IDENTITY)
    stdlib_digest = digest(_WITNESS, IDENTITY)

    assert msgspec_bytes == _WITNESS_MSGSPEC_BYTES
    assert stdlib_bytes == b'{"x":1e-05}'
    assert stdlib_bytes != msgspec_bytes
    assert stdlib_digest != msgspec_digest

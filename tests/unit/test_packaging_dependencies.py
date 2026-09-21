"""Packaging declarations that decide content-hash bytes, not just speed.

``polylogue.core.digest.IDENTITY`` -- the profile behind material-protocol v1
canonical record and manifest bytes -- declares ``encoder="core-json"``, so its
bytes are msgspec's. stdlib ``json`` does not agree: it switches to exponent
notation at a different magnitude (and than the orjson formatter every
pre-existing archive hash was computed under), so the same payload would get a
different SHA-256 under it.

That makes msgspec a *correctness* dependency. While it sat in an optional
``speed`` extra, ``pip install polylogue`` produced an install that silently
hashed differently from every other one. Three declarations now have to agree
before such an install can exist -- ``[project] dependencies``, the flake's own
program list, and ``polylogue.runtime.REQUIRED_NATIVE_PACKAGES`` -- and
``polylogue.core.json`` imports msgspec unconditionally so a fourth, silent
route is gone. These tests pin all of that, and the divergence that makes it
load-bearing.
"""

from __future__ import annotations

import hashlib
import json as _stdlib_json
import re
from pathlib import Path
from typing import Any

import tomllib

from polylogue.core.digest import IDENTITY, canonical_bytes, digest
from polylogue.runtime import REQUIRED_NATIVE_PACKAGES, probe_extensions, runtime_report

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


def test_the_digest_runs_on_msgspec_bytes() -> None:
    """A supported install never runs the digest on the stdlib formatter.

    Anti-vacuity: this asserts the bytes the live interpreter actually
    produces, not a re-read of a declaration, so a declaration that failed to
    reach the environment does not satisfy it. In an interpreter without
    msgspec, importing ``polylogue.core.json`` now fails outright and this
    errors rather than silently hashing differently.
    """
    assert canonical_bytes(_WITNESS, IDENTITY) == _WITNESS_MSGSPEC_BYTES


def test_msgspec_is_a_required_runtime_extension() -> None:
    """The startup contract refuses the install a lost msgspec would produce.

    Declaring the dependency is not the same as observing it: every guarded
    console script calls ``require_free_threaded_runtime`` before archive or
    network work, and that check only refuses for packages it is told are
    required. msgspec sat in an OPTIONAL tier while being identity-bearing,
    which made ``runtime_report()`` report ``pass`` on an install that would
    write foreign content hashes.

    Anti-vacuity: move ``msgspec`` back to an optional tier, or drop it from
    ``REQUIRED_NATIVE_PACKAGES``, and the probe below stops being taken and
    this goes red.
    """
    assert "msgspec" in REQUIRED_NATIVE_PACKAGES
    probed = {probe.name: probe for probe in probe_extensions()}
    assert "msgspec" in probed, "the contract must actually probe what it declares required"
    assert probed["msgspec"].safe
    assert runtime_report()["extensions_safe"] is True


def test_losing_msgspec_would_change_the_identity_rather_than_degrade() -> None:
    """Why the dependency is load-bearing, kept executable rather than asserted in prose.

    ``polylogue.core.json`` no longer has a stdlib codec to select, so this
    compares the canonical bytes against what stdlib json would have written
    for the same payload. That is the divergence the promotion closes.

    Anti-vacuity: if the two formatters were ever reconciled so canonical bytes
    no longer depend on the codec, this goes red -- and then the base
    dependency's stated justification, here and in pyproject.toml, needs
    rewriting rather than the test relaxing.
    """
    msgspec_bytes = canonical_bytes(_WITNESS, IDENTITY)
    stdlib_bytes = _stdlib_json.dumps(_WITNESS, sort_keys=True, separators=(",", ":")).encode("utf-8")

    assert msgspec_bytes == _WITNESS_MSGSPEC_BYTES
    assert stdlib_bytes == b'{"x":1e-05}'
    assert stdlib_bytes != msgspec_bytes
    assert digest(_WITNESS, IDENTITY) != hashlib.sha256(stdlib_bytes).hexdigest()

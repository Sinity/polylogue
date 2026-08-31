"""Immutable authority artifacts for selected verification.

pytest-testmon is a useful execution cache, but its SQLite database is a
mutable implementation detail.  This module records the authority which makes
one selected run meaningful: a canonical environment identity, an independent
corpus attestation, and explicit parent/child lineage.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from devtools.testmon_bootstrap import canonical_test_nodeid

GRAPH_PROTOCOL = 1
GRAPH_ROOTS = Path(".cache/verify/graph/roots")
GRAPH_CHILDREN = Path(".cache/verify/graph/children")
# A worktree may read immutable roots from another checkout (normally the main
# checkout that ran the complete corpus) instead of copying them. The roots are
# content-addressed and eligibility is re-validated on every read, so sharing
# is safe by construction; publication never writes through the indirection.
GRAPH_PARENT_ENV = "POLYLOGUE_VERIFY_GRAPH_PARENT"


def _root_bases(root: Path) -> tuple[Path, ...]:
    """Directories that may hold immutable graph roots, local first."""
    bases = [root / GRAPH_ROOTS]
    external = os.environ.get(GRAPH_PARENT_ENV)
    if external:
        candidate = Path(external)
        if candidate.name != "roots":
            candidate = candidate / GRAPH_ROOTS
        if candidate != bases[0]:
            bases.append(candidate)
    return tuple(bases)


# These are deliberately explicit.  Adding an input is an authority change,
# and the mutation matrix can enumerate this list rather than trusting a
# comment or an incidental dict implementation detail.
GRAPH_IDENTITY_INPUTS = (
    "tree",
    "code",
    "dependency_lock",
    "installed_distributions",
    "interpreter",
    "toolchain",
    "plugins",
    "harness",
    "corpus_attestation",
    "schemas",
    "configuration",
    "platform",
    "execution_policy",
)


class VerificationGraphError(RuntimeError):
    """The immutable verification authority cannot be trusted."""


def corpus_attestation_matches(
    expected: CorpusAttestation,
    nodeids: Iterable[str],
    *,
    fixture_files: Iterable[Path] = (),
    root: Path | None = None,
) -> bool:
    """Independently re-collect the authority inputs and compare exact bytes."""
    observed = attest_corpus(nodeids, fixture_files=fixture_files, root=root)
    return observed == expected


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [_canonical(item) for item in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"verification graph identity is not canonical: {type(value).__name__}")


def _digest(value: object) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CorpusAttestation:
    """Independent digest of the collected corpus and its fixture inputs."""

    digest: str
    nodeids: tuple[str, ...]
    fixture_digest: str


def attest_corpus(
    nodeids: Iterable[str],
    *,
    fixture_files: Iterable[Path] = (),
    root: Path | None = None,
) -> CorpusAttestation:
    """Hash collection output and fixture bytes without consulting testmon."""
    normalized = tuple(sorted({canonical_test_nodeid(str(nodeid)) for nodeid in nodeids}))
    fixture_rows: list[dict[str, str]] = []
    base = (root or Path.cwd()).resolve()
    for raw_path in sorted((Path(path) for path in fixture_files), key=lambda path: path.as_posix()):
        path = raw_path if raw_path.is_absolute() else base / raw_path
        relative = path.resolve().relative_to(base).as_posix()
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise VerificationGraphError(f"cannot attest corpus fixture {relative}: {exc}") from exc
        fixture_rows.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest()})
    fixture_digest = _digest(fixture_rows)
    identity_payload: dict[str, object] = {
        "protocol": GRAPH_PROTOCOL,
        "nodeids": normalized,
        "fixture_digest": fixture_digest,
    }
    return CorpusAttestation(_digest(identity_payload), normalized, fixture_digest)


def graph_identity(
    *,
    tree: object,
    code: object,
    dependency_lock: object,
    installed_distributions: object,
    interpreter: object,
    toolchain: object,
    plugins: object,
    harness: object,
    corpus_attestation: str,
    schemas: object,
    configuration: object,
    platform: object,
    execution_policy: object,
) -> str:
    """Return the content identity of every named verification input.

    No timestamps, process ids, random values, or paths outside the supplied
    authority inputs participate in this digest.
    """
    values = {
        "tree": tree,
        "code": code,
        "dependency_lock": dependency_lock,
        "installed_distributions": installed_distributions,
        "interpreter": interpreter,
        "toolchain": toolchain,
        "plugins": plugins,
        "harness": harness,
        "corpus_attestation": corpus_attestation,
        "schemas": schemas,
        "configuration": configuration,
        "platform": platform,
        "execution_policy": execution_policy,
    }
    payload = {name: values[name] for name in GRAPH_IDENTITY_INPUTS}
    return _digest({"protocol": GRAPH_PROTOCOL, "inputs": payload})


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(_canonical(payload), handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def publish_complete_root(
    root: Path,
    *,
    graph_digest: str,
    corpus: CorpusAttestation,
    run_id: str,
    terminal_status: str,
    complete: bool,
    selected: bool = False,
) -> Path | None:
    """Publish one eligible root only after all terminal checks pass."""
    if terminal_status != "success" or not complete or selected:
        return None
    payload = {
        "protocol": GRAPH_PROTOCOL,
        "kind": "verification-graph-root",
        "graph_digest": graph_digest,
        "corpus": {"digest": corpus.digest, "nodeids": corpus.nodeids, "fixture_digest": corpus.fixture_digest},
        "run_id": run_id,
        "terminal_status": terminal_status,
        "complete": True,
        "selected": False,
    }
    destination = root / GRAPH_ROOTS / graph_digest / "manifest.json"
    if destination.exists():
        existing = _read(destination)
        comparable_existing = dict(existing or {})
        comparable_expected = dict(payload)
        # Provenance is useful, but the root is content-addressed by the
        # graph and corpus. A retry of the same complete run must converge on
        # the already-published immutable object.
        comparable_existing.pop("run_id", None)
        comparable_expected.pop("run_id", None)
        if comparable_existing != _canonical(comparable_expected):
            raise VerificationGraphError(f"immutable graph root collision: {graph_digest}")
        return destination
    _atomic_json(destination, payload)
    return destination


def publish_selected_child(
    root: Path,
    *,
    parent_digest: str,
    graph_digest: str,
    selection: Sequence[str],
    change_lineage: Mapping[str, object],
    outcome: Mapping[str, object],
    run_id: str,
) -> Path:
    """Record selected evidence against an existing immutable parent root."""
    parent = root / GRAPH_ROOTS / parent_digest / "manifest.json"
    parent_payload = _read(parent)
    if not parent_payload or parent_payload.get("selected") is not False or parent_payload.get("complete") is not True:
        raise VerificationGraphError("selected verification requires an eligible immutable parent root")
    normalized = tuple(sorted({canonical_test_nodeid(str(nodeid)) for nodeid in selection}))
    payload = {
        "protocol": GRAPH_PROTOCOL,
        "kind": "verification-graph-child",
        "graph_digest": graph_digest,
        "parent_digest": parent_digest,
        "selection": normalized,
        "change_lineage": _canonical(change_lineage),
        "outcome": _canonical(outcome),
        "run_id": run_id,
        "selected": True,
    }
    destination = root / GRAPH_CHILDREN / f"{graph_digest}-{run_id}" / "manifest.json"
    _atomic_json(destination, payload)
    return destination


def eligible_root(root: Path, graph_digest: str) -> Path | None:
    """Return a root only when its immutable manifest is structurally eligible."""
    for base in _root_bases(root):
        path = base / graph_digest / "manifest.json"
        payload = _read(path)
        if payload is None or payload.get("kind") != "verification-graph-root":
            continue
        if payload.get("graph_digest") != graph_digest or payload.get("complete") is not True:
            continue
        if payload.get("terminal_status") != "success" or payload.get("selected") is not False:
            continue
        return path
    return None


def latest_eligible_root(root: Path) -> tuple[str, Path] | None:
    """Find the newest eligible root without treating directory order as authority."""
    candidates: list[tuple[str, float, Path]] = []
    manifests: list[Path] = []
    for base in _root_bases(root):
        if base.is_dir():
            manifests.extend(base.glob("*/manifest.json"))
    if not manifests:
        return None
    for manifest in manifests:
        payload = _read(manifest)
        if payload is None:
            continue
        digest = payload.get("graph_digest")
        if not isinstance(digest, str) or eligible_root(root, digest) != manifest:
            continue
        try:
            stamp = manifest.stat().st_mtime_ns
        except OSError:
            continue
        candidates.append((digest, stamp, manifest))
    if not candidates:
        return None
    digest, _stamp, manifest = max(candidates, key=lambda item: (item[1], item[0]))
    return digest, manifest


__all__ = [
    "GRAPH_CHILDREN",
    "GRAPH_IDENTITY_INPUTS",
    "GRAPH_ROOTS",
    "CorpusAttestation",
    "VerificationGraphError",
    "attest_corpus",
    "corpus_attestation_matches",
    "eligible_root",
    "latest_eligible_root",
    "graph_identity",
    "publish_complete_root",
    "publish_selected_child",
]

"""Benchmark corpora: a laid-out source tree plus a sealed manifest.

A corpus directory holds ``home/`` (a stand-in ``$HOME`` whose typed default
source roots -- ``.claude/projects``, ``.codex/sessions``, ``.gemini/tmp`` --
the daemon discovers exactly as it does in production), optionally
``exports/<name>/`` directories for export archives (ChatGPT, Claude.ai) that
the run stages into the scratch archive's inbox, and ``manifest.json``.

The manifest seals the input: every file's relative path, size and SHA-256,
its origin family, and one digest over the sorted triples. Two receipts are
comparable only when their corpus digests match.

``sample`` corpora are a seeded, stratified sample of an operator's real
source files, and ``files`` corpora are exactly the named real files (one
whale, say). Both stay private, with their manifests and receipts; only
aggregate numbers leave the machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import stat
import time
from collections import defaultdict, deque
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from polylogue.schemas.synthetic import SyntheticCorpus
    from polylogue.schemas.synthetic.composition import SyntheticComposition
    from polylogue.schemas.synthetic.models import SyntheticSessionChunks

MANIFEST_FORMAT: Final = "polylogue.fresh-build-corpus.v1"
MANIFEST_NAME: Final = "manifest.json"

#: Origin family by corpus-relative prefix. Longest prefix wins.
ORIGIN_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("home/.claude/projects/", "claude-code"),
    ("home/.codex/sessions/", "codex"),
    ("home/.gemini/tmp/", "gemini-cli"),
    ("exports/chatgpt/", "chatgpt"),
    ("exports/claude-ai/", "claude-ai"),
    ("home/.claude/history.jsonl", "claude-code"),
    ("home/.codex/", "codex"),
)


_CHECKOUT = Path(__file__).resolve().parents[2]


def refuse_inside_checkout(path: Path, what: str) -> None:
    """Refuse a corpus, work or scratch path inside this checkout.

    They hold copies of real transcripts and archives built from them; a
    tracked tree is public.
    """
    resolved = path.resolve()
    if resolved == _CHECKOUT or _CHECKOUT in resolved.parents:
        raise ValueError(f"{what} must be outside the checkout ({_CHECKOUT}): {path}")


def origin_for(relative: str) -> str:
    for prefix, origin in ORIGIN_PREFIXES:
        if relative.startswith(prefix):
            return origin
    return "other"


#: Directories of parser sidecars whose filesystem mtime production parsing
#: persists (as the sidecar event's timestamp), so it is part of the input.
_MTIME_SEMANTIC_DIRS: Final = frozenset({"tool-results", "tool-outputs"})


@dataclass(frozen=True, slots=True)
class CorpusFile:
    path: str
    bytes: int
    sha256: str
    origin: str
    #: Sealed only for sidecars, whose mtime reaches the indexed output.
    mtime_ns: int | None = None


def _semantic_mtime_ns(relative: str, path: Path) -> int | None:
    if not _MTIME_SEMANTIC_DIRS.intersection(relative.split("/")[:-1]):
        return None
    return path.stat().st_mtime_ns


def _sidecar_mtimes(files: Iterable[CorpusFile]) -> dict[str, int]:
    return {item.path: item.mtime_ns for item in files if item.mtime_ns is not None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_digest(files: Iterable[CorpusFile], parameters: dict[str, Any] | None = None) -> str:
    """Hash the file list together with the sealed *parameters*.

    ``parameters`` (a sample's ``population``, ``fraction``, ``seed``, ...)
    is not file-derived, so nothing here can recompute it from the corpus
    tree the way ``by_origin``/``files`` are recomputed and compared in
    :func:`verify_manifest`. Folding it into the digest instead means an edit
    that touches ``parameters`` alone -- e.g. an inflated or reduced
    ``population.<origin>.bytes`` -- no longer leaves ``digest`` matching,
    closing the gap where a run could qualify under an edited projection
    denominator while still carrying the originally sealed file bytes.
    """
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda entry: entry.path):
        mtime = "" if item.mtime_ns is None else f"\0{item.mtime_ns}"
        digest.update(f"{item.path}\0{item.bytes}\0{item.sha256}{mtime}\n".encode())
    if parameters is not None:
        digest.update(b"\0parameters\0")
        digest.update(json.dumps(parameters, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _hash_tree(root: Path) -> list[CorpusFile]:
    """Every file under ``home/`` and ``exports/`` with its current bytes."""
    files: list[CorpusFile] = []
    for top in ("home", "exports"):
        base = root / top
        # A linked top-level root is walked by os.walk (and by the daemon)
        # but skipped by the change stamp: the same unsealed-input hole as a
        # nested link.
        if base.is_symlink():
            raise ValueError("corpus holds a symbolic link; a sealed corpus holds only real files")
        if not base.exists():
            continue
        for directory, dirs, names in os.walk(base):
            # The daemon walks source roots following symlinks; a linked
            # directory would add unsealed files this walk never hashed.
            for name in [*dirs, *names]:
                if (Path(directory) / name).is_symlink():
                    raise ValueError("corpus holds a symbolic link; a sealed corpus holds only real files")
            for name in names:
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                files.append(
                    CorpusFile(
                        relative,
                        path.stat().st_size,
                        _sha256(path),
                        origin_for(relative),
                        _semantic_mtime_ns(relative, path),
                    )
                )
    files.sort(key=lambda entry: entry.path)
    return files


def seal(root: Path, *, kind: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Hash every file under ``home/`` and ``exports/`` and write the manifest."""
    _private_root(root)
    # The run's stand-in ``$HOME`` exists even for an export-only corpus.
    (root / "home").mkdir(mode=0o700, exist_ok=True)
    files = _hash_tree(root)
    by_origin: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for item in files:
        by_origin[item.origin]["files"] += 1
        by_origin[item.origin]["bytes"] += item.bytes
    manifest = {
        "format": MANIFEST_FORMAT,
        "kind": kind,
        "parameters": parameters,
        "digest": corpus_digest(files, parameters),
        "file_count": len(files),
        "total_bytes": sum(item.bytes for item in files),
        "by_origin": dict(sorted(by_origin.items())),
        "files": [[item.path, item.bytes, item.sha256, item.origin] for item in files],
        "sidecar_mtimes_ns": _sidecar_mtimes(files),
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    (root / MANIFEST_NAME).chmod(0o600)
    return manifest


def _private_root(root: Path) -> None:
    """Create a corpus root only its owner can enter.

    Sampled and named corpora are copies of private transcripts; with the
    common 022 umask they would otherwise be world-readable in shared
    scratch storage.
    """
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)


def _file_observation(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return (status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def _copy_private(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A source rewritten in place while it is copied (same size or not)
    # would seal a torn mixture; the file must be the same one, unchanged,
    # on both sides of the copy.
    # Stamps alone miss a same-size rewrite within one timestamp tick, so the
    # copy's content is compared with the source's as well.
    before = _file_observation(source)
    shutil.copyfile(source, destination)
    if _file_observation(source) != before or _sha256(source) != _sha256(destination):
        destination.unlink(missing_ok=True)
        raise ValueError("a source file changed while it was being copied; sample a quiescent population")
    destination.chmod(0o600)
    # A sidecar's mtime is parsed into its event timestamp; a copy keeps the
    # source's, so a sample reproduces the output its source would.
    stat = source.stat()
    os.utime(destination, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def load_manifest(root: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"not a fresh-build corpus manifest: {root / MANIFEST_NAME}")
    return manifest


def change_stamp(root: Path) -> dict[str, tuple[int, int, int]]:
    """``{relative path: (inode, mtime_ns, ctime_ns)}`` for every corpus file and directory.

    A write changes a file's ctime even when its bytes are later restored
    and its mtime set back, and creating, renaming or deleting an entry
    changes its directory's mtime and ctime, so two equal stamps mean no file
    was written, replaced, added or removed in between -- even one added and
    removed again. The endpoint content check alone cannot tell. The root
    itself is stamped under ``"."``.
    """
    stamps: dict[str, tuple[int, int, int]] = {}
    for path in [root, *sorted(root.rglob("*"))]:
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            continue
        status = path.stat()
        relative = path.relative_to(root).as_posix() if path != root else "."
        stamps[relative] = (status.st_ino, status.st_mtime_ns, status.st_ctime_ns)
    return stamps


def verify_manifest(root: Path, manifest: dict[str, Any]) -> None:
    """Refuse a corpus whose bytes no longer match its seal.

    Every file is re-hashed and the tree re-listed, so an edit that keeps a
    file's size, or a file added after sealing, is refused too.

    Errors give counts, never paths: corpus paths carry private project and
    session names, and these errors reach job and CI logs.
    """
    sealed = {row[0]: (row[1], row[2]) for row in manifest["files"]}
    current = _hash_tree(root)
    present = {item.path: (item.bytes, item.sha256) for item in current}
    if added := set(present) - set(sealed):
        raise ValueError(f"corpus has {len(added)} file(s) added since sealing")
    if missing := set(sealed) - set(present):
        raise ValueError(f"corpus lost {len(missing)} file(s) since sealing")
    resized = sum(present[relative][0] != size for relative, (size, _sha256) in sealed.items())
    if resized:
        raise ValueError(f"{resized} corpus file(s) changed size since sealing")
    edited = sum(present[relative][1] != sha256 for relative, (_size, sha256) in sealed.items())
    if edited:
        raise ValueError(f"{edited} corpus file(s) changed content since sealing")
    sealed_mtimes = manifest.get("sidecar_mtimes_ns", {})
    current_mtimes = _sidecar_mtimes(current)
    if retimed := sum(sealed_mtimes.get(path) != mtime for path, mtime in current_mtimes.items()):
        raise ValueError(f"{retimed} corpus sidecar(s) changed mtime since sealing")
    if corpus_digest(current, manifest.get("parameters")) != manifest["digest"]:
        raise ValueError("corpus manifest digest does not match its file list")
    # The digest covers the file rows only; the aggregates a receipt reads
    # (bytes, counts, per-origin split) are recomputed and compared too.
    by_origin: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for item in current:
        by_origin[item.origin]["files"] += 1
        by_origin[item.origin]["bytes"] += item.bytes
    expected = {
        "file_count": len(current),
        "total_bytes": sum(item.bytes for item in current),
        "by_origin": dict(sorted(by_origin.items())),
        "files": [[item.path, item.bytes, item.sha256, item.origin] for item in current],
        "sidecar_mtimes_ns": current_mtimes,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"corpus manifest {key} does not match the sealed files")


# ---------------------------------------------------------------------------
# private stratified sample of real sources


@dataclass(frozen=True, slots=True)
class SampleSource:
    """One real source root and where it lands inside the corpus."""

    #: The watch-source name whose declared layout admits the population.
    origin: str
    root: Path
    target: str
    #: Sample whole units rather than single files: every file whose first
    #: ``unit_depth`` path parts (suffix stripped) agree is one unit.
    session_units: bool = False
    unit_depth: int = 2


def default_sample_sources(home: Path) -> tuple[SampleSource, ...]:
    return (
        # <project>/<session>.jsonl, its subagents and its tool-results/
        # sidecars are one unit.
        SampleSource("claude-code", home / ".claude" / "projects", "home/.claude/projects", True),
        SampleSource("codex", home / ".codex" / "sessions", "home/.codex/sessions"),
        # Gemini keeps a project's transcripts under chats/ and their
        # persisted tool output under tool-outputs/session-<id>/; the project
        # is the smallest unit that holds both.
        SampleSource("gemini-cli", home / ".gemini" / "tmp", "home/.gemini/tmp", True, unit_depth=1),
    )


def _unit_key(source: SampleSource, path: Path) -> str:
    if not source.session_units:
        return str(path)
    relative = path.relative_to(source.root)
    parts = relative.parts
    return str(Path(*parts[: source.unit_depth]).with_suffix("")) if len(parts) >= source.unit_depth else str(relative)


def _unit_paths(source: SampleSource, named: Path) -> list[Path]:
    """The existing layout unit, walked without visiting other units."""
    from polylogue.sources.source_walk import layout_source_paths
    from polylogue.sources.walk_faults import WalkRefusedError

    relative = named.relative_to(source.root)
    key = _unit_key(source, named)
    if not source.session_units:
        start = named.parent

        def scope(path: Path) -> bool:
            return path == named
    else:
        start = source.root if source.unit_depth == 1 else source.root / relative.parts[0]

        # Names of siblings are inspected once; only the same existing unit
        # key reaches native type/link handling or subtree enumeration.
        def scope(path: Path) -> bool:
            return _unit_key(source, path) == key

    try:
        paths = layout_source_paths(source.origin, source.root, start=start, scope=scope)
    except WalkRefusedError as exc:
        raise ValueError(f"a {source.origin} selected source unit was refused ({len(exc.faults)} fault(s))") from exc
    if named not in paths:
        raise ValueError("named transcript is not admitted by its native source walk")
    return paths


def _units(source: SampleSource) -> list[tuple[str, list[Path], int]]:
    """Group the native watched population into the original sampling units."""
    if not source.root.is_dir():
        return []
    from polylogue.sources.source_walk import layout_source_paths
    from polylogue.sources.walk_faults import WalkRefusedError

    try:
        files = layout_source_paths(source.origin, source.root)
    except WalkRefusedError as exc:
        raise ValueError(f"a {source.origin} source subtree is unreadable ({exc})") from exc
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        grouped[_unit_key(source, path)].append(path)
    # Non-session sources retain native discovery order, which owns the seeded draw.
    items = sorted(grouped.items()) if source.session_units else grouped.items()
    return [(key, paths, sum(path.stat().st_size for path in paths)) for key, paths in items]


def _member_destination(out: Path, source: SampleSource, path: Path) -> Path:
    return out / source.target / path.relative_to(source.root)


def _assembly_closure(source: SampleSource, paths: Sequence[Path], out: Path | None = None) -> dict[Path, Path]:
    """Existing static inputs paired with their production corpus coordinates.

    Acquisition supplies lexical session coordinates. Resolve both the source
    and copied session with the same retained-assembly resolver; a resolved
    history outside a linked watched root must still accompany the copied main.
    """
    from polylogue.sources.retained_assembly import claude_code_sidecar_coordinates, codex_sidecar_coordinates

    resolver = {"claude-code": claude_code_sidecar_coordinates, "codex": codex_sidecar_coordinates}.get(source.origin)
    closure: dict[Path, Path] = {}
    if resolver is None:
        return closure
    members = set(paths)
    for path in paths:
        coordinates = resolver(str(path)) or ()
        if not coordinates:
            continue  # A generic sampling root has no declared install coordinates.
        targets = (resolver(str(_member_destination(out, source, path))) or ()) if out is not None else coordinates
        for coordinate, target in zip(coordinates, targets, strict=True):
            dependency = Path(coordinate)
            try:
                mode = dependency.lstat().st_mode
            except FileNotFoundError:
                continue  # Optional absence; the selected closure is checked again.
            if not stat.S_ISREG(mode):
                raise ValueError("assembly dependency must be a regular unlinked file")
            if dependency in members:
                continue
            destination = Path(target)
            if out is not None and out.resolve() not in destination.resolve().parents:
                raise ValueError("assembly dependency escapes the corpus")
            prior = closure.get(destination)
            if prior is not None and prior != dependency and _sha256(prior) != _sha256(dependency):
                raise ValueError("assembly dependencies collide at a corpus coordinate")
            closure[destination] = dependency
    return closure


class _CorpusCopy:
    """One copy per destination, with content and currentness proof."""

    def __init__(self) -> None:
        self.destinations: dict[Path, str] = {}
        self.observed: dict[Path, tuple[tuple[int, int, int, int], str]] = {}

    def expect(self, paths: Sequence[Path]) -> None:
        for path in paths:
            stamp = _file_observation(path)
            if path in self.observed:
                if self.observed[path][0] != stamp:
                    raise ValueError("source dependency changed between selected units")
                continue
            digest = _sha256(path)
            if _file_observation(path) != stamp:
                raise ValueError("source dependency changed during observation")
            self.observed[path] = (stamp, digest)

    def copy(self, path: Path, destination: Path) -> int:
        before = _file_observation(path)
        if destination in self.destinations:
            prior = self.observed.get(path)
            digest = prior[1] if prior is not None and prior[0] == before else _sha256(path)
            if digest != self.destinations[destination]:
                raise ValueError("source files collide at a corpus coordinate")
            added = 0
        else:
            _copy_private(path, destination)
            digest = _sha256(destination)
            self.destinations[destination] = digest
            added = destination.stat().st_size
        if _file_observation(path) != before:
            raise ValueError("source closure changed while copying")
        if path in self.observed and self.observed[path] != (before, digest):
            raise ValueError("source dependency changed before its copy")
        self.observed[path] = (before, digest)
        return added

    def verify(self) -> None:
        for path, (stamp, digest) in self.observed.items():
            try:
                unchanged = _file_observation(path) == stamp and _sha256(path) == digest
            except OSError:
                unchanged = False
            if not unchanged:
                raise ValueError("a source closure changed after it was copied")


def _copy_closure(out: Path, source: SampleSource, paths: Sequence[Path], copier: _CorpusCopy) -> dict[Path, Path]:
    closure = _assembly_closure(source, paths, out)
    copier.expect(list(closure.values()))
    for path in paths:
        copier.copy(path, _member_destination(out, source, path))
    for destination, path in closure.items():
        copier.copy(path, destination)
    return closure


def _size_bucket(size: int) -> int:
    return max(0, size.bit_length() - 10)


def sample_real(
    out: Path,
    *,
    seed: int,
    fraction: float,
    sources: Sequence[SampleSource],
) -> dict[str, Any]:
    """Copy a seeded byte-fraction of every (origin, size-bucket) stratum.

    Within each stratum units are shuffled with the seed and taken until the
    stratum's byte fraction is reached, rounding the last unit stochastically.
    Large units are sampled like any other: their own size buckets are
    strata, so a sample holds whales at their byte fraction and a full
    fraction holds every one.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"corpus directory must be absent or empty: {out}")
    _private_root(out)
    rng = random.Random(seed)
    copier = _CorpusCopy()
    selected_units = []
    drawn: dict[str, dict[str, int]] = {}
    population: dict[str, dict[str, int]] = {}
    census: dict[str, list[tuple[str, list[Path], int]]] = {}
    for source in sources:
        units = _units(source)
        census[source.origin] = units
        drawn[source.origin] = {"units": 0, "bytes": 0, "shared_dependency_bytes": 0}
        population[source.origin] = {"units": len(units), "bytes": sum(unit[2] for unit in units)}
        strata: dict[int, list[tuple[str, list[Path], int]]] = defaultdict(list)
        for unit in units:
            strata[_size_bucket(unit[2])].append(unit)
        for bucket in sorted(strata):
            members = strata[bucket]
            rng.shuffle(members)
            goal = fraction * sum(unit[2] for unit in members)
            taken = 0
            for _key, paths, size in members:
                # A full fraction takes every unit, zero-byte ones included.
                if fraction < 1 and taken >= goal:
                    break
                # Stochastic rounding keeps each stratum's expected sampled
                # bytes at its fraction, so rare large buckets are not
                # over-represented by a forced minimum of one unit. The unit
                # that crosses the goal is the one boundary draw: accepted or
                # not, the stratum ends there. Redrawing on every later unit
                # would make selection near certain.
                boundary = fraction < 1 and taken + size > goal
                if boundary and rng.random() > (goal - taken) / size:
                    break
                closure = _assembly_closure(source, paths, out)
                copier.expect(list(closure.values()))
                copied = 0
                for path in paths:
                    destination = _member_destination(out, source, path)
                    copier.copy(path, destination)
                    copied += destination.stat().st_size
                selected_units.append((source, paths, closure))
                drawn[source.origin]["units"] += 1
                drawn[source.origin]["bytes"] += copied
                if copied != size:
                    # The census (population and stratum goal) saw other bytes
                    # than the sample now holds: a live transcript grew. Report
                    # only origin and the changed unit's file count -- the
                    # operator source path is private and this can run in an
                    # AgentCTL/CI job whose log is not private (the later
                    # census-revalidation error below follows the same rule).
                    raise ValueError(
                        f"{source.origin} source changed while sampling ({len(paths)} file(s) in the unit)"
                    )
                taken += size
                if boundary:
                    break
    # Shared static metadata is added after the unchanged probabilistic draws.
    # Already drawn layout files are not charged a second time.
    for source, _paths, closure in selected_units:
        for destination, path in closure.items():
            drawn[source.origin]["shared_dependency_bytes"] += copier.copy(path, destination)
    # The population is the projection's denominator: a live source that
    # grew, shrank or lost a file while sampling makes it stale.
    for source in sources:
        recount = {key: size for key, _paths, size in _units(source)}
        if recount != {key: size for key, _paths, size in census[source.origin]}:
            raise ValueError(f"{source.origin} sources changed while sampling; sample a quiescent tree")
    for source, paths, closure in selected_units:
        if _assembly_closure(source, paths, out) != closure:
            raise ValueError("sampled dependency membership changed while copying")
    copier.verify()
    return seal(
        out,
        kind="sample",
        parameters={
            "seed": seed,
            "fraction": fraction,
            "population": population,
            "drawn_units": drawn,
            "projection_unknown_origins": sorted(
                origin for origin, row in drawn.items() if row["shared_dependency_bytes"]
            ),
        },
    )


def _selection_units(source: SampleSource) -> list[tuple[str, list[Path], list[Path]]]:
    """Whole session units plus existing static assembly dependencies.

    Coordinate resolvers are the production retained-assembly authority.
    Mutable database sources are separate routes, never copied as page images.
    """
    result = []
    for key, paths, _size in _units(source):
        if source.origin in {"claude-code", "codex"} and not any(
            path.suffix in {".jsonl", ".ndjson"} for path in paths
        ):
            continue
        dependencies = sorted(set(_assembly_closure(source, paths).values()))
        result.append((key, paths, dependencies))
    return result


def select_real(
    out: Path,
    *,
    seed: int,
    max_bytes: int,
    max_unit_bytes: int,
    max_units: int,
    sources: Sequence[SampleSource],
) -> dict[str, Any]:
    """Select complete session units under private-byte and total-closure budgets.

    Shared dependencies count once in the corpus budget. No file is truncated.
    Oversized units and uncovered strata remain visible in the sealed report.
    """
    if min(max_bytes, max_unit_bytes, max_units) <= 0:
        raise ValueError("selection budgets must be positive")
    if len({source.origin for source in sources}) != len(sources):
        raise ValueError("selection source origins must be unique")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"corpus directory must be absent or empty: {out}")
    sources = tuple(sorted(sources, key=lambda source: source.origin))
    rng = random.Random(seed)
    census_started = time.time()
    census = {source.origin: _selection_units(source) for source in sources}
    stamps = {
        path: _file_observation(path)
        for units in census.values()
        for _key, paths, dependencies in units
        for path in [*paths, *dependencies]
    }
    census_finished = time.time()
    queues: dict[tuple[str, int], list[tuple[str, list[Path], list[Path]]]] = defaultdict(list)
    strata: dict[tuple[str, int], dict[str, int]] = {}
    population = {}
    for source in sources:
        units = census[source.origin]
        all_paths = {path for _key, paths, dependencies in units for path in [*paths, *dependencies]}
        population[source.origin] = {"units": len(units), "bytes": sum(stamps[path][1] for path in all_paths)}
        for unit in units:
            private = sum(stamps[path][1] for path in unit[1])
            closure_bytes = sum(stamps[path][1] for path in [*unit[1], *unit[2]])
            key = (source.origin, _size_bucket(private))
            row = strata.setdefault(
                key,
                {
                    "population_units": 0,
                    "private_bytes": 0,
                    "oversized_units": 0,
                    "selected_units": 0,
                    "selected_bytes": 0,
                    "total_budget_skips": 0,
                },
            )
            row["population_units"] += 1
            row["private_bytes"] += private
            if private > max_unit_bytes or closure_bytes > max_bytes:
                row["oversized_units"] += 1
            else:
                queues[key].append(unit)
    for key in sorted(queues):
        members = queues[key]
        members.sort(key=lambda unit: unit[0])
        rng.shuffle(members)
    source_by_origin = {source.origin: source for source in sources}
    provider_order = sorted({origin for origin, _bucket in queues})
    rng.shuffle(provider_order)
    bucket_order: dict[str, list[int]] = {}
    for origin in sorted(provider_order):
        buckets = sorted(bucket for queued_origin, bucket in queues if queued_origin == origin)
        rng.shuffle(buckets)
        bucket_order[origin] = buckets
    provider_buckets = {origin: deque(buckets) for origin, buckets in bucket_order.items()}
    total_budget_skips = 0
    selected: list[tuple[SampleSource, tuple[str, list[Path], list[Path]]]] = []
    selected_paths: set[Path] = set()
    selected_destinations: set[Path] = set()
    total = 0
    while queues and len(selected) < max_units:
        for origin in provider_order:
            bucket_queue = provider_buckets[origin]
            if not bucket_queue:
                continue
            key = (origin, bucket_queue.popleft())
            members = queues[key]
            unit = members.pop()
            if members:
                bucket_queue.append(key[1])
            else:
                del queues[key]
            paths = {*unit[1], *unit[2]}
            source = source_by_origin[key[0]]
            mapped = {_member_destination(out, source, path): path for path in unit[1]}
            closure = _assembly_closure(source, unit[1], out)
            if set(closure.values()) != set(unit[2]):
                raise ValueError("selected dependency membership changed after census")
            mapped.update(closure)
            added = sum(
                stamps[path][1] for destination, path in mapped.items() if destination not in selected_destinations
            )
            if total + added > max_bytes:
                strata[key]["total_budget_skips"] += 1
                total_budget_skips += 1
                continue
            selected.append((source_by_origin[key[0]], unit))
            selected_paths.update(paths)
            selected_destinations.update(mapped)
            total += added
            strata[key]["selected_units"] += 1
            strata[key]["selected_bytes"] += added
            if len(selected) == max_units:
                break
    _private_root(out)
    copier = _CorpusCopy()
    copied_closures = []
    provenance = []
    for source, (unit_key, unit_paths, dependencies) in selected:
        closure = _copy_closure(out, source, unit_paths, copier)
        copied_closures.append((source, unit_paths, closure))
        provenance.append(
            {
                "origin": source.origin,
                "source_root": str(source.root.resolve()),
                "unit": unit_key,
                "files": len(unit_paths),
                "dependency_files": len(dependencies),
                "private_bytes": sum(stamps[path][1] for path in unit_paths),
                "shared_dependency_bytes": sum(stamps[path][1] for path in dependencies),
                "closure": "whole layout session unit and static assembly dependencies",
            }
        )
    # Unselected live sessions can keep growing. Only this corpus's selected
    # dependency closure must be stable, including newly appearing sidecars.
    for source in sources:
        chosen = {unit[0]: unit for selected_source, unit in selected if selected_source == source}
        if not chosen:
            continue
        current = {unit[0]: unit for unit in _selection_units(source) if unit[0] in chosen}
        if current != chosen:
            raise ValueError(f"{source.origin} selected dependency membership changed while copying")
    if any(_file_observation(path) != stamps[path] for path in selected_paths):
        raise ValueError("selected source closure changed while copying")
    for source, closure_paths, closure in copied_closures:
        if _assembly_closure(source, closure_paths, out) != closure:
            raise ValueError("selected dependency coordinates changed while copying")
    copier.verify()
    return seal(
        out,
        kind="selected-real",
        parameters={
            "seed": seed,
            "max_bytes": max_bytes,
            "max_unit_bytes": max_unit_bytes,
            "max_units": max_units,
            "selection_strategy": "seeded provider round-robin with seeded size-bucket round-robin",
            "selection_order": {"providers": provider_order, "size_buckets": bucket_order},
            "selection_semantics": "diagnostic coverage; not a probability sample or population estimator",
            "total_budget_skips": total_budget_skips,
            "population": population,
            "selected_units": len(selected),
            "provenance": provenance,
            "population_observation": {
                "started_at_unix": census_started,
                "finished_at_unix": census_finished,
                "coherence": "non-atomic timed census; unrelated source changes permitted",
            },
            "strata": [
                {"origin": origin, "size_bucket": bucket, **row, "missing": row["selected_units"] == 0}
                for (origin, bucket), row in sorted(strata.items())
            ],
            "sampled_providers": sorted({source.origin for source, _unit in selected}),
            "missing_providers": sorted(
                ({origin for _prefix, origin in ORIGIN_PREFIXES} | {"codex-state"} | set(source_by_origin))
                - {source.origin for source, _unit in selected}
            ),
            "route": "ordinary-daemon-cold-build-selected-static-roots",
            "finish_line": "terminal readiness and exact output fingerprint",
            "exclusions": [
                "oversized units",
                "units beyond selection budget",
                "mutable codex-state databases",
                "export and hook routes",
                "embeddings",
                "browser capture",
            ],
        },
    )


#: Export origins a corpus may stage under ``exports/<name>/``.
EXPORT_ORIGINS: Final = ("chatgpt", "claude-ai")


def corpus_from_files(
    out: Path,
    files: Sequence[Path],
    *,
    home: Path,
    exports: Sequence[tuple[str, Path]] = (),
    hooks: Path | None = None,
    hooks_fraction: float = 1.0,
) -> dict[str, Any]:
    """Seal a private corpus of the named real transcripts and their units.

    A named transcript brings the rest of its sampling unit -- its
    subagent transcripts and parser sidecars (``tool-results/``,
    ``tool-outputs/``) -- exactly as ``sample`` groups them, since the daemon
    reads those beside it from the real source tree. Each file keeps its
    position under ``home`` (so a Claude Code transcript
    stays under ``.claude/projects/...``) and must lie under one of the typed
    default source roots. This is how a single large source -- a whale -- is
    measured through the same benchmark as a stratified sample. ``exports``
    stages export files (``(origin, path)``) under ``exports/<origin>/``,
    the additional roots the run configures.
    """
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"corpus directory must be absent or empty: {out}")
    _private_root(out)
    from polylogue.core.enums import Provider
    from polylogue.sources.origin_specs import recognize_source_class

    # Membership and layout follow the lexical path, as production discovery
    # does: a transcript below a linked source directory is under its root.
    # The file itself is read through its resolved path and must be a real
    # file there (production does not ingest a linked file).
    home = Path(os.path.abspath(home))
    sources = [(Path(os.path.abspath(source.root)), source) for source in default_sample_sources(home)]
    copier = _CorpusCopy()
    selected = {}
    for named in files:
        file = Path(os.path.abspath(named))
        resolved = file.resolve(strict=True)
        if file.is_symlink() or not resolved.is_file():
            raise ValueError(f"{named} is not a regular transcript file")
        admitted = [source for root, source in sources if root in file.parents]
        if not admitted:
            raise ValueError(f"{named} is not under a default source root of {home}")
        # The watcher admits only its declared layout; any other file would
        # never be admitted, and the build could not go terminal.
        from polylogue.sources.live.watcher import WatchSource

        if not WatchSource(name=admitted[0].origin, root=admitted[0].root).accepts(file):
            raise ValueError(f"{named} is not a transcript its source root's declared layout admits")
        # The suffix makes a file observable, not a session: production's own
        # source classifier decides (a Gemini tool-output sidecar is raw-only).
        recognition = recognize_source_class(Provider(admitted[0].origin), resolved)
        if recognition is not None and recognition.source_class != "session":
            raise ValueError(f"{named} is not a session transcript: {recognition.reason}")
        source = admitted[0]
        key = (source.origin, _unit_key(source, file))
        if key in selected:
            continue
        unit = _unit_paths(source, file)
        closure = _copy_closure(out, source, unit, copier)
        selected[key] = (source, file, unit, closure)
    for source, named, unit, closure in selected.values():
        if _unit_paths(source, named) != unit or _assembly_closure(source, unit, out) != closure:
            raise ValueError("named source dependency membership changed while copying")
    copier.verify()
    staged: set[str] = set()
    for origin, file in exports:
        if origin not in EXPORT_ORIGINS:
            raise ValueError(f"unknown export origin {origin!r}; known: {', '.join(EXPORT_ORIGINS)}")
        resolved = file.resolve(strict=True)
        # Two exports with one basename would overwrite each other: here, in
        # one origin's directory, and in the run's archive inbox, which every
        # origin shares.
        if resolved.name in staged:
            raise ValueError(f"two --export files share the name {resolved.name!r}; rename one")
        staged.add(resolved.name)
        _copy_private(resolved, out / "exports" / origin / resolved.name)
    if not 0 < hooks_fraction <= 1:
        raise ValueError("hooks_fraction must be in (0, 1]")
    hook_count = 0
    if hooks is not None:
        source_argument = Path(hooks).expanduser()
        if source_argument.is_symlink():
            raise ValueError("hook spool root is a symbolic link")
        source_root = source_argument.resolve(strict=True)
        if not source_root.is_dir():
            raise ValueError(f"hook spool is not a directory: {hooks}")
        destination_root = out / "home" / ".polylogue-hook-spool"
        _copy_hook_tree(source_root, destination_root, fraction=hooks_fraction)
        hook_count = sum(1 for path in destination_root.rglob("*") if path.is_file())
    return seal(
        out,
        kind="files",
        parameters={
            "selection": "explicit",
            "files": len(files),
            "exports": len(exports),
            "hook_files": hook_count,
            "hook_fraction": hooks_fraction,
        },
    )


def _copy_hook_tree(source_root: Path, destination_root: Path, *, fraction: float) -> None:
    """Copy a deterministic file sample with per-file identity checks."""
    stack = [(source_root, destination_root, "")]
    while stack:
        source_dir, destination_dir, relative_dir = stack.pop()
        if source_dir.is_symlink() or not source_dir.is_dir():
            raise ValueError("hook spool contains a non-directory member")
        before = source_dir.stat()
        destination_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        with os.scandir(source_dir) as entries:
            members = sorted(entries, key=lambda entry: entry.name)
        observed = []
        directories = []
        for entry in members:
            path = Path(entry.path)
            status = entry.stat(follow_symlinks=False)
            observed.append((entry.name, status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns))
            target = destination_dir / entry.name
            if stat.S_ISDIR(status.st_mode):
                directories.append((path, target, f"{relative_dir}/{entry.name}".strip("/")))
            elif stat.S_ISREG(status.st_mode):
                relative = f"{relative_dir}/{entry.name}".strip("/")
                selector = int.from_bytes(hashlib.sha256(relative.encode()).digest()[:8], "big") / 2**64
                if fraction == 1 or selector < fraction:
                    _copy_private(path, target)
            else:
                raise ValueError("hook spool contains a symlink or special file")
        if fraction == 1:
            with os.scandir(source_dir) as entries:
                after = sorted(
                    (
                        entry.name,
                        entry.stat(follow_symlinks=False).st_dev,
                        entry.stat(follow_symlinks=False).st_ino,
                        entry.stat(follow_symlinks=False).st_size,
                        entry.stat(follow_symlinks=False).st_mtime_ns,
                    )
                    for entry in entries
                )
            if observed != after or (source_dir.stat().st_dev, source_dir.stat().st_ino) != (
                before.st_dev,
                before.st_ino,
            ):
                raise ValueError("hook spool changed while staging; retry from a quiescent snapshot")
        stack.extend(reversed(directories))


def corpus_from_synthetic(
    out: Path,
    *,
    corpus: SyntheticCorpus | SyntheticComposition,
    seed: int,
    count: int | None = None,
    target_bytes: int | None = None,
) -> dict[str, Any]:
    """Seal the original package driver's bytes for a fresh-build benchmark.

    A caller may supply a newly inferred neutral package to qualify generator
    mechanics. Its receipt does not prove the measured operator workload.
    """
    if count is None and target_bytes is None:
        raise ValueError("synthetic corpus needs a session count or byte target")
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError("synthetic session count must be a nonnegative exact integer")
    if target_bytes is not None and (type(target_bytes) is not int or target_bytes < 0):
        raise ValueError("synthetic byte target must be a nonnegative exact integer")
    from polylogue.schemas.synthetic.composition import SyntheticComposition

    if isinstance(corpus, SyntheticComposition):
        composition: SyntheticComposition | None = corpus
        corpora = corpus.corpora
    else:
        composition = None
        corpora = (corpus,)
    for selected in corpora:
        if selected.workload_profile is None:
            raise ValueError("package workload_profile_file is missing; re-infer and publish the package")
    layouts = {
        "claude-code": ("home/.claude/projects/schema", ".jsonl"),
        "codex": ("home/.codex/sessions/schema", ".jsonl"),
        "chatgpt": ("exports/chatgpt", ".json"),
        "claude-ai": ("exports/claude-ai", ".json"),
    }
    if any(selected.provider not in layouts for selected in corpora):
        raise ValueError("provider has no declared fresh-build synthetic source layout")
    refuse_inside_checkout(out, "synthetic corpus")
    if out.exists() and any(out.iterdir()):
        raise ValueError("synthetic corpus directory must be absent or empty")
    _private_root(out)
    written_bytes = 0
    selection_counts: dict[str, int] = {}

    def selected_sessions() -> Generator[tuple[SyntheticCorpus, SyntheticSessionChunks], None, None]:
        if composition is None:
            original = corpora[0]
            sessions = original.iter_sessions(count=count, target_bytes=target_bytes, seed=seed)
            try:
                for session in sessions:
                    yield original, session
            finally:
                sessions.close()
        else:
            for selected, selected_seed in composition.iter_choices(count=count, seed=seed):
                if target_bytes is not None and written_bytes >= target_bytes:
                    break
                sessions = selected.iter_sessions(count=1, seed=selected_seed)
                try:
                    for session in sessions:
                        yield selected, session
                finally:
                    sessions.close()

    sessions = selected_sessions()
    session_count = 0
    try:
        for selected, session in sessions:
            session_count += 1
            selection_counts[selected.provider] = selection_counts.get(selected.provider, 0) + 1
            directory, suffix = layouts[selected.provider]
            destination = out / directory
            destination.mkdir(parents=True, exist_ok=True, mode=0o700)
            index = session.index if composition is None else session_count - 1
            path = destination / f"session-{index:06d}{suffix}"
            with path.open("xb") as handle:
                path.chmod(0o600)
                for chunk in session.chunks:
                    handle.write(chunk)
                    written_bytes += len(chunk)
            for adjunct in session.adjuncts:
                adjunct_path = path.parent / path.stem / adjunct.directory / adjunct.name
                adjunct_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with adjunct_path.open("xb") as handle:
                    adjunct_path.chmod(0o600)
                    for chunk in adjunct.chunks:
                        handle.write(chunk)
                        written_bytes += len(chunk)
    finally:
        sessions.close()
    parameters: dict[str, Any] = {
        "seed": seed,
        "sessions": session_count,
        "target_bytes": target_bytes,
    }
    if composition is None:
        original = corpora[0]
        assert original.workload_profile is not None
        parameters.update(
            provider=original.provider,
            package_version=original.package_version,
            element_kind=original.element_kind,
            profile_id=original.workload_profile.get("profile_id"),
        )
    else:
        parameters.update(
            provider_populations=composition.populations,
            provider_weights=composition.weights,
            provider_profile_ids=composition.profile_ids,
            provider_selection_counts=selection_counts,
            population="retained-original-main-session-revisions-or-neutral-scopes",
        )
    return seal(out, kind="synthetic", parameters=parameters)

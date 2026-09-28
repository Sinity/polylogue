"""Benchmark corpora: a laid-out source tree plus a sealed manifest.

A corpus directory holds ``home/`` (a stand-in ``$HOME`` whose typed default
source roots -- ``.claude/projects``, ``.codex/sessions``, ``.gemini/tmp`` --
the daemon discovers exactly as it does in production), optionally
``exports/<name>/`` roots for export archives (ChatGPT, Claude.ai) that the
run configures as additional source roots, and ``manifest.json``.

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
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

MANIFEST_FORMAT: Final = "polylogue.fresh-build-corpus.v1"
MANIFEST_NAME: Final = "manifest.json"

#: Origin family by corpus-relative prefix. Longest prefix wins.
ORIGIN_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("home/.claude/projects/", "claude-code"),
    ("home/.codex/sessions/", "codex"),
    ("home/.gemini/tmp/", "gemini-cli"),
    ("exports/chatgpt/", "chatgpt"),
    ("exports/claude-ai/", "claude-ai"),
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


@dataclass(frozen=True, slots=True)
class CorpusFile:
    path: str
    bytes: int
    sha256: str
    origin: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_digest(files: Iterable[CorpusFile]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda entry: entry.path):
        digest.update(f"{item.path}\0{item.bytes}\0{item.sha256}\n".encode())
    return digest.hexdigest()


def _hash_tree(root: Path) -> list[CorpusFile]:
    """Every file under ``home/`` and ``exports/`` with its current bytes."""
    files: list[CorpusFile] = []
    for top in ("home", "exports"):
        base = root / top
        if not base.exists():
            continue
        for directory, _dirs, names in os.walk(base):
            for name in names:
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                files.append(CorpusFile(relative, path.stat().st_size, _sha256(path), origin_for(relative)))
    files.sort(key=lambda entry: entry.path)
    return files


def seal(root: Path, *, kind: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Hash every file under ``home/`` and ``exports/`` and write the manifest."""
    _private_root(root)
    files = _hash_tree(root)
    by_origin: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    for item in files:
        by_origin[item.origin]["files"] += 1
        by_origin[item.origin]["bytes"] += item.bytes
    manifest = {
        "format": MANIFEST_FORMAT,
        "kind": kind,
        "parameters": parameters,
        "digest": corpus_digest(files),
        "file_count": len(files),
        "total_bytes": sum(item.bytes for item in files),
        "by_origin": dict(sorted(by_origin.items())),
        "files": [[item.path, item.bytes, item.sha256, item.origin] for item in files],
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


def _copy_private(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o600)


def load_manifest(root: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"not a fresh-build corpus manifest: {root / MANIFEST_NAME}")
    return manifest


def verify_manifest(root: Path, manifest: dict[str, Any]) -> None:
    """Refuse a corpus whose bytes no longer match its seal.

    Every file is re-hashed and the tree re-listed, so an edit that keeps a
    file's size, or a file added after sealing, is refused too.
    """
    sealed = {row[0]: (row[1], row[2]) for row in manifest["files"]}
    current = _hash_tree(root)
    present = {item.path: (item.bytes, item.sha256) for item in current}
    if added := sorted(set(present) - set(sealed)):
        raise ValueError(f"corpus has files added since sealing: {', '.join(added[:5])}")
    if missing := sorted(set(sealed) - set(present)):
        raise ValueError(f"corpus lost files since sealing: {', '.join(missing[:5])}")
    for relative, (size, sha256) in sealed.items():
        if present[relative][0] != size:
            raise ValueError(f"corpus file changed size since sealing: {relative}")
        if present[relative][1] != sha256:
            raise ValueError(f"corpus file changed content since sealing: {relative}")
    if corpus_digest(current) != manifest["digest"]:
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
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"corpus manifest {key} does not match the sealed files")


# ---------------------------------------------------------------------------
# private stratified sample of real sources


@dataclass(frozen=True, slots=True)
class SampleSource:
    """One real source root and where it lands inside the corpus."""

    origin: str
    root: Path
    target: str
    suffixes: tuple[str, ...]
    #: Sample whole per-session units (a Claude Code transcript and its
    #: sibling subagent directory) rather than single files.
    session_units: bool = False


def default_sample_sources(home: Path) -> tuple[SampleSource, ...]:
    return (
        SampleSource("claude-code", home / ".claude" / "projects", "home/.claude/projects", (".jsonl",), True),
        SampleSource("codex", home / ".codex" / "sessions", "home/.codex/sessions", (".jsonl",)),
        SampleSource("gemini-cli", home / ".gemini" / "tmp", "home/.gemini/tmp", (".json", ".jsonl")),
    )


def _units(source: SampleSource) -> list[tuple[str, list[Path], int]]:
    """Group a source into sampling units: (key, files, bytes)."""
    if not source.root.is_dir():
        return []
    files = sorted(
        path
        for path in source.root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in source.suffixes
    )
    if not source.session_units:
        return [(str(path), [path], path.stat().st_size) for path in files]
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        relative = path.relative_to(source.root)
        parts = relative.parts
        # <project>/<session>.jsonl and <project>/<session>/subagents/... share
        # the key <project>/<session>.
        key = str(Path(*parts[:2]).with_suffix("")) if len(parts) >= 2 else str(relative)
        grouped[key].append(path)
    return [(key, paths, sum(path.stat().st_size for path in paths)) for key, paths in sorted(grouped.items())]


def _size_bucket(size: int) -> int:
    return max(0, size.bit_length() - 10)


def sample_real(
    out: Path,
    *,
    seed: int,
    fraction: float,
    sources: Sequence[SampleSource],
    whale_bytes: int = 64 << 20,
    max_whales_per_origin: int = 1,
) -> dict[str, Any]:
    """Copy a seeded byte-fraction of every (origin, size-bucket) stratum.

    Within each stratum units are shuffled with the seed and taken until the
    stratum's byte fraction is reached, rounding the last unit stochastically. Units above ``whale_bytes`` are capped per origin, because one
    whale dominates a small sample; the receipt's per-origin fit extrapolates
    whales from bytes rather than from their share of the sample.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"corpus directory must be absent or empty: {out}")
    _private_root(out)
    rng = random.Random(seed)
    population: dict[str, dict[str, int]] = {}
    for source in sources:
        units = _units(source)
        population[source.origin] = {"units": len(units), "bytes": sum(unit[2] for unit in units)}
        strata: dict[int, list[tuple[str, list[Path], int]]] = defaultdict(list)
        for unit in units:
            strata[_size_bucket(unit[2])].append(unit)
        whales = 0
        for bucket in sorted(strata):
            members = strata[bucket]
            rng.shuffle(members)
            goal = fraction * sum(unit[2] for unit in members)
            taken = 0
            for _key, paths, size in members:
                if taken >= goal:
                    break
                # Stochastic rounding keeps each stratum's expected sampled
                # bytes at its fraction, so rare large buckets are not
                # over-represented by a forced minimum of one unit. The unit
                # that crosses the goal is the one boundary draw: accepted or
                # not, the stratum ends there. Redrawing on every later unit
                # would make selection near certain.
                boundary = taken + size > goal
                if boundary and rng.random() > (goal - taken) / size:
                    break
                if size >= whale_bytes:
                    if whales >= max_whales_per_origin:
                        if boundary:
                            break
                        continue
                    whales += 1
                copied = 0
                for path in paths:
                    destination = out / source.target / path.relative_to(source.root)
                    _copy_private(path, destination)
                    copied += destination.stat().st_size
                if copied != size:
                    # The census (population and stratum goal) saw other bytes
                    # than the sample now holds: a live transcript grew.
                    raise ValueError(f"source changed while sampling: {paths[0]}")
                taken += size
                if boundary:
                    break
    return seal(
        out,
        kind="sample",
        parameters={
            "seed": seed,
            "fraction": fraction,
            "whale_bytes": whale_bytes,
            "max_whales_per_origin": max_whales_per_origin,
            "population": population,
        },
    )


def corpus_from_files(out: Path, files: Sequence[Path], *, home: Path) -> dict[str, Any]:
    """Seal a private corpus of exactly the named real files.

    Each file keeps its position under ``home`` (so a Claude Code transcript
    stays under ``.claude/projects/...``) and must lie under one of the typed
    default source roots. This is how a single large source -- a whale -- is
    measured through the same benchmark as a stratified sample.
    """
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"corpus directory must be absent or empty: {out}")
    _private_root(out)
    sources = [(source.root.resolve(), source.suffixes) for source in default_sample_sources(home)]
    home = home.resolve()
    for file in files:
        resolved = file.resolve(strict=True)
        admitted = [suffixes for root, suffixes in sources if root in resolved.parents]
        if not admitted:
            raise ValueError(f"{file} is not under a default source root of {home}")
        # The watcher cursors only its declared transcript suffixes; any other
        # file would never be admitted, and the build could not go terminal.
        if resolved.suffix.lower() not in admitted[0]:
            raise ValueError(f"{file} is not a transcript its source root admits ({', '.join(admitted[0])})")
        _copy_private(resolved, out / "home" / resolved.relative_to(home))
    return seal(out, kind="files", parameters={"selection": "explicit", "files": len(files)})

"""Hermeticity lint: a test must read its fixture, not the developer's machine.

A test that names ``~/.claude``, ``/realm/state`` or resolves ``$HOME`` at
runtime has stepped outside ``tests/conftest.py``'s isolation boundary no
matter what fixtures it also requests, so this is a source scan rather than a
runtime sandbox. Production modules are scanned for the same capture at import
time.

Pre-existing findings are exempted by fingerprint through a ratchet baseline;
a new one fails.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

PRODUCTION_PACKAGE = "polylogue"

#: Ambient paths a hermetic test must never read. These are real user and live
#: archive locations; a test touching one is reading the developer's machine.
AMBIENT_PATH_MARKERS: tuple[str, ...] = (
    "~/.codex",
    "~/.claude",
    "~/.config/polylogue",
    "/realm/state",
    "/realm/db",
    "/realm/data",
)

#: Call expressions that resolve an ambient location at runtime. ``Path.home``
#: and ``expanduser`` are the two ways to reach ``$HOME`` without writing a
#: literal, so a literal-only scan would miss them entirely.
AMBIENT_CALL_MARKERS: tuple[str, ...] = (
    "Path.home",
    "os.path.expanduser",
    "os.path.expandvars",
    "pathlib.Path.home",
)


@dataclass(frozen=True, slots=True)
class OracleAllowlistEntry:
    """One structured exemption. A bare name is not accepted anywhere."""

    path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class OracleFinding:
    """One machine-readable lint failure."""

    code: str
    path: str
    detail: str
    #: Line-independent identity used for baseline matching. ``detail`` stays
    #: human-readable (it carries the line number); ``fingerprint`` is what the
    #: ratchet compares, so an unrelated edit ABOVE a baselined finding no
    #: longer invalidates its exemption.
    fingerprint: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "detail": self.detail,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class OracleIntegrityReport:
    findings: tuple[OracleFinding, ...]
    scanned_modules: int = 0
    baselined: int = 0

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "scanned_modules": self.scanned_modules,
            "baselined": self.baselined,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


# ---------------------------------------------------------------------------
# Hermeticity sweep
# ---------------------------------------------------------------------------


def finding_fingerprint(code: str, path: str, source_line: str, ordinal: int) -> str:
    """Line-independent identity for one finding.

    Keying the ratchet on the rendered ``detail`` (which embeds a line number)
    made every baselined entry brittle: an unrelated edit higher in the file
    shifted the line, invalidated the exemption, and forced a regeneration --
    twice in one night, in PRs that had nothing to do with the finding.
    Dropping the line entirely was rejected earlier for a good reason: two
    ``~/.codex`` reads in one file would collapse onto a single key and a newly
    added one would inherit the old exemption, which is the ratchet hole cold
    review already found once.

    This keeps both properties by keying on the SOURCE LINE'S TEXT plus an
    ordinal among identical texts in the same file. Unrelated edits above do
    not change it; a genuinely new occurrence either has different text (new
    key) or is byte-identical to an existing one (same text, next ordinal, so
    still a new key).
    """
    digest = hashlib.sha256(source_line.strip().encode("utf-8")).hexdigest()[:16]
    return f"{code}:{path}:{digest}:{ordinal}"


class _FingerprintCounter:
    """Assign stable ordinals to findings sharing one source-line text."""

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str, str], int] = {}

    def fingerprint(self, code: str, path: str, source_line: str) -> str:
        key = (code, path, source_line.strip())
        ordinal = self._seen.get(key, 0)
        self._seen[key] = ordinal + 1
        return finding_fingerprint(code, path, source_line, ordinal)


def _docstring_nodes(tree: ast.Module) -> frozenset[int]:
    """Ids of every docstring constant, which are prose rather than reads."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            found.add(id(first.value))
    return frozenset(found)


def scan_hermeticity(path: Path, tree: ast.Module, source_lines: Sequence[str]) -> tuple[OracleFinding, ...]:
    """Flag ambient user/archive reads in a test module.

    This enforces an ESCAPE, it does not re-implement isolation. The sanctioned
    boundary is ``tests/conftest.py``'s ``workspace_env`` /
    ``_clear_polylogue_env``, which point XDG and ``POLYLOGUE_ARCHIVE_ROOT`` at
    ``tmp_path``. A test that names an ambient path in source has stepped
    outside that boundary no matter what fixtures it also requests, so the
    check is a source scan rather than a runtime sandbox.
    """
    findings: list[OracleFinding] = []
    counter = _FingerprintCounter()
    docstrings = _docstring_nodes(tree)

    def line_at(lineno: int) -> str:
        return source_lines[lineno - 1] if 0 < lineno <= len(source_lines) else ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                # A module/class/function docstring naming ``~/.codex`` is
                # documentation, not a filesystem read. Flagging it made
                # ``tests/unit/core/test_paths.py`` a violation for explaining
                # the precedence ladder it tests.
                continue
            for marker in AMBIENT_PATH_MARKERS:
                if marker in node.value:
                    findings.append(
                        OracleFinding(
                            "ambient_path_literal",
                            str(path),
                            f"line {node.lineno}: reads ambient path {marker!r}",
                            counter.fingerprint("ambient_path_literal", str(path), line_at(node.lineno)),
                        )
                    )
        elif isinstance(node, ast.Call):
            dotted = _dotted_name(node.func)
            if dotted is not None and dotted in AMBIENT_CALL_MARKERS:
                findings.append(
                    OracleFinding(
                        "ambient_path_call",
                        str(path),
                        f"line {node.lineno}: resolves an ambient location via {dotted}()",
                        counter.fingerprint("ambient_path_call", str(path), line_at(node.lineno)),
                    )
                )
    return tuple(findings)


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return None if parent is None else f"{parent}.{node.attr}"
    return None


# ---------------------------------------------------------------------------
# Production scan: import-time home capture
# ---------------------------------------------------------------------------


#: Nodes whose bodies execute later, not at module import.
_DEFERRED_NODES = (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)


def _walk_eager(node: ast.AST) -> Iterator[ast.AST]:
    """Walk an expression, pruning subtrees that are NOT evaluated eagerly.

    ``ast.walk`` descends into everything, so a lambda or comprehension body
    inside a module-level assignment would be reported as an import-time
    capture even though its body runs later. Pruning those subtrees is what
    makes ``RESOLVER = lambda: Path.home()`` correctly silent while
    ``PROVIDERS = {...Path.home()...}`` is correctly flagged.
    """
    if isinstance(node, _DEFERRED_NODES):
        # The assignment's own value can BE the deferred node
        # (``RESOLVER = lambda: Path.home()``), so the root needs the same
        # pruning as any child -- checking only children silently let that
        # exact shape through.
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk_eager(child)


def scan_import_time_home_capture(
    path: Path, tree: ast.Module, source_lines: Sequence[str]
) -> tuple[OracleFinding, ...]:
    """Flag ambient-location calls evaluated at MODULE IMPORT time.

    A different failure shape from the test-side scan, and one the test-side
    scan structurally cannot see because it only reads ``tests/**``. A
    production module-level constant computed from ``Path.home()`` or
    ``expanduser`` is evaluated once, at first import -- which in a test process
    happens BEFORE any per-test environment patching. Every later test then
    reads a value captured from the developer's real home directory no matter
    how carefully it patches ``HOME``/XDG, and no test-side fixture can undo it.

    Only module scope is reported. The identical call inside a function body is
    fine: it is evaluated per call, so it observes whatever environment is
    patched at that moment. That distinction is the whole check -- flagging
    every occurrence would report ~109 sites in this package and mean nothing.
    """
    findings: list[OracleFinding] = []
    counter = _FingerprintCounter()

    def line_at(lineno: int) -> str:
        return source_lines[lineno - 1] if 0 < lineno <= len(source_lines) else ""

    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if value is None:
            continue
        for node in _walk_eager(value):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted_name(node.func)
            if dotted is None or dotted not in AMBIENT_CALL_MARKERS:
                continue
            findings.append(
                OracleFinding(
                    "import_time_home_capture",
                    str(path),
                    f"line {node.lineno}: module-level constant captures an ambient location via {dotted}()",
                    counter.fingerprint("import_time_home_capture", str(path), line_at(node.lineno)),
                )
            )
    return tuple(findings)


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

HERMETICITY_ALLOWLIST: tuple[OracleAllowlistEntry, ...] = (
    OracleAllowlistEntry(
        "tests/unit/devtools",
        "devtools tests assert on path-policy machinery itself (basetemp "
        "selection, checkout guard), so ambient path names are the subject "
        "under test rather than an escape.",
    ),
    OracleAllowlistEntry(
        "tests/unit/config",
        "Config resolution tests assert the XDG/archive-root ladder's own "
        "behaviour and must name the paths that ladder resolves.",
    ),
)


def _allowlisted(path: Path, source_root: Path, allowlist: Sequence[OracleAllowlistEntry]) -> bool:
    try:
        relative = path.relative_to(source_root).as_posix()
    except ValueError:
        relative = path.as_posix()
    return any(relative == entry.path or relative.startswith(f"{entry.path}/") for entry in allowlist)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


#: Findings that already existed when the lint was introduced. This is a
#: RATCHET, exactly like ``docs/plans/layering-surface-baseline.json``: a
#: pre-existing violation is exempted so the gate can be adopted green today,
#: while any NEW violation fails. Baseline entries are a worklist, not an
#: endorsement -- WS-F's deletion sweeps consume this file directly.
BASELINE_PATH = Path("docs/plans/oracle-integrity-baseline.json")


def load_baseline(source_root: Path, *, path: Path | None = None) -> frozenset[str]:
    """Load the fingerprints that are exempt because they pre-existed.

    Keyed on the DETAIL as well as the file, per-entry, the way the layering
    baseline is. Keying on ``(code, path)`` alone exempted an entire FILE once
    any finding in it was baselined -- a reviewer appended a brand-new
    ``~/.codex`` read to an already-baselined file and the gate stayed green
    while ``baselined`` silently rose 34 -> 35. A ratchet that can be widened
    by editing an exempted file is not a ratchet.
    """
    baseline_path = path or (source_root / BASELINE_PATH)
    if not baseline_path.is_file():
        return frozenset()
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    fingerprints: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("fingerprint"), str) and entry["fingerprint"]:
            fingerprints.add(entry["fingerprint"])
    return frozenset(fingerprints)


def check_oracle_integrity(
    source_root: Path,
    *,
    test_root: Path | None = None,
    hermeticity_allowlist: Sequence[OracleAllowlistEntry] = HERMETICITY_ALLOWLIST,
    baseline: frozenset[str] | None = None,
) -> OracleIntegrityReport:
    """Scan the test corpus for ambient reads and production for home capture."""
    source_root = source_root.resolve()
    tests_dir = (test_root or source_root / "tests").resolve()
    production_root = source_root / PRODUCTION_PACKAGE

    findings: list[OracleFinding] = []
    scanned = 0

    for path in sorted(tests_dir.rglob("test_*.py")):
        scanned += 1
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            findings.append(OracleFinding("unparseable_test", str(path), str(exc)))
            continue
        if not _allowlisted(path, source_root, hermeticity_allowlist):
            findings.extend(scan_hermeticity(path.relative_to(source_root), tree, source.splitlines()))

    for path in sorted(production_root.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        findings.extend(scan_import_time_home_capture(path.relative_to(source_root), tree, source.splitlines()))

    exempt = load_baseline(source_root) if baseline is None else baseline
    retained = tuple(finding for finding in findings if finding.fingerprint not in exempt)
    return OracleIntegrityReport(
        findings=retained,
        baselined=len(findings) - len(retained),
        scanned_modules=scanned,
    )


__all__ = [
    "AMBIENT_CALL_MARKERS",
    "BASELINE_PATH",
    "AMBIENT_PATH_MARKERS",
    "HERMETICITY_ALLOWLIST",
    "OracleAllowlistEntry",
    "OracleFinding",
    "OracleIntegrityReport",
    "check_oracle_integrity",
    "load_baseline",
    "finding_fingerprint",
    "scan_hermeticity",
    "scan_import_time_home_capture",
]

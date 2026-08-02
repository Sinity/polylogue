"""Verify no raw-capture write path mutates bytes before they reach the hasher.

Background
----------

Polylogue-u19l found a concrete, confirmed bug: ``sources/live/batch.py``'s
``_append_payload_for_provider`` used to prepend a synthetic
``{"type":"session_meta","payload":{"id":...}}`` line ahead of every Codex
append-mode capture's real tail bytes, *before* those bytes were hashed and
stored as the raw's content (``session_meta + b"\\n" + payload``, the exact
shape this lint forbids). That made the stored blob architecturally never a
literal byte-slice of the live file, permanently defeating any live-source
byte-identity re-verification even when the live file was completely
untouched (~59 GB of quarantined raw rows, ~95% of which were, in fact,
directly re-verifiable once the synthetic header was accounted for).

The fix (PR #3539, polylogue-u19l) carries the identity as sidecar metadata
(``raw_sessions.native_id``) instead of splicing it into the hashed bytes.
This lint (polylogue-ds4b4 item 2) generalizes the regression test: it
statically forbids the *pattern* that produced the bug -- concatenating a
freshly synthesized literal (a bytes/str constant, or the result of a
``json.dumps``/``.encode()`` call) onto a name/attribute reference -- anywhere
in the raw-capture write-path modules, not just in the one function that
already got fixed.

What this lint checks
----------------------

For every module in ``WRITE_PATH_MODULES`` (the modules that construct or
carry the ``bytes`` that ultimately reach ``write_raw_payload``/
``BlobPublisher.write_from_bytes`` -- the content-hashing boundary for raw
session capture), parse the AST and flag every ``BinOp`` using ``+`` where at
least one operand is a freshly synthesized literal (a bytes/str ``Constant``,
an f-string, or a call to something that looks like serialization --
``dumps``/``encode``) and the other operand is not itself another literal of
the same kind (i.e. this is a synthesize-and-splice, not two literals being
joined for an unrelated purpose, e.g. building a log message).

Scope and false-positive discipline
------------------------------------

In scope: only the fixed module list below -- the modules on the raw-capture
write path. An unrelated string-concatenation elsewhere in the codebase (log
messages, error text, SQL fragments) never fires. Within scope, ordinary
byte-slicing, ``.join()``, and passing a bare ``Name``/``Attribute`` through
unmodified are unaffected -- only literal-onto-variable splicing trips the
gate. There is no ack/exception mechanism: unlike classifier-fingerprint
drift (where a change can be a deliberate, reviewed decision), byte-mutation
before the content hasher has no legitimate use case on this write path --
identity/metadata belongs in a sidecar column, never spliced into the hashed
payload. If a genuinely new, legitimate need for payload-adjacent literal
bytes arises, it belongs in ``WRITE_PATH_MODULES`` exclusion criteria (edit
this module's own scope, with review), not a per-line escape hatch.

Wired into ``devtools verify --lab`` alongside the other static policy
checks: archive-independent, sub-second.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass

from devtools import repo_root as _get_root

ROOT = _get_root()

# The raw-capture write path: every module that constructs, transforms, or
# forwards the bytes that ultimately reach a raw payload hasher
# (``write_raw_payload`` / ``BlobPublisher.write_from_bytes``) for
# ``raw_sessions`` capture. Deliberately a fixed, reviewed list, not a
# directory glob -- widening scope to "everything under sources/" would
# false-positive on unrelated string building far from any hashed payload.
WRITE_PATH_MODULES: tuple[str, ...] = (
    "polylogue/sources/live/batch.py",
    "polylogue/sources/live/batch_support.py",
    "polylogue/sources/live/append_ingest.py",
    "polylogue/pipeline/services/archive_ingest.py",
    "polylogue/pipeline/services/acquisition_records.py",
    "polylogue/pipeline/services/ingest_batch/_core.py",
    "polylogue/sources/source_acquisition_components.py",
    "polylogue/sources/drive/__init__.py",
    "polylogue/storage/sqlite/archive_tiers/archive.py",
    "polylogue/storage/sqlite/archive_tiers/revision_governance.py",
)

_SERIALIZE_CALL_NAMES = {"dumps", "json_dumps", "encode"}


@dataclass(frozen=True, slots=True)
class HashPurityViolation:
    path: str
    lineno: int
    col_offset: int
    detail: str


def _looks_synthesized(node: ast.expr) -> bool:
    """A literal or serialization-call operand -- the "freshly minted" half."""
    if isinstance(node, ast.Constant) and isinstance(node.value, bytes | str):
        return True
    if isinstance(node, ast.JoinedStr):  # f-string
        return True
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if name in _SERIALIZE_CALL_NAMES:
            return True
    return False


def _looks_like_captured_bytes(node: ast.expr) -> bool:
    """A bare reference operand -- plausibly the read/captured payload."""
    return isinstance(node, ast.Name | ast.Attribute | ast.Subscript)


def _scan_binop(node: ast.BinOp, *, path: str) -> HashPurityViolation | None:
    if not isinstance(node.op, ast.Add):
        return None
    left, right = node.left, node.right
    synthesized_left, synthesized_right = _looks_synthesized(left), _looks_synthesized(right)
    captured_left, captured_right = _looks_like_captured_bytes(left), _looks_like_captured_bytes(right)
    # Flag only an asymmetric splice: one side freshly synthesized, the other
    # a bare reference. Two literals joined together (e.g. building a fixed
    # log-message prefix) or two references concatenated (e.g. joining two
    # already-captured buffers) are not the hazard this lint targets.
    if (synthesized_left and captured_right) or (synthesized_right and captured_left):
        return HashPurityViolation(
            path=path,
            lineno=node.lineno,
            col_offset=node.col_offset,
            detail="literal/serialized value concatenated onto a bare reference before hashing",
        )
    return None


def scan_source_for_payload_concatenation(source: str, *, path: str) -> list[HashPurityViolation]:
    """Return every literal-onto-reference splice in *source*.

    Exposed standalone (not just via the CLI) so a test can feed a synthetic
    source-string fixture directly, mirroring
    ``verify_timestamp_doctrine.scan_ddl_for_text_timestamps``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    violations: list[HashPurityViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            violation = _scan_binop(node, path=path)
            if violation is not None:
                violations.append(violation)
    return violations


def _collect_write_path_violations(modules: tuple[str, ...] = WRITE_PATH_MODULES) -> list[HashPurityViolation]:
    violations: list[HashPurityViolation] = []
    for rel in modules:
        full_path = ROOT / rel
        if not full_path.exists():
            continue
        source = full_path.read_text(encoding="utf-8")
        violations.extend(scan_source_for_payload_concatenation(source, path=rel))
    return violations


def _format_report(violations: list[HashPurityViolation]) -> str:
    if not violations:
        return (
            f"Raw-payload hash purity intact: no write-path module concatenates a "
            f"synthesized literal onto captured bytes before hashing ({len(WRITE_PATH_MODULES)} modules scanned)."
        )
    lines = [f"Raw-payload hash-purity violations: {len(violations)}", ""]
    for violation in violations:
        lines.append(f"  {violation.path}:{violation.lineno}: {violation.detail}")
    lines.append("")
    lines.append(
        "Policy violation (polylogue-ds4b4/u19l): a raw-capture write path must never splice a "
        "synthesized literal (bytes/str constant, f-string, json.dumps()/.encode() result) onto "
        "captured bytes before they reach the content hasher. Carry identity/metadata as sidecar "
        "data (a separate column/return value), and reconstruct any parseable header at READ time "
        "instead, per _append_payload_for_provider's docstring in polylogue/sources/live/batch.py."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    violations = _collect_write_path_violations()

    if args.json:
        payload = {
            "violations": [
                {"path": v.path, "lineno": v.lineno, "col_offset": v.col_offset, "detail": v.detail} for v in violations
            ],
            "modules_scanned": list(WRITE_PATH_MODULES),
            "ok": not violations,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(violations))

    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())

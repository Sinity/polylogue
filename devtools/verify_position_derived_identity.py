"""Verify no parser mints comparison identity from positional/index data.

Background
----------

polylogue-hith/qkuq found and fixed a concrete bug: the Claude.ai attachment
parser's synthetic id (``att-<hash(message_id:name:index)>``) was seeded
partly by array index, so a re-export that reordered or inserted attachment
entries silently produced a different id for "the same" attachment,
manufacturing false divergence in revision-authority membership comparison.
The fix was not to remove the synthetic-id *generator* (still legitimate as
a last-resort storage/display id, seed no longer includes ``index``) but to
make ``attachment_identity_hash`` (``polylogue/pipeline/ids.py``) stop
reading either the real or synthetic attachment id at all for comparison
purposes.

polylogue-gysk3 found the identical hazard one level up:
``message_identity_hash`` (``polylogue/pipeline/ids.py``) hashes a message's
``id`` directly, and that id IS ``provider_message_id`` -- multiple parsers
construct it as ``f"msg-{index}"`` or similar whenever the raw record
carries no native id of its own. The production repair removed every known
instance. This lint (ds4b4 item 3) prevents the same shape returning in a
parser or shared parser helper.

What this lint checks
----------------------

Scans ``polylogue/sources/parsers/`` (and ``base_support.py``-style shared
parser helpers alongside it) for an assignment or keyword argument whose
target name is in ``IDENTITY_FIELD_NAMES`` (fields whose value is used as
cross-revision comparison identity, not just a display/storage label) where
the value expression is an f-string, ``str.format()`` call, or ``+``
concatenation that references a variable named like a loop index/position
(``index``, ``idx``, ``i``, ``pos``, ``position`` -- ``ENUMERATE_VAR_NAMES``).

No current finding needs an acknowledgement. The optional acknowledgement
manifest (``docs/plans/position-derived-identity-acks.json``) is absent while
the audit is clean and is created only by ``--ack`` for a newly discovered,
temporarily accepted finding with a tracked follow-up. An unacknowledged
occurrence fails the gate; a stale acknowledgement fails it too.

Scope and false-positive discipline
------------------------------------

Only ``IDENTITY_FIELD_NAMES`` trips this lint -- an ordinary loop variable
named ``index`` used for anything else (slicing, logging, an unrelated
counter) is untouched. ``ENUMERATE_VAR_NAMES`` is deliberately narrow (common
loop-counter spellings); a differently-named counter is a residual
false-negative, not a false-positive, and acceptable for a preventive lint of
this kind (the goal is to catch the common, already-observed shape, not
build a full data-flow/taint analysis). Position-derived values used for
purely *structural* addressing that is never compared across independent
captures of the same logical entity (e.g. ``blocks.block_id`` -- see
CLAUDE.md's data model section) are out of scope by construction: this lint
only matches the specific field-name list, not every f-string containing an
index.

Wired into ``devtools verify --lab`` (like classifier-fingerprints):
static, archive-independent, sub-second.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from devtools import repo_root as _get_root

ROOT = _get_root()
PARSERS_ROOT = ROOT / "polylogue" / "sources" / "parsers"
MANIFEST_PATH = ROOT / "docs" / "plans" / "position-derived-identity-acks.json"

#: Field names whose value is used as cross-revision comparison identity
#: (not just a storage/display label). Extend deliberately, with review --
#: this is the allowlist that scopes the whole lint.
IDENTITY_FIELD_NAMES = frozenset({"provider_message_id"})

#: Common loop-counter/index spellings. Deliberately narrow (see module
#: docstring's false-positive-discipline section).
ENUMERATE_VAR_NAMES = frozenset({"index", "idx", "i", "pos", "position"})

_REF_PATTERN = re.compile(r"^(polylogue-[a-z0-9][a-z0-9.]*|#\d+)$")
_MIN_REASON_LEN = 15


def _references_enumerate_var(node: ast.AST) -> bool:
    return any(isinstance(sub, ast.Name) and sub.id in ENUMERATE_VAR_NAMES for sub in ast.walk(node))


def _value_is_position_derived(value: ast.expr) -> bool:
    if isinstance(value, ast.JoinedStr):
        return _references_enumerate_var(value)
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
        return _references_enumerate_var(value)
    if isinstance(value, ast.Call):
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if name == "format" and _references_enumerate_var(value):
            return True
        # `str(x or f"...{index}")`, `str(x or f"...{index}")` etc: unwrap a
        # single-arg str()/BoolOp `or` chain to reach the fallback literal.
        if name == "str" and len(value.args) == 1:
            return _value_is_position_derived(value.args[0])
    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
        return any(_value_is_position_derived(v) for v in value.values)
    return False


@dataclass(frozen=True, slots=True)
class PositionIdentityFinding:
    qualname: str
    path: str
    lineno: int
    field: str


def _enclosing_function_name(stack: list[ast.AST]) -> str:
    for node in reversed(stack):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node.name
    return "<module>"


def _resolve_via_local_bindings(value: ast.expr, bindings: dict[str, ast.expr], *, depth: int = 0) -> bool:
    """Follow a bare-name reference back to its last local assignment.

    The common shape in this codebase is not inline construction at the
    identity-field keyword/assign site -- it is ``msg_id = ... or
    f"msg-{idx}"`` a few lines earlier, then ``provider_message_id=msg_id``.
    A one-hop-per-name, last-write-wins backward resolution (approximating
    straight-line control flow, which is what these parsers use) catches
    that without a full data-flow analysis. ``depth`` guards against
    pathological self-referential rebinding.
    """
    if depth > 5:
        return False
    if _value_is_position_derived(value):
        return True
    if isinstance(value, ast.Name) and value.id in bindings:
        return _resolve_via_local_bindings(bindings[value.id], bindings, depth=depth + 1)
    if isinstance(value, ast.Call) and len(value.args) == 1:
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if name == "str":
            return _resolve_via_local_bindings(value.args[0], bindings, depth=depth + 1)
    if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
        return any(_resolve_via_local_bindings(v, bindings, depth=depth + 1) for v in value.values)
    return False


def _scan_module(source: str, *, rel_path: str) -> list[PositionIdentityFinding]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    findings: list[PositionIdentityFinding] = []
    ordinal_by_key: dict[str, int] = {}
    stack: list[ast.AST] = []
    # Reset per function: a straight-line, last-write-wins map of local
    # simple-name bindings, rebuilt fresh at each FunctionDef so a name in
    # one function never leaks into another's resolution.
    bindings_by_function: dict[int, dict[str, ast.expr]] = {}

    def _current_bindings() -> dict[str, ast.expr]:
        for node in reversed(stack):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                return bindings_by_function.setdefault(id(node), {})
        return bindings_by_function.setdefault(0, {})

    def _record(field: str, value: ast.expr, *, lineno: int) -> None:
        if not _resolve_via_local_bindings(value, _current_bindings()):
            return
        func_name = _enclosing_function_name(stack)
        key = f"{rel_path}:{func_name}:{field}"
        ordinal = ordinal_by_key.get(key, 0)
        ordinal_by_key[key] = ordinal + 1
        qualname = key if ordinal == 0 else f"{key}#{ordinal}"
        findings.append(PositionIdentityFinding(qualname=qualname, path=rel_path, lineno=lineno, field=field))

    class _Visitor(ast.NodeVisitor):
        def generic_visit(self, node: ast.AST) -> None:
            stack.append(node)
            super().generic_visit(node)
            stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id in IDENTITY_FIELD_NAMES:
                        _record(target.id, node.value, lineno=node.lineno)
                    _current_bindings()[target.id] = node.value
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name) and node.value is not None:
                if node.target.id in IDENTITY_FIELD_NAMES:
                    _record(node.target.id, node.value, lineno=node.lineno)
                _current_bindings()[node.target.id] = node.value
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            for kw in node.keywords:
                if kw.arg in IDENTITY_FIELD_NAMES:
                    _record(kw.arg, kw.value, lineno=kw.value.lineno)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return findings


def collect_position_derived_identity(root: Path = PARSERS_ROOT) -> dict[str, PositionIdentityFinding]:
    """Return every in-scope finding, keyed by a line-shift-stable qualname.

    Exposed standalone so a test can feed a synthetic source string via
    ``_scan_module`` directly.
    """
    found: dict[str, PositionIdentityFinding] = {}
    if not root.exists():
        return found
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        for finding in _scan_module(source, rel_path=rel):
            found[finding.qualname] = finding
    return found


@dataclass(frozen=True, slots=True)
class AckEntry:
    reason: str
    ref: str


class ManifestJSON(TypedDict):
    acknowledged: dict[str, dict[str, str]]


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, AckEntry]:
    if not path.exists():
        return {}
    raw: ManifestJSON = json.loads(path.read_text(encoding="utf-8"))
    return {
        qualname: AckEntry(reason=str(data["reason"]), ref=str(data["ref"]))
        for qualname, data in raw.get("acknowledged", {}).items()
    }


def save_manifest(entries: dict[str, AckEntry], path: Path = MANIFEST_PATH) -> None:
    payload: ManifestJSON = {
        "acknowledged": {
            qualname: {"reason": entry.reason, "ref": entry.ref} for qualname, entry in sorted(entries.items())
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _validate_ack(reason: str, ref: str) -> str | None:
    if len(reason.strip()) < _MIN_REASON_LEN:
        return f"reason too short (must explain the tracked follow-up, >= {_MIN_REASON_LEN} chars)"
    if not _REF_PATTERN.match(ref):
        return "ref must be a bead id (polylogue-xxxx) or issue number (#N)"
    return None


@dataclass(frozen=True, slots=True)
class DriftReport:
    unacknowledged: tuple[str, ...]
    stale: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.unacknowledged and not self.stale


def compute_drift_report(
    current: dict[str, PositionIdentityFinding] | None = None,
    manifest: dict[str, AckEntry] | None = None,
) -> DriftReport:
    if current is None:
        current = collect_position_derived_identity()
    if manifest is None:
        manifest = load_manifest()
    unacknowledged = tuple(sorted(q for q in current if q not in manifest))
    stale = tuple(sorted(q for q in manifest if q not in current))
    return DriftReport(unacknowledged=unacknowledged, stale=stale)


def _format_report(report: DriftReport, current: dict[str, PositionIdentityFinding]) -> str:
    lines = [
        f"unacknowledged position-derived identity constructions: {len(report.unacknowledged)}",
        f"stale manifest entries (finding no longer exists): {len(report.stale)}",
    ]
    if report.unacknowledged:
        lines.append("")
        lines.append(
            "New position-derived comparison-identity constructions (see module docstring -- "
            "polylogue-hith/qkuq's already-fixed attachment-id bug, polylogue-gysk3's tracked "
            "message-id sibling):"
        )
        for qualname in report.unacknowledged:
            finding = current[qualname]
            lines.append(f"  {finding.path}:{finding.lineno} ({finding.field}) [{qualname}]")
        lines.append(
            "  Fix: either derive identity from provider-native data instead of position, or "
            "acknowledge with `devtools lab policy position-derived-identity --ack <qualname> "
            "--reason '...' --ref <bead-or-issue>` naming a tracked follow-up."
        )
    if report.stale:
        lines.append("")
        lines.append("Manifest entries whose finding no longer exists (remove them):")
        for qualname in report.stale:
            lines.append(f"  {qualname}")
        lines.append(f"  Fix: edit {MANIFEST_PATH.relative_to(ROOT)} and delete the stale entry.")
    if report.ok:
        lines.append("")
        lines.append("Position-derived identity policy intact.")
    return "\n".join(lines)


def _cmd_ack(qualname: str, *, reason: str, ref: str) -> int:
    current = collect_position_derived_identity()
    if qualname not in current:
        print(f"error: {qualname!r} is not a currently discovered finding", file=sys.stderr)
        return 2
    problem = _validate_ack(reason, ref)
    if problem is not None:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    manifest = load_manifest()
    manifest[qualname] = AckEntry(reason=reason, ref=ref)
    save_manifest(manifest)
    print(f"recorded {qualname}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--ack",
        metavar="QUALNAME",
        help="record an acknowledged position-derived identity finding, naming a tracked follow-up",
    )
    parser.add_argument("--reason", help="justification text for --ack (required with --ack)")
    parser.add_argument("--ref", help="bead id (polylogue-xxxx) or issue number (#N) for --ack (required with --ack)")
    args = parser.parse_args(argv)

    if args.ack:
        if not args.reason or not args.ref:
            parser.error("--ack requires --reason and --ref")
        return _cmd_ack(args.ack, reason=args.reason, ref=args.ref)

    current = collect_position_derived_identity()
    report = compute_drift_report(current)

    if args.json:
        print(
            json.dumps(
                {
                    "unacknowledged": list(report.unacknowledged),
                    "stale": list(report.stale),
                    "ok": report.ok,
                },
                indent=2,
            )
        )
    else:
        print(_format_report(report, current))

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

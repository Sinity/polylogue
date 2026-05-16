"""Static scan: no logger/print call interpolates raw secret variables.

The repository has several names that commonly hold secrets:

* ``api_key``, ``voyage_api_key``
* ``auth_token``, ``access_token``, ``refresh_token``, ``token``
* ``password``, ``secret``
* ``Bearer`` (HTTP header literal that often precedes a token)

A ``logger.<level>("...", auth_token)`` or
``print(f"... {api_key}")`` will quietly emit those values into log files,
crash reports, or sentry payloads. This test parses every production
module under ``polylogue/`` and fails when it finds a logging/printing
call that takes one of the above names as a positional/keyword arg or
interpolates it inside an f-string.

A small explicit whitelist (``_AUDITED_SITES``) records lines that have
been reviewed and proven safe (e.g. logging *that* a token was loaded
without logging its value). Adding to the whitelist requires a comment
naming the call shape and why it is safe.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_PRODUCTION_ROOT: Final = Path("polylogue")

_SECRET_NAMES: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "voyage_api_key",
        "anthropic_api_key",
        "openai_api_key",
        "auth_token",
        "access_token",
        "refresh_token",
        "bearer_token",
        "id_token",
        "password",
        "passphrase",
        "secret",
        "client_secret",
        "private_key",
    }
)

# Substrings of f-string text or string constants that strongly suggest the
# call is emitting a secret value rather than a label.
_SECRET_LITERAL_SUBSTRINGS: Final[tuple[str, ...]] = (
    "Bearer ",
    "Authorization: ",
    "api_key=",
    "token=",
    "password=",
)


_LOGGING_FUNC_NAMES: Final[frozenset[str]] = frozenset(
    {"debug", "info", "warning", "warn", "error", "critical", "exception", "log"}
)

# (relative_path, line) -> rationale. Use sparingly; prefer rewriting the call.
_AUDITED_SITES: Final[dict[tuple[str, int], str]] = {}


def _is_logging_or_print_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "print":
        return True
    return isinstance(func, ast.Attribute) and func.attr in _LOGGING_FUNC_NAMES


def _flatten_names(node: ast.expr) -> set[str]:
    """Collect every identifier reachable from ``node`` (Name + Attribute tail)."""
    found: set[str] = set()

    class _Walker(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name) -> None:
            found.add(n.id.lower())

        def visit_Attribute(self, n: ast.Attribute) -> None:
            found.add(n.attr.lower())
            self.generic_visit(n)

    _Walker().visit(node)
    return found


def _joined_str_contains_secret_substring(node: ast.JoinedStr) -> bool:
    for part in node.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            for needle in _SECRET_LITERAL_SUBSTRINGS:
                if needle in part.value:
                    return True
    return False


def _call_arg_references_secret(node: ast.Call) -> str | None:
    """Return a description of the secret reference, or None."""
    # f-strings: every formatted-value expression is inspected; their text
    # constants are scanned for "Bearer ..."-style fixtures.
    for arg in list(node.args) + [kw.value for kw in node.keywords]:
        if isinstance(arg, ast.JoinedStr):
            if _joined_str_contains_secret_substring(arg):
                return "f-string contains literal secret prefix"
            for fv in arg.values:
                if isinstance(fv, ast.FormattedValue):
                    names = _flatten_names(fv.value)
                    hit = names & _SECRET_NAMES
                    if hit:
                        return f"f-string interpolates secret name(s): {sorted(hit)}"
        elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            for needle in _SECRET_LITERAL_SUBSTRINGS:
                if needle in arg.value:
                    return f"string constant contains secret prefix {needle!r}"
        else:
            names = _flatten_names(arg)
            hit = names & _SECRET_NAMES
            if hit:
                return f"positional/keyword arg references secret name(s): {sorted(hit)}"
    return None


class _LogCallVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[tuple[str, int, str, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        if _is_logging_or_print_call(node):
            why = _call_arg_references_secret(node)
            if why is not None:
                rel = self.path.as_posix()
                if (rel, node.lineno) not in _AUDITED_SITES:
                    self.violations.append((rel, node.lineno, why, _read_line(self.path, node.lineno)))
        self.generic_visit(node)


def _read_line(path: Path, line_no: int) -> str:
    try:
        return path.read_text(encoding="utf-8").splitlines()[line_no - 1].strip()
    except (OSError, IndexError):
        return "<source unavailable>"


def test_no_secret_leak_in_log_or_print_calls() -> None:
    violations: list[tuple[str, int, str, str]] = []
    for path in sorted(_PRODUCTION_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            raise AssertionError(f"{path} failed to parse: {exc}") from exc
        visitor = _LogCallVisitor(path)
        visitor.visit(tree)
        violations.extend(visitor.violations)

    assert not violations, (
        "logger/print call interpolating a secret-bearing name:\n"
        + "\n".join(f"  {rel}:{line} [{why}]\n    {src}" for rel, line, why, src in violations)
        + "\n\nReplace the value with a stable identifier (length, hash prefix, "
        "or boolean 'present?'); or, if the call is provably safe, add an entry "
        "to _AUDITED_SITES with rationale."
    )


def test_audited_sites_are_real_violations() -> None:
    """Keep `_AUDITED_SITES` honest: every entry must still be a flagged call."""
    if not _AUDITED_SITES:
        return
    audited_by_path: dict[Path, set[int]] = {}
    for rel, line in _AUDITED_SITES:
        audited_by_path.setdefault(Path(rel), set()).add(line)
    stale: list[tuple[str, int]] = []
    for path, lines in audited_by_path.items():
        if not path.exists():
            stale.extend((path.as_posix(), line) for line in lines)
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        flagged: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_logging_or_print_call(node):
                continue
            if _call_arg_references_secret(node) is not None:
                flagged.add(node.lineno)
        stale.extend((path.as_posix(), line) for line in lines if line not in flagged)
    assert not stale, "Stale _AUDITED_SITES entries:\n" + "\n".join(f"  {rel}:{line}" for rel, line in stale)

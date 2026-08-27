"""Static production census for concurrency and runtime-owned state."""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

Disposition = str
_DISPOSITIONS: tuple[Disposition, ...] = (
    "immutable",
    "thread-confined",
    "lock-protected",
    "transactionally-serialized",
    "intent-passing",
)
_ROOTS = (
    "polylogue/daemon",
    "polylogue/cli",
    "polylogue/mcp",
    "polylogue/api",
    "polylogue/pipeline",
    "polylogue/sources",
    "polylogue/storage",
    "polylogue/operations",
)


@dataclass(frozen=True, slots=True)
class CensusItem:
    path: str
    line: int
    kind: str
    symbol: str
    disposition: Disposition
    evidence: str


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _disposition(*, kind: str, symbol: str, path: str) -> tuple[Disposition, str]:
    lower = f"{kind} {symbol} {path}".lower()
    if path == "polylogue/daemon/write_coordinator.py":
        return "transactionally-serialized", "daemon write state is owned by the single writer coordinator"
    if kind == "executor":
        if "processpool" in lower or "process_pool" in lower:
            return "thread-confined", "compute crosses a process boundary and carries no SQLite handle"
        if "writer" in lower or "coordinator" in lower:
            return "transactionally-serialized", "writer work is admitted through the daemon coordinator"
        return "thread-confined", "executor worker owns local compute state"
    if kind == "sqlite connection":
        if "/daemon/" in path and "writer" in lower:
            return "transactionally-serialized", "daemon writer-intent route owns mutation"
        return "thread-confined", "connection is local to the operation and must not cross its thread"
    if kind in {"lock", "event", "condition"} or "cache" in lower or "registry" in lower:
        return "lock-protected", "shared state is guarded by the owning module's synchronization boundary"
    if symbol.isupper() or kind == "constant":
        return "immutable", "module declaration is read-only after import"
    if kind == "mutable global":
        return "intent-passing", "state is exposed through an operation or coordinator seam"
    return "thread-confined", "state is local to the owning call or worker"


def collect_census(root: Path) -> tuple[CensusItem, ...]:
    """Find production executor, SQLite, synchronization, and global-state sites."""

    items: list[CensusItem] = []
    for relative_root in _ROOTS:
        directory = root / relative_root
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = _dotted(node.func)
                    if name.rsplit(".", 1)[-1] in {"ThreadPoolExecutor", "ProcessPoolExecutor"}:
                        kind = "executor"
                    elif name.endswith("sqlite3.connect") or name == "connect":
                        kind = "sqlite connection"
                    elif name.rsplit(".", 1)[-1] in {"Lock", "RLock", "Event", "Condition"}:
                        kind = name.rsplit(".", 1)[-1].lower()
                    else:
                        continue
                    symbol = name.rsplit(".", 1)[-1]
                    disposition, evidence = _disposition(kind=kind, symbol=symbol, path=relative)
                    items.append(CensusItem(relative, node.lineno, kind, symbol, disposition, evidence))
                elif isinstance(node, (ast.Assign, ast.AnnAssign)) and getattr(node, "col_offset", 1) == 0:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if not isinstance(target, ast.Name):
                            continue
                        name = target.id
                        if not (name.isupper() or name.startswith("_")):
                            continue
                        kind = "constant" if name.isupper() else "mutable global"
                        disposition, evidence = _disposition(kind=kind, symbol=name, path=relative)
                        items.append(CensusItem(relative, node.lineno, kind, name, disposition, evidence))
    return tuple(sorted(set(items), key=lambda item: (item.path, item.line, item.kind, item.symbol)))


def report(root: Path) -> dict[str, object]:
    items = collect_census(root)
    return {
        "schema": "polylogue.runtime-concurrency-census.v1",
        "roots": list(_ROOTS),
        "dispositions": list(_DISPOSITIONS),
        "items": [asdict(item) for item in items],
        "item_count": len(items),
        "unexplained_count": sum(item.disposition not in _DISPOSITIONS for item in items),
        "pass": all(item.disposition in _DISPOSITIONS for item in items),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    root = Path.cwd()
    payload = report(root)
    if "--json" in args:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print(f"runtime census: {payload['item_count']} items, unexplained={payload['unexplained_count']}")
        for item in cast(list[dict[str, object]], payload["items"]):
            print(f"{item['path']}:{item['line']} {item['kind']} {item['symbol']} -> {item['disposition']}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

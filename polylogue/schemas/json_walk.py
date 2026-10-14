"""Finite JSON graph custody shared by inference and synthetic verification."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence


def require_finite_json(value: object) -> None:
    """Reject cycles while preserving repeated positions in an acyclic graph."""
    active: set[int] = set()
    completed: set[int] = set()
    stack: list[tuple[int, Iterator[object]]] = []
    current = value
    while True:
        children: Iterator[object] | None = None
        if isinstance(current, Mapping):
            children = iter(current.values())
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            children = iter(current)
        if children is not None and id(current) not in completed:
            identity = id(current)
            if identity in active:
                raise ValueError("cyclic objects have no finite JSON representation")
            active.add(identity)
            stack.append((identity, children))
        while stack:
            identity, children = stack[-1]
            try:
                current = next(children)
                break
            except StopIteration:
                stack.pop()
                active.remove(identity)
                completed.add(identity)
        else:
            return


def clone_finite_json(value: object) -> object:
    """Copy schema containers without native recursion or duplicate alias work."""
    require_finite_json(value)
    copies: dict[int, dict[object, object] | list[object]] = {}
    pending: list[tuple[object, dict[object, object] | list[object]]] = []

    def copied(item: object) -> object:
        if not isinstance(item, (dict, list)):
            return item
        if id(item) not in copies:
            result: dict[object, object] | list[object] = {} if isinstance(item, dict) else []
            copies[id(item)] = result
            pending.append((item, result))
        return copies[id(item)]

    result = copied(value)
    while pending:
        original, target = pending.pop()
        if isinstance(original, dict) and isinstance(target, dict):
            for key, child in original.items():
                target[key] = copied(child)
        elif isinstance(original, list) and isinstance(target, list):
            target.extend(copied(child) for child in original)
    return result

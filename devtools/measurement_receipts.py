"""Commit measurement receipts beside the static ratchet baselines.

Gate classification: **not a gate**. This is a place to put evidence, not a
new obligation to produce it, and nothing schedules it.

The static verification gates already own their committed evidence:
``devtools/verify_schema_closure.py`` writes ``docs/plans/schema-closure-baseline.json``
and ``devtools/verify_oracle_integrity.py`` writes its hermeticity baseline.
Measurement evidence had no equivalent owner, so
``tests/benchmarks/baselines/`` held a single ``.gitkeep`` while
``tests/benchmarks/test_finished_build_measurement.py`` built a complete typed
receipt -- work identity, route, elapsed/CPU/RSS/storage, censuses, schema
identity, canonical logical digest -- and then ``print()``ed it. The receipt
was already constructed; only its destination was missing (polylogue-cjyfw).

**Boundary against ``tests/benchmarks/floors.json``.** That artifact is a
*ratchet* over scalar metrics with a per-metric tolerance, measured and checked
by the nightly ``tests/benchmarks/perf_floors.py`` lane; a metric that falls
past its tolerance is a regression finding. This route is the other half: it
commits a whole typed *receipt* for one named measurement, and its movement
rule is **explained, not ratcheted**. A number that moves is accepted once a
reason is recorded, because unlike a static closure a measurement legitimately
moves with the host it ran on. The two do not overlap and neither replaces the
other.

Usage::

    devtools bench baseline --list
    devtools bench baseline --record .cache/measurements/<name>.json \\
        --reason "first recorded arm for the sealed 516-raw finished build"

A benchmark emits its observed receipt with :func:`emit_receipt`, which writes
under ``.cache/measurements`` (disposable checkout scratch, overridable with
``POLYLOGUE_MEASUREMENT_RECEIPT_DIR``). Promoting one of those files to the
committed baseline is the deliberate, reviewable act ``--record`` performs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devtools import repo_root

#: The home the tree reserved for committed measurement evidence.
BASELINE_DIR = Path("tests/benchmarks/baselines")

#: Where an unpromoted observation lands. Disposable, gitignored scratch.
DEFAULT_RECEIPT_DIR = Path(".cache/measurements")

#: Redirects :func:`emit_receipt` when a runner owns its own output tree.
RECEIPT_DIR_ENV = "POLYLOGUE_MEASUREMENT_RECEIPT_DIR"

BASELINE_FORMAT = "polylogue.measurement-baseline.v1"


class MeasurementBaselineError(RuntimeError):
    """A baseline could not be read, written, or trusted."""


@dataclass(frozen=True, slots=True)
class Movement:
    """One leaf of the measurement that differs from the committed baseline."""

    path: str
    before: object
    after: object

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "before": self.before, "after": self.after}

    def __str__(self) -> str:
        return f"{self.path}: {self.before!r} -> {self.after!r}"


_ABSENT = object()


def _leaves(value: object, prefix: str = "") -> dict[str, object]:
    """Flatten a receipt to ``path -> scalar`` so nested numbers stay visible.

    A top-level comparison would report "the receipt changed" and lose which
    number moved, which is exactly the fact the recorded reason has to explain.
    """
    if isinstance(value, Mapping):
        leaves: dict[str, object] = {}
        for key in value:
            leaves.update(_leaves(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = {}
        for position, item in enumerate(value):
            leaves.update(_leaves(item, f"{prefix}[{position}]"))
        return leaves
    return {prefix or ".": value}


def measurement_movement(before: object, after: object) -> tuple[Movement, ...]:
    """Name every leaf that differs, including leaves added or dropped."""
    old = _leaves(before)
    new = _leaves(after)
    movements = [
        Movement(path, old.get(path, _ABSENT), new.get(path, _ABSENT))
        for path in sorted(set(old) | set(new))
        if old.get(path, _ABSENT) != new.get(path, _ABSENT)
    ]
    return tuple(movements)


def measurement_digest(measurement: object) -> str:
    """Digest the measurement payload so a baseline can name what it superseded."""
    encoded = json.dumps(measurement, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def host_fingerprint() -> dict[str, object]:
    """Record the measurement-relevant host facts, and no host identifier.

    A measurement is only comparable against one taken on a comparable
    machine, so the interpreter and CPU width belong in the committed file.
    The hostname and kernel release do not: they identify the operator's
    machine without making any number more interpretable.
    """
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "free_threaded": not bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
        "thread_inherit_context": (
            bool(thread_inherit_context)
            if (thread_inherit_context := getattr(sys.flags, "thread_inherit_context", None)) is not None
            else None
        ),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def receipt_dir(*, root: Path | None = None, env: Mapping[str, str] | None = None) -> Path:
    """Resolve where an observed (not yet committed) receipt is written."""
    environ = os.environ if env is None else env
    override = environ.get(RECEIPT_DIR_ENV)
    if override:
        return Path(override)
    return (repo_root() if root is None else root) / DEFAULT_RECEIPT_DIR


def emit_receipt(
    name: str,
    measurement: object,
    *,
    root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Write one observed measurement receipt and return its path.

    This is the destination a benchmark emits through instead of ``print``.
    It never touches the committed baseline: promoting an observation is the
    separate, deliberate ``devtools bench baseline --record`` step.
    """
    directory = receipt_dir(root=root, env=env)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    payload = {"name": name, "host": host_fingerprint(), "measurement": measurement}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def baseline_path(name: str, *, root: Path | None = None) -> Path:
    return (repo_root() if root is None else root) / BASELINE_DIR / f"{name}.json"


def load_baseline(path: Path) -> dict[str, Any] | None:
    """Read a committed baseline, or ``None`` when none is committed yet."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MeasurementBaselineError(f"{path} is not readable measurement JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format") != BASELINE_FORMAT:
        raise MeasurementBaselineError(f"{path} does not declare format {BASELINE_FORMAT!r}")
    return payload


def write_baseline(
    path: Path,
    *,
    name: str,
    measurement: object,
    host: Mapping[str, object],
    reason: str,
    supersedes: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Commit one measurement receipt with the reason its numbers stand."""
    if not reason.strip():
        raise MeasurementBaselineError("a committed measurement needs a non-empty reason")
    payload: dict[str, Any] = {
        "format": BASELINE_FORMAT,
        "name": name,
        "reason": reason.strip(),
        "host": dict(host),
        "measurement_digest": measurement_digest(measurement),
        "supersedes": dict(supersedes) if supersedes is not None else None,
        "measurement": measurement,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def committed_baselines(*, root: Path | None = None) -> tuple[dict[str, Any], ...]:
    """Every committed baseline, newest-format first and refusing junk."""
    directory = (repo_root() if root is None else root) / BASELINE_DIR
    if not directory.is_dir():
        return ()
    payloads = []
    for path in sorted(directory.glob("*.json")):
        payload = load_baseline(path)
        if payload is not None:
            payloads.append(payload)
    return tuple(payloads)


def _read_observation(path: Path) -> tuple[str | None, dict[str, object] | None, object]:
    """Split an emitted receipt into its name, host fingerprint and payload."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MeasurementBaselineError(f"{path} is not readable measurement JSON: {exc}") from exc
    if isinstance(payload, dict) and "measurement" in payload:
        host = payload.get("host")
        return (
            str(payload["name"]) if payload.get("name") else None,
            dict(host) if isinstance(host, Mapping) else None,
            payload["measurement"],
        )
    # A receipt written by a runner that does not use ``emit_receipt`` is
    # still recordable: the file itself is the measurement.
    return None, None, payload


def _record(args: argparse.Namespace, root: Path) -> tuple[int, dict[str, Any]]:
    observation = Path(args.record)
    if not observation.is_file():
        raise MeasurementBaselineError(f"no such measurement receipt: {observation}")
    emitted_name, emitted_host, measurement = _read_observation(observation)
    name = args.name or emitted_name or observation.stem
    host = emitted_host if emitted_host is not None else host_fingerprint()

    target = baseline_path(name, root=root)
    existing = load_baseline(target)
    movements = () if existing is None else measurement_movement(existing.get("measurement"), measurement)

    if existing is not None and not movements:
        return 0, {
            "action": "unchanged",
            "name": name,
            "path": str(target.relative_to(root)),
            "movements": [],
        }

    if not args.reason:
        return 1, {
            "action": "refused",
            "name": name,
            "path": str(target.relative_to(root)),
            "reason_required": (
                "a first committed measurement states why it stands"
                if existing is None
                else "a measurement that moved needs a recorded reason"
            ),
            "movements": [movement.to_dict() for movement in movements],
        }

    supersedes = (
        None
        if existing is None
        else {
            "measurement_digest": existing.get("measurement_digest"),
            "reason": existing.get("reason"),
        }
    )
    payload = write_baseline(
        target,
        name=name,
        measurement=measurement,
        host=host,
        reason=args.reason,
        supersedes=supersedes,
    )
    return 0, {
        "action": "recorded" if existing is None else "moved",
        "name": name,
        "path": str(target.relative_to(root)),
        "measurement_digest": payload["measurement_digest"],
        "reason": payload["reason"],
        "movements": [movement.to_dict() for movement in movements],
    }


def _list(root: Path) -> tuple[int, dict[str, Any]]:
    entries = [
        {
            "name": payload.get("name"),
            "reason": payload.get("reason"),
            "measurement_digest": payload.get("measurement_digest"),
            "host": payload.get("host"),
        }
        for payload in committed_baselines(root=root)
    ]
    return 0, {"action": "list", "baselines": entries, "directory": str(BASELINE_DIR)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools bench baseline",
        description="List or record committed measurement receipts under tests/benchmarks/baselines/.",
    )
    parser.add_argument("--list", action="store_true", help="List the committed measurement baselines.")
    parser.add_argument(
        "--record",
        metavar="RECEIPT",
        help="Promote an emitted measurement receipt to the committed baseline.",
    )
    parser.add_argument("--name", help="Baseline name; defaults to the receipt's own name.")
    parser.add_argument(
        "--reason",
        help="Why these numbers stand. Required for a first record and for any movement.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the result as JSON.")
    return parser


def _render(result: Mapping[str, object]) -> None:
    action = result.get("action")
    if action == "list":
        baselines = result.get("baselines")
        entries = baselines if isinstance(baselines, Sequence) else ()
        if not entries:
            print(f"No committed measurement baselines under {result.get('directory')}.")
            return
        for entry in entries:
            assert isinstance(entry, Mapping)
            print(f"{entry.get('name')}  {entry.get('measurement_digest')}")
            print(f"  reason: {entry.get('reason')}")
        return
    if action == "refused":
        print(f"refused: {result.get('reason_required')}", file=sys.stderr)
        movements = result.get("movements")
        for movement in movements if isinstance(movements, Sequence) else ():
            assert isinstance(movement, Mapping)
            print(f"  {movement.get('path')}: {movement.get('before')!r} -> {movement.get('after')!r}", file=sys.stderr)
        print("  pass --reason '<why this number moved>' to record it.", file=sys.stderr)
        return
    if action == "unchanged":
        print(f"{result.get('name')}: unchanged; {result.get('path')} already holds this measurement.")
        return
    moved = result.get("movements")
    count = len(moved) if isinstance(moved, Sequence) else 0
    print(f"{result.get('action')} {result.get('path')} ({count} moved value(s))")
    print(f"  reason: {result.get('reason')}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    try:
        if args.record:
            code, result = _record(args, root)
        else:
            code, result = _list(root)
    except MeasurementBaselineError as exc:
        if args.json:
            print(json.dumps({"action": "error", "error": str(exc)}, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        _render(result)
    return code


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())

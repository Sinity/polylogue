"""Read-only diagnostic binding of the daemon's published compute owner.

Loaded explicitly by the benchmark launcher and hashed in its receipt. This
observer never constructs, publishes or changes a product compute adapter.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any


def policy_state() -> dict[str, Any] | None:
    module = sys.modules.get("polylogue.core.compute")
    adapter = getattr(module, "_SHARED_COMPUTE_ADAPTER", None)
    if adapter is None:
        return None
    snapshot = adapter.snapshot().to_dict()
    # Counters change with work; only actual admission capacities belong to
    # the policy identity. These fields come from the live owner's snapshot.
    fields = ("capacity_units", "capacity_bytes", "capacity_slots", "foreground_allowance_capacity_bytes")
    class_fields = (
        "admission_class",
        "reservation_group",
        "reserved_units",
        "ceiling_units",
        "reserved_slots",
        "ceiling_slots",
    )
    checker = getattr(sys, "_is_gil_enabled", None)
    return {
        **{key: snapshot[key] for key in fields},
        "classes": [{key: row[key] for key in class_fields} for row in snapshot["classes"]],
        "gil_enabled_after_product_imports": bool(checker()) if checker else True,
        "adapter_type": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
    }


def product_binding() -> dict[str, Any]:
    """Bind actual imported modules, even if a launcher discarded PYTHONPATH."""
    candidate = Path(os.environ["POLYLOGUE_BENCH_CANDIDATE"]).resolve()
    modules = {}
    for name in ("polylogue", "polylogue.core.compute", "polylogue.daemon.commands"):
        module = sys.modules.get(name)
        actual = getattr(module, "__file__", None)
        relative = "polylogue/__init__.py" if name == "polylogue" else name.replace(".", "/") + ".py"
        expected = (candidate / relative).resolve()
        if actual is None:
            modules[name] = {"bound": False, "path": None, "sha256": None}
            continue
        path = Path(actual).resolve()
        modules[name] = {
            "bound": path == expected,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "outcome": "bound" if all(row["bound"] for row in modules.values()) else "mismatch",
        "candidate": str(candidate),
        "modules": modules,
    }


def start_from_environment() -> None:
    path = Path(os.environ["POLYLOGUE_BENCH_POLICY"])
    stop = threading.Event()
    states: list[dict[str, Any]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def capture() -> bool:
        try:
            state = policy_state()
            if state is not None:
                with lock:
                    if state not in states:
                        states.append(state)
                        return True
        except Exception as exc:
            with lock:
                name = type(exc).__name__
                if name not in errors:
                    errors.append(name)
                    return True
        return False

    def write() -> None:
        with lock:
            document = {
                "outcome": "bound" if states and not errors else "unavailable",
                "states": list(states),
                "errors": list(errors),
                "product_binding": product_binding(),
                "source": "live published compute adapter snapshot",
                "coverage": "periodic 50ms and normal process exit; transient policies may be missed",
            }
        staging = path.with_suffix(".tmp")
        staging.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
        staging.replace(path)

    def observe() -> None:
        while not stop.is_set():
            if capture() or not path.exists():
                write()
            stop.wait(0.05)

    thread = threading.Thread(target=observe, name="bench-policy-observer", daemon=True)
    thread.start()

    def finish() -> None:
        stop.set()
        thread.join()
        capture()
        write()

    atexit.register(finish)

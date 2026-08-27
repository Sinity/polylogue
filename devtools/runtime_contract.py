"""Managed runtime identity and extension compatibility verification."""

from __future__ import annotations

import json
import sys

from polylogue.runtime import RuntimeContractError, runtime_report


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    report = runtime_report()
    if "--json" in args:
        json.dump(report, sys.stdout, sort_keys=True, indent=2)
        sys.stdout.write("\n")
        return 0 if report["pass"] else 1
    else:
        runtime = report["runtime"]
        assert isinstance(runtime, dict)
        print(
            f"runtime: {runtime['implementation']} {runtime['version']} "
            f"gil_enabled={runtime['gil_enabled']} abi_flags={runtime['abi_flags']!r}"
        )
        extensions = report["extensions"]
        assert isinstance(extensions, list)
        for probe in extensions:
            assert isinstance(probe, dict)
            print(f"extension: {probe['name']} importable={probe['importable']} safe={probe['safe']}")
    if not report["pass"]:
        try:
            from polylogue.runtime import require_free_threaded_runtime

            require_free_threaded_runtime(consumer="devtools runtime verification")
        except RuntimeContractError as exc:
            print(f"runtime: FAIL: {exc}", file=sys.stderr)
        else:
            print("runtime: FAIL: one or more required extensions did not import", file=sys.stderr)
        return 1
    print("runtime: PASS: CPython 3.14 free-threading and required extensions verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

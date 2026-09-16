"""Run the repository's declared typed WebUI verification route."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from devtools import repo_root

#: What ``devtools gate webui`` claims the run covers, mapped to the
#: ``webui/package.json`` ``check`` steps that make each claim true. The gate
#: description restates these categories in prose; :func:`check_script_steps`
#: reads what the script actually composes, so dropping a step from the script
#: (or adding one no category owns) contradicts the claim instead of quietly
#: narrowing the gate.
CHECK_SCRIPT_CLAIMS: dict[str, tuple[str, ...]] = {
    "generation": ("generate:check", "check:client"),
    "contract": ("lint", "typecheck", "test:client-contracts"),
    "unit": ("test",),
    "build": ("build", "build:design-system"),
}

_NPM_RUN = re.compile(r"^npm\s+run\s+(?P<step>\S+)$")


def check_script_steps(root: Path | None = None) -> tuple[str, ...]:
    """The ``npm run`` steps ``webui/package.json``'s ``check`` script composes."""
    package = (root or repo_root()) / "webui" / "package.json"
    script = json.loads(package.read_text(encoding="utf-8"))["scripts"]["check"]
    steps: list[str] = []
    for fragment in script.split("&&"):
        match = _NPM_RUN.match(fragment.strip())
        if match is not None:
            steps.append(match.group("step"))
    return tuple(steps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable result envelope.")
    args = parser.parse_args(argv)

    root = repo_root()
    command = ["npm", "run", "check"]
    try:
        result = subprocess.run(command, cwd=root / "webui", check=False, text=True, capture_output=True)
    except OSError as exc:
        payload: dict[str, Any] = {
            "command": "devtools gate webui",
            "argv": command,
            "status": "blocked-env",
            "returncode": None,
            "output": str(exc),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"verify webui: blocked environment: {exc}")
        return 2

    output = (result.stdout + result.stderr).strip()
    payload = {
        "command": "devtools gate webui",
        "argv": command,
        "status": "green" if result.returncode == 0 else "red",
        "returncode": result.returncode,
        "output": output,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(output)
        print(f"verify webui: {'green' if result.returncode == 0 else 'red'}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

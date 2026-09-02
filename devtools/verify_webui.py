"""Run the repository's declared typed WebUI verification route."""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

from devtools import repo_root


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

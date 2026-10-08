"""Own the synthetic demo's resident daemon through physical shutdown."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from builtins import BaseExceptionGroup
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from watchfiles import watch


def _await_listener(process: subprocess.Popen[bytes], path: Path) -> None:
    # The atomic listener record proves this child's bind, not archive readiness.
    # Watch ticks only observe child exit/cancellation; they impose no deadline.
    with closing(watch(path.parent, yield_on_timeout=True, rust_timeout=250)) as changes:
        while True:
            status = process.poll()
            if status is not None:
                raise RuntimeError(f"demo resident daemon exited before binding (status {status})")
            if path.is_file():
                payload = json.loads(path.read_bytes())
                if not isinstance(payload, dict) or payload.get("pid") != process.pid:
                    raise RuntimeError("demo resident listener belongs to a different process")
                listeners = payload.get("listeners")
                api = listeners.get("api") if isinstance(listeners, dict) else None
                if (
                    not isinstance(listeners, dict)
                    or not isinstance(api, dict)
                    or api.get("host") != "127.0.0.1"
                    or type(api.get("port")) is not int
                    or not 0 < api["port"] <= 65535
                    or listeners.get("browser_capture") is not None
                ):
                    raise RuntimeError("demo resident listener record is invalid")
                return
            next(changes)


@contextmanager
def demo_resident(archive_root: Path) -> Iterator[dict[str, str]]:
    """Run only the demo's daemon and lend its isolated configuration to queries.

    The seed's one-shot writer must have settled before entering this scope.
    SIGINT follows the daemon's original shutdown path; wait retains ownership
    until that child physically exits, including when the caller is cancelled.
    """
    scratch = Path(tempfile.mkdtemp(prefix="polylogue-demo-resident-"))
    process: subprocess.Popen[bytes] | None = None
    primary: BaseException | None = None
    try:
        config = scratch / "polylogue.toml"
        config.write_text("", encoding="utf-8")
        env = {key: value for key, value in os.environ.items() if not key.startswith("POLYLOGUE_")}
        env.update(
            POLYLOGUE_ARCHIVE_ROOT=str(archive_root),
            POLYLOGUE_CONFIG=str(config),
            POLYLOGUE_SITE_CONFIG=str(config),
            POLYLOGUE_FORCE_PLAIN="1",
            POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS="0",
        )
        listener = scratch / "listeners.json"
        with (scratch / "daemon.log").open("wb") as output:
            process = subprocess.Popen(
                (
                    sys.executable,
                    "-c",
                    f"import sys; sys.path[:] = {sys.path!r}; "
                    "from polylogue.daemon.commands import main; main(prog_name='polylogued')",
                    "run",
                    "--no-watch",
                    "--no-source-catchup",
                    "--no-browser-capture",
                    "--api-host",
                    "127.0.0.1",
                    "--api-port",
                    "0",
                    "--listener-info-path",
                    str(listener),
                ),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            _await_listener(process, listener)
            yield env
    except BaseException as exc:
        primary = exc
        raise
    finally:
        errors: list[BaseException] = []
        if process is not None:
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError as exc:
                    if process.poll() is None:
                        errors.append(exc)
                except BaseException as exc:
                    errors.append(exc)
            # A second cancellation interrupts the wait, not child ownership.
            # Retain it and continue until the same process physically exits.
            status: int | None = None
            while status is None:
                try:
                    status = process.wait()
                except (KeyboardInterrupt, asyncio.CancelledError, InterruptedError) as exc:
                    errors.append(exc)
            if status != 0:
                errors.append(RuntimeError(f"demo resident daemon failed during shutdown (status {status})"))
        # Do not discard a live child's evidence or claim physical settlement.
        if process is None or process.poll() is not None:
            try:
                shutil.rmtree(scratch)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise BaseExceptionGroup(
                "demo resident settlement failed", ([primary] if primary else []) + errors
            ) from None

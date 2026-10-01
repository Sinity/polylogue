#!/usr/bin/env python3
import json
import os
import signal
import socket
import sys

if "run" not in sys.argv:
    raise SystemExit(0)


def record(event):
    with open(os.environ["SMOKE_RECEIPT"], "a") as output:
        print(json.dumps({"event": event}), file=output)


if os.environ.get("SMOKE_FAILURE") == "startup":
    raise SystemExit(7)


def stop(_signum, _frame):
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop)
try:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(os.environ["SMOKE_SOCKET"])
        server.listen()
        record("daemon_started")
        while True:
            connection, _ = server.accept()
            connection.close()
finally:
    record("daemon_stopped")

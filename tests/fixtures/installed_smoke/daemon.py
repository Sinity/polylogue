#!/usr/bin/env python3
import json
import os
import signal
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

if "run" not in sys.argv:
    raise SystemExit(0)


def record(event):
    with open(os.environ["SMOKE_RECEIPT"], "a") as output:
        print(json.dumps({"event": event}), file=output)


if os.environ.get("SMOKE_FAILURE") == "startup":
    raise SystemExit(7)


def stop(_signum, _frame):
    raise SystemExit(0)


class ReadyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        record("readiness_ready")
        body = b'{"status":"ready"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


api = HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--api-port") + 1])), ReadyHandler)
threading.Thread(target=api.serve_forever, daemon=True).start()
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

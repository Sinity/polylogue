"""Python access through Polylogue's declared CLI and its daemon."""

import json
import os
import socket
import subprocess
from pathlib import Path

from .models import Answer, _copy

__all__ = ["Client", "Refused", "read_answer"]

# The exit status the CLI gives each outcome that is not a refusal.
_EXIT_STATUS = {"ok": 0, "empty": 1, "degraded": 5}


def _failure(answer):
    if not isinstance(answer, dict):
        return None
    if answer.get("outcome") == "error":
        return answer["error"]
    value = answer.get("value", {})
    if isinstance(value, dict) and value.get("kind") == "steps":
        for step in value["reply"]["steps"]:
            if step["outcome"]["status"] == "failed":
                return step["outcome"]["error"]
    return None


class Refused(RuntimeError):
    """The complete refusal, including a corrected query when supplied."""

    def __init__(self, answer, returncode):
        self.answer = Answer(answer)
        self.returncode = returncode
        error = _failure(answer)
        self.kind = error["kind"]
        super().__init__(error.get("message", str(answer)))


def _refusal(answer, returncode):
    """The error to raise for a failure answer, checked against its exit status."""
    if returncode == 0:
        return RuntimeError("Polylogue returned an error with a successful exit status")
    return Refused(answer, returncode)


def read_answer(response):
    """The final answer of one socket reply.

    A reply is server-sent-event frames: zero or more events, then one
    `answer` frame. A reply that ends before its answer means the daemon
    went away.
    """
    event, data = None, []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\n").rstrip("\r")
        if line == "":
            if event == "answer":
                return json.loads("\n".join(data))
            event, data = None, []
        elif line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:"):].removeprefix(" "))
    raise RuntimeError("the daemon went away before it answered")


class Client:
    """Use the installed CLI; an explicit archive is passed only to its process.

    Writes are never retried here. A caller that retries a write must keep its
    request ID and pass it again through the declared command's arguments.
    """

    def __init__(self, archive=None, *, executable="polylogue"):
        self.executable = os.fspath(executable)
        self.archive = None if archive is None else str(Path(archive).expanduser().absolute())

    def _arguments(self, *arguments):
        command = [self.executable]
        if self.archive is not None:
            command.extend(["--archive", self.archive])
        return [*command, *arguments]

    def call(self, *arguments):
        """One declared command, with the same arguments and answer as CLI JSON."""
        result = subprocess.run(
            self._arguments("--json", *arguments),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        try:
            answer = json.loads(result.stdout)
        except (ValueError, UnicodeDecodeError) as error:
            said = result.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(
                f"Polylogue returned no JSON answer (exit {result.returncode})"
                + (f": {said}" if said else "")
            ) from error
        outcome = answer.get("outcome") if isinstance(answer, dict) else None
        if _failure(answer) is not None:
            raise _refusal(answer, result.returncode)
        if outcome not in _EXIT_STATUS:
            raise RuntimeError("Polylogue returned an invalid shared answer")
        if result.returncode != _EXIT_STATUS[outcome]:
            raise RuntimeError(f"Polylogue's {outcome} answer disagrees with exit {result.returncode}")
        return Answer(answer)

    def stream(self, *arguments):
        """Yield the finite line events, as plain dicts, through the final answer.

        Each read completes independently. The final answer ends the line;
        failures retain their earlier step reports and undo lines.
        Closing the iterator terminates and reaps the child process.
        """
        with subprocess.Popen(self._arguments("--json", *arguments), stdout=subprocess.PIPE) as process:
            final = None
            failed = None
            try:
                for line in process.stdout:
                    event = json.loads(line)
                    if "event" not in event:
                        if _failure(event):
                            raise _refusal(event, process.wait())
                        raise RuntimeError("Polylogue returned no line event")
                    if final is not None:
                        raise RuntimeError("Polylogue emitted an event after its final answer")
                    if event["event"] in {"steps", "answer"}:
                        if _failure(event["data"]) and failed is None:
                            failed = event["data"]
                    if event["event"] == "answer":
                        final = event["data"]
                    yield _copy(event)
                code = process.wait()
                if failed is not None:
                    raise _refusal(failed, code)
                if final is None:
                    raise RuntimeError("Polylogue's line stopped before its final answer")
                if final.get("outcome") not in _EXIT_STATUS:
                    raise RuntimeError("Polylogue returned an invalid shared answer")
                if code != _EXIT_STATUS[final["outcome"]]:
                    raise RuntimeError(f"Polylogue's line answer disagrees with exit {code}")
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait()

    def status(self):
        return self.call("status")

    def query(self, text, *, cursor=None, page_size=None):
        arguments = ["--", text]
        if page_size is not None:
            arguments.extend(["--page-size", str(page_size)])
        if cursor is not None:
            arguments.extend(["--cursor", cursor])
        return self.call(*arguments)

    def read(self, identity, *, view="transcript", page_tokens=None):
        arguments = ["--", identity, "read", "--view", view]
        if page_tokens is not None:
            arguments.extend(["--budget", str(page_tokens)])
        return self.call(*arguments)

    def follow_read(self, request):
        """Execute one generated rows request on the daemon socket.

        The structured request stays unchanged, including omitted and null
        deadlines, budgets and step cursors. No displayed command is parsed.
        """
        request = _copy(request)
        if request["operation"] != "rows":
            raise ValueError(f"unsupported read follow-up operation: {request['operation']}")
        unknown = request.keys() - {
            "operation", "arguments", "request_id", "deadline", "budget_tokens",
            "cursor", "fresh", "format",
        }
        if unknown:
            raise ValueError(f"unknown read follow-up fields: {sorted(unknown)}")
        archive = self.archive
        if archive is None:
            archive = self.status().json()["value"]["archive"]
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(os.path.join(archive, "daemon.sock"))
            connection.sendall(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
            with connection.makefile("rb") as response:
                answer = read_answer(response)
        if _failure(answer) is not None:
            raise Refused(answer, None)
        if answer.get("outcome") not in _EXIT_STATUS:
            raise RuntimeError("Polylogue returned an invalid shared answer")
        return Answer(answer)

    def feed(self, after=None):
        """Yield a finite feed. Save only the position on its completion line.

        The CLI validates framing and counts before forwarding completion.
        Closing the iterator early terminates and reaps the child process.
        """
        arguments = self._arguments("dev", "feed")
        if after is not None:
            arguments.extend(["--after", after])
        with subprocess.Popen(arguments, stdout=subprocess.PIPE) as process:
            try:
                for line in process.stdout:
                    answer = json.loads(line)
                    if answer.get("outcome") == "error":
                        raise _refusal(answer, process.wait())
                    yield Answer(answer)
                if process.wait():
                    raise RuntimeError(f"Polylogue's change feed failed (exit {process.returncode})")
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait()

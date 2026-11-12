"""Python access through Polylogue's declared CLI and its daemon."""

import json
import os
import subprocess
from pathlib import Path

from .models import Answer, _copy

__all__ = ["Client", "Refused"]


class Refused(RuntimeError):
    """The complete refusal, including a corrected query when supplied."""

    def __init__(self, answer, returncode):
        self.answer = Answer(answer)
        self.returncode = returncode
        error = answer["error"]
        self.kind = error.get("kind", "client")
        super().__init__(error.get("message", str(answer)))


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
        return [*command, "--json", *arguments]

    def call(self, *arguments):
        """One declared command, with the same arguments and answer as CLI JSON."""
        result = subprocess.run(
            self._arguments(*arguments),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        try:
            answer = json.loads(result.stdout)
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError(f"Polylogue returned no JSON answer (exit {result.returncode})") from error
        outcome = answer.get("outcome") if isinstance(answer, dict) else None
        if outcome == "error":
            if result.returncode == 0:
                raise RuntimeError("Polylogue returned an error with a successful exit status")
            raise Refused(answer, result.returncode)
        if outcome not in {"ok", "empty", "degraded"}:
            raise RuntimeError("Polylogue returned an invalid shared answer")
        expected = {"ok": 0, "empty": 1, "degraded": 5}[outcome]
        if result.returncode != expected:
            raise RuntimeError(f"Polylogue's {outcome} answer disagrees with exit {result.returncode}")
        return Answer(answer)

    def status(self):
        return self.call("status")

    def query(self, text, *, cursor=None, page_size=None):
        arguments = []
        if cursor is not None:
            arguments.extend(["--cursor", cursor])
        if page_size is not None:
            arguments.extend(["--limit", str(page_size)])
        return self.call(*arguments, "--", text)

    def read(self, identity, *, view="transcript", page_tokens=None):
        arguments = ["read", identity, "--view", view]
        if page_tokens is not None:
            arguments.extend(["--page-tokens", str(page_tokens)])
        return self.call(*arguments)

    def follow_read(self, request):
        """Execute a generated read/attachment action, including a previous page.

        The request is passed as structured arguments; displayed commands are
        never parsed. Omitted and explicit null deadlines/budgets stay distinct.
        """
        request = _copy(request)
        operation = request["operation"]
        if operation not in {"read", "attachment"}:
            raise ValueError(f"unsupported read follow-up operation: {operation}")
        unknown = request.keys() - {
            "operation", "arguments", "request_id", "deadline", "budget_tokens",
            "cursor", "fresh", "format",
        }
        if unknown:
            raise ValueError(f"unknown read follow-up fields: {sorted(unknown)}")
        # These shared-envelope options are not offered by this product. They
        # must not be discarded or reinterpreted as CLI output switches.
        if request.get("fresh", False) or request.get("format", "text") != "text":
            raise ValueError("unsupported read follow-up freshness or envelope format")
        arguments = [operation, "--request", json.dumps(request["arguments"], ensure_ascii=False),
                     "--request-id", request["request_id"]]
        for field, flag in [("deadline", "--deadline"), ("budget_tokens", "--budget")]:
            if field in request:
                value = request[field]
                arguments.extend([flag, "none" if value is None else str(value)])
        if request.get("cursor") is not None:
            arguments.extend(["--cursor", request["cursor"]])
        return self.call(*arguments)

    def changes(self, since=None):
        """Yield a finite feed. Save only the position on its completion line.

        The CLI validates framing and counts before forwarding completion.
        Closing the iterator early terminates and reaps the child process.
        """
        arguments = self._arguments("changes")
        if since is not None:
            arguments.extend(["--since", since])
        with subprocess.Popen(arguments, stdout=subprocess.PIPE) as process:
            try:
                for line in process.stdout:
                    answer = json.loads(line)
                    if answer.get("outcome") == "error":
                        raise Refused(answer, process.wait())
                    yield Answer(answer)
                if process.wait():
                    raise RuntimeError(f"Polylogue's change feed failed (exit {process.returncode})")
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait()

"""Python access through Polylogue's declared CLI and its daemon."""

import json
import os
import subprocess
from pathlib import Path

from .models import Answer

__all__ = ["Client", "Refused"]


class Refused(RuntimeError):
    """The complete refusal, including a corrected query when supplied."""

    def __init__(self, answer, returncode):
        self.answer = Answer(answer)
        self.returncode = returncode
        error = answer.get("error", answer if answer.get("result") == "error" else {})
        self.kind = error.get("kind", "client")
        super().__init__(error.get("message", str(answer)))


class Client:
    """Use the installed CLI; an explicit archive is passed only to its process.

    Writes are never retried here. A caller that retries a write must keep its
    request ID and pass it again through the declared command's arguments.
    """

    def __init__(self, archive=None, *, executable="polylogue"):
        self.executable = os.fspath(executable)
        self.archive = None if archive is None else str(Path(archive).expanduser().resolve())

    def _environment(self):
        env = os.environ.copy()
        if self.archive is not None:
            env["POLYLOGUE_ARCHIVE"] = self.archive
        return env

    def call(self, *arguments):
        """One declared command, with the same arguments and answer as CLI JSON."""
        result = subprocess.run(
            [self.executable, "--json", *arguments], env=self._environment(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        try:
            answer = json.loads(result.stdout)
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError(f"Polylogue returned no JSON answer (exit {result.returncode})") from error
        if result.returncode:
            raise Refused(answer, result.returncode)
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

    def read(self, identity, *, view="full", budget=None, cursor=None):
        arguments = ["read", identity, "--view", view]
        if budget is not None:
            arguments.extend(["--budget", str(budget)])
        if cursor is not None:
            arguments.extend(["--cursor", cursor])
        return self.call(*arguments)

    def changes(self, since=None):
        """Yield a finite feed. Save only the position on its completion line.

        The CLI validates framing and counts before forwarding completion.
        Closing the iterator early terminates and reaps the child process.
        """
        arguments = [self.executable, "--json", "changes"]
        if since is not None:
            arguments.extend(["--since", since])
        with subprocess.Popen(arguments, env=self._environment(), stdout=subprocess.PIPE) as process:
            try:
                for line in process.stdout:
                    answer = json.loads(line)
                    if "error" in answer or answer.get("result") == "error":
                        raise Refused(answer, process.wait())
                    yield Answer(answer)
                if process.wait():
                    raise RuntimeError(f"Polylogue's change feed failed (exit {process.returncode})")
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait()

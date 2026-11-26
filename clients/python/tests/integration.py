"""Public client contracts against a real synthetic daemon, started by e2e."""

import base64
import io
import json
import sys
import socket
import threading
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from polylogue.models import Answer
from polylogue.client import Client, Refused, read_answer

ARCHIVE, EXECUTABLE = sys.argv[1:3]
del sys.argv[1:3]

PYTHON_SESSION = "session:claude-code:python"
PYTHON_TEXT = "a synthetic Python client session"


def text_items(items):
    return [item["kind"]["text"] for item in items if item["kind"]["kind"] == "text"]


def read_page(answer):
    return answer["value"]["reply"]["steps"][0]["outcome"]["reply"]["reply"]


def attachment_bytes(answer):
    return read_page(answer)["content_base64"]


class ClientTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.capture = Path(directory.name) / "answer.json"
        self.arguments = Path(directory.name) / "arguments.json"
        proxy = Path(directory.name) / "polylogue"
        proxy.write_text(
            "#!/usr/bin/env python3\n"
            "import json, subprocess, sys\nfrom pathlib import Path\n"
            f"Path({str(self.arguments)!r}).write_text(json.dumps(sys.argv[1:]))\n"
            f"result = subprocess.run([{EXECUTABLE!r}, *sys.argv[1:]], capture_output=True)\n"
            f"Path({str(self.capture)!r}).write_bytes(result.stdout)\n"
            "sys.stdout.buffer.write(result.stdout)\n"
            "sys.stderr.buffer.write(result.stderr)\n"
            "raise SystemExit(result.returncode)\n"
        )
        proxy.chmod(0o700)
        self.archive_alias = Path(directory.name) / "a"
        self.archive_alias.symlink_to(Path(ARCHIVE), target_is_directory=True)
        self.client = Client(self.archive_alias, executable=proxy)
        self.direct = Client(self.archive_alias, executable=EXECUTABLE)
        self.assertEqual(self.client.archive, str(self.archive_alias.absolute()))

    def checked(self, call):
        answer = call().json()
        # Compare the complete output of this very execution, including its
        # own captured clock, saved Run identity and observed archive position.
        self.assertEqual(answer, json.loads(self.capture.read_bytes()))
        self.assertEqual(json.loads(self.arguments.read_text())[:2], ["--archive", str(self.archive_alias.absolute())])
        return answer

    def checked_follow(self, request):
        # Forward this execution to the real daemon, recording both sides of
        # the socket boundary independently of Client's JSON handling.
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(Path(directory) / "daemon.sock"))
                listener.listen(1)
                observed = {}
                def forward():
                    with listener.accept()[0] as connection:
                        with connection.makefile("rb") as incoming:
                            raw = incoming.readline()
                        observed["request"] = json.loads(raw)
                        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
                            upstream.connect(str(self.archive_alias / "daemon.sock"))
                            upstream.sendall(raw)
                            upstream.shutdown(socket.SHUT_WR)
                            with upstream.makefile("rb") as response:
                                raw_answer = response.read()
                        observed["answer"] = read_answer(io.BytesIO(raw_answer))
                        connection.sendall(raw_answer)
                worker = threading.Thread(target=forward, daemon=True)
                worker.start()
                client = Client(directory, executable=EXECUTABLE)
                refusal = None
                try:
                    answer = client.follow_read(request).json()
                except Refused as error:
                    refusal, answer = error, error.answer.json()
                worker.join()
                self.assertEqual(observed["request"], request)
                self.assertEqual(answer, observed["answer"])
                if refusal is not None:
                    raise refusal
                return answer

    def test_declared_archive_alias_is_preserved_for_calls_and_feeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / ("long-archive-target-" * 8)
            target.mkdir()
            alias = root / "short"
            alias.symlink_to(target, target_is_directory=True)
            client = Client(alias, executable=EXECUTABLE)
            declared = str(alias.absolute())
            self.assertNotEqual(declared, str(alias.resolve()))
            self.assertEqual(client.archive, declared)
            self.assertEqual(client._arguments("status"), [EXECUTABLE, "--archive", declared, "status"])
            self.assertEqual(
                client._arguments("dev", "feed"),
                [EXECUTABLE, "--archive", declared, "dev", "feed"],
            )

    def test_ordered_line_stream_reads_every_step_and_finishes_once(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            events = list(self.direct.stream(
                "--request-id", "python-ordered", "--", PYTHON_SESSION,
                "note", "Python first line note",
                "read", "--to", str(first), "--format", "json",
                "tag", "python-later",
                "read", "--to", str(second), "--format", "json"))
            for path in [first, second]:
                document = json.loads(path.read_bytes())
                self.assertEqual(text_items(document["items"]), [PYTHON_TEXT])

        self.assertEqual([event["event"] for event in events].count("complete"),2)
        self.assertEqual(events[-1]["event"],"answer")
        reports = [report for event in events if event["event"] in {"steps","answer"}
            for report in event["data"]["value"]["reply"]["steps"]]
        self.assertEqual([report["verb"] for report in reports],["note","read","tag","read"])
        self.assertEqual([report["request_id"] for report in reports],
            ["python-ordered-1","python-ordered-2","python-ordered-3","python-ordered-4"])

    def test_query_and_read_match_cli_json(self):
        query = self.checked(lambda: self.client.query("sessions"))
        self.assertEqual(query["outcome"], "ok")
        self.assertEqual(
            sorted(row["id"] for row in query["value"]["reply"]["rows"]),
            ["claude-code:python", "claude-code:python-image", "claude-code:python-pages"],
        )
        read = self.checked(lambda: self.client.read(PYTHON_SESSION))
        self.assertEqual(read["outcome"], "ok")
        self.assertTrue(read_page(read)["document"]["items"])
        self.assertIn("read_at", read_page(read)["snapshot"])
        self.assertEqual(text_items(read_page(read)["document"]["items"]), [PYTHON_TEXT])

    def test_generated_read_calls_continue_without_losing_their_envelope(self):
        target = "session:claude-code:python-pages"
        whole = read_page(self.direct.call(target, "read", "--budget", "100000").json())["document"]["items"]
        answer = self.checked(lambda: self.client.read(target, page_tokens=600))
        snapshot = read_page(answer)["snapshot"]
        complete, partial, seen_calls = [], {}, set()
        count = 0
        while True:
            page = read_page(answer)
            # Every executed page writes its own run, which records when it
            # ran and the archive position its run row advanced; everything
            # else continues the first page's question and index read
            # position (docs/query.md, Runs and Paging).
            ran = dict(page["snapshot"])
            first = dict(snapshot)
            self.assertIsInstance(ran.pop("ran_at"), str)
            first.pop("ran_at")
            ran["read_at"], first["read_at"] = dict(ran["read_at"]), dict(first["read_at"])
            self.assertGreaterEqual(ran["read_at"].pop("archive"), first["read_at"].pop("archive"))
            self.assertEqual(ran, first)
            for read_item in page["document"]["items"]:
                item = read_item["kind"]
                if item["kind"] != "fragment":
                    self.assertNotIn(read_item, complete)
                    complete.append(read_item)
                    continue
                position = item["position"]
                key = json.dumps([position["selection"], position["item"], position["item_hash"]], sort_keys=True)
                text, length = partial.get(key, ("", 0))
                self.assertEqual(position["character"], length)
                self.assertEqual(item["until"] - length, len(item["text"]))
                text += item["text"]
                self.assertLessEqual(item["until"], item["total"])
                if item["until"] == item["total"]:
                    decoded = json.loads(text)
                    self.assertNotIn(decoded, complete)
                    complete.append(decoded)
                    partial.pop(key, None)
                else:
                    partial[key] = (text, item["until"])
            ready = page["document"]["footer"].get("next")
            if ready is None:
                break
            cursor = ready["arguments"]["steps"][0]["cursor"]
            self.assertNotIn(cursor, seen_calls)
            seen_calls.add(cursor)
            answer = self.checked_follow(ready)
            self.assertEqual(ready["operation"], "rows")
            self.assertNotIn("deadline", ready)
            self.assertNotIn("budget_tokens", ready)
            count += 1
            self.assertLess(count, 100)
        self.assertGreater(count, 0)
        self.assertFalse(partial)
        self.assertEqual(complete, whole)
        self.assertEqual(text_items(complete), ["Python λ🙂 continuation. " * 300])

    def test_read_follow_ups_preserve_nulls_and_previous_pages(self):
        first = read_page(self.direct.call("session:claude-code:python-pages", "read", "--budget", "600", "--deadline", "none").json())
        next_request = first["document"]["footer"]["next"]
        self.assertIsNone(next_request["deadline"])
        second = read_page(self.checked_follow(next_request))
        previous = second["document"]["footer"]["previous"]
        restored = read_page(self.checked_follow(previous))
        self.assertEqual(restored["document"]["items"], first["document"]["items"])
        # The request budget belongs to the listing; the read step's own
        # budget and cursor remain unchanged.
        for budget in [None, 37]:
            request = {**next_request, "budget_tokens": budget}
            followed = read_page(self.checked_follow(request))
            self.assertEqual(followed["document"]["items"], second["document"]["items"])
            self.assertEqual(request["arguments"]["steps"][0]["budget"], 600)

    def test_rows_follow_up_preserves_the_listing_request(self):
        request = {"operation": "rows", "arguments": {"selection": "sessions", "page_size": 1},
                   "request_id": "python-rows-follow", "deadline": None}
        answer = self.checked_follow(request)
        self.assertEqual(answer["value"]["kind"], "listing")
        self.assertEqual(len(answer["value"]["reply"]["rows"]), 1)

    def test_generated_attachment_action_and_unsupported_operations(self):
        page = read_page(self.direct.read("session:claude-code:python-image").json())
        attachment = next(item["kind"] for item in page["document"]["items"]
                          if item["kind"]["kind"] == "attachment")
        request = attachment["open"]
        self.assertEqual(request["operation"], "rows")
        self.assertTrue(request["arguments"]["selection"].startswith("attachment:"))
        answer = self.checked_follow(request)
        self.assertEqual(base64.b64decode(attachment_bytes(answer)), b"synthetic Python attachment")
        for operation in ["query", "set"]:
            with self.assertRaises(ValueError):
                self.client.follow_read({**request, "operation": operation})
        with self.assertRaises(Refused) as raised:
            self.checked_follow({**request, "fresh": True})
        self.assertEqual(raised.exception.kind, "bad_request")
        self.assertIsNone(raised.exception.returncode)
        formatted = self.checked_follow({**request, "format": "json"})
        self.assertEqual(base64.b64decode(attachment_bytes(formatted)), b"synthetic Python attachment")

    def test_refusals_keep_the_whole_answer(self):
        with self.assertRaises(Refused) as raised:
            self.client.read("session:claude-code:missing")
        self.assertEqual(raised.exception.kind, "not_found")
        self.assertEqual(raised.exception.answer.json(), json.loads(self.capture.read_bytes()))

    def test_error_answer_with_a_successful_exit_status_is_a_protocol_error(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "polylogue"
            fake.write_text(
                "#!/bin/sh\n"
                """echo '{"outcome":"error","error":{"kind":"bad_request","message":"no"}}'\n"""
            )
            fake.chmod(0o700)
            client = Client(directory, executable=fake)
            with self.assertRaises(RuntimeError) as raised:
                client.status()
            self.assertNotIsInstance(raised.exception, Refused)
            with self.assertRaises(RuntimeError) as raised:
                list(client.stream("status"))
            self.assertNotIsInstance(raised.exception, Refused)
            with self.assertRaises(RuntimeError) as raised:
                list(client.feed())
            self.assertNotIsInstance(raised.exception, Refused)

    def test_empty_answers_keep_their_payload_and_exit_contract(self):
        answer = self.checked(lambda: self.client.query('sessions where repo:"python-missing"'))
        self.assertEqual(answer["outcome"], "empty")
        self.assertEqual(answer["value"]["reply"]["rows"], [])

    def test_answers_are_frozen_without_changing_json_values(self):
        value = {"unknown": None, "zero": 0, "id": "a:b:c", "large": 2**62, "rows": [{"x": "ż"}]}
        answer = Answer(value)
        value["rows"][0]["x"] = "changed"
        self.assertEqual(answer.data["rows"][0]["x"], "ż")
        with self.assertRaises(TypeError):
            answer.data["rows"][0]["x"] = "changed"
        with self.assertRaises(FrozenInstanceError):
            answer.data = None
        self.assertEqual(hash(answer), hash(Answer(answer.json())))
        self.assertEqual({answer: 1}[Answer(answer.json())], 1)
        copy = answer.json()
        self.assertIsNone(copy["unknown"])
        self.assertEqual(copy["large"], 2**62)
        copy["rows"].clear()
        self.assertEqual(len(answer.data["rows"]), 1)

    def test_resent_write_is_idempotent(self):
        arguments = ("set", "signal.interrupts_per_hour", "9", "--request-id", "python-write")
        first = self.checked(lambda: self.client.call(*arguments))["value"]
        again = self.checked(lambda: self.client.call(*arguments))["value"]
        self.assertEqual(first["id"], again["id"])
        self.assertFalse(first["repeat"])
        self.assertTrue(again["repeat"])
        current = self.checked(lambda: self.client.call("set", "signal.interrupts_per_hour"))["value"]
        self.assertEqual(current["value"], {"type": "interrupt_budget", "value": 9})
        self.assertEqual(first["id"], "python-write")
        self.assertEqual(current["authored"]["reference"], "annotation:python-write@python-write")

    def test_change_feed_can_continue_from_its_completion(self):
        first = [line.json() for line in self.direct.feed()]
        self.assertEqual(first[0]["type"], "begin")
        self.assertEqual(first[-1]["type"], "complete")
        self.assertTrue(first[-1]["replacement"])
        following = [line.json() for line in self.direct.feed(first[-1]["position"])]
        self.assertEqual(following[0]["position"], first[-1]["position"])
        self.assertEqual(following[-1]["type"], "complete")
        self.assertFalse(following[-1]["replacement"])
        generation = first[-1]["position"].split(":")[0]
        with self.assertRaises(Refused) as raised:
            list(self.direct.feed(f"{generation}:9223372036854775807:9223372036854775807"))
        self.assertEqual(raised.exception.kind, "bad_request")


if __name__ == "__main__":
    unittest.main()

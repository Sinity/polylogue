"""Public client contracts against a real synthetic daemon, started by e2e."""

import json
import shlex
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from polylogue.models import Answer
from polylogue.client import Client, Refused

ARCHIVE, EXECUTABLE = sys.argv[1:3]
del sys.argv[1:3]


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
        # own captured clock, durable Run identity and observed archive position.
        self.assertEqual(answer, json.loads(self.capture.read_bytes()))
        self.assertEqual(json.loads(self.arguments.read_text())[:2], ["--archive", str(self.archive_alias.absolute())])
        return answer

    def test_declared_archive_alias_is_preserved_for_calls_and_feeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / ("long-archive-target-" * 8)
            actual.mkdir()
            alias = root / "short"
            alias.symlink_to(actual, target_is_directory=True)
            client = Client(alias, executable=EXECUTABLE)
            declared = str(alias.absolute())
            self.assertNotEqual(declared, str(alias.resolve()))
            self.assertEqual(client.archive, declared)
            self.assertEqual(client._arguments("status")[:3], [EXECUTABLE, "--archive", declared])
            self.assertEqual(client._arguments("changes")[:3], [EXECUTABLE, "--archive", declared])

    def test_query_and_read_match_cli_json(self):
        query = self.checked(lambda: self.client.query("sessions"))
        self.assertEqual(query["outcome"], "ok")
        self.assertTrue(query["value"]["rows"])
        read = self.checked(lambda: self.client.read("session:claude-code:python"))
        self.assertIn(read["outcome"], {"ok", "degraded"})
        self.assertTrue(read["value"]["document"]["items"])
        self.assertIn("read_at", read["value"]["snapshot"])
        self.assertEqual(
            [
                item["kind"]["text"]
                for item in read["value"]["document"]["items"]
                if item["kind"]["kind"] == "text"
            ],
            ["a synthetic Python client session"],
        )

    def test_generated_read_calls_continue_without_losing_their_envelope(self):
        target = "session:claude-code:python-pages"
        whole = self.direct.call("read", target, "--page-tokens", "100000").json()["value"]["document"]["items"]
        answer = self.checked(lambda: self.client.read(target, page_tokens=600))
        snapshot = answer["value"]["snapshot"]
        complete, partial, seen_calls = [], {}, set()
        count = 0
        while True:
            page = answer["value"]
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
            self.assertNotIn(ready["cli"], seen_calls)
            seen_calls.add(ready["cli"])
            words = shlex.split(ready["cli"], posix=True)
            self.assertEqual(words[0], "polylogue")
            answer = self.checked(lambda: self.client.call(*words[1:]))
            # The subprocess receives the generated command intact as argv.
            self.assertEqual(json.loads(self.arguments.read_text())[3:], words[1:])
            count += 1
            self.assertLess(count, 100)
        self.assertGreater(count, 0)
        self.assertFalse(partial)
        self.assertEqual(complete, whole)
        self.assertEqual([item["kind"]["text"] for item in complete if item["kind"]["kind"] == "text"],
                         ["Python λ🙂 continuation. " * 300])

    def test_refusals_keep_the_whole_answer(self):
        with self.assertRaises(Refused) as raised:
            self.client.read("session:claude-code:missing")
        self.assertEqual(raised.exception.kind, "not_found")
        self.assertEqual(raised.exception.answer.json(), json.loads(self.capture.read_bytes()))

    def test_empty_answers_keep_their_payload_and_exit_contract(self):
        answer = self.checked(lambda: self.client.query('sessions where repo:"python-missing"'))
        self.assertEqual(answer["outcome"], "empty")
        self.assertEqual(answer["value"]["rows"], [])

    def test_answers_are_frozen_without_changing_json_values(self):
        value = {"unknown": None, "zero": 0, "id": "a:b:c", "large": 2**62, "rows": [{"x": "ż"}]}
        answer = Answer(value)
        value["rows"][0]["x"] = "changed"
        self.assertEqual(answer.data["rows"][0]["x"], "ż")
        with self.assertRaises(TypeError):
            answer.data["rows"][0]["x"] = "changed"
        with self.assertRaises(FrozenInstanceError):
            answer.data = None
        copy = answer.json()
        self.assertIsNone(copy["unknown"])
        self.assertEqual(copy["large"], 2**62)
        copy["rows"].clear()
        self.assertEqual(len(answer.data["rows"]), 1)

    def test_resent_write_is_idempotent(self):
        arguments = ("setting", "signal.interrupts_per_hour", "9", "--request-id", "python-write")
        first = self.checked(lambda: self.client.call(*arguments))["value"]
        again = self.checked(lambda: self.client.call(*arguments))["value"]
        self.assertEqual(first["id"], again["id"])
        self.assertFalse(first["repeat"])
        self.assertTrue(again["repeat"])
        current = self.checked(lambda: self.client.call("setting", "signal.interrupts_per_hour"))["value"]
        self.assertEqual(current["value"], {"type": "interrupt_budget", "value": 9})
        self.assertEqual(first["id"], "python-write")
        self.assertEqual(current["authored"]["reference"], "assertion:python-write@python-write")

    def test_change_feed_can_continue_from_its_completion(self):
        first = [line.json() for line in self.direct.changes()]
        self.assertEqual(first[0]["type"], "begin")
        self.assertEqual(first[-1]["type"], "complete")
        self.assertTrue(first[-1]["replacement"])
        following = [line.json() for line in self.direct.changes(first[-1]["position"])]
        self.assertEqual(following[0]["position"], first[-1]["position"])
        self.assertEqual(following[-1]["type"], "complete")
        self.assertFalse(following[-1]["replacement"])
        generation = first[-1]["position"].split(":")[0]
        with self.assertRaises(Refused) as raised:
            list(self.direct.changes(f"{generation}:9223372036854775807:9223372036854775807"))
        self.assertEqual(raised.exception.kind, "bad_request")


if __name__ == "__main__":
    unittest.main()

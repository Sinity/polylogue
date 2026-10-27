"""Public client contracts against a real synthetic daemon, started by e2e."""

import json
import os
import subprocess
import unittest
from dataclasses import FrozenInstanceError

from polylogue.models import Answer
from polylogue.sdk import Client, Refused


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.client = Client(os.environ["POLYLOGUE_ARCHIVE"], executable=os.environ["POLYLOGUE_BIN"])

    def test_query_and_read_match_cli_json(self):
        for arguments, answer in [
            (["--", "sessions"], self.client.query("sessions")),
            (["read", "claude-code:python", "--view", "full"], self.client.read("claude-code:python")),
        ]:
            output = subprocess.run([self.client.executable, "--json", *arguments], check=True, capture_output=True)
            self.assertEqual(answer.json(), json.loads(output.stdout))

    def test_refusals_keep_the_whole_answer(self):
        with self.assertRaises(Refused) as raised:
            self.client.read("claude-code:missing")
        self.assertEqual(raised.exception.kind, "not_found")
        output = subprocess.run([self.client.executable, "--json", "read", "claude-code:missing"], capture_output=True)
        self.assertEqual(raised.exception.answer.json(), json.loads(output.stdout))

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

    def test_change_feed_can_continue_from_its_completion(self):
        first = [line.json() for line in self.client.changes()]
        self.assertEqual(first[0]["type"], "begin")
        self.assertEqual(first[-1]["type"], "complete")
        self.assertTrue(first[-1]["replacement"])
        following = [line.json() for line in self.client.changes(first[-1]["position"])]
        self.assertEqual(following[0]["position"], first[-1]["position"])
        self.assertEqual(following[-1]["type"], "complete")
        self.assertFalse(following[-1]["replacement"])
        generation = first[-1]["position"].split(":")[0]
        with self.assertRaises(Refused) as raised:
            list(self.client.changes(f"{generation}:9223372036854775807:9223372036854775807"))
        self.assertEqual(raised.exception.kind, "bad_request")


if __name__ == "__main__":
    unittest.main()

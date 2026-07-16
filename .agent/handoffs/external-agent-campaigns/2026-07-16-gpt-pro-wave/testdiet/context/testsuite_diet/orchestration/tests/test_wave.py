from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORCHESTRATION = HERE.parent
WAVE = ORCHESTRATION / "wave.py"
FAKE = HERE / "fake_attested_launcher.py"


class WaveRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        self.runs = self.base / "runs"
        self.events = self.base / "events.jsonl"
        (self.root / "context.md").write_text("context\n", encoding="utf-8")
        (self.root / ".gitignore").write_text(".agent/\n", encoding="utf-8")
        control = self.root / ".agent/scratch/testsuite_diet/control.md"
        control.parent.mkdir(parents=True)
        control.write_text("coordinator control input\n", encoding="utf-8")

    def job(
        self, job_id: str, *, wave: int, model: str, mission: str = "Implement the assigned behavioral result"
    ) -> dict[str, object]:
        target = f"owned/{job_id}.py"
        (self.root / target).parent.mkdir(parents=True, exist_ok=True)
        (self.root / target).write_text("value = 1\n", encoding="utf-8")
        return {
            "id": job_id,
            "wave": wave,
            "model": model,
            "effort": "high",
            "mission": mission,
            "required_reads": ["context.md"],
            "write_files": [target],
            "avoid_files": [f"avoid/{job_id}.py"],
            "acceptance": "Assigned file changes and the named focused check passes",
            "focused_tests": [f"devtools test tests/unit/test_{job_id.replace('-', '_')}.py"],
        }

    def manifest(self, jobs: list[dict[str, object]], name: str = "manifest.json") -> Path:
        path = self.base / name
        path.write_text(json.dumps(jobs), encoding="utf-8")
        return path

    def commit_root(self) -> None:
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=self.root, check=False)
        if staged.returncode != 0:
            commit_env = dict(os.environ)
            commit_env.update(
                {
                    "GIT_AUTHOR_NAME": "Wave Runner Test",
                    "GIT_AUTHOR_EMAIL": "wave-runner-test@example.invalid",
                    "GIT_COMMITTER_NAME": "Wave Runner Test",
                    "GIT_COMMITTER_EMAIL": "wave-runner-test@example.invalid",
                }
            )
            subprocess.run(["git", "commit", "-q", "-m", "test fixture"], cwd=self.root, check=True, env=commit_env)

    def command(
        self,
        *parts: str,
        env: dict[str, str] | None = None,
        prepare_clean: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if prepare_clean:
            self.commit_root()
        command = ["python", str(WAVE), "--root", str(self.root), "--runs-root", str(self.runs), *parts]
        return subprocess.run(command, check=False, text=True, capture_output=True, env=env)

    def test_validate_rejects_missing_reads_and_same_wave_write_collision(self) -> None:
        jobs = [self.job("one", wave=1, model="gpt-5.6-terra"), self.job("two", wave=1, model="gpt-5.6-luna")]
        jobs[1]["write_files"] = jobs[0]["write_files"]
        jobs[1]["required_reads"] = ["missing.md"]
        result = self.command("validate", str(self.manifest(jobs)))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("write collision", result.stdout)
        self.assertIn("does not exist", result.stdout)

    def test_validate_rejects_same_wave_read_write_hazard(self) -> None:
        jobs = [self.job("reader", wave=1, model="gpt-5.6-terra"), self.job("writer", wave=1, model="gpt-5.6-terra")]
        jobs[0]["required_reads"] = jobs[1]["write_files"]
        result = self.command("validate", str(self.manifest(jobs)))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("read/write collision", result.stdout)

    def test_validate_rejects_sol_as_a_worker_model(self) -> None:
        job = self.job("worker", wave=1, model="gpt-5.6-sol")
        result = self.command("validate", str(self.manifest([job])))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("worker model", result.stdout)

    def test_render_is_byte_stable_and_preserves_mixed_model_contract(self) -> None:
        jobs = [
            self.job("terra", wave=1, model="gpt-5.6-terra"),
            self.job("luna", wave=2, model="gpt-5.6-luna"),
        ]
        manifest = self.manifest(jobs)
        first = self.command("render", str(manifest), "--run-id", "render-a")
        second = self.command("render", str(manifest), "--run-id", "render-b")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        for job in jobs:
            a = (self.runs / "render-a/prompts" / f"{job['id']}.prompt").read_bytes()
            b = (self.runs / "render-b/prompts" / f"{job['id']}.prompt").read_bytes()
            self.assertEqual(a, b)
            self.assertTrue(a.startswith(b"POLYLOGUE SHARED-WORKTREE TESTSUITE-DIET WORKER v1\n"))
            self.assertIn(str(job["model"]).encode(), a)

    def test_run_bounds_parallelism_and_collects_attested_structured_receipts(self) -> None:
        jobs = [self.job("foundation", wave=1, model="gpt-5.6-terra")]
        jobs.extend(
            self.job(f"parallel-{index}", wave=2, model="gpt-5.6-luna" if index == 4 else "gpt-5.6-terra")
            for index in range(5)
        )
        manifest = self.manifest(jobs)
        env = dict(os.environ)
        env["TESTSUITE_DIET_FAKE_EVENTS"] = str(self.events)
        env["TESTSUITE_DIET_FAKE_DELAY"] = "0.1"
        result = self.command(
            "run",
            str(manifest),
            "--run-id",
            "mixed-run",
            "--launcher",
            str(FAKE),
            "--max-concurrency",
            "4",
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads((self.runs / "mixed-run/run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "completed")
        self.assertEqual(record["max_concurrency"], 4)
        self.assertEqual({item["state"] for item in record["jobs"].values()}, {"completed"})
        self.assertEqual(record["jobs"]["parallel-4"]["model"], "gpt-5.6-luna")

        active = 0
        maximum = 0
        wave_one_end = 0.0
        wave_two_start = float("inf")
        for line in self.events.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if "foundation" in event["job_id"] and event["kind"] == "end":
                wave_one_end = event["time"]
            if "parallel-" in event["job_id"] and event["kind"] == "start":
                wave_two_start = min(wave_two_start, event["time"])
            active += 1 if event["kind"] == "start" else -1
            maximum = max(maximum, active)
        self.assertGreaterEqual(wave_two_start, wave_one_end)
        self.assertGreaterEqual(maximum, 2)
        self.assertLessEqual(maximum, 4)

        for job in jobs:
            final = json.loads((self.runs / "mixed-run/final" / f"{job['id']}.json").read_text(encoding="utf-8"))
            self.assertEqual(final["changed_files"], job["write_files"])
            attested_path = Path(record["jobs"][job["id"]]["artifacts"]["attested"])
            attested = json.loads(attested_path.read_text(encoding="utf-8"))
            self.assertEqual(attested["model"], job["model"])
            self.assertEqual(attested["effort"], "high")

    def test_incomplete_packet_blocks_without_speculative_edit(self) -> None:
        job = self.job("blocked", wave=1, model="gpt-5.6-terra", mission="INCOMPLETE_PACKET lacks an oracle")
        later = self.job("later", wave=2, model="gpt-5.6-terra")
        target = self.root / str(job["write_files"][0])
        before = target.read_bytes()
        result = self.command(
            "run",
            str(self.manifest([job, later], "blocked.json")),
            "--run-id",
            "blocked-run",
            "--launcher",
            str(FAKE),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads((self.runs / "blocked-run/run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "completed-with-blockers")
        self.assertEqual(record["jobs"]["blocked"]["state"], "blocked")
        self.assertEqual(record["jobs"]["later"]["state"], "skipped-upstream-wave")
        self.assertEqual(target.read_bytes(), before)

    def test_prose_only_completion_is_rejected(self) -> None:
        job = self.job("prose", wave=1, model="gpt-5.6-terra", mission="PROSE_ONLY claims completion")
        result = self.command(
            "run",
            str(self.manifest([job], "prose.json")),
            "--run-id",
            "prose-run",
            "--launcher",
            str(FAKE),
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        record = json.loads((self.runs / "prose-run/run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "invalid")
        self.assertIn("prose-only completion", " ".join(record["jobs"]["prose"]["errors"]))

    def test_run_rejects_dirty_execution_checkout(self) -> None:
        job = self.job("dirty", wave=1, model="gpt-5.6-terra")
        self.commit_root()
        (self.root / "context.md").write_text("user change\n", encoding="utf-8")
        result = self.command(
            "run",
            str(self.manifest([job], "dirty.json")),
            "--run-id",
            "dirty-run",
            "--launcher",
            str(FAKE),
            prepare_clean=False,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        record = json.loads((self.runs / "dirty-run/run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "invalid")
        self.assertIn("must be clean", " ".join(record["errors"]))

    def test_run_rejects_outside_assignment_edits(self) -> None:
        job = self.job("outside", wave=1, model="gpt-5.6-terra", mission="UNASSIGNED_EDIT")
        result = self.command(
            "run",
            str(self.manifest([job], "outside.json")),
            "--run-id",
            "outside-run",
            "--launcher",
            str(FAKE),
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        record = json.loads((self.runs / "outside-run/run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "invalid")
        self.assertIn("outside its assignment", " ".join(record["errors"]))

    def test_run_rejects_ignored_control_plane_edits(self) -> None:
        job = self.job("control", wave=1, model="gpt-5.6-terra", mission="CONTROL_PLANE_EDIT")
        later = self.job("later", wave=2, model="gpt-5.6-terra")
        result = self.command(
            "run",
            str(self.manifest([job, later], "control.json")),
            "--run-id",
            "control-run",
            "--launcher",
            str(FAKE),
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        record = json.loads((self.runs / "control-run/run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "invalid")
        self.assertEqual(record["jobs"]["later"]["state"], "skipped-upstream-wave")
        self.assertIn("control tree", " ".join(record["errors"]))


if __name__ == "__main__":
    unittest.main()

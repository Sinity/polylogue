"""Tests for showcase baseline verification."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


class TestVerifyShowcaseImportable:
    """Verify the script is importable and has expected interface."""

    def test_module_imports(self) -> None:
        """devtools.lab_scenario is importable."""
        assert importlib.import_module("devtools.lab_scenario") is not None

    def test_main_callable(self) -> None:
        """main() function exists and is callable."""
        from devtools.lab_scenario import main

        assert callable(main)

    def test_get_tier_0_exercises_returns_list(self) -> None:
        """get_tier_0_exercises returns a non-empty list of exercises."""
        from devtools.lab_scenario import get_tier_0_exercises

        exercises = get_tier_0_exercises()
        assert isinstance(exercises, list)
        assert len(exercises) > 0

    def test_tier_0_exercises_are_structural_or_sources(self) -> None:
        """All tier 0 exercises belong to structural or sources groups."""
        from devtools.lab_scenario import get_tier_0_exercises

        exercises = get_tier_0_exercises()
        for ex in exercises:
            assert ex.group in ("structural", "sources"), f"Exercise {ex.name!r} has unexpected group {ex.group!r}"

    def test_tier_0_exercises_need_no_data(self) -> None:
        """Tier 0 exercises should not require database data."""
        from devtools.lab_scenario import get_tier_0_exercises

        exercises = get_tier_0_exercises()
        for ex in exercises:
            assert not ex.needs_data, f"Tier 0 exercise {ex.name!r} requires data"


class TestBaselineComparison:
    """Test the compare_outputs logic."""

    def test_identical_outputs_no_drift(self) -> None:
        """Identical outputs produce no drift."""
        from devtools.lab_scenario import compare_outputs

        outputs = {"help-main": "Usage: polylogue\n", "version": "polylogue 1.0\n"}
        baselines = {"help-main": "Usage: polylogue\n", "version": "polylogue 1.0\n"}

        drifts = compare_outputs(outputs, baselines)
        assert drifts == []

    def test_changed_output_detected(self) -> None:
        """Changed output is detected as drift."""
        from devtools.lab_scenario import compare_outputs

        current = {"help-main": "Usage: polylogue [OPTIONS]\n"}
        baselines = {"help-main": "Usage: polylogue\n"}

        drifts = compare_outputs(current, baselines)
        assert len(drifts) == 1
        assert "CHANGED: help-main" in drifts[0]

    def test_new_exercise_detected(self) -> None:
        """New exercise (no baseline) is detected."""
        from devtools.lab_scenario import compare_outputs

        current = {"help-main": "Usage\n", "new-cmd": "New output\n"}
        baselines = {"help-main": "Usage\n"}

        drifts = compare_outputs(current, baselines)
        assert any("NEW: new-cmd" in d for d in drifts)

    def test_removed_exercise_detected(self) -> None:
        """Removed exercise (in baseline but not current) is detected."""
        from devtools.lab_scenario import compare_outputs

        current = {"help-main": "Usage\n"}
        baselines = {"help-main": "Usage\n", "old-cmd": "Gone\n"}

        drifts = compare_outputs(current, baselines)
        assert any("REMOVED: old-cmd" in d for d in drifts)

    def test_empty_baselines_all_new(self) -> None:
        """With no baselines, all exercises are new."""
        from devtools.lab_scenario import compare_outputs

        current = {"a": "1\n", "b": "2\n"}
        drifts = compare_outputs(current, {})
        assert len(drifts) == 2
        assert all("NEW:" in d for d in drifts)


class TestBaselinePersistence:
    """Test baseline save/load cycle."""

    def test_save_and_load_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Baselines saved to disk can be loaded back identically."""
        import devtools.lab_scenario as mod

        # Redirect BASELINE_DIR to tmp_path
        test_dir = tmp_path / "baselines"
        monkeypatch.setattr(mod, "BASELINE_DIR", test_dir)

        outputs = {"help-main": "Usage: polylogue\n", "version": "1.0.0\n"}

        mod.save_baselines(outputs)
        loaded = mod.load_baselines()

        assert loaded == outputs

    def test_load_missing_dir_returns_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loading from non-existent directory returns empty dict."""
        import devtools.lab_scenario as mod

        monkeypatch.setattr(mod, "BASELINE_DIR", tmp_path / "nonexistent")
        assert mod.load_baselines() == {}


class TestVerifyShowcaseBaselinesControlFlow:
    """verify_showcase_baselines control flow without subprocess overhead.

    run_tier_0() is mocked so we test the comparison/drift logic, not
    the exercise subprocesses (those are a CI step, not a unit test).
    """

    def test_all_match_returns_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When all exercise outputs match baselines, return 0."""
        import devtools.lab_scenario as mod

        baseline_dir = tmp_path / "baselines"
        baseline_dir.mkdir()
        (baseline_dir / "cmd-a.txt").write_text("output A\n", encoding="utf-8")
        monkeypatch.setattr(mod, "BASELINE_DIR", baseline_dir)
        monkeypatch.setattr(mod, "run_tier_0", lambda: {"cmd-a": "output A\n"})
        assert mod.verify_showcase_baselines(update=False) == 0

    def test_drift_detected_returns_one(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When exercise output differs from baseline, return 1."""
        import devtools.lab_scenario as mod

        baseline_dir = tmp_path / "baselines"
        baseline_dir.mkdir()
        (baseline_dir / "cmd-a.txt").write_text("expected\n", encoding="utf-8")
        monkeypatch.setattr(mod, "BASELINE_DIR", baseline_dir)
        monkeypatch.setattr(mod, "run_tier_0", lambda: {"cmd-a": "actual\n"})
        assert mod.verify_showcase_baselines(update=False) == 1

    def test_update_saves_baselines(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--update writes current outputs to the baseline directory."""
        import devtools.lab_scenario as mod

        baseline_dir = tmp_path / "baselines"
        monkeypatch.setattr(mod, "BASELINE_DIR", baseline_dir)
        monkeypatch.setattr(mod, "run_tier_0", lambda: {"cmd-a": "fresh output\n"})
        assert mod.verify_showcase_baselines(update=True) == 0
        assert (baseline_dir / "cmd-a.txt").read_text() == "fresh output\n"

    def test_update_refuses_failed_exercise_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--update does not save blank output when the exercise runner fails."""
        import devtools.lab_scenario as mod

        baseline_dir = tmp_path / "baselines"
        monkeypatch.setattr(mod, "BASELINE_DIR", baseline_dir)

        def _failed_run() -> dict[str, str]:
            raise RuntimeError("tier-0 showcase exercises failed:\n  - help-main: exit -1: invoke crashed")

        monkeypatch.setattr(mod, "run_tier_0", _failed_run)

        assert mod.verify_showcase_baselines(update=True) == 1
        assert not baseline_dir.exists()

    def test_main_returns_1_when_no_baselines_and_no_update(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """verify_showcase_baselines returns 1 when no baselines exist and update=False.

        This is the pure-logic unit path: run_tier_0 is mocked so no subprocesses
        are spawned.  The E2E subprocess path belongs in the slow/lab lane.
        """
        import devtools.lab_scenario as mod

        monkeypatch.setattr(mod, "BASELINE_DIR", tmp_path / "nonexistent_baselines")
        monkeypatch.setattr(mod, "run_tier_0", lambda: {"help-main": "Usage: polylogue\n"})
        assert mod.verify_showcase_baselines(update=False) == 1

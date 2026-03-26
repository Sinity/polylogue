"""Tests for VHS tape generation from showcase exercises."""

from __future__ import annotations

from polylogue.showcase.exercises import Exercise, vhs_exercises
from polylogue.showcase.vhs import generate_all_tapes, generate_tape


class TestGenerateTape:
    """generate_tape produces valid tape content."""

    def test_basic_tape_has_output_directive(self):
        ex = Exercise(
            name="test-ex",
            group="structural",
            description="Test exercise",
            args=["--help"],
            vhs_capture=True,
        )
        tape = generate_tape(ex)
        assert "Output test-ex.gif" in tape

    def test_tape_has_set_directives(self):
        ex = Exercise(
            name="test-ex",
            group="structural",
            description="Test exercise",
            args=["--help"],
            vhs_capture=True,
        )
        tape = generate_tape(ex)
        assert "Set FontSize" in tape
        assert "Set Width" in tape
        assert "Set Height" in tape
        assert "Set Padding" in tape

    def test_tape_includes_correct_command(self):
        ex = Exercise(
            name="test-cmd",
            group="structural",
            description="Test",
            args=["run", "--preview"],
            vhs_capture=True,
        )
        tape = generate_tape(ex)
        assert 'Type "polylogue run --preview"' in tape
        assert "Enter" in tape

    def test_tape_with_no_args(self):
        ex = Exercise(
            name="default",
            group="structural",
            description="Default",
            args=[],
            vhs_capture=True,
        )
        tape = generate_tape(ex)
        assert 'Type "polylogue"' in tape

    def test_tape_with_capture_steps(self):
        ex = Exercise(
            name="custom",
            group="structural",
            description="Custom steps",
            args=["--help"],
            vhs_capture=True,
            capture_steps=(
                'Type "polylogue --help"',
                "Enter",
                "Sleep 3s",
                'Type "polylogue --version"',
                "Enter",
            ),
        )
        tape = generate_tape(ex)
        # Custom steps should be present
        assert 'Type "polylogue --help"' in tape
        assert 'Type "polylogue --version"' in tape
        # Auto-generated command should NOT be present (since capture_steps overrides)
        lines = tape.splitlines()
        type_lines = [line for line in lines if line.startswith("Type")]
        assert len(type_lines) == 2

    def test_custom_dimensions(self):
        ex = Exercise(
            name="dim",
            group="structural",
            description="Dimensions",
            args=["--help"],
            vhs_capture=True,
        )
        tape = generate_tape(ex, font_size=22, padding=30)
        assert "Set FontSize 22" in tape
        assert "Set Padding 30" in tape

    def test_tape_has_exercise_description_comment(self):
        ex = Exercise(
            name="desc",
            group="structural",
            description="My cool exercise",
            args=["--help"],
            vhs_capture=True,
        )
        tape = generate_tape(ex)
        assert "# My cool exercise" in tape


class TestGenerateAllTapes:
    """generate_all_tapes returns expected exercises."""

    def test_returns_dict_of_tape_content(self):
        tapes = generate_all_tapes()
        assert isinstance(tapes, dict)
        assert len(tapes) > 0
        for name, content in tapes.items():
            assert isinstance(name, str)
            assert isinstance(content, str)
            assert "Output" in content

    def test_only_vhs_capture_exercises(self):
        vhs_names = {e.name for e in vhs_exercises()}
        tapes = generate_all_tapes()
        assert set(tapes.keys()) == vhs_names

    def test_skips_non_capturable_in_explicit_list(self):
        exs = [
            Exercise(name="cap", group="g", description="C", vhs_capture=True),
            Exercise(name="nocap", group="g", description="N", vhs_capture=False),
        ]
        tapes = generate_all_tapes(exs)
        assert "cap" in tapes
        assert "nocap" not in tapes

    def test_writes_to_output_dir(self, tmp_path):
        exs = [
            Exercise(name="write-test", group="g", description="W", vhs_capture=True, args=["--help"]),
        ]
        tapes = generate_all_tapes(exs, output_dir=tmp_path / "tapes")
        assert (tmp_path / "tapes" / "write-test.tape").exists()
        content = (tmp_path / "tapes" / "write-test.tape").read_text()
        assert content == tapes["write-test"]


class TestTapeContentCorrectness:
    """Tape content includes correct commands for real exercises."""

    def test_help_main_tape(self):
        exs = [e for e in vhs_exercises() if e.name == "help-main"]
        assert len(exs) == 1
        tape = generate_tape(exs[0])
        assert 'Type "polylogue --help"' in tape

    def test_stats_default_tape(self):
        exs = [e for e in vhs_exercises() if e.name == "stats-default"]
        assert len(exs) == 1
        tape = generate_tape(exs[0])
        # stats-default has no args → just "polylogue"
        assert 'Type "polylogue"' in tape

"""Tests for direct VHS tape generation specs."""

from __future__ import annotations

from pathlib import Path

from polylogue.showcase.vhs import VHSTapeSpec, default_tape_specs, generate_all_tapes, generate_tape


class TestGenerateTape:
    """generate_tape produces valid tape content."""

    def test_basic_tape_has_output_directive(self) -> None:
        spec = VHSTapeSpec(name="test-ex", description="Test exercise", display_command=("polylogue", "--help"))

        tape = generate_tape(spec)

        assert "Output test-ex.gif" in tape

    def test_tape_has_set_directives(self) -> None:
        spec = VHSTapeSpec(name="test-ex", description="Test exercise", display_command=("polylogue", "--help"))

        tape = generate_tape(spec)

        assert "Set FontSize" in tape
        assert "Set Width" in tape
        assert "Set Height" in tape
        assert "Set Padding" in tape

    def test_tape_includes_correct_command(self) -> None:
        spec = VHSTapeSpec(name="test-cmd", description="Test", display_command=("polylogue", "ops", "status"))

        tape = generate_tape(spec)

        assert 'Type "polylogue ops status"' in tape
        assert "Enter" in tape

    def test_tape_with_no_args(self) -> None:
        spec = VHSTapeSpec(name="default", description="Default")

        tape = generate_tape(spec)

        assert 'Type "polylogue"' in tape

    def test_tape_with_capture_steps(self) -> None:
        spec = VHSTapeSpec(
            name="custom",
            description="Custom steps",
            display_command=("polylogue", "--help"),
            capture_steps=(
                'Type "polylogue --help"',
                "Enter",
                "Sleep 3s",
                'Type "polylogue --version"',
                "Enter",
            ),
        )

        tape = generate_tape(spec)

        assert 'Type "polylogue --help"' in tape
        assert 'Type "polylogue --version"' in tape
        type_lines = [line for line in tape.splitlines() if line.startswith("Type")]
        assert len(type_lines) == 2

    def test_custom_dimensions(self) -> None:
        spec = VHSTapeSpec(name="dim", description="Dimensions", display_command=("polylogue", "--help"))

        tape = generate_tape(spec, font_size=22, padding=30)

        assert "Set FontSize 22" in tape
        assert "Set Padding 30" in tape

    def test_tape_has_spec_description_comment(self) -> None:
        spec = VHSTapeSpec(name="desc", description="My cool tape", display_command=("polylogue", "--help"))

        tape = generate_tape(spec)

        assert "# My cool tape" in tape


class TestGenerateAllTapes:
    """generate_all_tapes returns expected direct tape specs."""

    def test_returns_dict_of_tape_content(self) -> None:
        tapes = generate_all_tapes()

        assert isinstance(tapes, dict)
        assert len(tapes) > 0
        for name, content in tapes.items():
            assert isinstance(name, str)
            assert isinstance(content, str)
            assert "Output" in content

    def test_uses_direct_default_tape_inventory(self) -> None:
        expected_names = {spec.name for spec in default_tape_specs()}

        tapes = generate_all_tapes()

        assert set(tapes.keys()) == expected_names

    def test_accepts_explicit_specs(self) -> None:
        specs = [
            VHSTapeSpec(name="one", description="One", display_command=("polylogue", "--help")),
            VHSTapeSpec(name="two", description="Two", display_command=("polylogue", "ops", "status")),
        ]

        tapes = generate_all_tapes(specs)

        assert set(tapes) == {"one", "two"}

    def test_writes_to_output_dir(self, tmp_path: Path) -> None:
        specs = [
            VHSTapeSpec(name="write-test", description="W", display_command=("polylogue", "--help")),
        ]

        tapes = generate_all_tapes(specs, output_dir=tmp_path / "tapes")

        assert (tmp_path / "tapes" / "write-test.tape").exists()
        content = (tmp_path / "tapes" / "write-test.tape").read_text()
        assert content == tapes["write-test"]


class TestDefaultTapeContent:
    """Default tape specs preserve the existing visual evidence flows."""

    def test_default_tape_names(self) -> None:
        assert [spec.name for spec in default_tape_specs()] == [
            "help-main",
            "stats-default",
            "query-list",
            "doctor-readiness",
            "query-latest-md",
        ]

    def test_help_main_tape(self) -> None:
        spec = next(spec for spec in default_tape_specs() if spec.name == "help-main")

        tape = generate_tape(spec)

        assert 'Type "polylogue --help"' in tape

    def test_stats_default_tape(self) -> None:
        spec = next(spec for spec in default_tape_specs() if spec.name == "stats-default")

        tape = generate_tape(spec)

        assert 'Type "polylogue"' in tape

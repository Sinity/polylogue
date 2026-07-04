"""Tests for direct VHS tape generation specs."""

from __future__ import annotations

from pathlib import Path

from devtools.visual_vhs import VHSTapeSpec, default_tape_specs, generate_all_tapes, generate_tape


class TestGenerateTape:
    """generate_tape produces valid tape content."""

    def test_basic_tape_has_output_directive(self) -> None:
        spec = VHSTapeSpec(name="test-flow", description="Test flow", display_command=("polylogue", "--help"))

        tape = generate_tape(spec)

        assert "Output test-flow.gif" in tape

    def test_tape_has_set_directives(self) -> None:
        spec = VHSTapeSpec(name="test-flow", description="Test flow", display_command=("polylogue", "--help"))

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

    def test_spec_dimensions_are_used_by_default(self) -> None:
        spec = VHSTapeSpec(
            name="spec-dim",
            description="Dimensions",
            display_command=("polylogue", "--help"),
            output_width=90,
            output_height=20,
            font_size=18,
            padding=12,
        )

        tape = generate_tape(spec)

        assert "Set FontSize 18" in tape
        assert "Set Width 990" in tape
        assert "Set Height 440" in tape
        assert "Set Padding 12" in tape

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
            "demo-tour",
            "query-tour",
            "reader-evidence-tour",
            "browser-capture-tour",
        ]

    def test_demo_tour_tape(self) -> None:
        spec = next(spec for spec in default_tape_specs() if spec.name == "demo-tour")

        tape = generate_tape(spec)

        assert 'Type "polylogue demo tour --out-dir demo-tour --force"' in tape
        assert 'Type "cat demo-tour/report.md"' in tape

    def test_query_tour_uses_demo_archive_not_live_archive(self) -> None:
        spec = next(spec for spec in default_tape_specs() if spec.name == "query-tour")

        tape = generate_tape(spec)

        assert "POLYLOGUE_ARCHIVE_ROOT=query-tour/archive" in tape
        assert 'Type "polylogue read --all -n 1"' not in tape
        assert 'Type "polylogue --latest -f markdown"' not in tape

    def test_reader_evidence_tour_uses_visual_smoke_lane(self) -> None:
        spec = next(spec for spec in default_tape_specs() if spec.name == "reader-evidence-tour")

        tape = generate_tape(spec)

        assert "devtools lab smoke run reader-visual-smoke" in tape
        assert "reader-visual-smoke.json" in tape

    def test_browser_capture_tour_uses_deterministic_provider_smoke(self) -> None:
        spec = next(spec for spec in default_tape_specs() if spec.name == "browser-capture-tour")

        tape = generate_tape(spec)

        assert "devtools workspace dev-loop --browser-provider-smoke --json" in tape
        assert "browser_provider_smoke" in tape
        assert "provider_statuses" in tape
        assert "POLYLOGUE_ARCHIVE_ROOT" not in tape

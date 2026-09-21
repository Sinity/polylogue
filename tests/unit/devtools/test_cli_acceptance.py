"""Contracts for ``devtools verify cli-acceptance`` (polylogue-dpm6o).

The command renders the public CLI at four widths and three color lanes, lints
the rendered text, and delegates the latency lane. Two kinds of test live here:

* **Live surface** -- the real gallery is captured and must satisfy every
  declared lint property. What makes it red: any regression in the CLI that
  emits ANSI off a TTY, lets ``NO_COLOR`` change already-plain text, drops the
  "what to run next" line out of an error, mangles the user's rejected token,
  raises out of a public invocation, exits outside the declared table, or adds
  a help paragraph that overflows a terminal past the recorded baseline.
* **Mutants** -- each lint property is fed a hand-built frame that violates it
  and must produce exactly that finding. Without these, a lint that silently
  stopped checking anything would still pass the live-surface test. Deleting a
  check from :func:`devtools.cli_acceptance.lint_findings` turns the matching
  mutant green-to-red here.
"""

from __future__ import annotations

import json

import pytest

from devtools import cli_acceptance
from devtools.cli_acceptance import (
    COLOR_LANES,
    GALLERY_SAMPLES,
    GALLERY_WIDTHS,
    LINT_PROPERTIES,
    REFLOW_OVERFLOW_BASELINE,
    Finding,
    GalleryFrame,
    GallerySample,
    build_gallery,
    lint_findings,
    measure_overflow,
    render_gallery_markdown,
    scenario_findings,
)
from devtools.command_catalog import COMMANDS

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def gallery() -> tuple[GalleryFrame, ...]:
    return build_gallery()


def _codes(findings: tuple[Finding, ...]) -> set[str]:
    return {finding.code for finding in findings}


# ---------------------------------------------------------------------------
# The live CLI surface
# ---------------------------------------------------------------------------


class TestLiveSurface:
    def test_the_gallery_covers_every_declared_sample_width_and_lane(self, gallery: tuple[GalleryFrame, ...]) -> None:
        assert len(gallery) == len(GALLERY_SAMPLES) * len(GALLERY_WIDTHS) * len(COLOR_LANES)
        assert {frame.sample for frame in gallery} == {sample.name for sample in GALLERY_SAMPLES}
        assert {frame.width for frame in gallery} == set(GALLERY_WIDTHS)
        assert {frame.lane for frame in gallery} == {lane for lane, _ in COLOR_LANES}

    def test_every_captured_frame_has_text(self, gallery: tuple[GalleryFrame, ...]) -> None:
        """Anti-vacuity: an empty capture would satisfy most lint checks trivially."""
        assert all(frame.text.strip() for frame in gallery)

    def test_the_live_cli_satisfies_every_declared_lint_property(self, gallery: tuple[GalleryFrame, ...]) -> None:
        findings = lint_findings(gallery)
        assert findings == (), "\n".join(f"{f.code}: {f.subject}: {f.message}" for f in findings)

    def test_every_declared_invocation_terminates_typed(self, gallery: tuple[GalleryFrame, ...]) -> None:
        assert scenario_findings(gallery) == ()

    def test_the_overflow_baseline_is_exact_for_every_sample_and_width(self, gallery: tuple[GalleryFrame, ...]) -> None:
        """The ratchet covers the whole surface, and no row over-allows today."""
        measured = measure_overflow(gallery)
        assert set(measured) == set(REFLOW_OVERFLOW_BASELINE)
        assert measured == dict(REFLOW_OVERFLOW_BASELINE)

    def test_the_2026_09_21_raise_bought_surface_that_is_still_on_the_screen(
        self, gallery: tuple[GalleryFrame, ...]
    ) -> None:
        """polylogue-1khzy: the five raised rows are paid for, not re-recorded.

        The raise was attributed entirely to f8a63f0b6 (#5249). Each string
        below is one of the lines that diff introduced, so the recorded
        numbers stay tied to output a reader actually gets.

        Anti-vacuity: withdraw any of this surface -- drop ``--to``/``--out``,
        drop the ``--json`` alias, or stop publishing the ``md``/``jsonl``
        short spellings -- and the corresponding assertion here goes red while
        ``test_the_overflow_baseline_is_exact_for_every_sample_and_width``
        independently goes red the other way, because the measured count then
        falls below the recorded baseline. Neither can be silenced alone.
        """
        by_sample = {frame.sample: frame.text for frame in gallery if frame.lane == "default" and frame.width == 80}
        root, read = by_sample["root-help"], by_sample["read-help"]

        # root-help: three new option lines (+3 @40, +1 @60).
        assert "--to [terminal|stdout|browser|clipboard|file]" in root
        assert "--out PATH" in root
        # root-help and read-help: the published short spellings (+1 @80 each).
        for text in (root, read):
            assert "|jsonl|" in text
            assert "|md|" in text
        # read-help: the --json alias line (+1 @40).
        assert "Alias for --format json." in read

    def test_the_rendered_gallery_shows_each_sample_at_each_width(self, gallery: tuple[GalleryFrame, ...]) -> None:
        markdown = render_gallery_markdown(gallery)
        for sample in GALLERY_SAMPLES:
            assert f"## {sample.name}" in markdown
        for width in GALLERY_WIDTHS:
            assert f"### {width} columns" in markdown


# ---------------------------------------------------------------------------
# Lint mutants: one violating frame per property
# ---------------------------------------------------------------------------


def _frame(
    text: str, *, sample: str = "version", width: int = 120, lane: str = "default", exit_code: int = 0
) -> GalleryFrame:
    return GalleryFrame(sample=sample, width=width, lane=lane, exit_code=exit_code, text=text)


_META = GallerySample("version", ("--version",), "meta")
_ERROR = GallerySample("boom", ("boom",), "error", echo="boom")


class TestLintMutants:
    def test_an_ansi_escape_off_tty_is_reported(self) -> None:
        findings = lint_findings((_frame("\x1b[31mred\x1b[0m"),), samples=(_META,), baseline={"version@120": 0})
        assert "no_ansi_off_tty" in _codes(findings)

    def test_a_raw_control_character_is_reported(self) -> None:
        findings = lint_findings((_frame("before\x07after"),), samples=(_META,), baseline={"version@120": 0})
        assert "control_safety" in _codes(findings)

    def test_a_bidi_override_is_reported(self) -> None:
        findings = lint_findings((_frame("before\u202eafter"),), samples=(_META,), baseline={"version@120": 0})
        assert "control_safety" in _codes(findings)

    def test_a_color_lane_that_changes_the_text_is_reported(self) -> None:
        frames = (
            _frame("plain", lane="default"),
            _frame("PLAIN", lane="no-color"),
        )
        findings = lint_findings(frames, samples=(_META,), baseline={"version@120": 0})
        assert "color_is_not_meaning" in _codes(findings)

    def test_identical_color_lanes_are_not_reported(self) -> None:
        """Anti-vacuity for the check above: matching lanes must stay silent."""
        frames = (_frame("plain", lane="default"), _frame("plain", lane="no-color"))
        assert _codes(lint_findings(frames, samples=(_META,), baseline={"version@120": 0})) == set()

    def test_a_dropped_user_token_is_reported(self) -> None:
        frames = (_frame("Error: unknown input", sample="boom", exit_code=2),)
        findings = lint_findings(frames, samples=(_ERROR,), baseline={"boom@120": 0})
        assert "echo_preserved" in _codes(findings)

    def test_a_silent_non_zero_exit_is_reported(self) -> None:
        frames = (_frame("boom polylogue", sample="boom", exit_code=1),)
        findings = lint_findings(frames, samples=(_ERROR,), baseline={"boom@120": 0})
        assert "failure_is_textual" in _codes(findings)

    def test_an_error_without_a_next_command_is_reported(self) -> None:
        frames = (_frame("Error: boom", sample="boom", exit_code=2),)
        findings = lint_findings(frames, samples=(_ERROR,), baseline={"boom@120": 0})
        assert "repair_named" in _codes(findings)

    def test_a_traceback_is_reported(self) -> None:
        text = "Traceback (most recent call last):\n  File x\nValueError"
        findings = lint_findings((_frame(text),), samples=(_META,), baseline={"version@120": 0})
        assert "no_traceback" in _codes(findings)

    def test_an_undeclared_exit_code_is_reported(self) -> None:
        frames = (_frame("Error: boom polylogue status", sample="boom", exit_code=3),)
        findings = lint_findings(frames, samples=(_ERROR,), baseline={"boom@120": 0})
        assert "declared_exit_code" in _codes(findings)

    def test_growing_overflow_past_the_baseline_is_reported(self) -> None:
        frames = (_frame("x" * 200, width=80),)
        findings = lint_findings(frames, samples=(_META,), baseline={"version@80": 0})
        assert "reflow_overflow_ratchet" in _codes(findings)

    def test_an_unrecorded_sample_width_is_reported(self) -> None:
        """A new sample or width cannot slip in without a recorded baseline."""
        findings = lint_findings((_frame("ok", width=80),), samples=(_META,), baseline={})
        assert "reflow_overflow_ratchet" in _codes(findings)

    def test_an_empty_gallery_is_not_a_pass(self) -> None:
        assert _codes(lint_findings(())) == {"empty_gallery"}

    def test_every_declared_property_has_a_mutant_in_this_module(self) -> None:
        """The property table and this suite cannot drift apart silently."""
        source = __import__("pathlib").Path(__file__).read_text(encoding="utf-8")
        for code, _statement in LINT_PROPERTIES:
            assert f'"{code}" in _codes' in source or f'== {{"{code}"}}' in source, code


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------


class TestCommandRegistration:
    def test_the_command_is_a_subcommand_of_an_existing_root_verb(self) -> None:
        """The twelve-root fold stays folded: acceptance lives under ``verify``."""
        spec = COMMANDS["verify cli-acceptance"]
        assert spec.command_path == ("verify", "cli-acceptance")
        assert spec.module == "devtools.cli_acceptance"
        assert callable(spec.resolve_main())

    def test_the_json_report_names_the_properties_and_findings(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = cli_acceptance.main(["--skip-benchmarks", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert exit_code == 0
        assert payload["ok"] is True
        assert payload["findings"] == []
        assert [entry["code"] for entry in payload["properties"]] == [code for code, _ in LINT_PROPERTIES]
        assert payload["frames"] == len(GALLERY_SAMPLES) * len(GALLERY_WIDTHS) * len(COLOR_LANES)

    def test_a_lint_finding_makes_the_command_exit_non_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Anti-vacuity: the exit code is derived from findings, not hard-coded."""
        monkeypatch.setattr(
            cli_acceptance,
            "lint_findings",
            lambda *args, **kwargs: (Finding("no_traceback", "synthetic", "injected"),),
        )
        assert cli_acceptance.main(["--skip-benchmarks"]) == 1

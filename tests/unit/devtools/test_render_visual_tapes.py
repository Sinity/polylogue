from __future__ import annotations

from pathlib import Path

from devtools import render_visual_tapes


def test_committed_tapes_match_generated_specs_on_head() -> None:
    """The committed tapes under docs/examples/visual-tapes/ must always equal
    what the current DEFAULT_TAPE_SPECS generate -- a regression here means a
    spec changed (e.g. gained a transcript step) without regenerating the
    committed artifact, exactly the drift polylogue-3tl.17 exists to catch."""
    from devtools.visual_vhs import default_tape_specs, generate_all_tapes

    tapes = generate_all_tapes(default_tape_specs())
    drift = render_visual_tapes.committed_tape_drift(tapes)
    assert drift == {}


def test_committed_tape_drift_flags_a_seeded_content_change(tmp_path: Path) -> None:
    """Seeded failure: a generated tape whose content diverges from the
    committed file must be reported, not silently accepted."""
    committed_dir = tmp_path / "committed"
    committed_dir.mkdir()
    (committed_dir / "demo-tour.tape").write_text("Type demo-tour-old\n", encoding="utf-8")

    tapes = {"demo-tour": "Type demo-tour-new\n"}
    drift = render_visual_tapes.committed_tape_drift(tapes, committed_dir=committed_dir)

    assert set(drift) == {"demo-tour"}
    committed_text, generated_text = drift["demo-tour"]
    assert committed_text == "Type demo-tour-old\n"
    assert generated_text == "Type demo-tour-new\n"


def test_committed_tape_drift_flags_a_missing_committed_file(tmp_path: Path) -> None:
    """Seeded failure: a spec with no committed counterpart at all must be
    reported (committed=None), not skipped -- a brand-new spec ships without
    its evidence artifact otherwise."""
    committed_dir = tmp_path / "committed"
    committed_dir.mkdir()

    tapes = {"new-spec": "Type something\n"}
    drift = render_visual_tapes.committed_tape_drift(tapes, committed_dir=committed_dir)

    assert set(drift) == {"new-spec"}
    committed_text, _generated_text = drift["new-spec"]
    assert committed_text is None


def test_committed_tape_drift_ignores_matching_content(tmp_path: Path) -> None:
    committed_dir = tmp_path / "committed"
    committed_dir.mkdir()
    (committed_dir / "demo-tour.tape").write_text("Type same\n", encoding="utf-8")

    tapes = {"demo-tour": "Type same\n"}
    drift = render_visual_tapes.committed_tape_drift(tapes, committed_dir=committed_dir)

    assert drift == {}

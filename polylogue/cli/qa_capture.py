"""Optional QA capture helpers."""

from __future__ import annotations


def run_vhs_capture(env, showcase_result, json_output: bool) -> None:
    """Run VHS tape captures if available."""
    try:
        from polylogue.showcase.vhs import (
            check_vhs_available,
            generate_all_tapes,
        )
        from polylogue.showcase.vhs import (
            run_vhs_capture as _capture_tape,
        )
    except ImportError:
        return

    output_dir = showcase_result.output_dir
    if output_dir is None:
        return

    tapes_dir = output_dir / "tapes"
    captures_dir = output_dir / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    exercises = [entry.exercise for entry in showcase_result.results]
    tapes = generate_all_tapes(exercises, output_dir=tapes_dir)

    if check_vhs_available():
        for name in tapes:
            tape_path = tapes_dir / f"{name}.tape"
            gif_path = captures_dir / f"{name}.gif"
            ok = _capture_tape(tape_path, gif_path)
            if not json_output:
                status = "ok" if ok else "FAILED"
                env.ui.console.print(f"  VHS {name}: {status}")
    elif not json_output:
        env.ui.console.print(
            f"VHS binary not found — {len(tapes)} tape(s) generated in {tapes_dir} but not recorded"
        )


__all__ = ["run_vhs_capture"]

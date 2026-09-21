"""The measurement half of the baseline story (polylogue-cjyfw).

Static gates already commit their evidence; measurement evidence had no
owner, so a benchmark that built a complete typed receipt printed it and the
number was gone with the terminal scrollback. These prove the route that
gives it a home, and the rule that keeps a committed number honest:
**explained, not ratcheted**.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.command_catalog import COMMAND_SPECS
from devtools.measurement_receipts import (
    BASELINE_DIR,
    BASELINE_FORMAT,
    RECEIPT_DIR_ENV,
    MeasurementBaselineError,
    committed_baselines,
    emit_receipt,
    host_fingerprint,
    load_baseline,
    main,
    measurement_movement,
    receipt_dir,
)

_MEASUREMENT = {
    "receipt": {"arm": "sealed", "resources": {"elapsed_seconds": 12.5, "storage_bytes": 4096}},
    "verdict": {"conclusion": "single-arm-observation"},
}


def _emit(root: Path, name: str, measurement: object, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(RECEIPT_DIR_ENV, str(root / "observations"))
    return emit_receipt(name, measurement, root=root)


def _record(root: Path, receipt: Path, monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr("devtools.measurement_receipts.repo_root", lambda: root)
    return main(["--record", str(receipt), *argv])


def test_receipt_dir_prefers_the_declared_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A runner that owns its output tree must not be forced into the checkout.

    Anti-vacuity: ignore ``RECEIPT_DIR_ENV`` in ``receipt_dir`` and the first
    assertion goes red; hard-code the override and the second does.
    """
    monkeypatch.setenv(RECEIPT_DIR_ENV, str(tmp_path / "elsewhere"))
    assert receipt_dir(root=tmp_path) == tmp_path / "elsewhere"

    monkeypatch.delenv(RECEIPT_DIR_ENV)
    assert receipt_dir(root=tmp_path) == tmp_path / ".cache" / "measurements"


def test_emit_writes_a_named_reloadable_observation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The emitted file is the whole receipt, not a rendering of it.

    Anti-vacuity: drop ``measurement`` from the emitted payload, or stringify
    it, and the round-trip equality goes red.
    """
    path = _emit(tmp_path, "arm", _MEASUREMENT, monkeypatch)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"] == "arm"
    assert payload["measurement"] == _MEASUREMENT
    assert set(payload["host"]) == set(host_fingerprint())


def test_host_fingerprint_carries_no_host_identifier() -> None:
    """A committed file is public; the machine's name is not measurement evidence.

    Anti-vacuity: add ``platform.node()`` or the kernel release to
    ``host_fingerprint`` and this goes red.
    """
    fingerprint = host_fingerprint()

    assert "cpu_count" in fingerprint
    assert not {"hostname", "node", "release", "platform"} & set(fingerprint)


def test_movement_names_the_nested_leaf_that_changed() -> None:
    """A "the receipt changed" verdict cannot be explained; a named leaf can.

    Anti-vacuity: compare only top-level keys and the reported path collapses
    to ``receipt``, so the exact-path assertion goes red.
    """
    after = json.loads(json.dumps(_MEASUREMENT))
    after["receipt"]["resources"]["elapsed_seconds"] = 19.0

    movements = measurement_movement(_MEASUREMENT, after)

    assert [movement.path for movement in movements] == ["receipt.resources.elapsed_seconds"]
    assert movements[0].before == 12.5
    assert movements[0].after == 19.0


def test_movement_reports_a_dropped_and_an_added_leaf() -> None:
    """A field that vanished is a movement too, not an unchanged baseline.

    Anti-vacuity: iterate only the observed payload's keys and the dropped
    leaf disappears from the report, reddening this.
    """
    after = {"receipt": {"arm": "sealed", "resources": {"elapsed_seconds": 12.5}}, "note": "new"}

    paths = {movement.path for movement in measurement_movement(_MEASUREMENT, after)}

    assert "receipt.resources.storage_bytes" in paths
    assert "note" in paths


def test_first_record_refuses_without_a_stated_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A committed number nobody explained is a number nobody can re-judge.

    Anti-vacuity: default ``--reason`` to a placeholder and the refusal exit
    code goes to 0, reddening this.
    """
    receipt = _emit(tmp_path, "arm", _MEASUREMENT, monkeypatch)

    assert _record(tmp_path, receipt, monkeypatch) == 1
    assert not (tmp_path / BASELINE_DIR / "arm.json").exists()
    assert "refused" in capsys.readouterr().err


def test_recording_commits_the_receipt_with_its_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed artifact is the receipt plus why it stands.

    Anti-vacuity: write the reason to a side file instead of the baseline and
    the format/reason assertions go red.
    """
    receipt = _emit(tmp_path, "arm", _MEASUREMENT, monkeypatch)

    assert _record(tmp_path, receipt, monkeypatch, "--reason", "first sealed arm") == 0

    committed = load_baseline(tmp_path / BASELINE_DIR / "arm.json")
    assert committed is not None
    assert committed["format"] == BASELINE_FORMAT
    assert committed["reason"] == "first sealed arm"
    assert committed["measurement"] == _MEASUREMENT
    assert committed["supersedes"] is None


def test_an_unchanged_measurement_needs_no_new_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-running a benchmark that reproduced its numbers is not a movement.

    Anti-vacuity: always demand ``--reason`` and this exits 1 instead of 0.
    """
    receipt = _emit(tmp_path, "arm", _MEASUREMENT, monkeypatch)
    assert _record(tmp_path, receipt, monkeypatch, "--reason", "first sealed arm") == 0

    assert _record(tmp_path, receipt, monkeypatch) == 0
    assert "unchanged" in capsys.readouterr().out


def test_a_moved_number_refuses_until_the_move_is_explained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Explained-not-ratcheted: the move is allowed, the silence is not.

    A measurement legitimately moves with the host, so this route must not
    refuse the direction the way a static ratchet does -- it must refuse the
    *unexplained* overwrite, and name the leaf that moved.

    Anti-vacuity: accept a record whose leaves differ without ``--reason``
    and the refusal assertions go red; refuse the explained record and the
    ``moved`` assertions do.
    """
    first = _emit(tmp_path, "arm", _MEASUREMENT, monkeypatch)
    assert _record(tmp_path, first, monkeypatch, "--reason", "first sealed arm") == 0
    capsys.readouterr()

    slower = json.loads(json.dumps(_MEASUREMENT))
    slower["receipt"]["resources"]["elapsed_seconds"] = 31.0
    moved = _emit(tmp_path, "arm", slower, monkeypatch)

    assert _record(tmp_path, moved, monkeypatch) == 1
    refusal = capsys.readouterr().err
    assert "receipt.resources.elapsed_seconds" in refusal
    assert load_baseline(tmp_path / BASELINE_DIR / "arm.json")["measurement"] == _MEASUREMENT  # type: ignore[index]

    assert _record(tmp_path, moved, monkeypatch, "--reason", "measured under a loaded host") == 0
    committed = load_baseline(tmp_path / BASELINE_DIR / "arm.json")
    assert committed is not None
    assert committed["measurement"] == slower
    assert committed["supersedes"]["reason"] == "first sealed arm"


def test_a_file_without_the_declared_format_is_refused(tmp_path: Path) -> None:
    """A stray JSON file in the baseline home must not read as a baseline.

    Anti-vacuity: drop the format check in ``load_baseline`` and this goes red.
    """
    stray = tmp_path / "stray.json"
    stray.write_text('{"elapsed": 1}\n', encoding="utf-8")

    with pytest.raises(MeasurementBaselineError, match=BASELINE_FORMAT):
        load_baseline(stray)


def test_the_route_is_a_declared_bench_subcommand() -> None:
    """The command surface is generated, and the twelve-root fold stays folded.

    Anti-vacuity: rename the spec, drop it from ``COMMAND_SPECS``, or promote
    it to a thirteenth root verb and this goes red.
    """
    spec = next(spec for spec in COMMAND_SPECS if spec.name == "bench baseline")

    assert spec.command_path == ("bench", "baseline")
    assert spec.module == "devtools.measurement_receipts"
    assert callable(spec.resolve_main())


def test_the_reserved_home_holds_a_real_committed_measurement() -> None:
    """The directory reserved for measurement evidence actually holds some.

    This is polylogue-cjyfw's own anti-vacuity condition: before the route
    existed, ``tests/benchmarks/baselines/`` held a lone ``.gitkeep`` and a
    grep of tracked JSON for the arm names returned nothing. Deleting the
    committed baseline makes this red.
    """
    baselines = committed_baselines()

    assert baselines, "tests/benchmarks/baselines/ holds no committed measurement"
    for payload in baselines:
        assert payload["reason"].strip()
        assert payload["measurement"]
        assert payload["measurement_digest"]

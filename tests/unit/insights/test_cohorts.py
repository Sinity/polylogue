from __future__ import annotations

import weakref
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.analysis.cohorts import (
    CohortCandidate,
    CohortSpec,
    compare_cohort_manifests,
    compile_cohort_manifest,
)


def _candidate(
    ref: str,
    *,
    repo: str = "polylogue",
    model: str = "fable",
    day: str = "2026-07-10",
    template: str | None = None,
    excluded: str | None = None,
) -> CohortCandidate:
    return CohortCandidate(
        object_ref=ref,
        dimensions={"repo": repo, "model": model, "day": day},
        template_key=template,
        exclusion_reason=excluded,
    )


def test_manifest_is_byte_stable_across_input_order_and_general_object_refs() -> None:
    spec = CohortSpec(
        population_query="assertions where kind:decision",
        archive_cursor="index:24:cursor-a",
        seed="stable-seed",
        requested_size=3,
        strata=("repo", "model"),
    )
    candidates = [
        _candidate("assertion:z", repo="sinex", model="opus"),
        _candidate("assertion:a"),
        _candidate("assertion:c", repo="sinex", model="opus"),
        _candidate("assertion:b"),
    ]

    first = compile_cohort_manifest(spec, candidates)
    second = compile_cohort_manifest(spec, list(reversed(candidates)))

    assert first.to_json() == second.to_json()
    assert first.selected_refs == second.selected_refs
    assert first.population_count == 4
    assert {dict(item.key)["repo"] for item in first.stratum_counts} == {"polylogue", "sinex"}


def test_manifest_records_exclusions_shortfall_and_template_sensitivity() -> None:
    spec = CohortSpec(
        population_query="delegations where mapping_state:resolved",
        archive_cursor="index:24:cursor-a",
        seed="stable-seed",
        requested_size=4,
        strata=("repo", "day", "model"),
        exact_template_cap=1,
    )
    manifest = compile_cohort_manifest(
        spec,
        [
            _candidate("delegation:one", template="same"),
            _candidate("delegation:two", template="same"),
            _candidate("delegation:three", template="other", repo="sinex"),
            _candidate("delegation:excluded", excluded="edge_only"),
        ],
    )

    assert len(manifest.selected_refs) == 2
    assert manifest.shortfall == 2
    assert manifest.excluded_counts == (("edge_only", 1),)
    selected_templates = {
        candidate.template_key
        for candidate in [
            _candidate("delegation:one", template="same"),
            _candidate("delegation:two", template="same"),
            _candidate("delegation:three", template="other", repo="sinex"),
        ]
        if candidate.object_ref in manifest.selected_refs
    }
    assert selected_templates == {"same", "other"}
    assert dict(manifest.template_counts) == {"other": 1, "same": 2, "unknown": 1}


def test_manifest_drift_is_explicit_for_population_and_cursor_changes() -> None:
    candidates = [_candidate("message:a"), _candidate("message:b")]
    initial = compile_cohort_manifest(CohortSpec("messages where role:user", "index:24:a", "seed", 2), candidates)
    changed_population = compile_cohort_manifest(
        CohortSpec("messages where role:user", "index:24:a", "seed", 2),
        [*candidates, _candidate("message:c")],
    )
    changed_cursor = compile_cohort_manifest(
        CohortSpec("messages where role:user", "index:24:b", "seed", 2), candidates
    )

    population_drift = compare_cohort_manifests(initial, changed_population)
    cursor_drift = compare_cohort_manifests(initial, changed_cursor)

    assert population_drift.changed is True
    assert changed_population.manifest_id != initial.manifest_id
    assert cursor_drift.changed is True
    assert cursor_drift.cursor_changed is True
    assert changed_cursor.manifest_id != initial.manifest_id


def test_cohort_cancellation_reaches_population_and_rank_work() -> None:
    import pytest

    candidates = [CohortCandidate(f"session:neutral-{index}") for index in range(100)]
    calls = 0

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls == 150:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        compile_cohort_manifest(CohortSpec("neutral", "original-frame", "seed", 1), candidates, checkpoint=checkpoint)
    assert calls == 150


def test_spooled_cohorts_preserve_original_full_manifest_bytes() -> None:
    import json
    from pathlib import Path

    fixture = json.loads((Path(__file__).parents[2] / "fixtures/cohort_manifest_equivalence.json").read_text())
    for case in fixture["cases"]:
        spec = CohortSpec(**{**case["spec"], "strata": tuple(case["spec"]["strata"])})
        candidates = (CohortCandidate(**row) for row in reversed(fixture["candidates"]))
        actual = compile_cohort_manifest(spec, candidates).to_json()
        assert actual == json.dumps(case["expected"], sort_keys=True, separators=(",", ":"))


def test_spooled_cohort_releases_population_operands_and_physical_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile
    import weakref
    from pathlib import Path

    import pytest

    original = tempfile.TemporaryDirectory
    directories: list[Path] = []

    def track_directory(*, prefix: str, dir: Path | None = None) -> tempfile.TemporaryDirectory[str]:
        directory = original(prefix=prefix, dir=tmp_path if dir is None else dir)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(tempfile, "TemporaryDirectory", track_directory)
    refs: list[weakref.ReferenceType[CohortCandidate]] = []

    def candidates() -> Iterator[CohortCandidate]:
        for index in range(1000):
            # Input memory must stay constant even when the sample has one row.
            assert sum(ref() is not None for ref in refs) <= 1
            candidate = CohortCandidate(f"session:neutral-{index}", template_key="same")
            refs.append(weakref.ref(candidate))
            yield candidate
            del candidate

    manifest = compile_cohort_manifest(CohortSpec("neutral", "original-frame", "seed", 1), candidates())
    assert manifest.population_count == 1000
    assert len(manifest.selected_refs) == 1
    assert all(ref() is None for ref in refs)
    assert directories and all(not directory.exists() for directory in directories)

    def cancelled() -> Iterator[CohortCandidate]:
        yield CohortCandidate("session:first")
        raise InterruptedError("original cancellation")

    with pytest.raises(InterruptedError):
        compile_cohort_manifest(CohortSpec("neutral", "frame", "seed", 1), cancelled())
    assert all(not directory.exists() for directory in directories)
    for invalid in ([CohortCandidate("")], [CohortCandidate("same"), CohortCandidate("same")]):
        with pytest.raises(ValueError):
            compile_cohort_manifest(CohortSpec("neutral", "frame", "seed", 1), iter(invalid))
    assert all(not directory.exists() for directory in directories)


def test_spooled_cohort_cancellation_after_input_exhaustion_closes_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    original = tempfile.TemporaryDirectory
    directories: list[Path] = []
    exhausted = False

    def track_directory(*, prefix: str, dir: Path | None = None) -> tempfile.TemporaryDirectory[str]:
        directory = original(prefix=prefix, dir=tmp_path if dir is None else dir)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(tempfile, "TemporaryDirectory", track_directory)

    def candidates() -> Iterator[CohortCandidate]:
        nonlocal exhausted
        for index in range(1000):
            yield CohortCandidate(f"session:neutral-{index}")
        exhausted = True

    def checkpoint() -> None:
        if exhausted:
            raise InterruptedError("cancelled after population acquisition")

    with pytest.raises(InterruptedError):
        compile_cohort_manifest(CohortSpec("neutral", "frame", "seed", 1), candidates(), checkpoint=checkpoint)
    assert exhausted
    assert directories and all(not directory.exists() for directory in directories)

from __future__ import annotations

from polylogue.insights.cohorts import (
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

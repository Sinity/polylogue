from __future__ import annotations

from devtools.validation_family_models import (
    ValidationLaneCompositeSpec,
    ValidationLaneFamily,
    ValidationLaneStageSpec,
    compile_validation_lane_families,
)


def test_validation_lane_family_compiles_composite_entries_with_family_metadata() -> None:
    family = ValidationLaneFamily(
        name="runtime-substrate",
        description="Runtime substrate family.",
        lanes=(
            ValidationLaneCompositeSpec(
                name="runtime-substrate-contracts",
                description="Local runtime-substrate contract lane.",
                timeout_s=120,
                members=("query-routing", "semantic-stack"),
            ),
            ValidationLaneCompositeSpec(
                name="runtime-substrate-hardening",
                description="Full runtime-substrate hardening lane.",
                timeout_s=240,
                members=("runtime-substrate-contracts", "runtime-substrate-live"),
            ),
        ),
    )

    entries = family.compile_entries()

    assert [entry.name for entry in entries] == [
        "runtime-substrate-contracts",
        "runtime-substrate-hardening",
    ]
    assert all(entry.family == "runtime-substrate" for entry in entries)
    assert entries[0].sub_lanes == ("query-routing", "semantic-stack")
    assert entries[0].origin == "authored.validation-lane.composite-family"


def test_compile_validation_lane_families_indexes_entries_by_name() -> None:
    compiled = compile_validation_lane_families(
        (
            ValidationLaneFamily(
                name="domain-read-model",
                description="Domain read-model family.",
                lanes=(
                    ValidationLaneCompositeSpec(
                        name="domain-read-model-live",
                        description="Live lane.",
                        timeout_s=180,
                        members=("live-insights-small",),
                    ),
                ),
            ),
        )
    )

    assert tuple(compiled) == ("domain-read-model-live",)
    assert compiled["domain-read-model-live"].family == "domain-read-model"


def test_validation_lane_family_from_stages_derives_stage_names_and_members() -> None:
    family = ValidationLaneFamily.from_stages(
        name="runtime-substrate",
        description="Runtime substrate family.",
        stages=(
            ValidationLaneStageSpec(
                suffix="contracts",
                description="Contracts lane.",
                timeout_s=120,
                members=("query-routing", "semantic-stack"),
            ),
            ValidationLaneStageSpec(
                suffix="hardening",
                description="Hardening lane.",
                timeout_s=240,
                member_stages=("contracts", "live"),
            ),
        ),
    )

    entries = family.compile_entries()

    assert [entry.name for entry in entries] == [
        "runtime-substrate-contracts",
        "runtime-substrate-hardening",
    ]
    assert entries[1].sub_lanes == (
        "runtime-substrate-contracts",
        "runtime-substrate-live",
    )

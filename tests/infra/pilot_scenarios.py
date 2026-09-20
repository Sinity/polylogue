"""Small semantic witnesses for the demand-driven resource pilot.

The workload-artifact manifest deliberately contains construction facts only.
These four witnesses therefore live beside the tests that own their meaning.
Each case has two wire representations and one separately authored expectation;
the representation normalizer is intentionally tiny and local rather than a
second scenario registry or a general-purpose harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PilotExpectation:
    """Independent semantic facts asserted by a pilot witness."""

    matched_witness: tuple[str, ...] = ()
    weighted_total: int | None = None
    window_members: tuple[str, ...] = ()
    local_native: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class PilotScenario:
    """One compact semantic witness with schema-shaped representations."""

    name: str
    seed: int
    representations: tuple[dict[str, Any], ...]
    expected: PilotExpectation


def same_witness_matching() -> PilotScenario:
    """Match one logical witness despite flat and nested wire shapes."""

    return PilotScenario(
        name="same-witness-matching",
        seed=17,
        representations=(
            {"witness": {"provider": "codex", "native": "n-17", "local": "l-17"}},
            {"record": {"source": "codex", "identity": {"native_id": "n-17", "local_id": "l-17"}}},
        ),
        expected=PilotExpectation(matched_witness=("codex", "n-17")),
    )


def multiplicity_sensitive_sum() -> PilotScenario:
    """Retain duplicate contributions when summing a sparse provider payload."""

    return PilotScenario(
        name="multiplicity-sensitive-sum",
        seed=23,
        representations=(
            {"amounts": [2, 2, 3]},
            {"items": [{"amount": 2}, {"amount": 2}, {"amount": 3}]},
        ),
        expected=PilotExpectation(weighted_total=7),
    )


def sparse_nested_windows() -> PilotScenario:
    """Preserve sparse, nested window membership and its declared order."""

    return PilotScenario(
        name="sparse-nested-windows",
        seed=31,
        representations=(
            {"windows": [{"start": 2, "end": 3}, {"start": 8, "end": 8}]},
            {"timeline": {"windows": [{"range": {"from": 2, "to": 3}}, {"range": {"from": 8, "to": 8}}]}},
        ),
        expected=PilotExpectation(window_members=("2", "3", "8")),
    )


def local_native_preservation() -> PilotScenario:
    """Keep local and provider-native identities as distinct facts."""

    return PilotScenario(
        name="local-native-preservation",
        seed=43,
        representations=(
            {"local_id": "archive:l-43", "native_id": "codex:n-43"},
            {"identity": {"local": "archive:l-43", "native": "codex:n-43"}},
        ),
        expected=PilotExpectation(local_native=(("archive:l-43", "codex:n-43"),)),
    )


def pilot_scenarios() -> tuple[PilotScenario, ...]:
    """Return the bounded pilot witnesses; no corpus-wide generation occurs."""

    return (
        same_witness_matching(),
        multiplicity_sensitive_sum(),
        sparse_nested_windows(),
        local_native_preservation(),
    )


def project_representation(case: PilotScenario, representation: dict[str, Any]) -> PilotExpectation:
    """Normalize one representation to facts without consulting storage or an oracle."""

    if case.name == "same-witness-matching":
        witness = representation.get("witness")
        if isinstance(witness, dict):
            return PilotExpectation(
                matched_witness=(str(witness["provider"]), str(witness["native"])),
            )
        record = representation["record"]
        identity = record["identity"]
        return PilotExpectation(matched_witness=(str(record["source"]), str(identity["native_id"])))

    if case.name == "multiplicity-sensitive-sum":
        amounts = representation.get("amounts")
        if amounts is None:
            amounts = [item["amount"] for item in representation["items"]]
        return PilotExpectation(weighted_total=sum(int(amount) for amount in amounts))

    if case.name == "sparse-nested-windows":
        windows = representation.get("windows")
        if windows is None:
            windows = representation["timeline"]["windows"]
            windows = [{"start": item["range"]["from"], "end": item["range"]["to"]} for item in windows]
        members = tuple(
            str(value) for window in windows for value in range(int(window["start"]), int(window["end"]) + 1)
        )
        return PilotExpectation(window_members=members)

    if case.name == "local-native-preservation":
        identity = representation.get("identity", representation)
        local = identity["local_id"] if "local_id" in identity else identity["local"]
        native = identity["native_id"] if "native_id" in identity else identity["native"]
        return PilotExpectation(local_native=((str(local), str(native)),))

    raise ValueError(f"unknown pilot scenario {case.name!r}")


__all__ = [
    "PilotExpectation",
    "PilotScenario",
    "local_native_preservation",
    "multiplicity_sensitive_sum",
    "pilot_scenarios",
    "project_representation",
    "same_witness_matching",
    "sparse_nested_windows",
]

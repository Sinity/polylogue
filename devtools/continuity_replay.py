"""Run a privacy-safe continuity oracle against recorded route observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - exercised by the script entry point
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polylogue.product.continuity_scenarios import continuity_scenario


def evaluate_replay(
    scenario_name: str,
    fixture: dict[str, object],
    observation: dict[str, object],
) -> dict[str, object]:
    """Classify an observed black-box walk against a fixture-derived oracle."""
    scenario = continuity_scenario(scenario_name)
    refs = observation.get("refs", ())
    if not isinstance(refs, list):
        raise ValueError("observation.refs must be a JSON list")
    calls = observation.get("calls", 0)
    if not isinstance(calls, int):
        raise ValueError("observation.calls must be an integer")
    failure = observation.get("failure")
    failure_value = str(failure) if failure is not None else None
    page_bytes = observation.get("page_bytes")
    if page_bytes is not None and not isinstance(page_bytes, int):
        raise ValueError("observation.page_bytes must be an integer")
    cancel_seconds = observation.get("cancel_seconds")
    if cancel_seconds is not None and not isinstance(cancel_seconds, (int, float)):
        raise ValueError("observation.cancel_seconds must be numeric")
    result = scenario.classify(
        fixture,
        observed_refs=[str(ref) for ref in refs],
        calls=calls,
        observed_failure=failure_value if failure_value in scenario.failure_taxonomy else None,  # type: ignore[arg-type]
        page_bytes=page_bytes,
        cancel_seconds=float(cancel_seconds) if cancel_seconds is not None else None,
        continuation_state_lost=bool(observation.get("continuation_state_lost", False)),
        non_progressing_continuation=bool(observation.get("non_progressing_continuation", False)),
    )
    expected_record = scenario.oracle_record(fixture)
    return {
        "scenario": scenario.scenario_id,
        "classification": result,
        "expected_refs": list(scenario.oracle(fixture)),
        "expected_answer": {str(key): value for key, value in expected_record.items() if key != "refs"},
        "observed_refs": refs,
        "calls": calls,
        "budget": {
            "max_calls": scenario.budget.max_calls,
            "max_page_bytes": scenario.budget.max_page_bytes,
            "max_cancel_seconds": scenario.budget.max_cancel_seconds,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario")
    parser.add_argument("fixture", type=Path)
    parser.add_argument("observation", type=Path)
    args = parser.parse_args(argv)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    observation = json.loads(args.observation.read_text(encoding="utf-8"))
    result = evaluate_replay(args.scenario, fixture, observation)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["classification"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate_replay", "main"]

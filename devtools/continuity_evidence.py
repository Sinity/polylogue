"""Replay continuity and query-discovery behavior together.

The continuity scenario suite exercises seven workflows plus the
parallel-agent incident variant against a deterministic synthetic archive.
The executable query-discovery catalog describes the plans a cold client can
formulate. This module runs those two existing capabilities together without
reimplementing them.

This module is that wiring, not a fourth reimplementation:

- :func:`check_discovery_coverage` reuses the real
  :data:`polylogue.archive.query.discovery.QUERY_DISCOVERY_EXAMPLES` catalog
  to prove every ``query``-tool route step any continuity scenario executes
  has a declared positive example of the same unit-source/route shape --
  i.e. a cold model relying on discovery alone could have found that plan
  family, not just executed it once the runner already knew it.
- :func:`run_continuity_evidence` calls
  :func:`devtools.continuity_replay.replay_archive` unmodified against either
  a supplied archive root (an authorized live-scale replay) or a freshly
  seeded synthetic corpus (the default, privacy-safe CI lane), then combines
  both executable lanes into one JSON artifact. :func:`redact_report`
  strips raw evidence prose from that
  artifact (keeping refs/hashes/counts) for the live-archive lane; the
  synthetic lane never touches private content so redaction there is a no-op
  proof of the same mechanism, not a load-bearing privacy boundary.

The report states whether it used a supplied live archive. It does not copy a
tracker item's acceptance prose into product output or infer tracker closure
from the three lane statuses.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, TextIO, cast

if __package__ in {None, ""}:  # pragma: no cover - exercised by the script entry point
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devtools.continuity_replay import replay_archive
from polylogue.archive.query.discovery import QUERY_DISCOVERY_EXAMPLES
from polylogue.core.json import JSONDocument, JSONValue, require_json_document
from polylogue.product.continuity_scenarios import CONTINUITY_SCENARIOS, ContinuityScenarioSpec, continuity_scenario
from tests.infra.continuity import load_continuity_catalog, seed_continuity_archive

# ── Discovery-coverage lane ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DiscoveryCoverageGap:
    """One continuity route step whose plan family has no discovery example."""

    scenario_id: str
    step_id: str
    plan_atom: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiscoveryCoverageReport:
    """Whether every continuity-scenario query plan is independently discoverable."""

    checked_steps: int
    covered_steps: int
    gaps: tuple[DiscoveryCoverageGap, ...]

    @property
    def status(self) -> Literal["pass", "fail"]:
        return "pass" if not self.gaps else "fail"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checked_steps": self.checked_steps,
            "covered_steps": self.covered_steps,
            "gaps": [gap.to_dict() for gap in self.gaps],
        }


def check_discovery_coverage(
    scenarios: Sequence[ContinuityScenarioSpec],
) -> DiscoveryCoverageReport:
    """Prove every ``query``-tool continuity route step is independently discoverable.

    A continuity scenario's route steps prove the runner *can execute* a
    plan; they say nothing about whether a cold model, given only the
    published query-discovery catalog (``archive/query/discovery.py``), could
    have *formulated* that same plan family without hidden knowledge. This
    cross-checks each ``query``-tool step's unit-source against
    ``QUERY_DISCOVERY_EXAMPLES`` -- the same catalog z9gh.3 generates MCP
    schemas/completions from -- so a shipped scenario whose plan family the
    discovery catalog does not teach shows up as a named gap rather than a
    silent success.
    """

    catalog_atoms = {f"query:{example.unit_source}" for example in QUERY_DISCOVERY_EXAMPLES if example.route == "query"}
    gaps: list[DiscoveryCoverageGap] = []
    checked = 0
    for scenario in scenarios:
        for step in scenario.route_steps:
            if step.tool != "query":
                continue
            checked += 1
            atom = step.plan_atom
            if atom not in catalog_atoms:
                gaps.append(
                    DiscoveryCoverageGap(
                        scenario_id=scenario.scenario_id,
                        step_id=step.step_id,
                        plan_atom=atom,
                        reason=f"no declared query-discovery example teaches {atom!r}",
                    )
                )
    return DiscoveryCoverageReport(checked_steps=checked, covered_steps=checked - len(gaps), gaps=tuple(gaps))


# ── Redaction ─────────────────────────────────────────────────────────

_REDACTABLE_KEYS = frozenset({"label", "claim_text", "reason", "response_sha256"})


def _redact_value(key: str, value: JSONValue) -> JSONValue:
    if key in _REDACTABLE_KEYS and isinstance(value, str) and value:
        return f"redacted:sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
    return value


def redact_report(document: JSONValue) -> JSONValue:
    """Strip raw evidence prose from a mandate report, keeping refs/counts/hashes.

    Recursively walks the report replacing any string value stored under a
    label/claim-text/reason-shaped key with a stable hash of itself. Refs,
    ids, statuses, and counts (everything the mandate AC actually needs
    cited) pass through unchanged -- only free-text prose that could carry
    private archive content is hashed.
    """

    if isinstance(document, dict):
        return {key: _redact_value(key, redact_report(value)) for key, value in document.items()}
    if isinstance(document, list):
        return [redact_report(item) for item in document]
    return document


# ── Orchestration ──────────────────────────────────────────────────────


async def run_continuity_evidence(
    *,
    archive_root: Path | None = None,
    scenario_names: Sequence[str] | None = None,
    redact: bool = True,
    keep_archive: bool = False,
) -> JSONDocument:
    """Run the continuity and discovery lanes as one artifact.

    When ``archive_root`` is ``None`` (the default), a fresh, privacy-safe
    synthetic continuity corpus is seeded and torn down automatically -- the
    CI/deterministic lane. Passing an authorized live archive root runs the
    identical mechanism against it (the live-scale lane); pair that with
    ``redact=True`` (the default) so evidence prose never leaves this
    process's stdout/artifact file.
    """

    started_ns = time.perf_counter_ns()
    catalog = load_continuity_catalog()
    live_archive = archive_root is not None
    workdir: TemporaryDirectory[str] | None = None

    resolved_root: Path
    if archive_root is None:
        workdir = TemporaryDirectory(prefix="polylogue-continuity-evidence-")
        resolved_root = Path(workdir.name) / "archive"
        seed_continuity_archive(resolved_root, catalog=catalog)
    else:
        resolved_root = archive_root

    try:
        continuity_report = await replay_archive(resolved_root, catalog, scenario_names=scenario_names)
    finally:
        if workdir is not None and not keep_archive:
            workdir.cleanup()

    scenarios = (
        CONTINUITY_SCENARIOS if scenario_names is None else tuple(continuity_scenario(name) for name in scenario_names)
    )
    discovery_report = check_discovery_coverage(scenarios)
    overall_status: Literal["pass", "fail"] = (
        "pass" if continuity_report.get("status") == "pass" and discovery_report.status == "pass" else "fail"
    )
    report: dict[str, object] = {
        "schema_version": 3,
        "live_archive": live_archive,
        "archive_root": str(resolved_root.resolve()) if keep_archive or live_archive else None,
        "elapsed_ms": round((time.perf_counter_ns() - started_ns) / 1_000_000, 3),
        "status": overall_status,
        "continuity": continuity_report,
        "discovery_coverage": discovery_report.to_dict(),
    }
    document = require_json_document(report, context="continuity evidence report")
    return cast(JSONDocument, redact_report(document)) if redact else document


def _scenario_names(value: str) -> tuple[str, ...] | None:
    if value == "all":
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Authorized live archive to replay against; omit for the default synthetic CI lane.",
    )
    parser.add_argument("--scenario", default="all", help="all or a comma-separated scenario id list")
    parser.add_argument("--no-redact", action="store_true", help="Disable evidence redaction (CI/synthetic lane only)")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = asyncio.run(
        run_continuity_evidence(
            archive_root=args.archive_root,
            scenario_names=_scenario_names(args.scenario),
            redact=not args.no_redact,
            keep_archive=args.keep_archive,
        )
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    out = stdout or sys.stdout
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, file=out)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DiscoveryCoverageGap",
    "DiscoveryCoverageReport",
    "check_discovery_coverage",
    "main",
    "redact_report",
    "run_continuity_evidence",
]

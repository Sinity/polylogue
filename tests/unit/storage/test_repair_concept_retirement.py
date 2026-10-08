"""Repair is no longer a product concept (polylogue-6kur).

The generic repair framework was a plan/actuator/executability layer over three
historical incident strategies. Its apply half was deleted first, which left the
worse residue: a census that classified accepted heads into "executable"
repair plans, named an actuator for each, and published durable blockers whose
promised remedy no code could run. polylogue-6kur deletes the promise instead
of re-plumbing it.

What survives is a read-only frontier census whose every non-current state is
either an explicit retryable obligation that ordinary acquisition discharges,
or a typed permanent refusal. This module is that boundary's ratchet: each
assertion names the exact production shape whose reintroduction turns it red.

The operator-adjudication route that once survived here
(``ops maintenance operation-recovery --confirm`` and MCP
``recovery_adjudicate``) is deleted too: an interrupted mutation's outcome must
be decidable from durable evidence, not operator testimony (polylogue-aw070).
"""

from __future__ import annotations

import ast
import importlib
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.maintenance.declarations import MAINTENANCE_COMMAND_DECLARATIONS
from polylogue.storage import raw_reconciler as raw_reconciler_mod
from polylogue.storage.raw_reconciler import RawAuthorityFrontierState
from polylogue.storage.sqlite.archive_tiers import raw_admission, source_write
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root

#: Modules and symbols the generic repair framework owned. Each entry is a real
#: definition at the parent commit, not a hypothetical.
RETIRED_MODULES = ("polylogue.storage.raw_convergence",)

RETIRED_RECONCILER_SYMBOLS = (
    "RawAuthorityFrontierCensus",
    "inspect_raw_authority_frontier",
    "_reconcile_frontier_obligations",
    # The actuator taxonomy and its executability gate: every member named a
    # remedy, and no dispatcher existed to run one.
    "RawAuthorityActuator",
    "_APPLY_DISPATCHED_ACTUATORS",
    "_EXECUTABLE_STATES",
    # The bridge that asked the three historical incident inspectors for
    # proofs and turned them into executable frontier plans.
    "_StrategyOverride",
    "_strategy_overrides",
    "_quarantine_override_key",
    "_browser_strategy_witness",
    "_quarantine_strategy_witness",
    "_duplicate_strategy_witness",
    "_duplicate_alias_siblings",
    # The operator-judgment promotion loop: an accepted judgment promoted a
    # blocked plan into an "executable successor" nothing executed.
    "_record_judgment_candidate",
    "_apply_judgment_dispositions",
)

RETIRED_FRONTIER_STATES = (
    "SAFELY_REKEYABLE",
    "DUPLICATE_ALIAS",
    "CONFLICTING_AUTHORITY_NEEDS_JUDGMENT",
)

#: The durable source-tier write that existed solely as the browser-origin
#: copy-forward repair's named exemption from ``admit_raw_observation``.
RETIRED_SOURCE_WRITERS = ("ReconstructedRawRow", "insert_reconstructed_raw_row")

#: Anti-vacuity partner for every absence above: the read-only census and its
#: durable obligation ledger are RETAINED, so a change that deleted the whole
#: raw-authority family instead of the repair framework fails here.
RETAINED_RECONCILER_SYMBOLS = (
    "RawAuthorityFrontierState",
    "RawAuthorityFrontierItem",
    "_classify_frontier_row",
    "_verify_blob_bytes",
    "_frontier_blocker_identity",
)

_PRODUCT_ROOTS = ("polylogue",)


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[])


def _write_codex_raw(root: Path, *, native_id: str, source_path: str) -> str:
    payload = (
        f'{{"type":"session_meta","payload":{{"id":"{native_id}"}}}}\n'
        f'{{"type":"response_item","payload":{{"type":"message","id":"m-1",'
        f'"role":"user","content":[{{"type":"input_text","text":"authored content"}}]}}}}\n'
    ).encode()
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path=source_path,
            canonical_source_path=source_path,
            acquired_at_ms=1,
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _product_sources() -> tuple[Path, ...]:
    root = _repo_root()
    return tuple(sorted(path for package in _PRODUCT_ROOTS for path in (root / package).rglob("*.py")))


@pytest.mark.parametrize("module_name", RETIRED_MODULES)
def test_no_product_module_imports_a_retired_repair_substrate(module_name: str) -> None:
    """No import edge reaches a deleted repair substrate, at any nesting depth.

    An ``import`` inside a function body is the exact shape the frontier census
    used to reach ``raw_convergence``, so a top-level-only check would not have
    caught it. Parse instead of grepping: a docstring naming the old module is
    prose, not an authority.

    Anti-vacuity: restoring ``from polylogue.storage.raw_convergence import
    inspect_quarantined_accepted_raws`` inside any function in the tree fails
    this, and so does recreating the module itself.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)

    offenders: list[str] = []
    for path in _product_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == module_name or name.startswith(f"{module_name}.") for name in names):
                offenders.append(f"{path.relative_to(_repo_root())}:{node.lineno}")
    assert offenders == []


@pytest.mark.parametrize("symbol", RETIRED_RECONCILER_SYMBOLS)
def test_frontier_census_declares_no_repair_actuator_machinery(symbol: str) -> None:
    """The census classifies evidence; it owns no remedy vocabulary.

    Anti-vacuity: re-adding ``RawAuthorityActuator`` (or any strategy-override
    helper) to ``storage/raw_reconciler.py`` fails this parametrization, and
    the retained-symbol test below keeps it from passing by deleting the census.
    """
    assert not hasattr(raw_reconciler_mod, symbol)


@pytest.mark.parametrize("symbol", RETAINED_RECONCILER_SYMBOLS)
def test_frontier_census_and_its_obligation_ledger_are_retained(symbol: str) -> None:
    """Anti-vacuity partner: deleting the census wholesale is not the outcome."""
    assert hasattr(raw_reconciler_mod, symbol)


@pytest.mark.parametrize("state", RETIRED_FRONTIER_STATES)
def test_no_frontier_state_promises_a_remedy(state: str) -> None:
    """Every surviving state is a fact about evidence, not a queued repair.

    ``safely_rekeyable`` and ``duplicate_alias`` existed only as the two
    executable states the deleted strategies produced;
    ``conflicting_authority_needs_judgment`` existed only to mint a judgment
    assertion whose acceptance promoted a plan nothing applied.

    Anti-vacuity: re-adding any of the three to ``RawAuthorityFrontierState``
    fails immediately.
    """
    assert not hasattr(RawAuthorityFrontierState, state)


@pytest.mark.parametrize("symbol", RETIRED_SOURCE_WRITERS)
def test_source_tier_has_no_repair_exemption_from_raw_admission(symbol: str) -> None:
    """Every ``raw_sessions`` row in a live archive comes through admission.

    ``insert_reconstructed_raw_row`` was the one declared exemption, and its
    only caller was the browser-origin copy-forward repair. With that strategy
    gone the exemption is an unreferenced durable-source write bypass.

    Anti-vacuity: restoring either symbol to
    ``storage/sqlite/archive_tiers/source_write.py`` fails this, while the
    assertion below keeps the real admission chokepoint in place.
    """
    assert not hasattr(source_write, symbol)
    assert hasattr(raw_admission, "admit_raw_observation")


def test_maintenance_surface_offers_no_generic_repair_command() -> None:
    """No declared CLI command is a generic repair/cleanup/doctor umbrella.

    The maintenance family is the only registered operator surface that could
    host one; every member is checked against the live Click tree by
    ``tests/unit/devtools/test_declaration_binding_registries.py``, so this reads the same declarations
    the CLI actually registers rather than a doc.

    Anti-vacuity: declaring ``_command("repair", ...)`` (or ``doctor`` /
    ``cleanup`` / ``fix``) in ``polylogue/maintenance/declarations.py`` fails
    this, and the retained blob/embedding commands below keep it from passing
    by emptying the family.
    """
    forbidden = ("repair", "doctor", "cleanup", "fix-", "break-glass")
    offenders = [
        declaration.cli_name
        for declaration in MAINTENANCE_COMMAND_DECLARATIONS
        if any(token in declaration.cli_name for token in forbidden)
    ]
    assert offenders == []
    declared = {declaration.cli_name for declaration in MAINTENANCE_COMMAND_DECLARATIONS}
    assert {"blob-gc", "verify-archive", "raw-authority-frontier"} <= declared


@pytest.mark.asyncio
@pytest.mark.timeout(0)
async def test_frontier_census_reports_a_blocked_head_without_promising_a_remedy(tmp_path: Path) -> None:
    from contextlib import closing
    from dataclasses import asdict

    from polylogue.storage.frontier_inspection import inspect_prepared_raw_authority_frontier
    from tests.infra.archive_templates import run_archive_fixture_write
    from tests.infra.live_ingest import prepared_live_convergence_owner

    def acquire() -> str:
        bootstrap_archive_root(tmp_path)
        return _write_codex_raw(tmp_path, native_id="blocked-head", source_path="blocked.jsonl")

    raw_id = await run_archive_fixture_write(tmp_path, acquire)
    async with prepared_live_convergence_owner(tmp_path) as owner:
        receipts = (await owner.ingest_retained_raw_ids((raw_id,))).require_complete()
        assert sum(len(receipt.written_session_ids) for receipt in receipts) == 1

        def quarantine() -> None:
            with closing(sqlite3.connect(tmp_path / "source.db")) as conn, conn:
                conn.execute("UPDATE raw_sessions SET revision_authority='quarantined' WHERE raw_id=?", (raw_id,))

        await run_archive_fixture_write(tmp_path, quarantine)
        outcome = await owner.run_convergence_sync(
            "fixture.frontier.blocked",
            inspect_prepared_raw_authority_frontier,
            tmp_path,
            input_demand=owner._compute_adapter.amend_current_input_demand,
            check_physical_dependencies=True,
        )
    assert not outcome.healthy and outcome.blocking_head_checks == 1
    payload = asdict(outcome)
    assert "executable_plan_count" not in payload and "actuator" not in payload
    with closing(sqlite3.connect(tmp_path / "source.db")) as source_conn:
        rows = source_conn.execute(
            "SELECT expected_json,observed_json,resolved_at_ms FROM raw_authority_blockers WHERE resolved_at_ms IS NULL"
        ).fetchall()
    assert len(rows) == 1 and rows[0][2] is None
    expected, observed = json.loads(rows[0][0]), json.loads(rows[0][1])
    assert expected["input_raw_ids"] == [raw_id]
    assert observed["state"] == "unresolved_provenance"
    assert "actuator" not in expected["authority_witness"]
    assert "actuator" not in observed and "strategy_witness" not in expected["authority_witness"]


def test_no_product_module_is_named_as_a_repair_substrate() -> None:
    """AC7, structural half: no module in the product tree is named for repair.

    Module names are the part of "named or described as a generic repair
    substrate" that a test can decide. ``storage/repair.py`` (7,534 lines) and
    ``storage/raw_convergence.py`` (2,915) were both real modules under those
    names; this keeps the next one from landing quietly.

    The surviving occurrences of the word live inside modules with other names
    and other jobs -- FTS index convergence, derived-row convergence inside the
    write transaction, the declaration kernel's ``repair_command`` remedy
    pointer, and the ``contradicted_then_repaired`` behavioural measurement.
    None of them is a substrate, and none is checkable by name.

    Anti-vacuity: adding ``polylogue/storage/repair.py`` (or ``doctor.py`` /
    ``cleanup.py`` / any ``*_repair.py``) fails this.
    """
    forbidden = ("repair", "doctor", "cleanup")
    offenders = [
        str(path.relative_to(_repo_root()))
        for path in _product_sources()
        if any(token in path.stem.lower() for token in forbidden)
    ]
    assert offenders == []

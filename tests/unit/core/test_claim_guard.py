"""Unit tests for claim-guard readiness behavior."""

from __future__ import annotations

from polylogue.readiness.claim_guard import ClaimGuard, DerivedDomainReadiness, derive_claim_guard


def _base_kwargs() -> dict[str, object]:
    return {
        "archive_schema_ready": True,
        "schema_mismatches": (),
        "missing_tiers": (),
        "derived_domains": (
            DerivedDomainReadiness("raw_materialization", True, "ready"),
            DerivedDomainReadiness("raw_frontier_integrity", True, "ready"),
            DerivedDomainReadiness("session_profiles", True, "ready"),
            DerivedDomainReadiness("fts", True, "ready"),
        ),
        "search_ready": True,
        "search_summary": "ready",
        "active_writer": False,
        "active_writer_summary": "",
    }


def test_fully_ready_archive_claims_all_four_states() -> None:
    guard = derive_claim_guard(**_base_kwargs())  # type: ignore[arg-type]
    assert isinstance(guard, ClaimGuard)
    payload = guard.to_dict()
    assert payload["openable"]["value"] is True
    assert payload["converged"]["value"] is True
    assert payload["search_ready"]["value"] is True
    assert payload["perf_measurable"]["value"] is True
    assert all(entry["signal"] for entry in payload.values())


def test_schema_mismatch_blocks_openable_and_converged() -> None:
    kwargs = _base_kwargs()
    kwargs["archive_schema_ready"] = False
    kwargs["schema_mismatches"] = ["index"]
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["openable"]["value"] is False
    assert "index" in str(guard["openable"]["reason"])
    assert guard["converged"]["value"] is False
    assert "not openable" in str(guard["converged"]["reason"])


def test_missing_tiers_block_openable_with_named_tiers() -> None:
    kwargs = _base_kwargs()
    kwargs["archive_schema_ready"] = False
    kwargs["missing_tiers"] = ["user", "embeddings"]
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["openable"]["value"] is False
    assert "embeddings" in str(guard["openable"]["reason"])
    assert "user" in str(guard["openable"]["reason"])


def test_openable_but_not_converged_reports_raw_materialization_reason() -> None:
    kwargs = _base_kwargs()
    kwargs["derived_domains"] = (
        DerivedDomainReadiness("raw_materialization", False, "raw evidence pending materialization"),
    )
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["openable"]["value"] is True
    assert guard["converged"]["value"] is False
    assert guard["converged"]["reason"] == "raw evidence pending materialization"


def test_raw_frontier_integrity_not_ready_blocks_converged_with_reason() -> None:
    kwargs = _base_kwargs()
    kwargs["derived_domains"] = (
        DerivedDomainReadiness(
            "raw_frontier_integrity", False, "1 accepted append head(s) have a broken predecessor chain"
        ),
    )
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["openable"]["value"] is True
    assert guard["converged"]["value"] is False
    assert guard["converged"]["reason"] == "1 accepted append head(s) have a broken predecessor chain"


def test_raw_materialization_not_ready_takes_precedence_over_frontier_integrity() -> None:
    kwargs = _base_kwargs()
    kwargs["derived_domains"] = (
        DerivedDomainReadiness("raw_materialization", False, "raw evidence pending materialization"),
        DerivedDomainReadiness(
            "raw_frontier_integrity", False, "1 ingest cursor(s) committed past accepted raw material"
        ),
    )
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["converged"]["value"] is False
    assert guard["converged"]["reason"] == "raw evidence pending materialization"


def test_convergence_debt_is_not_a_claim_guard_input() -> None:
    kwargs = _base_kwargs()
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["converged"]["value"] is True
    assert "derived_domain_readiness" in str(guard["converged"]["signal"])


def test_profile_inspection_blocks_convergence() -> None:
    kwargs = _base_kwargs()
    kwargs["derived_domains"] = (DerivedDomainReadiness("session_profiles", False, "session profiles incomplete"),)
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["converged"]["value"] is False
    assert guard["converged"]["reason"] == "session profiles incomplete"
    assert guard["converged"]["signal"] == "derived_domain_readiness.session_profiles"


def test_search_not_ready_reports_component_summary() -> None:
    kwargs = _base_kwargs()
    kwargs["search_ready"] = False
    kwargs["search_summary"] = "fts index incomplete"
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["search_ready"]["value"] is False
    assert guard["search_ready"]["reason"] == "fts index incomplete"


def test_active_writer_blocks_perf_measurable_with_reason() -> None:
    kwargs = _base_kwargs()
    kwargs["active_writer"] = True
    kwargs["active_writer_summary"] = "2 live ingest attempt(s) running"
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["perf_measurable"]["value"] is False
    assert guard["perf_measurable"]["reason"] == "2 live ingest attempt(s) running"


def test_active_writer_without_summary_falls_back_to_generic_reason() -> None:
    kwargs = _base_kwargs()
    kwargs["active_writer"] = True
    kwargs["active_writer_summary"] = ""
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["perf_measurable"]["value"] is False
    assert "in flight" in str(guard["perf_measurable"]["reason"])

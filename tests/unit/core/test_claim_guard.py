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


def test_incomplete_inspection_withholds_converged_instead_of_denying_it() -> None:
    """An unfinished inspection is a third state, never a negative claim.

    converged:false must mean PROVEN not converged. A domain whose inspection
    ran out of budget cannot certify convergence and cannot refute it either,
    so the claim is withheld (value None, determinate False) and the reason
    names the gap.

    Anti-vacuity: collapse indeterminate domains back into the not-ready scan
    and ``value`` becomes False with a reason that describes the archive, so
    both the None assertion and the determinate assertion go red.
    """
    kwargs = _base_kwargs()
    kwargs["derived_domains"] = (
        DerivedDomainReadiness("raw_materialization", True, "ready"),
        DerivedDomainReadiness(
            "session_summary",
            False,
            "session-summary inspection deadline exceeded",
            determinate=False,
        ),
    )
    entry = derive_claim_guard(**kwargs).converged  # type: ignore[arg-type]

    assert entry.value is None
    assert entry.determinate is False
    assert "inspection incomplete for session_summary" in entry.reason
    assert entry.to_dict()["value"] is None
    assert entry.to_dict()["determinate"] is False


def test_a_proven_unready_domain_outranks_an_unmeasured_one() -> None:
    """A real counterexample is reported as false even beside an unknown.

    Anti-vacuity: pick the first non-ready domain in list order instead of
    preferring the determinate one, and this reports the unmeasured domain's
    withheld claim instead of the proven failure.
    """
    kwargs = _base_kwargs()
    kwargs["derived_domains"] = (
        DerivedDomainReadiness("session_summary", False, "not measured", determinate=False),
        DerivedDomainReadiness("fts", False, "fts index incomplete"),
    )
    entry = derive_claim_guard(**kwargs).converged  # type: ignore[arg-type]

    assert entry.value is False
    assert entry.determinate is True
    assert entry.reason == "fts index incomplete"
    assert entry.signal == "derived_domain_readiness.fts"


def test_unreadable_writer_evidence_withholds_perf_measurable() -> None:
    """Unreadable writer evidence is not proof that nothing is writing.

    ``_live_ingest_attempt_summary_info`` returns a zero-count summary on an
    unreadable attempt ledger, which used to render as
    ``perf_measurable: true, reason="no concurrent archive write/rebuild
    detected"`` (polylogue-g88v4). Anti-vacuity: dropping
    ``active_writer_determinate`` back to its default, or deriving ``value``
    from ``active_writer`` alone, makes this entry determinate again and turns
    this test red.
    """
    kwargs = _base_kwargs()
    kwargs["active_writer"] = False
    kwargs["active_writer_determinate"] = False
    kwargs["active_writer_summary"] = "ingest workload inspection unavailable; cannot rule out a concurrent writer"
    guard = derive_claim_guard(**kwargs).to_dict()  # type: ignore[arg-type]

    assert guard["perf_measurable"]["value"] is None
    assert guard["perf_measurable"]["determinate"] is False
    assert "cannot rule out" in str(guard["perf_measurable"]["reason"])


def test_a_zero_raw_artifact_denominator_withholds_raw_materialization() -> None:
    """A vacuously-satisfied readiness predicate certifies nothing.

    ``raw_materialization_ready`` is the invariant "every raw artifact is
    materialized", and it reads every blocking counter as zero on an archive
    that holds no raw artifacts at all -- so a pristine archive published
    ``converged: true`` off an inspection that examined no rows
    (polylogue-njcms). This is the same zero-denominator shape
    :func:`search_unmeasured_reason` already refuses for FTS coverage
    (polylogue-o6oct).

    Anti-vacuity, both directions: deleting the zero-denominator branch makes
    the first assertion red, and widening it to any falsy/absent count makes
    the second and third red -- a populated archive must stay determinate, and
    a producer that reported no denominator at all must keep its own verdict
    rather than have a real refutation masked into "unknown".
    """
    from polylogue.readiness.claim_guard import raw_materialization_unmeasured_reason

    complete = {"available": True, "raw_authority_parser_census": {"available": True}}

    assert raw_materialization_unmeasured_reason({**complete, "raw_artifact_count": 0}) is not None
    assert raw_materialization_unmeasured_reason({**complete, "raw_artifact_count": 4}) is None
    assert raw_materialization_unmeasured_reason(complete) is None

"""The admission proof certifies only records a parsing owner accounted for (polylogue-mg7jx).

Two defects let the proof certify input nobody lowered:

* ``AdmissionObserver`` marked every record without a ``future_``-style
  sentinel as materialized, so an ordinary-shaped record a parser discarded
  counted as parsed, and a one-pass stream's denominator was only what the
  parser pulled before it returned.
* ``ParseAccounting.assert_conserved`` compared counts, not members, so a
  range or outcome outside ``[0, expected)`` balanced the ledger (the
  universe check landed with #5798; these cases pin each mutation).

Each test names the input and the proof the old code accepted.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from polylogue.core.enums import Provider
from polylogue.sources.parsers.base_models import (
    AdmissionDisposition,
    AdmissionOutcome,
    AdmissionUnit,
    AdmissionUnknownReason,
    ParseAccounting,
    ParsedSession,
)
from polylogue.sources.parsers.base_support import AdmissionLedger, AdmissionObserver, parser_admission


def _session() -> ParsedSession:
    return ParsedSession(source_name=Provider.CODEX, provider_session_id="probe", messages=[])


@parser_admission("probe")
def _discarding_parser(payload: Iterable[object], fallback_id: str) -> ParsedSession:
    """A parser that reads its input and lowers none of it."""
    for _ in payload:
        pass
    return _session()


@parser_admission("probe")
def _early_returning_parser(payload: Iterable[object], fallback_id: str) -> ParsedSession:
    """A parser that pulls one record, claims it in its own ledger, and returns."""
    first = next(iter(payload))
    ledger = AdmissionLedger()
    ledger.expect(AdmissionUnit.OUTER_RECORD, 1)
    ledger.materialized(AdmissionUnit.OUTER_RECORD, 0, str(first))
    return _session().model_copy(update={"unit_accounting": ledger.close()})


@parser_admission("probe")
def _stopping_parser(payload: Iterable[object], fallback_id: str) -> ParsedSession:
    """A parser without a ledger that pulls one record of a stream and returns."""
    next(iter(payload))
    return _session()


def test_an_ordinary_record_a_parser_discarded_is_not_certified() -> None:
    """``[{"type": "ordinary_new_record"}]`` was certified materialized ``[(0, 1)]`` with no event."""
    with pytest.raises(ValueError, match="gave no disposition for 1 of 1 outer records"):
        _discarding_parser([{"type": "ordinary_new_record"}], "probe")


def test_a_stream_the_parser_leaves_early_counts_every_record() -> None:
    """A 3-record stream the parser takes 1 of was certified with ``expected=1``."""
    records = iter([{"type": "a"}, {"type": "b"}, {"type": "c"}])
    with pytest.raises(ValueError, match="accounted for 1 of 3 outer records"):
        _early_returning_parser(records, "probe")


def test_a_stream_left_early_without_a_ledger_is_not_certified() -> None:
    records = iter([{"type": "a"}, {"type": "b"}, {"type": "c"}])
    with pytest.raises(ValueError, match="gave no disposition for 3 of 3 outer records"):
        _stopping_parser(records, "probe")


def test_one_whole_document_is_settled_by_the_parser_that_consumed_it() -> None:
    """The single-document route still proves its one outer record."""

    @parser_admission("probe")
    def parse(payload: dict[str, object], fallback_id: str) -> ParsedSession:
        return _session()

    accounting = parse({"conversation": {}}, "probe").unit_accounting
    assert accounting is not None
    assert accounting.expected == {AdmissionUnit.OUTER_RECORD: 1}
    assert accounting.materialized_ordinals == {AdmissionUnit.OUTER_RECORD: [(0, 1)]}


def test_parser_dispositions_settle_a_stream() -> None:
    observer = AdmissionObserver(record_stream=True)
    observer.observe({"type": "message"}, lowered=True)
    observer.observe({"type": "noise"}, lowered=False)

    accounting = observer.apply(_session(), "probe").unit_accounting

    assert accounting is not None
    assert accounting.materialized_ordinals == {AdmissionUnit.OUTER_RECORD: [(0, 1)]}
    assert [(outcome.ordinal, outcome.disposition) for outcome in accounting.outcomes] == [
        (1, AdmissionDisposition.TYPED_UNKNOWN)
    ]


def _unknown(ordinal: int) -> AdmissionOutcome:
    return AdmissionOutcome(
        unit=AdmissionUnit.OUTER_RECORD,
        ordinal=ordinal,
        key="k",
        disposition=AdmissionDisposition.TYPED_UNKNOWN,
        reason=AdmissionUnknownReason.UNRECOGNIZED_TYPE,
    )


@pytest.mark.parametrize(
    ("accounting", "refusal"),
    [
        pytest.param(
            ParseAccounting(
                expected={AdmissionUnit.OUTER_RECORD: 1},
                materialized_ordinals={AdmissionUnit.OUTER_RECORD: [(99, 100)]},
            ),
            "invalid materialized admission range",
            id="out-of-domain-range",
        ),
        pytest.param(
            ParseAccounting(
                expected={AdmissionUnit.OUTER_RECORD: 2},
                materialized_ordinals={AdmissionUnit.OUTER_RECORD: [(0, 1), (5, 6)]},
            ),
            "invalid materialized admission range",
            id="wrong-member-range",
        ),
        pytest.param(
            ParseAccounting(
                expected={AdmissionUnit.OUTER_RECORD: 2},
                outcomes=[_unknown(3)],
                materialized_ordinals={AdmissionUnit.OUTER_RECORD: [(0, 1)]},
            ),
            "outside denominator",
            id="same-count-substitution",
        ),
        pytest.param(
            ParseAccounting(expected={AdmissionUnit.OUTER_RECORD: 1}, outcomes=[_unknown(-1)]),
            "outside denominator",
            id="negative-outcome-ordinal",
        ),
        pytest.param(
            ParseAccounting(expected={}, materialized_ordinals={AdmissionUnit.BLOCK: [(0, 1)]}),
            "invalid materialized admission range",
            id="undeclared-unit",
        ),
        pytest.param(
            ParseAccounting(expected={AdmissionUnit.OUTER_RECORD: 2}, outcomes=[_unknown(0)]),
            "mismatch",
            id="count-mismatch",
        ),
    ],
)
def test_conservation_refuses_a_ledger_that_does_not_cover_its_universe(
    accounting: ParseAccounting, refusal: str
) -> None:
    """Each ledger balanced or was accepted before; the universe check refuses it."""
    with pytest.raises(ValueError, match=refusal):
        accounting.assert_conserved()


def test_conservation_accepts_an_exact_cover() -> None:
    ParseAccounting(
        expected={AdmissionUnit.OUTER_RECORD: 3},
        outcomes=[_unknown(1)],
        materialized_ordinals={AdmissionUnit.OUTER_RECORD: [(0, 1), (2, 3)]},
    ).assert_conserved()

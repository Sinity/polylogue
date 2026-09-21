"""The collection-cost measurement must narrow, and must not launder failures.

polylogue-016xl states its target as a whole-corpus ``--collect-only`` peak.
Two ways a measurement command silently lies: it collects more than it was
asked for and reports the corpus number as the selection's, or it reports a
broken collection as a budget verdict. Both are pinned here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from devtools import collection_cost
from devtools.pytest_invocation import CLOSED_WORLD_COLLECTION_ARGS
from devtools.verify_test_collection import collection_command

_ROOT = Path(__file__).resolve().parents[3]


def test_an_unnarrowed_measurement_is_exactly_the_gates_invocation() -> None:
    """The gate and the measurement must collect the same corpus.

    Anti-vacuity: re-derive the pytest line here instead of reusing
    ``collection_command`` and the two drift apart the first time the declared
    plugin set or ini override changes -- this equality goes red.
    """
    assert collection_cost.collection_argv([], root=_ROOT) == list(collection_command(root=_ROOT))


def test_a_narrowed_measurement_replaces_the_corpus_root_instead_of_adding_to_it() -> None:
    """The defect this guards is real and was observed.

    Appending a file to the declared invocation left the corpus root ``tests``
    in place, so ``bench collection <one file>`` collected 23,547 tests in 93 s
    and reported that as the file's cost. Substituting gives 27 tests in 1.9 s.

    Anti-vacuity: append rather than substitute and the corpus root survives,
    so both assertions below go red.
    """
    argv = collection_cost.collection_argv(["tests/unit/devtools/test_collection_cost.py"], root=_ROOT)

    corpus_root = CLOSED_WORLD_COLLECTION_ARGS[-1]
    assert corpus_root not in argv
    assert argv.count("tests/unit/devtools/test_collection_cost.py") == 1
    # Everything else about the declared invocation survives.
    unchanged = [item for item in collection_command(root=_ROOT) if item != corpus_root]
    assert [item for item in argv if item != "tests/unit/devtools/test_collection_cost.py"] == unchanged


def test_a_count_pytest_did_not_report_is_none_and_not_zero() -> None:
    """An unavailable measurement must not become a measured zero.

    Anti-vacuity: default the count to 0 and a collection that crashed before
    reporting anything reads as an empty-but-successful selection.
    """
    assert collection_cost._collected_count("Traceback (most recent call last):\nImportError") is None
    assert collection_cost._collected_count("23,547 tests collected in 89.26s") == 23547
    assert collection_cost._collected_count("1 test collected in 0.1s") == 1
    # The last count wins: pytest can print an interim line before the summary.
    assert collection_cost._collected_count("5 tests collected\n7 tests collected") == 7


def _measured(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "polylogue.collection-cost",
        "selection": ["<whole corpus>"],
        "collected": 23547,
        "wall_clock_s": 92.78,
        "peak_rss_mib": 581.3,
        "peak_rss_delta_mib": 545.1,
        "returncode": 0,
        "tail": [],
    }
    payload.update(overrides)
    return payload


def test_over_budget_and_a_broken_collection_exit_differently(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A budget breach is not a collection error, and neither is the other.

    Anti-vacuity: return 1 for both and a reader cannot tell "the corpus got
    bigger" from "a module stopped importing" without parsing prose -- which
    is the failure this whole command exists to make legible.
    """
    monkeypatch.setattr(collection_cost, "measure_collection", lambda selection, *, root: _measured())
    assert collection_cost.main(["--budget-mib", "430"]) == collection_cost.OVER_BUDGET_EXIT
    assert "OVER 430 MiB" in capsys.readouterr().out

    monkeypatch.setattr(
        collection_cost, "measure_collection", lambda selection, *, root: _measured(returncode=2, collected=None)
    )
    assert collection_cost.main(["--budget-mib", "430"]) == 2


def test_without_a_budget_a_measurement_is_never_a_verdict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The target has never been met, so enforcing it by default hides it.

    Anti-vacuity: default ``--budget-mib`` to the corpus target and every
    unrelated invocation of this command starts failing, which is how a
    measurement stops being run at all.
    """
    monkeypatch.setattr(collection_cost, "measure_collection", lambda selection, *, root: _measured())

    assert collection_cost.main([]) == 0
    output = capsys.readouterr().out
    assert "581.3 MiB" in output
    assert "budget" not in output

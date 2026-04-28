from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from polylogue.cli import products_rendering


@dataclass(frozen=True)
class _ModelItem:
    value: str

    def model_dump(self, *, mode: str) -> object:
        assert mode == "json"
        return {"value": self.value}


def test_model_payload_and_payloads_cover_model_dump_and_passthrough() -> None:
    dumped = products_rendering.model_payload(_ModelItem("alpha"))
    passthrough = products_rendering.model_payload({"value": "beta"})

    assert dumped == {"value": "alpha"}
    assert passthrough == {"value": "beta"}
    assert products_rendering.model_payloads([_ModelItem("alpha"), {"value": "beta"}]) == [
        {"value": "alpha"},
        {"value": "beta"},
    ]


def test_emit_product_list_emits_count_and_json_ready_items() -> None:
    with patch("polylogue.cli.products_rendering.emit_success") as emit_success:
        products_rendering.emit_product_list(
            key="items",
            items=[_ModelItem("alpha"), {"value": "beta"}],
        )

    emit_success.assert_called_once_with(
        {
            "count": 2,
            "items": [{"value": "alpha"}, {"value": "beta"}],
        }
    )


def test_summarize_archive_debt_counts_actionable_items_and_issue_rows() -> None:
    items = [
        _ModelItem("healthy"),
        type("DebtRow", (), {"healthy": False, "issue_count": 2})(),
        type("DebtRow", (), {"healthy": False, "issue_count": 0})(),
        type("DebtRow", (), {"healthy": True, "issue_count": 5})(),
    ]

    assert products_rendering.summarize_archive_debt(items) == {
        "tracked_items": 4,
        "actionable_items": 2,
        "issue_rows": 7,
    }

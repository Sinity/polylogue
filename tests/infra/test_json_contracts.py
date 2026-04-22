"""Tests for shared JSON assertion contracts."""

from __future__ import annotations

import pytest

from tests.infra.json_contracts import (
    extract_json_object,
    extract_json_result,
    json_array_field,
    json_int,
    json_object_list,
    parse_json_object,
)


def test_extract_json_object_skips_banner_lines() -> None:
    payload = extract_json_object('banner\n{"status":"ok","result":{"count":2}}\n', context="sample")

    assert payload["status"] == "ok"
    assert parse_json_object('{"value": 1}')["value"] == 1


def test_extract_json_result_unwraps_success_envelope() -> None:
    payload = extract_json_result('debug\n{"status":"ok","result":{"count":2}}\n')

    assert payload == {"count": 2}


def test_json_object_list_and_numeric_contracts() -> None:
    payload = parse_json_object('{"rows":[{"count":2},{"count":3}]}')
    rows = json_object_list(json_array_field(payload, "rows"))

    assert [json_int(row["count"]) for row in rows] == [2, 3]


def test_parse_json_object_rejects_arrays() -> None:
    with pytest.raises(AssertionError, match="not a JSON object"):
        parse_json_object("[1, 2, 3]", context="array")

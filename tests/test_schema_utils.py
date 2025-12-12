from polylogue.schema import stamp_payload


def test_stamp_payload_dual_keys_and_versions():
    payload = {"foo_bar": 1, "nested_val": {"inner_key": "x"}}
    stamped = stamp_payload(payload)

    # versions injected
    assert "schemaVersion" in stamped
    assert "polylogueVersion" in stamped

    # dual casing
    assert stamped["foo_bar"] == 1
    assert stamped["fooBar"] == 1
    assert stamped["nested_val"]["inner_key"] == "x"
    assert stamped["nestedVal"]["innerKey"] == "x"

use super::*;

#[test]
fn canonical_retains_tokens_presence_and_decoded_utf16_keys() {
    let raw = br#"{"z":[1.00,1e+02,-0,184467440737095516160,null],"\ud800":"\udfff","a":{}}"#;
    let value = RawJson::parse(raw).unwrap();
    assert_eq!(
        value.canonical(),
        br#"{"a":{},"z":[1.00,1e+02,-0,184467440737095516160,null],"\ud800":"\udfff"}"#
    );
    let key = NativeId::from_utf16(&[0xd800]);
    assert_eq!(
        value.as_object().unwrap()[&key].native().unwrap().utf16(),
        [0xdfff]
    );
    assert!(value.get("missing").is_none());
    assert!(value["z"][4].is_null());
    assert_eq!(serde_json::to_vec(&value).unwrap(), value.canonical());
    let restored: RawJson = serde_json::from_slice(&value.canonical()).unwrap();
    assert_eq!(restored, value);
}

#[test]
fn duplicate_decoded_keys_refuse_instead_of_overwriting() {
    for raw in [
        br#"{"a":1,"\u0061":2}"#.as_slice(),
        br#"{"\ud800":1,"\ud800":2}"#,
    ] {
        assert!(RawJson::parse(raw).is_err());
    }
    // Literal transport-looking strings are distinct from decoded lone units.
    assert!(RawJson::parse(br#"{"~u:eda080":1,"\ud800":2}"#).is_ok());
}

#[test]
fn surrogate_repairs_are_counted_once_at_original_offsets() {
    let log = TextRepairs::default();
    let raw = br#"{"key":"a\ud800b\udfffc","opaque":"\ud800"}"#;
    let value = RawJson::parse_with_repairs(raw, 100, &log).unwrap();
    assert_eq!(log.count(), 0);
    assert_eq!(
        value["key"].native().unwrap().utf16(),
        [97, 0xd800, 98, 0xdfff, 99]
    );
    assert_eq!(log.count(), 0);
    assert_eq!(value["key"].text(), Some("a\u{fffd}b\u{fffd}c"));
    value["key"].text();
    assert_eq!(
        log.entries(),
        [TextRepair {
            record: 100,
            byte_offset: 109,
            count: 2
        }]
    );
    value["opaque"].opaque().text();
    assert_eq!(log.count(), 2);
}

#[test]
fn embedded_json_repairs_rebase_to_outer_original_bytes() {
    let log = TextRepairs::default();
    let raw = br#"{"body":"{\"text\":\"\\ud800\"}"}"#;
    let outer = RawJson::parse_with_repairs(raw, 700, &log).unwrap();
    let RawJson::String(body) = &outer["body"] else {
        panic!()
    };
    let inner = body.json().unwrap();
    assert_eq!(log.count(), 0);
    assert_eq!(inner["text"].text(), Some("\u{fffd}"));
    let at = raw.windows(7).position(|b| b == br#"\\ud800"#).unwrap();
    assert_eq!(
        log.entries(),
        [TextRepair {
            record: 700,
            byte_offset: 700 + at as u64,
            count: 1
        }]
    );
}

#[test]
fn depth_limit_is_checked_before_recursing() {
    for depth in [JSON_DEPTH_LIMIT, JSON_DEPTH_LIMIT + 1, 10000] {
        let raw = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        let result = RawJson::parse(raw.as_bytes());
        if depth <= JSON_DEPTH_LIMIT {
            assert!(result.is_ok());
        } else {
            assert!(matches!(
                result,
                Err(RawJsonError::Capacity {
                    limit: JSON_DEPTH_LIMIT,
                    ..
                })
            ));
        }
    }
    assert!(RawJson::parse(br#"{"text":"[{{[["}"#).is_ok());
}

#[test]
fn malformed_and_invalid_utf8_are_refused() {
    for raw in [b"\xff".as_slice(), b"[1,]", b"{}{}", b"\"\\x\"", b"01"] {
        assert!(RawJson::parse(raw).is_err());
    }
}

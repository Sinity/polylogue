use super::*;

#[test]
fn parses_rfc3339_with_offset_only() {
    let t = Timestamp::parse_rfc3339("2026-01-02T03:04:05.678Z").unwrap();
    assert_eq!(t.unix_ms(), 1_767_323_045_678);
    assert!(Timestamp::parse_rfc3339("2026-01-02T03:04:05").is_none());
    assert_eq!(
        Timestamp::parse_rfc3339("2026-01-02T04:04:05.678+01:00"),
        Some(t)
    );
}

#[test]
fn serializes_as_rfc3339() {
    let t = Timestamp::from_unix_ms(0).unwrap();
    assert_eq!(
        serde_json::to_string(&t).unwrap(),
        "\"1970-01-01T00:00:00Z\""
    );
}

#[test]
fn nanosecond_creation_refuses_values_outside_supported_instants() {
    let min = jiff::Timestamp::MIN.as_nanosecond();
    let max = jiff::Timestamp::MAX.as_nanosecond();
    assert_eq!(Timestamp::from_unix_ns(min).unwrap().unix_ns(), min);
    assert_eq!(Timestamp::from_unix_ns(max).unwrap().unix_ns(), max);
    for value in [
        i128::MIN,
        min - 1,
        max + 1,
        i128::MAX,
        1712000001000i128 * 1_000_000_000,
    ] {
        assert!(Timestamp::from_unix_ns(value).is_none());
    }
}

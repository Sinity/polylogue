use super::*;
use serde_json::json;

#[test]
fn native_encodings_keep_declared_precision() {
    for (value, precision) in [
        (
            rfc3339(&json!("2026-01-02T03:04:05.250Z")),
            Precision::Instant,
        ),
        (rfc3339(&json!("2026-01-02T03:04:05Z")), Precision::Second),
        (epoch_seconds(&json!(1767323045)), Precision::Second),
        (epoch_seconds(&json!(1767323045.25)), Precision::Instant),
        (epoch_millis(&json!(1767323045000i64)), Precision::Instant),
    ] {
        let time = value.unwrap();
        assert_eq!(time.evidence(), TimeEvidence::intrinsic(precision));
        assert_eq!(time.timestamp().unix_ms() / 1000, 1767323045);
    }
}

#[test]
fn decimal_seconds_and_otlp_nanos_are_exact() {
    let decimal: Json = serde_json::from_str("1767323045.123456789").unwrap();
    let at = epoch_seconds(&decimal).unwrap();
    assert_eq!(at.unix_ns(), 1767323045123456789);
    assert_eq!(epoch_nanos(&json!("1767323045123456789")), Some(at));
    assert_eq!(epoch_seconds(&json!(-0.25)).unwrap().unix_ns(), -250000000);
    assert_eq!(
        epoch_seconds(&serde_json::from_str::<Json>("1.23e-7").unwrap())
            .unwrap()
            .unix_ns(),
        123
    );
    assert_eq!(
        epoch_seconds(&serde_json::from_str::<Json>("0.0000000001").unwrap()),
        None
    );
}

#[test]
fn unreadable_or_unzoned_times_stay_unknown() {
    for value in [json!("yesterday"), json!(5), json!("2026-01-02T03:04:05")] {
        assert_eq!(rfc3339(&value), None);
    }
    assert_eq!(epoch_millis(&json!("5")), None);
    assert_eq!(epoch_millis(&json!(1.5)), None);
    assert_eq!(parts(None), (None, TimeEvidence::UNKNOWN));
}

#[test]
fn explicit_observation_and_file_semantics_preserve_resolution() {
    let at = rfc3339(&json!("2026-01-02T03:04:05Z")).unwrap();
    assert_eq!(at.captured().evidence().quality, TimeQuality::Captured);
    assert_eq!(at.file_mtime().evidence().quality, TimeQuality::FileMtime);
    assert_eq!(at.captured().evidence().precision, Some(Precision::Second));
    assert_eq!(at.file_mtime().timestamp(), at.timestamp());
}

#[test]
fn persisted_parser_time_rejects_inconsistent_evidence() {
    let invalid = json!({"at":"2026-01-02T03:04:05Z", "evidence":{"quality":"unknown","precision":null,"until":null}});
    assert!(serde_json::from_value::<SourceTime>(invalid).is_err());
    let valid = rfc3339(&json!("2026-01-02T03:04:05Z")).unwrap();
    assert_eq!(
        serde_json::from_value::<SourceTime>(serde_json::to_value(valid).unwrap()).unwrap(),
        valid
    );
}

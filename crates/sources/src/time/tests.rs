use super::*;
use serde_json::json;

#[test]
fn reads_each_encoding() {
    let t = Timestamp::parse_rfc3339("2026-01-02T03:04:05.250Z").unwrap();
    assert_eq!(rfc3339(&json!("2026-01-02T03:04:05.250Z")), Some(t));
    assert_eq!(epoch_millis(&json!(t.unix_ms())), Some(t));
    assert_eq!(epoch_seconds(&json!(1_767_323_045.25)), Some(t));
    assert_eq!(epoch_seconds(&json!(0)), Timestamp::from_unix_ms(0));
}

#[test]
fn unreadable_times_stay_unknown() {
    assert_eq!(rfc3339(&json!("yesterday")), None);
    assert_eq!(rfc3339(&json!(5)), None);
    assert_eq!(epoch_millis(&json!("5")), None);
    assert_eq!(epoch_millis(&json!(1.5)), None);
}

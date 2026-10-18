//! Reading the times sources write. Each origin says which encoding a field
//! uses; nothing here guesses between seconds and milliseconds.

use polylogue_model::{Json, Timestamp};

/// An RFC 3339 string with an offset.
pub fn rfc3339(v: &Json) -> Option<Timestamp> {
    Timestamp::parse_rfc3339(v.as_str()?)
}

/// Seconds since the Unix epoch, integer or fractional.
pub fn epoch_seconds(v: &Json) -> Option<Timestamp> {
    let n = v.as_number()?;
    if let Some(i) = n.as_i64() {
        return Timestamp::from_unix_ns(i128::from(i) * 1_000_000_000);
    }
    let f = n.as_f64()?;
    if !f.is_finite() {
        return None;
    }
    // Fractional seconds carry at most microsecond precision in practice;
    // rounding to it avoids float noise in the nanoseconds.
    let micros = (f * 1e6).round();
    Timestamp::from_unix_ns(micros as i128 * 1000)
}

/// Milliseconds since the Unix epoch.
pub fn epoch_millis(v: &Json) -> Option<Timestamp> {
    Timestamp::from_unix_ms(v.as_number()?.as_i64()?)
}

#[cfg(test)]
mod tests {
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
}

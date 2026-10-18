//! Provider timestamps. Every adapter reads RFC 3339 text to the millisecond.
use polylogue_model::{Precision, ProviderTime, RawJson as Value, Timestamp};

pub(crate) fn parse(text: &str) -> Option<Timestamp> {
    let at = text.parse::<jiff::Timestamp>().ok()?;
    Some(Timestamp(at.as_millisecond()))
}

/// A time the provider recorded for the item itself.
pub(crate) fn intrinsic(at: Timestamp) -> ProviderTime {
    ProviderTime::Intrinsic {
        at,
        precision: Precision::Instant,
        until: None,
    }
}

pub(crate) fn parse_intrinsic(text: &str) -> Option<ProviderTime> {
    parse(text).map(intrinsic)
}

pub(crate) fn intrinsic_field(value: Option<&Value>) -> Option<ProviderTime> {
    value.and_then(Value::as_str).and_then(parse_intrinsic)
}

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

#[test]
fn source_time_evidence_distinguishes_unknown_instants_and_periods() {
    let at = Timestamp::parse_rfc3339("2026-01-02T03:04:05.123456789Z").unwrap();
    assert!(TimeEvidence::UNKNOWN.validate(None).is_ok());
    assert_eq!(
        TimeEvidence::UNKNOWN.validate(Some(at)),
        Err(TimeEvidenceError::UnknownQuality)
    );
    let instant = TimeEvidence::intrinsic(Precision::Instant);
    assert!(instant.validate(Some(at)).is_ok());
    assert_eq!(instant.validate(None), Err(TimeEvidenceError::MissingTime));
    let until = Timestamp::from_unix_ns(at.unix_ns() + 1).unwrap();
    let period = TimeEvidence {
        quality: TimeQuality::Intrinsic,
        precision: Some(Precision::Bounded),
        until: Some(until),
    };
    assert!(period.validate(Some(at)).is_ok());
    assert_eq!(
        TimeEvidence {
            until: Some(at),
            ..period
        }
        .validate(Some(at)),
        Err(TimeEvidenceError::InvalidBounds)
    );
    assert_eq!(
        TimeEvidence {
            until: None,
            ..period
        }
        .validate(Some(at)),
        Err(TimeEvidenceError::MissingBounds)
    );
    assert_eq!(
        TimeEvidence {
            until: Some(until),
            ..instant
        }
        .validate(Some(at)),
        Err(TimeEvidenceError::InstantHasBounds)
    );
    assert_eq!(
        TimeEvidence {
            precision: None,
            ..instant
        }
        .validate(Some(at)),
        Err(TimeEvidenceError::MissingPrecision)
    );
    let encoded = serde_json::to_string(&(at, period)).unwrap();
    let decoded: (Timestamp, TimeEvidence) = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, (at, period));
    assert_eq!(decoded.0.unix_ns(), at.unix_ns());
}

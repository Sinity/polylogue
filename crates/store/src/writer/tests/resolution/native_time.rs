//! Hand-computed temporal cases through admitted bytes, the writer and sole compiler.
use super::*;
use polylogue_model::{Precision, TimeEvidence, TimeQuality};

fn stamp(text: &str) -> Timestamp {
    Timestamp::parse_rfc3339(text).unwrap()
}

fn values(reader: &crate::Reader<'_>, text: &str, field: &str) -> Vec<String> {
    read_values(
        reader,
        text,
        field,
        query_language::time::DaySetting::new(
            jiff::tz::TimeZone::UTC,
            jiff::civil::Time::midnight(),
        )
        .unwrap(),
    )
    .unwrap_or_else(|error| panic!("{text}: {error}"))
}

fn read_values(
    reader: &crate::Reader<'_>,
    text: &str,
    field: &str,
    day: query_language::time::DaySetting,
) -> crate::Result<Vec<String>> {
    let options = polylogue_query::Options {
        clock: query_language::time::Clock::new("2026-10-29T12:00:00Z".parse().unwrap(), day),
        origins: vec!["claude-code".into()],
        page_size: None,
        random_seed: Some(1),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        kept: Default::default(),
        fields: vec![field.into()],
        chat_pages: vec![],
    };
    let program =
        crate::query::compile(text, &options).unwrap_or_else(|error| panic!("{text}: {error}"));
    let parameters = program
        .params
        .iter()
        .map(|p| match p {
            crate::query::Param::Int(n) => crate::SqlValue::Int(*n),
            crate::query::Param::Real(n) => crate::SqlValue::Real(*n),
            crate::query::Param::Text(s) => crate::SqlValue::Text(s.clone()),
        })
        .collect::<Vec<_>>();
    let rows = reader.with_clock(&program.clock, || reader.query(&program.sql, &parameters))?;
    let mut values = rows
        .iter()
        .map(|row| row[field].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    values.sort();
    Ok(values)
}

#[test]
fn native_time_spans_overlap_bounds_and_comparisons_preserve_unknown_and_nanoseconds() {
    let w = test_archive();
    let bounded = TimeEvidence {
        quality: TimeQuality::Intrinsic,
        precision: Some(Precision::Bounded),
        until: Some(stamp("2026-10-29T11:00:00Z")),
    };
    let mut message = text_message("bounded", "temporal", Role::User, "native interval", None);
    message
        .set_time(Some(stamp("2026-10-29T10:00:00Z")), bounded)
        .unwrap();
    let mut input = batch(
        "native-temporal",
        0,
        0,
        vec![row(1, message)],
        &["temporal"],
    );
    input.parse[0].output.sessions[0]
        .set_time(Some(stamp("2026-10-29T10:00:00Z")), bounded)
        .unwrap();
    for (index, (kind, at, precision, until)) in [
        (
            "bounded",
            Some("2026-10-29T10:00:00Z"),
            Some(Precision::Bounded),
            Some("2026-10-29T11:00:00Z"),
        ),
        (
            "calendar",
            Some("2026-01-01T00:00:00Z"),
            Some(Precision::Year),
            Some("2027-01-01T00:00:00Z"),
        ),
        (
            "calendar-no-boundary",
            Some("2026-01-01T00:00:00Z"),
            Some(Precision::Year),
            None,
        ),
        (
            "late",
            Some("2026-10-29T11:00:00Z"),
            Some(Precision::Instant),
            None,
        ),
        (
            "nano-one",
            Some("2026-10-29T10:30:00.000000001Z"),
            Some(Precision::Instant),
            None,
        ),
        (
            "nano-two",
            Some("2026-10-29T10:30:00.000000002Z"),
            Some(Precision::Instant),
            None,
        ),
        (
            "second",
            Some("2026-10-29T10:30:00Z"),
            Some(Precision::Second),
            None,
        ),
        (
            "minute",
            Some("2026-10-29T10:30:00Z"),
            Some(Precision::Minute),
            None,
        ),
        (
            "negative-one",
            Some("1969-12-31T23:59:59.999999999Z"),
            Some(Precision::Instant),
            None,
        ),
        (
            "negative-two",
            Some("1969-12-31T23:59:59.999999998Z"),
            Some(Precision::Instant),
            None,
        ),
        ("missing-time", None, None, None),
    ]
    .into_iter()
    .enumerate()
    {
        let mut event = Event {
            session: Some(sid("temporal")),
            kind: kind.into(),
            at: None,
            at_quality: TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            after: None,
            range: Some((index as u64 + 2, index as u64 + 3)),
            data: serde_json::json!({"synthetic":kind}),
        };
        event
            .set_time(
                at.map(stamp),
                TimeEvidence {
                    quality: if at.is_some() {
                        TimeQuality::Intrinsic
                    } else {
                        TimeQuality::Unknown
                    },
                    precision,
                    until: until.map(stamp),
                },
            )
            .unwrap();
        input.parse[0].output.events.push(event);
    }
    let mut document = polylogue_model::Document::new(
        polylogue_model::DocumentId::parse("claude-code:bounded-document").unwrap(),
        polylogue_model::DocumentPurpose::Context,
    );
    document.blocks.push(Block::Text {
        text: "native document".into(),
    });
    document
        .set_time(Some(stamp("2026-10-29T10:00:00Z")), bounded)
        .unwrap();
    input.parse[0]
        .output
        .documents
        .push(polylogue_model::DocumentRow {
            offset: 12,
            end: 13,
            document,
        });
    let bytes = serde_json::to_vec(&input.parse[0].output).unwrap();
    commit_event_bytes(&w, input, &bytes);
    w.writer.derive().unwrap();
    // A stale nonnull point cache cannot turn unknown native evidence into a time.
    let connection = crate::archive::open_write(&w.archive.index_db(), 64).unwrap();
    connection
        .execute(
            "UPDATE event SET at=?1 WHERE kind='missing-time'",
            [stamp("2026-10-29T10:30:00Z").unix_ms()],
        )
        .unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    for (condition, expected) in [
        ("kind:bounded since:2026-10-29T10:30:00Z", vec!["bounded"]),
        ("kind:bounded until:2026-10-29T10:30:00Z", vec!["bounded"]),
        ("kind:bounded since:2026-10-29T11:00:00Z", vec![]),
        ("kind:bounded until:2026-10-29T10:00:00Z", vec![]),
        (
            "since:2026-10-29T10:30:00.000000001Z until:2026-10-29T10:30:00.000000002Z",
            vec!["bounded", "calendar", "minute", "nano-one", "second"],
        ),
        (
            "since:2026-10-29T11:00:00Z (until:2026-10-29T10:00:00Z)",
            vec![],
        ),
        ("at = 2026-10-29T10:30:00.000000001Z", vec!["nano-one"]),
        (
            "at > 2026-10-29T10:30:00.000000001Z",
            vec!["late", "nano-two"],
        ),
        ("kind:bounded at >= 2026-10-29T10:30:00Z", vec![]),
        ("kind:bounded NOT at >= 2026-10-29T10:30:00Z", vec![]),
        ("kind:bounded at < 2026-10-29T11:00:00Z", vec!["bounded"]),
        ("kind:calendar at:2026", vec!["calendar"]),
        ("kind:calendar-no-boundary at:2026", vec![]),
        ("kind:calendar-no-boundary at:unknown", vec![]),
        ("kind:second since:2026-10-29T10:30:00.5Z", vec!["second"]),
        ("kind:second at >= 2026-10-29T10:30:00.5Z", vec![]),
        ("kind:minute since:2026-10-29T10:30:59.9Z", vec!["minute"]),
        ("kind:minute at >= 2026-10-29T10:30:59.9Z", vec![]),
        ("kind:minute at <= 2026-10-29T10:31:00Z", vec!["minute"]),
        ("at = 1969-12-31T23:59:59.999999999Z", vec!["negative-one"]),
        (
            "since:1969-12-31T23:59:59.999999999Z until:1970-01-01T00:00:00Z",
            vec!["negative-one"],
        ),
        ("at:unknown", vec!["missing-time"]),
        ("NOT since:2026-10-29T10:30:00Z kind:missing-time", vec![]),
        ("kind:bounded until:2026-10-29", vec!["bounded"]),
        ("kind:bounded at < 2026-10-29", vec![]),
    ] {
        assert_eq!(
            values(&reader, &format!("events where {condition}"), "kind"),
            expected,
            "{condition}"
        );
    }
    assert_eq!(
        values(&reader, "messages where since:2026-10-29T10:30:00Z", "id"),
        ["claude-code:bounded"]
    );
    assert_eq!(
        values(&reader, "blocks where since:2026-10-29T10:30:00Z", "kind"),
        ["text"]
    );
    assert_eq!(
        values(&reader, "documents where since:2026-10-29T10:30:00Z", "id"),
        ["claude-code:bounded-document"]
    );
    assert_eq!(
        values(&reader, "sessions where until:2026-10-29T10:30:00Z", "id"),
        ["claude-code:temporal"]
    );
    assert_eq!(
        values(
            &reader,
            "sessions where exists event(kind:missing-time at:unknown)",
            "id"
        ),
        ["claude-code:temporal"]
    );
    assert!(
        values(
            &reader,
            "sessions where exists event(kind:missing-time NOT at < 2026-10-29T11:00:00Z)",
            "id"
        )
        .is_empty()
    );
    assert!(
        values(
            &reader,
            "sessions where NOT exists event(kind:missing-time at >= 2026-10-29T11:00:00Z)",
            "id"
        )
        .is_empty()
    );
    assert_eq!(
        values(
            &reader,
            "sessions where exists event(kind:calendar-no-boundary NOT at:unknown)",
            "id"
        ),
        ["claude-code:temporal"]
    );
    drop(reader);
    drop(pool);
    connection
        .execute(
            "UPDATE event SET fields=X'00' WHERE kind='missing-time'",
            [],
        )
        .unwrap();
    let corrupt_reads = ReadPool::open(&w.archive).unwrap();
    let corrupt_reader = corrupt_reads.reader().unwrap();
    let failure = read_values(
        &corrupt_reader,
        "events where kind:missing-time at:unknown",
        "kind",
        query_language::time::DaySetting::new(
            jiff::tz::TimeZone::UTC,
            jiff::civil::Time::midnight(),
        )
        .unwrap(),
    )
    .unwrap_err();
    assert!(
        matches!(&failure, crate::StoreError::Sqlite(_)),
        "{failure:?}"
    );
}

#[test]
fn native_time_day_overlap_uses_the_resolved_owner_boundary_across_both_dst_changes() {
    let w = test_archive();
    let mut input = batch("native-zones", 0, 0, vec![], &["zones"]);
    for (index, (kind, start, end)) in [
        ("spring", "2026-03-29T02:00:00Z", "2026-03-29T03:00:00Z"),
        ("fold", "2026-10-25T03:00:00Z", "2026-10-25T04:00:00Z"),
    ]
    .into_iter()
    .enumerate()
    {
        let mut event = Event {
            session: Some(sid("zones")),
            kind: kind.into(),
            at: None,
            at_quality: TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            after: None,
            range: Some((index as u64, index as u64 + 1)),
            data: serde_json::json!({"synthetic":kind}),
        };
        event
            .set_time(
                Some(stamp(start)),
                TimeEvidence {
                    quality: TimeQuality::Intrinsic,
                    precision: Some(Precision::Bounded),
                    until: Some(stamp(end)),
                },
            )
            .unwrap();
        input.parse[0].output.events.push(event);
    }
    let bytes = serde_json::to_vec(&input.parse[0].output).unwrap();
    commit_event_bytes(&w, input, &bytes);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    for (zone, expected) in [("UTC", vec![]), ("Europe/Warsaw", vec!["spring"])] {
        let day = query_language::time::DaySetting::new(
            jiff::tz::TimeZone::get(zone).unwrap(),
            jiff::civil::Time::constant(4, 0, 0, 0),
        )
        .unwrap();
        // Warsaw's 04:00 is 02:00Z after the spring change, versus UTC 04:00Z.
        for condition in ["kind:spring at:2026-03-29", "kind:spring since:2026-03-29"] {
            assert_eq!(
                read_values(
                    &reader,
                    &format!("events where {condition}"),
                    "kind",
                    day.clone()
                )
                .unwrap(),
                expected,
                "{zone}: {condition}"
            );
        }
    }
    for (zone, within, before) in [
        ("UTC", vec![], vec!["fold"]),
        ("Europe/Warsaw", vec!["fold"], vec![]),
    ] {
        let day = query_language::time::DaySetting::new(
            jiff::tz::TimeZone::get(zone).unwrap(),
            jiff::civil::Time::constant(4, 0, 0, 0),
        )
        .unwrap();
        // Warsaw's 04:00 is 03:00Z after the fold; each span endpoint is literal above.
        for condition in ["kind:fold at:2026-10-25", "kind:fold since:2026-10-25"] {
            assert_eq!(
                read_values(
                    &reader,
                    &format!("events where {condition}"),
                    "kind",
                    day.clone()
                )
                .unwrap(),
                within,
                "{zone}: {condition}"
            );
        }
        assert_eq!(
            read_values(
                &reader,
                "events where kind:fold until:2026-10-24",
                "kind",
                day
            )
            .unwrap(),
            before,
            "{zone}: until includes the prior complete logical day"
        );
    }
}

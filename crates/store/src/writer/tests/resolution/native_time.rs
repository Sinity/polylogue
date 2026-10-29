//! Hand-computed temporal laws through admitted bytes, the writer and sole compiler.
use super::*;
use polylogue_model::{Precision, TimeEvidence, TimeQuality};

fn stamp(text: &str) -> Timestamp {
    Timestamp::parse_rfc3339(text).unwrap()
}

fn values(reader: &crate::Reader<'_>, text: &str, field: &str) -> Vec<String> {
    let options = polylogue_query::Options {
        clock: query_language::time::Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::UTC,
                jiff::civil::Time::midnight(),
            )
            .unwrap(),
        ),
        origins: vec!["claude-code".into()],
        command_words: vec![],
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
    let rows = reader
        .with_clock(&program.clock, || reader.query(&program.sql, &parameters))
        .unwrap_or_else(|error| panic!("{text}: {error}"));
    let mut values = rows
        .iter()
        .map(|row| row[field].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    values.sort();
    values
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
        ("unknown", None, None, None),
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
            "UPDATE event SET at=?1 WHERE kind='unknown'",
            [stamp("2026-10-29T10:30:00Z").unix_ms()],
        )
        .unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    for (condition, expected) in [
        ("kind:bounded since 2026-10-29T10:30:00Z", vec!["bounded"]),
        ("kind:bounded until 2026-10-29T10:30:00Z", vec!["bounded"]),
        ("kind:bounded since 2026-10-29T11:00:00Z", vec![]),
        ("kind:bounded until 2026-10-29T10:00:00Z", vec![]),
        (
            "since 2026-10-29T10:30:00.000000001Z until 2026-10-29T10:30:00.000000002Z",
            vec!["bounded", "calendar", "nano-one", "second"],
        ),
        (
            "since 2026-10-29T11:00:00Z (until 2026-10-29T10:00:00Z)",
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
        ("kind:second since 2026-10-29T10:30:00.5Z", vec!["second"]),
        ("kind:second at >= 2026-10-29T10:30:00.5Z", vec![]),
        ("at:unknown", vec!["unknown"]),
        ("NOT since 2026-10-29T10:30:00Z kind:unknown", vec![]),
        ("kind:bounded until 2026-10-29", vec!["bounded"]),
        ("kind:bounded at < 2026-10-29", vec![]),
    ] {
        assert_eq!(
            values(&reader, &format!("events where {condition}"), "kind"),
            expected,
            "{condition}"
        );
    }
    assert_eq!(
        values(&reader, "messages where since 2026-10-29T10:30:00Z", "id"),
        ["claude-code:bounded"]
    );
    assert_eq!(
        values(&reader, "blocks where since 2026-10-29T10:30:00Z", "kind"),
        ["text"]
    );
    assert_eq!(
        values(&reader, "documents where since 2026-10-29T10:30:00Z", "id"),
        ["claude-code:bounded-document"]
    );
    assert_eq!(
        values(&reader, "sessions where until 2026-10-29T10:30:00Z", "id"),
        ["claude-code:temporal"]
    );
    assert_eq!(
        values(
            &reader,
            "sessions where exists event(kind:unknown at:unknown)",
            "id"
        ),
        ["claude-code:temporal"]
    );
    assert!(
        values(
            &reader,
            "sessions where exists event(kind:unknown NOT at < 2026-10-29T11:00:00Z)",
            "id"
        )
        .is_empty()
    );
}

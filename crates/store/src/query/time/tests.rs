use super::*;
fn fixture() -> Clock {
    Clock::new(
        "2026-10-07T10:00:00Z".parse().unwrap(),
        query_language::time::DaySetting::new(
            jiff::tz::TimeZone::get("Europe/Warsaw").unwrap(),
            jiff::civil::Time::constant(4, 0, 0, 0),
        )
        .unwrap(),
    )
}
fn ms(text: &str) -> i64 {
    text.parse::<jiff::Timestamp>().unwrap().as_millisecond()
}
#[test]
fn source_time_expressions_use_one_shared_clock() {
    let clock = fixture();
    assert_eq!(clock.now.to_string(), "2026-10-07T10:00:00Z");
    for (text, expected) in [
        ("7d", "2026-09-30T10:00:00Z"),
        ("3h", "2026-10-07T07:00:00Z"),
        ("09:00", "2026-10-07T07:00:00Z"),
        ("2026-10-03T14:30", "2026-10-03T12:30:00Z"),
    ] {
        assert_eq!(
            resolve(text, &clock).unwrap().value.start_ms(),
            ms(expected),
            "{text}"
        );
    }
    let day = resolve("2026-10-05", &clock).unwrap().value;
    assert_eq!(day.start_ms(), ms("2026-10-05T02:00:00Z"));
    assert_eq!(day.end_ms().unwrap(), ms("2026-10-06T02:00:00Z"));
    assert_eq!(
        resolve("today", &clock).unwrap().value,
        resolve("2026-10-07", &clock).unwrap().value
    );
    assert_eq!(
        resolve("this week", &clock).unwrap().value.start_ms(),
        day.start_ms()
    );
    assert!(resolve("soon", &clock).is_err());
}
#[test]
fn exact_shared_durations_and_product_money_keep_storage_units() {
    for (text, expected) in [
        ("5m", 300_000),
        ("1.5h", 5_400_000),
        ("250ms", 250),
        ("2w", 1_209_600_000),
    ] {
        assert_eq!(duration_ms(text), Some(expected));
    }
    assert_eq!(duration_ms("5x"), None);
    assert_eq!(duration_ms("0.0000001ms"), None);
    assert_eq!(money_nusd("2.90"), Some(2_900_000_000));
    assert_eq!(money_nusd("$1usd"), Some(1_000_000_000));
}

#[test]
fn fractional_query_bounds_select_exact_integer_index_instants() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    let read = |value: Time| {
        conn.prepare("SELECT column1 FROM (VALUES(-1),(0),(1),(2)) WHERE column1>=?1 AND column1<?2 ORDER BY column1").unwrap().query_map([value.start_ms(),value.end_ms().unwrap()],|row|row.get::<_,i64>(0)).unwrap().collect::<rusqlite::Result<Vec<_>>>().unwrap()
    };
    let at = |text: &str| text.parse::<jiff::Timestamp>().unwrap();
    assert_eq!(
        read(Time::Span {
            start: at("1970-01-01T00:00:00.0002Z"),
            end: at("1970-01-01T00:00:00.0012Z")
        }),
        [1]
    );
    assert!(
        read(Time::Instant {
            at: at("1970-01-01T00:00:00.0005Z")
        })
        .is_empty()
    );
    assert!(
        read(Time::Instant {
            at: at("1969-12-31T23:59:59.9995Z")
        })
        .is_empty()
    );
    assert_eq!(
        read(Time::Instant {
            at: at("1970-01-01T00:00:00.001Z")
        }),
        [1]
    );
}

#[test]
fn fractional_compiled_since_until_and_comparisons_keep_index_boundaries_exclusive() {
    use crate::fixture::{Fixture, MessageFacts, SessionFacts};
    let temp = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(temp.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:fractional",
            ..Default::default()
        })
        .unwrap();
    for (id, at) in [("before", -1), ("zero", 0), ("one", 1), ("two", 2)] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "user",
                    written_by: "human",
                    at: Some(at),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let options = polylogue_query::Options {
        clock: Clock::new(
            "2026-10-07T10:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::UTC,
                jiff::civil::Time::midnight(),
            )
            .unwrap(),
        ),
        origins: vec!["test".into()],
        command_words: vec![],
        page_size: None,
        random_seed: Some(1),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        kept: Default::default(),
        fields: vec!["id".into()],
        chat_pages: vec![],
    };
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    for (text, expected) in [
        (
            "messages where since 1970-01-01T00:00:00.0005Z | sort by at",
            vec!["one", "two"],
        ),
        (
            "messages where until 1970-01-01T00:00:00.0005Z | sort by at",
            vec!["before", "zero"],
        ),
        (
            "messages where at >= 1970-01-01T00:00:00.0005Z | sort by at",
            vec!["one", "two"],
        ),
        (
            "messages where at < 1970-01-01T00:00:00.0012Z | sort by at",
            vec!["before", "zero", "one"],
        ),
        ("messages where at = 1970-01-01T00:00:00.0005Z", vec![]),
    ] {
        let program = crate::query::compile(text, &options).unwrap();
        let params = program
            .params
            .iter()
            .map(|param| match param {
                crate::query::Param::Int(value) => crate::SqlValue::Int(*value),
                crate::query::Param::Real(value) => crate::SqlValue::Real(*value),
                crate::query::Param::Text(value) => crate::SqlValue::Text(value.clone()),
            })
            .collect::<Vec<_>>();
        let rows = reader
            .with_clock(&program.clock, || reader.query(&program.sql, &params))
            .unwrap();
        assert_eq!(
            rows.iter()
                .map(|row| row["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            expected,
            "{text}"
        );
    }
}

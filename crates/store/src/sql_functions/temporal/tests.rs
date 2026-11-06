use super::*;
use polylogue_model::{TimeEvidence, TimeQuality};
use rusqlite::params;

fn conn() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::sql_functions::register(&conn).unwrap();
    conn.execute_batch("CREATE TABLE source(time BLOB NOT NULL)")
        .unwrap();
    conn
}
fn native(at: &str, quality: TimeQuality, until: Option<&str>) -> Vec<u8> {
    let at = Timestamp::parse_rfc3339(at).unwrap();
    crate::encoding::time_bytes(
        Some(at),
        TimeEvidence {
            quality,
            precision: Some(if until.is_some() {
                Precision::Bounded
            } else {
                Precision::Instant
            }),
            until: until.map(|value| Timestamp::parse_rfc3339(value).unwrap()),
        },
    )
    .unwrap()
}
fn last(conn: &Connection, times: &[Vec<u8>]) -> Vec<u8> {
    conn.execute("DELETE FROM source", []).unwrap();
    for time in times {
        conn.execute("INSERT INTO source(time) VALUES(?1)", [time])
            .unwrap();
    }
    conn.query_row("SELECT pl_time_latest(time) FROM source", [], |r| r.get(0))
        .unwrap()
}
fn attribute(conn: &Connection, bytes: &[u8], name: &str) -> Option<String> {
    conn.query_row(
        "SELECT pl_time_attribute(?1,'latest',?2)",
        params![bytes, name],
        |r| r.get(0),
    )
    .unwrap()
}
fn before(conn: &Connection, bytes: &[u8], at: &str) -> Option<bool> {
    let predicate = NativePredicate::Compare {
        condition: query_language::time::TimeCondition::Before {
            at: at.parse().unwrap(),
        },
        invert: false,
    };
    conn.query_row(
        "SELECT pl_time_test(?1,'latest',?2)",
        params![bytes, serde_json::to_string(&predicate).unwrap()],
        |r| r.get(0),
    )
    .unwrap()
}

#[test]
fn latest_activity_keeps_the_possible_maximum_of_whole_periods() {
    let conn = conn();
    let coarse = native(
        "1970-01-01T00:00:00Z",
        TimeQuality::Intrinsic,
        Some("1970-01-01T00:00:10Z"),
    );
    let point = native("1970-01-01T00:00:05Z", TimeQuality::Intrinsic, None);
    let bytes = last(&conn, &[coarse.clone(), point.clone()]);
    let value = evidence(Some(&bytes), "latest").unwrap();
    assert_eq!(value.at.unwrap().to_string(), "1970-01-01T00:00:05Z");
    assert_eq!(value.until.unwrap().to_string(), "1970-01-01T00:00:10Z");
    assert_eq!(
        attribute(&conn, &bytes, "precision").as_deref(),
        Some("bounded")
    );
    assert_eq!(before(&conn, &bytes, "1970-01-01T00:00:07Z"), None);
    assert_eq!(before(&conn, &bytes, "1970-01-01T00:00:10Z"), Some(true));
    assert_eq!(before(&conn, &bytes, "1970-01-01T00:00:05Z"), Some(false));
    assert_eq!(bytes, last(&conn, &[point, coarse]));
}

#[test]
fn latest_activity_does_not_invent_a_single_source_quality() {
    let conn = conn();
    let captured = native(
        "1970-01-01T00:00:00Z",
        TimeQuality::Captured,
        Some("1970-01-01T00:00:10Z"),
    );
    let intrinsic = native("1970-01-01T00:00:05Z", TimeQuality::Intrinsic, None);
    let mixed = last(&conn, &[captured, intrinsic.clone()]);
    assert_eq!(attribute(&conn, &mixed, "quality"), None);
    assert_eq!(
        attribute(&conn, &mixed, "precision").as_deref(),
        Some("bounded")
    );
    let older = native("1970-01-01T00:00:01Z", TimeQuality::Captured, None);
    let precise = last(&conn, &[older, intrinsic]);
    assert_eq!(
        attribute(&conn, &precise, "quality").as_deref(),
        Some("intrinsic")
    );
    assert_eq!(
        attribute(&conn, &precise, "precision").as_deref(),
        Some("instant")
    );
}

#[test]
fn latest_activity_cannot_resolve_a_source_calendar_period_in_the_reader_zone() {
    let conn = conn();
    let unresolved = crate::encoding::time_bytes(
        Timestamp::parse_rfc3339("1970-01-01T00:00:00Z"),
        TimeEvidence::intrinsic(Precision::Day),
    )
    .unwrap();
    let point = native("1970-01-01T00:00:05Z", TimeQuality::Intrinsic, None);
    let bytes = last(&conn, &[unresolved, point]);
    assert_eq!(attribute(&conn, &bytes, "precision"), None);
    assert_eq!(before(&conn, &bytes, "1970-01-02T00:00:00Z"), None);
    let known = serde_json::to_string(&NativePredicate::Known).unwrap();
    assert!(
        conn.query_row(
            "SELECT pl_time_test(?1,'latest',?2)",
            params![bytes, known],
            |r| r.get::<_, bool>(0)
        )
        .unwrap()
    );
}

fn relates(
    conn: &Connection,
    a: &[u8],
    b: &[u8],
    relation: &str,
    window_ns: Option<&str>,
    point: &str,
) -> Option<bool> {
    conn.query_row(
        "SELECT pl_time_order(?1,'tuple',?2,'tuple',?3,?4,?5)",
        params![a, b, relation, window_ns, point],
        |r| r.get(0),
    )
    .unwrap()
}
fn distance(conn: &Connection, a: &[u8], b: &[u8], point: &str) -> f64 {
    conn.query_row(
        "SELECT pl_time_distance(?1,'tuple',?2,'tuple',?3)",
        params![a, b, point],
        |r| r.get(0),
    )
    .unwrap()
}

#[test]
fn temporal_order_uses_whole_half_open_periods_and_the_outer_anchor() {
    let conn = conn();
    let day = native(
        "1970-01-01T00:00:00Z",
        TimeQuality::Intrinsic,
        Some("1970-01-01T00:00:10Z"),
    );
    let point = native("1970-01-01T00:00:05Z", TimeQuality::Intrinsic, None);
    let late = native("1970-01-01T00:00:15Z", TimeQuality::Intrinsic, None);
    assert_eq!(
        relates(&conn, &day, &point, "before", None, "when"),
        Some(false)
    );
    assert_eq!(
        relates(&conn, &day, &point, "after", None, "when"),
        Some(false)
    );
    assert_eq!(
        relates(&conn, &point, &day, "during", None, "when"),
        Some(true)
    );
    assert_eq!(
        relates(&conn, &day, &point, "containing", None, "when"),
        Some(true)
    );
    assert_eq!(
        relates(&conn, &day, &late, "before", Some("20000000000"), "when"),
        Some(true)
    );
    assert_eq!(
        relates(&conn, &day, &point, "near", Some("5000000000"), "when"),
        Some(false)
    );
    assert_eq!(distance(&conn, &day, &late, "when"), 15.0);
    assert_eq!(distance(&conn, &day, &late, "start"), 15.0);
    assert_eq!(distance(&conn, &day, &late, "end"), 5.0);
    assert_eq!(distance(&conn, &late, &day, "end"), -15.0);
    let start = native("1970-01-01T00:00:00Z", TimeQuality::Intrinsic, None);
    assert_eq!(
        relates(&conn, &day, &start, "containing", None, "when"),
        Some(true)
    );
    assert_eq!(
        relates(&conn, &day, &day, "containing", None, "when"),
        Some(true)
    );
    assert_eq!(
        relates(&conn, &point, &point, "containing", None, "when"),
        Some(true)
    );
    assert_eq!(
        relates(&conn, &point, &day, "containing", None, "when"),
        Some(false)
    );
    // The span's end is excluded from membership.
    let end = native("1970-01-01T00:00:10Z", TimeQuality::Intrinsic, None);
    assert_eq!(
        relates(&conn, &day, &end, "containing", None, "when"),
        Some(false)
    );
    assert_eq!(
        relates(&conn, &end, &day, "during", None, "when"),
        Some(false)
    );
}

#[test]
fn temporal_order_preserves_nanoseconds_at_large_absolute_timestamps() {
    let conn = conn();
    let first = native(
        "9999-12-30T00:00:00.000000001Z",
        TimeQuality::Operator,
        None,
    );
    let next = native(
        "9999-12-30T00:00:00.000000002Z",
        TimeQuality::Operator,
        None,
    );
    assert_eq!(
        relates(&conn, &first, &next, "before", Some("1"), "when"),
        Some(true)
    );
    assert_eq!(
        relates(&conn, &first, &next, "near", Some("0"), "when"),
        Some(false)
    );
    assert_eq!(distance(&conn, &first, &next, "when"), 1e-9);
    assert_eq!(distance(&conn, &next, &first, "when"), -1e-9);
}

#[test]
fn temporal_order_keeps_unknown_separate_from_a_known_nonmatch() {
    let conn = conn();
    let unknown = crate::encoding::time_bytes(None, TimeEvidence::UNKNOWN).unwrap();
    let point = native("1970-01-01T00:00:05Z", TimeQuality::Intrinsic, None);
    assert_eq!(
        relates(&conn, &unknown, &point, "before", None, "when"),
        None
    );
    assert_eq!(
        relates(&conn, &point, &unknown, "before", None, "when"),
        None
    );
    assert_eq!(
        relates(&conn, &point, &point, "before", None, "when"),
        Some(false)
    );
    let empty: Option<Vec<u8>> = conn
        .query_row("SELECT pl_time_latest(time) FROM source", [], |r| r.get(0))
        .unwrap();
    assert_eq!(empty, None);
}

#[test]
fn session_last_activity_field_reads_real_message_evidence() {
    use crate::fixture::{Fixture, MessageFacts, SessionFacts};
    use crate::query::fields::{self, Aliases};
    use crate::query::units::Unit;
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:clock",
            last: Some(5000),
            ..Default::default()
        })
        .unwrap();
    let coarse = fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:coarse",
                at: Some(0),
                role: "assistant",
                written_by: "model",
                ..Default::default()
            },
        )
        .unwrap();
    fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:point",
                at: Some(5000),
                role: "assistant",
                written_by: "model",
                ..Default::default()
            },
        )
        .unwrap();
    let conn = Connection::open(fixture.archive.index_db()).unwrap();
    crate::sql_functions::register(&conn).unwrap();
    conn.execute(
        "UPDATE message SET at_time=?2 WHERE pk=?1",
        params![
            coarse,
            native(
                "1970-01-01T00:00:00Z",
                TimeQuality::Intrinsic,
                Some("1970-01-01T00:00:10Z")
            )
        ],
    )
    .unwrap();
    let field = fields::native_time(Unit::Sessions, "last_active", &Aliases::plain()).unwrap();
    let sql = format!(
        "SELECT pl_time_attribute({}, '{}', 'precision'), pl_time_order({}, '{}', ?1, 'tuple', 'before', NULL, 'when') FROM session s WHERE s.id='claude-code:clock'",
        field.bytes,
        field.storage.code(),
        field.bytes,
        field.storage.code()
    );
    let later = native("1970-01-01T00:00:08Z", TimeQuality::Intrinsic, None);
    let (precision, before): (Option<String>, Option<bool>) = conn
        .query_row(&sql, [later], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap();
    assert_eq!(precision.as_deref(), Some("bounded"));
    assert_eq!(
        before,
        Some(false),
        "the possible last activity extends beyond the inner point"
    );
}

#[test]
fn nearest_distance_keys_preserve_nanoseconds_across_large_distances() {
    let conn = conn();
    let origin = native("1970-01-01T00:00:00Z", TimeQuality::Intrinsic, None);
    let first = native(
        "9999-12-30T00:00:00.000000001Z",
        TimeQuality::Intrinsic,
        None,
    );
    let second = native(
        "9999-12-30T00:00:00.000000002Z",
        TimeQuality::Intrinsic,
        None,
    );
    let key = |outer: &[u8], inner: &[u8]| -> String {
        conn.query_row(
            "SELECT pl_time_distance_key(?1,'tuple',?2,'tuple','when')",
            params![outer, inner],
            |row| row.get(0),
        )
        .unwrap()
    };
    let first_key = key(&origin, &first);
    let second_key = key(&origin, &second);
    assert_eq!(first_key.len(), 39);
    assert!(first_key < second_key);
    assert_eq!(
        second_key.parse::<u128>().unwrap() - first_key.parse::<u128>().unwrap(),
        1
    );
    assert_eq!(first_key, key(&first, &origin));
    let unknown: Option<String> = conn
        .query_row(
            "SELECT pl_time_distance_key(NULL,'tuple',?1,'tuple','when')",
            [&first],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(unknown, None);
}

#[test]
fn matched_occurrence_preserves_uncertainty_without_inventing_state_endpoints() {
    let conn = conn();
    let outer = native("1970-01-01T00:00:05Z", TimeQuality::Intrinsic, None);
    let inner = native(
        "1970-01-01T00:00:00Z",
        TimeQuality::Intrinsic,
        Some("1970-01-01T00:00:10Z"),
    );
    assert_eq!(
        relates(&conn, &outer, &inner, "before", None, "match.when"),
        Some(false)
    );
    assert_eq!(
        relates(&conn, &outer, &inner, "after", None, "match.when"),
        Some(false)
    );
    assert_eq!(
        relates(
            &conn,
            &outer,
            &inner,
            "near",
            Some("5000000000"),
            "match.when"
        ),
        Some(false)
    );
    assert_eq!(
        relates(
            &conn,
            &outer,
            &inner,
            "near",
            Some("10000000000"),
            "match.when"
        ),
        Some(true)
    );
    for selector in ["match.start", "match.end"] {
        assert_eq!(
            relates(&conn, &outer, &inner, "before", None, selector),
            None
        );
        let distance: Option<f64> = conn
            .query_row(
                "SELECT pl_time_distance(?1,'tuple',?2,'tuple',?3)",
                params![&outer, &inner, selector],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(distance, None);
    }
    assert_eq!(
        relates(&conn, &outer, &inner, "during", None, "match.when"),
        None
    );
    assert_eq!(
        relates(&conn, &inner, &outer, "containing", None, "match.when"),
        None
    );
}

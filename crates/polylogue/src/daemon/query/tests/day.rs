//! The logical day is supplied by the operation owner.
use super::*;
#[test]
fn executions_keep_their_explicit_day_in_the_resolved_context() {
    let archive = morrow();
    let mut request = ask("since:2026-10-05");
    request.clock.day = fixture_clock(0, "Pacific/Kiritimati", 0).day;
    let result = run_q(&archive, request).unwrap();
    assert_eq!(
        result["resolved"]["clock"]["day"],
        json!({"zone":"Pacific/Kiritimati","start":"00:00:00"})
    );
}

#[test]
fn until_date_uses_logical_day_end_and_until_instant_stays_exclusive() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:time-bounds",
            ..Default::default()
        })
        .unwrap();
    // This configured logical day crosses Warsaw's autumn clock change.
    for (id, timestamp) in [
        ("before", "2026-10-24T01:59:59Z"),
        ("start", "2026-10-24T02:00:00Z"),
        ("inside", "2026-10-24T12:00:00Z"),
        ("last", "2026-10-25T02:59:59Z"),
        ("end", "2026-10-25T03:00:00Z"),
    ] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "user",
                    written_by: "human",
                    at: Some(
                        timestamp
                            .parse::<jiff::Timestamp>()
                            .unwrap()
                            .as_millisecond(),
                    ),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    for (text, expected) in [
        (
            "messages where until 2026-10-24 | sort by at",
            vec!["before", "start", "inside", "last"],
        ),
        (
            "messages where at < 2026-10-24 | sort by at",
            vec!["before"],
        ),
        (
            "messages where until 2026-10-24T12:00:00Z | sort by at",
            vec!["before", "start"],
        ),
        (
            "messages where since 2026-10-24 until 2026-10-24 | sort by at",
            vec!["start", "inside", "last"],
        ),
    ] {
        let mut request = ask(text);
        request.clock.day = fixture_clock(0, "Europe/Warsaw", 240).day;
        let answer = run(&reader, request).unwrap();
        let actual: Vec<&str> = answer["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].as_str().unwrap())
            .collect();
        assert_eq!(actual, expected, "{text}");
    }
}

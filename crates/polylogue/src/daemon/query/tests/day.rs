//! The logical day comes from the operator's settings.

use super::*;

#[test]
fn the_logical_day_follows_its_settings() {
    let mut w = morrow();
    let mut a = ask("since:2026-10-05");
    a.day = None;
    let machine_default = run_q(&w, a).unwrap();
    w.fx.setting("day.zone", "Pacific/Kiritimati").unwrap();
    w.fx.setting("day.logical_start", "00:00").unwrap();
    let pool = w.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    let day = logical_day(&reader).unwrap();
    assert_eq!(day.zone_name(), Some("Pacific/Kiritimati"));
    assert_eq!(day.start_minutes, 0);
    drop(reader);
    let mut a = ask("since:2026-10-05");
    a.day = None;
    let resolved = run_q(&w, a).unwrap();
    assert_eq!(resolved["resolved"]["zone"], "Pacific/Kiritimati");
    assert_eq!(resolved["resolved"]["day_start_minutes"], 0);
    assert_ne!(resolved["resolved"], machine_default["resolved"]);
    w.fx.setting("day.zone", "Mars/Olympus").unwrap();
    let mut a = ask("kind:*");
    a.day = None;
    let refused = run_q(&w, a).unwrap_err();
    assert!(refused.message.contains("day.zone"), "{}", refused.message);
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
        request.day = Some(LogicalDay::named("Europe/Warsaw", 240).unwrap());
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

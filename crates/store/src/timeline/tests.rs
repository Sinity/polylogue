use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

fn add(f: &mut Fixture, session: i64, id: &str, at: Option<i64>) {
    f.message(
        session,
        MessageFacts {
            id,
            at,
            role: "assistant",
            written_by: "model",
            model: Some("synthetic"),
            cost_nusd: Some(5_000_000),
            ..Default::default()
        },
    )
    .unwrap();
}
fn at(s: &str) -> i64 {
    s.parse::<jiff::Timestamp>().unwrap().as_millisecond()
}
fn all(date: Option<jiff::civil::Date>, across_years: bool, undated: bool) -> Selection<'static> {
    Selection {
        date,
        across_years,
        undated,
        responses: None,
        model: None,
        unknown_model: false,
    }
}

#[test]
fn a_session_crossing_logical_start_and_dst_uses_each_records_time() {
    let temp = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(temp.path()).unwrap();
    let session = f
        .session(SessionFacts {
            id: "codex:overnight",
            kind: "main",
            started: Some(at("2024-03-29T12:00:00Z")),
            ..Default::default()
        })
        .unwrap();
    for (id, time) in [
        ("codex:before", "2024-03-31T01:59:59Z"),
        ("codex:start", "2024-03-31T02:00:00Z"),
        ("codex:last", "2024-04-01T01:59:59Z"),
        ("codex:next", "2024-04-01T02:00:00Z"),
    ] {
        add(&mut f, session, id, Some(at(time)));
    }
    f.event(session, "synthetic", Some(at("2024-03-31T03:00:00Z")))
        .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let day = LogicalDay::named("Europe/Warsaw", 240).unwrap();
    let date = "2024-03-31".parse().unwrap();
    let rows = page(&reader, &day, &all(Some(date), false, false), None, 50).unwrap();
    assert_eq!(rows["rows"].as_array().unwrap().len(), 3);
    assert_eq!(rows["rows"][0]["id"], "codex:start");
    assert_eq!(rows["rows"][2]["id"], "codex:last");
    let counts = calendar(&reader, &day, date).unwrap();
    assert_eq!(
        counts.iter().find(|r| r["date"] == "2024-03-31").unwrap()["records"],
        3
    );
    let bounds = day.day(date);
    let selection = crate::measure::Selection {
        since: Some(bounds.0),
        until: Some(bounds.1),
        ..Default::default()
    };
    let usage = crate::measure::usage(&reader, &selection).unwrap();
    assert_eq!(usage.known_response_messages, 2);
    assert_eq!(usage.by_model[0].messages, 2);
    let responses = Selection {
        date: None,
        across_years: false,
        undated: false,
        responses: Some(&selection),
        model: Some("synthetic"),
        unknown_model: false,
    };
    let rows = page(&reader, &day, &responses, None, 50).unwrap();
    assert_eq!(rows["rows"].as_array().unwrap().len(), 2);
}

#[test]
fn every_leap_year_record_and_undated_record_is_reachable() {
    let temp = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(temp.path()).unwrap();
    let session = f
        .session(SessionFacts {
            id: "codex:years",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    for year in [1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024] {
        add(
            &mut f,
            session,
            &format!("codex:{year}"),
            Some(at(&format!("{year}-02-29T12:00:00Z"))),
        );
    }
    add(
        &mut f,
        session,
        "codex:ancient",
        Some(
            LogicalDay::named("UTC", 240)
                .unwrap()
                .start_of(jiff::civil::date(-400, 2, 29))
                + 8 * 3_600_000,
        ),
    );
    add(&mut f, session, "codex:unknown", None);
    add(&mut f, session, "codex:a-unrepresentable", Some(i64::MAX));
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let day = LogicalDay::named("UTC", 240).unwrap();
    let selection = all(Some("2024-02-29".parse().unwrap()), true, false);
    let mut after = None;
    let mut ids = Vec::new();
    loop {
        let result = page(&reader, &day, &selection, after.as_ref(), 3).unwrap();
        ids.extend(
            result["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r["id"].as_str().unwrap().to_owned()),
        );
        after = result
            .get("after")
            .filter(|p| !p.is_null())
            .map(|p| serde_json::from_value(p.clone()).unwrap());
        if after.is_none() {
            break;
        }
    }
    assert_eq!(ids.len(), 9);
    assert_eq!(ids[0], "codex:ancient");
    assert_eq!(ids[8], "codex:2024");
    let ancient = page(
        &reader,
        &day,
        &all(Some(jiff::civil::date(-400, 2, 29)), true, false),
        None,
        50,
    )
    .unwrap();
    assert_eq!(ancient["rows"].as_array().unwrap().len(), 1);
    let first = page(&reader, &day, &all(None, false, true), None, 1).unwrap();
    let next: Position = serde_json::from_value(first["after"].clone()).unwrap();
    let second = page(&reader, &day, &all(None, false, true), Some(&next), 1).unwrap();
    assert_eq!(first["rows"][0]["id"], "codex:a-unrepresentable");
    assert_eq!(second["rows"][0]["id"], "codex:unknown");
    let unknown = page(&reader, &day, &all(None, false, true), None, 50).unwrap();
    assert_eq!(unknown["rows"].as_array().unwrap().len(), 2);
    assert!(
        unknown["rows"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["date"].is_null())
    );
}

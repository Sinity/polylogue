use super::*;
use crate::fixture::{Fixture, SessionFacts};

fn kinds_from(
    reader: &Reader<'_>,
    selection: &PreparedSelection<'_>,
    from: Option<&OrderPosition>,
) -> Vec<String> {
    let mut out = vec![];
    reader
        .visit_selection::<SelectionError>(
            selection,
            from.map_or(crate::read::VisitFrom::First, crate::read::VisitFrom::At),
            None,
            |row| {
                out.push(row.values["kind"].as_str().unwrap().to_owned());
                Ok(ControlFlow::Continue(()))
            },
        )
        .unwrap();
    out
}

fn positions(reader: &Reader<'_>, selection: &PreparedSelection<'_>) -> Vec<OrderPosition> {
    let mut out = vec![];
    reader
        .visit_selection::<SelectionError>(selection, crate::read::VisitFrom::First, None, |row| {
            out.push(row.position);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    out
}

#[test]
fn row_continuation_resumes_at_its_start_key() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:session",
            ..Default::default()
        })
        .unwrap();
    fixture.event(session, "synthetic-a", None).unwrap();
    fixture.event(session, "synthetic-b", None).unwrap();
    fixture.event(session, "synthetic-c", None).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let selection = reader
        .prepare_selection("events | sort by id", &options())
        .unwrap();
    let keys = positions(&reader, &selection);
    drop(reader);
    drop(pool);
    // An earlier body that cannot decode is never read by a continuation.
    let conn = crate::archive::open_write(&fixture.archive.index_db(), 1).unwrap();
    conn.execute(
        "UPDATE event SET data=X'ff' WHERE id=(SELECT min(id) FROM event)",
        [],
    )
    .unwrap();
    drop(conn);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let selection = reader
        .prepare_selection("events | sort by id", &options())
        .unwrap();
    assert_eq!(
        kinds_from(&reader, &selection, Some(&keys[1])),
        ["synthetic-b", "synthetic-c"]
    );
    assert_eq!(
        kinds_from(&reader, &selection, Some(&keys[2])),
        ["synthetic-c"]
    );
}

fn options() -> Options {
    let clock = query_language::time::Clock::new(
        "2026-10-29T12:00:00Z".parse().unwrap(),
        query_language::time::DaySetting::new(
            jiff::tz::TimeZone::get("Etc/UTC").unwrap(),
            jiff::civil::Time::midnight(),
        )
        .unwrap(),
    );
    Options {
        clock,
        origins: vec!["test".into()],
        page_size: Some(1),
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        here: None,
        seen_marks_of: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        provider_catalog: None,
    }
}

#[test]
fn reverse_selection_mirrors_nulls_ties_and_selection_stages() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:session",
            ..Default::default()
        })
        .unwrap();
    fixture.event(session, "unknown-a", None).unwrap();
    fixture.event(session, "early", Some(1000)).unwrap();
    fixture.event(session, "late-a", Some(2000)).unwrap();
    fixture.event(session, "late-b", Some(2000)).unwrap();
    fixture.event(session, "unknown-b", None).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let options = options();
    for (query, expected) in [
        (
            "events | sort by at desc, id asc",
            vec!["late-a", "late-b", "early", "unknown-a", "unknown-b"],
        ),
        (
            "events | sort by at asc, id desc",
            vec!["early", "late-b", "late-a", "unknown-b", "unknown-a"],
        ),
        (
            "events | sort by at desc, id asc | offset 1 | limit 3",
            vec!["late-b", "early", "unknown-a"],
        ),
    ] {
        let selection = reader.prepare_selection(query, &options).unwrap();
        let keys = positions(&reader, &selection);
        assert_eq!(kinds_from(&reader, &selection, None), expected, "{query}");
        for (end, key) in keys.iter().enumerate() {
            let mut reverse = vec![];
            reader
                .visit_selection::<SelectionError>(
                    &selection,
                    crate::read::VisitFrom::Before(key),
                    None,
                    |row| {
                        reverse.push(row.values["kind"].as_str().unwrap().to_owned());
                        Ok(ControlFlow::Continue(()))
                    },
                )
                .unwrap();
            let before: Vec<String> = expected[..end]
                .iter()
                .rev()
                .map(|kind| (*kind).to_owned())
                .collect();
            assert_eq!(reverse, before, "{query}, before row {end}");
            let after: Vec<String> = expected[end..]
                .iter()
                .map(|kind| (*kind).to_owned())
                .collect();
            assert_eq!(
                kinds_from(&reader, &selection, Some(key)),
                after,
                "{query}, from row {end}"
            );
        }
    }
}

#[test]
fn reverse_callback_stops_before_poisoned_earlier_body() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:session",
            ..Default::default()
        })
        .unwrap();
    fixture.event(session, "poison", None).unwrap();
    fixture.event(session, "previous", None).unwrap();
    fixture.event(session, "start", None).unwrap();
    let starts: Vec<(String, OrderPosition)> = {
        let pool = fixture.reads().unwrap();
        let reader = pool.reader().unwrap();
        ["events | sort by id", "events | sort by id | limit 3"]
            .into_iter()
            .map(|query| {
                let selection = reader.prepare_selection(query, &options()).unwrap();
                (query.to_owned(), positions(&reader, &selection)[2].clone())
            })
            .collect()
    };
    let conn = crate::archive::open_write(&fixture.archive.index_db(), 1).unwrap();
    conn.execute("UPDATE event SET data=X'ff' WHERE id='e1'", [])
        .unwrap();
    drop(conn);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    for (query, start) in starts {
        let selection = reader.prepare_selection(&query, &options()).unwrap();
        let mut found = vec![];
        reader
            .visit_selection::<SelectionError>(
                &selection,
                crate::read::VisitFrom::Before(&start),
                None,
                |row| {
                    found.push(row.values["kind"].clone());
                    Ok(ControlFlow::Break(()))
                },
            )
            .unwrap();
        assert_eq!(found, vec![Value::String("previous".into())]);
        assert!(
            reader
                .visit_selection::<SelectionError>(
                    &selection,
                    crate::read::VisitFrom::Before(&start),
                    None,
                    |_| Ok(ControlFlow::Continue(()))
                )
                .is_err()
        );
    }
}

#[test]
fn last_selection_visit_reverses_the_complete_forward_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:reverse",
            ..Default::default()
        })
        .unwrap();
    for (kind, at) in [
        ("unknown", None),
        ("early", Some(1000)),
        ("late-a", Some(2000)),
        ("late-b", Some(2000)),
    ] {
        fixture.event(session, kind, at).unwrap();
    }
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    for (query, expected) in [
        (
            "events | sort by at desc, id asc",
            vec!["late-a", "late-b", "early", "unknown"],
        ),
        (
            "events | sort by at desc, id asc | offset 1 | limit 2",
            vec!["late-b", "early"],
        ),
    ] {
        let selection = reader.prepare_selection(query, &options()).unwrap();
        let forward = kinds_from(&reader, &selection, None);
        assert_eq!(forward, expected, "{query}");
        let mut reverse = Vec::new();
        reader
            .visit_selection::<SelectionError>(
                &selection,
                crate::read::VisitFrom::Last,
                None,
                |row| {
                    reverse.push(row.values["kind"].as_str().unwrap().to_owned());
                    Ok(ControlFlow::Continue(()))
                },
            )
            .unwrap();
        assert_eq!(
            reverse,
            forward.into_iter().rev().collect::<Vec<_>>(),
            "{query}"
        );
    }
}

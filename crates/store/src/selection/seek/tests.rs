use super::*;
use crate::fixture::{Fixture, SessionFacts};

#[test]
fn compact_row_seek_does_not_decode_preceding_answer_bodies() {
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
    let conn = crate::archive::open_write(&fixture.archive.index_db(), 1).unwrap();
    conn.execute(
        "UPDATE event SET data=X'ff' WHERE id=(SELECT min(id) FROM event)",
        [],
    )
    .unwrap();
    drop(conn);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let options = options();
    let selection = reader
        .prepare_selection("events | sort by id", &options)
        .unwrap();
    assert!(
        reader
            .visit_selection::<SelectionError>(&selection, None, None, |_| Ok(
                ControlFlow::Continue(())
            ))
            .is_err()
    );
    let mut rows = vec![];
    reader
        .visit_selection_from::<SelectionError>(&selection, 1, None, |ordinal, row| {
            rows.push((
                ordinal,
                row.values["kind"].clone(),
                row.values["data"].clone(),
            ));
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(
        rows,
        vec![(
            1,
            Value::String("synthetic-b".into()),
            serde_json::json!({})
        )]
    );
    assert_eq!(
        reader
            .visit_selection_from::<SelectionError>(&selection, 3, None, |_, _| Ok(
                ControlFlow::Continue(())
            ))
            .unwrap_err()
            .kind(),
        "stale_cursor"
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
        command_words: vec![],
        page_size: Some(1),
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        fields: vec![],
        kept: Default::default(),
        chat_pages: vec![],
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
        let mut forward = vec![];
        reader
            .visit_selection::<SelectionError>(&selection, None, None, |row| {
                forward.push(row.values["kind"].as_str().unwrap().to_owned());
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        assert_eq!(forward, expected, "{query}");
        for end_row in 0..expected.len() {
            let mut reverse = vec![];
            reader
                .visit_selection_before::<SelectionError>(
                    &selection,
                    end_row as u64,
                    None,
                    |ordinal, row| {
                        reverse.push((ordinal, row.values["kind"].as_str().unwrap().to_owned()));
                        Ok(ControlFlow::Continue(()))
                    },
                )
                .unwrap();
            let expected = expected[..end_row]
                .iter()
                .enumerate()
                .rev()
                .map(|(i, kind)| (i as u64, (*kind).to_owned()))
                .collect::<Vec<_>>();
            assert_eq!(reverse, expected, "{query}, before row {end_row}");
        }
        assert_eq!(
            reader
                .visit_selection_before::<SelectionError>(&selection, 99, None, |_, _| Ok(
                    ControlFlow::Continue(())
                ))
                .unwrap_err()
                .kind(),
            "stale_cursor"
        );
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
    fixture.event(session, "anchor", None).unwrap();
    let conn = crate::archive::open_write(&fixture.archive.index_db(), 1).unwrap();
    conn.execute("UPDATE event SET data=X'ff' WHERE id='e1'", [])
        .unwrap();
    drop(conn);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let options = options();
    for query in ["events | sort by id", "events | sort by id | limit 3"] {
        let selection = reader.prepare_selection(query, &options).unwrap();
        let mut found = vec![];
        reader
            .visit_selection_before::<SelectionError>(&selection, 2, None, |ordinal, row| {
                found.push((ordinal, row.values["kind"].clone()));
                Ok(ControlFlow::Break(()))
            })
            .unwrap();
        assert_eq!(found, vec![(1, Value::String("previous".into()))]);
        assert!(
            reader
                .visit_selection_before::<SelectionError>(&selection, 2, None, |_, _| Ok(
                    ControlFlow::Continue(())
                ))
                .is_err()
        );
    }
}

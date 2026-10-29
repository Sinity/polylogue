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
    let clock = query_language::time::Clock::new(
        "2026-10-29T12:00:00Z".parse().unwrap(),
        query_language::time::DaySetting::new(
            jiff::tz::TimeZone::get("Etc/UTC").unwrap(),
            jiff::civil::Time::midnight(),
        )
        .unwrap(),
    );
    let options = Options {
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
    };
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

use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
fn options() -> Options {
    Options {
        clock: query_language::time::Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("Europe/Warsaw").unwrap(),
                jiff::civil::time(4, 0, 0, 0),
            )
            .unwrap(),
        ),
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
fn fixture() -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (id, last) in [("test:one", 1), ("test:two", 2), ("test:three", 3)] {
        let session = f
            .session(SessionFacts {
                id,
                last: Some(last),
                ..Default::default()
            })
            .unwrap();
        f.message(
            session,
            MessageFacts {
                id: &format!("{id}:m"),
                role: "assistant",
                written_by: "model",
                text: Some("synthetic body"),
                ..Default::default()
            },
        )
        .unwrap();
    }
    (dir, f)
}
#[test]
fn prepared_selection_streams_total_answer_and_stops_at_callback() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let prepared = reader.prepare_selection("sessions", &options()).unwrap();
    let mut refs = vec![];
    let mut after = None;
    reader
        .visit_selection::<SelectionError>(&prepared, None, None, |row| {
            refs.push(row.reference.unwrap().to_string());
            after = Some(row.position);
            Ok(ControlFlow::Break(()))
        })
        .unwrap();
    assert_eq!(refs, ["session:test:three"]);
    reader
        .visit_selection::<SelectionError>(&prepared, after.as_ref(), None, |row| {
            refs.push(row.reference.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(
        refs,
        ["session:test:three", "session:test:two", "session:test:one"]
    );
}
#[test]
fn prepared_selection_identity_resolves_latest_without_hashing_execution() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let latest = reader.prepare_selection("latest", &options()).unwrap();
    let explicit = reader
        .prepare_selection("session:test:three", &options())
        .unwrap();
    assert_eq!(latest.question_id(), explicit.question_id());
    let mut changed = options();
    changed.page_size = Some(300);
    changed.random_seed = Some(999);
    assert_eq!(
        latest.question_id(),
        reader
            .prepare_selection("latest", &changed)
            .unwrap()
            .question_id()
    );
}
#[test]
fn prepared_selection_retains_exact_block_reference() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let target = "block:test:one:m#0";
    let prepared = reader.prepare_selection(target, &options()).unwrap();
    let mut refs = vec![];
    reader
        .visit_selection::<SelectionError>(&prepared, None, None, |row| {
            refs.push(row.reference.unwrap().to_string());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(refs, [target]);
}

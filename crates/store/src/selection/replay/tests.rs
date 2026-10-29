use super::*;
use crate::fixture::Fixture;
use query_language::cursor::Continuation;

fn options() -> Options {
    Options {
        clock: Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("Etc/UTC").unwrap(),
                jiff::civil::Time::midnight(),
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

#[test]
fn semantic_preparation_does_not_certify_replay_custody() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(dir.path()).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let selection = reader
        .prepare_query_selection("sessions", &options())
        .unwrap();
    assert_eq!(
        selection
            .authenticated_execution(&reader)
            .err()
            .unwrap()
            .kind(),
        "not_yet_available"
    );
}

#[test]
fn foreign_seals_cannot_authorize_even_a_well_formed_replay_context() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(dir.path()).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let options = options();
    let selection = reader
        .prepare_query_selection("sessions", &options)
        .unwrap();
    let mut key = reader.cursor_key().unwrap();
    key[0] ^= 1;
    let cursor = CursorSeal::new(key)
        .seal(&Continuation {
            context: selection.cursor_context().clone(),
            after: OrderPosition {
                values: vec![],
                row_identity: String::new(),
            },
        })
        .unwrap();
    let error = reader
        .prepare_authenticated_run_selection(
            &cursor,
            selection.question_id(),
            None,
            RunReplayContext {
                current_clock: &options.clock,
                current_versions: &BTreeMap::new(),
                ranking_identity: None,
                page_size: None,
                near: None,
            },
        )
        .err()
        .unwrap();
    assert!(matches!(
        error,
        SelectionError::Language(query_language::Error {
            kind: query_language::ErrorKind::InvalidQuery,
            ..
        })
    ));
}

#[test]
fn an_authentic_cursor_without_root_custody_binding_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(dir.path()).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let options = options();
    let selection = reader
        .prepare_query_selection("sessions", &options)
        .unwrap();
    let cursor = CursorSeal::new(reader.cursor_key().unwrap())
        .seal(&Continuation {
            context: selection.cursor_context().clone(),
            after: OrderPosition {
                values: vec![],
                row_identity: String::new(),
            },
        })
        .unwrap();
    let error = reader
        .prepare_authenticated_run_selection(
            &cursor,
            selection.question_id(),
            None,
            RunReplayContext {
                current_clock: &options.clock,
                current_versions: &BTreeMap::new(),
                ranking_identity: None,
                page_size: None,
                near: None,
            },
        )
        .err()
        .unwrap();
    assert_eq!(error.kind(), "invalid_cursor");
    assert!(error.to_string().contains("read_root_run"));
}

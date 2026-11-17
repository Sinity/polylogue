use super::*;
use std::ops::ControlFlow;

fn copied_event(copy: &SessionId) -> Event {
    Event {
        session: Some(copy.clone()),
        kind: "permission_mode".into(),
        at: None,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: None,
        range: Some((7, 8)),
        data: serde_json::json!({"mode":"plan"}),
    }
}

fn selected(w: &TestArchive, sessions: &[SessionId]) -> Vec<(String, Vec<Event>)> {
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    sessions
        .iter()
        .map(|session| {
            let mut events = Vec::new();
            reader
                .visit_event_records(session, 0, |record| {
                    events.push(record.event);
                    ControlFlow::Continue(())
                })
                .unwrap();
            (session.to_string(), events)
        })
        .collect()
}

fn move_events(restart_after_install: bool) -> Vec<(String, Vec<Event>)> {
    let mut w = test_archive();
    let copy = SessionId::parse("ai-studio:content-copy").unwrap();
    let known = SessionId::parse("ai-studio:drive-file").unwrap();
    let ambiguous = SessionId::parse("ai-studio:other-file").unwrap();
    let mut input = batch("copied", Route::Capture, 0, vec![], &[]);
    input.parse[0]
        .output
        .sessions
        .push(Session::new(copy.clone(), SessionKind::Main));
    input.parse[0].output.events.push(copied_event(&copy));
    w.store_reading(input, Some(&SOURCE_RECORD)).unwrap();
    w.writer.derive().unwrap();
    if restart_after_install {
        w.writer.stop();
        w.archive = w.archive.reopened();
        w.writer = Writer::start(&w.archive).unwrap();
    }
    let declare = |input: &str, id: &SessionId| {
        let mut input = batch(input, Route::Live, 0, vec![], &[]);
        let mut session = Session::new(id.clone(), SessionKind::Main);
        session.aliases.push(copy.clone());
        input.parse[0].output.sessions.push(session);
        input
    };
    w.store_reading(declare("native", &known), Some(&SOURCE_RECORD))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let moved = selected(&w, &[copy.clone(), known.clone()]);
    assert!(moved[0].1.is_empty());
    let mut expected = copied_event(&copy);
    expected.session = Some(known.clone());
    assert_eq!(moved[1].1, [expected]);
    w.store_reading(declare("ambiguous", &ambiguous), Some(&SOURCE_RECORD))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    w.writer.derive().unwrap();
    let released = selected(&w, &[copy.clone(), known, ambiguous]);
    assert_eq!(released[0].1, [copied_event(&copy)]);
    assert!(released[1].1.is_empty());
    assert!(released[2].1.is_empty());
    released
}

#[test]
fn installed_build_moves_copy_events_like_a_restarted_writer() {
    assert_eq!(move_events(false), move_events(true));
}

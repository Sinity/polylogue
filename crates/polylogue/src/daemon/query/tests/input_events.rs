use super::*;
use polylogue_model::{Event, Output, Session, SessionId, SessionKind};
use polylogue_store::ReadPool;
use polylogue_store::fixture::writer::WriterWorld;

#[test]
fn input_observations_are_typed_general_rows_and_do_not_enter_session_scopes() {
    let mut w = WriterWorld::new();
    let sid = SessionId::parse("hermes:conversation").unwrap();
    let data = json!({"approved":false,"exit_code":null,"paths":[" λ.rs "],"nested":{"count":0}});
    let event = |session| Event {
        session,
        kind: "hermes_verification_event".into(),
        at: None,
        after: None,
        range: Some((0, 100)),
        data: data.clone(),
    };
    w.write_output(
        "ledger",
        "hermes",
        "/neutral/verification_evidence.db.jsonl",
        Output {
            sessions: vec![Session::new(sid.clone(), SessionKind::Main)],
            events: vec![event(None), event(Some(sid.clone()))],
            ..Output::default()
        },
        100,
    );
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let all = run(&reader, ask("events")).unwrap();
    assert_eq!(all["count"], 2);
    let rows = all["rows"].as_array().unwrap();
    assert_eq!(rows.iter().filter(|r| r["session"].is_null()).count(), 1);
    for row in rows {
        assert_eq!(row["input"], "ledger");
        assert_eq!(row["source"], "hermes");
        assert_eq!(row["data"], data);
    }
    let mut scoped = ask("events where self");
    scoped.session = Some(sid.to_string());
    assert_eq!(run(&reader, scoped).unwrap()["count"], 1);
    assert_eq!(run(&reader, ask("sessions | events")).unwrap()["count"], 1);
    assert_eq!(reader.events(&sid).unwrap().len(), 1);
    let fields = ["input".into(), "source".into(), "data".into()];
    let mut projection = ask("events");
    projection.fields = &fields;
    let projected = run(&reader, projection).unwrap();
    assert_eq!(projected["rows"][0]["data"], data);
    assert_eq!(
        run(&reader, ask("events where text:λ.rs")).unwrap()["count"],
        2
    );
    assert_eq!(
        run(&reader, ask("events where text:missing")).unwrap()["count"],
        0
    );
    assert_eq!(
        run(&reader, ask("events where text:0")).unwrap()["count"],
        0,
        "numbers do not become text evidence"
    );
    assert_eq!(
        run(&reader, ask("events where text:approved")).unwrap()["count"],
        0,
        "field labels are not supplied textual values"
    );
    let kept = keep(&reader, ask("events")).unwrap();
    assert_eq!(kept["rows"][0]["data"], data);
}

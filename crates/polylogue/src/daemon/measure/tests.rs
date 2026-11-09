use super::run;
use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};
use query_language::time::DaySetting;

#[test]
fn measures_on_a_fresh_reader_see_current_tags_without_a_preceding_query() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for id in ["codex:shown", "codex:hidden"] {
        let session = f
            .session(SessionFacts {
                id,
                kind: "main",
                ..Default::default()
            })
            .unwrap();
        f.message(
            session,
            MessageFacts {
                id: &format!("{id}:reply"),
                at: Some(1_700_000_000_000),
                model: Some("test"),
                role: "assistant",
                written_by: "model",
                cost_nusd: Some(5_000_000),
                ..Default::default()
            },
        )
        .unwrap();
    }
    f.note("hidden", None, None, "session:codex:hidden")
        .unwrap();
    let day = DaySetting::new(jiff::tz::TimeZone::UTC, "00:00:00".parse().unwrap()).unwrap();
    let pool = f.reads().unwrap();

    let reader = pool.reader().unwrap();
    let usage = run(&reader, "usage", None, 1_700_000_100_000, &day, None).unwrap();
    assert_eq!(usage["known_response_messages"], 2);
    assert_eq!(usage["hidden_sessions"], 1);
    drop(reader);

    let reader = pool.reader().unwrap();
    run(&reader, "quota", None, 1_700_000_100_000, &day, None).unwrap();
}

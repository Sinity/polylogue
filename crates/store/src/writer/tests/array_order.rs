use super::*;

#[test]
fn reverse_event_replay_cannot_regress_an_already_observed_logical_position() {
    let w = world();
    let row = MessageRow {
        offset: 100,
        end: 200,
        message: text_message("a", "s", Role::User, "A", None),
    };
    let mut seed = batch("array", 2, 0, vec![row], &["s"]);
    seed.parse[0].output.events.push(Event {
        session: Some(sid("s")),
        kind: "active_leaf".into(),
        at: None,
        after: Some(MessageId::parse("claude-code:a").unwrap()),
        range: Some((100, 200)),
        data: Json::Null,
    });
    seed.parse[0].order_from = Some(9);
    w.writer.send(seed.clone()).unwrap();
    w.writer.flush().unwrap();
    seed.change = Change::Reparse { chain: 1 };
    seed.parse[0].order_from = Some(1);
    w.writer.send(seed).unwrap();
    let result = w.writer.flush().unwrap();
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let rows = reader
        .query("SELECT source_order FROM event_source", &[])
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["source_order"], 9);
}

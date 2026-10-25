use super::*;

#[test]
fn a_parse_from_the_first_byte_restates_event_positions() {
    let w = test_archive();
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
        .query("SELECT seq AS source_order FROM event", &[])
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["source_order"], 1);
}

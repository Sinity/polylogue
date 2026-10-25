use super::*;

/// One record holding several messages (a format that writes a whole
/// exchange per line): the messages share the record's offset, and their
/// IDs sort against transcript order.
#[test]
fn messages_one_record_holds_read_in_record_order() {
    let w = test_archive();
    let record = |offset, ids: &[(&str, Role, &str)]| {
        ids.iter()
            .map(|(id, role, text)| MessageRow {
                offset,
                end: offset + 100,
                message: text_message(id, "s", *role, text, Some("end_turn")),
            })
            .collect::<Vec<_>>()
    };
    let mut rows = record(
        0,
        &[
            ("z", Role::User, "first"),
            ("m", Role::Assistant, "second"),
            ("a", Role::User, "third"),
        ],
    );
    rows.extend(record(100, &[("0", Role::Assistant, "fourth")]));
    let mut first = batch("lines", 0, 0, rows, &["s"]);
    first.length = 200;
    first.parse[0].parsed_upto = 200;
    w.writer.send(first).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(texts(&w, "s"), ["first", "second", "third", "fourth"]);

    // A record appended later with the same shape keeps its own order.
    let later = batch(
        "lines",
        0,
        200,
        record(
            200,
            &[("y", Role::User, "fifth"), ("b", Role::Assistant, "sixth")],
        ),
        &["s"],
    );
    w.writer.send(later).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(
        texts(&w, "s"),
        ["first", "second", "third", "fourth", "fifth", "sixth"]
    );
}

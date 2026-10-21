use super::*;

pub(super) fn send(
    w: &World,
    change: Change,
    version: u32,
    order: u64,
    range: (u64, u64),
    text: &str,
) {
    let mut b = batch(
        "input",
        0,
        0,
        vec![MessageRow {
            offset: range.0,
            end: range.1,
            message: text_message("a", "s", Role::User, text, None),
        }],
        &["s"],
    );
    b.change = change;
    b.parse[0].parser_version = version;
    b.parse[0].order_from = Some(order);
    w.writer.send(b).unwrap();
    let result = w.writer.flush().unwrap();
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn a_new_parser_can_replace_an_earlier_logical_position_without_changing_input_bytes() {
    let w = world();
    send(&w, Change::First, 1, 0, (1, 2), "old interpretation");
    send(
        &w,
        Change::Reparse { chain: 1 },
        2,
        1,
        (30, 40),
        "new interpretation",
    );
    let reads = ReadPool::open(&w.archive).unwrap();
    let r = reads.reader().unwrap();
    assert_eq!(
        r.messages(&sid("s"), false).unwrap()[0].text(),
        "new interpretation"
    );
    assert_eq!(
        r.query(
            "SELECT source_start,source_end,source_order,parser_version FROM message_source",
            &[]
        )
        .unwrap(),
        vec![
            serde_json::json!({"source_start":30,"source_end":40,"source_order":1,"parser_version":2})
        ]
    );
    assert_eq!(
        r.query("SELECT seq FROM message", &[]).unwrap()[0]["seq"],
        1
    );
}

#[test]
fn same_parser_duplicates_and_older_parser_replay_cannot_replace_the_new_interpretation() {
    let w = world();
    send(&w, Change::First, 1, 0, (1, 2), "old interpretation");
    send(
        &w,
        Change::Reparse { chain: 1 },
        2,
        1,
        (30, 40),
        "new interpretation",
    );
    send(
        &w,
        Change::Reparse { chain: 1 },
        2,
        9,
        (50, 60),
        "later duplicate",
    );
    send(
        &w,
        Change::Reparse { chain: 1 },
        1,
        0,
        (1, 2),
        "old interpretation",
    );
    let reads = ReadPool::open(&w.archive).unwrap();
    let r = reads.reader().unwrap();
    assert_eq!(
        r.messages(&sid("s"), false).unwrap()[0].text(),
        "new interpretation"
    );
    assert_eq!(
        r.query(
            "SELECT source_start,source_end,source_order,parser_version FROM message_source",
            &[]
        )
        .unwrap(),
        vec![
            serde_json::json!({"source_start":30,"source_end":40,"source_order":1,"parser_version":2})
        ]
    );
}

#[test]
fn parser_version_numbers_from_different_inputs_do_not_override_route_preference() {
    for reverse in [false, true] {
        let w = world();
        let mut sources = [("copy", 2, 100, "copy"), ("live", 0, 1, "live")];
        if reverse {
            sources.reverse();
        }
        for (input, rank, version, text) in sources {
            let mut b = batch(
                input,
                rank,
                0,
                vec![MessageRow {
                    offset: 0,
                    end: 10,
                    message: text_message("a", "s", Role::User, text, None),
                }],
                &["s"],
            );
            b.parse[0].parser_version = version;
            w.writer.send(b).unwrap();
        }
        assert!(w.writer.flush().unwrap().errors.is_empty());
        let reads = ReadPool::open(&w.archive).unwrap();
        let r = reads.reader().unwrap();
        assert_eq!(r.messages(&sid("s"), false).unwrap()[0].text(), "live");
    }
}

#[test]
fn newer_snapshot_scope_wins_even_when_an_older_scope_has_a_higher_parser_number() {
    let w = world();
    for (start, version, text, change) in [
        (0, 2, "old scope", Change::First),
        (100, 1, "new scope", Change::Appended { from: 100 }),
        (0, 100, "reparsed old scope", Change::Reparse { chain: 1 }),
    ] {
        let mut b = batch(
            "input",
            0,
            0,
            vec![MessageRow {
                offset: start,
                end: start + 100,
                message: text_message("a", "s", Role::User, text, None),
            }],
            &["s"],
        );
        b.change = change;
        b.length = 200;
        b.parse[0].parser_version = version;
        b.parse[0].snapshot = Some(polylogue_model::Snapshot {
            start,
            end: start + 100,
            covered: vec![sid("s")],
            complete: true,
        });
        w.writer.send(b).unwrap();
        assert!(w.writer.flush().unwrap().errors.is_empty());
    }
    let reads = ReadPool::open(&w.archive).unwrap();
    let r = reads.reader().unwrap();
    assert_eq!(r.messages(&sid("s"), false).unwrap()[0].text(), "new scope");
}

#[test]
fn newer_revision_chain_wins_over_a_higher_parser_number_on_an_old_chain() {
    let w = world();
    send(&w, Change::First, 99, 0, (1, 2), "old chain");
    send(&w, Change::Rewritten, 1, 3, (30, 40), "new chain");
    send(
        &w,
        Change::Reparse { chain: 1 },
        100,
        0,
        (1, 2),
        "reparsed old chain",
    );
    let reads = ReadPool::open(&w.archive).unwrap();
    let r = reads.reader().unwrap();
    assert_eq!(r.messages(&sid("s"), false).unwrap()[0].text(), "new chain");
    assert_eq!(
        r.query(
            "SELECT chain,parser_version FROM message_source WHERE preferred=1",
            &[]
        )
        .unwrap(),
        vec![serde_json::json!({"chain":2,"parser_version":1})]
    );
}

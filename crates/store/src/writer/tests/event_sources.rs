use super::*;

fn send_event(w: &World, input: &str, change: Change, version: u32, order: u64, range: (u64, u64)) {
    let mut b = batch(input, 0, 0, vec![], &["s"]);
    b.change = change;
    b.parse[0].parser_version = version;
    b.parse[0].order_from = Some(order);
    b.parse[0].output.events.push(polylogue_model::Event {
        session: sid("s"),
        kind: "notice".into(),
        at: None,
        after: None,
        range: Some(range),
        data: serde_json::json!("same event"),
    });
    b.parse[0].accounting = Accounting {
        framed: 1,
        ..Default::default()
    };
    b.parse[0]
        .accounting
        .add("notice", &Disposition::Materialized);
    w.writer.send(b).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
}

fn shown_range(w: &World) -> Option<(u64, u64)> {
    let pool = ReadPool::open(&w.archive).unwrap();
    let events = pool.reader().unwrap().events(&sid("s")).unwrap();
    assert_eq!(events.len(), 1);
    events[0].range
}

#[test]
fn identical_events_keep_revision_ranges_and_old_replay_cannot_replace_the_current_range() {
    let w = world();
    send_event(&w, "input", Change::First, 1, 0, (10, 20));
    send_event(&w, "input", Change::Rewritten, 1, 0, (40, 50));
    assert_eq!(shown_range(&w), Some((40, 50)));
    send_event(&w, "input", Change::Reparse { chain: 1 }, 1, 0, (10, 20));
    assert_eq!(shown_range(&w), Some((40, 50)));
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let observations = r
        .query(
            "SELECT chain,source_start,source_end FROM event_source ORDER BY chain",
            &[],
        )
        .unwrap();
    assert_eq!(
        observations,
        vec![
            serde_json::json!({"chain":1,"source_start":10,"source_end":20}),
            serde_json::json!({"chain":2,"source_start":40,"source_end":50})
        ]
    );
    assert_eq!(
        r.query("SELECT COUNT(*) AS n FROM event", &[]).unwrap()[0]["n"],
        1
    );
}

#[test]
fn an_events_current_source_is_independent_of_input_arrival_order() {
    for reverse in [false, true] {
        let w = world();
        let mut sources = [("input-a", (10, 20)), ("input-z", (40, 50))];
        if reverse {
            sources.reverse();
        }
        for (input, range) in sources {
            send_event(&w, input, Change::First, 1, 0, range);
        }
        assert_eq!(shown_range(&w), Some((10, 20)));
    }
}

#[test]
fn parser_replacement_resets_record_order_but_same_parser_replay_keeps_the_latest_record() {
    let w = world();
    send_event(&w, "input", Change::First, 1, 100, (10, 20));
    send_event(&w, "input", Change::Reparse { chain: 1 }, 2, 1, (40, 50));
    assert_eq!(shown_range(&w), Some((40, 50)));
    send_event(&w, "input", Change::Reparse { chain: 1 }, 2, 3, (60, 70));
    send_event(&w, "input", Change::Reparse { chain: 1 }, 2, 1, (40, 50));
    assert_eq!(shown_range(&w), Some((60, 70)));
}

#[test]
fn a_complete_snapshot_without_the_event_promotes_another_inputs_range() {
    let w = world();
    send_event(&w, "input-a", Change::First, 1, 0, (10, 20));
    send_event(&w, "input-z", Change::First, 1, 0, (40, 50));
    assert_eq!(shown_range(&w), Some((10, 20)));
    let mut replacement = batch("input-a", 0, 100, vec![], &["s"]);
    replacement.parse[0].snapshot = Some(polylogue_model::Snapshot {
        start: 100,
        end: 200,
        covered: vec![sid("s")],
        complete: true,
    });
    w.writer.send(replacement).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    assert_eq!(shown_range(&w), Some((40, 50)));
}

#[test]
fn removing_one_input_promotes_another_inputs_event_and_removing_both_hides_it() {
    let w = world();
    send_event(&w, "input-a", Change::First, 1, 0, (10, 20));
    send_event(&w, "input-z", Change::First, 1, 0, (40, 50));
    let blobs = std::sync::Arc::new(
        hashed_blobs::Store::open(w.archive.root.clone(), vec![], 0o600).unwrap(),
    );
    for input in ["input-a", "input-z"] {
        w.writer
            .excise(
                crate::excise::ExciseRequest {
                    id: format!("remove-{input}"),
                    scope: crate::excise::Scope::Input(input.into()),
                    reason: None,
                    dry_run: false,
                },
                std::sync::Arc::clone(&blobs),
            )
            .unwrap();
        if input == "input-a" {
            assert_eq!(shown_range(&w), Some((40, 50)));
        }
    }
    let pool = ReadPool::open(&w.archive).unwrap();
    assert!(pool.reader().unwrap().events(&sid("s")).unwrap().is_empty());
}

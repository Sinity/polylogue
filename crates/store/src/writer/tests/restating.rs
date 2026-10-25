use super::*;

fn restating(input: &str, at: u64, texts: &[(&str, &str)]) -> Batch {
    let rows = texts
        .iter()
        .map(|(id, text)| MessageRow {
            offset: at,
            end: at + 100,
            message: text_message(id, "s", Role::User, text, None),
        })
        .collect();
    let mut b = batch(input, 0, at, rows, &["s"]);
    b.parse[0].restates = true;
    b.parse[0].at_end = true;
    b
}
fn read(w: &World) -> Vec<String> {
    let pool = ReadPool::open(&w.archive).unwrap();
    pool.reader()
        .unwrap()
        .messages(&sid("s"), false)
        .unwrap()
        .iter()
        .flat_map(|m| {
            m.blocks
                .iter()
                .filter_map(|b| b.text().map(ToOwned::to_owned))
        })
        .collect()
}

fn active(w: &World) -> Vec<Json> {
    let pool = ReadPool::open(&w.archive).unwrap();
    pool.reader()
        .unwrap()
        .query("SELECT id FROM message WHERE on_active_path=1", &[])
        .unwrap()
}

#[test]
fn repeated_active_leaf_uses_record_order_in_both_replay_orders() {
    for reverse in [false, true] {
        let w = world();
        let mut seed = batch("a", 0, 0, vec![], &[]);
        seed.length = 300;
        seed.parse.clear();
        w.writer.send(seed).unwrap();
        w.writer.flush().unwrap();
        let mut observations = Vec::new();
        for (at, leaf) in [(0, "a"), (100, "b"), (200, "a")] {
            let mut b = restating("a", at, &[("a", "A"), ("b", "B")]);
            b.change = Change::Reparse { chain: 1 };
            b.length = 300;
            b.parse[0].output.events.push(polylogue_model::Event {
                session: Some(sid("s")),
                kind: "active_leaf".into(),
                at: None,
                after: Some(MessageId::parse(&format!("claude-code:{leaf}")).unwrap()),
                range: None,
                data: Json::Null,
            });
            observations.push(b);
        }
        if reverse {
            observations.reverse();
        }
        for (index, b) in observations.into_iter().enumerate() {
            w.writer.send(b).unwrap();
            w.writer.derive().unwrap();
            if !reverse {
                let expected = ["claude-code:a", "claude-code:b", "claude-code:a"][index];
                assert_eq!(
                    active(&w),
                    [serde_json::json!({"id": expected})],
                    "observation {index}"
                );
            }
        }
        assert_eq!(
            active(&w),
            [serde_json::json!({"id": "claude-code:a"})],
            "reverse replay: {reverse}"
        );
    }
}

#[test]
fn active_leaf_across_inputs_uses_known_time_and_stable_content_ties() {
    for timed in [false, true] {
        let mut forward = None;
        for reverse in [false, true] {
            let w = world();
            let mut observations = [("a", "a", 200), ("z", "b", 100)];
            if reverse {
                observations.reverse();
            }
            for (input, leaf, at) in observations {
                let mut b = restating(input, 0, &[("a", "A"), ("b", "B")]);
                b.parse[0].output.events.push(polylogue_model::Event {
                    session: Some(sid("s")),
                    kind: "active_leaf".into(),
                    at: timed.then(|| Timestamp::from_unix_ms(at).unwrap()),
                    after: Some(MessageId::parse(&format!("claude-code:{leaf}")).unwrap()),
                    range: None,
                    data: Json::Null,
                });
                w.writer.send(b).unwrap();
            }
            w.writer.derive().unwrap();
            let selected = active(&w);
            assert_eq!(selected.len(), 1);
            if timed {
                assert_eq!(selected, [serde_json::json!({"id": "claude-code:a"})]);
            }
            if let Some(expected) = &forward {
                assert_eq!(&selected, expected, "timed: {timed}");
            } else {
                forward = Some(selected);
            }
        }
    }
}

#[test]
fn empty_and_nonempty_path_markers_replace_each_other_in_both_replay_orders() {
    for last_empty in [false, true] {
        for reverse in [false, true] {
            let w = world();
            let mut seed = batch("a", 0, 0, vec![], &[]);
            seed.length = 300;
            seed.parse.clear();
            w.writer.send(seed).unwrap();
            w.writer.flush().unwrap();
            let mut observations = Vec::new();
            for (at, empty) in [(0, last_empty), (100, !last_empty), (200, last_empty)] {
                let mut b = restating("a", at, &[("a", "A"), ("b", "B")]);
                b.change = Change::Reparse { chain: 1 };
                b.length = 300;
                b.parse[0].output.events.push(polylogue_model::Event {
                    session: Some(sid("s")),
                    kind: if empty {
                        "active_path_empty"
                    } else {
                        "active_leaf"
                    }
                    .into(),
                    at: None,
                    after: (!empty).then(|| MessageId::parse("claude-code:a").unwrap()),
                    range: None,
                    data: Json::Null,
                });
                observations.push(b);
            }
            if reverse {
                observations.reverse();
            }
            for b in observations {
                w.writer.send(b).unwrap();
                w.writer.derive().unwrap();
            }
            let pool = ReadPool::open(&w.archive).unwrap();
            let reader = pool.reader().unwrap();
            let rows = reader
                .query("SELECT id,on_active_path FROM message ORDER BY id", &[])
                .unwrap();
            assert_eq!(
                rows,
                vec![
                    serde_json::json!({"id":"claude-code:a","on_active_path":if last_empty {0}else{1}}),
                    serde_json::json!({"id":"claude-code:b","on_active_path":0})
                ],
                "empty:{last_empty}, reverse:{reverse}"
            );
            assert_eq!(read(&w), ["A", "B"]);
        }
    }
}

#[test]
fn absent_or_malformed_leaf_is_unknown_and_rule_upgrade_rederives_an_empty_path() {
    for invalid in [
        None,
        Some(("active_leaf", None)),
        Some(("active_path_empty", Some("a"))),
    ] {
        let w = world();
        let mut b = restating("a", 0, &[("a", "A")]);
        if let Some((kind, after)) = invalid {
            b.parse[0].output.events.push(polylogue_model::Event {
                session: Some(sid("s")),
                kind: kind.into(),
                at: None,
                after: after.map(|id| MessageId::parse(&format!("claude-code:{id}")).unwrap()),
                range: None,
                data: Json::Null,
            });
        }
        w.writer.send(b).unwrap();
        w.writer.derive().unwrap();
        let pool = ReadPool::open(&w.archive).unwrap();
        let activity = || {
            pool.reader()
                .unwrap()
                .query("SELECT on_active_path FROM message", &[])
                .unwrap()[0]["on_active_path"]
                .clone()
        };
        assert!(activity().is_null());
        let mut b = restating("a", 100, &[("a", "A")]);
        b.parse[0].output.events.push(polylogue_model::Event {
            session: Some(sid("s")),
            kind: "active_path_empty".into(),
            at: None,
            after: None,
            range: None,
            data: Json::Null,
        });
        w.writer.send(b).unwrap();
        w.writer.derive().unwrap();
        assert_eq!(activity(), 0);
        let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
        conn.execute_batch("UPDATE message SET on_active_path=NULL; UPDATE derive_state SET version='4' WHERE name='active_path';").unwrap();
        w.writer.derive().unwrap();
        assert_eq!(activity(), 0);
        assert_eq!(
            pool.reader()
                .unwrap()
                .query(
                    "SELECT version FROM derive_state WHERE name='active_path'",
                    &[]
                )
                .unwrap()[0]["version"],
            "5"
        );
    }
}

#[test]
fn empty_path_uses_the_same_cross_input_marker_order_as_an_active_leaf() {
    for timed in [false, true] {
        let mut forward = None;
        for reverse in [false, true] {
            let w = world();
            let mut inputs = [("empty", true, 200), ("leaf", false, 100)];
            if reverse {
                inputs.reverse();
            }
            for (input, empty, at) in inputs {
                let mut b = restating(input, 0, &[("a", "A")]);
                b.parse[0].output.events.push(polylogue_model::Event {
                    session: Some(sid("s")),
                    kind: if empty {
                        "active_path_empty"
                    } else {
                        "active_leaf"
                    }
                    .into(),
                    at: timed.then(|| Timestamp::from_unix_ms(at).unwrap()),
                    after: (!empty).then(|| MessageId::parse("claude-code:a").unwrap()),
                    range: None,
                    data: Json::Null,
                });
                w.writer.send(b).unwrap();
            }
            w.writer.derive().unwrap();
            let pool = ReadPool::open(&w.archive).unwrap();
            let value = pool
                .reader()
                .unwrap()
                .query("SELECT on_active_path FROM message", &[])
                .unwrap()[0]["on_active_path"]
                .clone();
            if timed {
                assert_eq!(value, 0);
            }
            if let Some(expected) = &forward {
                assert_eq!(&value, expected);
            } else {
                forward = Some(value);
            }
        }
    }
}

#[test]
fn a_later_restating_record_replaces_the_earlier_version_and_keeps_absent_messages() {
    let w = world();
    w.writer
        .send(restating("a", 0, &[("a", "old A"), ("b", "old B")]))
        .unwrap();
    w.writer
        .send(restating("a", 100, &[("b", "new B")]))
        .unwrap();
    w.writer.flush().unwrap();
    assert_eq!(read(&w), ["old A", "new B"]);
    // A later observation may carry earlier content again.
    w.writer
        .send(restating("a", 200, &[("b", "old B")]))
        .unwrap();
    w.writer.flush().unwrap();
    assert_eq!(read(&w), ["old A", "old B"]);
}

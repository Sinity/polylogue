use super::*;

fn snapshot(input: &str, at: u64, complete: bool, texts: &[(&str, &str)]) -> Batch {
    let rows = texts
        .iter()
        .map(|(id, text)| MessageRow {
            offset: at,
            end: at + 100,
            message: text_message(id, "s", Role::User, text, None),
        })
        .collect();
    let mut b = batch(input, 0, at, rows, &["s"]);
    b.parse[0].snapshot = Some(polylogue_model::Snapshot {
        start: at,
        end: at + 100,
        covered: vec![sid("s")],
        complete,
    });
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
        .query("SELECT id FROM current_message WHERE on_active_path=1", &[])
        .unwrap()
}

#[test]
fn repeated_active_leaf_uses_snapshot_order_in_both_replay_orders() {
    for reverse in [false, true] {
        let w = world();
        let mut seed = batch("a", 0, 0, vec![], &[]);
        seed.length = 300;
        seed.parse.clear();
        w.writer.send(seed).unwrap();
        w.writer.flush().unwrap();
        let mut observations = Vec::new();
        for (at, leaf) in [(0, "a"), (100, "b"), (200, "a")] {
            let mut b = snapshot("a", at, false, &[("a", "A"), ("b", "B")]);
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
                let mut b = snapshot(input, 0, false, &[("a", "A"), ("b", "B")]);
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
fn partial_snapshots_keep_absent_members_and_complete_snapshots_retire_them() {
    let w = world();
    w.writer
        .send(snapshot("a", 0, true, &[("a", "old A"), ("b", "old B")]))
        .unwrap();
    w.writer
        .send(snapshot("a", 100, false, &[("b", "partial B")]))
        .unwrap();
    w.writer.flush().unwrap();
    assert_eq!(read(&w), ["old A", "partial B"]);
    w.writer
        .send(snapshot("a", 200, true, &[("b", "complete B")]))
        .unwrap();
    w.writer.derive().unwrap();
    assert_eq!(read(&w), ["complete B"]);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let history = reader
        .message_versions(&MessageId::parse("claude-code:a").unwrap())
        .unwrap();
    assert_eq!(history.len(), 1);
    assert!(!history[0].current);
    assert_eq!(reader.status().unwrap().messages, 1);
}

#[test]
fn reversed_snapshot_replay_and_other_source_promotion_match_forward_arrival() {
    for reverse in [false, true] {
        let w = world();
        let mut snapshots = vec![
            snapshot("a", 0, true, &[("a", "old A"), ("b", "old B")]),
            snapshot("a", 100, false, &[("b", "partial B")]),
            snapshot("a", 200, true, &[("b", "complete B")]),
        ];
        // Retained bytes already exist during rebuild; bind every replay to
        // one physical chain without manufacturing new archive revisions.
        let mut seed = batch("a", 0, 0, vec![], &[]);
        seed.length = 300;
        seed.parse.clear();
        w.writer.send(seed).unwrap();
        w.writer.flush().unwrap();
        if reverse {
            snapshots.reverse();
        }
        for mut s in snapshots {
            s.change = Change::Reparse { chain: 1 };
            s.length = 300;
            w.writer.send(s).unwrap();
        }
        w.writer
            .send(batch(
                "b",
                1,
                0,
                vec![MessageRow {
                    offset: 0,
                    end: 100,
                    message: text_message("a", "s", Role::User, "alternate current A", None),
                }],
                &["s"],
            ))
            .unwrap();
        w.writer.derive().unwrap();
        let mut result = read(&w);
        result.sort();
        assert_eq!(result, ["alternate current A", "complete B"]);
    }
}

#[test]
fn omitted_metadata_stays_known_with_partial_or_complete_population() {
    for (reverse, complete) in [(false, false), (true, false), (false, true), (true, true)] {
        let w = world();
        let mut first = snapshot("a", 0, true, &[("a", "A")]);
        first.parse[0].output.sessions[0].title = Some(polylogue_model::Title {
            text: "Known title".into(),
            source: TitleSource::Custom,
        });
        first.parse[0].output.sessions[0].cwd = vec!["/synthetic/project".into()];
        let later = snapshot("a", 100, complete, &[("b", "B")]);
        let mut seed = batch("a", 0, 0, vec![], &[]);
        seed.length = 200;
        seed.parse.clear();
        w.writer.send(seed).unwrap();
        w.writer.flush().unwrap();
        let mut steps = vec![first, later];
        if reverse {
            steps.reverse();
        }
        for mut step in steps {
            step.change = Change::Reparse { chain: 1 };
            step.length = 200;
            w.writer.send(step).unwrap();
        }
        w.writer.derive().unwrap();
        let pool = ReadPool::open(&w.archive).unwrap();
        let session = pool.reader().unwrap().session(&sid("s")).unwrap().unwrap();
        assert_eq!(session.title.unwrap().text, "Known title");
        assert_eq!(session.cwd, ["/synthetic/project"]);
    }
}

#[test]
fn removing_an_input_promotes_only_remaining_current_versions_and_prunes_history() {
    let w = world();
    w.writer
        .send(snapshot("a", 0, true, &[("a", "sensitive old version")]))
        .unwrap();
    w.writer
        .send(snapshot(
            "a",
            100,
            true,
            &[("a", "preferred current version")],
        ))
        .unwrap();
    let mut remaining = batch(
        "b",
        1,
        0,
        vec![MessageRow {
            offset: 0,
            end: 100,
            message: text_message("a", "s", Role::User, "remaining current version", None),
        }],
        &["s"],
    );
    remaining.parse[0].parser_version = 7;
    w.writer.send(remaining).unwrap();
    w.writer.derive().unwrap();
    let store = std::sync::Arc::new(
        hashed_blobs::Store::open(w.archive.root.clone(), vec![], 0o600).unwrap(),
    );
    w.writer
        .excise(
            crate::excise::ExciseRequest {
                id: "remove-a".into(),
                scope: crate::excise::Scope::Input("a".into()),
                reason: None,
                dry_run: false,
            },
            store,
        )
        .unwrap();
    w.writer.derive().unwrap();
    assert_eq!(read(&w), ["remaining current version"]);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let history = reader
        .message_versions(&MessageId::parse("claude-code:a").unwrap())
        .unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].input, "b");
    assert!(history[0].current);
    let rows = reader
        .query("SELECT count(*) AS n FROM content_version", &[])
        .unwrap();
    assert_eq!(rows[0]["n"], 1);
    assert_eq!(
        reader
            .query("SELECT parser_version FROM message", &[])
            .unwrap()[0]["parser_version"],
        7
    );
}

#[test]
fn retired_latest_member_leaves_current_activity_usage_and_prose() {
    let w = world();
    let mut first = snapshot(
        "a",
        0,
        true,
        &[("a", "retired prose"), ("b", "current prose")],
    );
    first.parse[0].output.messages[0].message.at = Timestamp::from_unix_ms(20);
    first.parse[0].output.messages[0].message.usage.input = Some(900);
    first.parse[0].output.messages[0].message.role = Role::Assistant;
    first.parse[0].output.messages[1].message.at = Timestamp::from_unix_ms(10);
    let mut second = snapshot("a", 100, true, &[("b", "current prose")]);
    second.parse[0].output.messages[0].message.at = Timestamp::from_unix_ms(10);
    w.writer.send(first).unwrap();
    w.writer.send(second).unwrap();
    w.writer.derive().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT last_activity_at FROM session", &[])
            .unwrap()[0]["last_activity_at"],
        10
    );
    assert_eq!(
        reader
            .prose_texts(0, 100, None)
            .unwrap()
            .iter()
            .map(|v| v.text.as_str())
            .collect::<Vec<_>>(),
        ["current prose"]
    );
    let usage = crate::measure::usage(&reader, &crate::measure::Scope::default()).unwrap();
    assert_eq!(usage.tokens.input.tokens, 0);
    assert_eq!(
        reader
            .query(
                "SELECT count(*) AS n FROM text_fts WHERE text_fts MATCH 'retired'",
                &[]
            )
            .unwrap()[0]["n"],
        0
    );
}

#[test]
fn complete_scope_retires_copies_by_source_placement_after_other_source_retires() {
    let w = world();
    w.writer
        .send(snapshot("a", 0, true, &[("x", "shared X")]))
        .unwrap();
    let mut copy = snapshot("b", 0, true, &[("x", "shared X")]);
    copy.parse[0].output.sessions[0] = Session::new(sid("child"), SessionKind::Subagent);
    copy.parse[0].output.messages[0].message.session = sid("child");
    copy.parse[0].snapshot.as_mut().unwrap().covered = vec![sid("child")];
    w.writer.send(copy).unwrap();
    w.writer.send(snapshot("a", 100, true, &[])).unwrap();
    w.writer.derive().unwrap();
    assert_eq!(read(&w), ["shared X"]);
    let mut empty = snapshot("b", 100, true, &[]);
    empty.parse[0].output.sessions[0] = Session::new(sid("child"), SessionKind::Subagent);
    empty.parse[0].snapshot.as_mut().unwrap().covered = vec![sid("child")];
    w.writer.send(empty).unwrap();
    w.writer.derive().unwrap();
    assert!(read(&w).is_empty());
    let pool = ReadPool::open(&w.archive).unwrap();
    assert_eq!(pool.reader().unwrap().status().unwrap().messages, 0);
}

#[test]
fn scopes_in_one_input_keep_separate_membership_for_the_same_message() {
    let w = world();
    let mut first = snapshot("a", 0, true, &[("x", "shared X")]);
    let mut copy = first.parse[0].output.messages[0].clone();
    copy.message.session = sid("child");
    first.parse[0].output.messages.push(copy);
    first.parse[0]
        .output
        .sessions
        .push(Session::new(sid("child"), SessionKind::Subagent));
    first.parse[0]
        .snapshot
        .as_mut()
        .unwrap()
        .covered
        .push(sid("child"));
    first.parse[0].accounting = accounting(2);
    w.writer.send(first).unwrap();
    w.writer.derive().unwrap();
    let mut empty = snapshot("a", 100, true, &[]);
    empty.parse[0].output.sessions[0] = Session::new(sid("child"), SessionKind::Subagent);
    empty.parse[0].snapshot.as_mut().unwrap().covered = vec![sid("child")];
    w.writer.send(empty).unwrap();
    w.writer.derive().unwrap();
    assert_eq!(read(&w), ["shared X"]);
    w.writer.send(snapshot("a", 200, true, &[])).unwrap();
    w.writer.derive().unwrap();
    assert!(read(&w).is_empty());
}

#[test]
fn normalized_history_shares_prose_and_keeps_only_bounded_nonprose_copies() {
    let w = world();
    let a = "a".repeat(100_000);
    let b = "b".repeat(100_000);
    for (at, text) in [(0, a.as_str()), (100, b.as_str()), (200, b.as_str())] {
        w.writer
            .send(snapshot("a", at, false, &[("x", text)]))
            .unwrap();
    }
    w.writer.flush().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let rows=reader.query("SELECT (SELECT count(*) FROM text) AS texts,(SELECT count(*) FROM content_version) AS versions,(SELECT sum(length(attrs)) FROM content_version)+(SELECT sum(length(body)) FROM version_block) AS metadata_bytes",&[]).unwrap();
    assert_eq!(rows[0]["texts"], 2);
    assert_eq!(rows[0]["versions"], 2);
    assert!(rows[0]["metadata_bytes"].as_u64().unwrap() < 2048);
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
                let mut b = snapshot("a", at, false, &[("a", "A"), ("b", "B")]);
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
                .query(
                    "SELECT id,on_active_path FROM current_message ORDER BY id",
                    &[],
                )
                .unwrap();
            assert_eq!(
                rows,
                vec![
                    serde_json::json!({"id":"claude-code:a","on_active_path":if last_empty {0}else{1}}),
                    serde_json::json!({"id":"claude-code:b","on_active_path":0})
                ],
                "empty:{last_empty}, reverse:{reverse}"
            );
            assert_eq!(read(&w), ["A", "B"], "retention does not follow visibility");
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
        let mut b = snapshot("a", 0, false, &[("a", "A")]);
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
                .query("SELECT on_active_path FROM current_message", &[])
                .unwrap()[0]["on_active_path"]
                .clone()
        };
        assert!(activity().is_null());
        let mut b = snapshot("a", 100, false, &[("a", "A")]);
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
                let mut b = snapshot(input, 0, false, &[("a", "A")]);
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
                .query("SELECT on_active_path FROM current_message", &[])
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

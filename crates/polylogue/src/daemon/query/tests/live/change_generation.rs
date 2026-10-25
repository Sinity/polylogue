use super::*;
use polylogue_store::{UserRequest, UserWrite};
use polylogue_testkit::{ClaudeHome, claude_user, delete_index};

#[test]
fn a_rebuilt_index_refuses_old_change_cursors_before_query_composition() {
    let home = ClaudeHome::new();
    home.append("s.jsonl", &claude_user("m1", "s", "stored exact text"));
    let (mut engine, lock) = home.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    for n in 2..=6 {
        home.append(
            "s.jsonl",
            &claude_user(&format!("m{n}"), "s", &format!("later record {n}")),
        );
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    let initial = {
        let pool = home.reads();
        run(&pool.reader().unwrap(), ask("sessions", T0)).unwrap()
    };
    let cursor = initial["read_at"]["cursor"].as_str().unwrap();
    engine.writer().unwrap().user(UserRequest {
        id: "saved-feed".into(), machine: "synthetic".into(), at: Timestamp::from_unix_ms(T0).unwrap(),
        write: UserWrite::AddNote { note: serde_json::from_value(json!({"kind":"saved_query","key":"old-feed","value":format!("changes where after:{cursor}"),"author":"operator","targets":["repo:synthetic"]})).unwrap() },
    }).unwrap();
    drop(engine);
    drop(lock);
    delete_index(&home.root);
    // The rebuild uses stored bytes, even after the provider file disappears.
    std::fs::remove_file(home.file("s.jsonl")).unwrap();
    let (mut engine, _lock) = home.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let texts = home.texts("s");
    assert_eq!(texts.len(), 6);
    assert_eq!(texts[0], "stored exact text");
    let pool = home.reads();
    let reader = pool.reader().unwrap();
    for query in [
        format!("changes where after:{cursor}"),
        format!("changes where after:{cursor} AND kind:session"),
        format!("changes where after:{cursor} | limit 1"),
        format!("changes where after:{cursor} | group by kind | count"),
        format!("sessions where exists change(after:{cursor}) | count"),
        format!("changes where NOT after:{cursor} OR kind:note"),
        "query:old-feed".into(),
    ] {
        let error = run(&reader, ask(&query, T0)).unwrap_err();
        assert_eq!(error.kind, "resync_required", "{query}: {error:?}");
        assert_eq!(
            error.detail.unwrap()["excisions_query"],
            "changes where kind:excision"
        );
        assert_eq!(
            keep(&reader, ask(&query, T0)).unwrap_err().kind,
            "resync_required",
            "keep {query}"
        );
    }
    let fresh = run(&reader, ask("sessions", T0)).unwrap();
    assert_ne!(fresh["read_at"]["generation"], initial["read_at"]["generation"]);
    let after = format!(
        "changes where after:{}",
        fresh["read_at"]["cursor"].as_str().unwrap()
    );
    assert_eq!(run(&reader, ask(&after, T0)).unwrap()["count"], 0);
}

#[test]
fn fresh_change_cursors_keep_two_counters_and_include_late_notes_and_excisions() {
    let home = ClaudeHome::new();
    home.append(
        "gone.jsonl",
        &claude_user("g1", "gone", "remove this exact text"),
    );
    home.append(
        "kept.jsonl",
        &claude_user("k1", "kept", "keep this exact text"),
    );
    let (mut engine, _lock) = home.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let pool = home.reads();
    let pinned = pool.reader().unwrap();
    let initial = run(&pinned, ask("sessions", T0)).unwrap();
    let after = format!(
        "changes where after:{}",
        initial["read_at"]["cursor"].as_str().unwrap()
    );
    engine
        .writer()
        .unwrap()
        .user(UserRequest {
            id: "000-late-note".into(),
            machine: "synthetic".into(),
            at: Timestamp::from_unix_ms(T0 - 86_400_000).unwrap(),
            write: UserWrite::AddNote {
                note: serde_json::from_value(
                    json!({"kind":"note","value":"late arrival","author":"operator","targets":["repo:synthetic"]}),
                )
                .unwrap(),
            },
        })
        .unwrap();
    assert_eq!(
        run(&pinned, ask(&after, T0)).unwrap()["count"],
        0,
        "one reader sees both counters at one moment"
    );
    drop(pinned);
    drop(pool);
    engine
        .excise(polylogue_store::excise::ExciseRequest {
            id: "remove-gone".into(),
            target: polylogue_store::excise::ExciseTarget::Session(sid("gone")),
            reason: None,
            dry_run: false,
        })
        .unwrap();
    let pool = home.reads();
    let reader = pool.reader().unwrap();
    let changes = run(&reader, ask(&after, T0)).unwrap();
    assert_eq!(changes["read_at"]["generation"], initial["read_at"]["generation"]);
    assert!(
        changes["read_at"]["index"].as_i64().unwrap() > initial["read_at"]["index"].as_i64().unwrap()
    );
    assert!(
        changes["read_at"]["archive"].as_i64().unwrap() > initial["read_at"]["archive"].as_i64().unwrap()
    );
    let rows = changes["rows"].as_array().unwrap();
    assert!(
        rows.iter()
            .any(|r| r["kind"] == "note" && r["ref"] == "000-late-note")
    );
    assert!(
        rows.iter()
            .any(|r| r["kind"] == "removed" && r["ref"] == "claude-code:gone")
    );
    let excision = rows.iter().find(|r| r["kind"] == "excision").unwrap();
    assert_eq!(excision["ref"], "remove-gone");
    assert_eq!(excision["target"], "session:claude-code:gone");
    assert_eq!(excision["sessions"], json!(["claude-code:gone"]));
    assert!(
        excision["messages"]
            .as_array()
            .unwrap()
            .iter()
            .all(Value::is_string)
    );
    let replay = run(&reader, ask("changes where kind:excision", T0)).unwrap();
    assert_eq!(replay["rows"], json!([excision]));
    let after = format!(
        "changes where after:{}",
        changes["read_at"]["cursor"].as_str().unwrap()
    );
    assert_eq!(run(&reader, ask(&after, T0)).unwrap()["count"], 0);
}

#[test]
fn selection_output_preserves_unknown_and_empty_and_refuses_scalars() {
    for (text, expected) in [
        (None, Value::Null),
        (Some("[]"), json!([])),
        (Some("[\"claude-code:a\"]"), json!(["claude-code:a"])),
    ] {
        let mut value = text.map_or(Value::Null, |text| json!(text));
        shape_value(ColKind::Ids, &mut value).unwrap();
        assert_eq!(value, expected);
    }
    for text in ["true", "1", "\"bad\"", "[1]", "{}", "["] {
        let mut value = json!(text);
        assert_eq!(
            shape_value(ColKind::Ids, &mut value).unwrap_err().kind,
            "store"
        );
    }
}

#[test]
fn scalar_sqlite_selection_values_are_not_treated_as_missing() {
    for mut value in [
        json!(true),
        json!(1),
        json!({"sessions":[]}),
        json!([null]),
        json!([""]),
    ] {
        assert_eq!(
            shape_value(ColKind::Ids, &mut value).unwrap_err().kind,
            "store"
        );
    }
}

#[test]
fn excision_changes_list_the_removed_sessions_and_messages() {
    use polylogue_store::fixture::Fixture;
    for (sessions, messages) in [
        (&[][..], &[][..]),
        (
            &["claude-code:a"][..],
            &["claude-code:a1", "claude-code:a2"][..],
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
        fixture
            .excision("remove", "session:claude-code:a", sessions, messages)
            .unwrap();
        let pool = fixture.reads().unwrap();
        let result = run(
            &pool.reader().unwrap(),
            ask("changes where kind:excision", T0),
        )
        .unwrap();
        assert_eq!(result["rows"][0]["sessions"], json!(sessions));
        assert_eq!(result["rows"][0]["messages"], json!(messages));
    }
}

#[test]
fn public_keep_refuses_a_stale_change_cursor_without_keeping_a_result() {
    use crate::daemon::{Request, Response, service::handle};
    let daemon = crate::test_daemon::start("synthetic source");
    let result = handle(
        Request::Keep {
            request_id: "stale-keep".into(),
            text: "changes where after:obsolete:0:0 | limit 1".into(),
            name: None,
        },
        &daemon.daemon,
    );
    assert!(matches!(result, Response::Error { kind, .. } if kind == "resync_required"));
    let pool = polylogue_store::ReadPool::open(&daemon.daemon.archive).unwrap();
    assert_eq!(
        pool.reader()
            .unwrap()
            .query(
                "SELECT count(*) AS n FROM a.kept_result WHERE id='stale-keep'",
                &[]
            )
            .unwrap()[0]["n"],
        0
    );
}

#[test]
fn kept_change_selections_preserve_their_json_type_when_read_back() {
    use polylogue_store::fixture::Fixture;
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    fixture
        .excision("remove", "input:synthetic", &["claude-code:a"], &[])
        .unwrap();
    let pool = fixture.reads().unwrap();
    let mut kept = keep(
        &pool.reader().unwrap(),
        ask("changes where kind:excision", T0),
    )
    .unwrap();
    fixture
        .kept_result(
            "valid-ids",
            "changes where kind:excision",
            &kept.to_string(),
        )
        .unwrap();
    let result = run(&pool.reader().unwrap(), ask("result:valid-ids", T0)).unwrap();
    assert_eq!(result["rows"][0]["sessions"], json!(["claude-code:a"]));
    assert_eq!(result["rows"][0]["messages"], json!([]));
    kept["rows"][0]["sessions"] = json!("[\"claude-code:a\"]");
    fixture
        .kept_result(
            "invalid-ids",
            "changes where kind:excision",
            &kept.to_string(),
        )
        .unwrap();
    assert_eq!(
        run(&pool.reader().unwrap(), ask("result:invalid-ids", T0))
            .unwrap_err()
            .kind,
        "store"
    );
}

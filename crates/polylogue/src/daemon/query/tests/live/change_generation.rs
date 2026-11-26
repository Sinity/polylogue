use super::*;
use polylogue_model::user::{UserRequest, UserWrite};
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
    let cursor = initial["read_position"]["cursor"].as_str().unwrap();
    engine
        .writer()
        .archive()
        .user(polylogue_store::writer::UserSubmission {
            request: UserRequest {
                authorship: polylogue_model::authorship::Authorship {
                    author: polylogue_model::Author::Operator,
                    via: polylogue_model::authorship::Channel::Cli,
                },
                id: "saved-feed".into(),
                machine: "synthetic".into(),
                at: Timestamp::from_unix_ms(T0).unwrap(),
                write: UserWrite::Save(Box::new(polylogue_model::user::Save {
                    name: Some("old-feed".into()),
                    query: format!("feed where after:{cursor}"),
                    denominator: None,
                    rows_json: r#"{"unit":"feed","columns":[],"rows":[]}"#.into(),
                    read_position: "{}".into(),
                    watch: None,
                    outcomes: None,
                })),
            },
            scope: None,
            preparation: Some(polylogue_store::user_writes::PreparationContext::new(vec![
                "claude-code".into(),
            ])),
            parent: None,
        })
        .unwrap();
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
        format!("feed where after:{cursor}"),
        format!("feed where after:{cursor} AND kind:session"),
        format!("feed where after:{cursor} | limit 1"),
        format!("feed where after:{cursor} | group by kind | count"),
        format!("sessions where exists feed(after:{cursor}) | count"),
        format!("feed where NOT after:{cursor} OR kind:annotation"),
        "query:old-feed".into(),
    ] {
        let error = run(&reader, ask(&query, T0)).unwrap_err();
        assert_eq!(error.code(), "resync_required", "{query}: {error:?}");
        assert!(
            error.message.starts_with("the index generation changed"),
            "{query}: {}",
            error.message
        );
        assert_eq!(
            save_rows(&reader, ask(&query, T0)).unwrap_err().code(),
            "resync_required",
            "keep {query}"
        );
    }
    let fresh = run(&reader, ask("sessions", T0)).unwrap();
    assert_ne!(
        fresh["read_position"]["generation"],
        initial["read_position"]["generation"]
    );
    let after = format!(
        "feed where after:{}",
        fresh["read_position"]["cursor"].as_str().unwrap()
    );
    assert_eq!(run(&reader, ask(&after, T0)).unwrap()["count"], 0);
}

#[test]
fn fresh_change_cursors_keep_two_counters_and_include_late_notes() {
    let home = ClaudeHome::new();
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
        "feed where after:{}",
        initial["read_position"]["cursor"].as_str().unwrap()
    );
    engine
        .writer()
        .archive().user(UserRequest {
    authorship: polylogue_model::authorship::Authorship {
        author: polylogue_model::Author::Operator,
        via: polylogue_model::authorship::Channel::Cli,
    },
            id: "000-late-note".into(),
            machine: "synthetic".into(),
            at: Timestamp::from_unix_ms(T0 - 86_400_000).unwrap(),
            write: UserWrite::Assert {
                annotations: vec![serde_json::from_value(json!({"kind":"note","key":null,"value":"late arrival","links":[{"role":"about","reference":"repo:synthetic"}],"valid_until":null})).unwrap()],
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
    home.append("kept.jsonl", &claude_user("k2", "kept", "a later message"));
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let pool = home.reads();
    let reader = pool.reader().unwrap();
    let changes = run(&reader, ask(&after, T0)).unwrap();
    assert_eq!(
        changes["read_position"]["generation"],
        initial["read_position"]["generation"]
    );
    assert!(
        changes["read_position"]["index"].as_i64().unwrap()
            > initial["read_position"]["index"].as_i64().unwrap()
    );
    assert!(
        changes["read_position"]["archive"].as_i64().unwrap()
            > initial["read_position"]["archive"].as_i64().unwrap()
    );
    let rows = changes["rows"].as_array().unwrap();
    let mut changed: Vec<(&str, &str)> = rows
        .iter()
        .map(|r| (r["kind"].as_str().unwrap(), r["ref"].as_str().unwrap()))
        .collect();
    changed.sort_unstable();
    assert_eq!(
        changed,
        [
            ("annotation", "000-late-note"),
            ("session", "claude-code:kept")
        ],
        "{changes}"
    );
    let after = format!(
        "feed where after:{}",
        changes["read_position"]["cursor"].as_str().unwrap()
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
        shape_value(ColKind::Ids, &jiff::tz::TimeZone::UTC, &mut value).unwrap();
        assert_eq!(value, expected);
    }
    for text in ["true", "1", "\"bad\"", "[1]", "{}", "["] {
        let mut value = json!(text);
        assert_eq!(
            shape_value(ColKind::Ids, &jiff::tz::TimeZone::UTC, &mut value)
                .unwrap_err()
                .code(),
            polylogue_model::Code::Store
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
            shape_value(ColKind::Ids, &jiff::tz::TimeZone::UTC, &mut value)
                .unwrap_err()
                .code(),
            polylogue_model::Code::Store
        );
    }
}

#[test]
fn save_refuses_a_stale_change_cursor_without_keeping_a_result() {
    let daemon = crate::test_daemon::start("synthetic source");
    let result = crate::test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection":"feed where after:obsolete:0:0 | limit 1", "steps":[{"verb":"save"}]}),
    )
    .id("stale-keep")
    .answer(&daemon.daemon);
    assert!(
        matches!(result, command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("resync_required"))
    );
    let pool = polylogue_store::ReadPool::open(&daemon.daemon.archive).unwrap();
    assert_eq!(
        pool.reader()
            .unwrap()
            .query(
                "SELECT count(*) AS n FROM a.saved_rows WHERE id='stale-keep-1'",
                &[]
            )
            .unwrap()[0]["n"],
        0
    );
}

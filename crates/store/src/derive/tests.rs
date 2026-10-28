use crate::fixture::writer::*;
use polylogue_model::{Json, RepoLookup, Session, SessionKind};
use serde_json::json;

fn one(w: &WriterArchive, sql: &str) -> Json {
    w.rows(sql).into_iter().next().unwrap_or(Json::Null)
}

#[test]
fn every_derivation_runs_after_what_it_reads() {
    for (i, d) in super::ALL.iter().enumerate() {
        for a in d.after {
            assert!(
                super::ALL[..i].iter().any(|x| x.name == *a),
                "{} reads {a}, which does not run before it",
                d.name
            );
        }
    }
}

#[test]
fn a_shell_call_s_command_is_normalized_by_the_derive_step() {
    let mut w = WriterArchive::new();
    w.write("a", bash("a", 1, 1_000, "cargo test -p x", true, "ok"));
    assert_eq!(
        one(&w, "SELECT action, command_norm FROM tool_call"),
        json!({"action": "shell", "command_norm": "cargo test"})
    );
    // Nothing changed: nothing is derived again.
    assert_eq!(w.writer.derive().unwrap(), 0);
}

#[test]
fn a_call_left_behind_has_passed_without_result() {
    let mut w = WriterArchive::new();
    let mut msgs = call(
        "a",
        1,
        1_000,
        "Bash",
        Some("sleep 1"),
        Json::Null,
        Ok("unused"),
    );
    // Drop the result: the call never gets one.
    msgs.truncate(1);
    w.write("a", msgs);
    assert_eq!(
        one(&w, "SELECT outcome_reason FROM tool_call"),
        json!({"outcome_reason": "no_result"})
    );
    // A later model response: the harness moved on without a result.
    w.write("a", bash("a", 2, 2_000, "true", true, ""));
    assert_eq!(
        w.rows("SELECT outcome_reason FROM tool_call ORDER BY at"),
        vec![
            json!({"outcome_reason": "passed_without_result"}),
            json!({"outcome_reason": null})
        ]
    );
}

#[test]
fn a_session_without_a_title_is_named_by_its_first_prompt() {
    let mut w = WriterArchive::new();
    w.write(
        "a",
        prose("a", 1, 1_000, "\n  Fix the flaky crash test\nplease"),
    );
    assert_eq!(
        one(&w, "SELECT title, derived_title FROM session"),
        json!({"title": null, "derived_title": "Fix the flaky crash test"})
    );
}

#[test]
fn a_session_takes_its_repository_from_the_nearest_lookup() {
    let mut w = WriterArchive::new();
    let mut s = Session::new(sid("a"), SessionKind::Main);
    s.offer_cwd("/r/project/x/sub");
    w.lookups.push(RepoLookup {
        path: "/r/project/x".into(),
        remote: Some("https://example.com/x".into()),
        root_commit: Some("abc".into()),
    });
    w.write_session(s.clone(), prose("a", 1, 1_000, "hi"));
    assert_eq!(
        one(&w, "SELECT repo, repo_root FROM session"),
        json!({"repo": "https://example.com/x", "repo_root": "abc"})
    );
    // The lookup changes: the session is attributed again.
    w.lookups.push(RepoLookup {
        path: "/r/project/x".into(),
        remote: Some("https://example.com/y".into()),
        root_commit: Some("abc".into()),
    });
    w.write_session(s, vec![]);
    assert_eq!(
        one(&w, "SELECT repo FROM session"),
        json!({"repo": "https://example.com/y"})
    );
}

#[test]
fn a_repository_the_records_name_wins_over_the_lookup() {
    let mut w = WriterArchive::new();
    let mut s = Session::new(sid("a"), SessionKind::Main);
    s.offer_cwd("/r/project/x");
    s.repo = Some("git@own:repo".into());
    w.lookups.push(RepoLookup {
        path: "/r/project/x".into(),
        remote: Some("https://example.com/x".into()),
        root_commit: Some("abc".into()),
    });
    w.write_session(s, prose("a", 1, 1_000, "hi"));
    assert_eq!(
        one(&w, "SELECT repo, repo_root FROM session"),
        json!({"repo": "git@own:repo", "repo_root": "abc"})
    );
}

/// Two inputs about one session, written in the given order.
fn two_inputs(early_first: bool) -> Json {
    let mut w = WriterArchive::new();
    let mut early = Session::new(sid("s"), SessionKind::Subagent);
    early.offer_cwd("/a");
    early.offer_title("early", polylogue_model::TitleSource::Generated);
    early.git_branch = Some("main".into());
    early.tool_version = Some("1.0".into());
    early.repo = Some("https://example.test/early.git".into());
    early.mode = Some("interactive".into());
    let mut late = Session::new(sid("s"), SessionKind::Main);
    late.offer_cwd("/b");
    late.offer_cwd("/a");
    late.offer_title("late", polylogue_model::TitleSource::Generated);
    late.git_branch = Some("dev".into());
    late.repo = Some("https://example.test/late.git".into());
    late.mode = Some("bg".into());
    let writes = [
        ("in-early", early, prose("s", 1, 1_000, "first")),
        ("in-late", late, prose("s", 2, 5_000, "second")),
    ];
    let order: Vec<usize> = if early_first { vec![0, 1] } else { vec![1, 0] };
    for i in order {
        let (input, s, m) = writes[i].clone();
        w.write_from(input, s, m);
    }
    one(
        &w,
        "SELECT kind, title, title_source, cwd_json, cwd, branch, harness_version, record_repo, mode FROM session",
    )
}

#[test]
fn session_facts_do_not_depend_on_arrival_order() {
    // A session's own facts: kind from the earliest input, an
    // equal-precedence title and the branch from the latest, working
    // directories by first use.
    let expected = json!({
        "kind": "subagent", "title": "late", "title_source": "generated",
        "cwd_json": "[\"/a\",\"/b\"]", "cwd": "/b", "branch": "dev", "harness_version": "1.0", "record_repo": "https://example.test/late.git", "mode": "bg"
    });
    assert_eq!(two_inputs(true), expected);
    assert_eq!(two_inputs(false), expected);
}

#[test]
fn session_read_and_export_preserve_every_recorded_fact() {
    let mut w = WriterArchive::new();
    let mut session = Session::new(sid("export:identity"), SessionKind::Subagent);
    session.offer_title("An exact title", polylogue_model::TitleSource::Custom);
    session.offer_cwd("/first");
    session.offer_cwd("/second");
    session.git_branch = Some("topic".into());
    session.repo = Some("https://example.test/source.git".into());
    session.tool_version = Some("1.2.3".into());
    session.started = polylogue_model::Timestamp::from_unix_ms(1234);
    session.mode = Some("bg".into());
    w.write_session(
        session.clone(),
        prose("export:identity", 1, 1234, "A message"),
    );
    let reads = crate::ReadPool::open(&w.archive).unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(reader.session(&session.id).unwrap(), Some(session.clone()));
    let exported = reader.session_export(&session.id).unwrap().unwrap();
    assert_eq!(exported.session, session);
    assert_eq!(exported.messages.len(), 1);
    assert_eq!(exported.messages[0].session, exported.session.id);
}

fn versioned_rules(version: i64) -> [super::Derivation; 4] {
    use super::{Derivation, Version};
    [
        Derivation {
        products: &[],
            name: "upstream",
            version: Version::Rules(version),
            after: &[],
            run: if version == 1 {
                |tx, pk, _| {
                    tx.execute(
                        "UPDATE session SET derived_title = 'before' WHERE pk = ?1",
                        [pk],
                    )?;
                    Ok(())
                }
            } else {
                |tx, pk, _| {
                    tx.execute(
                        "UPDATE session SET derived_title = 'after' WHERE pk = ?1",
                        [pk],
                    )?;
                    Ok(())
                }
            },
            finish: None,
        },
        Derivation {
        products: &[],
            name: "dependent",
            version: Version::Rules(1),
            after: &["upstream"],
            run: |tx, pk, _| {
                tx.execute(
                    "UPDATE session SET repo = derived_title WHERE pk = ?1",
                    [pk],
                )?;
                Ok(())
            },
            finish: None,
        },
        Derivation {
        products: &[],
            name: "unrelated",
            version: Version::Rules(1),
            after: &[],
            run: |tx, pk, _| {
                tx.execute(
                    "UPDATE session SET repo_root = 'unrelated' WHERE pk = ?1",
                    [pk],
                )?;
                Ok(())
            },
            finish: None,
        },
        super::cost::DERIVATION,
    ]
}

fn synthetic_catalog(version: &str) -> polylogue_model::Catalog {
    serde_json::from_value(json!({"version": version, "models": {}})).unwrap()
}

#[test]
fn upstream_rule_change_invalidates_unchanged_dependents_only() {
    let mut index = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_index(&index).unwrap();
    index
        .execute(
            "INSERT INTO session(id, origin, native_id, kind, changed)
        VALUES ('test:s', 'test', 's', 'main', 1)",
            [],
        )
        .unwrap();
    let catalog = synthetic_catalog("first");
    assert_eq!(
        super::run_all(&mut index, &catalog, &versioned_rules(1)).unwrap(),
        4
    );
    assert_eq!(
        index
            .query_row("SELECT repo FROM session", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "before"
    );
    assert_eq!(
        super::run_all(&mut index, &catalog, &versioned_rules(2)).unwrap(),
        2
    );
    assert_eq!(
        index
            .query_row("SELECT repo FROM session", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "after"
    );
    let changed: i64 = index
        .query_row("SELECT changed FROM session", [], |r| r.get(0))
        .unwrap();
    assert!(changed > 1);
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM derive_state WHERE from_changed<>?1",
                [changed],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    assert_eq!(
        super::run_all(&mut index, &catalog, &versioned_rules(2)).unwrap(),
        0
    );
    assert_eq!(
        super::run_all(
            &mut index,
            &synthetic_catalog("second"),
            &versioned_rules(2)
        )
        .unwrap(),
        1
    );
    assert_eq!(
        index
            .query_row("SELECT repo_root FROM session", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        "unrelated"
    );
}

#[test]
fn dependency_identity_includes_transitive_rules_and_ignores_list_order() {
    let catalog = synthetic_catalog("same");
    let mut first = versioned_rules(1).to_vec();
    first.push(super::Derivation {
        products: &[],
        name: "grandchild",
        version: super::Version::Rules(1),
        after: &["upstream", "dependent"],
        run: |_, _, _| Ok(()),
        finish: None,
    });
    let original = super::versions(&first, &catalog).unwrap();
    first[4].after = &["dependent", "upstream"];
    assert_eq!(super::versions(&first, &catalog).unwrap(), original);
    first[0].version = super::Version::Rules(2);
    let changed = super::versions(&first, &catalog).unwrap();
    for i in [0, 1, 4] {
        assert_ne!(original[i], changed[i]);
    }
    for i in [2, 3] {
        assert_eq!(original[i], changed[i]);
    }
}

#[test]
fn invalid_dependency_declarations_are_refused_before_any_rows_change() {
    let mut missing = versioned_rules(1);
    missing[1].after = &["missing"];
    let mut cyclic = versioned_rules(1);
    cyclic[0].after = &["dependent"];
    let mut backward = versioned_rules(1);
    backward.swap(0, 1);
    let mut duplicate = versioned_rules(1);
    duplicate[2].name = "upstream";
    for (rules, reason) in [
        (missing, "missing"),
        (cyclic, "cycle"),
        (backward, "must follow"),
        (duplicate, "duplicate"),
    ] {
        let mut index = rusqlite::Connection::open_in_memory().unwrap();
        crate::schema::create_index(&index).unwrap();
        index
            .execute(
                "INSERT INTO session(id, origin, native_id, kind, changed)
            VALUES ('test:s', 'test', 's', 'main', 1)",
                [],
            )
            .unwrap();
        let error = super::run_all(&mut index, &synthetic_catalog("same"), &rules).unwrap_err();
        assert!(
            matches!(error, crate::StoreError::Invalid(ref detail) if detail.contains(reason)),
            "{error}"
        );
        assert_eq!(
            index
                .query_row("SELECT count(*) FROM derive_state", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            index
                .query_row("SELECT derived_title FROM session", [], |r| r
                    .get::<_, Option<String>>(0))
                .unwrap(),
            None
        );
    }
}

#[test]
fn a_failed_derivation_rolls_back_values_counters_and_markers() {
    use super::{Derivation, Version};
    let mut index = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_index(&index).unwrap();
    index.execute_batch("INSERT INTO session(id,origin,native_id,kind,changed) VALUES ('test:a','test','a','main',1),('test:b','test','b','main',1); INSERT INTO meta VALUES ('write_counter','1')").unwrap();
    let mut rule = Derivation {
        products: &[],
        name: "synthetic",
        version: Version::Rules(1),
        after: &[],
        run: |tx, pk, _| {
            tx.execute(
                "UPDATE session SET derived_title='before' WHERE pk=?1",
                [pk],
            )?;
            Ok(())
        },
        finish: None,
    };
    let catalog = synthetic_catalog("same");
    assert_eq!(
        super::run_all(&mut index, &catalog, std::slice::from_ref(&rule)).unwrap(),
        2
    );
    let before: String = index
        .query_row(
            "SELECT value FROM meta WHERE key='write_counter'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    rule.version = Version::Rules(2);
    rule.run = |tx, pk, _| {
        tx.execute("UPDATE session SET derived_title='after' WHERE pk=?1", [pk])?;
        let changed: i64 = tx.query_row(
            "SELECT count(*) FROM session WHERE derived_title='after'",
            [],
            |r| r.get(0),
        )?;
        if changed == 2 {
            return Err(crate::StoreError::Invalid("late derived failure".into()));
        }
        Ok(())
    };
    assert!(
        super::run_all(&mut index, &catalog, std::slice::from_ref(&rule))
            .unwrap_err()
            .to_string()
            .contains("late derived failure")
    );
    assert_eq!(
        index
            .query_row(
                "SELECT value FROM meta WHERE key='write_counter'",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
        before
    );
    assert_eq!(index.query_row("SELECT count(*) FROM session WHERE derived_title='before' AND changed=CAST(?1 AS INTEGER)", [&before], |r|r.get::<_,i64>(0)).unwrap(), 2);
    assert_eq!(index.query_row("SELECT count(*) FROM derive_state WHERE version='1' AND from_changed=CAST(?1 AS INTEGER)", [&before], |r|r.get::<_,i64>(0)).unwrap(), 2);
    rule.run = |tx, pk, _| {
        tx.execute("UPDATE session SET derived_title='after' WHERE pk=?1", [pk])?;
        Ok(())
    };
    assert_eq!(
        super::run_all(&mut index, &catalog, std::slice::from_ref(&rule)).unwrap(),
        2
    );
    assert_eq!(
        super::run_all(&mut index, &catalog, std::slice::from_ref(&rule)).unwrap(),
        0
    );
}

#[test]
fn stored_products_declare_validity_and_executable_evidence() {
    use polylogue_model::derived::Validity;
    let products=super::products();
    assert_eq!(products.iter().map(|p|p.catalog).collect::<Vec<_>>(),["K23","K15","K16"]);
    for p in products {
        assert!(!p.reads.is_empty());assert!(!p.unknown.is_empty());
        match p.validity {
            Validity::Exact=>assert!(!p.tests.is_empty()),
            Validity::Parameterized=>{assert!(!p.tests.is_empty());assert!(!p.parameters.is_empty());assert!(!p.figures.is_empty());},
            Validity::Extracted|Validity::Judged=>assert!(!p.figures.is_empty()),
        }
    }
    assert_eq!(polylogue_model::derived::markdown(&polylogue_model::derived::all()),include_str!("../../../../docs/derived.md"));
}

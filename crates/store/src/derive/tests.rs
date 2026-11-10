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
        root_commits: vec!["abc".into()],
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
        root_commits: vec!["abc".into()],
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
        root_commits: vec!["abc".into()],
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
            dirtied_by: &[],
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
            class: super::Class::Session,
        },
        Derivation {
            products: &[],
            name: "dependent",
            version: Version::Rules(1),
            after: &["upstream"],
            dirtied_by: &[],
            run: |tx, pk, _| {
                tx.execute(
                    "UPDATE session SET repo = derived_title WHERE pk = ?1",
                    [pk],
                )?;
                Ok(())
            },
            class: super::Class::Session,
        },
        Derivation {
            products: &[],
            name: "unrelated",
            version: Version::Rules(1),
            after: &[],
            dirtied_by: &[],
            run: |tx, pk, _| {
                tx.execute(
                    "UPDATE session SET repo_root = 'unrelated' WHERE pk = ?1",
                    [pk],
                )?;
                Ok(())
            },
            class: super::Class::Session,
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
        super::run_all(&mut index, &catalog, &versioned_rules(1), None).unwrap(),
        4
    );
    assert_eq!(
        index
            .query_row("SELECT repo FROM session", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "before"
    );
    assert_eq!(
        super::run_all(&mut index, &catalog, &versioned_rules(2), None).unwrap(),
        2
    );
    assert_eq!(
        index
            .query_row("SELECT repo FROM session", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "after"
    );
    // Deriving changes no session's own rows (docs/measures.md, convergence).
    let changed: i64 = index
        .query_row("SELECT changed FROM session", [], |r| r.get(0))
        .unwrap();
    assert_eq!(changed, 1);
    assert_eq!(
        super::run_all(&mut index, &catalog, &versioned_rules(2), None).unwrap(),
        0
    );
    assert_eq!(
        super::run_all(
            &mut index,
            &synthetic_catalog("second"),
            &versioned_rules(2),
            None
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
        dirtied_by: &[],
        run: |_, _, _| Ok(()),
        class: super::Class::Session,
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
        let error =
            super::run_all(&mut index, &synthetic_catalog("same"), &rules, None).unwrap_err();
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
        dirtied_by: &[],
        run: |tx, pk, _| {
            tx.execute(
                "UPDATE session SET derived_title='before' WHERE pk=?1",
                [pk],
            )?;
            Ok(())
        },
        class: super::Class::Session,
    };
    let catalog = synthetic_catalog("same");
    assert_eq!(
        super::run_all(&mut index, &catalog, std::slice::from_ref(&rule), None).unwrap(),
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
        super::run_all(&mut index, &catalog, std::slice::from_ref(&rule), None)
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
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM session WHERE derived_title='before' AND changed=1",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        2
    );
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM derive_state WHERE version='1'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        2
    );
    // Both sessions are still due for the new version.
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM derive_due", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
    rule.run = |tx, pk, _| {
        tx.execute("UPDATE session SET derived_title='after' WHERE pk=?1", [pk])?;
        Ok(())
    };
    assert_eq!(
        super::run_all(&mut index, &catalog, std::slice::from_ref(&rule), None).unwrap(),
        2
    );
    assert_eq!(
        super::run_all(&mut index, &catalog, std::slice::from_ref(&rule), None).unwrap(),
        0
    );
}

#[test]
fn stored_products_declare_validity_and_executable_evidence() {
    use polylogue_model::derived::Validity;
    let products = super::products();
    assert_eq!(
        products.iter().map(|p| p.catalog).collect::<Vec<_>>(),
        ["K23", "K15", "K16"]
    );
    for p in products {
        assert!(!p.reads.is_empty());
        assert!(!p.unknown.is_empty());
        match p.validity {
            Validity::Exact => assert!(!p.tests.is_empty()),
            Validity::Parameterized => {
                assert!(!p.tests.is_empty());
                assert!(!p.parameters.is_empty());
                assert!(!p.figures.is_empty());
            }
            Validity::Extracted | Validity::Judged => assert!(!p.figures.is_empty()),
        }
    }
    assert_eq!(
        polylogue_model::derived::markdown(&polylogue_model::derived::all()),
        include_str!("../../../../docs/derived.md")
    );
}

/// A subagent forked with its parent's context: its first user message is
/// the harness's fork instructions, and its metadata names the task.
fn forked_subagent(w: &mut WriterArchive, description: Option<&str>) {
    use polylogue_model::{Event, Output, TimeQuality};
    w.write("parent", prose("parent", 1, 1_000, "Plan the migration"));
    let child = sid("agent-fork");
    let messages = prose(
        "agent-fork",
        1,
        2_000,
        "<fork-boilerplate>\nYou are a worker fork.\n</fork-boilerplate>\nRun the audit",
    );
    let mut events = vec![Event {
        session: Some(child.clone()),
        kind: "fork_context".into(),
        at: None,
        at_quality: TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: None,
        range: Some((0, 1)),
        data: json!({"type":"fork-context-ref","parentSessionId":"parent"}),
    }];
    if let Some(description) = description {
        events.push(Event {
            session: Some(child.clone()),
            kind: "subagent_meta".into(),
            at: None,
            at_quality: TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            after: None,
            range: Some((1, 2)),
            data: json!({"agentType":"fork","description":description}),
        });
    }
    let output = Output {
        sessions: vec![Session::new(child.clone(), SessionKind::Subagent)],
        messages: messages
            .into_iter()
            .enumerate()
            .map(|(i, message)| polylogue_model::MessageRow {
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset: i as u64,
                end: i as u64 + 1,
                message,
            })
            .collect(),
        links: vec![polylogue_model::Link {
            kind: polylogue_model::LinkKind::Spawn,
            parent: sid("parent"),
            child,
            at: Some(polylogue_model::MessageId::parse("claude-code:parent-p1").unwrap()),
            call_id: None,
            evidence: "fork-context-ref".into(),
        }],
        events,
        ..Output::default()
    };
    w.write_output(
        "input-agent-fork",
        "claude-code",
        "/p/agent-fork",
        output,
        2,
    );
    w.writer.derive().unwrap();
}

#[test]
fn fork_title_from_meta_description() {
    let mut w = WriterArchive::new();
    forked_subagent(&mut w, Some("Audit the identity schemes"));
    assert_eq!(
        one(
            &w,
            "SELECT derived_title FROM session WHERE id = 'claude-code:agent-fork'"
        ),
        json!({"derived_title": "Audit the identity schemes"})
    );
    // Without metadata the fork instructions are still never the title.
    let mut w = WriterArchive::new();
    forked_subagent(&mut w, None);
    assert_eq!(
        one(
            &w,
            "SELECT derived_title FROM session WHERE id = 'claude-code:agent-fork'"
        ),
        json!({"derived_title": null})
    );
}

#[test]
fn fork_with_parents_shows_inherited_count() {
    let mut w = WriterArchive::new();
    forked_subagent(&mut w, Some("Audit"));
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let child = sid("agent-fork");
    let selection = reader.select_messages(&child, true).unwrap();
    assert!(selection.parents().complete, "{:?}", selection.parents());
    // The parent's prompt up to the branch point comes before the fork's
    // own message.
    let keys = reader.message_keys_with_parents(&child).unwrap();
    assert_eq!(keys.len(), 2, "{keys:?}");
}

/// SQLite steps of a derive pass after one of `sessions` derived sessions
/// changed.
fn derive_steps(sessions: i64) -> usize {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut index = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_index(&index).unwrap();
    index
        .execute(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?1)
             INSERT INTO session(id, origin, native_id, kind, changed)
             SELECT 'test:s' || i, 'test', 's' || i, 'main', 1 FROM n",
            [sessions],
        )
        .unwrap();
    let catalog = synthetic_catalog("same");
    let rules = versioned_rules(1);
    super::run_all(&mut index, &catalog, &rules, None).unwrap();
    index
        .execute("UPDATE session SET changed = 2 WHERE pk = 1", [])
        .unwrap();
    let steps = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&steps);
    index
        .progress_handler(
            1,
            Some(move || {
                counted.fetch_add(1, Ordering::Relaxed);
                false
            }),
        )
        .unwrap();
    // The three session derivations; cost is a row value written with its row.
    assert_eq!(
        super::run_all(&mut index, &catalog, &rules, None).unwrap(),
        3
    );
    index.progress_handler(0, None::<fn() -> bool>).unwrap();
    steps.load(Ordering::Relaxed)
}

#[test]
fn derive_discovery_steps_independent_of_session_count() {
    assert_eq!(derive_steps(10), derive_steps(2_000));
}

#[test]
fn noop_derive_leaves_session_changed() {
    let mut w = WriterArchive::new();
    w.write("a", prose("a", 1, 1_000, "a first prompt"));
    let state = |w: &WriterArchive| {
        (
            w.rows("SELECT id, changed FROM session ORDER BY id"),
            w.rows("SELECT value FROM meta WHERE key = 'write_counter'"),
        )
    };
    let before = state(&w);
    assert_eq!(w.writer.derive().unwrap(), 0);
    assert_eq!(state(&w), before);
    // A derive that has work moves no session's `changed` either: the later
    // session keeps the counter of the write that stored its rows.
    w.write("b", prose("b", 1, 2_000, "another prompt"));
    let (sessions, counter) = state(&w);
    assert_eq!(sessions[0], before.0[0]);
    let counter: i64 = counter[0]["value"].as_str().unwrap().parse().unwrap();
    assert!(sessions[1]["changed"].as_i64().unwrap() < counter);
}

#[test]
fn user_write_served_between_derive_commits() {
    let w = WriterArchive::new();
    let sessions = 2_000;
    let output = polylogue_model::Output {
        sessions: (0..sessions)
            .map(|n| Session::new(sid(&format!("s{n}")), SessionKind::Main))
            .collect(),
        messages: (0..sessions)
            .map(|n| polylogue_model::MessageRow {
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset: n,
                end: n + 1,
                message: prose(&format!("s{n}"), 1, 1_000, "a prompt").remove(0),
            })
            .collect(),
        ..polylogue_model::Output::default()
    };
    let mut w = w;
    w.write_output("input-many", "claude-code", "/p/many", output, sessions);
    // As after their rows changed: every session is due again.
    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    index
        .execute(
            "INSERT INTO derive_due(session_pk, rows_changed) SELECT pk, 1 FROM session WHERE true
             ON CONFLICT(session_pk) DO UPDATE SET rows_changed = 1",
            [],
        )
        .unwrap();
    let due = || -> i64 {
        index
            .query_row("SELECT count(*) FROM derive_due", [], |r| r.get(0))
            .unwrap()
    };
    let sender = w.writer.sender().unwrap();
    let object = polylogue_query::plan::prepare(
        "sessions where title:between",
        &polylogue_query::Bindings::default(),
        |_| Ok(()),
    )
    .unwrap()
    .query_object()
    .unwrap();
    let barrier = sender.pause_user_dispatch_for_test(
        &object.id.to_string(),
        crate::writer::DispatchPhase::BeforeAdmission,
    );
    std::thread::scope(|scope| {
        let deriving = scope.spawn(|| sender.derive().unwrap());
        // The user write is queued once the derive has committed its first
        // transaction.
        while due() == i64::try_from(sessions).unwrap() {
            std::thread::yield_now();
        }
        let writing = scope.spawn(|| sender.query_object(object.clone()).unwrap());
        barrier.wait().unwrap();
        let remaining = due();
        assert!(
            remaining > 0 && remaining < i64::try_from(sessions).unwrap(),
            "{remaining}"
        );
        barrier.release();
        writing.join().unwrap();
        assert!(deriving.join().unwrap() > 0);
    });
    assert_eq!(due(), 0);
}

#[test]
fn a_changed_lookup_derives_only_the_repository_of_sessions_under_it() {
    let mut w = WriterArchive::new();
    for (name, cwd) in [
        ("a", "/r/project/x/sub"),
        ("b", "/r/project/x"),
        ("c", "/r/project/xy"),
        ("d", "/r/other"),
    ] {
        let mut s = Session::new(sid(name), SessionKind::Main);
        s.offer_cwd(cwd);
        w.write_session(s, prose(name, 1, 1_000, "hi"));
    }
    let state = |w: &WriterArchive| w.rows("SELECT id, repo, changed FROM session ORDER BY id");
    let before = state(&w);
    // A lookup collector's record, written on an input of its own, with
    // the derive step left to run separately so its work can be counted.
    let mut lookup = crate::fixture::writer::step(
        "input-lookups",
        "repo-lookup",
        "repo-lookup:/r/project/x",
        polylogue_model::Output {
            repo_lookups: vec![RepoLookup {
                path: "/r/project/x".into(),
                remote: Some("https://example.com/x".into()),
                root_commits: vec!["abc".into()],
            }],
            ..polylogue_model::Output::default()
        },
        0,
        1,
    );
    let mut raw = lookup.clone();
    raw.parse.clear();
    let stored = w
        .writer
        .store_bytes(w.archive.prepare_bytes(&b"r\n"[..]).unwrap(), raw)
        .unwrap();
    lookup.change = crate::Change::Reparse {
        chain: stored.chain,
    };
    lookup.observation = None;
    w.writer.send(lookup).unwrap();
    assert!(w.writer.flush_rows().unwrap().errors.is_empty());
    // Two sessions lie under the lookup; each runs the repository alone.
    assert_eq!(w.writer.derive().unwrap(), 2);
    let after = state(&w);
    let repo = "https://example.com/x";
    for (before, after) in before.iter().zip(&after) {
        let under = ["claude-code:a", "claude-code:b"].contains(&after["id"].as_str().unwrap());
        assert_eq!(after["changed"], before["changed"], "{after}");
        assert_eq!(
            after["repo"],
            if under { json!(repo) } else { Json::Null },
            "{after}"
        );
    }
}

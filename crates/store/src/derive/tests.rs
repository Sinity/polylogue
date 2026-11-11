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
    use super::{Derivation, Output, Version};
    [
        Derivation {
            products: &[],
            name: "upstream",
            version: Version::Rules(version),
            after: &[],
            dirtied_by: &[],
            compute: if version == 1 {
                |_, _, _| {
                    Ok(Output::Columns(vec![(
                        "derived_title",
                        Some("before".into()),
                    )]))
                }
            } else {
                |_, _, _| {
                    Ok(Output::Columns(vec![(
                        "derived_title",
                        Some("after".into()),
                    )]))
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
            compute: |conn, pk, _| {
                let title = conn.query_row(
                    "SELECT derived_title FROM session WHERE pk = ?1",
                    [pk],
                    |r| r.get(0),
                )?;
                Ok(Output::Columns(vec![("repo", title)]))
            },
            class: super::Class::Session,
        },
        Derivation {
            products: &[],
            name: "unrelated",
            version: Version::Rules(1),
            after: &[],
            dirtied_by: &[],
            compute: |_, _, _| {
                Ok(Output::Columns(vec![(
                    "repo_root",
                    Some("unrelated".into()),
                )]))
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
    // Deriving changes no session's own rows (docs/measures.md, convergence).
    let changed: i64 = index
        .query_row("SELECT changed FROM session", [], |r| r.get(0))
        .unwrap();
    assert_eq!(changed, 1);
    assert_eq!(
        super::run_all(&mut index, &catalog, &versioned_rules(2)).unwrap(),
        0
    );
    assert_eq!(
        super::run_all(
            &mut index,
            &synthetic_catalog("second"),
            &versioned_rules(2),
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
    use super::Output;
    first.push(super::Derivation {
        products: &[],
        name: "grandchild",
        version: super::Version::Rules(1),
        after: &["upstream", "dependent"],
        dirtied_by: &[],
        compute: |_, _, _| Ok(Output::Columns(Vec::new())),
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
    use super::{Derivation, Output, Version};
    let mut index = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_index(&index).unwrap();
    index.execute_batch("INSERT INTO session(id,origin,native_id,kind,changed) VALUES ('test:a','test','a','main',1),('test:b','test','b','main',1); INSERT INTO meta VALUES ('write_counter','1')").unwrap();
    let mut rule = Derivation {
        products: &[],
        name: "synthetic",
        version: Version::Rules(1),
        after: &[],
        dirtied_by: &[],
        compute: |_, _, _| {
            Ok(Output::Columns(vec![(
                "derived_title",
                Some("before".into()),
            )]))
        },
        class: super::Class::Session,
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
    // Sessions are taken newest first: the second one's write fails after
    // the first one's applied, in the same transaction.
    rule.compute = |_, pk, _| {
        let column = if pk == 1 {
            "no_such_column"
        } else {
            "derived_title"
        };
        Ok(Output::Columns(vec![(column, Some("after".into()))]))
    };
    assert!(
        super::run_all(&mut index, &catalog, std::slice::from_ref(&rule))
            .unwrap_err()
            .to_string()
            .contains("no_such_column")
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
    rule.compute = |_, _, _| {
        Ok(Output::Columns(vec![(
            "derived_title",
            Some("after".into()),
        )]))
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
    super::run_all(&mut index, &catalog, &rules).unwrap();
    index
        .execute("UPDATE session SET changed = 2 WHERE pk = 1", [])
        .unwrap();
    // Statements the first pass cached are prepared again after a schema
    // change, a number of times that depends on that pass's history; the
    // measured pass starts with none cached.
    index.flush_prepared_statement_cache();
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
    assert_eq!(super::run_all(&mut index, &catalog, &rules).unwrap(), 3);
    index.progress_handler(0, None::<fn() -> bool>).unwrap();
    steps.load(Ordering::Relaxed)
}

#[test]
fn derive_discovery_steps_independent_of_session_count() {
    assert_eq!(derive_steps(10), derive_steps(2_000));
}

/// Writes sessions `from..to` from one input of their own, each a prompt and
/// 20 shell calls, so shell calls are most of the archive's calls, as in a
/// real one.
fn write_shell_sessions(w: &mut WriterArchive, from: u32, to: u32) {
    let mut sessions = Vec::new();
    let mut messages = Vec::new();
    for n in from..to {
        let name = format!("bulk{n}");
        sessions.push(Session::new(sid(&name), SessionKind::Main));
        messages.extend(prose(&name, 0, 1_000, "run the tests"));
        for call in 1..=20 {
            messages.extend(bash(
                &name,
                call,
                1_000 + i64::from(call) * 10,
                "cargo test",
                true,
                "ok",
            ));
        }
    }
    let records = messages.len() as u64;
    let output = polylogue_model::Output {
        sessions,
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
        ..polylogue_model::Output::default()
    };
    w.write_output(
        &format!("input-bulk-{from}"),
        "claude-code",
        &format!("/p/bulk-{from}"),
        output,
        records,
    );
}

/// SQLite steps of running every derivation on the session `name` from the
/// start, as a first build does, on an index without planner statistics, as
/// it is until its first build ends.
fn steps_to_derive_from_start(w: &WriterArchive, name: &str) -> usize {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    crate::sql_functions::register(&index).unwrap();
    index
        .execute_batch("DROP TABLE IF EXISTS sqlite_stat1")
        .unwrap();
    let pk: i64 = index
        .query_row(
            "SELECT pk FROM session WHERE id = ?1",
            [format!("claude-code:{name}")],
            |r| r.get(0),
        )
        .unwrap();
    let tx = index.transaction().unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    let inputs = super::Inputs {
        catalog: polylogue_model::Catalog::builtin(),
        from_start: true,
    };
    let steps = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&steps);
    tx.progress_handler(
        1,
        Some(move || {
            counted.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    for derivation in super::ALL {
        (derivation.compute)(&tx, pk, &inputs)
            .unwrap()
            .apply(&tx, pk, &changes)
            .unwrap();
    }
    tx.progress_handler(0, None::<fn() -> bool>).unwrap();
    drop(changes);
    tx.rollback().unwrap();
    steps.load(Ordering::Relaxed)
}

/// Each derivation reads its session's rows through an index led by the
/// session (docs/measures.md: derived data is computed at session grain), so
/// its cost does not grow with the calls other sessions hold, even where
/// one action is most of the archive's calls and the planner has no
/// statistics.
#[test]
fn deriving_a_session_costs_the_same_however_many_calls_other_sessions_hold() {
    let mut w = WriterArchive::new();
    write_shell_sessions(&mut w, 0, 10);
    let mut target = prose("target", 0, 1_000, "fix the build");
    target.extend(bash("target", 1, 1_010, "cargo build", false, "error"));
    target.extend(bash("target", 2, 1_020, "cargo build", true, "ok"));
    w.write("target", target);
    // The first build ended at the first derive and gathered statistics.
    let analyzed: bool = rusqlite::Connection::open(w.archive.index_db())
        .unwrap()
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_stat1 WHERE tbl = 'tool_call')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(analyzed);
    // Rows of other sessions on both sides of the target's in every index,
    // so a range read ends at a neighbour's row in both measurements.
    for from in [10, 20] {
        write_shell_sessions(&mut w, from, from + 10);
    }
    let before = steps_to_derive_from_start(&w, "target");
    for from in (30..110).step_by(10) {
        write_shell_sessions(&mut w, from, from + 10);
    }
    // Not exactly equal: the passed-without-result update takes a few steps
    // fewer once more rows surround the session's. A read of other
    // sessions' calls adds thousands.
    let after = steps_to_derive_from_start(&w, "target");
    assert!(after <= before, "{before} steps became {after}");

    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let plan: Vec<String> = index
        .prepare(&format!(
            "EXPLAIN QUERY PLAN {}",
            super::tool_calls::SHELL_CALLS
        ))
        .unwrap()
        .query_map([1], |r| r.get(3))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(plan.len(), 1, "{plan:?}");
    assert!(
        plan[0].starts_with("SEARCH tool_call USING INDEX tool_call_")
            && plan[0].contains("(session_pk=?"),
        "{plan:?}"
    );
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

/// Waits up to a minute for `done`; a writer that waited for the held
/// derivation would never answer.
fn within_a_minute<T>(done: &std::sync::mpsc::Receiver<T>, what: &str) -> T {
    done.recv_timeout(std::time::Duration::from_secs(60))
        .unwrap_or_else(|_| panic!("{what} waited for the held derivation"))
}

#[test]
fn a_held_derivation_does_not_hold_parse_commits() {
    let mut w = WriterArchive::new();
    w.write_underived("a", prose("a", 1, 1_000, "the first session"));
    let sender = w.writer.sender().unwrap();
    // The derive thread stops after computing its first wave, as a slow
    // derivation would, before the writer is sent anything.
    let barrier = sender.pause_derive_for_test();
    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let title = |id: &str| -> Option<String> {
        index
            .query_row(
                "SELECT derived_title FROM session WHERE id = ?1",
                [format!("claude-code:{id}")],
                |r| r.get(0),
            )
            .unwrap()
    };
    std::thread::scope(|scope| {
        let deriving = scope.spawn(|| sender.derive().unwrap());
        barrier.wait().unwrap();
        let (written, done) = std::sync::mpsc::sync_channel(1);
        let w = &mut w;
        let writing = scope.spawn(move || {
            for n in 0..3 {
                w.write_underived(
                    &format!("b{n}"),
                    prose(&format!("b{n}"), 1, 2_000, "parsed meanwhile"),
                );
                written.send(()).unwrap();
            }
        });
        for _ in 0..3 {
            within_a_minute(&done, "a parse commit");
        }
        writing.join().unwrap();
        for n in 0..3 {
            let rows: i64 = index
                .query_row(
                    "SELECT count(*) FROM message m JOIN session s ON s.pk = m.session_pk
                     WHERE s.id = ?1",
                    [format!("claude-code:b{n}")],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(rows, 1);
        }
        // Nothing of the held pass is stored yet.
        assert_eq!(title("a"), None);
        barrier.release();
        assert!(deriving.join().unwrap() > 0);
    });
    assert_eq!(title("a").as_deref(), Some("the first session"));
    // The sessions written during the pass are derived by the next one.
    assert_eq!(title("b0"), None);
    assert!(w.writer.derive().unwrap() > 0);
    assert_eq!(title("b0").as_deref(), Some("parsed meanwhile"));
}

#[test]
fn a_value_computed_from_rows_that_changed_is_not_stored() {
    let mut w = WriterArchive::new();
    w.write_underived("a", prose("a", 1, 1_000, "the first prompt"));
    let sender = w.writer.sender().unwrap();
    let barrier = sender.pause_derive_for_test();
    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let state = || -> (Option<i64>, i64, i64, i64) {
        index
            .query_row(
                "SELECT (SELECT sp.last_at FROM session_span sp WHERE sp.session_pk = s.pk),
                        s.message_count,
                        (SELECT count(*) FROM derive_due d WHERE d.session_pk = s.pk),
                        (SELECT count(*) FROM derive_state d WHERE d.session_pk = s.pk)
                 FROM session s WHERE s.id = 'claude-code:a'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .unwrap()
    };
    std::thread::scope(|scope| {
        let deriving = scope.spawn(|| sender.derive().unwrap());
        // Computed from the session's first message; the second arrives
        // before the writer is sent the values.
        barrier.wait().unwrap();
        let (written, done) = std::sync::mpsc::sync_channel(1);
        let w = &mut w;
        scope.spawn(move || {
            w.write_underived("a", prose("a", 2, 3_000, "a later prompt"));
            written.send(()).unwrap();
        });
        within_a_minute(&done, "the append");
        barrier.release();
        assert_eq!(deriving.join().unwrap(), 0);
    });
    // The counts fold kept up with both messages; nothing the pass computed
    // was stored, and the session is still due.
    assert_eq!(state(), (None, 2, 1, 0));
    assert!(w.writer.derive().unwrap() > 0);
    assert_eq!(state(), (Some(3_000), 2, 0, 11));
}

#[test]
fn a_stop_reaches_the_derive_worker_before_writer_teardown() {
    let mut w = WriterArchive::new();
    w.write_underived("a", prose("a", 1, 1_000, "the first session"));
    let sender = w.writer.sender().unwrap();
    let stop = w.writer.stop_handle();
    let barrier = sender.pause_derive_for_test();
    std::thread::scope(|scope| {
        let deriving = scope.spawn(|| sender.derive());
        barrier.wait().unwrap();
        stop.request();
        barrier.release();
        assert!(matches!(
            deriving.join().unwrap(),
            Err(crate::StoreError::WriterGone)
        ));
    });
    // The pass remains due instead of being reported fully derived. Its
    // already committed waves are valid and the next writer finishes it.
    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM derive_due", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    drop(index);
    drop(w.writer);
    let writer = crate::writer::Writer::start(&w.archive).unwrap();
    assert!(writer.derive().unwrap() > 0);
    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM derive_due", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

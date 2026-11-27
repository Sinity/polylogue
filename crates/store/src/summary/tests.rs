use crate::fixture::writer::*;
use polylogue_model::{Json, RepoLookup, Session, SessionKind};
use serde_json::json;

fn one(w: &WriterArchive, sql: &str) -> Json {
    w.rows(sql).into_iter().next().unwrap_or(Json::Null)
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
fn a_prompt_title_is_cut_after_eighty_characters_with_an_ellipsis() {
    let exact = "a".repeat(80);
    assert_eq!(super::title::title_of(&exact), Some(exact.clone()));
    let long = "b".repeat(81);
    assert_eq!(
        super::title::title_of(&long),
        Some(format!("{}…", "b".repeat(80)))
    );
    // Only the first non-blank line is used.
    assert_eq!(
        super::title::title_of("\n  \n  second line \nthird"),
        Some("second line".to_owned())
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
        one(&w, "SELECT title, prompt_title FROM session"),
        json!({"title": null, "prompt_title": "Fix the flaky crash test"})
    );
}

#[test]
fn a_session_takes_its_repository_from_the_nearest_lookup() {
    let mut w = WriterArchive::new();
    let mut s = Session::new(sid("a"), SessionKind::Main);
    s.offer_cwd("/r/project/x/sub");
    w.repo_lookups.push(RepoLookup {
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
    w.repo_lookups.push(RepoLookup {
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
    w.repo_lookups.push(RepoLookup {
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
fn session_read_preserves_every_recorded_fact() {
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
    let messages = reader.messages(&session.id, false).unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].session, session.id);
    assert_eq!(
        messages[0].blocks,
        [polylogue_model::Block::Text {
            text: "A message".into()
        }]
    );
    assert_eq!(
        messages[0].at,
        polylogue_model::Timestamp::from_unix_ms(1234)
    );
}

#[test]
fn stored_products_declare_validity_and_tests() {
    use polylogue_model::derived::Validity;
    let products = super::products();
    assert_eq!(
        products.iter().map(|p| p.catalog).collect::<Vec<_>>(),
        ["K6", "K23", "K15", "K16"]
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
                response: None,
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset: i as u64,
                end: i as u64 + 1,
                message,
            })
            .collect(),
        links: vec![polylogue_model::Link {
            kind: polylogue_model::LinkKind::Spawn,
            copies_prefix: true,
            reason: None,
            parent: sid("parent"),
            child,
            branch_point: Some(polylogue_model::MessageId::parse("claude-code:parent-p1").unwrap()),
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
fn a_display_title_without_a_prose_prompt_is_unknown() {
    // The fork instructions are not a prompt, and the spawning call's
    // description is not a title: the display title stays unknown.
    for description in [Some("Audit the identity schemes"), None] {
        let mut w = WriterArchive::new();
        forked_subagent(&mut w, description);
        assert_eq!(
            one(
                &w,
                "SELECT prompt_title FROM session WHERE id = 'claude-code:agent-fork'"
            ),
            json!({"prompt_title": null})
        );
    }
}

#[test]
fn fork_with_parents_shows_inherited_count() {
    let mut w = WriterArchive::new();
    forked_subagent(&mut w, Some("Audit"));
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let child = sid("agent-fork");
    let selection = reader.select_messages(&child, true).unwrap();
    assert!(selection.parents().complete(), "{:?}", selection.parents());
    // The parent's prompt up to the branch point comes before the fork's
    // own message.
    let keys = reader.message_keys_with_parents(&child).unwrap();
    assert_eq!(keys.len(), 2, "{keys:?}");
}

/// SQLite steps of a summary pass after one of `sessions` summarized
/// sessions changed.
fn summary_steps(sessions: i64) -> usize {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut index = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_index(&index).unwrap();
    crate::sql_functions::register(&index).unwrap();
    index
        .execute(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?1)
             INSERT INTO session(id, origin, native_id, kind, changed)
             SELECT 'test:s' || i, 'test', 's' || i, 'main', 1 FROM n",
            [sessions],
        )
        .unwrap();
    crate::summary::mark_every_session(&index).unwrap();
    super::run(&mut index).unwrap();
    crate::summary::mark(&index, 1, &super::Key::start()).unwrap();
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
    assert_eq!(super::run(&mut index).unwrap(), 1);
    index.progress_handler(0, None::<fn() -> bool>).unwrap();
    steps.load(Ordering::Relaxed)
}

#[test]
fn summary_discovery_steps_independent_of_session_count() {
    assert_eq!(summary_steps(10), summary_steps(2_000));
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
                response: None,
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

/// SQLite steps of computing every summary of the session `name` from its
/// first row, as after a rebuild, on an index without planner statistics.
fn steps_to_summarize_from_start(w: &WriterArchive, name: &str) -> usize {
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
    index
        .execute("DELETE FROM summary_state WHERE session_pk = ?1", [pk])
        .unwrap();
    crate::summary::mark(&index, pk, &super::Key::start()).unwrap();
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
    assert_eq!(super::run(&mut index).unwrap(), 1);
    index.progress_handler(0, None::<fn() -> bool>).unwrap();
    steps.load(Ordering::Relaxed)
}

/// Each summary reads its session's rows through an index led by the
/// session (docs/measures.md: derived data is computed at session grain), so
/// its cost does not grow with the calls other sessions hold, even where
/// one action is most of the archive's calls and the planner has no
/// statistics.
#[test]
fn summarizing_a_session_costs_the_same_however_many_calls_other_sessions_hold() {
    let mut w = WriterArchive::new();
    write_shell_sessions(&mut w, 0, 10);
    let mut target = prose("target", 0, 1_000, "fix the build");
    target.extend(bash("target", 1, 1_010, "cargo build", false, "error"));
    target.extend(bash("target", 2, 1_020, "cargo build", true, "ok"));
    w.write("target", target);
    let reads = crate::ReadPool::serving(&w.archive, 1, 8 << 20).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .query(
                "SELECT command_norm FROM tool_call
                 WHERE session_pk=(SELECT pk FROM session WHERE id='claude-code:target')
                 ORDER BY at",
                &[],
            )
            .unwrap(),
        vec![
            json!({"command_norm":"cargo build"}),
            json!({"command_norm":"cargo build"}),
        ]
    );
    drop(reads);
    // Rows of other sessions on both sides of the target's in every index,
    // so a range read ends at a neighbour's row in both measurements.
    for from in [10, 20] {
        write_shell_sessions(&mut w, from, from + 10);
    }
    let before = steps_to_summarize_from_start(&w, "target");
    for from in (30..110).step_by(10) {
        write_shell_sessions(&mut w, from, from + 10);
    }
    // Not exactly equal: the passed-without-result update takes a few steps
    // fewer once more rows surround the session's. A read of other
    // sessions' calls adds thousands.
    let after = steps_to_summarize_from_start(&w, "target");
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
    w.write_underived("b", prose("b", 1, 2_000, "another prompt"));
    let stored_at: i64 = state(&w).1[0]["value"].as_str().unwrap().parse().unwrap();
    let stored = w.rows("SELECT changed FROM session WHERE id = 'claude-code:b'")[0]["changed"]
        .as_i64()
        .unwrap();
    assert!(stored <= stored_at);
    assert!(w.writer.derive().unwrap() > 0);
    let (sessions, counter) = state(&w);
    let counter: i64 = counter[0]["value"].as_str().unwrap().parse().unwrap();
    assert!(counter > stored_at, "the derive wrote after the rows");
    assert_eq!(sessions[0], before.0[0]);
    assert_eq!(sessions[1]["changed"].as_i64().unwrap(), stored);
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
                response: None,
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
    // The first derive installs the built index; then, as after their rows
    // changed, every session is due again.
    w.writer.derive().unwrap();
    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    index
        .execute(
            "INSERT INTO summary_due(session_pk, from_order) SELECT pk, X'' FROM session WHERE true
             ON CONFLICT(session_pk) DO UPDATE SET from_order = X''",
            [],
        )
        .unwrap();
    let due = || -> i64 {
        index
            .query_row("SELECT count(*) FROM summary_due", [], |r| r.get(0))
            .unwrap()
    };
    let sender = w.writer.sender();
    let object = polylogue_query::plan::prepare(
        "sessions where title:between",
        &polylogue_query::Bindings::default(),
        |_| Ok(()),
    )
    .unwrap()
    .query_object()
    .unwrap();
    let pause = sender.pause_user_dispatch_for_test(
        &object.id.to_string(),
        crate::writer::DispatchPhase::BeforeParentCheck,
    );
    std::thread::scope(|scope| {
        let deriving = scope.spawn(|| sender.derive().unwrap());
        // The user write is queued once the derive has committed its first
        // transaction.
        while due() == i64::try_from(sessions).unwrap() {
            std::thread::yield_now();
        }
        let writing = scope.spawn(|| sender.archive().query_object(object.clone(), None).unwrap());
        pause.wait().unwrap();
        let remaining = due();
        assert!(
            remaining > 0 && remaining < i64::try_from(sessions).unwrap(),
            "{remaining}"
        );
        pause.release();
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
    let sender = w.writer.sender();
    // The derive thread stops after computing its first wave, as a slow
    // derivation would, before the writer is sent anything.
    let pause = sender.pause_derive_for_test();
    // The pass's first derive installs the index built so far; this test
    // reads it once the pass is held.
    let installed = w.archive.root.join("index.db");
    let index = || rusqlite::Connection::open(&installed).unwrap();
    let title = |id: &str| -> Option<String> {
        index()
            .query_row(
                "SELECT prompt_title FROM session WHERE id = ?1",
                [format!("claude-code:{id}")],
                |r| r.get(0),
            )
            .unwrap()
    };
    std::thread::scope(|scope| {
        let deriving = scope.spawn(|| sender.derive().unwrap());
        pause.wait().unwrap();
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
            let rows: i64 = index()
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
        pause.release();
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
    let sender = w.writer.sender();
    let pause = sender.pause_derive_for_test();
    // The pass's first derive installs the index built so far; this test
    // reads it once the pass has run.
    let installed = w.archive.root.join("index.db");
    let state = || -> (Option<i64>, i64, i64, i64) {
        rusqlite::Connection::open(&installed)
            .unwrap()
            .query_row(
                "SELECT (SELECT sp.last_at FROM session_span sp WHERE sp.session_pk = s.pk),
                        s.message_count,
                        (SELECT count(*) FROM summary_due d WHERE d.session_pk = s.pk),
                        (SELECT count(*) FROM summary_state d WHERE d.session_pk = s.pk)
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
        pause.wait().unwrap();
        let (written, done) = std::sync::mpsc::sync_channel(1);
        let w = &mut w;
        scope.spawn(move || {
            w.write_underived("a", prose("a", 2, 3_000, "a later prompt"));
            written.send(()).unwrap();
        });
        within_a_minute(&done, "the append");
        pause.release();
        assert_eq!(deriving.join().unwrap(), 0);
    });
    // Nothing the pass computed was stored, and the session is still due.
    assert_eq!(state(), (None, 0, 1, 0));
    assert!(w.writer.derive().unwrap() > 0);
    assert_eq!(state(), (Some(3_000), 2, 0, super::ALL.len() as i64));
}

#[test]
fn a_stop_reaches_the_derive_worker_before_writer_teardown() {
    let mut w = WriterArchive::new();
    w.write_underived("a", prose("a", 1, 1_000, "the first session"));
    let sender = w.writer.sender();
    let stop = w.archive.stop().clone();
    let pause = sender.pause_derive_for_test();
    std::thread::scope(|scope| {
        let deriving = scope.spawn(|| sender.derive());
        pause.wait().unwrap();
        stop.cancel();
        pause.release();
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
            .query_row("SELECT count(*) FROM summary_due", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    drop(index);
    drop(w.writer);
    let writer = crate::writer::Writer::start(&w.archive.reopened()).unwrap();
    assert!(writer.derive().unwrap() > 0);
    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM summary_due", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

mod revealed_lines;

/// The summary worker's cost of an append follows the append: one message
/// appended to a session of 10,000 is folded from the state the previous
/// run kept, reading only the new row, and gives what folding every row
/// from the start gives; its transcript lines are written from the message
/// before it.
#[test]
fn appending_one_message_to_a_long_session_reads_the_appended_row() {
    let mut w = WriterArchive::new();
    let mut messages = Vec::new();
    for n in 0..10_000 {
        messages.extend(prose("s", n, 1_000 + i64::from(n) * 20, "a question"));
    }
    w.write("s", messages);
    w.write_underived("s", prose("s", 10_000, 500_000, "one more question"));
    let index = crate::archive::open_read(&w.archive.index_db()).unwrap();
    crate::sql_functions::register(&index).unwrap();
    let (pk, from): (i64, Vec<u8>) = index
        .query_row(
            "SELECT s.pk, d.from_order FROM session s JOIN summary_due d ON d.session_pk = s.pk
             WHERE s.id = 'claude-code:s'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    let run = super::fold::Run { conn: &index };
    let appended = run.session(pk, &super::Key(from.clone())).unwrap();
    assert_eq!(appended.read, 1);
    let whole = run.session(pk, &super::Key::start()).unwrap();
    assert_eq!(whole.read, 10_001);
    // The lines start from the last message before the new one: two units
    // placed, one message made from its blocks, its two lines written.
    let before = super::lines::counted::now();
    let lines = super::lines::compute(&index, pk, &super::Key(from)).unwrap();
    let after = super::lines::counted::now();
    assert_eq!(lines.read, 2);
    assert_eq!(lines.written(), 2);
    assert_eq!(after[2] - before[2], 1);
    assert_eq!(appended.output.counts, whole.output.counts);
    assert_eq!(appended.output.values, whole.output.values);
    assert_eq!(
        serde_json::to_value(&appended.output.span).unwrap(),
        serde_json::to_value(&whole.output.span).unwrap()
    );
}

/// A transcript read made right after a commit, before any summary pass,
/// shows what the commit wrote: the reader asks the summary worker for the
/// session and waits for it.
#[test]
fn a_transcript_read_after_a_commit_shows_the_new_lines() {
    let mut w = WriterArchive::new();
    w.write("s", prose("s", 0, 1_000, "a first question"));
    let pool = crate::ReadPool::open(&w.archive).unwrap();
    let before = pool
        .reader()
        .unwrap()
        .item_index(&sid("s"))
        .unwrap()
        .unwrap();
    w.write_underived("s", prose("s", 1, 2_000, "a later question"));
    let due: i64 = rusqlite::Connection::open(w.archive.index_db())
        .unwrap()
        .query_row("SELECT count(*) FROM summary_due", [], |r| r.get(0))
        .unwrap();
    assert_eq!(due, 1);
    let reader = pool.reader().unwrap();
    let after = reader.item_index(&sid("s")).unwrap().unwrap();
    assert_eq!(after.entries.len(), before.entries.len() * 2);
    assert_eq!(after.entries[..before.entries.len()], before.entries[..]);
    assert_eq!(after.append_seq, before.append_seq + 1);
    assert_eq!(after.rewrite_version, before.rewrite_version);
    assert_eq!(reader.restarts(), 1);
    assert_eq!(
        after.entries,
        super::lines::computed_entries(reader.conn().unwrap(), after.session_pk).unwrap()
    );
}

/// A call's lines are found through `transcript_call`, by the session and
/// the call ID, without reading the session's lines.
#[test]
fn a_call_lookup_reads_no_line_of_its_session() {
    let mut w = WriterArchive::new();
    w.write("s", prose("s", 0, 1_000, "a question"));
    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    for (sql, wanted) in [
        (
            "SELECT position FROM transcript_call WHERE session_pk = ?1 AND call_id = ?2 ORDER BY position",
            "SEARCH transcript_call USING PRIMARY KEY (session_pk=? AND call_id=?)",
        ),
        (
            "SELECT c.call_id, c.position, c.class
             FROM json_each(?2) j CROSS JOIN transcript_call c
             WHERE c.session_pk = ?1 AND c.call_id = j.value",
            "SEARCH c USING PRIMARY KEY (session_pk=? AND call_id=?)",
        ),
    ] {
        let plan: Vec<String> = index
            .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
            .unwrap()
            .query_map(rusqlite::params![1, "[]"], |r| r.get(3))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert!(plan.iter().any(|step| step == wanted), "{plan:?}");
        assert!(
            !plan.iter().any(|step| step.contains("transcript_line")),
            "{plan:?}"
        );
    }
}

/// A run streams the session's rows and keeps states that do not grow with
/// the session: folding from the first row across 100 Git commits keeps
/// states of the same size near the end as near the start
/// (the links found are written, not carried in the state).
#[test]
fn a_run_from_the_first_row_keeps_states_flat_in_the_rows() {
    let mut w = WriterArchive::new();
    let mut messages = Vec::new();
    for n in 0..2_000u32 {
        let at = 1_000 + i64::from(n) * 20;
        if n % 20 == 0 {
            messages.extend(bash(
                "s",
                n,
                at,
                &format!("git commit -m step{n}"),
                true,
                &format!("[main {:07x}] step", 0x1000000 + n),
            ));
        } else {
            messages.extend(prose("s", n, at, "a question"));
        }
    }
    w.write_underived("s", messages);
    let index = crate::archive::open_read(&w.archive.index_db()).unwrap();
    crate::sql_functions::register(&index).unwrap();
    let pk: i64 = index
        .query_row(
            "SELECT pk FROM session WHERE id = 'claude-code:s'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let run = super::fold::Run { conn: &index }
        .session(pk, &super::Key::start())
        .unwrap();
    assert!(run.read >= 2_000, "{}", run.read);
    let sizes: Vec<usize> = run.kept.iter().map(|k| k.state.len()).collect();
    let (early, late) = (sizes[0], sizes[sizes.len() - 2]);
    // Counters gain digits; nothing is held per row or per commit.
    assert!(late <= early + 64, "a state of {early} bytes became {late}");
    assert_eq!(run.output.links.unwrap().commits().len(), 100);
}

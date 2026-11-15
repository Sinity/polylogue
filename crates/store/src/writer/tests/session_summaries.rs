use super::*;
use rusqlite::trace::{TraceEvent, TraceEventCodes};
use std::cell::Cell;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Copy, Debug, Default)]
struct Work {
    statements: u64,
    vm: u64,
    contribution_reads: u64,
    missing_old_reads: u64,
    script_vm: u64,
}
thread_local! { static WORK: Cell<Work> = Cell::new(Work::default()); }
fn observe(event: TraceEvent<'_>) {
    if let TraceEvent::Profile(statement, _) = event {
        WORK.with(|work| {
            let mut w = work.get();
            w.statements += 1;
            let steps = statement.get_status(rusqlite::StatementStatus::VmStep) as u64;
            let sql = statement.sql();
            // These row insertions and summary reads each own one execution
            // with the cache disabled; reused resolver statements are excluded.
            if sql.starts_with("INSERT INTO message(")
                || sql.starts_with("INSERT INTO block(")
                || sql.starts_with("INSERT INTO tool_call(")
                || sql.starts_with("UPDATE session SET")
                || sql.contains("run_by_script=1")
                || sql.starts_with("SELECT session_pk,duplicate_of")
            {
                w.vm += steps;
            }
            if sql.contains("run_by_script=1") {
                w.script_vm += steps;
            }
            w.contribution_reads += u64::from(
                statement
                    .sql()
                    .starts_with("SELECT session_pk,duplicate_of IS NULL"),
            );
            w.missing_old_reads +=
                u64::from(sql.starts_with("SELECT session_pk, parent_id, session_order, id"));
            work.set(w);
        });
    }
}
fn connection(legacy: bool) -> Connection {
    let c = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&c).unwrap();
    c.pragma_update(None, "foreign_keys", true).unwrap();
    c.set_prepared_statement_cache_capacity(0);
    if legacy {
        c.execute_batch(include_str!(
            "../../schema/session_counts_before_transaction.sql"
        ))
        .unwrap();
    }
    c
}
fn apply(tx: &Transaction<'_>, mut b: Batch, legacy: bool) {
    resolve::prepare_index(tx, &mut b, 1, false).unwrap();
    let changes = crate::change_summary::ChangeSummary::start(tx).unwrap();
    write_rows(
        &crate::text_rows::TextIds::new(tx, false).unwrap(),
        tx,
        &b,
        &b.parse[0],
        b.parse[0].resolution.as_ref().unwrap(),
        1,
        None,
        Some(&fixture_receipt()),
        &changes,
    )
    .unwrap();
    if !legacy {
        changes.finish_summaries(tx, Some(1)).unwrap();
    }
    let pks = tx
        .prepare("SELECT pk FROM session")
        .unwrap()
        .query_map([], |r| r.get::<_, i64>(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    for pk in pks {
        assert_eq!(
            crate::derive::counts::read(tx, pk, true).unwrap(),
            crate::derive::counts::compute(tx, pk).unwrap()
        );
    }
}
fn fresh(count: usize, input: &str, session: &str) -> Batch {
    batch(
        input,
        Route::Live,
        0,
        (0..count)
            .map(|n| {
                let mut m = text_message(
                    &format!("{input}-{n}"),
                    session,
                    if n % 2 == 0 {
                        Role::User
                    } else {
                        Role::Assistant
                    },
                    "one two three",
                    None,
                );
                if n % 2 != 0 {
                    m.blocks.push(Block::Thinking {
                        text: Some("four five".into()),
                        signature: None,
                        redacted: false,
                    });
                }
                row(n as u64, m)
            })
            .collect(),
        &[session],
    )
}
#[test]
fn fresh_message_counts_write_once_and_do_not_read_back_each_contribution() {
    let measure = |count, history, legacy, history_session| {
        let mut c = connection(legacy);
        if history > 0 {
            let tx = c.transaction().unwrap();
            let messages = (0..history)
                .map(|n| {
                    let mut m = text_message(
                        &format!("old-{n}"),
                        history_session,
                        Role::Assistant,
                        "",
                        None,
                    );
                    m.blocks = vec![Block::ToolCall {
                        call_id: Some(format!("old-call-{n}")),
                        tool: Some("run".into()),
                        action: polylogue_model::ToolAction::Code,
                        input: Json::Null,
                        written_text: None,
                        command: None,
                        path: None,
                        run_by_script: false,
                    }];
                    row(n as u64, m)
                })
                .collect();
            apply(
                &tx,
                batch("old", Route::Live, 0, messages, &[history_session]),
                legacy,
            );
            tx.commit().unwrap();
        }
        let updates = Arc::new(AtomicU64::new(0));
        let seen = Arc::clone(&updates);
        c.update_hook(Some(move |action, _: &str, table: &str, _: i64| {
            if action == rusqlite::hooks::Action::SQLITE_UPDATE && table == "session" {
                seen.fetch_add(1, Ordering::Relaxed);
            }
        }))
        .unwrap();
        let tx = c.transaction().unwrap();
        WORK.with(|w| w.set(Work::default()));
        tx.trace_v2(TraceEventCodes::SQLITE_TRACE_PROFILE, Some(observe));
        // Only the application belongs to this observation; independent full
        // count verification is performed after stopping the trace below.
        let mut b = fresh(count, "fresh", "s");
        resolve::prepare_index(&tx, &mut b, 1, false).unwrap();
        let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
        write_rows(
            &crate::text_rows::TextIds::new(&tx, false).unwrap(),
            &tx,
            &b,
            &b.parse[0],
            b.parse[0].resolution.as_ref().unwrap(),
            1,
            None,
            Some(&fixture_receipt()),
            &changes,
        )
        .unwrap();
        if !legacy {
            changes.finish_summaries(&tx, Some(1)).unwrap();
        }
        tx.trace_v2(TraceEventCodes::empty(), None);
        let w = WORK.with(Cell::get);
        let updates = updates.load(Ordering::Relaxed);
        let pk: i64 = tx
            .query_row("SELECT pk FROM session WHERE id='claude-code:s'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(
            crate::derive::counts::read(&tx, pk, true).unwrap(),
            crate::derive::counts::compute(&tx, pk).unwrap()
        );
        (w, updates)
    };
    let small = measure(1, 0, false, "unrelated");
    let large = measure(1024, 0, false, "unrelated");
    let history = measure(1024, 4096, false, "unrelated");
    let same_session = measure(1024, 4096, false, "s");
    let old = measure(1024, 4096, true, "s");
    eprintln!(
        "session count writer work: small={small:?} large={large:?} history={history:?} same_session={same_session:?} row-trigger control={old:?}"
    );
    // One activity update and one final summary update, independent of message count.
    assert_eq!(small.1, 2);
    assert_eq!(large.1, 2);
    assert_eq!(history.1, 2);
    assert_eq!(same_session.1, 2);
    assert_eq!(small.0.script_vm, large.0.script_vm);
    assert_eq!(large.0.script_vm, history.0.script_vm);
    assert_eq!(history.0.script_vm, same_session.0.script_vm);
    assert_eq!(large.0.contribution_reads, 0);
    assert_eq!(history.0.contribution_reads, 0);
    assert_eq!(small.0.missing_old_reads, 0);
    assert_eq!(large.0.missing_old_reads, 0);
    assert_eq!(history.0.missing_old_reads, 0);
    assert_eq!(same_session.0.missing_old_reads, 0);
    assert!(old.1 >= 1024);
    assert!(same_session.0.vm < old.0.vm);
    // PROFILE counts top-level executions; trigger programs are included in VM steps.
}

#[test]
fn count_contributions_keep_script_changes_late_results_and_rollback_exact() {
    use polylogue_model::{ErrorReason, ToolAction, ToolOutcome};
    let mut c = connection(false);
    let mut code = text_message("code", "s", Role::Assistant, "", None);
    code.blocks = vec![Block::ToolCall {
        call_id: Some("code".into()),
        tool: Some("run".into()),
        action: ToolAction::Code,
        input: Json::Null,
        written_text: None,
        command: None,
        path: None,
        run_by_script: false,
    }];
    {
        let tx = c.transaction().unwrap();
        apply(
            &tx,
            batch("code-input", Route::Live, 0, vec![row(0, code)], &["s"]),
            false,
        );
        tx.commit().unwrap();
    }
    let before: i64 = c
        .query_row("SELECT tool_call_count FROM session", [], |r| r.get(0))
        .unwrap();
    assert_eq!(before, 1);
    let mut script = text_message("command", "s", Role::Assistant, "", None);
    script.blocks = vec![Block::ToolCall {
        call_id: Some("command".into()),
        tool: Some("shell".into()),
        action: ToolAction::Shell,
        input: Json::Null,
        written_text: None,
        command: Some("false".into()),
        path: None,
        run_by_script: true,
    }];
    {
        let tx = c.transaction().unwrap();
        apply(
            &tx,
            batch("script-input", Route::Live, 0, vec![row(0, script)], &["s"]),
            false,
        );
        tx.commit().unwrap();
    }
    let mut result = text_message("result", "s", Role::User, "", None);
    result.blocks = vec![Block::ToolResult {
        reported_duration_ms: None,
        call_id: Some("command".into()),
        outcome: ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(1)),
        },
        output: Some("failed".into()),
        data: None,
        attachments: vec![],
    }];
    let late = batch("late-input", Route::Live, 0, vec![row(0, result)], &["s"]);
    {
        let tx = c.transaction().unwrap();
        apply(&tx, late.clone(), false);
        assert_eq!(
            tx.query_row("SELECT error_count FROM session", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
    }
    assert_eq!(
        c.query_row("SELECT error_count FROM session", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    {
        let tx = c.transaction().unwrap();
        apply(&tx, late, false);
        tx.commit().unwrap();
    }
    let tx = c.transaction().unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    let call: i64 = tx
        .query_row("SELECT pk FROM tool_call WHERE run_by_script=1", [], |r| {
            r.get(0)
        })
        .unwrap();
    changes.call(&tx, call).unwrap();
    tx.execute("UPDATE tool_call SET run_by_script=0 WHERE pk=?1", [call])
        .unwrap();
    changes.finish_summaries(&tx, Some(2)).unwrap();
    let pk: i64 = tx
        .query_row("SELECT pk FROM session", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        crate::derive::counts::read(&tx, pk, true).unwrap(),
        crate::derive::counts::compute(&tx, pk).unwrap()
    );
    assert_eq!(
        tx.query_row("SELECT tool_call_count,error_count FROM session", [], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
        })
        .unwrap(),
        (2, 1)
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
            .get::<_, i64>(
            0
        ))
        .unwrap(),
        0
    );
}

#[test]
fn reopened_missing_count_baseline_is_folded_without_blessing_a_stale_version() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("index.db");
    {
        let mut c = Connection::open(&path).unwrap();
        crate::schema::create_index(&c).unwrap();
        let tx = c.transaction().unwrap();
        apply(&tx, fresh(8, "old", "s"), false);
        tx.commit().unwrap();
        c.execute("UPDATE session SET word_count=NULL", []).unwrap();
        c.execute("INSERT INTO derive_state(name,session_pk,version)SELECT 'counts',pk,'stale' FROM session",[]).unwrap();
    }
    let mut c = Connection::open(&path).unwrap();
    crate::schema::ensure_index(&c).unwrap();
    assert_eq!(
        c.query_row("SELECT word_count FROM session", [], |r| r
            .get::<_, Option<i64>>(0))
            .unwrap(),
        None
    );
    {
        let tx = c.transaction().unwrap();
        apply(&tx, fresh(1, "new", "s"), false);
        tx.commit().unwrap();
    }
    assert_eq!(
        c.query_row("SELECT message_count FROM session", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        9
    );
    assert_eq!(
        c.query_row(
            "SELECT version FROM derive_state WHERE name='counts'",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        "stale"
    );
    let current = crate::session_sql::current("s", "counts");
    assert!(
        !c.query_row(&format!("SELECT {current} FROM session s"), [], |r| r
            .get::<_, bool>(0))
            .unwrap()
    );
}

#[test]
fn fresh_installed_counts_are_invalidated_by_same_transaction_visibility_or_session_changes() {
    for mutation in [
        "duplicate_of='another'",
        "on_active_path=0",
        "session_pk=(SELECT pk FROM session WHERE id='claude-code:to')",
    ] {
        let mut c = connection(false);
        let tx = c.transaction().unwrap();
        let mut message = text_message("fresh", "s", Role::Assistant, "", None);
        message.blocks.clear();
        let mut b = batch("fresh", Route::Live, 0, vec![row(0, message)], &["s", "to"]);
        resolve::prepare_index(&tx, &mut b, 1, false).unwrap();
        let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
        write_rows(
            &crate::text_rows::TextIds::new(&tx, false).unwrap(),
            &tx,
            &b,
            &b.parse[0],
            b.parse[0].resolution.as_ref().unwrap(),
            1,
            None,
            Some(&fixture_receipt()),
            &changes,
        )
        .unwrap();
        let pk = tx
            .query_row("SELECT pk FROM message", [], |r| r.get::<_, i64>(0))
            .unwrap();
        changes.message_metadata(&tx, pk).unwrap();
        tx.execute(&format!("UPDATE message SET {mutation} WHERE pk=?1"), [pk])
            .unwrap();
        changes.finish_summaries(&tx, Some(1)).unwrap();
        let sessions = tx
            .prepare("SELECT pk FROM session")
            .unwrap()
            .query_map([], |r| r.get::<_, i64>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        for session in sessions {
            assert_eq!(
                crate::derive::counts::read(&tx, session, true).unwrap(),
                crate::derive::counts::compute(&tx, session).unwrap(),
                "{mutation}"
            );
        }
        assert_eq!(
            tx.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
                .get::<_, i64>(
                0
            ))
            .unwrap(),
            0
        );
    }
}

#[test]
fn compaction_moves_and_removal_keep_both_sessions_exact_across_rollback() {
    let mut c = connection(false);
    let mut b = batch("events", Route::Live, 0, vec![], &["s", "to"]);
    b.parse[0].output.events.push(Event {
        session: Some(sid("s")),
        kind: "compaction".into(),
        at: None,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: None,
        range: None,
        data: serde_json::to_value(polylogue_model::CompactionBoundary::default()).unwrap(),
    });
    {
        let tx = c.transaction().unwrap();
        apply(&tx, b, false);
        tx.commit().unwrap();
    }
    for remove in [false, true] {
        let tx = c.transaction().unwrap();
        let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
        let event = tx
            .query_row("SELECT pk FROM event", [], |r| r.get::<_, i64>(0))
            .unwrap();
        // A later real mutation must upgrade an earlier derived-only refresh.
        changes.event_items(&tx, event).unwrap();
        changes.event(&tx, event).unwrap();
        if remove {
            tx.execute("DELETE FROM event_source WHERE event_pk=?1", [event])
                .unwrap();
            tx.execute("DELETE FROM event WHERE pk=?1", [event])
                .unwrap();
        } else {
            let target = tx
                .query_row(
                    "SELECT pk FROM session WHERE id='claude-code:to'",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .unwrap();
            changes.session(&tx, target).unwrap();
            tx.execute(
                "UPDATE event SET session_pk=?2 WHERE pk=?1",
                params![event, target],
            )
            .unwrap();
        }
        changes.finish_summaries(&tx, Some(2)).unwrap();
        let sessions = tx
            .prepare("SELECT pk FROM session")
            .unwrap()
            .query_map([], |r| r.get::<_, i64>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        for session in sessions {
            assert_eq!(
                crate::derive::counts::read(&tx, session, true).unwrap(),
                crate::derive::counts::compute(&tx, session).unwrap()
            );
        }
        assert_eq!(
            tx.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
                .get::<_, i64>(
                0
            ))
            .unwrap(),
            0
        );
    }
    assert_eq!(
        c.query_row(
            "SELECT compaction_count FROM session WHERE id='claude-code:s'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
}

#[test]
fn fresh_contributions_match_the_full_fold_for_every_recorded_author() {
    let mut c = connection(false);
    let tx = c.transaction().unwrap();
    let mut messages = Vec::new();
    for (n, role, kind) in [
        (0, Role::User, Some(PromptKind::Prose)),
        (1, Role::User, Some(PromptKind::Command)),
        (2, Role::User, None),
        (3, Role::User, Some(PromptKind::InjectedContext)),
        (4, Role::User, Some(PromptKind::Model)),
        (5, Role::Assistant, None),
        (6, Role::Assistant, Some(PromptKind::Harness)),
        (7, Role::System, None),
    ] {
        let mut m = text_message(&format!("author-{n}"), "s", role, "one two three", None);
        m.prompt_kind = kind;
        messages.push(row(n, m));
    }
    apply(
        &tx,
        batch("authors", Route::Live, 0, messages, &["s"]),
        false,
    );
    let got=tx.query_row("SELECT human_prompt_count,human_command_count,human_word_count,model_word_count,word_count FROM session",[],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?,r.get::<_,i64>(4)?))).unwrap();
    assert_eq!(got, (1, 1, 6, 6, 12));
}

#[test]
fn background_outcomes_change_counts_only_for_the_named_call() {
    use polylogue_model::{ErrorReason, ToolAction, ToolOutcome};
    let mut c = connection(false);
    let mut call = text_message("call", "s", Role::Assistant, "", None);
    call.blocks = vec![Block::ToolCall {
        call_id: Some("named".into()),
        tool: Some("run".into()),
        action: ToolAction::Shell,
        input: Json::Null,
        written_text: None,
        command: None,
        path: None,
        run_by_script: false,
    }];
    {
        let tx = c.transaction().unwrap();
        apply(
            &tx,
            batch("call", Route::Live, 0, vec![row(0, call)], &["s"]),
            false,
        );
        tx.commit().unwrap();
    }
    for (n, call_id, errors) in [(0, None, 0), (1, Some("missing"), 0), (2, Some("named"), 1)] {
        let mut b = batch(&format!("background-{n}"), Route::Live, 0, vec![], &["s"]);
        b.parse[0]
            .output
            .background
            .push(polylogue_model::batch::BackgroundUpdate {
                result_at: None,
                result_at_evidence: Default::default(),
                reported_duration_ms: None,
                session: sid("s"),
                task_id: "synthetic-task".into(),
                call_id: call_id.map(str::to_owned),
                outcome: ToolOutcome::Error {
                    reason: Some(ErrorReason::ExitCode(1)),
                },
                output_file: None,
            });
        let tx = c.transaction().unwrap();
        apply(&tx, b, false);
        assert_eq!(
            tx.query_row("SELECT error_count FROM session", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            errors
        );
        tx.commit().unwrap();
    }
}

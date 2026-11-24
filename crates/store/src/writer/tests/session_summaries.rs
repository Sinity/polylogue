use super::*;

fn connection() -> Connection {
    let c = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&c).unwrap();
    c.pragma_update(None, "foreign_keys", true).unwrap();
    c
}
fn apply(tx: &Transaction<'_>, b: Batch) {
    let step = b.parse[0].clone();
    write_test_rows(tx, &b, &step, 1, 1, false).unwrap();
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
fn count_contributions_keep_script_changes_late_results_and_rollback() {
    use polylogue_model::{ErrorReason, ToolAction, ToolOutcome};
    let mut c = connection();
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
        apply(&tx, late.clone());
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
        apply(&tx, late);
        tx.commit().unwrap();
    }
    let tx = c.transaction().unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    let (call, session, call_id): (i64, i64, String) = tx
        .query_row(
            "SELECT pk,session_pk,call_id FROM tool_call WHERE run_by_script=1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    changes.call_result(&tx, session, &call_id).unwrap();
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
fn missing_count_baseline_is_recomputed_on_reopen() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("index.db");
    {
        let mut c = Connection::open(&path).unwrap();
        crate::schema::create_index(&c).unwrap();
        let tx = c.transaction().unwrap();
        apply(&tx, fresh(8, "old", "s"));
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
        apply(&tx, fresh(1, "new", "s"));
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
fn compaction_moves_and_removal_keep_both_sessions_across_rollback() {
    let mut c = connection();
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
        apply(&tx, b);
        tx.commit().unwrap();
    }
    for remove in [false, true] {
        let tx = c.transaction().unwrap();
        let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
        let event = tx
            .query_row("SELECT pk FROM event", [], |r| r.get::<_, i64>(0))
            .unwrap();
        changes.event(&tx, event).unwrap();
        if remove {
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
        for (id, expected) in [("s", 0), ("to", i64::from(!remove))] {
            assert_eq!(
                tx.query_row(
                    "SELECT compaction_count FROM session WHERE id=?1",
                    [format!("claude-code:{id}")],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
                expected,
                "{id} after {}",
                if remove { "removal" } else { "move" }
            );
        }
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
fn fresh_contributions_match_the_full_recompute() {
    let mut c = connection();
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
    apply(&tx, batch("authors", Route::Live, 0, messages, &["s"]));
    let got=tx.query_row("SELECT human_prompt_count,human_command_count,human_word_count,model_word_count,word_count FROM session",[],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?,r.get::<_,i64>(4)?))).unwrap();
    assert_eq!(got, (1, 1, 6, 6, 12));
}

#[test]
fn background_outcomes_change_counts_only_for_the_named_call() {
    use polylogue_model::{ErrorReason, ToolAction, ToolOutcome};
    let mut c = connection();
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
        apply(&tx, b);
        assert_eq!(
            tx.query_row("SELECT error_count FROM session", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            errors
        );
        tx.commit().unwrap();
    }
}

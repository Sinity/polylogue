use super::*;

fn connection() -> Connection {
    let c = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&c).unwrap();
    crate::sql_functions::register(&c).unwrap();
    c.pragma_update(None, "foreign_keys", true).unwrap();
    c
}
/// Writes `b`'s rows in one transaction, then computes the summaries of
/// the sessions it marked due; each session's stored counts then equal
/// those computed from all its rows.
fn apply(c: &mut Connection, b: Batch) {
    let step = b.parse[0].clone();
    {
        let tx = c.transaction().unwrap();
        write_test_rows(&tx, &b, &step, 1, 1).unwrap();
        tx.commit().unwrap();
    }
    summarize(c);
}

fn summarize(c: &mut Connection) {
    crate::summary::run(c).unwrap();
    let pks = c
        .prepare("SELECT pk FROM session")
        .unwrap()
        .query_map([], |r| r.get::<_, i64>(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    for pk in pks {
        assert_eq!(
            crate::summary::counts::read(c, pk, true).unwrap(),
            crate::summary::counts::compute(c, pk).unwrap()
        );
    }
}

#[test]
fn counts_follow_script_calls_and_late_results() {
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
    apply(
        &mut c,
        batch("code-input", Route::Live, 0, vec![row(0, code)], &["s"]),
    );
    let calls = |c: &Connection| -> (i64, i64) {
        c.query_row("SELECT tool_call_count,error_count FROM session", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap()
    };
    assert_eq!(calls(&c), (1, 0));
    // A call run by a script: the code call counts through its commands.
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
    apply(
        &mut c,
        batch("script-input", Route::Live, 0, vec![row(0, script)], &["s"]),
    );
    assert_eq!(calls(&c), (1, 0));
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
    apply(
        &mut c,
        batch("late-input", Route::Live, 0, vec![row(0, result)], &["s"]),
    );
    assert_eq!(calls(&c), (1, 1));
    // Without a script call the code call counts again.
    let pk: i64 = c
        .query_row("SELECT pk FROM session", [], |r| r.get(0))
        .unwrap();
    c.execute("UPDATE tool_call SET run_by_script=0", [])
        .unwrap();
    crate::summary::mark(&c, pk, &crate::summary::Key::start()).unwrap();
    summarize(&mut c);
    assert_eq!(calls(&c), (2, 1));
}

/// An event moved to another session, or removed, outside a re-parse:
/// what the change summary records of it before the change marks both
/// sessions due from where it was.
#[test]
fn a_moved_or_removed_compaction_is_counted_where_it_is() {
    for remove in [false, true] {
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
        apply(&mut c, b);
        let count = |c: &Connection, id: &str| -> i64 {
            c.query_row(
                "SELECT compaction_count FROM session WHERE id=?1",
                [format!("claude-code:{id}")],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!((count(&c, "s"), count(&c, "to")), (1, 0));
        {
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
                tx.execute(
                    "UPDATE event SET session_pk=(SELECT pk FROM session WHERE id='claude-code:to')
                     WHERE pk=?1",
                    [event],
                )
                .unwrap();
            }
            changes.mark_due(&tx, 2).unwrap();
            tx.commit().unwrap();
        }
        summarize(&mut c);
        assert_eq!(
            (count(&c, "s"), count(&c, "to")),
            (0, i64::from(!remove)),
            "after {}",
            if remove { "removal" } else { "move" }
        );
    }
}

#[test]
fn fresh_contributions_match_the_full_recompute() {
    let mut c = connection();
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
    apply(&mut c, batch("authors", Route::Live, 0, messages, &["s"]));
    let got=c.query_row("SELECT human_prompt_count,human_command_count,human_word_count,model_word_count,word_count FROM session",[],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?,r.get::<_,i64>(4)?))).unwrap();
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
    apply(
        &mut c,
        batch("call", Route::Live, 0, vec![row(0, call)], &["s"]),
    );
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
        apply(&mut c, b);
        assert_eq!(
            c.query_row("SELECT error_count FROM session", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            errors
        );
    }
}

use crate::fixture::writer::*;
use polylogue_model::Json;
use serde_json::json;

#[test]
fn no_task_from_prose() {
    // Law 90: a Bead ID in a prompt, or read with `bd show`, claims
    // nothing; a successful claim holds until a successful close.
    let mut w = WriterWorld::new();
    let mut a = prose("a", 1, 1_000, "please work on plnext-ony.7 next");
    a.extend(bash(
        "a",
        2,
        2_000,
        "bd show plnext-ony.7",
        true,
        "plnext-ony.7 · Work graph",
    ));
    w.write("a", a);
    w.write(
        "b",
        bash(
            "b",
            1,
            1_000,
            "bd --actor me update plnext-ony.7 --claim",
            true,
            "✓ claimed",
        ),
    );
    assert_eq!(
        w.live_task("a"),
        json!({"live_task": null, "live_task_evidence": null})
    );
    assert_eq!(
        w.live_task("b"),
        json!({"live_task": "plnext-ony.7", "live_task_evidence": "claim_command"})
    );
    let attempts = w.rows(
        "SELECT s.id AS session, n.state FROM work_node n JOIN session s ON s.pk = n.session_pk
         WHERE n.kind = 'attempt'",
    );
    assert_eq!(
        attempts,
        vec![json!({"session": "claude-code:b", "state": "open"})]
    );

    w.write(
        "b",
        bash(
            "b",
            2,
            5_000,
            "bd close plnext-ony.7 --reason done",
            true,
            "✓ closed",
        ),
    );
    assert_eq!(
        w.live_task("b"),
        json!({"live_task": null, "live_task_evidence": null})
    );
    let attempts = w.rows("SELECT state FROM work_node WHERE kind = 'attempt'");
    assert_eq!(attempts, vec![json!({"state": "closed"})]);
}

#[test]
fn a_failed_claim_claims_nothing() {
    let mut w = WriterWorld::new();
    w.write(
        "s",
        bash(
            "s",
            1,
            1_000,
            "bd update plnext-1 --status in_progress",
            false,
            "error: not found",
        ),
    );
    assert_eq!(
        w.rows("SELECT COUNT(*) AS n FROM work_node"),
        vec![json!({"n": 0})]
    );
}

#[test]
fn checks_and_commits_while_a_claim_is_open_belong_to_its_attempt() {
    let mut w = WriterWorld::new();
    let mut m = bash("s", 1, 1_000, "bd update plnext-9 --claim", true, "");
    m.extend(bash(
        "s",
        2,
        2_000,
        "cd /src && cargo test -p store 2>&1 | tail -20",
        false,
        "FAILED",
    ));
    m.extend(bash(
        "s",
        3,
        3_000,
        "cargo nextest run -p store",
        true,
        "ok",
    ));
    m.extend(bash(
        "s",
        4,
        4_000,
        "git add -A && git commit -m 'store: fix'",
        true,
        "[lane-x 1a2b3c4] store: fix\n 1 file changed",
    ));
    m.extend(bash(
        "s",
        5,
        5_000,
        "gh pr create --fill",
        true,
        "https://github.com/Sinity/polylogue/pull/6300\n",
    ));
    m.extend(bash("s", 6, 6_000, "bd close plnext-9", true, ""));
    // After the close: a check that belongs to no attempt.
    m.extend(bash("s", 7, 7_000, "cargo clippy", true, ""));
    w.write("s", m);

    let attempt = w.rows("SELECT state, attrs_json FROM work_node WHERE kind = 'attempt'");
    assert_eq!(attempt[0]["state"], "closed");
    let attrs: Json = serde_json::from_str(attempt[0]["attrs_json"].as_str().unwrap()).unwrap();
    assert_eq!(
        attrs,
        json!({"checks_passed": 1, "checks_failed": 1, "checks_unknown": 0, "artifacts": 2})
    );

    let checks = w.rows(
        "SELECT n.title, n.state, (SELECT COUNT(*) FROM work_edge e WHERE e.from_pk = n.pk) AS edges
         FROM work_node n WHERE kind = 'verification' ORDER BY n.at",
    );
    assert_eq!(
        checks,
        vec![
            json!({"title": "cargo test", "state": "error", "edges": 1}),
            json!({"title": "cargo nextest", "state": "ok", "edges": 1}),
            json!({"title": "cargo clippy", "state": "ok", "edges": 0}),
        ]
    );
    assert_eq!(
        w.rows("SELECT commit_hash, evidence FROM commit_link"),
        vec![json!({"commit_hash": "1a2b3c4", "evidence": "commit-output"})]
    );
    assert_eq!(
        w.rows("SELECT repo, number FROM pr_link"),
        vec![json!({"repo": "Sinity/polylogue", "number": 6300})]
    );
    let produced = w.rows(
        "SELECT a.title FROM work_edge e JOIN work_node a ON a.pk = e.to_pk
         WHERE e.kind = 'produces' ORDER BY a.at",
    );
    assert_eq!(
        produced,
        vec![
            json!({"title": "1a2b3c4"}),
            json!({"title": "Sinity/polylogue#6300"})
        ]
    );
}

#[test]
fn a_bead_claimed_by_two_sessions_is_one_goal() {
    let mut w = WriterWorld::new();
    w.write(
        "a",
        bash("a", 1, 1_000, "bd update plnext-5 --claim", true, ""),
    );
    w.write(
        "b",
        bash("b", 1, 2_000, "bd update plnext-5 --claim", true, ""),
    );
    assert_eq!(
        w.rows(
            "SELECT g.key, COUNT(e.from_pk) AS attempts FROM work_node g
             JOIN work_edge e ON e.to_pk = g.pk AND e.kind = 'pursues'
             WHERE g.kind = 'goal' GROUP BY g.pk"
        ),
        vec![json!({"key": "bead:plnext-5", "attempts": 2})]
    );
}

#[test]
fn declared_todo_items_are_session_goals() {
    let mut w = WriterWorld::new();
    let todos = |a: &str, b: &str| {
        json!({"todos": [
            {"content": "port the parser", "status": a, "activeForm": "Porting"},
            {"content": "write the laws", "status": b, "activeForm": "Writing"},
        ]})
    };
    let mut m = call(
        "s",
        1,
        1_000,
        "TodoWrite",
        None,
        todos("in_progress", "pending"),
        Ok(""),
    );
    m.extend(call(
        "s",
        2,
        2_000,
        "TodoWrite",
        None,
        todos("completed", "in_progress"),
        Ok(""),
    ));
    w.write("s", m);
    let goals = w.rows(
        "SELECT g.title, a.state FROM work_node g
         JOIN work_edge e ON e.to_pk = g.pk JOIN work_node a ON a.pk = e.from_pk
         WHERE g.kind = 'goal' ORDER BY g.at, g.title",
    );
    assert_eq!(
        goals,
        vec![
            json!({"title": "port the parser", "state": "completed"}),
            json!({"title": "write the laws", "state": "open"}),
        ]
    );
    assert_eq!(
        w.live_task("s")["live_task"],
        Json::Null,
        "a todo item is never a task"
    );
}

#[test]
fn deriving_again_changes_nothing() {
    let mut w = WriterWorld::new();
    w.write(
        "s",
        bash("s", 1, 1_000, "bd update plnext-1 --claim", true, ""),
    );
    let before = w.rows("SELECT key, kind, title, state FROM work_node ORDER BY key");
    assert_eq!(w.writer.derive().unwrap(), 0);
    let after = w.rows("SELECT key, kind, title, state FROM work_node ORDER BY key");
    assert_eq!(before, after);
}

#[test]
fn rows_an_excision_leaves_behind_are_swept() {
    // Excision deletes a session's nodes by their session; their edges go
    // with them, and a shared goal no one pursues any more goes at the next
    // derive.
    let mut w = WriterWorld::new();
    w.write(
        "a",
        bash("a", 1, 1_000, "bd update plnext-5 --claim", true, ""),
    );
    w.write(
        "b",
        bash("b", 1, 2_000, "bd update plnext-5 --claim", true, ""),
    );
    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let excise = |s: &str| {
        index
            .execute(
                "DELETE FROM work_node WHERE session_pk = (SELECT pk FROM session WHERE id = ?1)",
                [format!("claude-code:{s}")],
            )
            .unwrap();
    };
    excise("b");
    w.writer.derive().unwrap();
    let dangling = "SELECT COUNT(*) AS n FROM work_edge e WHERE NOT EXISTS
        (SELECT 1 FROM work_node n WHERE n.pk = e.from_pk)";
    assert_eq!(w.rows(dangling), vec![json!({"n": 0})]);
    assert_eq!(
        w.rows("SELECT COUNT(*) AS n FROM work_node WHERE kind = 'goal'"),
        vec![json!({"n": 1})],
        "a still pursues the goal"
    );
    excise("a");
    w.writer.derive().unwrap();
    assert_eq!(
        w.rows("SELECT COUNT(*) AS n FROM work_node"),
        vec![json!({"n": 0})]
    );
    assert_eq!(
        w.rows("SELECT COUNT(*) AS n FROM work_edge"),
        vec![json!({"n": 0})]
    );
}

#[test]
fn a_spawned_session_pursues_the_task_it_was_given() {
    let mut w = WriterWorld::new();
    w.write("parent", prose("parent", 1, 1_000, "coordinate"));
    let mut child = prose(
        "child",
        1,
        2_000,
        "Sinex lane B: replay and excision.\nDetails follow.",
    );
    child.extend(bash(
        "child",
        2,
        3_000,
        "nix develop --command cargo nextest run",
        true,
        "ok",
    ));
    w.links.push(polylogue_model::Link {
        kind: polylogue_model::LinkKind::Spawn,
        parent: sid("parent"),
        child: sid("child"),
        at: None,
        call_id: None,
        evidence: "test".into(),
    });
    w.write("child", child);
    let rows = w.rows(
        "SELECT g.title AS goal, a.state, c.title AS check_title FROM work_node g
         JOIN work_edge p ON p.to_pk = g.pk AND p.kind = 'pursues'
         JOIN work_node a ON a.pk = p.from_pk
         JOIN work_edge k ON k.to_pk = a.pk AND k.kind = 'checks'
         JOIN work_node c ON c.pk = k.from_pk",
    );
    assert_eq!(
        rows,
        vec![json!({
            "goal": "Sinex lane B: replay and excision.",
            "state": null,
            "check_title": "cargo nextest"
        })]
    );
}

#[test]
fn a_spawned_task_skips_harness_wrapper_blocks() {
    let fork = "<fork-boilerplate>\nYou are a worker fork.\n</fork-boilerplate>\n\nYour directive: fix the parser";
    assert_eq!(
        super::task_line(fork).as_deref(),
        Some("Your directive: fix the parser")
    );
    let reminder = "<system-reminder>ctx</system-reminder>\nRun the suite";
    assert_eq!(super::task_line(reminder).as_deref(), Some("Run the suite"));
    // Text that merely starts with markup is the task.
    assert_eq!(
        super::task_line("<b>fix X</b> please").as_deref(),
        Some("<b>fix X</b> please")
    );
    // An unclosed block still yields its first plain line.
    assert_eq!(
        super::task_line("<brief>\nDo the thing").as_deref(),
        Some("Do the thing")
    );
    assert_eq!(super::task_line("\n\n"), None);
}

#[test]
fn orphan_goals_are_removed_even_when_no_sessions_remain() {
    let mut index = rusqlite::Connection::open_in_memory().unwrap();
    crate::schema::create_index(&index).unwrap();
    index
        .execute(
            "INSERT INTO work_node(key, kind, title) VALUES ('bead:gone', 'goal', 'gone')",
            [],
        )
        .unwrap();
    assert_eq!(crate::derive::run(&mut index).unwrap(), 0);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM work_node", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn command_only_work_does_not_read_tool_result_bodies() {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};

    let mut w = WriterWorld::new();
    let mut messages = bash(
        "s",
        1,
        1_000,
        "cargo test",
        true,
        "large test output".repeat(4096).as_str(),
    );
    messages.extend(bash(
        "s",
        2,
        2_000,
        "git commit -m failed",
        false,
        "failed commit output",
    ));
    messages.extend(bash(
        "s",
        3,
        3_000,
        "gh pr view",
        true,
        "unrelated pull request output",
    ));
    messages.extend(bash(
        "s",
        4,
        4_000,
        "bd update plnext-9 --claim",
        true,
        "claimed",
    ));
    w.write("s", messages);
    let mut index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let session_pk: i64 = index
        .query_row("SELECT pk FROM session WHERE id='claude-code:s'", [], |r| {
            r.get(0)
        })
        .unwrap();
    index
        .authorizer(Some(|context: AuthContext<'_>| match context.action {
            AuthAction::Read {
                table_name: "text" | "block",
                column_name: "body",
            } => Authorization::Deny,
            _ => Authorization::Allow,
        }))
        .unwrap();
    let tx = index.transaction().unwrap();
    super::derive_session(&tx, session_pk, "claude-code:s").unwrap();
    tx.commit().unwrap();
    index
        .authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    assert_eq!(w.live_task("s")["live_task"], "plnext-9");
    assert_eq!(
        w.rows("SELECT count(*) AS n FROM work_node WHERE kind='verification'"),
        vec![json!({"n":1})]
    );
    assert_eq!(
        w.rows("SELECT count(*) AS n FROM commit_link"),
        vec![json!({"n":0})]
    );
}

#[test]
fn declared_plan_items_keep_their_states_without_command_text() {
    let mut w = WriterWorld::new();
    w.write(
        "s",
        call(
            "s",
            1,
            1_000,
            "update_plan",
            None,
            json!({"plan":[{"step":"inspect the writer", "status":"completed"},
                       {"step":"remove unused reads", "status":"in_progress"}]}),
            Ok("updated"),
        ),
    );
    assert_eq!(w.rows("SELECT g.title, a.state FROM work_node g JOIN work_edge e ON e.to_pk=g.pk JOIN work_node a ON a.pk=e.from_pk WHERE g.kind='goal' ORDER BY g.title"),
        vec![json!({"title":"inspect the writer", "state":"completed"}),json!({"title":"remove unused reads", "state":"open"})]);
}

#[test]
fn compound_artifact_commands_keep_text_and_legacy_packed_result_provenance() {
    for legacy_body in [false, true] {
        let mut w = WriterWorld::new();
        let output = "[work 1a2b3c4] store: improve work reads\nhttps://github.com/Sinity/polylogue/pull/6300\n";
        w.write(
            "s",
            bash(
                "s",
                1,
                1_000,
                "git commit -m 'store: improve work reads' && gh pr create --fill",
                true,
                output,
            ),
        );
        let mut index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
        if legacy_body {
            let body = crate::encoding::pack_json(
                &json!({"type":"tool_result", "outcome":{"kind":"ok"}, "output":output}),
            )
            .unwrap();
            index.execute("UPDATE block SET text_id=NULL, body=?1 WHERE pk IN (SELECT result_block_pk FROM tool_call)", [body]).unwrap();
        }
        let session_pk = index
            .query_row("SELECT pk FROM session WHERE id='claude-code:s'", [], |r| {
                r.get(0)
            })
            .unwrap();
        let tx = index.transaction().unwrap();
        super::derive_session(&tx, session_pk, "claude-code:s").unwrap();
        tx.commit().unwrap();
        assert_eq!(w.rows("SELECT c.commit_hash, c.evidence, c.evidence_block_pk=t.call_block_pk AS original_call FROM commit_link c JOIN tool_call t ON t.session_pk=c.session_pk"),
            vec![json!({"commit_hash":"1a2b3c4", "evidence":"commit-output", "original_call":1})]);
        assert_eq!(
            w.rows("SELECT repo, number, evidence FROM pr_link"),
            vec![json!({"repo":"Sinity/polylogue", "number":6300, "evidence":"gh-output"})]
        );
    }
}

fn checks_with_unknown_outcomes() -> WriterWorld {
    use polylogue_model::{Block, ToolOutcome, UnknownReason};
    let mut w = WriterWorld::new();
    let mut messages = bash("s", 1, 1_000, "bd update plnext-9 --claim", true, "claimed");
    for (n, outcome) in [
        (2, None),
        (3, Some(ToolOutcome::NoResult)),
        (
            4,
            Some(ToolOutcome::Unknown {
                reason: UnknownReason::StatusFieldUnreadable,
            }),
        ),
        (
            5,
            Some(ToolOutcome::Unknown {
                reason: UnknownReason::RunningInBackground,
            }),
        ),
    ] {
        let mut check = bash(
            "s",
            n,
            i64::from(n) * 1_000,
            "cargo test",
            true,
            "status output",
        );
        if let Some(value) = outcome {
            let Block::ToolResult { outcome, .. } = &mut check[1].blocks[0] else {
                unreachable!()
            };
            *outcome = value;
        } else {
            check.truncate(1);
        }
        messages.extend(check);
    }
    messages.extend(bash("s", 6, 6_000, "cargo check", true, "ok"));
    messages.extend(bash("s", 7, 7_000, "cargo clippy", false, "error"));
    w.write("s", messages);
    w
}

fn attempt_totals(w: &WriterWorld) -> Json {
    let rows = w.rows("SELECT attrs_json FROM work_node WHERE kind='attempt'");
    assert_eq!(rows.len(), 1);
    serde_json::from_str(rows[0]["attrs_json"].as_str().unwrap()).unwrap()
}

#[test]
fn unknown_checks_stay_separate_from_failures_and_later_completion_updates_totals() {
    use polylogue_model::{Block, Role, ToolOutcome};
    let mut w = checks_with_unknown_outcomes();
    assert_eq!(
        attempt_totals(&w),
        json!({"checks_passed":1,"checks_failed":1,"checks_unknown":4,"artifacts":0})
    );
    assert_eq!(w.rows("SELECT t.call_id,n.state,n.tool_call_pk=t.pk AS original_call FROM work_node n JOIN tool_call t ON t.pk=n.tool_call_pk WHERE n.kind='verification' ORDER BY t.at"), vec![
        json!({"call_id":"t2","state":"unknown","original_call":1}),
        json!({"call_id":"t3","state":"unknown","original_call":1}),
        json!({"call_id":"t4","state":"unknown","original_call":1}),
        json!({"call_id":"t5","state":"unknown","original_call":1}),
        json!({"call_id":"t6","state":"ok","original_call":1}),
        json!({"call_id":"t7","state":"error","original_call":1}),
    ]);
    w.write(
        "s",
        vec![message(
            "completed",
            "s",
            Role::User,
            8_000,
            vec![Block::ToolResult {
                call_id: Some("t5".into()),
                outcome: ToolOutcome::Ok,
                output: Some("finished".into()),
                data: None,
                attachments: vec![],
            }],
        )],
    );
    assert_eq!(
        attempt_totals(&w),
        json!({"checks_passed":2,"checks_failed":1,"checks_unknown":3,"artifacts":0})
    );
    assert_eq!(w.rows("SELECT n.state,n.tool_call_pk=t.pk AS original_call FROM work_node n JOIN tool_call t ON t.pk=n.tool_call_pk WHERE t.call_id='t5'"),vec![json!({"state":"ok","original_call":1})]);
}

#[test]
fn ordinary_derivation_corrects_attempt_totals_from_the_previous_work_rule() {
    let w = checks_with_unknown_outcomes();
    let mut old = super::DERIVATION;
    old.version = crate::derive::Version::Rules(1);
    let old_version = crate::derive::versions(
        &[crate::derive::tool_calls::DERIVATION, old],
        polylogue_model::Catalog::builtin(),
    )
    .unwrap()
    .pop()
    .unwrap();
    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    index
        .execute(
            "UPDATE work_node SET attrs_json=?1 WHERE kind='attempt'",
            [json!({"checks_passed":1,"checks_failed":5,"artifacts":0}).to_string()],
        )
        .unwrap();
    index
        .execute(
            "UPDATE derive_state SET version=?1 WHERE name='work'",
            [old_version],
        )
        .unwrap();
    assert_eq!(w.writer.derive().unwrap(), 1);
    assert_eq!(
        attempt_totals(&w),
        json!({"checks_passed":1,"checks_failed":1,"checks_unknown":4,"artifacts":0})
    );
    assert_eq!(w.writer.derive().unwrap(), 0);
}

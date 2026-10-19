use crate::writer::{Batch, InputKey, Observation, ParseStep};
use crate::{Archive, ReadPool, Writer};
use polylogue_model::{
    Accounting, Block, Disposition, ErrorReason, Json, Message, MessageId, MessageRow, Output,
    PromptKind, Role, Session, SessionId, SessionKind, Timestamp, ToolOutcome, Usage,
};
use serde_json::json;

struct World {
    _dir: tempfile::TempDir,
    archive: Archive,
    _lock: crate::ArchiveLock,
    writer: Writer,
    offset: u64,
    /// Links sent with the next write.
    links: Vec<polylogue_model::Link>,
}

fn world() -> World {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    World {
        _dir: dir,
        archive,
        _lock: lock,
        writer,
        offset: 0,
        links: vec![],
    }
}

fn sid(n: &str) -> SessionId {
    SessionId::parse(&format!("claude-code:{n}")).unwrap()
}

fn message(id: &str, session: &str, role: Role, at: i64, blocks: Vec<Block>) -> Message {
    Message {
        id: MessageId::parse(&format!("claude-code:{session}-{id}")).unwrap(),
        session: sid(session),
        parent: None,
        role,
        at: Timestamp::from_unix_ms(at),
        model: None,
        prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
        blocks,
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
    }
}

/// One tool call and its result, as two messages. `result` is the output,
/// `Ok` for a successful call and `Err` for a failed one.
fn call(
    session: &str,
    n: u32,
    at: i64,
    tool: &str,
    command: Option<&str>,
    input: Json,
    result: Result<&str, &str>,
) -> Vec<Message> {
    let (ok, output) = match result {
        Ok(o) => (true, o),
        Err(o) => (false, o),
    };
    let call_id = format!("t{n}");
    let c = message(
        &format!("c{n}"),
        session,
        Role::Assistant,
        at,
        vec![Block::ToolCall {
            call_id: Some(call_id.clone()),
            tool: Some(tool.into()),
            input,
            command: command.map(Into::into),
            path: None,
        }],
    );
    let outcome = if ok {
        ToolOutcome::Ok
    } else {
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(1)),
        }
    };
    let r = message(
        &format!("r{n}"),
        session,
        Role::User,
        at + 1,
        vec![Block::ToolResult {
            call_id: Some(call_id),
            outcome,
            output: Some(output.into()),
            data: None,
            attachments: vec![],
        }],
    );
    vec![c, r]
}

fn bash(session: &str, n: u32, at: i64, command: &str, ok: bool, output: &str) -> Vec<Message> {
    let result = if ok { Ok(output) } else { Err(output) };
    call(session, n, at, "Bash", Some(command), Json::Null, result)
}

fn prose(session: &str, n: u32, at: i64, text: &str) -> Vec<Message> {
    vec![message(
        &format!("p{n}"),
        session,
        Role::User,
        at,
        vec![Block::Text { text: text.into() }],
    )]
}

impl World {
    /// Writes `messages` as one appended step of the session's input.
    fn write(&mut self, session: &str, messages: Vec<Message>) {
        let start = self.offset;
        let rows: Vec<MessageRow> = messages
            .into_iter()
            .enumerate()
            .map(|(i, message)| MessageRow {
                offset: start + i as u64,
                end: start + i as u64 + 1,
                message,
            })
            .collect();
        let n = rows.len() as u64;
        self.offset += n + 1;
        let mut accounting = Accounting {
            framed: n,
            ..Accounting::default()
        };
        for _ in 0..n {
            accounting.add("record", &Disposition::Materialized);
        }
        let batch = Batch {
            input: InputKey {
                id: format!("input-{session}"),
                format: "claude-code".into(),
                route_rank: 0,
            },
            change: if start == 0 {
                crate::Change::First
            } else {
                crate::Change::Appended { from: start }
            },
            observation: Some(Observation {
                machine: "m".into(),
                host: "h".into(),
                path: format!("/p/{session}"),
                facts: None,
                seen_at_ms: 1,
            }),
            continues: false,
            piece: None,
            blobs: vec![],
            length: self.offset,
            release: Default::default(),
            parse: Some(ParseStep {
                output: Output {
                    sessions: vec![Session::new(sid(session), SessionKind::Main)],
                    messages: rows,
                    links: std::mem::take(&mut self.links),
                    ..Output::default()
                },
                accounting,
                start,
                parsed_upto: self.offset,
                parser_version: 1,
                hashes: vec![],
                packed: Default::default(),
                bodies: vec![],
                events: vec![],
            }),
        };
        self.writer.send(batch).unwrap();
        self.writer.flush().unwrap();
        self.writer.derive().unwrap();
    }

    fn rows(&self, sql: &str) -> Vec<Json> {
        let pool = ReadPool::open(&self.archive).unwrap();
        pool.reader().unwrap().query(sql, &[]).unwrap()
    }

    fn live_task(&self, session: &str) -> Json {
        self.rows(&format!(
            "SELECT live_task, live_task_evidence FROM session WHERE id = 'claude-code:{session}'"
        ))[0]
            .clone()
    }
}

#[test]
fn no_task_from_prose() {
    // Law 90: a Bead ID in a prompt, or read with `bd show`, claims
    // nothing; a successful claim holds until a successful close.
    let mut w = world();
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
    let mut w = world();
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
    let mut w = world();
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
        json!({"checks_passed": 1, "checks_failed": 1, "artifacts": 2})
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
    let mut w = world();
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
    let mut w = world();
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
    let mut w = world();
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
    let mut w = world();
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
    let mut w = world();
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

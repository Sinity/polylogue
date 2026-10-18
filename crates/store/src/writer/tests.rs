use super::*;
use crate::ReadPool;
use polylogue_model::{Disposition, Json, LinkKind, MessageId, Usage};

struct World {
    _dir: tempfile::TempDir,
    archive: Archive,
    _lock: crate::ArchiveLock,
    writer: Writer,
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
    }
}

fn sid(n: &str) -> SessionId {
    SessionId::parse(&format!("claude-code:{n}")).unwrap()
}

fn text_message(id: &str, session: &str, role: Role, text: &str, stop: Option<&str>) -> Message {
    Message {
        id: MessageId::parse(&format!("claude-code:{id}")).unwrap(),
        session: sid(session),
        parent: None,
        role,
        at: Timestamp::from_unix_ms(1_000),
        model: None,
        prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
        blocks: vec![Block::Text { text: text.into() }],
        usage: Usage::default(),
        stop_reason: stop.map(Into::into),
        reported_cost_usd: None,
        duration_ms: None,
    }
}

fn accounting(n: u64) -> Accounting {
    let mut a = Accounting {
        framed: n,
        ..Accounting::default()
    };
    for _ in 0..n {
        a.add("user", &Disposition::Materialized);
    }
    a
}

fn batch(input: &str, rank: u8, start: u64, rows: Vec<MessageRow>, sessions: &[&str]) -> Batch {
    let n = rows.len() as u64;
    Batch {
        input: InputKey {
            id: input.into(),
            format: "claude-code".into(),
            route_rank: rank,
        },
        change: if start == 0 {
            Change::First
        } else {
            Change::Appended { from: start }
        },
        observation: Some(Observation {
            machine: "m".into(),
            host: "h".into(),
            path: format!("/p/{input}"),
            facts: None,
            seen_at_ms: 1,
        }),
        continues: false,
        piece: None,
        blobs: vec![],
        length: start + 100,
        parse: Some(ParseStep {
            output: Output {
                sessions: sessions
                    .iter()
                    .map(|s| Session::new(sid(s), SessionKind::Main))
                    .collect(),
                messages: rows,
                ..Output::default()
            },
            accounting: accounting(n),
            start,
            parsed_upto: start + 100,
            parser_version: 1,
            hashes: vec![],
            packed: Default::default(),
            bodies: vec![],
            events: vec![],
        }),
    }
}

fn row(offset: u64, message: Message) -> MessageRow {
    MessageRow { offset, message }
}

fn texts(w: &World, session: &str) -> Vec<String> {
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    r.messages(&sid(session), false)
        .unwrap()
        .iter()
        .map(Message::text)
        .collect()
}

#[test]
fn messages_read_back_in_order_and_unchanged() {
    let w = world();
    let a = text_message("a", "s", Role::User, "hello", None);
    let b = text_message("b", "s", Role::Assistant, "hi", Some("end_turn"));
    w.writer
        .send(batch(
            "i1",
            0,
            0,
            vec![row(50, b.clone()), row(10, a.clone())],
            &["s"],
        ))
        .unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let pool = ReadPool::open(&w.archive).unwrap();
    let got = pool.reader().unwrap().messages(&sid("s"), false).unwrap();
    assert_eq!(got, vec![a, b]);
}

#[test]
fn the_same_record_sent_longer_replaces_itself() {
    let w = world();
    let short = text_message("r", "s", Role::Assistant, "part", None);
    let long = text_message("r", "s", Role::Assistant, "part and more", Some("end_turn"));
    w.writer
        .send(batch("i1", 0, 0, vec![row(0, short)], &["s"]))
        .unwrap();
    w.writer
        .send(batch("i1", 0, 100, vec![row(0, long)], &["s"]))
        .unwrap();
    w.writer.flush().unwrap();
    assert_eq!(texts(&w, "s"), ["part and more"]);
}

#[test]
fn within_one_input_the_earliest_record_wins() {
    let w = world();
    let first = text_message("u", "s", Role::User, "original", None);
    let dup = text_message("u", "s", Role::User, "conflicting duplicate", None);
    w.writer
        .send(batch(
            "i1",
            0,
            0,
            vec![row(10, first), row(90, dup)],
            &["s"],
        ))
        .unwrap();
    w.writer.flush().unwrap();
    assert_eq!(texts(&w, "s"), ["original"]);
}

#[test]
fn across_inputs_a_complete_version_beats_an_open_one_whichever_comes_first() {
    for order in [[0, 1], [1, 0]] {
        let w = world();
        let open = text_message("r", "s", Role::Assistant, "open", None);
        let done = text_message("r", "s", Role::Assistant, "done", Some("end_turn"));
        let batches = [
            batch("live", 0, 0, vec![row(0, open)], &["s"]),
            batch("import", 2, 0, vec![row(0, done)], &["s"]),
        ];
        for i in order {
            w.writer.send(batches[i].clone()).unwrap();
        }
        w.writer.flush().unwrap();
        assert_eq!(texts(&w, "s"), ["done"], "order {order:?}");
    }
}

#[test]
fn a_winning_version_from_another_session_moves_the_message_with_its_blocks() {
    let w = world();
    let mut open = text_message("r", "first", Role::Assistant, "open", None);
    open.blocks.push(Block::ToolCall {
        call_id: Some("t1".into()),
        tool: Some("Bash".into()),
        input: Json::Null,
        command: Some("ls".into()),
        path: None,
    });
    let done = text_message("r", "second", Role::Assistant, "done", Some("end_turn"));
    w.writer
        .send(batch("a", 0, 0, vec![row(0, open)], &["first"]))
        .unwrap();
    w.writer
        .send(batch("b", 0, 0, vec![row(0, done)], &["second"]))
        .unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(texts(&w, "first").is_empty());
    assert_eq!(texts(&w, "second"), ["done"]);
}

#[test]
fn a_tool_result_pairs_with_its_call_in_either_order() {
    for result_first in [false, true] {
        let w = world();
        let mut call = text_message("c", "s", Role::Assistant, "", Some("tool_use"));
        call.blocks = vec![Block::ToolCall {
            call_id: Some("t1".into()),
            tool: Some("Bash".into()),
            input: Json::Null,
            command: Some("cargo test".into()),
            path: None,
        }];
        let mut result = text_message("r", "s", Role::User, "", None);
        result.blocks = vec![Block::ToolResult {
            call_id: Some("t1".into()),
            outcome: ToolOutcome::Error {
                reason: Some(ErrorReason::ExitCode(101)),
            },
            output: Some("test failed".into()),
            data: None,
            attachments: vec![],
        }];
        let (first, second) = if result_first {
            (result, call)
        } else {
            (call, result)
        };
        w.writer
            .send(batch("i", 0, 0, vec![row(0, first)], &["s"]))
            .unwrap();
        w.writer
            .send(batch("i", 0, 100, vec![row(100, second)], &["s"]))
            .unwrap();
        w.writer.flush().unwrap();
        let pool = ReadPool::open(&w.archive).unwrap();
        let r = pool.reader().unwrap();
        let rows = r
            .query(
                "SELECT outcome, exit_code, result_block_pk IS NOT NULL AS paired FROM tool_call",
                &[],
            )
            .unwrap();
        assert_eq!(
            rows,
            vec![serde_json::json!({"outcome": "error", "exit_code": 101, "paired": 1})],
            "result_first {result_first}"
        );
        let hits = r
            .query(
                "SELECT COUNT(*) AS n FROM tool_call_trigram WHERE tool_call_trigram MATCH 'cargo'",
                &[],
            )
            .unwrap();
        assert_eq!(hits[0]["n"], 1);
    }
}

#[test]
fn prose_and_tool_output_are_searchable_in_the_same_transaction() {
    let w = world();
    let mut m = text_message("a", "s", Role::User, "find the walrus", None);
    m.blocks.push(Block::ToolResult {
        call_id: None,
        outcome: ToolOutcome::Ok,
        output: Some("compiler said narwhal".into()),
        data: None,
        attachments: vec![],
    });
    w.writer
        .send(batch("i", 0, 0, vec![row(0, m)], &["s"]))
        .unwrap();
    w.writer.flush().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let n = |sql: &str| r.query(sql, &[]).unwrap()[0]["n"].clone();
    assert_eq!(
        n("SELECT COUNT(*) AS n FROM text_fts WHERE text_fts MATCH 'walrus'"),
        1
    );
    assert_eq!(
        n("SELECT COUNT(*) AS n FROM output_fts WHERE output_fts MATCH 'narwhal'"),
        1
    );
    assert_eq!(
        n("SELECT COUNT(*) AS n FROM text_fts WHERE text_fts MATCH 'narwhal'"),
        0
    );
}

#[test]
fn session_facts_merge_and_a_pending_link_finds_its_child() {
    let w = world();
    let mut b1 = batch("i", 0, 0, vec![], &[]);
    let p = b1.parse.as_mut().unwrap();
    let mut s = Session::new(sid("parent"), SessionKind::Main);
    s.offer_cwd("/a");
    p.output.sessions = vec![s.clone()];
    p.output.links = vec![Link {
        kind: LinkKind::Spawn,
        parent: sid("parent"),
        child: sid("agent-1"),
        at: None,
        call_id: Some("toolu_1".into()),
        evidence: "agent_progress".into(),
    }];
    w.writer.send(b1).unwrap();
    let mut b2 = batch("j", 0, 0, vec![], &[]);
    let p = b2.parse.as_mut().unwrap();
    s.offer_cwd("/b");
    s.offer_title("named", TitleSource::Custom);
    p.output.sessions = vec![s, Session::new(sid("agent-1"), SessionKind::Subagent)];
    p.output.links = vec![Link {
        kind: LinkKind::Spawn,
        parent: sid("parent"),
        child: sid("agent-1"),
        at: None,
        call_id: None,
        evidence: "agentId".into(),
    }];
    w.writer.send(b2).unwrap();
    w.writer.flush().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let parent = r.session(&sid("parent")).unwrap().unwrap();
    assert_eq!(parent.cwd, ["/a", "/b"]);
    assert_eq!(parent.title.unwrap().text, "named");
    let link = r
        .query(
            "SELECT child_pk IS NOT NULL AS resolved, spawning_call FROM link",
            &[],
        )
        .unwrap();
    assert_eq!(
        link,
        vec![serde_json::json!({"resolved": 1, "spawning_call": "toolu_1"})]
    );
}

#[test]
fn copied_messages_are_stored_once_and_read_logically() {
    let w = world();
    // The parent's own file.
    let a = text_message("a", "parent", Role::User, "first", None);
    w.writer
        .send(batch(
            "parent-file",
            0,
            0,
            vec![row(0, a.clone())],
            &["parent"],
        ))
        .unwrap();
    // The child's file repeats the parent's message, then adds its own.
    let b = text_message("b", "child", Role::User, "second", None);
    let mut child = batch(
        "child-file",
        0,
        0,
        vec![row(0, a), row(50, b)],
        &["parent", "child"],
    );
    child.parse.as_mut().unwrap().output.links = vec![Link {
        kind: LinkKind::Copy,
        parent: sid("parent"),
        child: sid("child"),
        at: None,
        call_id: None,
        evidence: "sessionId".into(),
    }];
    w.writer.send(child).unwrap();
    w.writer.flush().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let n = r.query("SELECT COUNT(*) AS n FROM message", &[]).unwrap();
    assert_eq!(n[0]["n"], 2);
    let physical: Vec<_> = r
        .messages(&sid("child"), false)
        .unwrap()
        .iter()
        .map(Message::text)
        .collect();
    let logical: Vec<_> = r
        .messages(&sid("child"), true)
        .unwrap()
        .iter()
        .map(Message::text)
        .collect();
    assert_eq!(physical, ["second"]);
    assert_eq!(logical, ["first", "second"]);
}

#[test]
fn a_step_whose_counts_do_not_add_up_is_refused_alone() {
    let w = world();
    let good = batch(
        "good",
        0,
        0,
        vec![row(0, text_message("g", "s", Role::User, "ok", None))],
        &["s"],
    );
    let mut bad = batch(
        "bad",
        0,
        0,
        vec![row(0, text_message("x", "t", Role::User, "x", None))],
        &["t"],
    );
    bad.parse.as_mut().unwrap().accounting.framed = 5;
    w.writer.send(good).unwrap();
    w.writer.send(bad).unwrap();
    let report = w.writer.flush().unwrap();
    assert_eq!(report.errors.len(), 1, "{:?}", report.errors);
    assert_eq!(texts(&w, "s"), ["ok"]);
    assert!(texts(&w, "t").is_empty());
}

#[test]
fn revisions_extents_and_parse_state_follow_appends() {
    let w = world();
    let store = hashed_blobs::Store::open(w.archive.root.clone(), vec![], 0o600).unwrap();
    let mut b1 = batch("i", 0, 0, vec![], &[]);
    b1.piece = Some(Piece::prepare(&store, b"line one\n").unwrap());
    b1.length = 9;
    b1.parse.as_mut().unwrap().parsed_upto = 9;
    let mut b2 = batch("i", 0, 9, vec![], &[]);
    b2.piece = Some(Piece::prepare(&store, b"line two\npartial").unwrap());
    b2.length = 25;
    b2.parse.as_mut().unwrap().parsed_upto = 18;
    w.writer.send(b1).unwrap();
    w.writer.send(b2).unwrap();
    w.writer.flush().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let revs = r
        .query(
            "SELECT number, chain, produced, length FROM a.revision ORDER BY number",
            &[],
        )
        .unwrap();
    assert_eq!(
        revs,
        vec![
            serde_json::json!({"number": 1, "chain": 1, "produced": "first", "length": 9}),
            serde_json::json!({"number": 2, "chain": 1, "produced": "appended", "length": 25}),
        ]
    );
    let ext = r
        .query("SELECT offset, length FROM a.extent ORDER BY offset", &[])
        .unwrap();
    assert_eq!(
        ext,
        vec![
            serde_json::json!({"offset": 0, "length": 9}),
            serde_json::json!({"offset": 9, "length": 16}),
        ]
    );
    let known = r.known_inputs("m").unwrap();
    assert_eq!(known.len(), 1);
    assert_eq!(known[0].stored, 25);
    assert_eq!(known[0].parsed_upto, 18);
    let state = r.query("SELECT status FROM parse_state", &[]).unwrap();
    assert_eq!(state[0]["status"], "parsing");
}

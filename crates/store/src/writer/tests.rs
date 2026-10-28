use super::*;
use crate::ReadPool;
use polylogue_model::{
    Json, LinkKind, MessageId, PromptKind, RecordOutcome, SessionKind, TitleSource, Usage,
};

mod extent_checkpoint;
mod input_events;

struct TestArchive {
    _dir: tempfile::TempDir,
    archive: Archive,
    _lock: crate::ArchiveLock,
    writer: Writer,
}

fn test_archive() -> TestArchive {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    TestArchive {
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
        production: Default::default(),
    }
}

fn counts(n: u64) -> RecordCounts {
    let mut a = RecordCounts {
        records: n,
        ..RecordCounts::default()
    };
    for _ in 0..n {
        a.add("user", &RecordOutcome::Used);
    }
    a
}

fn batch(input: &str, rank: u8, start: u64, rows: Vec<MessageRow>, sessions: &[&str]) -> Batch {
    let n = rows.len() as u64;
    Batch {
        input: InputKey {
            id: input.into(),
            format: "claude-code".into(),
            arrival_rank: rank,
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
            import: None,
        }),
        continues: false,
        new_blob: None,
        blobs: vec![],
        length: start + 100,
        parse: vec![{
            let mut step = ParseStep::new(
                Output {
                    sessions: sessions
                        .iter()
                        .map(|s| Session::new(sid(s), SessionKind::Main))
                        .collect(),
                    messages: rows,
                    ..Output::default()
                },
                counts(n),
                start..start + 100,
                true,
            );
            step.reading_version = "synthetic=1".into();
            step
        }],
        release: InFlight::default(),
    }
}

fn row(offset: u64, message: Message) -> MessageRow {
    MessageRow {
        offset,
        end: offset,
        message,
    }
}

fn texts(w: &TestArchive, session: &str) -> Vec<String> {
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
    let w = test_archive();
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
    let w = test_archive();
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
    let w = test_archive();
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
        let w = test_archive();
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
    let w = test_archive();
    let mut open = text_message("r", "first", Role::Assistant, "open", None);
    open.blocks.push(Block::ToolCall {
        run_by_script: false,
        call_id: Some("t1".into()),
        tool: Some("Bash".into()),
        action: polylogue_model::ToolAction::Shell,
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
        let w = test_archive();
        let mut call = text_message("c", "s", Role::Assistant, "", Some("tool_use"));
        call.blocks = vec![Block::ToolCall {
            run_by_script: false,
            call_id: Some("t1".into()),
            tool: Some("Bash".into()),
            action: polylogue_model::ToolAction::Shell,
            input: Json::Null,
            command: Some("cargo test".into()),
            path: None,
        }];
        let mut result = text_message("r", "s", Role::User, "", None);
        result.blocks = vec![Block::ToolResult {
            reported_duration_ms: None,
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
        w.writer.derive().unwrap();
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
fn prose_and_tool_output_are_searchable_once_a_first_build_ends() {
    let w = test_archive();
    let mut m = text_message("a", "s", Role::User, "find the walrus", None);
    m.blocks.push(Block::ToolResult {
        reported_duration_ms: None,
        call_id: None,
        outcome: ToolOutcome::Ok,
        output: Some("compiler said narwhal".into()),
        data: None,
        attachments: vec![],
    });
    w.writer
        .send(batch("i", 0, 0, vec![row(0, m)], &["s"]))
        .unwrap();
    w.writer.derive().unwrap();
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

/// After a first build, new text reaches the search index from the search
/// thread, without a flush, within 2 s; `status`
/// counts it as pending until then.
#[test]
fn the_search_thread_indexes_live_writes_without_a_flush() {
    let w = test_archive();
    w.writer
        .send(batch(
            "i",
            0,
            0,
            vec![row(0, text_message("a", "s", Role::User, "first", None))],
            &["s"],
        ))
        .unwrap();
    w.writer.derive().unwrap(); // ends the first build
    w.writer
        .send(batch(
            "j",
            0,
            0,
            vec![row(
                0,
                text_message("b", "t", Role::User, "the okapi grazes", None),
            )],
            &["t"],
        ))
        .unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    loop {
        let r = pool.reader().unwrap();
        let hits = r
            .query(
                "SELECT COUNT(*) AS n FROM text_fts WHERE text_fts MATCH 'okapi'",
                &[],
            )
            .unwrap()[0]["n"]
            .clone();
        if hits == 1 {
            assert_eq!(r.status().unwrap().search_pending, 0);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

#[test]
fn session_facts_merge_and_a_pending_link_finds_its_child() {
    let w = test_archive();
    let mut b1 = batch("i", 0, 0, vec![], &[]);
    let p = b1.parse.first_mut().unwrap();
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
    let p = b2.parse.first_mut().unwrap();
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
    w.writer.derive().unwrap();
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
    let w = test_archive();
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
    child.parse.first_mut().unwrap().output.links = vec![Link {
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
    let w = test_archive();
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
    bad.parse.first_mut().unwrap().counts.records = 5;
    w.writer.send(good).unwrap();
    w.writer.send(bad).unwrap();
    let report = w.writer.flush().unwrap();
    assert_eq!(report.errors.len(), 1, "{:?}", report.errors);
    assert_eq!(texts(&w, "s"), ["ok"]);
    assert!(texts(&w, "t").is_empty());
}

#[test]
fn revisions_extents_and_parse_state_follow_appends() {
    let w = test_archive();
    let store = hashed_blobs::Store::open(w.archive.root.clone(), vec![], 0o600).unwrap();
    let mut b1 = batch("i", 0, 0, vec![], &[]);
    b1.new_blob = Some(NewBlob::prepare(&store, b"line one\n").unwrap());
    b1.length = 9;
    b1.parse.first_mut().unwrap().parsed_upto = 9;
    let mut b2 = batch("i", 0, 9, vec![], &[]);
    b2.new_blob = Some(NewBlob::prepare(&store, b"line two\npartial").unwrap());
    b2.length = 25;
    b2.parse.first_mut().unwrap().parsed_upto = 18;
    b2.parse.first_mut().unwrap().at_end = false;
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
    assert!(!known[0].waiting, "an incomplete chunk still has work");
    let state = r.query("SELECT status FROM parse_state", &[]).unwrap();
    assert_eq!(state[0]["status"], "parsing");
}

#[test]
fn waiting_at_eof_does_not_hide_a_stored_append_without_parse_progress() {
    let w = test_archive();
    let store = hashed_blobs::Store::open(w.archive.root.clone(), vec![], 0o600).unwrap();
    let mut first = batch("i", 0, 0, vec![], &[]);
    first.new_blob = Some(NewBlob::prepare(&store, b"line\npartial").unwrap());
    first.length = 12;
    let parse = first.parse.first_mut().unwrap();
    parse.parsed_upto = 5;
    parse.at_end = true;
    w.writer.send(first).unwrap();
    w.writer.flush().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    assert!(pool.reader().unwrap().known_inputs("m").unwrap()[0].waiting);

    let mut append = batch("i", 0, 12, vec![], &[]);
    append.new_blob = Some(NewBlob::prepare(&store, b"rest").unwrap());
    append.length = 16;
    append.parse.clear();
    w.writer.send(append).unwrap();
    w.writer.flush().unwrap();
    let known = pool.reader().unwrap().known_inputs("m").unwrap();
    assert_eq!(known[0].stored, 16);
    assert_eq!(known[0].parsed_upto, 5);
    assert!(
        !known[0].waiting,
        "the end-of-file check covered only twelve bytes"
    );
}

#[test]
fn a_copy_in_a_spawned_session_never_takes_the_message_from_its_origin() {
    // A compaction call repeats its parent's message; whichever file is read
    // first, the message stays in the parent with the parent's record.
    for main_first in [true, false] {
        let w = test_archive();
        let original = text_message("h1", "s", Role::User, "original", None);
        let copy = text_message("h1", "agent-c", Role::User, "copy", None);
        let main = batch("main", 0, 0, vec![row(0, original)], &["s"]);
        let mut spawned = batch("spawned", 0, 0, vec![row(0, copy)], &[]);
        if let Some(p) = spawned.parse.first_mut() {
            p.output.sessions = vec![Session::new(sid("agent-c"), SessionKind::Compaction)];
        }
        let (first, second) = if main_first {
            (main, spawned)
        } else {
            (spawned, main)
        };
        w.writer.send(first).unwrap();
        w.writer.send(second).unwrap();
        let report = w.writer.flush().unwrap();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(texts(&w, "s"), ["original"], "main first: {main_first}");
        assert!(texts(&w, "agent-c").is_empty(), "main first: {main_first}");
    }
}

#[test]
fn a_release_runs_once_when_its_batch_is_dropped() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    let released = Arc::new(AtomicU64::new(0));
    let counter = Arc::clone(&released);
    let mut b = batch("a", 0, 0, vec![], &[]);
    b.release = InFlight::new(move || {
        counter.fetch_add(7, Ordering::SeqCst);
    });
    let copy = b.clone();
    drop(copy);
    assert_eq!(released.load(Ordering::SeqCst), 0, "a clone holds nothing");
    // The writer keeps a batch until its transaction commits, then drops it.
    let w = test_archive();
    w.writer.send(b).unwrap();
    w.writer.flush().unwrap();
    assert_eq!(released.load(Ordering::SeqCst), 7);
}

fn session_in(id: &str, cwd: &str, repo: Option<&str>) -> Session {
    let mut s = Session::new(sid(id), SessionKind::Main);
    s.offer_cwd(cwd);
    s.repo = repo.map(Into::into);
    s
}

fn lookup_batch(input: &str, path: &str, remote: Option<&str>, root: &str) -> Batch {
    let mut b = batch(input, 0, 0, vec![], &[]);
    b.input.format = "polylogue-repo-lookup".into();
    let p = b.parse.first_mut().unwrap();
    p.output.repo_lookups.push(polylogue_model::RepoLookup {
        path: path.into(),
        remote: remote.map(Into::into),
        root_commit: Some(root.into()),
    });
    p.counts = counts(1);
    b
}

fn repo_of(w: &TestArchive, session: &str) -> (Option<String>, Option<String>) {
    // Repository attribution is derived.
    w.writer.derive().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let rows = pool
        .reader()
        .unwrap()
        .query(
            "SELECT repo, repo_root FROM session WHERE id = ?1",
            &[crate::SqlValue::Text(sid(session).to_string())],
        )
        .unwrap();
    let s = |k: &str| rows[0][k].as_str().map(str::to_owned);
    (s("repo"), s("repo_root"))
}

#[test]
fn sessions_take_the_repository_of_the_nearest_lookup_above_their_cwd() {
    let w = test_archive();
    let mut b = batch("i1", 0, 0, vec![], &[]);
    let out = &mut b.parse.first_mut().unwrap().output;
    out.sessions
        .push(session_in("deep", "/work/outer/inner/src", None));
    out.sessions
        .push(session_in("outer", "/work/outer/docs", None));
    out.sessions
        .push(session_in("near", "/work/outer-sibling", None));
    out.sessions
        .push(session_in("own", "/work/outer", Some("git@host:o/own.git")));
    w.writer.send(b).unwrap();
    // The lookup arrives after its sessions: the writer joins them then.
    w.writer
        .send(lookup_batch(
            "l1",
            "/work/outer",
            Some("https://host/o/outer.git"),
            "r1",
        ))
        .unwrap();
    w.writer.flush().unwrap();
    assert_eq!(
        repo_of(&w, "deep"),
        (Some("https://host/o/outer.git".into()), Some("r1".into()))
    );
    // A path prefix that is not a whole component is another directory.
    assert_eq!(repo_of(&w, "near"), (None, None));
    // The session's own record names its repository; the root still joins.
    assert_eq!(
        repo_of(&w, "own"),
        (Some("git@host:o/own.git".into()), Some("r1".into()))
    );
    // A nested repository is nearer than the outer one.
    w.writer
        .send(lookup_batch("l2", "/work/outer/inner", None, "r2"))
        .unwrap();
    w.writer.flush().unwrap();
    // It has no remote, so the outer remote no longer describes the session.
    assert_eq!(repo_of(&w, "deep"), (None, Some("r2".into())));
    assert_eq!(
        repo_of(&w, "outer"),
        (Some("https://host/o/outer.git".into()), Some("r1".into()))
    );
    // A session written after its lookup joins as it is written.
    let mut b = batch("i2", 0, 0, vec![], &[]);
    let out = &mut b.parse.first_mut().unwrap().output;
    out.sessions.push(session_in("late", "/work/outer/x", None));
    w.writer.send(b).unwrap();
    w.writer.flush().unwrap();
    assert_eq!(
        repo_of(&w, "late"),
        (Some("https://host/o/outer.git".into()), Some("r1".into()))
    );
}

fn call_message(id: &str, call: &str) -> Message {
    let mut m = text_message(id, "s", Role::Assistant, "", None);
    m.blocks = vec![Block::ToolCall {
        run_by_script: false,
        call_id: Some(call.into()),
        tool: Some("Bash".into()),
        action: polylogue_model::ToolAction::Shell,
        input: Json::Null,
        command: Some("cargo test".into()),
        path: None,
    }];
    m
}

fn reasons(w: &TestArchive) -> Vec<(String, String)> {
    // Outcome reasons of calls left behind are derived.
    w.writer.derive().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    pool.reader()
        .unwrap()
        .query(
            "SELECT call_id, outcome_reason FROM tool_call ORDER BY call_id",
            &[],
        )
        .unwrap()
        .iter()
        .map(|r| {
            (
                r["call_id"].as_str().unwrap().to_owned(),
                r["outcome_reason"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

#[test]
fn a_call_the_session_went_past_without_a_result_will_never_get_one() {
    let w = test_archive();
    w.writer
        .send(batch(
            "i1",
            0,
            0,
            vec![row(10, call_message("a", "c1"))],
            &["s"],
        ))
        .unwrap();
    w.writer.flush().unwrap();
    // The last response's call may still be answered.
    assert_eq!(reasons(&w), [("c1".into(), "no_result".into())]);
    let later = vec![row(20, call_message("b", "c2"))];
    w.writer.send(batch("i1", 0, 100, later, &["s"])).unwrap();
    w.writer.flush().unwrap();
    assert_eq!(
        reasons(&w),
        [
            ("c1".into(), "passed_without_result".into()),
            ("c2".into(), "no_result".into())
        ]
    );
}

#[test]
fn dropping_the_writer_drains_accepted_work_with_sender_clones_alive() {
    let TestArchive {
        _dir,
        archive,
        _lock,
        writer,
    } = test_archive();
    let sender = writer.sender().unwrap();
    sender
        .send(batch(
            "shutdown",
            0,
            0,
            vec![row(
                0,
                text_message("kept", "s", Role::User, "durable before stop", None),
            )],
            &["s"],
        ))
        .unwrap();
    let (reply, received) = sync_channel(1);
    sender
        .0
        .send_user((
            Box::new(crate::UserRequest::now(
                crate::user_writes::new_request_id(),
                crate::UserWrite::SetSetting {
                    key: "shutdown-test".into(),
                    value: "stored".into(),
                    author: polylogue_model::Author::Operator,
                    machine: None,
                },
            )),
            reply,
        ))
        .unwrap();
    let (done, stopped) = sync_channel(1);
    let thread = std::thread::spawn(move || {
        drop(writer);
        done.send(()).unwrap();
    });
    stopped
        .recv()
        .expect("writer drop waited for stored senders");
    thread.join().unwrap();
    assert!(received.recv().unwrap().is_ok());
    assert!(matches!(sender.derive(), Err(StoreError::WriterGone)));
    let reads = ReadPool::open(&archive).unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(
        reader.messages(&sid("s"), false).unwrap()[0].text(),
        "durable before stop"
    );
    assert_eq!(
        reader.setting("shutdown-test", "").unwrap().as_deref(),
        Some("stored")
    );
}

#[test]
fn source_arrival_order_does_not_change_message_order_or_derived_title() {
    let view = |inputs: [&str; 2]| {
        let w = test_archive();
        for input in inputs {
            let rows = if input == "input-a" {
                vec![
                    row(
                        0,
                        text_message("earlier", "s1", Role::User, "First question", None),
                    ),
                    row(
                        10,
                        text_message("shared", "s1", Role::User, "Shared question", None),
                    ),
                ]
            } else {
                vec![
                    row(
                        0,
                        text_message("shared", "s1", Role::User, "Shared question", None),
                    ),
                    row(
                        10,
                        text_message("later", "s1", Role::User, "Later question", None),
                    ),
                ]
            };
            w.writer.send(batch(input, 0, 0, rows, &["s1"])).unwrap();
            w.writer.flush().unwrap();
        }
        w.writer.derive().unwrap();
        let reads = ReadPool::open(&w.archive).unwrap();
        let r = reads.reader().unwrap();
        let messages = r
            .messages(&sid("s1"), false)
            .unwrap()
            .into_iter()
            .map(|m| m.id.to_string())
            .collect::<Vec<_>>();
        let title = r
            .query(
                "SELECT derived_title FROM session WHERE id = 'claude-code:s1'",
                &[],
            )
            .unwrap();
        (messages, title)
    };
    let forward = view(["input-a", "input-z"]);
    let reversed = view(["input-z", "input-a"]);
    assert_eq!(forward, reversed);
    assert_eq!(
        forward.0,
        [
            "claude-code:earlier",
            "claude-code:shared",
            "claude-code:later"
        ]
    );
    assert_eq!(forward.1[0]["derived_title"], "First question");
}

mod restating;

mod array_order;

mod record_order;

mod metadata;

mod statement_cache;

#[cfg(test)]
mod prepared;

mod observation_order;

mod excluded;
mod text_projection;

mod failed_inputs;

#[test]
fn first_build_stays_in_wal_and_finishes_while_a_read_stays_open() {
    let w = test_archive();
    let reader_pool = ReadPool::open(&w.archive).unwrap();
    let reader = reader_pool.reader().unwrap();
    assert_eq!(
        reader.query("PRAGMA main.journal_mode", &[]).unwrap()[0]["journal_mode"],
        "wal"
    );
    assert_eq!(
        reader
            .query("SELECT value FROM meta WHERE key='unsynced'", &[])
            .unwrap()[0]["value"],
        "1"
    );
    // This open read predates both the write and the durable build
    // completion. Neither commit may require it to be dropped.
    w.writer
        .send(batch(
            "first",
            0,
            0,
            vec![MessageRow {
                offset: 0,
                end: 100,
                message: text_message("m1", "a", Role::User, "synthetic first build", None),
            }],
            &["a"],
        ))
        .unwrap();
    w.writer.derive().unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    let current = reader_pool.reader().unwrap();
    assert_eq!(
        current
            .query("SELECT value FROM meta WHERE key='unsynced'", &[])
            .unwrap()[0]["value"],
        "0"
    );
    assert_eq!(current.status().unwrap().messages, 1);
    assert_eq!(
        reader
            .query("SELECT value FROM meta WHERE key='unsynced'", &[])
            .unwrap()[0]["value"],
        "1"
    );
    assert_eq!(reader.status().unwrap().messages, 0);
    drop(current);
    drop(w.writer);
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
}

mod authorship;

mod hash_order;

mod documents;

#[test]
fn a_user_write_goes_ahead_of_queued_batches() {
    let w = test_archive();
    let sender = w.writer.sender().unwrap();
    // The writer stops inside batch `a` (its release runs after `a`
    // commits) until the gate opens.
    let (entered, inside) = sync_channel(1);
    let (open, gate) = sync_channel::<()>(1);
    let mut a = batch("a", 0, 0, vec![], &[]);
    a.release = InFlight::new(move || {
        entered.send(()).unwrap();
        gate.recv().unwrap();
    });
    sender.send(a).unwrap();
    inside.recv().unwrap();
    // Batches fill the queue behind it; one more sender waits for room.
    let queued = std::thread::spawn({
        let sender = sender.clone();
        move || {
            for name in ["b", "c", "d", "e", "f"] {
                sender.send(batch(name, 0, 0, vec![], &[])).unwrap();
            }
        }
    });
    std::thread::sleep(std::time::Duration::from_millis(100));
    let user = std::thread::spawn({
        let sender = sender.clone();
        move || {
            sender.user(crate::UserRequest::now(
                crate::user_writes::new_request_id(),
                crate::UserWrite::SetSetting {
                    key: "ahead".into(),
                    value: "yes".into(),
                    author: polylogue_model::Author::Operator,
                    machine: None,
                },
            ))
        }
    });
    std::thread::sleep(std::time::Duration::from_millis(100));
    open.send(()).unwrap();
    assert!(user.join().unwrap().is_ok());
    queued.join().unwrap();
    w.writer.flush().unwrap();
    let reads = ReadPool::open(&w.archive).unwrap();
    let reader = reads.reader().unwrap();
    let rows = reader
        .query(
            "SELECT (SELECT commit_seq FROM a.setting WHERE key = 'ahead') AS setting,
                    (SELECT min(commit_seq) FROM a.observation WHERE path IN ('/p/b', '/p/c', '/p/d', '/p/e', '/p/f')) AS batches",
            &[],
        )
        .unwrap();
    let (setting, batches) = (
        rows[0]["setting"].as_i64().unwrap(),
        rows[0]["batches"].as_i64().unwrap(),
    );
    assert!(
        setting < batches,
        "the setting committed at {setting}, the queued batches from {batches}"
    );
}

mod attachments;
mod reparse;
#[test]
fn an_excision_does_not_finish_unrelated_first_build_work() {
    let w = test_archive();
    w.writer
        .send(batch(
            "removed-input",
            0,
            0,
            vec![MessageRow {
                offset: 0,
                end: 100,
                message: text_message(
                    "removed-message",
                    "removed",
                    Role::User,
                    "removed text",
                    None,
                ),
            }],
            &["removed"],
        ))
        .unwrap();
    w.writer
        .send(batch(
            "kept-input",
            0,
            0,
            vec![MessageRow {
                offset: 0,
                end: 100,
                message: text_message("kept-message", "kept", Role::User, "kept text", None),
            }],
            &["kept"],
        ))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let store = Arc::new(w.archive.blobs().unwrap());
    w.writer
        .excise(
            crate::excise::ExciseRequest {
                id: "remove-during-build".into(),
                target: crate::excise::ExciseTarget::Session(sid("removed")),
                reason: None,
                dry_run: false,
            },
            store,
        )
        .unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT value FROM meta WHERE key='unsynced'", &[])
            .unwrap()[0]["value"],
        "1"
    );
    assert_eq!(reader.status().unwrap().messages, 1);
    drop(reader);
    w.writer.derive().unwrap();
    assert_eq!(pool.reader().unwrap().status().unwrap().messages, 1);
}

mod segments;

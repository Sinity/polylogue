//! The loop over files a test writes: first builds, appends, rewrites,
//! chunked reads, imports, excision and export round trips.

use polylogue_engine::{Engine, ExciseRequest, ExciseTarget, Excised};
use polylogue_model::{Message, SessionId};
use polylogue_store::{Archive, ReadPool};
use polylogue_testkit::{
    ClaudeHome, claude_assistant as assistant, claude_user as user, delete_index,
};
use std::fs;
use std::path::Path;

mod excision_selected;
mod export_round_trip;
mod large_sessions;
mod live_first;
mod older_chains;
mod tool_captures;

#[test]
fn a_first_build_stores_and_parses_and_a_second_pass_reads_nothing() {
    let w = ClaudeHome::new();
    w.append(
        "s1.jsonl",
        &(user("u1", "s1", "hello") + &assistant("a1", "s1", "hi")),
    );
    let (mut e, _lock) = w.engine(64 << 20);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(pass.inputs_changed, 1);
    assert_eq!(w.texts("s1"), ["hello", "hi"]);
    let again = e.run_pass().unwrap();
    assert_eq!(again.inputs_changed, 0);
    assert_eq!(again.bytes_read, 0);
}

#[test]
fn an_unchanged_incomplete_tail_waits_without_rereading_stored_bytes() {
    let w = ClaudeHome::new();
    let partial = user("u2", "s1", "still being written");
    w.append("s1.jsonl", &(user("u1", "s1", "complete") + &partial[..20]));
    {
        let (mut e, _lock) = w.engine(64 << 20);
        e.run_pass().unwrap();
        assert_eq!(w.texts("s1"), ["complete"]);
        let quiet = e.run_pass().unwrap();
        assert_eq!(quiet.bytes_read, 0, "unchanged pending bytes are not work");
    }
    let (mut e, _lock) = w.engine(64 << 20);
    let quiet = e.run_pass().unwrap();
    assert_eq!(
        quiet.bytes_read, 0,
        "restart preserves the pending-tail wait"
    );
    w.append("s1.jsonl", &partial[20..]);
    let grown = e.run_pass().unwrap();
    assert!(grown.errors.is_empty(), "{:?}", grown.errors);
    assert_eq!(w.texts("s1"), ["complete", "still being written"]);
    let whole = ClaudeHome::new();
    whole.append("s1.jsonl", &(user("u1", "s1", "complete") + &partial));
    let (mut engine, _lock) = whole.engine(64 << 20);
    engine.run_pass().unwrap();
    assert_eq!(w.export("claude-code:s1"), whole.export("claude-code:s1"));
}

#[test]
fn waiting_tail_retries_after_the_source_changes() {
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &(user("u1", "s1", "before") + "{\"type\":"));
    let (mut e, _lock) = w.engine(64 << 20);
    e.run_pass().unwrap();
    assert_eq!(e.run_pass().unwrap().bytes_read, 0);
    fs::write(w.file("s1.jsonl"), user("u1", "s1", "rewritten")).unwrap();
    e.run_pass().unwrap();
    assert_eq!(w.texts("s1"), ["rewritten"]);
}

#[test]
fn appends_cost_only_their_bytes_and_a_partial_line_waits() {
    let w = ClaudeHome::new();
    let first = user("u1", "s1", "one");
    w.append("s1.jsonl", &first);
    let (mut e, _lock) = w.engine(64 << 20);
    e.run_pass().unwrap();
    let second = user("u2", "s1", "two");
    let (head, rest) = second.split_at(10);
    w.append("s1.jsonl", head);
    let pass = e.run_pass().unwrap();
    assert_eq!(pass.bytes_read, 10, "only the appended bytes are read");
    assert_eq!(w.texts("s1"), ["one"]);
    w.append("s1.jsonl", rest);
    let pass = e.run_pass().unwrap();
    assert_eq!(
        pass.bytes_read,
        second.len() as u64,
        "the waiting line and its rest"
    );
    assert_eq!(w.texts("s1"), ["one", "two"]);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.revision"), 3);
    assert_eq!(
        w.count("SELECT SUM(length) AS n FROM a.extent"),
        (first.len() + second.len()) as i64
    );
}

#[test]
fn after_a_restart_state_is_rebuilt_only_when_the_input_changes() {
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &user("u1", "s1", "one"));
    w.append("s2.jsonl", &user("v1", "s2", "other"));
    {
        let (mut e, _lock) = w.engine(64 << 20);
        e.run_pass().unwrap();
    }
    let (mut e, _lock) = w.engine(64 << 20);
    let quiet = e.run_pass().unwrap();
    assert_eq!(quiet.bytes_read, 0, "nothing is re-parsed at start-up");
    let line = user("u2", "s1", "two");
    w.append("s1.jsonl", &line);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(
        pass.bytes_read > line.len() as u64,
        "s1 was parsed again from the start"
    );
    assert_eq!(w.texts("s1"), ["one", "two"]);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM message"), 3);
}

#[test]
fn a_rewritten_file_starts_a_new_chain_and_its_content_wins() {
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &user("u1", "s1", "draft"));
    let (mut e, _lock) = w.engine(64 << 20);
    e.run_pass().unwrap();
    fs::write(w.file("s1.jsonl"), user("u1", "s1", "final text, longer")).unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(w.texts("s1"), ["final text, longer"]);
    assert_eq!(
        w.count("SELECT COUNT(DISTINCT chain) AS n FROM a.revision"),
        2
    );
}

#[test]
fn large_files_go_in_chunks_and_a_record_longer_than_a_chunk_still_parses() {
    let w = ClaudeHome::new();
    let mut text = String::new();
    for i in 0..40 {
        text += &user(&format!("u{i}"), "s1", &format!("message {i}"));
    }
    text += &user("big", "s1", &"x".repeat(2000));
    w.append("s1.jsonl", &text);
    let (mut e, _lock) = w.engine(300);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let got = w.texts("s1");
    assert_eq!(got.len(), 41);
    assert_eq!(got[40].len(), 2000);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.revision"), 1);
    assert_eq!(
        w.count("SELECT SUM(length) AS n FROM a.extent"),
        text.len() as i64
    );
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM parse_state WHERE status = 'done'"),
        1
    );
}

#[test]
fn a_vanished_file_is_observed_gone_and_its_messages_stay() {
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &user("u1", "s1", "kept"));
    let (mut e, _lock) = w.engine(64 << 20);
    e.run_pass().unwrap();
    fs::remove_file(w.file("s1.jsonl")).unwrap();
    e.run_pass().unwrap();
    assert_eq!(w.texts("s1"), ["kept"]);
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM a.observation WHERE class = 'deleted'"),
        1
    );
}

#[test]
fn a_deleted_index_is_built_again_from_stored_bytes_even_when_the_files_are_gone() {
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &user("u1", "s1", "first"));
    {
        let (mut e, _lock) = w.engine(64 << 20);
        e.run_pass().unwrap();
        // An append stored as its own new blob, so the chain has two extents.
        w.append("s1.jsonl", &assistant("a1", "s1", "second"));
        e.run_pass().unwrap();
    }
    fs::remove_file(w.file("s1.jsonl")).unwrap();
    delete_index(&w.root);
    let (mut e, _lock) = w.engine(64 << 20);
    e.run_pass().unwrap();
    assert_eq!(w.texts("s1"), ["first", "second"]);
}

#[test]
fn a_stored_input_is_parsed_from_its_stored_bytes_in_chunks() {
    let w = ClaudeHome::new();
    for i in 0..20 {
        w.append("s2.jsonl", &user(&format!("u{i}"), "s2", &format!("m{i}")));
    }
    {
        let (mut e, _lock) = w.engine(64 << 20);
        e.run_pass().unwrap();
    }
    fs::remove_file(w.file("s2.jsonl")).unwrap();
    delete_index(&w.root);
    // A chunk far smaller than the file: the stored bytes are read by range.
    let (mut e, _lock) = w.engine(300);
    e.run_pass().unwrap();
    assert_eq!(w.texts("s2").len(), 20);
}

#[test]
fn importing_the_same_bytes_twice_stores_them_once() {
    let w = ClaudeHome::new();
    let outside = w.dir.path().join("export/s9.jsonl");
    fs::create_dir_all(outside.parent().unwrap()).unwrap();
    fs::write(&outside, user("x1", "s9", "imported")).unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    let first = e.import(std::slice::from_ref(&outside)).unwrap();
    assert_eq!(first.imported, std::slice::from_ref(&outside));
    assert_eq!(w.texts("s9"), ["imported"]);
    let second = e.import(std::slice::from_ref(&outside)).unwrap();
    assert_eq!(second.already_stored, [outside]);
}

#[test]
fn importing_a_directory_imports_every_file_under_it() {
    let w = ClaudeHome::new();
    let spool = w.dir.path().join("old-spool");
    fs::create_dir_all(spool.join("nested")).unwrap();
    fs::write(spool.join("a.jsonl"), user("d1", "sa", "first")).unwrap();
    fs::write(spool.join("nested/b.jsonl"), user("d2", "sb", "second")).unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    let report = e.import(std::slice::from_ref(&spool)).unwrap();
    assert_eq!(
        report.imported,
        [spool.join("a.jsonl"), spool.join("nested/b.jsonl")]
    );
    assert_eq!(w.texts("sa"), ["first"]);
    assert_eq!(w.texts("sb"), ["second"]);
}

fn messages_of(root: &Path, session: &str) -> Vec<Message> {
    let pool = ReadPool::open(&Archive::at(root)).unwrap();
    let id = SessionId::parse(session).unwrap();
    pool.reader().unwrap().messages(&id, false).unwrap()
}

fn chatgpt_conversation(id: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id, "conversation_id": id, "title": "T", "create_time": 1700000000.0,
        "current_node": "n1",
        "mapping": {
            "n1": {"id": "n1", "parent": null, "children": [], "message": {
                "id": format!("{id}-m1"), "author": {"role": "user"}, "create_time": 1700000000.0,
                "content": {"content_type": "multimodal_text", "parts": [
                    {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-Img"},
                    text
                ]},
                "status": "finished_successfully", "recipient": "all", "metadata": {}
            }}
        }
    })
}

#[test]
fn a_zip_export_imports_its_conversations_and_keeps_its_other_files() {
    use std::io::Write as _;
    let w = ClaudeHome::new();
    let zip_path = w.dir.path().join("chatgpt-data.zip");
    let mut zip = zip::ZipWriter::new(fs::File::create(&zip_path).unwrap());
    let opts = zip::write::SimpleFileOptions::default();
    zip.start_file("conversations-000.json", opts).unwrap();
    let convs = serde_json::json!([chatgpt_conversation("c1", "look at this")]);
    zip.write_all(&serde_json::to_vec(&convs).unwrap()).unwrap();
    zip.start_file("file-Img-photo.png", opts).unwrap();
    zip.write_all(b"\x89PNG not really").unwrap();
    zip.finish().unwrap();

    let (mut e, _lock) = w.engine(64 << 20);
    let report = e.import(std::slice::from_ref(&zip_path)).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(report.imported.len(), 3, "{:?}", report.imported);
    assert!(report.imported.contains(&zip_path));
    let msgs = messages_of(&w.root, "chatgpt:c1");
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].text(), "look at this");
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM a.input WHERE format = 'export-file'"),
        2,
        "the original ZIP and image are kept as bytes"
    );
    let again = e.import(std::slice::from_ref(&zip_path)).unwrap();
    assert_eq!(again.already_stored.len(), 3);
    assert!(again.imported.is_empty());
}

#[test]
fn array_conversations_parse_when_the_chunk_is_smaller_than_one_record() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    let convs = serde_json::json!([
        chatgpt_conversation("c1", "one"),
        chatgpt_conversation("c2", "two")
    ]);
    fs::write(&path, serde_json::to_vec_pretty(&convs).unwrap()).unwrap();
    // The array path stores one conversation independently of the line chunk.
    let (mut e, _lock) = w.engine(64);
    let report = e.import(std::slice::from_ref(&path)).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(messages_of(&w.root, "chatgpt:c1")[0].text(), "one");
    assert_eq!(messages_of(&w.root, "chatgpt:c2")[0].text(), "two");
}

fn capture_file(texts: &[&str]) -> Vec<u8> {
    let turns: Vec<serde_json::Value> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| serde_json::json!({"provider_turn_id": format!("t{i}"), "role": "user", "text": t, "ordinal": i}))
        .collect();
    serde_json::to_vec_pretty(&serde_json::json!({
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "source": "browser-extension",
        "provenance": {"captured_at": "2026-02-01T00:00:00Z"},
        "provider_meta": {},
        "raw_provider_payload": {},
        "session": {"provider": "grok", "provider_session_id": "g1", "turns": turns}
    }))
    .unwrap()
}

#[test]
fn a_capture_rewritten_in_place_is_parsed_again_whole() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".local/state/polylogue/browser-capture/grok");
    fs::create_dir_all(&dir).unwrap();
    let file = dir.join("g1.json");
    fs::write(&file, capture_file(&["one"])).unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let texts = |root: &Path| -> Vec<String> {
        messages_of(root, "grok:g1")
            .iter()
            .map(Message::text)
            .collect()
    };
    assert_eq!(texts(&w.root), ["one"]);
    // The extension rewrites the whole file when it captures again.
    fs::write(&file, capture_file(&["one", "two"])).unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(texts(&w.root), ["one", "two"]);
}

fn excise(e: &mut Engine, target: &str, dry_run: bool) -> Excised {
    e.excise(ExciseRequest {
        id: format!("req-{target}-{dry_run}"),
        target: ExciseTarget::parse(target).unwrap(),
        reason: Some("test".into()),
        dry_run,
    })
    .unwrap()
}

/// Whether any stored input byte range holds `needle`.
fn stored(root: &Path, needle: &str) -> bool {
    !polylogue_store::fixture::text_search::stored_blobs_containing(&Archive::at(root), needle)
        .unwrap()
        .is_empty()
}

#[test]
fn an_excised_session_is_gone_from_every_file_and_never_comes_back() {
    let w = ClaudeHome::new();
    w.append(
        "s1.jsonl",
        &(user("u1", "s1", "the launch code is zebra42") + &assistant("a1", "s1", "noted")),
    );
    w.append(
        "s2.jsonl",
        &(user("u2", "s2", "unrelated") + &assistant("a2", "s2", "fine")),
    );
    let (mut e, lock) = w.engine(64 << 20);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert!(stored(&w.root, "zebra42"));

    let preview = excise(&mut e, "session:claude-code:s1", true);
    assert!(preview.dry_run);
    assert_eq!(preview.messages, 2);
    assert_eq!(w.texts("s1"), ["the launch code is zebra42", "noted"]);

    let done = excise(&mut e, "session:claude-code:s1", false);
    assert_eq!(done.sessions, ["claude-code:s1"]);
    assert_eq!(done.inputs_removed.len(), 1);
    assert!(done.blobs_deleted >= 1, "{done:?}");
    assert!(w.texts("s1").is_empty());
    assert_eq!(w.texts("s2"), ["unrelated", "fine"]);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM session WHERE id = 'claude-code:s1'"),
        0
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM text_fts WHERE text_fts MATCH 'zebra42'"),
        0
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM removed_session WHERE session_id = 'claude-code:s1'"),
        1
    );
    assert!(!stored(&w.root, "zebra42"));

    // The file keeps growing; nothing of it is collected again.
    w.append("s1.jsonl", &user("u3", "s1", "zebra42 again"));
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert!(w.texts("s1").is_empty());
    // Nor after a restart.
    drop(e);
    drop(lock);
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert!(w.texts("s1").is_empty());
    assert!(!stored(&w.root, "zebra42"));
    // Nor by import of the same session from another file.
    let copy = w.dir.path().join("copy.jsonl");
    fs::write(&copy, user("u1", "s1", "the launch code is zebra42")).unwrap();
    e.import(std::slice::from_ref(&copy)).unwrap();
    assert!(w.texts("s1").is_empty());
}

/// Index columns holding `needle`.
fn indexed(root: &Path, needle: &str) -> Vec<String> {
    polylogue_store::fixture::text_search::index_columns_containing(&Archive::at(root), needle)
        .unwrap()
}

/// A session whose secret sits in its title, working directory, branch,
/// prose, a tool command and the tool's output.
fn secret_session(session: &str, secret: &str) -> String {
    let line = |v: serde_json::Value| format!("{v}\n");
    [
        line(serde_json::json!({
            "type": "user", "uuid": "u1", "sessionId": session,
            "timestamp": "2026-03-01T10:00:00Z",
            "cwd": format!("/work/{secret}/proj"), "gitBranch": format!("fix-{secret}"),
            "message": {"role": "user", "content": format!("deploy with {secret} please")}
        })),
        line(serde_json::json!({
            "type": "custom-title", "sessionId": session,
            "customTitle": format!("rotate {secret}")
        })),
        line(serde_json::json!({
            "type": "assistant", "uuid": "a1", "parentUuid": "u1", "sessionId": session,
            "timestamp": "2026-03-01T10:00:01Z",
            "message": {"id": "msg_a1", "role": "assistant", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash",
                             "input": {"command": format!("echo {secret} > key.txt")}}]}
        })),
        line(serde_json::json!({
            "type": "user", "uuid": "u2", "parentUuid": "a1", "sessionId": session,
            "timestamp": "2026-03-01T10:00:02Z",
            "message": {"role": "user", "content": [{"type": "tool_result",
                "tool_use_id": "toolu_1", "content": format!("wrote {secret}")}]}
        })),
    ]
    .concat()
}

#[test]
fn a_removed_session_leaves_no_index_column_or_stored_byte_holding_its_text() {
    let w = ClaudeHome::new();
    let secret = "sk-abc123-synthetic";
    w.append("secret.jsonl", &secret_session("secret", secret));
    w.append("kept.jsonl", &user("k1", "kept", "kept text"));
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert!(!indexed(&w.root, secret).is_empty());
    assert!(stored(&w.root, secret));
    let done = excise(&mut e, "session:claude-code:secret", false);
    assert_eq!(done.inputs_removed.len(), 1, "{done:?}");
    assert_eq!(indexed(&w.root, secret), Vec::<String>::new());
    assert!(!stored(&w.root, secret));
    assert_eq!(w.texts("kept"), ["kept text"]);
}

#[test]
fn excising_a_session_from_a_file_holding_another_session_refuses() {
    let w = ClaudeHome::new();
    // One file holding records of two sessions (a resumed session's file
    // names the session each record belongs to).
    w.append(
        "mix.jsonl",
        &(user("u1", "keep", "keep this")
            + &user("u2", "drop", "drop pelican77")
            + &user("u3", "keep", "and this")),
    );
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.run_pass().unwrap().errors.is_empty());
    let before = (w.export("claude-code:keep"), w.export("claude-code:drop"));
    for dry_run in [true, false] {
        let refused = e
            .excise(ExciseRequest {
                id: format!("drop-{dry_run}"),
                target: ExciseTarget::parse("session:claude-code:drop").unwrap(),
                reason: None,
                dry_run,
            })
            .unwrap_err();
        assert!(
            refused
                .to_string()
                .contains("also holds session claude-code:keep"),
            "{refused}"
        );
        assert_eq!(
            (w.export("claude-code:keep"), w.export("claude-code:drop")),
            before
        );
        assert!(stored(&w.root, "pelican77"));
        assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 0);
    }
}

#[test]
fn every_origin_s_shell_and_file_calls_reach_failing_commands_and_files() {
    let w = ClaudeHome::new();
    let chats = w.home.join(".gemini/tmp/proj/chats");
    fs::create_dir_all(&chats).unwrap();
    let call = |id: &str, name: &str, args: serde_json::Value, status: &str| {
        serde_json::json!({"id": id, "name": name, "args": args, "status": status,
            "result": [{"functionResponse": {"id": id, "name": name,
                                             "response": {"output": "out"}}}]})
    };
    let gemini = serde_json::json!({
    "startTime": "2026-03-14T18:01:43Z", "directories": ["/work/g"],
    "messages": [
        {"id": "u1", "timestamp": "2026-03-14T18:01:43Z", "type": "user",
         "content": [{"text": "build it"}]},
        {"id": "m1", "timestamp": "2026-03-14T18:01:50Z", "type": "gemini", "content": "ok",
         "toolCalls": [
             call("c1", "run_shell_command", serde_json::json!({"command": "cargo build"}), "error"),
             call("c2", "read_file", serde_json::json!({"absolute_path": "/work/g/src/main.rs"}), "success"),
         ]},
    ]});
    // The file is recognized by its first key.
    let gemini = gemini.to_string().replacen('{', r#"{"sessionId":"g1","#, 1);
    fs::write(chats.join("session-g1.json"), gemini).unwrap();

    let rollouts = w.home.join(".codex/sessions/2026/03/01");
    fs::create_dir_all(&rollouts).unwrap();
    let thread = "019a0000-0000-7000-8000-000000000001";
    let env = |n: u64, ty: &str, payload: serde_json::Value| {
        serde_json::json!({"timestamp": format!("2026-03-01T10:00:{n:02}.000Z"),
                           "type": ty, "payload": payload})
        .to_string()
            + "\n"
    };
    let codex = env(
        1,
        "session_meta",
        serde_json::json!({"id": thread, "timestamp": "2026-03-01T10:00:00.000Z", "cwd": "/work/c"}),
    ) + &env(
        2,
        "response_item",
        serde_json::json!({"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "test it"}]}),
    ) + &env(
        3,
        "response_item",
        serde_json::json!({"type": "function_call",
            "name": "shell_command", "arguments": "{\"command\":\"cargo test\"}", "call_id": "k1"}),
    ) + &env(
        4,
        "response_item",
        serde_json::json!({"type": "function_call_output",
            "call_id": "k1", "output": "Exit code: 1\nWall time: 1 seconds\nOutput:\nfailed"}),
    );
    fs::write(
        rollouts.join(format!("rollout-2026-03-01T10-00-00-{thread}.jsonl")),
        codex,
    )
    .unwrap();

    let (mut e, _lock) = w.engine(64 << 20);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);

    let reads = w.reads();
    let reader = reads.reader().unwrap();
    let calls = reader
        .query(
            "SELECT tool, action, command_norm, path FROM tool_call ORDER BY tool",
            &[],
        )
        .unwrap();
    let row = |tool: &str| {
        calls
            .iter()
            .find(|r| r["tool"] == tool)
            .unwrap_or_else(|| panic!("no {tool} call in {calls:?}"))
            .clone()
    };
    for tool in ["run_shell_command", "shell_command"] {
        assert_eq!(row(tool)["action"], "shell", "{tool}");
        assert!(row(tool)["command_norm"].is_string(), "{tool}: {calls:?}");
    }
    // The files unit counts calls whose action is a file action.
    assert_eq!(row("read_file")["action"], "file_read");
    assert_eq!(row("read_file")["path"], "/work/g/src/main.rs");

    let failing = polylogue_store::measure::tool_calls::failing_commands(
        &reader,
        &polylogue_store::measure::Selection::default(),
    )
    .unwrap();
    let repositories: Vec<&str> = failing.iter().map(|f| f.repository.as_str()).collect();
    assert!(repositories.contains(&"cwd:/work/g"), "{repositories:?}");
    assert!(repositories.contains(&"cwd:/work/c"), "{repositories:?}");
}

#[test]
fn rebuilding_multiple_inputs_without_provider_files_preserves_order_and_title() {
    let w = ClaudeHome::new();
    w.append(
        "a.jsonl",
        &(user("first", "s1", "First question") + &user("shared", "s1", "Shared question")),
    );
    w.append(
        "z.jsonl",
        &(user("shared", "s1", "Shared question") + &user("last", "s1", "Last question")),
    );
    let view = || {
        let reads = w.reads();
        let r = reads.reader().unwrap();
        let title = r
            .query(
                "SELECT derived_title FROM session WHERE id = 'claude-code:s1'",
                &[],
            )
            .unwrap();
        (w.texts("s1"), title)
    };
    {
        let (mut engine, _lock) = w.engine(64);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    let before = view();
    fs::remove_file(w.file("a.jsonl")).unwrap();
    fs::remove_file(w.file("z.jsonl")).unwrap();
    delete_index(&w.root);
    {
        let (mut engine, _lock) = w.engine(64);
        let rebuilt = engine.run_pass().unwrap();
        assert!(rebuilt.errors.is_empty(), "{:?}", rebuilt.errors);
        assert!(rebuilt.bytes_read > 0);
    }
    assert_eq!(view(), before);
}

mod excision_exports;

mod otel_exports;

mod gemini_takeout;

mod excision_recovery;

mod empty_inputs;

mod hermes_state;

mod antigravity;

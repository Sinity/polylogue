//! The loop over files a test writes: first builds, appends, rewrites,
//! chunked reads, imports and export round trips.

use polylogue_engine::{Engine, ImportOptions};
use polylogue_model::{Message, SessionId};
use polylogue_store::{Archive, ReadPool};
use polylogue_testkit::{
    ClaudeHome, claude_assistant as assistant, claude_user as user, delete_index,
};
use std::fs;
use std::path::Path;

mod export_round_trip;
mod hook_events;
mod large_sessions;
mod live_first;
mod reading_version;
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
fn after_a_restart_only_the_changed_input_resumes_its_durable_checkpoint() {
    let w = ClaudeHome::new();
    let first = user("u1", "s1", "one");
    w.append("s1.jsonl", &first);
    w.append("s2.jsonl", &user("v1", "s2", "other"));
    let checkpoint = |engine: &Engine, path: &Path| {
        let pool = ReadPool::open(&engine.archive).unwrap();
        let reader = pool.reader().unwrap();
        let known = reader
            .known_inputs(&engine.config.machine)
            .unwrap()
            .into_iter()
            .find(|known| known.path_bytes.as_deref() == Some(path.as_os_str().as_encoded_bytes()))
            .unwrap();
        let chain = known.chain.unwrap();
        let version = polylogue_sources::origin::by_name("claude-code")
            .unwrap()
            .reading_version();
        let stored = reader
            .parser_checkpoint(&known.input_id, chain, &version)
            .unwrap()
            .unwrap();
        assert!(stored.checkpoint.is_some(), "the continuation is durable");
        (known.input_id, chain, stored)
    };
    let (before_changed, before_unchanged) = {
        let (mut e, _lock) = w.engine(64 << 20);
        let pass = e.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        (
            checkpoint(&e, &w.file("s1.jsonl")),
            checkpoint(&e, &w.file("s2.jsonl")),
        )
    };
    assert_eq!(before_changed.2.parsed_upto, first.len() as u64);
    let (mut e, _lock) = w.engine(64 << 20);
    let quiet = e.run_pass().unwrap();
    assert!(quiet.errors.is_empty(), "{quiet:?}");
    assert_eq!(quiet.bytes_read, 0, "nothing is re-parsed at start-up");
    assert_eq!(quiet.bytes_stored, 0);
    assert_eq!(checkpoint(&e, &w.file("s1.jsonl")), before_changed);
    assert_eq!(checkpoint(&e, &w.file("s2.jsonl")), before_unchanged);
    let old_message = polylogue_model::MessageId::parse("claude-code:u1").unwrap();
    let original_admission = w
        .reads()
        .reader()
        .unwrap()
        .message_admission(&old_message)
        .unwrap()
        .unwrap();
    let line = user("u2", "s1", "two");
    w.append("s1.jsonl", &line);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(
        pass.bytes_read,
        line.len() as u64,
        "only the new parser range is read"
    );
    assert_eq!(pass.bytes_stored, line.len() as u64);
    let after_changed = checkpoint(&e, &w.file("s1.jsonl"));
    assert_eq!(after_changed.0, before_changed.0);
    assert_eq!(after_changed.1, before_changed.1);
    assert_eq!(
        after_changed.2.parsed_upto,
        (first.len() + line.len()) as u64
    );
    assert_eq!(checkpoint(&e, &w.file("s2.jsonl")), before_unchanged);
    assert_eq!(
        w.reads()
            .reader()
            .unwrap()
            .message_admission(&old_message)
            .unwrap(),
        Some(original_admission)
    );
    assert_eq!(w.texts("s1"), ["one", "two"]);
    assert_eq!(w.texts("s2"), ["other"]);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM message"), 3);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.revision"), 3);
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
    let first = e
        .import_with_options(
            std::slice::from_ref(&outside),
            &ImportOptions {
                format: Some("claude-code".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(first.refused.is_empty(), "{first:?}");
    assert!(first.pass.errors.is_empty(), "{first:?}");
    assert_eq!(first.imported, std::slice::from_ref(&outside));
    assert_eq!(w.texts("s9"), ["imported"]);
    let second = e
        .import_with_options(
            std::slice::from_ref(&outside),
            &ImportOptions {
                format: Some("claude-code".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(second.refused.is_empty(), "{second:?}");
    assert!(second.pass.errors.is_empty(), "{second:?}");
    assert_eq!(second.already_stored, [outside]);
}

#[test]
fn import_without_final_newline_reads_last_record() {
    let w = ClaudeHome::new();
    let outside = w.dir.path().join("export/s8.jsonl");
    fs::create_dir_all(outside.parent().unwrap()).unwrap();
    let bytes = user("x1", "s8", "first") + &assistant("x2", "s8", "last");
    fs::write(&outside, bytes.trim_end_matches('\n')).unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    let report = e
        .import_with_options(
            std::slice::from_ref(&outside),
            &ImportOptions {
                format: Some("claude-code".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    assert_eq!(w.texts("s8"), ["first", "last"]);
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM parse_state WHERE status = 'done'"),
        1
    );
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM parse_state"), 1);
}

#[test]
fn growing_live_file_without_newline_still_waits() {
    let w = ClaudeHome::new();
    let bytes = user("u1", "s1", "first") + &assistant("a1", "s1", "last");
    w.append("s1.jsonl", bytes.trim_end_matches('\n'));
    let (mut e, _lock) = w.engine(64 << 20);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(w.texts("s1"), ["first"]);
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM parse_state WHERE status = 'waiting'"),
        1
    );
    w.append("s1.jsonl", "\n");
    e.run_pass().unwrap();
    assert_eq!(w.texts("s1"), ["first", "last"]);
}

#[test]
fn importing_a_directory_imports_every_file_under_it() {
    let w = ClaudeHome::new();
    let spool = w.dir.path().join("old-spool");
    fs::create_dir_all(spool.join("nested")).unwrap();
    fs::write(spool.join("a.jsonl"), user("d1", "sa", "first")).unwrap();
    fs::write(spool.join("nested/b.jsonl"), user("d2", "sb", "second")).unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    let report = e
        .import_with_options(
            std::slice::from_ref(&spool),
            &ImportOptions {
                format: Some("claude-code".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
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

mod otel_exports;

mod gemini_takeout;

mod empty_inputs;

mod hermes_state;

mod antigravity;

//! Inputs whose origin reads them as exports (a Gemini CLI session, a Hermes
//! profile's runtime records), found by the daemon's pass while the pool
//! reads other inputs, are read item by item, never range by range.
use super::*;
use polylogue_model::SessionId;
use serde_json::json;

/// A Gemini CLI session holding `turns` question and answer pairs, as the
/// CLI rewrites it after each turn.
fn gemini_turns(turns: usize) -> Vec<u8> {
    let messages: Vec<_> = (1..=turns)
        .flat_map(|k| {
            [
                json!({"id": format!("u{k}"), "timestamp": "2026-08-01T10:00:01.000Z",
                       "type": "user", "content": format!("question {k}")}),
                json!({"id": format!("g{k}"), "timestamp": "2026-08-01T10:00:02.000Z",
                       "type": "gemini", "content": format!("answer {k}")}),
            ]
        })
        .collect();
    serde_json::to_vec(&json!({
        "sessionId": "routed", "projectHash": "synthetic-project",
        "startTime": "2026-08-01T10:00:00.000Z", "lastUpdated": "2026-08-01T10:00:02.000Z",
        "kind": "chat", "messages": messages
    }))
    .unwrap()
}

fn atof_mark(n: u32) -> String {
    format!(
        "{}\n",
        json!({"atof_version": "0.1", "kind": "mark", "category": "agent",
               "uuid": format!("00000000-0000-4000-8000-00000000000{n}"),
               "timestamp": "2026-01-02T03:04:05Z", "name": format!("synthetic.mark.{n}"),
               "metadata": {"session_id": "run-alpha"}, "data": {"n": n}})
    )
}

fn append(path: &Path, text: &str) {
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
}

/// The changes the archive recorded for the input at `path`, in order.
fn revisions(engine: &Engine, path: &Path) -> Vec<String> {
    let reader = engine.reads.reader().unwrap();
    let known = reader
        .known_input_at("test-machine", path.as_os_str().as_encoded_bytes())
        .unwrap()
        .unwrap();
    reader
        .query(
            "SELECT produced FROM a.revision WHERE input=?1 ORDER BY number",
            &[polylogue_store::SqlValue::Text(known.input_id)],
        )
        .unwrap()
        .iter()
        .map(|row| row["produced"].as_str().unwrap().to_owned())
        .collect()
}

fn expected_texts(n: usize) -> Vec<String> {
    (1..=n)
        .flat_map(|k| [format!("question {k}"), format!("answer {k}")])
        .collect()
}

fn expected_marks(n: u32) -> Vec<String> {
    (1..=n).map(|k| format!("synthetic.mark.{k}")).collect()
}

#[test]
fn gemini_cli_and_hermes_files_are_read_and_so_are_their_changes_found_during_a_pool_run() {
    let home = ClaudeHome::new();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
    engine.config.workers = 1;
    // The pool looks for changed live inputs on every turn, so the Gemini
    // CLI and Hermes changes are found there while it reads the Claude Code
    // file.
    engine.config.live_every = Duration::ZERO;
    let filler = "synthetic filler ".repeat(200);
    let claude_lines = |from: u32, to: u32| {
        (from..to)
            .map(|n| claude_user(&format!("u{n}"), "big", &format!("{n} {filler}")))
            .collect::<String>()
    };
    home.append("big.jsonl", &claude_lines(0, 20));
    let gemini = home
        .home
        .join(".gemini/tmp/h/chats/session-2026-08-01.json");
    std::fs::create_dir_all(gemini.parent().unwrap()).unwrap();
    std::fs::write(&gemini, gemini_turns(1)).unwrap();
    let atof = home
        .home
        .join(".hermes/profiles/research/observability/nemo-relay/atof/events.jsonl");
    std::fs::create_dir_all(atof.parent().unwrap()).unwrap();
    std::fs::write(&atof, atof_mark(1) + &atof_mark(2)).unwrap();

    let session = SessionId::parse("gemini-cli:routed").unwrap();
    let texts = |engine: &Engine| -> Vec<String> {
        engine
            .reads
            .reader()
            .unwrap()
            .messages(&session, false)
            .unwrap()
            .iter()
            .map(polylogue_model::Message::text)
            .collect()
    };
    let marks = |engine: &Engine| -> Vec<String> {
        engine
            .reads
            .reader()
            .unwrap()
            .query(
                "SELECT json_extract((SELECT pl_text_at(g.body,e.content_start+e.content_length,e.data_length) FROM content_group g WHERE g.id=e.content_group),'$.name') AS name FROM event e \
                 WHERE kind='hermes_observer_mark' ORDER BY source_start",
                &[],
            )
            .unwrap()
            .iter()
            .map(|row| row["name"].as_str().unwrap().to_owned())
            .collect()
    };
    let unfinished = |engine: &Engine| {
        engine
            .reads
            .reader()
            .unwrap()
            .query(
                "SELECT count(*) AS n FROM parse_state WHERE NOT (state='parsed' AND waiting=0)",
                &[],
            )
            .unwrap()[0]["n"]
            .as_i64()
            .unwrap()
    };

    let first = engine.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    assert_eq!(texts(&engine), expected_texts(1));
    assert_eq!(marks(&engine), expected_marks(2));

    // The CLI rewrites its session with a new turn and Hermes appends a
    // record while the pool reads the Claude Code file's append: the pool
    // finds both changes itself.
    home.append("big.jsonl", &claude_lines(20, 40));
    std::fs::write(&gemini, gemini_turns(2)).unwrap();
    append(&atof, &atof_mark(3));
    let claude = home.file("big.jsonl");
    let facts = FileFacts::of(&std::fs::metadata(&claude).unwrap());
    let claude_plan = engine.plan(&claude, facts, &mut 0).unwrap().unwrap();
    let mut live = Pass::default();
    engine
        .execute_pool(vec![claude_plan], Vec::new(), &[], &mut live)
        .unwrap();
    assert!(live.errors.is_empty(), "{:?}", live.errors);
    assert_eq!(texts(&engine), expected_texts(2));
    assert_eq!(marks(&engine), expected_marks(3));
    assert_eq!(unfinished(&engine), 0);

    // The same changes planned by a pass are read by its own plans, while
    // the pool reads the Claude Code file first.
    home.append("big.jsonl", &claude_lines(40, 60));
    std::fs::write(&gemini, gemini_turns(3)).unwrap();
    append(&atof, &atof_mark(4));
    let third = engine.run_pass().unwrap();
    assert!(third.errors.is_empty(), "{:?}", third.errors);
    assert_eq!(texts(&engine), expected_texts(3));
    assert_eq!(marks(&engine), expected_marks(4));
    assert_eq!(unfinished(&engine), 0);
    // Each change is stored once, by the plan that reads it.
    assert_eq!(
        revisions(&engine, &gemini),
        ["first", "rewritten", "rewritten"]
    );
    assert_eq!(revisions(&engine, &atof), ["first", "appended", "appended"]);
    assert_eq!(engine.run_pass().unwrap().inputs_changed, 0);
}

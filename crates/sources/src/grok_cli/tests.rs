use super::*;
use crate::parse::run;
use serde_json::json;

fn grok() -> GrokCli {
    GrokCli {
        place: Place::from_path(Path::new(
            "/h/.grok/sessions/%2Fworkspace%2Fprojects%2Fexample/01a0b507/chat_history.jsonl",
        ))
        .unwrap(),
    }
}

fn lines(records: &[Json]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in records {
        out.extend(serde_json::to_vec(r).unwrap());
        out.push(b'\n');
    }
    out
}

fn history() -> Vec<Json> {
    vec![
        json!({"type": "system", "content": "You are a CLI."}),
        json!({"type": "user", "content": [{"type": "text", "text": "fix it"}], "prompt_index": 0}),
        json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "look first"}],
               "encrypted_content": "xx", "status": "completed"}),
        json!({"type": "assistant", "content": "Reading.", "model_id": "grok-4.6",
               "model_fingerprint": "fp", "reasoning_effort": "high",
               "tool_calls": [{"id": "call-1", "name": "read_file",
                               "arguments": "{\"target_file\": \"a.rs\"}"}]}),
        json!({"type": "tool_result", "tool_call_id": "call-1", "content": "fn main() {}"}),
        json!({"type": "user", "content": [{"type": "text", "text": "<reminder>"}],
               "synthetic_reason": "todo_reminder"}),
    ]
}

#[test]
fn reasoning_joins_the_next_answer_and_messages_are_named_by_content() {
    let input = lines(&history());
    let step = run(&grok(), &mut ParserState::default(), &input, 0).unwrap();
    let msgs: Vec<&Message> = step.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(msgs.len(), 5);
    assert_eq!(msgs[0].session.to_string(), "grok-cli:01a0b507");
    assert_eq!(step.output.sessions[0].cwd, ["/workspace/projects/example"]);
    let answer = msgs[2];
    assert!(
        matches!(&answer.blocks[0], Block::Thinking { text: Some(t), .. } if t == "look first")
    );
    match &answer.blocks[2] {
        Block::ToolCall { path, input, .. } => {
            assert_eq!(path.as_deref(), Some("a.rs"));
            assert_eq!(input["target_file"], "a.rs");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(answer.production.effort.as_deref(), Some("high"));
    assert_eq!(msgs[3].prompt_kind, Some(PromptKind::Harness));
    assert_eq!(msgs[4].prompt_kind, Some(PromptKind::Harness));
    assert_eq!(msgs[1].prompt_kind, Some(PromptKind::Prose));
    let native = msgs[1].id.native().as_str();
    let hash = native
        .strip_prefix("01a0b507~user-")
        .and_then(|rest| rest.strip_suffix("-0"))
        .unwrap();
    assert!(
        hash.len() == 16 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "{native}"
    );
    assert!(
        step.output
            .messages
            .iter()
            .all(|r| r.id_kind == polylogue_model::IdKind::Content)
    );
    assert_eq!(step.counts.records, 6);
}

#[test]
fn appending_bytes_gives_the_same_messages() {
    let input = lines(&history());
    let whole = run(&grok(), &mut ParserState::default(), &input, 0).unwrap();
    for cut in [1, 40, 120, input.len() / 2, input.len() - 3] {
        let mut parser_state = ParserState::default();
        let first = run(&grok(), &mut parser_state, &input[..cut], 0).unwrap();
        let used = first.consumed as usize;
        let second = run(&grok(), &mut parser_state, &input[used..], used as u64).unwrap();
        let joined: Vec<&MessageRow> = first
            .output
            .messages
            .iter()
            .chain(&second.output.messages)
            .collect();
        let expect: Vec<&MessageRow> = whole.output.messages.iter().collect();
        assert_eq!(joined, expect, "cut at {cut}");
    }
}

#[test]
fn the_working_directory_is_decoded_from_the_path() {
    assert_eq!(percent_decode("%2Fhome%2Fsinity"), "/home/sinity");
    assert_eq!(percent_decode("plain%zz"), "plain%zz");
}

#[test]
fn grok_cli_inserted_line_keeps_later_ids() {
    let original = history();
    let mut rewritten = original.clone();
    rewritten.insert(
        1,
        json!({"type": "user", "content": [{"type": "text", "text": "an inserted prompt"}]}),
    );
    let ids = |records: &[Json]| -> Vec<String> {
        run(&grok(), &mut ParserState::default(), &lines(records), 0)
            .unwrap()
            .output
            .messages
            .iter()
            .map(|row| row.message.id.to_string())
            .collect()
    };
    let before = ids(&original);
    let after = ids(&rewritten);
    assert_eq!(after.len(), before.len() + 1);
    assert_eq!(after[0], before[0]);
    assert_eq!(after[2..], before[1..]);
    assert!(!before.contains(&after[1]));
}

#[test]
fn repeated_messages_are_told_apart_by_occurrence() {
    let prompt = json!({"type": "user", "content": [{"type": "text", "text": "again"}]});
    let step = run(
        &grok(),
        &mut ParserState::default(),
        &lines(&[prompt.clone(), prompt]),
        0,
    )
    .unwrap();
    let ids: Vec<_> = step
        .output
        .messages
        .iter()
        .map(|row| row.message.id.native().as_str().to_owned())
        .collect();
    assert_eq!(ids.len(), 2);
    assert!(ids[0].ends_with("-0") && ids[1].ends_with("-1"));
    assert_eq!(ids[0].trim_end_matches("-0"), ids[1].trim_end_matches("-1"));
}

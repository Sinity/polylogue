use super::*;
use crate::parse::run;
use serde_json::json;

fn grok() -> GrokCli {
    GrokCli {
        place: Place::from_path(Path::new(
            "/h/.grok/sessions/%2Frealm%2Fproject%2Fsinnix/01a0b507/chat_history.jsonl",
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
fn reasoning_joins_the_next_answer_and_messages_are_named_by_offset() {
    let input = lines(&history());
    let step = run(&grok(), &mut Carry::default(), &input, 0).unwrap();
    let msgs: Vec<&Message> = step.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(msgs.len(), 5);
    assert_eq!(msgs[0].session.to_string(), "grok-cli:01a0b507");
    assert_eq!(step.output.sessions[0].cwd, ["/realm/project/sinnix"]);
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
    assert!(msgs[1].id.native().as_str().starts_with("01a0b507@"));
    assert_eq!(step.accounting.framed, 6);
}

#[test]
fn appending_in_pieces_gives_the_same_messages() {
    let input = lines(&history());
    let whole = run(&grok(), &mut Carry::default(), &input, 0).unwrap();
    for cut in [1, 40, 120, input.len() / 2, input.len() - 3] {
        let mut carry = Carry::default();
        let first = run(&grok(), &mut carry, &input[..cut], 0).unwrap();
        let used = first.consumed as usize;
        let second = run(&grok(), &mut carry, &input[used..], used as u64).unwrap();
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

use super::*;
use crate::parse::run;
use serde_json::json;
use std::collections::HashMap;

fn main_file(stem: &str) -> ClaudeCode {
    ClaudeCode {
        file: File::Main { stem: stem.into() },
    }
}

fn jsonl(records: &[Json]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in records {
        out.extend(serde_json::to_vec(r).unwrap());
        out.push(b'\n');
    }
    out
}

/// Applies a run of steps the way the writer does for one input: a row
/// at the same offset replaces the earlier one, and of several versions of
/// one message the earliest in the input is preferred.
fn final_messages(steps: &[crate::parse::Step]) -> Vec<Message> {
    let mut rows: BTreeMap<u64, Message> = BTreeMap::new();
    for step in steps {
        for row in &step.output.messages {
            rows.insert(row.offset, row.message.clone());
        }
    }
    let mut seen = HashMap::new();
    rows.into_values()
        .filter(|m| seen.insert(m.id.clone(), ()).is_none())
        .collect()
}

fn assistant(uuid: &str, msg_id: &str, block: Json) -> Json {
    json!({"type":"assistant","uuid":uuid,"sessionId":"s1","timestamp":"2026-01-01T00:00:00Z",
           "message":{"id":msg_id,"role":"assistant","model":"claude-opus-5-5","content":[block],
                      "usage":{"input_tokens":3,"output_tokens":5}}})
}

fn conversation() -> Vec<Json> {
    vec![
        json!({"type":"user","uuid":"u1","sessionId":"s1","timestamp":"2026-01-01T00:00:00Z",
               "cwd":"/repo","gitBranch":"main","version":"2.1.0",
               "message":{"role":"user","content":"run the tests"}}),
        assistant(
            "a1",
            "msg_1",
            json!({"type":"thinking","thinking":"plan","signature":"sig"}),
        ),
        assistant("a2", "msg_1", json!({"type":"text","text":"Running."})),
        assistant(
            "a3",
            "msg_1",
            json!({"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"cargo test"}}),
        ),
        json!({"type":"user","uuid":"u2","parentUuid":"a3","sessionId":"s1","timestamp":"2026-01-01T00:00:01Z",
               "message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","is_error":true,"content":"failed\nExit code 101"}]}}),
        assistant(
            "a4",
            "msg_2",
            json!({"type":"text","text":"One test fails."}),
        ),
    ]
}

#[test]
fn a_response_split_over_records_is_one_message() {
    let mut carry = Carry::default();
    let step = run(&main_file("s1"), &mut carry, &jsonl(&conversation()), 0).unwrap();
    let msgs = final_messages(&[step]);
    let ids: Vec<_> = msgs.iter().map(|m| m.id.native().as_str()).collect();
    assert_eq!(ids, ["u1", "a1", "u2", "a4"]);
    assert_eq!(msgs[1].blocks.len(), 3);
    assert_eq!(
        msgs[0].prompt_kind,
        Some(polylogue_model::PromptKind::Prose)
    );
    assert_eq!(
        msgs[2].prompt_kind,
        Some(polylogue_model::PromptKind::Harness)
    );
    let polylogue_model::Block::ToolResult {
        outcome, call_id, ..
    } = &msgs[2].blocks[0]
    else {
        panic!("{:?}", msgs[2].blocks)
    };
    assert_eq!(call_id.as_deref(), Some("toolu_1"));
    assert_eq!(
        *outcome,
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(101))
        }
    );
}

#[test]
fn parsing_in_pieces_equals_parsing_at_once() {
    let bytes = jsonl(&conversation());
    let whole = {
        let mut carry = Carry::default();
        final_messages(&[run(&main_file("s1"), &mut carry, &bytes, 0).unwrap()])
    };
    for cut in 0..=bytes.len() {
        let parser = main_file("s1");
        let mut carry = Carry::default();
        let first = run(&parser, &mut carry, &bytes[..cut], 0).unwrap();
        // The carry travels as JSON between steps, as it does in the engine.
        let mut carry: Carry =
            serde_json::from_value(serde_json::to_value(&carry).unwrap()).unwrap();
        let used = first.consumed as usize;
        let second = run(&parser, &mut carry, &bytes[used..], used as u64).unwrap();
        assert_eq!(final_messages(&[first, second]), whole, "cut at {cut}");
    }
}

#[test]
fn a_late_continuation_reopens_its_response() {
    let records = vec![
        assistant("a1", "msg_1", json!({"type":"text","text":"part one"})),
        assistant("b1", "msg_2", json!({"type":"text","text":"other"})),
        assistant("a2", "msg_1", json!({"type":"text","text":"part two"})),
    ];
    let mut carry = Carry::default();
    let step = run(&main_file("s1"), &mut carry, &jsonl(&records), 0).unwrap();
    let msgs = final_messages(&[step]);
    assert_eq!(msgs.len(), 2);
    let first = msgs
        .iter()
        .find(|m| m.id.native().as_str() == "a1")
        .unwrap();
    assert_eq!(first.text(), "part one\n\npart two");
}

#[test]
fn a_repeated_uuid_keeps_the_first_record() {
    let records = vec![
        json!({"type":"user","uuid":"u1","sessionId":"s1","message":{"role":"user","content":"original"}}),
        json!({"type":"user","uuid":"u1","sessionId":"s1","message":{"role":"user","content":"conflicting"}}),
    ];
    let mut carry = Carry::default();
    let step = run(&main_file("s1"), &mut carry, &jsonl(&records), 0).unwrap();
    assert_eq!(step.output.messages.len(), 2);
    let msgs = final_messages(&[step]);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].text(), "original");
}

#[test]
fn copied_history_stays_in_its_original_session() {
    let records = vec![
        json!({"type":"user","uuid":"old","sessionId":"s1","message":{"role":"user","content":"earlier"}}),
        json!({"type":"user","uuid":"new","sessionId":"s2","message":{"role":"user","content":"resumed"}}),
    ];
    let mut carry = Carry::default();
    let step = run(&main_file("s2"), &mut carry, &jsonl(&records), 0).unwrap();
    let out = &step.output;
    assert_eq!(out.messages[0].message.session.native().as_str(), "s1");
    assert_eq!(out.messages[1].message.session.native().as_str(), "s2");
    assert_eq!(out.links.len(), 1);
    let l = &out.links[0];
    assert_eq!(
        (
            l.kind,
            l.parent.native().as_str(),
            l.child.native().as_str()
        ),
        (LinkKind::Copy, "s1", "s2")
    );
}

#[test]
fn subagent_and_sidechain_sessions_link_to_their_parent() {
    let sub = ClaudeCode {
        file: File::Subagent {
            stem: "agent-x1".into(),
        },
    };
    let records = vec![
        json!({"type":"user","uuid":"u","sessionId":"s1","agentId":"x1","isSidechain":true,
                              "message":{"role":"user","content":"do the subtask"}}),
    ];
    let mut carry = Carry::default();
    let out = run(&sub, &mut carry, &jsonl(&records), 0).unwrap().output;
    assert_eq!(
        out.messages[0].message.session.native().as_str(),
        "agent-x1"
    );
    assert_eq!(out.sessions[0].kind, SessionKind::Subagent);
    assert_eq!(out.links[0].kind, LinkKind::Spawn);
    assert_eq!(out.links[0].parent.native().as_str(), "s1");

    let records = vec![
        json!({"type":"user","uuid":"u","sessionId":"s1","isSidechain":true,
                              "message":{"role":"user","content":"side"}}),
    ];
    let mut carry = Carry::default();
    let out = run(&main_file("s1"), &mut carry, &jsonl(&records), 0)
        .unwrap()
        .output;
    assert_eq!(
        out.messages[0].message.session.native().as_str(),
        "s1/sidechain"
    );
    assert_eq!(out.links[0].kind, LinkKind::Spawn);
}

#[test]
fn titles_follow_precedence() {
    let records = vec![
        json!({"type":"custom-title","customTitle":"Mine","sessionId":"s1"}),
        json!({"type":"ai-title","aiTitle":"Generated","sessionId":"s1"}),
        json!({"type":"summary","summary":"Summary","leafUuid":"x"}),
    ];
    let mut carry = Carry::default();
    let out = run(&main_file("s1"), &mut carry, &jsonl(&records), 0)
        .unwrap()
        .output;
    assert_eq!(out.sessions[0].title.as_ref().unwrap().text, "Mine");
}

#[test]
fn background_completion_updates_the_outcome() {
    let record = json!({"type":"queue-operation","operation":"enqueue","sessionId":"s1",
        "content":"<task-notification><task-id>bg7</task-id><status>failed</status></task-notification>"});
    let mut carry = Carry::default();
    let out = run(&main_file("s1"), &mut carry, &jsonl(&[record]), 0)
        .unwrap()
        .output;
    assert_eq!(out.background.len(), 1);
    assert_eq!(out.background[0].task_id, "bg7");
    assert_eq!(
        out.background[0].outcome,
        ToolOutcome::Error { reason: None }
    );
    let record = json!({"type":"queue-operation","sessionId":"s1",
        "content":"<task-notification>\n<task-id>bg8</task-id>\n<tool-use-id>toolu_8</tool-use-id>\n<status>failed</status>\n<summary>Background command \"x\" failed with exit code 7</summary>\n</task-notification>"});
    let mut carry = Carry::default();
    let out = run(&main_file("s1"), &mut carry, &jsonl(&[record]), 0)
        .unwrap()
        .output;
    assert_eq!(out.background[0].call_id.as_deref(), Some("toolu_8"));
    assert_eq!(
        out.background[0].outcome,
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(7))
        }
    );
    assert_eq!(out.events[0].kind, "queue");
}

#[test]
fn every_record_is_accounted_for() {
    let mut bytes = jsonl(&[
        json!({"type":"progress","data":{"type":"bash_progress"},"sessionId":"s1"}),
        json!({"type":"mystery","sessionId":"s1"}),
        json!({"type":"user","uuid":"u","sessionId":"s1","newField":1,"message":{"role":"user","content":"x"}}),
    ]);
    bytes.extend(b"{not json\n");
    let mut carry = Carry::default();
    let step = run(&main_file("s1"), &mut carry, &bytes, 0).unwrap();
    let acc = &step.accounting;
    assert_eq!(acc.framed, 4);
    assert_eq!(acc.accounted(), 4);
    assert_eq!(acc.materialized["user"], 1);
    assert!(
        acc.unknown
            .contains_key("mystery: unrecognized record type")
    );
    assert_eq!(acc.refused.len(), 1);
    assert_eq!(acc.unread_fields["user.newField"], 1);
}

#[test]
fn file_roles_come_from_the_path() {
    let file = |p: &str| File::from_path(Path::new(p));
    let main = |s: &str| Some(File::Main { stem: s.into() });
    let sub = |s: &str| Some(File::Subagent { stem: s.into() });
    assert_eq!(file("/h/.claude/projects/p/s1.jsonl"), main("s1"));
    // A copy kept in the session's own directory.
    assert_eq!(file("/h/.claude/projects/p/s1/s1.jsonl"), main("s1"));
    assert_eq!(file("/h/.claude/projects/p/s1/other.jsonl"), None);
    assert_eq!(
        file("/h/.claude/projects/p/s1.orphaned-17840-d0d1.jsonl"),
        main("s1")
    );
    assert_eq!(
        file("/h/.claude/projects/p/s1/subagents/agent-a.jsonl"),
        sub("agent-a")
    );
    // The older layout beside the session file.
    assert_eq!(file("/h/.claude/projects/p/agent-a.jsonl"), sub("agent-a"));
    assert_eq!(
        file("/h/.claude/projects/p/s1/subagents/workflows/wf_1/agent-b.jsonl"),
        sub("agent-b")
    );
    assert_eq!(
        file("/h/.claude/projects/p/s1/subagents/workflows/wf_1/journal.jsonl"),
        Some(File::Journal {
            session: "s1".into(),
            run: "wf_1".into()
        })
    );
    assert_eq!(file("/h/.claude/projects/p/s1/subagents/other.jsonl"), None);
    assert_eq!(file("/h/.claude/projects/p/s1.json"), None);
    let path = Path::new("/h/.claude/projects/p/s.jsonl");
    assert!(recognize(path, b"  {\"type\""));
    assert!(!recognize(path, b"[1,2]"));
}

#[test]
fn a_session_names_its_plan_once_per_slug() {
    let rec = |uuid: &str, slug: &str| {
        json!({"type":"permission-mode","uuid":uuid,"sessionId":"s1","slug":slug,
               "timestamp":"2026-01-01T00:00:00Z","permissionMode":"plan"})
    };
    let bytes = jsonl(&[
        rec("u1", "quiet-river"),
        rec("u2", "quiet-river"),
        rec("u3", "loud-sea"),
    ]);
    let step = run(&main_file("s1"), &mut Carry::default(), &bytes, 0).unwrap();
    let plans: Vec<_> = step
        .output
        .events
        .iter()
        .filter(|e| e.kind == "plan")
        .map(|e| e.data["slug"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(plans, ["quiet-river", "loud-sea"]);
}

#[test]
fn usage_keeps_one_hour_cache_writes_and_the_speed() {
    let rec = json!({"type":"assistant","uuid":"a1","sessionId":"s1","timestamp":"2026-01-01T00:00:00Z",
        "message":{"id":"msg_1","role":"assistant","model":"claude-opus-5-5",
                   "content":[{"type":"text","text":"ok"}],
                   "usage":{"input_tokens":3,"output_tokens":5,"cache_read_input_tokens":100,
                            "cache_creation_input_tokens":900,
                            "cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":900},
                            "service_tier":"standard","speed":"fast"}}});
    let step = run(&main_file("s1"), &mut Default::default(), &jsonl(&[rec]), 0).unwrap();
    let u = &final_messages(&[step])[0].usage;
    assert_eq!(u.cache_write, Some(900));
    assert_eq!(u.cache_write_1h, Some(900));
    assert_eq!(u.service_tier.as_deref(), Some("standard"));
    assert_eq!(u.speed.as_deref(), Some("fast"));
}

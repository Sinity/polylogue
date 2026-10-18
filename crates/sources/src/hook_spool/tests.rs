use super::*;
use crate::parse::run;
use serde_json::json;

fn lines(records: &[Json]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in records {
        out.extend(serde_json::to_vec(r).unwrap());
        out.push(b'\n');
    }
    out
}

#[test]
fn each_line_is_an_event_on_the_harness_session() {
    let input = lines(&[
        json!({"event_type": "SessionStart", "session_id": "s1", "timestamp": "2026-09-29T08:31:48Z",
               "provider": "claude-code",
               "payload": {"session_id": "s1", "cwd": "/repo", "hook_event_name": "SessionStart",
                           "source": "startup"}}),
        json!({"event_id": "e2", "event_type": "PreToolUse", "observed_at_ms": 1790531444000_i64,
               "provider": "claude-code", "session_id": "s1", "timestamp": "2026-09-27T17:50:44Z",
               "payload": {"tool_name": "Bash", "tool_input": {"command": "ls"}}}),
        json!({"event_type": "Stop", "provider": "codex", "payload": {}}),
    ]);
    let step = run(&HookSpool, &mut Carry::default(), &input, 0).unwrap();
    let kinds: Vec<&str> = step.output.events.iter().map(|e| e.kind.as_str()).collect();
    assert_eq!(kinds, ["hook_session_start", "hook_pre_tool_use"]);
    let pre = &step.output.events[1];
    assert_eq!(pre.session.to_string(), "claude-code:s1");
    assert_eq!(pre.data["event_id"], "e2");
    assert_eq!(pre.data["tool_input"]["command"], "ls");
    let (start, end) = pre.range.unwrap();
    assert_eq!(&input[start as usize..end as usize], {
        let first_len = input.iter().position(|&b| b == b'\n').unwrap() + 1;
        let second_len = input[first_len..].iter().position(|&b| b == b'\n').unwrap();
        &input[first_len..first_len + second_len]
    });
    assert_eq!(step.output.sessions[0].cwd, ["/repo"]);
    assert_eq!(step.accounting.unknown["Stop: event names no session"], 1);
    assert_eq!(step.accounting.framed, 3);
}

#[test]
fn event_kinds_are_snake_case() {
    assert_eq!(event_kind("PreToolUse"), "hook_pre_tool_use");
    assert_eq!(event_kind("UserPromptSubmit"), "hook_user_prompt_submit");
    assert_eq!(event_kind("session-end"), "hook_session_end");
}

#[test]
fn recognizes_both_layouts() {
    assert!(recognize(
        Path::new("/s/polylogue/hooks/claude-code-s1.jsonl"),
        b"{\"event_type\": \"SessionStart\""
    ));
    assert!(recognize(
        Path::new("/s/polylogue/hooks/carriers/codex/2026-09-27/aggregate-1.ndjson"),
        b"{\"event_id\":\"a0\""
    ));
    assert!(!recognize(
        Path::new("/s/polylogue/hooks/x.jsonl"),
        b"{\"type\": \"user\""
    ));
}

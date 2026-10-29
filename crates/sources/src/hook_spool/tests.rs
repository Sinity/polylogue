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
    let step = run(&HookSpool, &mut ParserState::default(), &input, 0).unwrap();
    let kinds: Vec<&str> = step.output.events.iter().map(|e| e.kind.as_str()).collect();
    assert_eq!(kinds, ["hook_session_start", "hook_pre_tool_use"]);
    let pre = &step.output.events[1];
    assert_eq!(pre.session.as_ref().unwrap().to_string(), "claude-code:s1");
    assert_eq!(pre.data["event_id"], "e2");
    assert_eq!(pre.data["tool_input"]["command"], "ls");
    let (start, end) = pre.range.unwrap();
    assert_eq!(&input[start as usize..end as usize], {
        let first_len = input.iter().position(|&b| b == b'\n').unwrap() + 1;
        let second_len = input[first_len..].iter().position(|&b| b == b'\n').unwrap();
        &input[first_len..first_len + second_len]
    });
    assert!(
        step.output.sessions.is_empty(),
        "a hook line made a session"
    );
    assert_eq!(step.counts.skipped["Stop: event names no session"], 1);
    assert_eq!(step.counts.records, 3);
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

/// The line `polylogue hook` writes (crates/polylogue/src/hook.rs).
#[test]
fn the_hook_commands_carrier_lines_are_read() {
    let input = lines(&[json!({
        "at_ns": "1792400000123456789", "event": "PreToolUse", "harness_pid": 4242,
        "payload": {"session_id": "s9", "cwd": "/repo", "tool_name": "Bash"}
    })]);
    assert!(recognize(
        Path::new("/s/polylogue/hooks/4242.ndjson"),
        &input[..40]
    ));
    let mut parser_state = ParserState::default();
    let step = run(&HookSpool, &mut parser_state, &input, 0).unwrap();
    let e = &step.output.events[0];
    assert_eq!(e.kind, "hook_pre_tool_use");
    assert_eq!(e.session.as_ref().unwrap().to_string(), "claude-code:s9");
    assert_eq!(e.at.unwrap().unix_ms(), 1_792_400_000_123);
    assert!(
        step.counts.unread_fields.is_empty(),
        "{:?}",
        step.counts.unread_fields
    );
    assert!(LOCATIONS.contains(&"polylogue/hooks/*.ndjson"));
}

/// Two invocations of the hook with the same payload are two events; the
/// line's own time and process tell them apart.
#[test]
fn each_invocation_carries_its_own_time_and_process() {
    let line = |at_ns: &str| {
        json!({"at_ns": at_ns, "event": "PreToolUse", "harness_pid": 4242, "provider": "claude-code",
               "session_id": "s9", "payload": {"session_id": "s9", "tool_name": "Bash"}})
    };
    let input = lines(&[line("1792400000123456789"), line("1792400000123456790")]);
    let step = run(&HookSpool, &mut ParserState::default(), &input, 0).unwrap();
    let [first, second] = &step.output.events[..] else {
        panic!("{:?}", step.output.events)
    };
    assert_ne!(
        first.at, second.at,
        "native nanosecond observations remain distinct"
    );
    assert_eq!(first.at.unwrap().unix_ms(), second.at.unwrap().unix_ms());
    assert_eq!(first.data["at_ns"], "1792400000123456789");
    assert_eq!(second.data["at_ns"], "1792400000123456790");
    assert_eq!(first.data["harness_pid"], 4242);
    assert_eq!(first.data["tool_name"], "Bash");
    assert_ne!(first.data, second.data);
}

#[test]
fn native_work_evidence_keeps_its_exact_hook_record_range_and_unknowns() {
    let git =
        json!({"common_dir":"/synthetic/.git","root_commits":["abc"],"head":"def","branch":null});
    let terminal = json!({"host":"synthetic","adapter":"kitty","process_id":22,"window_id":"4","parent_process_id":1,"boot_id":"boot"});
    let input = lines(&[
        json!({"event_id":"native-one","event":"SessionStart","provider":"codex","session_id":"session","at_ns":"1000000",
        "git_identity":git,"terminal_identity":terminal,"boot_id":"boot","repository":"https://example.test/project.git",
        "native_unavailable":["optional observation unavailable"],
        "payload":{"objectives":[{"kind":"task","reference":"demo-1","repository":"https://example.test/project.git"}]}}),
    ]);
    let step = run(&HookSpool, &mut ParserState::default(), &input, 0).unwrap();
    let event = &step.output.events[0];
    assert_eq!(event.range, Some((0, input.len() as u64 - 1)));
    assert_eq!(event.data["git_identity"], git);
    assert_eq!(event.data["terminal_identity"], terminal);
    assert_eq!(event.data["objectives"][0]["reference"], "demo-1");
    assert_eq!(event.data["boot_id"], "boot");
    assert_eq!(
        event.data["native_unavailable"],
        json!(["optional observation unavailable"])
    );
    assert!(step.output.sessions.is_empty());
}

#[test]
fn native_nanosecond_observation_keeps_quality_and_missing_clock_stays_unknown() {
    use polylogue_model::{Precision, TimeQuality};
    let input = lines(&[
        json!({"event_type":"Stop","provider":"codex","session_id":"s","at_ns":"1792400000123456789"}),
        json!({"event_type":"Stop","provider":"codex","session_id":"s"}),
    ]);
    let step = run(&HookSpool, &mut ParserState::default(), &input, 0).unwrap();
    let event = &step.output.events[0];
    assert_eq!(event.at.unwrap().unix_ns(), 1792400000123456789);
    assert_eq!(event.at_quality, TimeQuality::Captured);
    assert_eq!(event.at_precision, Some(Precision::Instant));
    assert_eq!(event.data["at_ns"], "1792400000123456789");
    assert_eq!(step.output.events[1].at, None);
    assert_eq!(step.output.events[1].at_quality, TimeQuality::Unknown);
}

use super::*;
use crate::parse::run;
use serde_json::json;

/// The rule as a file found at a location meets it: its first bytes alone.
fn recognize(path: &std::path::Path, head: &[u8]) -> bool {
    super::recognize(path, &crate::document::Head::new(head))
}

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
    let step = run(
        &HookSpool::default(),
        &mut ParserState::default(),
        &input,
        0,
    )
    .unwrap();
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
fn unreadable_hook_payload_has_its_recorded_reason() {
    let input = lines(&[
        json!({"event_id":"e1", "event":"Stop", "provider":"claude-code",
            "unreadable_payload":{"bytes_base64":"/w==", "reason":"hook payload is not JSON"}}),
        json!({"event_id":"e2", "event":"Stop", "provider":"claude-code",
            "session_id":"s1", "payload":{"cwd":"/repo"}}),
    ]);
    let step = run(
        &HookSpool::default(),
        &mut ParserState::default(),
        &input,
        0,
    )
    .unwrap();
    assert_eq!(step.counts.unreadable["Stop: hook payload is not JSON"], 1);
    assert!(step.counts.unread_fields.is_empty());
    let [event] = &step.output.events[..] else {
        panic!("unexpected events: {:?}", step.output.events);
    };
    assert_eq!(event.data["event_id"], "e2");
    assert_eq!(event.data["cwd"], "/repo");
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
    let step = run(&HookSpool::default(), &mut parser_state, &input, 0).unwrap();
    let e = &step.output.events[0];
    assert_eq!(e.kind, "hook_pre_tool_use");
    assert_eq!(e.session.as_ref().unwrap().to_string(), "claude-code:s9");
    assert_eq!(e.at.unwrap().unix_ms(), 1_792_400_000_123);
    assert!(
        step.counts.unread_fields.is_empty(),
        "{:?}",
        step.counts.unread_fields
    );
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
    let step = run(
        &HookSpool::default(),
        &mut ParserState::default(),
        &input,
        0,
    )
    .unwrap();
    let [first, second] = &step.output.events[..] else {
        panic!("{:?}", step.output.events)
    };
    assert_ne!(
        first.at, second.at,
        "source nanosecond observations remain distinct"
    );
    assert_eq!(first.at.unwrap().unix_ms(), second.at.unwrap().unix_ms());
    assert_eq!(first.data["at_ns"], "1792400000123456789");
    assert_eq!(second.data["at_ns"], "1792400000123456790");
    assert_eq!(first.data["harness_pid"], 4242);
    assert_eq!(first.data["tool_name"], "Bash");
    assert_ne!(first.data, second.data);
}

#[test]
fn work_live_events_keep_their_hook_record_range() {
    let git =
        json!({"common_dir":"/synthetic/.git","root_commits":["abc"],"head":"def","branch":null});
    let terminal = json!({"host":"synthetic","adapter":"kitty","process_id":22,"window_id":"4","parent_process_id":1,"boot_id":"boot"});
    let input = lines(&[
        json!({"event_id":"native-one","event":"SessionStart","provider":"codex","session_id":"session","at_ns":"1000000",
        "git_identity":git,"terminal_identity":terminal,"boot_id":"boot","repository":"https://example.test/project.git",
        "unstated":["optional observation unavailable"],
        "payload":{"objectives":[{"kind":"task","reference":"demo-1","repository":"https://example.test/project.git"}]}}),
    ]);
    let step = run(
        &HookSpool::default(),
        &mut ParserState::default(),
        &input,
        0,
    )
    .unwrap();
    let event = &step.output.events[0];
    assert_eq!(event.range, Some((0, input.len() as u64 - 1)));
    assert_eq!(event.data["git_identity"], git);
    assert_eq!(event.data["terminal_identity"], terminal);
    assert_eq!(event.data["objectives"][0]["reference"], "demo-1");
    assert_eq!(event.data["boot_id"], "boot");
    assert_eq!(
        event.data["unstated"],
        json!(["optional observation unavailable"])
    );
    assert!(step.output.sessions.is_empty());
}

#[test]
fn nanosecond_times_keep_quality_and_unknown_clock() {
    use polylogue_model::{Precision, TimeQuality};
    let input = lines(&[
        json!({"event_type":"Stop","provider":"codex","session_id":"s","at_ns":"1792400000123456789"}),
        json!({"event_type":"Stop","provider":"codex","session_id":"s"}),
    ]);
    let step = run(
        &HookSpool::default(),
        &mut ParserState::default(),
        &input,
        0,
    )
    .unwrap();
    let event = &step.output.events[0];
    assert_eq!(event.at.unwrap().unix_ns(), 1792400000123456789);
    assert_eq!(event.at_quality, TimeQuality::Captured);
    assert_eq!(event.at_precision, Some(Precision::Instant));
    assert_eq!(event.data["at_ns"], "1792400000123456789");
    assert_eq!(step.output.events[1].at, None);
    assert_eq!(step.output.events[1].at_quality, TimeQuality::Unknown);
}

fn emission_line() -> Json {
    json!({"event_id":"emission1", "event":CONTEXT_EMITTED,
        "provider":"claude-code", "session_id":"s1", "at_ns":"1792400000123456789",
        "payload":{"invocation":"invocation1", "session":"claude-code:s1",
            "hook_event":"SessionStart", "deliveries":["delivery:receipt1", "delivery:receipt2"]}})
}

#[test]
fn hook_emission_keeps_producer_references() {
    let line = emission_line();
    let bytes = lines(&[line]);
    let step = run(
        &HookSpool::default(),
        &mut ParserState::default(),
        &bytes,
        0,
    )
    .unwrap();
    let [event] = &step.output.events[..] else {
        panic!("one emission event")
    };
    assert_eq!(event.data["invocation"], "invocation1");
    assert_eq!(event.data["session"], "claude-code:s1");
    assert_eq!(event.data["hook_event"], "SessionStart");
    assert_eq!(
        event.data["deliveries"],
        json!(["delivery:receipt1", "delivery:receipt2"])
    );
    assert_eq!(event.data["event_id"], "emission1");
    assert!(step.output.messages.is_empty());
    assert!(step.output.hook_fields.is_empty());
}

/// Lines a later build's hook writes, from that build's rollback fixture:
/// this reader must take every one after a rollback.
#[test]
fn rollback_lines_from_the_next_build_are_read() {
    let input = include_bytes!("rollback-lines.ndjson");
    assert!(recognize(
        Path::new("/home/op/.local/state/polylogue/hooks/4242.ndjson"),
        input
    ));
    for line in input.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        assert!(
            recognize(Path::new("/s/polylogue/hooks/4242.ndjson"), line),
            "{}",
            String::from_utf8_lossy(line)
        );
    }
    let step = run(&HookSpool::default(), &mut ParserState::default(), input, 0).unwrap();
    assert_eq!(step.counts.records, 6);
    assert!(
        step.counts.unreadable.is_empty(),
        "{:?}",
        step.counts.unreadable
    );
    let sessions: Vec<String> = step
        .output
        .events
        .iter()
        .map(|e| e.session.as_ref().unwrap().to_string())
        .collect();
    assert_eq!(
        sessions,
        [
            "claude-code:rollback-claude",
            "codex:rollback-codex",
            "gemini-cli:rollback-gemini",
            "hermes:rollback-hermes@profile-caa890356da51a01",
            "claude-code:rollback-claude",
        ]
    );
    let kinds: Vec<&str> = step.output.events.iter().map(|e| e.kind.as_str()).collect();
    assert_eq!(
        kinds,
        [
            "hook_session_start",
            "hook_pre_tool_use",
            "hook_after_tool",
            "hook_pre_llm_call",
            "hook_context_emitted",
        ]
    );
    assert_eq!(
        step.counts.skipped["Notification: event names no session"],
        1
    );
    assert_eq!(step.output.events[1].data["tool_name"], "shell");
    assert_eq!(
        step.output.events[0].data["event_id"],
        "019b7600-0000-7000-8000-000000000001"
    );
}

#[test]
fn process_and_registry_lines_are_live_events() {
    let input = lines(&[
        json!({"at_ns": "1790531444000000000", "route": "process", "event": "RootGone",
               "provider": "claude-code", "session_id": "s1",
               "payload": {"pid": 41, "start": 900, "boot_id": "b1"}}),
        json!({"at_ns": "1790531445000000000", "route": "registry", "event": "Status",
               "provider": "codex", "session_id": "t1",
               "payload": {"status": "busy", "pid": 42, "start": 901, "boot_id": "b1"}}),
        json!({"at_ns": "1790531446000000000", "route": "elsewhere", "event": "Status",
               "provider": "codex", "session_id": "t1", "payload": {}}),
        json!({"at_ns": "1790531447000000000", "event": "PreToolUse", "harness_pid": 42,
               "harness_start": 901, "boot_id": "b1", "provider": "claude-code",
               "session_id": "s1", "payload": {"tool_name": "Edit"}}),
    ]);
    let step = run(
        &HookSpool::default(),
        &mut ParserState::default(),
        &input,
        0,
    )
    .unwrap();
    let kinds: Vec<&str> = step.output.events.iter().map(|e| e.kind.as_str()).collect();
    assert_eq!(
        kinds,
        ["process_root_gone", "registry_status", "hook_pre_tool_use"]
    );
    assert_eq!(step.output.events[0].data["pid"], 41);
    assert_eq!(step.output.events[1].data["status"], "busy");
    // The hook line keeps the whole process identity.
    assert_eq!(step.output.events[2].data["harness_start"], 901);
    assert_eq!(step.counts.records, 4);
    for kind in &kinds {
        assert!(polylogue_model::live::is_live_event(kind), "{kind}");
    }
}

#[test]
fn a_provider_or_route_that_is_not_a_string_is_unreadable() {
    let input = lines(&[
        json!({"event_type": "Stop", "session_id": "s", "provider": 3, "at_ns": "1792400000123456789"}),
        json!({"event_type": "Status", "session_id": "s", "provider": "claude-code", "route": 5}),
    ]);
    let step = run(
        &HookSpool::default(),
        &mut ParserState::default(),
        &input,
        0,
    )
    .unwrap();
    assert!(step.output.events.is_empty());
    assert_eq!(step.counts.unreadable.len(), 2);
    assert_eq!(step.counts.unreadable["Stop: provider is not a string"], 1);
    assert_eq!(step.counts.unreadable["Status: route is not a string"], 1);
}

#[test]
fn a_per_session_or_dated_files_name_gives_the_provider_its_lines_omit() {
    let input = lines(&[
        json!({"event_type": "Stop", "session_id": "019a-0001", "timestamp": "2026-09-29T08:31:48Z",
               "payload": {}}),
        json!({"event_type": "Stop", "session_id": "019a-0001", "provider": "gemini-cli",
               "timestamp": "2026-09-29T08:31:49Z", "payload": {}}),
    ]);
    let sessions = |path: &str| -> Vec<String> {
        let Parse::Lines(parser) = open(Path::new(path)).unwrap() else {
            panic!("a spool file is read by lines");
        };
        let (step, _) = parser.step(None, &input, 0, false).unwrap();
        step.output
            .events
            .iter()
            .map(|event| event.session.as_ref().unwrap().to_string())
            .collect()
    };
    // The origin name holds dashes, and so does the session ID: the
    // longest origin name before a dash is the provider.
    assert_eq!(
        sessions("/s/polylogue/hooks/claude-code-019a-0001.jsonl"),
        ["claude-code:019a-0001", "gemini-cli:019a-0001"]
    );
    assert_eq!(
        sessions("/s/polylogue/hooks/carriers/codex/2026-09-29/one.ndjson"),
        ["codex:019a-0001", "gemini-cli:019a-0001"]
    );
    // A file whose name gives no provider still refuses such lines.
    let Parse::Lines(parser) = open(Path::new("/s/polylogue/hooks/unnamed.jsonl")).unwrap() else {
        panic!("a spool file is read by lines");
    };
    let (step, _) = parser.step(None, &input, 0, false).unwrap();
    assert_eq!(step.counts.unreadable["Stop: no provider"], 1);
}

use super::*;
use crate::parse::{Step, run};
use polylogue_model::{ErrorReason, UnknownReason};
use serde_json::json;

const THREAD: &str = "019a0000-0000-7000-8000-000000000001";
const PARENT: &str = "019a0000-0000-7000-8000-0000000000aa";

fn rollout() -> Rollout {
    Rollout {
        thread: Some(THREAD.into()),
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

fn env(ordinal: u64, ty: &str, payload: Json) -> Json {
    json!({"timestamp": format!("2026-03-01T10:00:{:02}.000Z", ordinal % 60), "ordinal": ordinal, "type": ty, "payload": payload})
}

fn meta(extra: Json) -> Json {
    let mut p = json!({"id": THREAD, "timestamp": "2026-03-01T10:00:00.000Z", "cwd": "/work/repo",
                       "originator": "codex_cli_rs", "cli_version": "0.160.0", "source": "cli",
                       "model_provider": "openai", "base_instructions": {"text": "..."},
                       "git": {"branch": "main", "commit_hash": "abc"}});
    if let (Some(p), Some(e)) = (p.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            p.insert(k.clone(), v.clone());
        }
    }
    env(1, "session_meta", p)
}

fn user(o: u64, text: &str) -> Json {
    env(
        o,
        "response_item",
        json!({"type": "message", "role": "user",
        "content": [{"type": "input_text", "text": text}]}),
    )
}

fn turn() -> Vec<Json> {
    vec![
        meta(json!({})),
        env(
            2,
            "turn_context",
            json!({"turn_id": "t1", "cwd": "/work/repo", "model": "gpt-6.1-sol",
            "approval_policy": "never", "sandbox_policy": {"type": "danger-full-access"}, "summary": "auto"}),
        ),
        user(
            3,
            "<environment_context>\n  <cwd>/work/repo</cwd>\n</environment_context>",
        ),
        user(4, "run the tests"),
        env(
            5,
            "response_item",
            json!({"type": "reasoning", "summary": [{"type": "summary_text", "text": "Run cargo."}],
            "content": null, "encrypted_content": "gAAA"}),
        ),
        env(
            6,
            "response_item",
            json!({"type": "function_call", "name": "exec_command",
            "arguments": "{\"cmd\":\"cargo test\",\"workdir\":\"/work/repo\"}", "call_id": "call_1"}),
        ),
        env(
            7,
            "event_msg",
            json!({"type": "token_count",
            "info": {"last_token_usage": {"input_tokens": 1000, "cached_input_tokens": 600,
                     "output_tokens": 50, "reasoning_output_tokens": 20, "total_tokens": 1050}},
            "rate_limits": {"primary": {"used_percent": 12.0, "window_minutes": 300}}}),
        ),
        env(
            8,
            "event_msg",
            json!({"type": "item_completed", "thread_id": THREAD, "turn_id": "t1",
            "item": {"type": "CommandExecution", "id": "call_1", "command": ["zsh", "-lc", "cargo test"],
                     "status": "completed", "exit_code": 101, "duration": {"secs": 2, "nanos": 0},
                     "aggregated_output": "test failed", "source": "unified_exec_startup"},
            "completed_at_ms": 1}),
        ),
        env(
            9,
            "response_item",
            json!({"type": "function_call_output", "call_id": "call_1",
            "output": "Chunk ID: a\nWall time: 2.0 seconds\nProcess exited with code 101\nOutput:\ntest failed"}),
        ),
        env(
            10,
            "event_msg",
            json!({"type": "agent_message", "message": "One test fails.", "phase": "final_answer"}),
        ),
        env(
            11,
            "response_item",
            json!({"type": "message", "role": "assistant", "phase": "final_answer",
            "content": [{"type": "output_text", "text": "One test fails."}]}),
        ),
        env(
            12,
            "event_msg",
            json!({"type": "token_count",
            "info": {"last_token_usage": {"input_tokens": 1200, "cached_input_tokens": 1000,
                     "output_tokens": 10, "reasoning_output_tokens": 0, "total_tokens": 1210}},
            "rate_limits": {"primary": {"used_percent": 12.0, "window_minutes": 300}}}),
        ),
        env(
            13,
            "event_msg",
            json!({"type": "task_complete", "turn_id": "t1", "last_agent_message": "One test fails."}),
        ),
    ]
}

/// Applies steps the way the writer does for one input: a row at the same
/// offset replaces the earlier one.
fn final_messages(steps: &[Step]) -> Vec<Message> {
    let mut rows: BTreeMap<u64, Message> = BTreeMap::new();
    for step in steps {
        for row in &step.output.messages {
            rows.insert(row.offset, row.message.clone());
        }
    }
    rows.into_values().collect()
}

fn texts(msgs: &[Message]) -> Vec<String> {
    msgs.iter().map(Message::text).collect()
}

#[test]
fn a_turn_becomes_one_message_per_item() {
    let mut parser_state = ParserState::default();
    let step = run(&rollout(), &mut parser_state, &jsonl(&turn()), 0).unwrap();
    assert!(
        step.counts.unreadable.is_empty(),
        "{:?}",
        step.counts.unreadable
    );
    let msgs = final_messages(std::slice::from_ref(&step));
    let roles: Vec<_> = msgs.iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        [
            Role::User,
            Role::User,
            Role::Assistant,
            Role::Assistant,
            Role::User,
            Role::Assistant
        ]
    );
    assert_eq!(msgs[0].prompt_kind, Some(PromptKind::Harness));
    assert_eq!(msgs[1].prompt_kind, Some(PromptKind::Prose));
    assert_eq!(msgs[2].text(), "Run cargo.");
    assert_eq!(msgs[5].text(), "One test fails.");
    assert_eq!(msgs[5].model.as_deref(), Some("gpt-6.1-sol"));
    // The mirror of the final answer is not stored twice.
    assert_eq!(
        texts(&msgs)
            .iter()
            .filter(|t| *t == "One test fails.")
            .count(),
        1
    );
    let session = &step.output.sessions[0];
    assert_eq!(session.id.to_string(), format!("codex:{THREAD}"));
    assert_eq!(session.cwd, ["/work/repo"]);
    assert_eq!(session.git_branch.as_deref(), Some("main"));
    assert_eq!(session.tool_version.as_deref(), Some("0.160.0"));
}

fn readings(step: &Step) -> Vec<polylogue_model::UsageReport> {
    step.output
        .events
        .iter()
        .filter(|e| e.kind == polylogue_model::reports::USAGE_REPORT)
        .map(|e| serde_json::from_value(e.data.clone()).unwrap())
        .collect()
}

#[test]
fn usage_is_independent_of_items_and_partial_counters_do_not_prove_completion() {
    let step = run(&rollout(), &mut ParserState::default(), &jsonl(&turn()), 0).unwrap();
    let readings = readings(&step);
    assert_eq!(readings.len(), 2);
    assert_eq!(readings[0].usage.input, Some(400));
    assert_eq!(readings[0].usage.output, Some(30));
    assert_eq!(readings[1].usage.input, Some(200));
    assert!(readings.iter().all(|r| r.response.is_none()
        && r.measured_completion.is_none()
        && r.measured_at.is_none()));
    assert!(
        step.output
            .messages
            .iter()
            .all(|row| row.message.usage.is_empty())
    );
}

#[test]
fn rate_limits_are_stored_independently_when_they_change() {
    let step = run(&rollout(), &mut ParserState::default(), &jsonl(&turn()), 0).unwrap();
    assert_eq!(
        step.output
            .events
            .iter()
            .filter(|e| e.kind == "quota_report")
            .count(),
        1
    );
    assert_eq!(
        step.output
            .events
            .iter()
            .filter(|e| e.kind == "token_count")
            .count(),
        2
    );
}

#[test]
fn the_thread_item_gives_the_call_its_exit_code_and_duration() {
    let mut parser_state = ParserState::default();
    let step = run(&rollout(), &mut parser_state, &jsonl(&turn()), 0).unwrap();
    let msgs = final_messages(&[step]);
    let Block::ToolCall { command, tool, .. } = &msgs[3].blocks[0] else {
        panic!("{:?}", msgs[3].blocks)
    };
    assert_eq!(command.as_deref(), Some("cargo test"));
    assert_eq!(tool.as_deref(), Some("exec_command"));
    let Block::ToolResult {
        outcome,
        reported_duration_ms,
        ..
    } = &msgs[4].blocks[0]
    else {
        panic!()
    };
    assert_eq!(
        *outcome,
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(101))
        }
    );
    assert_eq!(*reported_duration_ms, Some(2000));
}

#[test]
fn parsing_in_appended_bytes_equals_parsing_at_once() {
    let input = jsonl(&turn());
    let mut whole_carry = ParserState::default();
    let whole = final_messages(&[run(&rollout(), &mut whole_carry, &input, 0).unwrap()]);
    let mut cut = 0;
    while cut < input.len() {
        let mut parser_state = ParserState::default();
        let first = run(&rollout(), &mut parser_state, &input[..cut], 0).unwrap();
        // The parser state crosses a restart as JSON.
        let mut parser_state: ParserState =
            serde_json::from_value(serde_json::to_value(&parser_state).unwrap()).unwrap();
        let used = first.consumed as usize;
        let second = run(&rollout(), &mut parser_state, &input[used..], used as u64).unwrap();
        assert_eq!(final_messages(&[first, second]), whole, "cut at {cut}");
        cut += 37;
    }
}

#[test]
fn a_command_left_running_is_updated_when_it_ends() {
    let records = vec![
        meta(json!({})),
        env(
            2,
            "response_item",
            json!({"type": "function_call", "name": "exec_command",
            "arguments": "{\"cmd\":\"cargo build\"}", "call_id": "call_a"}),
        ),
        env(
            3,
            "response_item",
            json!({"type": "function_call_output", "call_id": "call_a",
            "output": "Chunk ID: x\nWall time: 1.0 seconds\nProcess running with session ID 4242\nOutput:\n"}),
        ),
        env(
            4,
            "response_item",
            json!({"type": "function_call", "name": "write_stdin",
            "arguments": "{\"session_id\":4242,\"chars\":\"\"}", "call_id": "call_b"}),
        ),
        env(
            5,
            "response_item",
            json!({"type": "function_call_output", "call_id": "call_b",
            "output": "Chunk ID: y\nWall time: 5.0 seconds\nProcess exited with code 1\nOutput:\nerror"}),
        ),
    ];
    let mut parser_state = ParserState::default();
    let step = run(&rollout(), &mut parser_state, &jsonl(&records), 0).unwrap();
    let msgs = final_messages(std::slice::from_ref(&step));
    let Block::ToolResult { outcome, .. } = &msgs[1].blocks[0] else {
        panic!()
    };
    assert_eq!(
        *outcome,
        ToolOutcome::Unknown {
            reason: UnknownReason::RunningInBackground
        }
    );
    let [update] = step.output.background.as_slice() else {
        panic!("{:?}", step.output.background)
    };
    assert_eq!(update.call_id.as_deref(), Some("call_a"));
    assert_eq!(
        update.outcome,
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(1))
        }
    );
    assert!(parser_state.running.is_empty());
}

#[test]
fn commands_a_script_ran_are_recovered_as_calls() {
    let records = vec![
        meta(json!({})),
        env(
            2,
            "response_item",
            json!({"type": "custom_tool_call", "status": "completed", "call_id": "call_s",
            "name": "exec", "input": "await tools.exec_command({cmd: 'ls'})"}),
        ),
        env(
            3,
            "event_msg",
            json!({"type": "item_completed", "thread_id": THREAD, "turn_id": "t1",
            "item": {"type": "CommandExecution", "id": "call_inner", "command": ["zsh", "-lc", "ls"],
                     "cwd": "file:///work", "status": "completed", "exit_code": 0, "aggregated_output": "a\nb",
                     "duration": {"secs": 0, "nanos": 5_000_000}, "source": "unified_exec_startup"},
            "completed_at_ms": 1}),
        ),
        env(
            4,
            "response_item",
            json!({"type": "custom_tool_call_output", "call_id": "call_s",
            "output": [{"type": "input_text", "text": "Script completed\nWall time 0.1 seconds\nOutput:\na\nb"}]}),
        ),
    ];
    let mut parser_state = ParserState::default();
    let step = run(&rollout(), &mut parser_state, &jsonl(&records), 0).unwrap();
    let msgs = final_messages(&[step]);
    assert_eq!(msgs.len(), 3);
    let recovered = &msgs[1];
    assert_eq!(recovered.id.native().as_str(), "call_inner");
    assert_eq!(recovered.duration_ms, Some(5));
    let Block::ToolCall { command, tool, .. } = &recovered.blocks[0] else {
        panic!()
    };
    assert_eq!(command.as_deref(), Some("ls"));
    assert_eq!(tool.as_deref(), Some("exec_command"));
    let Block::ToolResult {
        outcome, output, ..
    } = &recovered.blocks[1]
    else {
        panic!()
    };
    assert_eq!(*outcome, ToolOutcome::Ok);
    assert_eq!(output.as_deref(), Some("a\nb"));
    let Block::ToolResult { outcome, .. } = &msgs[2].blocks[0] else {
        panic!()
    };
    assert_eq!(*outcome, ToolOutcome::Ok);
}

#[test]
fn a_subagents_copied_history_is_stored_once_in_the_parent() {
    let records = vec![
        meta(
            json!({"thread_source": "subagent", "parent_thread_id": PARENT, "forked_from_id": PARENT,
                    "agent_nickname": "Volta", "subagent_history_start_ordinal": 5}),
        ),
        env(
            2,
            "compacted",
            json!({"message": "", "replacement_history": [
            {"type": "message", "id": "msg_p1", "role": "user", "content": [{"type": "input_text", "text": "parent prompt"}]}]}),
        ),
        env(
            3,
            "response_item",
            json!({"type": "message", "id": "msg_p2", "role": "assistant",
            "content": [{"type": "output_text", "text": "parent answer"}]}),
        ),
        env(
            4,
            "response_item",
            json!({"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "copied without an id"}]}),
        ),
        env(
            5,
            "response_item",
            json!({"type": "message", "id": "msg_c1", "role": "user",
            "content": [{"type": "input_text", "text": "your task"}]}),
        ),
    ];
    let mut parser_state = ParserState::default();
    let step = run(&rollout(), &mut parser_state, &jsonl(&records), 0).unwrap();
    let msgs = final_messages(std::slice::from_ref(&step));
    let placed: Vec<_> = msgs
        .iter()
        .map(|m| {
            (
                m.id.native().as_str().to_owned(),
                m.session.native().as_str().to_owned(),
            )
        })
        .collect();
    assert_eq!(
        placed,
        [
            ("msg_p2".into(), PARENT.into()),
            ("msg_c1".into(), THREAD.into())
        ]
    );
    assert_eq!(step.counts.skipped.values().sum::<u64>(), 2);
    let session = &step.output.sessions[0];
    assert_eq!(session.kind, SessionKind::Subagent);
    assert_eq!(session.title.as_ref().unwrap().text, "Volta");
    let [spawn] = step.output.links.as_slice() else {
        panic!("{:?}", step.output.links)
    };
    assert_eq!(spawn.kind, LinkKind::Spawn);
    assert_eq!(spawn.parent.native().as_str(), PARENT);
    assert_eq!(spawn.at.as_ref().unwrap().native().as_str(), "msg_p2");
}

#[test]
fn spawned_agents_are_linked_from_the_call() {
    let records = vec![
        meta(json!({})),
        env(
            2,
            "response_item",
            json!({"type": "function_call", "name": "spawn_agent",
            "arguments": "{\"message\":\"go\"}", "call_id": "call_sp"}),
        ),
        env(
            3,
            "response_item",
            json!({"type": "function_call_output", "call_id": "call_sp",
            "output": "{\"agent_id\":\"019a0000-0000-7000-8000-0000000000bb\",\"nickname\":\"Euler\"}"}),
        ),
    ];
    let mut parser_state = ParserState::default();
    let step = run(&rollout(), &mut parser_state, &jsonl(&records), 0).unwrap();
    let [l] = step.output.links.as_slice() else {
        panic!()
    };
    assert_eq!(l.kind, LinkKind::Spawn);
    assert_eq!(l.call_id.as_deref(), Some("call_sp"));
    assert_eq!(
        l.child.native().as_str(),
        "019a0000-0000-7000-8000-0000000000bb"
    );
}

#[test]
fn rollouts_from_2025_have_a_header_and_bare_items() {
    let records = vec![
        json!({"id": THREAD, "timestamp": "2025-08-11T16:59:28.000Z", "instructions": null,
               "git": {"commit_hash": "abc", "branch": "main"}}),
        json!({"record_type": "state"}),
        json!({"type": "message", "id": null, "role": "user",
               "content": [{"type": "input_text", "text": "hello"}]}),
        json!({"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAA"}),
        json!({"type": "function_call", "id": "fc_1", "name": "shell",
               "arguments": "{\"command\":[\"bash\",\"-lc\",\"ls\"]}", "call_id": "call_l"}),
        json!({"type": "function_call_output", "call_id": "call_l",
               "output": "{\"output\":\"a\\n\",\"metadata\":{\"exit_code\":0,\"duration_seconds\":0.1}}"}),
    ];
    let mut parser_state = ParserState::default();
    let step = run(&rollout(), &mut parser_state, &jsonl(&records), 0).unwrap();
    assert!(step.counts.unreadable.is_empty(), "{:?}", step.counts);
    let msgs = final_messages(std::slice::from_ref(&step));
    assert_eq!(msgs.len(), 4);
    assert!(msgs.iter().all(|m| m.at.is_none()));
    let Block::Thinking { redacted, text, .. } = &msgs[1].blocks[0] else {
        panic!()
    };
    assert!(*redacted && text.is_none());
    let Block::ToolResult { outcome, .. } = &msgs[3].blocks[0] else {
        panic!()
    };
    assert_eq!(*outcome, ToolOutcome::Ok);
    assert_eq!(
        step.output.sessions[0].started,
        polylogue_model::Timestamp::parse_rfc3339("2025-08-11T16:59:28.000Z")
    );
}

#[test]
fn every_record_is_accounted_for() {
    let mut records = turn();
    records.push(json!({"type": "brand_new_envelope", "payload": {"x": 1}}));
    records.push(json!([1, 2]));
    let mut parser_state = ParserState::default();
    let step = run(&rollout(), &mut parser_state, &jsonl(&records), 0).unwrap();
    assert_eq!(step.counts.counted(), step.counts.records);
    assert_eq!(
        step.counts
            .skipped
            .get("brand_new_envelope: unrecognized record type"),
        Some(&1)
    );
}

#[test]
fn files_are_told_apart_by_name() {
    let f = |p: &str| File::from_path(Path::new(p));
    assert_eq!(
        f(
            "sessions/2026/03/01/rollout-2026-03-01T10-00-00-019a0000-0000-7000-8000-000000000001.jsonl"
        ),
        Some(File::Rollout {
            thread: Some(THREAD.into())
        })
    );
    assert_eq!(f("history.jsonl"), Some(File::History));
    assert_eq!(f("session_index.jsonl"), Some(File::SessionIndex));
    assert_eq!(f("sessions/notes.jsonl"), None);
}

#[test]
fn the_index_names_threads_and_the_history_keeps_prompts() {
    let names = jsonl(&[
        json!({"id": THREAD, "thread_name": "first", "updated_at": "2026-03-01T10:00:00Z"}),
        json!({"id": THREAD, "thread_name": "renamed", "updated_at": "2026-03-01T11:00:00Z"}),
    ]);
    let mut parser_state = index::Names::default();
    let step = run(&index::SessionIndex, &mut parser_state, &names, 0).unwrap();
    assert_eq!(step.output.sessions.len(), 1);
    assert_eq!(
        step.output.sessions[0].title.as_ref().unwrap().text,
        "renamed"
    );
    let history =
        jsonl(&[json!({"session_id": THREAD, "ts": 1_772_359_200, "text": "run the tests"})]);
    let mut parser_state = index::NoState::default();
    let step = run(&index::History, &mut parser_state, &history, 0).unwrap();
    let [e] = step.output.events.as_slice() else {
        panic!()
    };
    assert_eq!(e.kind, "prompt");
    assert_eq!(e.at.unwrap().unix_ms(), 1_772_359_200_000);
}

#[test]
fn the_thread_settings_give_later_usage_its_service_tier() {
    let mut records = turn();
    records.insert(
        1,
        env(
            1,
            "event_msg",
            json!({"type": "thread_settings",
                   "thread_settings": {"model": "gpt-6.1-sol", "service_tier": "flex"}}),
        ),
    );
    let mut parser_state = ParserState::default();
    let step = run(&rollout(), &mut parser_state, &jsonl(&records), 0).unwrap();
    let readings = readings(&step);
    let priced: Vec<_> = readings.iter().collect();
    assert!(!priced.is_empty());
    assert!(
        priced
            .iter()
            .all(|m| m.usage.service_tier.as_deref() == Some("flex"))
    );
}

#[test]
fn rate_limits_read_as_windows_classified_by_length_not_name() {
    let q = json!({"limit_id": "codex", "plan_type": "pro", "credits": {"balance": "12.5"},
        "primary": {"used_percent": 42.0, "window_minutes": 10080, "resets_at": 1_790_600_000},
        "secondary": null});
    let r = quota(&q).unwrap();
    assert_eq!(
        (r.plan.as_deref(), r.credits.as_deref()),
        (Some("pro"), Some("12.5"))
    );
    assert_eq!(r.windows.len(), 1);
    let w = &r.windows[0];
    assert_eq!((w.limit.limit.as_ref(), w.limit.minutes), ("codex", 10_080));
    assert!(!w.limit.from_first_response);
    assert_eq!(w.used_percent, Some(42.0));
    assert_eq!(w.resets_at.map(|t| t.unix_ms()), Some(1_790_600_000_000));
    // Older rollouts give the reset in seconds after the record.
    let q =
        json!({"primary": {"used_percent": "1", "window_minutes": 300, "resets_in_seconds": 60}});
    let w = &quota(&q).unwrap().windows[0];
    assert_eq!(w.resets_at.map(|t| t.unix_ms()), None);
    assert_eq!(w.used_percent, Some(1.0));
}

#[test]
fn absent_quota_facts_remain_absent_and_extreme_resets_do_not_overflow() {
    assert!(quota(&json!({})).is_none());
    for field in ["resets_at", "resets_in_seconds"] {
        let q = json!({"primary": {"window_minutes": 300, field: i64::MAX}});
        let reading = quota(&q).unwrap();
        let window = &reading.windows[0];
        assert_eq!(window.used_percent, None);
        assert_eq!(window.resets_at, None);
        assert_eq!(reading.plan, None);
        assert_eq!(reading.credits, None);
    }
}

mod reports;

fn native_background_start() -> Vec<Json> {
    vec![
        meta(json!({})),
        env(
            2,
            "response_item",
            json!({"type":"function_call","name":"exec_command","arguments":"{\"cmd\":\"build\"}","call_id":"call_a"}),
        ),
        env(
            3,
            "response_item",
            json!({"type":"function_call_output","call_id":"call_a","output":"Chunk ID: x\nWall time: 1.0 seconds\nProcess running with session ID 4242\nOutput:\n"}),
        ),
    ]
}

#[test]
fn native_background_poll_preserves_event_precision_and_missing_clock() {
    use polylogue_model::{Precision, TimeEvidence, Timestamp};
    for (clock, precision) in [
        (Some("2026-03-01T10:00:05Z"), Some(Precision::Second)),
        (
            Some("2026-03-01T10:00:05.123456789Z"),
            Some(Precision::Instant),
        ),
        (None, None),
    ] {
        let mut records = native_background_start();
        records.push(env(4,"response_item",json!({"type":"function_call","name":"write_stdin","arguments":"{\"session_id\":4242,\"chars\":\"\"}","call_id":"call_b"})));
        let mut done = env(
            5,
            "response_item",
            json!({"type":"function_call_output","call_id":"call_b","output":"Chunk ID: y\nWall time: 5.0 seconds\nProcess exited with code 0\nOutput:\ndone"}),
        );
        if let Some(clock) = clock {
            done["timestamp"] = json!(clock);
        } else {
            done.as_object_mut().unwrap().remove("timestamp");
        }
        records.push(done);
        let step = run(&rollout(), &mut ParserState::default(), &jsonl(&records), 0).unwrap();
        let [update] = step.output.background.as_slice() else {
            panic!("{:?}", step.output.background)
        };
        assert_eq!(update.result_at, clock.and_then(Timestamp::parse_rfc3339));
        assert_eq!(
            update.result_at_evidence,
            precision.map_or(TimeEvidence::UNKNOWN, TimeEvidence::intrinsic)
        );
    }
}

#[test]
fn native_background_modern_and_legacy_completion_keep_their_clock_evidence() {
    use polylogue_model::{Precision, TimeEvidence};
    for payload in [
        json!({"type":"item_completed","item":{"type":"CommandExecution","id":"call_a","status":"completed","exit_code":0}}),
        json!({"type":"exec_command_end","call_id":"call_a","exit_code":0}),
    ] {
        let mut records = native_background_start();
        let mut done = env(4, "event_msg", payload);
        done["timestamp"] = json!("2026-03-01T10:00:04.123456789Z");
        records.push(done);
        let step = run(&rollout(), &mut ParserState::default(), &jsonl(&records), 0).unwrap();
        let [update] = step.output.background.as_slice() else {
            panic!("{:?}", step.output.background)
        };
        assert_eq!(update.result_at.unwrap().unix_ns(), 1772359204123456789);
        assert_eq!(
            update.result_at_evidence,
            TimeEvidence::intrinsic(Precision::Instant)
        );
    }
}

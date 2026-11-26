use super::*;
use crate::parse::{Step, run};
use polylogue_model::{ErrorReason, UnknownReason};
use serde_json::json;
use std::collections::BTreeMap;

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
    for step in &crate::chain_names::tests::settled(steps) {
        for row in &step.output.messages {
            rows.insert(row.offset, row.message.clone());
        }
    }
    rows.into_values().collect()
}

/// The events and links of steps, in a canonical order: the session facts
/// written when a pass ends move with where the pass ends.
fn facts(steps: &[&Step]) -> Vec<String> {
    let mut facts: Vec<String> = steps
        .iter()
        .flat_map(|step| {
            step.output
                .events
                .iter()
                .map(|event| serde_json::to_string(event).unwrap())
                .chain(
                    step.output
                        .links
                        .iter()
                        .map(|link| serde_json::to_string(link).unwrap()),
                )
        })
        .collect();
    facts.sort();
    facts
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

/// Rate limits that return to an earlier reading are a new report at their
/// own time: A, B, A at three times are three reports, in order.
#[test]
fn a_reading_that_returns_is_a_new_report_in_its_place() {
    let limits = |used: f64| {
        json!({"type": "token_count", "info": null,
               "rate_limits": {"primary": {"used_percent": used, "window_minutes": 300}}})
    };
    let records = [
        meta(json!({})),
        env(2, "event_msg", limits(12.0)),
        env(3, "event_msg", limits(40.0)),
        env(4, "event_msg", limits(12.0)),
    ];
    let step = run(&rollout(), &mut ParserState::default(), &jsonl(&records), 0).unwrap();
    let reports: Vec<(Option<i64>, Json)> = step
        .output
        .events
        .iter()
        .filter(|e| e.kind == "quota_report")
        .map(|e| (e.at.map(|at| at.unix_ms()), e.data["reading"].clone()))
        .collect();
    let at = |second: i64| Some(1_772_359_200_000 + second * 1_000);
    assert_eq!(
        reports.iter().map(|(at, _)| *at).collect::<Vec<_>>(),
        [at(2), at(3), at(4)]
    );
    assert_eq!(reports[0].1, reports[2].1);
    assert_ne!(reports[0].1, reports[1].1);
    let ids: std::collections::BTreeSet<&str> = step
        .output
        .events
        .iter()
        .filter(|e| e.kind == "quota_report")
        .filter_map(|e| e.data["report_id"].as_str())
        .collect();
    assert_eq!(ids.len(), 3);
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
    let whole_step = run(&rollout(), &mut whole_carry, &input, 0).unwrap();
    let whole_facts = facts(&[&whole_step]);
    let whole = final_messages(&[whole_step]);
    let mut cut = 0;
    while cut < input.len() {
        let mut parser_state = ParserState::default();
        let first = run(&rollout(), &mut parser_state, &input[..cut], 0).unwrap();
        let mut parser_state: ParserState =
            serde_json::from_value(serde_json::to_value(&parser_state).unwrap()).unwrap();
        let used = first.consumed as usize;
        let second = run(&rollout(), &mut parser_state, &input[used..], used as u64).unwrap();
        assert_eq!(facts(&[&first, &second]), whole_facts, "cut at {cut}");
        assert_eq!(final_messages(&[first, second]), whole, "cut at {cut}");
        cut += 37;
    }
}

#[test]
fn a_command_left_running_is_updated_when_it_ends() {
    let mut records = vec![
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
    // A later poll cannot report the completed original call again.
    records.extend([
        env(
            6,
            "response_item",
            json!({"type":"function_call","name":"write_stdin",
            "arguments":"{\"session_id\":4242,\"chars\":\"\"}","call_id":"call_c"}),
        ),
        env(
            7,
            "response_item",
            json!({"type":"function_call_output","call_id":"call_c",
            "output":"Process exited with code 1\nOutput:\nerror"}),
        ),
    ]);
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
    assert_eq!(
        step.output.prefix_copies,
        [polylogue_model::PrefixCopy {
            parent: SessionId::parse(&format!("codex:{PARENT}")).unwrap(),
            child: SessionId::parse(&format!("codex:{THREAD}")).unwrap(),
            explicit_members: false,
        }]
    );
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
    assert_eq!(
        spawn.branch_point.as_ref().unwrap().native().as_str(),
        "msg_p2"
    );
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

/// An agent reports many activity items; its spawn is one statement,
/// stated by the first of them, however the bytes are split.
#[test]
fn a_subagents_activity_states_its_spawn_once() {
    let child = "019a0000-0000-7000-8000-0000000000cc";
    let activity = |ordinal: u64, id: &str| {
        env(
            ordinal,
            "event_msg",
            json!({"type": "item_completed", "thread_id": THREAD, "turn_id": "t1",
            "item": {"type": "SubAgentActivity", "id": id, "agent_thread_id": child},
            "completed_at_ms": 1}),
        )
    };
    let records = vec![
        meta(json!({})),
        activity(2, "act_1"),
        activity(3, "act_2"),
        activity(4, "act_3"),
    ];
    let spawns = |steps: &[Step]| -> Vec<(String, Option<String>)> {
        steps
            .iter()
            .flat_map(|step| &step.output.links)
            .filter(|link| link.kind == LinkKind::Spawn)
            .map(|link| {
                (
                    link.child.native().as_str().to_owned(),
                    link.call_id.clone(),
                )
            })
            .collect()
    };
    let expected = vec![(child.to_owned(), Some("act_1".to_owned()))];
    let mut parser_state = ParserState::default();
    let whole = run(&rollout(), &mut parser_state, &jsonl(&records), 0).unwrap();
    assert_eq!(spawns(&[whole]), expected);
    let bytes = jsonl(&records);
    let split = bytes.iter().position(|b| *b == b'\n').unwrap() + 1;
    let split = split + bytes[split..].iter().position(|b| *b == b'\n').unwrap() + 1;
    let mut parser_state = ParserState::default();
    let first = run(&rollout(), &mut parser_state, &bytes[..split], 0).unwrap();
    let parser_state_json = serde_json::to_string(&parser_state).unwrap();
    let mut parser_state: ParserState = serde_json::from_str(&parser_state_json).unwrap();
    let rest = run(&rollout(), &mut parser_state, &bytes[split..], split as u64).unwrap();
    assert_eq!(spawns(&[first, rest]), expected);
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
    records.push(json!({"type": "brand_new_record_type", "payload": {"x": 1}}));
    records.push(json!([1, 2]));
    let mut parser_state = ParserState::default();
    let step = run(&rollout(), &mut parser_state, &jsonl(&records), 0).unwrap();
    assert_eq!(step.counts.counted(), step.counts.records);
    assert_eq!(
        step.counts
            .skipped
            .get("brand_new_record_type: unrecognized record type"),
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
    assert_eq!(readings.len(), 2);
    assert!(
        readings
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

fn background_start_records() -> Vec<Json> {
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
fn background_poll_keeps_precision_and_missing_clock() {
    use polylogue_model::{Precision, TimeAccuracy, Timestamp};
    for (clock, precision) in [
        (Some("2026-03-01T10:00:05Z"), Some(Precision::Second)),
        (
            Some("2026-03-01T10:00:05.123456789Z"),
            Some(Precision::Instant),
        ),
        (None, None),
    ] {
        let mut records = background_start_records();
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
            precision.map_or(TimeAccuracy::UNKNOWN, TimeAccuracy::intrinsic)
        );
    }
}

#[test]
fn background_completion_keeps_its_time_accuracy() {
    use polylogue_model::{Precision, TimeAccuracy};
    for payload in [
        json!({"type":"item_completed","item":{"type":"CommandExecution","id":"call_a","status":"completed","exit_code":0}}),
        json!({"type":"exec_command_end","call_id":"call_a","exit_code":0}),
    ] {
        let mut records = background_start_records();
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
            TimeAccuracy::intrinsic(Precision::Instant)
        );
    }
}

#[test]
fn resumed_rollout_records_the_provider_session_relation() {
    let records = [meta(json!({})), meta(json!({"id": PARENT}))];
    let mut state = ParserState::default();
    let out = run(&rollout(), &mut state, &jsonl(&records), 0)
        .unwrap()
        .output;
    let [link] = out.links.as_slice() else {
        panic!("{:?}", out.links)
    };
    assert_eq!(link.kind, LinkKind::Resume);
    assert_eq!(link.parent.native().as_str(), PARENT);
    assert_eq!(link.child.native().as_str(), THREAD);
    assert!(!link.copies_prefix);
}

#[test]
fn copied_history_branch_point_survives_split_appends() {
    let fork = "fork-source";
    let records = [
        meta(json!({"thread_source":"subagent","parent_thread_id":PARENT,
            "forked_from_id":fork,"subagent_history_start_ordinal":4})),
        env(
            2,
            "response_item",
            json!({"type":"message","id":"copied-first","role":"user",
            "content":[{"type":"input_text","text":"first copied context"}]}),
        ),
        env(
            3,
            "response_item",
            json!({"type":"message","id":"copied-last","role":"assistant",
            "content":[{"type":"output_text","text":"last copied context"}]}),
        ),
        user(4, "own task"),
    ];
    let input = jsonl(&records);
    for cut in [
        0,
        jsonl(&records[..1]).len(),
        jsonl(&records[..2]).len(),
        jsonl(&records[..3]).len(),
    ] {
        let mut state = ParserState::default();
        let first = run(&rollout(), &mut state, &input[..cut], 0).unwrap();
        assert!(!first.output.links.iter().any(|link| {
            link.branch_point
                .as_ref()
                .is_some_and(|id| id.native().as_str() == "copied-first")
        }));
        let mut state: ParserState =
            serde_json::from_value(serde_json::to_value(state).unwrap()).unwrap();
        let second = run(&rollout(), &mut state, &input[cut..], cut as u64).unwrap();
        let links: Vec<_> = first
            .output
            .links
            .iter()
            .chain(&second.output.links)
            .collect();
        let with_branch_point: Vec<_> = links
            .iter()
            .filter(|link| link.branch_point.is_some())
            .collect();
        assert_eq!(with_branch_point.len(), 1, "cut {cut}");
        assert_eq!(with_branch_point[0].kind, LinkKind::Fork);
        assert_eq!(with_branch_point[0].parent.native().as_str(), fork);
        assert_eq!(
            with_branch_point[0]
                .branch_point
                .as_ref()
                .unwrap()
                .native()
                .as_str(),
            "copied-last"
        );
        assert!(with_branch_point[0].copies_prefix);
        assert!(
            links
                .iter()
                .filter(|link| link.kind == LinkKind::Spawn)
                .all(|link| link.branch_point.is_none() && !link.copies_prefix)
        );
        assert_eq!(second.output.sessions[0].kind, SessionKind::Subagent);
        assert!(
            run(&rollout(), &mut state, b"", input.len() as u64)
                .unwrap()
                .output
                .links
                .is_empty()
        );
    }
}

#[test]
fn verified_answers_keep_their_payload_and_actual_call_across_appends() {
    for native_id in [None, Some("question-message")] {
        let mut call = json!({"type":"function_call", "name":"request_user_input",
            "call_id":"question-call", "arguments":"{\"questions\":[{\"question\":\"Which colour?\"}]}"});
        if let Some(native_id) = native_id {
            call["id"] = json!(native_id);
            call["namespace"] = json!("functions");
        }
        let first_bytes = jsonl(&[meta(json!({})), env(2, "response_item", call)]);
        let payload = json!({"type":"verified_answer", "call_id":"question-call", "turn_id":"t1",
            "acceptance_order":2, "questions":[{"question":"Which colour?", "answer":"Blue",
                "future_answer_detail":{"value":7}}], "future_field":["retained"]});
        let second_bytes = jsonl(&[
            user(3, "An intervening message is not the answer's call."),
            env(4, "retained_context", payload.clone()),
        ]);
        let mut state = ParserState::default();
        let first = run(&rollout(), &mut state, &first_bytes, 0).unwrap();
        let actual_call = first.output.messages[0].message.id.clone();
        let mut state: ParserState =
            serde_json::from_value(serde_json::to_value(state).unwrap()).unwrap();
        let second = run(&rollout(), &mut state, &second_bytes, first.consumed).unwrap();
        let answer = second
            .output
            .events
            .iter()
            .find(|e| e.kind == "verified_answer")
            .unwrap();
        assert_eq!(answer.data, payload);
        assert_eq!(answer.after, Some(actual_call));
        assert_eq!(answer.session, session_id(THREAD));
        let written = serde_json::to_value(answer).unwrap();
        assert_eq!(written["at"], "2026-03-01T10:00:04Z");
        assert_eq!(written["at_quality"], "intrinsic");
        assert_eq!(written["at_precision"], "instant");
        assert_eq!(second.output.messages.len(), 1);
        assert_eq!(second.counts.used["retained_context"], 1);
        // The record starts after the intervening message's line and ends
        // before its own line break.
        let intervening =
            jsonl(&[user(3, "An intervening message is not the answer's call.")]).len();
        assert_eq!(
            answer.range,
            Some((
                (first_bytes.len() + intervening) as u64,
                (first_bytes.len() + second_bytes.len() - 1) as u64
            ))
        );
    }
}

#[test]
fn realtime_markers_keep_their_time_and_payload_without_a_message_anchor() {
    let started = json!({"type":"realtime_session_started", "id":"marker-start",
        "realtime_session_id":"realtime-one", "future_field":{"value":1}});
    let closed = json!({"type":"realtime_session_closed", "id":"marker-close",
        "realtime_session_id":"realtime-one", "outcome":"ended",
        "timestamp":"2026-03-01T10:00:09.125Z"});
    let mut closing = env(4, "realtime_item", closed.clone());
    closing["timestamp"] = Json::Null;
    let step = run(
        &rollout(),
        &mut ParserState::default(),
        &jsonl(&[
            meta(json!({})),
            user(2, "A message before the realtime activity."),
            env(3, "realtime_item", started.clone()),
            closing,
        ]),
        0,
    )
    .unwrap();
    let events: Vec<_> = step
        .output
        .events
        .iter()
        .filter(|e| e.kind.starts_with("realtime_session_"))
        .collect();
    assert_eq!(events.len(), 2);
    for (event, payload, time) in [
        (events[0], started, "2026-03-01T10:00:03Z"),
        (events[1], closed, "2026-03-01T10:00:09.125Z"),
    ] {
        assert_eq!(event.data, payload);
        assert_eq!(event.after, None);
        assert_eq!(event.session, session_id(THREAD));
        assert_eq!(serde_json::to_value(event).unwrap()["at"], time);
    }
    assert_eq!(step.output.messages.len(), 1);
    assert_eq!(step.counts.used["realtime_item"], 2);
}

#[test]
fn verified_answers_with_no_known_call_keep_an_unknown_anchor_and_time() {
    let payload = json!({"type":"verified_answer", "call_id":"unseen-call", "questions":[]});
    let mut answer = env(3, "retained_context", payload.clone());
    answer["timestamp"] = Json::Null;
    let step = run(
        &rollout(),
        &mut ParserState::default(),
        &jsonl(&[
            meta(json!({})),
            user(2, "An unrelated previous message."),
            answer,
        ]),
        0,
    )
    .unwrap();
    let answer = step
        .output
        .events
        .iter()
        .find(|e| e.kind == "verified_answer")
        .unwrap();
    assert_eq!(answer.data, payload);
    assert_eq!(answer.after, None);
    assert_eq!(answer.at, None);
    assert_eq!(
        serde_json::to_value(answer).unwrap()["at_quality"],
        "unknown"
    );
}

#[test]
fn unknown_retained_and_realtime_payloads_name_what_could_not_be_read() {
    let records = [
        env(1, "retained_context", json!({"type":"future_context"})),
        env(2, "realtime_item", json!({"type":"future_realtime"})),
        env(3, "retained_context", json!({})),
        env(4, "realtime_item", Json::Null),
        env(
            5,
            "unknown_context",
            json!({"type":"verified_answer", "call_id":"unseen"}),
        ),
    ];
    let step = run(&rollout(), &mut ParserState::default(), &jsonl(&records), 0).unwrap();
    assert_eq!(
        step.counts.skipped,
        BTreeMap::from([
            (
                "retained_context: retained_context payload type future_context has no reader"
                    .into(),
                1
            ),
            (
                "realtime_item: realtime_item payload type future_realtime has no reader".into(),
                1
            ),
            (
                "retained_context: retained_context payload type missing has no reader".into(),
                1
            ),
            (
                "realtime_item: realtime_item payload type missing has no reader".into(),
                1
            ),
            ("unknown_context: unrecognized record type".into(), 1),
        ])
    );
    assert_eq!(step.counts.records, 5);
}

#[test]
fn a_rollout_is_named_by_its_header_in_either_shape() {
    let bare = format!(r#"{{"id":"{THREAD}","timestamp":"2025-08-11T16:59:28.000Z"}}"#);
    assert_eq!(
        rollout_id(bare.as_bytes()).unwrap().as_deref(),
        Some(THREAD)
    );
    let meta = format!(r#"{{"type":"session_meta","payload":{{"id":"{THREAD}"}}}}"#);
    assert_eq!(
        rollout_id(meta.as_bytes()).unwrap().as_deref(),
        Some(THREAD)
    );
    let item = r#"{"type":"message","id":"m1","role":"user"}"#;
    assert_eq!(rollout_id(item.as_bytes()).unwrap(), None);
}

#[test]
fn a_spawned_threads_nickname_in_its_spawn_source_titles_it() {
    // Codex writes the spawn as `source.subagent.thread_spawn`, naming the
    // parent thread and the agent's nickname.
    let records = [meta(json!({"source": {"subagent": {"thread_spawn": {
        "parent_thread_id": PARENT, "depth": 1, "agent_nickname": "Volta", "agent_role": "explorer"}}}}))];
    let out = run(&rollout(), &mut ParserState::default(), &jsonl(&records), 0)
        .unwrap()
        .output;
    let session = out
        .sessions
        .iter()
        .find(|s| s.id.native().as_str() == THREAD)
        .unwrap();
    assert_eq!(session.kind, SessionKind::Subagent);
    let title = session.title.as_ref().unwrap();
    assert_eq!(
        (title.text.as_str(), title.source),
        ("Volta", TitleSource::AgentName)
    );
    assert!(out.links.iter().any(|l| l.kind == LinkKind::Spawn
        && l.parent.native().as_str() == PARENT
        && l.child.native().as_str() == THREAD));
}

/// Encrypted reasoning and a compaction's restated history are named where
/// the record holds them, so the index need not copy them; the place given
/// reads back as the value parsed, for enveloped and 2025 bare items alike.
#[test]
fn encrypted_reasoning_and_restated_history_are_named_in_their_records() {
    let pointed = |bytes: &[u8], field: &polylogue_model::StoredField| -> Json {
        let record: Json =
            serde_json::from_slice(bytes[field.start as usize..field.end as usize].trim_ascii())
                .unwrap();
        record.pointer(&field.pointer).unwrap().clone()
    };
    let mut records = turn();
    records.push(env(
        30,
        "compacted",
        json!({"message": "", "replacement_history": [
            {"type": "message", "id": "msg_p1", "role": "user",
             "content": [{"type": "input_text", "text": "restated prompt"}]}]}),
    ));
    let bytes = jsonl(&records);
    let step = run(&rollout(), &mut ParserState::default(), &bytes, 0).unwrap();
    let reasoning = step
        .output
        .messages
        .iter()
        .position(|row| matches!(row.message.blocks[0], Block::Thinking { .. }))
        .unwrap();
    let fields = &step.output.stored_fields[&reasoning];
    assert_eq!(fields.len(), 1);
    assert_eq!(
        (fields[0].block, fields[0].field.as_str()),
        (Some(0), "/signature")
    );
    assert_eq!(pointed(&bytes, &fields[0]), json!("gAAA"));
    let compaction = step
        .output
        .events
        .iter()
        .position(|event| event.kind == "compaction")
        .unwrap();
    let fields = &step.output.stored_event_fields[&compaction];
    assert_eq!(
        (fields[0].block, fields[0].field.as_str()),
        (None, "/replacement_history")
    );
    assert_eq!(
        pointed(&bytes, &fields[0]),
        step.output.events[compaction].data["replacement_history"]
    );
    // A 2025 rollout's items are bare.
    let bare = jsonl(&[
        json!({"id": THREAD, "timestamp": "2025-08-11T16:59:28.000Z", "instructions": null}),
        json!({"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gBBB"}),
    ]);
    let step = run(&rollout(), &mut ParserState::default(), &bare, 0).unwrap();
    let fields = &step.output.stored_fields[&0];
    assert_eq!(fields[0].pointer, "/encrypted_content");
    assert_eq!(pointed(&bare, &fields[0]), json!("gBBB"));
}

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
    let mut carry = Carry::default();
    let step = run(&rollout(), &mut carry, &jsonl(&turn()), 0).unwrap();
    assert!(
        step.accounting.refused.is_empty(),
        "{:?}",
        step.accounting.refused
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

#[test]
fn usage_lands_on_the_item_before_the_count_with_each_token_counted_once() {
    let mut carry = Carry::default();
    let step = run(&rollout(), &mut carry, &jsonl(&turn()), 0).unwrap();
    let msgs = final_messages(&[step]);
    let call = &msgs[3];
    assert_eq!(
        call.usage,
        Usage {
            input: Some(400),
            cached_input: Some(600),
            cache_write: None,
            output: Some(30),
            reasoning: Some(20),
            service_tier: None,
        }
    );
    assert_eq!(msgs[5].usage.input, Some(200));
    assert!(msgs[2].usage.is_empty());
}

#[test]
fn rate_limits_are_stored_when_they_change() {
    let mut carry = Carry::default();
    let step = run(&rollout(), &mut carry, &jsonl(&turn()), 0).unwrap();
    let kinds: Vec<_> = step.output.events.iter().map(|e| e.kind.as_str()).collect();
    assert_eq!(kinds, ["turn_context", "rate_limits", "task_complete"]);
}

#[test]
fn the_thread_item_gives_the_call_its_exit_code_and_duration() {
    let mut carry = Carry::default();
    let step = run(&rollout(), &mut carry, &jsonl(&turn()), 0).unwrap();
    let msgs = final_messages(&[step]);
    let Block::ToolCall { command, tool, .. } = &msgs[3].blocks[0] else {
        panic!("{:?}", msgs[3].blocks)
    };
    assert_eq!(command.as_deref(), Some("cargo test"));
    assert_eq!(tool.as_deref(), Some("exec_command"));
    let Block::ToolResult { outcome, data, .. } = &msgs[4].blocks[0] else {
        panic!()
    };
    assert_eq!(
        *outcome,
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(101))
        }
    );
    assert_eq!(data.as_ref().unwrap()["duration_ms"], 2000);
}

#[test]
fn parsing_in_pieces_equals_parsing_at_once() {
    let input = jsonl(&turn());
    let mut whole_carry = Carry::default();
    let whole = final_messages(&[run(&rollout(), &mut whole_carry, &input, 0).unwrap()]);
    let mut cut = 0;
    while cut < input.len() {
        let mut carry = Carry::default();
        let first = run(&rollout(), &mut carry, &input[..cut], 0).unwrap();
        // The carry crosses a restart as JSON.
        let mut carry: Carry =
            serde_json::from_value(serde_json::to_value(&carry).unwrap()).unwrap();
        let used = first.consumed as usize;
        let second = run(&rollout(), &mut carry, &input[used..], used as u64).unwrap();
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
    let mut carry = Carry::default();
    let step = run(&rollout(), &mut carry, &jsonl(&records), 0).unwrap();
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
    assert!(carry.running.is_empty());
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
    let mut carry = Carry::default();
    let step = run(&rollout(), &mut carry, &jsonl(&records), 0).unwrap();
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
    let mut carry = Carry::default();
    let step = run(&rollout(), &mut carry, &jsonl(&records), 0).unwrap();
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
    assert_eq!(step.accounting.unknown.values().sum::<u64>(), 2);
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
    let mut carry = Carry::default();
    let step = run(&rollout(), &mut carry, &jsonl(&records), 0).unwrap();
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
    let mut carry = Carry::default();
    let step = run(&rollout(), &mut carry, &jsonl(&records), 0).unwrap();
    assert!(step.accounting.refused.is_empty(), "{:?}", step.accounting);
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
    let mut carry = Carry::default();
    let step = run(&rollout(), &mut carry, &jsonl(&records), 0).unwrap();
    assert_eq!(step.accounting.accounted(), step.accounting.framed);
    assert_eq!(
        step.accounting
            .unknown
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
    let mut carry = index::Names::default();
    let step = run(&index::SessionIndex, &mut carry, &names, 0).unwrap();
    assert_eq!(step.output.sessions.len(), 1);
    assert_eq!(
        step.output.sessions[0].title.as_ref().unwrap().text,
        "renamed"
    );
    let history =
        jsonl(&[json!({"session_id": THREAD, "ts": 1_772_359_200, "text": "run the tests"})]);
    let mut carry = index::NoCarry::default();
    let step = run(&index::History, &mut carry, &history, 0).unwrap();
    let [e] = step.output.events.as_slice() else {
        panic!()
    };
    assert_eq!(e.kind, "prompt");
    assert_eq!(e.at.unwrap().unix_ms(), 1_772_359_200_000);
}

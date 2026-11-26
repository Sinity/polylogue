use super::*;
use crate::parse::run;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};

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

/// Settles a run of steps' chain-local names and applies them the way the
/// writer does for one input: a row at the same offset replaces the earlier
/// one, and of several versions of one message the earliest in the input is
/// shown.
fn final_messages(steps: &[crate::parse::Step]) -> Vec<Message> {
    let mut rows: BTreeMap<u64, Message> = BTreeMap::new();
    for step in &crate::chain_names::tests::settled(steps) {
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
    let mut parser_state = ParserState::default();
    let step = run(
        &main_file("s1"),
        &mut parser_state,
        &jsonl(&conversation()),
        0,
    )
    .unwrap();
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
fn parsing_in_appended_bytes_equals_parsing_at_once() {
    let bytes = jsonl(&conversation());
    let whole = {
        let mut parser_state = ParserState::default();
        final_messages(&[run(&main_file("s1"), &mut parser_state, &bytes, 0).unwrap()])
    };
    for cut in 0..=bytes.len() {
        let parser = main_file("s1");
        let mut parser_state = ParserState::default();
        let first = run(&parser, &mut parser_state, &bytes[..cut], 0).unwrap();
        let mut parser_state: ParserState =
            serde_json::from_value(serde_json::to_value(&parser_state).unwrap()).unwrap();
        let used = first.consumed as usize;
        let second = run(&parser, &mut parser_state, &bytes[used..], used as u64).unwrap();
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
    let mut parser_state = ParserState::default();
    let step = run(&main_file("s1"), &mut parser_state, &jsonl(&records), 0).unwrap();
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
    let mut parser_state = ParserState::default();
    let step = run(&main_file("s1"), &mut parser_state, &jsonl(&records), 0).unwrap();
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
    let mut parser_state = ParserState::default();
    let step = run(&main_file("s2"), &mut parser_state, &jsonl(&records), 0).unwrap();
    let out = &step.output;
    assert_eq!(out.messages[0].message.session.native().as_str(), "s1");
    assert_eq!(out.messages[1].message.session.native().as_str(), "s2");
    assert!(out.links.is_empty());
    assert_eq!(
        out.prefix_copies,
        [polylogue_model::PrefixCopy {
            explicit_members: false,
            parent: session_id("s1").unwrap(),
            child: session_id("s2").unwrap(),
        }]
    );
}

#[test]
fn subagent_links_its_carrier_only_by_sidechain_flag() {
    let sub = ClaudeCode {
        file: File::Subagent {
            stem: "agent-x1".into(),
        },
    };
    let records = vec![
        json!({"type":"user","uuid":"u","sessionId":"s1","agentId":"x1","isSidechain":true,
                              "message":{"role":"user","content":"do the subtask"}}),
    ];
    let mut parser_state = ParserState::default();
    let out = run(&sub, &mut parser_state, &jsonl(&records), 0)
        .unwrap()
        .output;
    assert_eq!(
        out.messages[0].message.session.native().as_str(),
        "agent-x1"
    );
    assert_eq!(out.sessions[0].kind, SessionKind::Subagent);
    assert!(out.links.is_empty());
    assert_eq!(out.messages[0].message.prompt_kind, Some(PromptKind::Model));

    let records = vec![
        json!({"type":"user","uuid":"u","sessionId":"s1","isSidechain":true,
                              "message":{"role":"user","content":"side"}}),
    ];
    let mut parser_state = ParserState::default();
    let out = run(&main_file("s1"), &mut parser_state, &jsonl(&records), 0)
        .unwrap()
        .output;
    assert_eq!(
        out.messages[0].message.session.native().as_str(),
        "s1~side-u"
    );
    assert_eq!(out.links[0].kind, LinkKind::Spawn);
}

#[test]
fn an_agent_calls_result_links_the_agent_it_started() {
    let records = vec![json!({"type":"user","uuid":"r1","sessionId":"s1",
               "toolUseResult":{"isAsync":true,"status":"async_launched","agentId":"a9"},
               "message":{"role":"user","content":[
                   {"type":"tool_result","tool_use_id":"toolu_9","content":"launched"}]}})];
    let mut parser_state = ParserState::default();
    let out = run(&main_file("s1"), &mut parser_state, &jsonl(&records), 0)
        .unwrap()
        .output;
    let l = &out.links[0];
    assert_eq!(l.kind, LinkKind::Spawn);
    assert_eq!(l.parent.native().as_str(), "s1");
    assert_eq!(l.child.native().as_str(), "agent-a9");
    assert_eq!(l.call_id.as_deref(), Some("toolu_9"));
    assert_eq!(l.evidence, "toolUseResult.agentId");
}

#[test]
fn titles_follow_precedence() {
    let records = vec![
        json!({"type":"custom-title","customTitle":"Mine","sessionId":"s1"}),
        json!({"type":"ai-title","aiTitle":"Generated","sessionId":"s1"}),
        json!({"type":"summary","summary":"Summary","leafUuid":"x"}),
    ];
    let mut parser_state = ParserState::default();
    let out = run(&main_file("s1"), &mut parser_state, &jsonl(&records), 0)
        .unwrap()
        .output;
    assert_eq!(out.sessions[0].title.as_ref().unwrap().text, "Mine");
}

#[test]
fn background_completion_updates_the_outcome() {
    let record = json!({"type":"queue-operation","operation":"enqueue","sessionId":"s1",
        "content":"<task-notification><task-id>bg7</task-id><status>failed</status></task-notification>"});
    let mut parser_state = ParserState::default();
    let out = run(&main_file("s1"), &mut parser_state, &jsonl(&[record]), 0)
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
    let mut parser_state = ParserState::default();
    let out = run(&main_file("s1"), &mut parser_state, &jsonl(&[record]), 0)
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
    let mut parser_state = ParserState::default();
    let step = run(&main_file("s1"), &mut parser_state, &bytes, 0).unwrap();
    let acc = &step.counts;
    assert_eq!(acc.records, 4);
    assert_eq!(acc.counted(), 4);
    assert_eq!(acc.used["user"], 1);
    assert!(
        acc.skipped
            .contains_key("mystery: unrecognized record type")
    );
    assert_eq!(acc.unreadable.len(), 1);
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
    let step = run(&main_file("s1"), &mut ParserState::default(), &bytes, 0).unwrap();
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
                            "cache_creation":{"ephemeral_5m_input_tokens":300,"ephemeral_1h_input_tokens":600},
                            "service_tier":"standard","speed":"fast"}}});
    let step = run(&main_file("s1"), &mut Default::default(), &jsonl(&[rec]), 0).unwrap();
    let u = &final_messages(&[step])[0].usage;
    assert_eq!(u.cache_write, Some(900));
    assert_eq!(u.cache_write_1h, Some(600));
    assert_eq!(u.service_tier.as_deref(), Some("standard"));
    assert_eq!(u.speed.as_deref(), Some("fast"));
}

fn usage_of(usage: Json) -> polylogue_model::Usage {
    let rec = json!({"type":"assistant","uuid":"a1","sessionId":"s1","timestamp":"2026-01-01T00:00:00Z",
        "message":{"id":"msg_1","role":"assistant","model":"claude-opus-5-5",
                   "content":[{"type":"text","text":"ok"}],"usage":usage}});
    let step = run(&main_file("s1"), &mut Default::default(), &jsonl(&[rec]), 0).unwrap();
    final_messages(&[step])[0].usage.clone()
}

#[test]
fn thinking_is_inside_output_tokens() {
    let u = usage_of(json!({"output_tokens":305,"output_tokens_details":{"thinking_tokens":100}}));
    assert_eq!(u.output, Some(205));
    assert_eq!(u.reasoning, Some(100));
    let u = usage_of(json!({"output_tokens":5}));
    assert_eq!((u.output, u.reasoning), (Some(5), None));
}

#[test]
fn thinking_above_output_leaves_output_unknown() {
    let u = usage_of(json!({"output_tokens":10,"output_tokens_details":{"thinking_tokens":11}}));
    assert_eq!(u.output, None);
    assert_eq!(u.reasoning, Some(11));
}

#[test]
fn background_notice_keeps_clock_precision_or_unknown() {
    use polylogue_model::{Precision, TimeAccuracy, Timestamp};
    for (clock, precision) in [
        (Some("2026-03-01T10:00:05Z"), Some(Precision::Second)),
        (
            Some("2026-03-01T10:00:05.123456789Z"),
            Some(Precision::Instant),
        ),
        (None, None),
    ] {
        let mut record = json!({"type":"queue-operation","sessionId":"s1","content":"<task-notification><task-id>bg7</task-id><status>completed</status></task-notification>"});
        if let Some(clock) = clock {
            record["timestamp"] = json!(clock);
        }
        let step = run(
            &main_file("s1"),
            &mut ParserState::default(),
            &jsonl(&[record]),
            0,
        )
        .unwrap();
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

mod hook_fields;

#[test]
fn a_response_joins_its_records_across_any_records_and_steps_between() {
    let mut records = vec![assistant(
        "a1",
        "response-a",
        json!({"type":"text","text":"first"}),
    )];
    for name in ["b", "c", "d", "e", "f"] {
        records.push(assistant(name, name, json!({"type":"text","text":name})));
    }
    records.push(assistant(
        "a2",
        "response-a",
        json!({"type":"text","text":"last"}),
    ));
    let bytes = jsonl(&records);
    let mut state = ParserState::default();
    let whole = final_messages(&[run(&main_file("s1"), &mut state, &bytes, 0).unwrap()]);
    assert_eq!(whole.len(), 6);
    assert_eq!(
        whole
            .iter()
            .find(|m| m.id.native().as_str() == "a1")
            .unwrap()
            .text(),
        "first\n\nlast"
    );
    let parser = Parse::Lines(Box::new(main_file("s1")));
    for cut in 0..=bytes.len() {
        let (first, state) = parser.step(None, &bytes[..cut], 0).unwrap();
        let used = first.consumed as usize;
        let state = serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
        let (second, _) = parser
            .step(Some(&state), &bytes[used..], used as u64)
            .unwrap();
        assert_eq!(final_messages(&[first, second]), whole, "cut {cut}");
    }
}

#[test]
fn a_child_resolves_its_parent_through_aliases_however_far_back() {
    let mut records = vec![
        assistant("a1", "response-a", json!({"type":"text","text":"first"})),
        assistant("a2", "response-a", json!({"type":"text","text":"last"})),
    ];
    for n in 0..=5000 {
        records.push(json!({"type":"progress","uuid":format!("progress-{n}"),"parentUuid":"a1","sessionId":"s1","data":{"type":"bash_progress"}}));
    }
    records.push(json!({"type":"user","uuid":"child","parentUuid":"a2","sessionId":"s1","message":{"role":"user","content":"follow-up"}}));
    let mut state = ParserState::default();
    let messages =
        final_messages(&[run(&main_file("s1"), &mut state, &jsonl(&records), 0).unwrap()]);
    assert_eq!(
        messages
            .iter()
            .find(|m| m.id.native().as_str() == "child")
            .unwrap()
            .parent,
        message_id("a1")
    );
}

#[test]
fn explicit_resume_and_distinct_fork_points_survive_source_deduplication() {
    let records = vec![
        json!({"type":"user","uuid":"a","sessionId":"child","resumeSessionId":"prior",
            "forkedFrom":{"sessionId":"parent","messageUuid":"first"},
            "message":{"role":"user","content":"first branch observation"}}),
        json!({"type":"user","uuid":"b","sessionId":"child",
            "forkedFrom":{"sessionId":"parent","messageUuid":"second"},
            "message":{"role":"user","content":"second branch observation"}}),
    ];
    let mut state = ParserState::default();
    let out = run(&main_file("child"), &mut state, &jsonl(&records), 0)
        .unwrap()
        .output;
    let resume = out
        .links
        .iter()
        .find(|l| l.kind == LinkKind::Resume)
        .unwrap();
    assert_eq!(resume.parent.native().as_str(), "prior");
    assert_eq!(resume.child.native().as_str(), "child");
    assert!(!resume.copies_prefix);
    let branch_points: Vec<_> = out
        .links
        .iter()
        .filter(|l| l.kind == LinkKind::Fork)
        .map(|l| l.branch_point.as_ref().unwrap().native().as_str())
        .collect();
    assert_eq!(branch_points, ["first", "second"]);
}

#[test]
fn inline_compaction_keeps_one_summary_message_and_normalized_kept_coordinates() {
    let records = [
        json!({"type":"user","uuid":"first","sessionId":"s1","message":{"content":"original"}}),
        json!({"type":"system","subtype":"compact_boundary","uuid":"boundary","sessionId":"s1",
            "logicalParentUuid":"first","compactMetadata":{"trigger":"auto","preTokens":182000,"postTokens":14500,
            "preservedSegment":{"headUuid":"first","tailUuid":"first","anchorUuid":"summary"}}}),
        json!({"type":"user","uuid":"summary","parentUuid":"boundary","sessionId":"s1","isCompactSummary":true,
            "message":{"content":"Known compressed context."}}),
    ];
    let out = crate::chain_names::tests::settled(&[run(
        &main_file("s1"),
        &mut ParserState::default(),
        &jsonl(&records),
        0,
    )
    .unwrap()])
    .remove(0)
    .output;
    assert_eq!(
        out.messages
            .iter()
            .map(|row| row.message.id.to_string())
            .collect::<Vec<_>>(),
        ["claude-code:first", "claude-code:summary"]
    );
    let summary = &out.messages[1].message;
    assert_eq!(summary.text(), "Known compressed context.");
    assert_eq!(summary.prompt_kind, Some(PromptKind::CompactionSummary));
    assert_eq!(
        polylogue_model::WrittenBy::of(summary),
        polylogue_model::WrittenBy::Harness
    );
    assert_eq!(
        summary.parent.as_ref().unwrap().to_string(),
        "claude-code:first"
    );
    let events: Vec<_> = out
        .events
        .iter()
        .filter(|event| event.kind == "compaction")
        .collect();
    assert_eq!(events.len(), 1);
    let boundary: polylogue_model::CompactionBoundary =
        serde_json::from_value(events[0].data.clone()).unwrap();
    assert_eq!(
        (
            boundary.pre_tokens,
            boundary.post_tokens,
            boundary.trigger.as_deref()
        ),
        (Some(182000), Some(14500), Some("auto"))
    );
    let kept = boundary.kept_range.unwrap();
    assert_eq!(
        (kept.first.to_string(), kept.last.to_string()),
        ("claude-code:first".into(), "claude-code:first".into())
    );
    assert_eq!(
        boundary.kept_after.unwrap().to_string(),
        "claude-code:summary"
    );
}

#[test]
fn a_nested_agent_carrier_does_not_invent_an_immediate_parent() {
    let parser = ClaudeCode {
        file: File::Subagent {
            stem: "agent-nested".into(),
        },
    };
    let records = [
        json!({"type":"user","uuid":"task","sessionId":"root-carrier","agentId":"nested",
        "isSidechain":true,"message":{"content":"A delegated task"}}),
    ];
    let out = run(&parser, &mut ParserState::default(), &jsonl(&records), 0)
        .unwrap()
        .output;
    assert!(out.links.is_empty());
    assert_eq!(
        out.messages[0].message.session.to_string(),
        "claude-code:agent-nested"
    );
    assert_eq!(out.messages[0].message.prompt_kind, Some(PromptKind::Model));
}

#[test]
fn rewritten_compactor_markers_report_unknown_session() {
    let parser = ClaudeCode {
        file: File::Subagent {
            stem: "agent-acompact-one".into(),
        },
    };
    let records = [
        json!({"type":"user","uuid":"replayed","sessionId":"root-carrier",
        "agentId":"acompact-one","isSidechain":true,"message":{"content":"kept context"}}),
    ];
    let out = run(&parser, &mut ParserState::default(), &jsonl(&records), 0)
        .unwrap()
        .output;
    assert!(out.links.is_empty());
    let issue = out
        .events
        .iter()
        .find(|event| event.kind == "compaction_ownership_unknown")
        .unwrap();
    assert!(
        issue.data["reason"]
            .as_str()
            .unwrap()
            .contains("do not establish")
    );
    assert_eq!(
        out.messages[0].message.session.to_string(),
        "claude-code:agent-acompact-one"
    );
    assert_eq!(out.messages[0].message.prompt_kind, None);
    assert_eq!(
        polylogue_model::WrittenBy::of(&out.messages[0].message),
        polylogue_model::WrittenBy::Unknown
    );
}

/// An event after a later record of a response follows the latest message
/// opened before it, read at once or in two steps.
#[test]
fn an_event_after_a_reopened_response_follows_the_latest_opened_message() {
    let records = vec![
        assistant("a1", "msg_1", json!({"type":"text","text":"one"})),
        assistant("b1", "msg_2", json!({"type":"text","text":"two"})),
        json!({"type":"user","uuid":"u1","parentUuid":"b1","sessionId":"s1",
               "message":{"role":"user","content":"go on"}}),
        assistant("a2", "msg_1", json!({"type":"text","text":"more"})),
        json!({"type":"system","subtype":"informational","uuid":"e1","sessionId":"s1"}),
    ];
    let bytes = jsonl(&records);
    let after = |steps: Vec<crate::parse::Step>| {
        crate::chain_names::tests::settled(&steps)
            .iter()
            .flat_map(|step| step.output.events.clone())
            .find(|event| event.kind == "informational")
            .unwrap()
            .after
    };
    let whole = after(vec![
        run(&main_file("s1"), &mut ParserState::default(), &bytes, 0).unwrap(),
    ]);
    assert_eq!(whole, message_id("u1"));
    let cut = jsonl(&records[..4]).len();
    let mut state = ParserState::default();
    let first = run(&main_file("s1"), &mut state, &bytes[..cut], 0).unwrap();
    let second = run(&main_file("s1"), &mut state, &bytes[cut..], cut as u64).unwrap();
    assert_eq!(after(vec![first, second]), whole);
}

/// A file history snapshot names no session, so it is stated for the file's
/// own session, yet after a later record of a resumed session's response it
/// follows the latest message that session opened, read at once or in two
/// steps.
#[test]
fn a_sessionless_event_follows_the_latest_opened_message_of_any_session() {
    let resumed = |uuid: &str, msg_id: &str, text: &str| {
        let mut record = assistant(uuid, msg_id, json!({"type":"text","text":text}));
        record["sessionId"] = json!("s2");
        record
    };
    let records = vec![
        assistant("a1", "msg_1", json!({"type":"text","text":"own"})),
        resumed("b1", "msg_2", "resumed"),
        resumed("c1", "msg_3", "another"),
        resumed("b2", "msg_2", "resumed, continued"),
        json!({"type":"file-history-snapshot","messageId":"b2","snapshot":{}}),
    ];
    let bytes = jsonl(&records);
    let after = |steps: Vec<crate::parse::Step>| {
        crate::chain_names::tests::settled(&steps)
            .iter()
            .flat_map(|step| step.output.events.clone())
            .find(|event| event.kind == "file_history")
            .unwrap()
            .after
    };
    let whole = after(vec![
        run(&main_file("s1"), &mut ParserState::default(), &bytes, 0).unwrap(),
    ]);
    assert_eq!(whole, message_id("c1"));
    let cut = jsonl(&records[..4]).len();
    let mut state = ParserState::default();
    let first = run(&main_file("s1"), &mut state, &bytes[..cut], 0).unwrap();
    let second = run(&main_file("s1"), &mut state, &bytes[cut..], cut as u64).unwrap();
    assert_eq!(after(vec![first, second]), whole);
}

#[test]
fn a_title_summary_is_not_a_compaction_boundary() {
    // Claude Code writes `{"type":"summary","summary":…,"leafUuid":…}` as the
    // session's title line; compactions are `isCompactSummary` user records.
    let records = [
        json!({"type":"user","uuid":"first","sessionId":"s1","message":{"content":"Fix the parser"}}),
        json!({"type":"summary","summary":"Parser fix","leafUuid":"first"}),
    ];
    let out = run(
        &main_file("s1"),
        &mut ParserState::default(),
        &jsonl(&records),
        0,
    )
    .unwrap()
    .output;
    assert_eq!(
        out.messages
            .iter()
            .map(|row| row.message.id.to_string())
            .collect::<Vec<_>>(),
        ["claude-code:first"]
    );
    assert!(out.events.iter().all(|event| event.kind != "compaction"));
    let title = out.sessions[0].title.as_ref().unwrap();
    assert_eq!(
        (title.text.as_str(), title.source),
        ("Parser fix", TitleSource::Summary)
    );
}

#[test]
fn a_compaction_summary_is_recorded() {
    let boundary = json!({"type":"system","subtype":"compact_boundary","uuid":"boundary","sessionId":"s1",
        "logicalParentUuid":"first","compactMetadata":{"trigger":"manual","preTokens":90000}});
    let summary = json!({"type":"user","uuid":"summary","parentUuid":"boundary","sessionId":"s1",
        "isCompactSummary":true,"message":{"content":"Known compressed context."}});
    let first = jsonl(&[
        json!({"type":"user","uuid":"first","sessionId":"s1","message":{"content":"original"}}),
        boundary.clone(),
    ]);
    let second = jsonl(std::slice::from_ref(&summary));
    // The summary arrives in a later step than its boundary, as when the
    // file is read while Claude Code writes it.
    let parser = main_file("s1");
    let mut state = ParserState::default();
    let one = run(&parser, &mut state, &first, 0).unwrap();
    assert!(
        one.output
            .events
            .iter()
            .all(|event| event.kind != "compaction")
    );
    let two = run(&parser, &mut state, &second, first.len() as u64).unwrap();
    let events: Vec<_> = two
        .output
        .events
        .iter()
        .filter(|event| event.kind == "compaction")
        .collect();
    assert_eq!(events.len(), 1);
    // The event depends on both records: its range runs from the boundary
    // record's start to the summary record's end (each line without its
    // newline).
    let boundary_line = serde_json::to_vec(&boundary).unwrap().len() as u64;
    let boundary_start = first.len() as u64 - 1 - boundary_line;
    let summary_end = (first.len() + second.len()) as u64 - 1;
    assert_eq!(events[0].range, Some((boundary_start, summary_end)));
    assert_eq!(
        events[0].after.as_ref().map(ToString::to_string).as_deref(),
        Some("claude-code:first")
    );
    let recorded: polylogue_model::CompactionBoundary =
        serde_json::from_value(events[0].data.clone()).unwrap();
    assert_eq!(
        recorded
            .summary
            .as_ref()
            .map(ToString::to_string)
            .as_deref(),
        Some("claude-code:summary")
    );
    assert_eq!(
        (recorded.trigger.as_deref(), recorded.pre_tokens),
        (Some("manual"), Some(90000))
    );

    // A boundary that no summary follows is sent with the next record,
    // naming no summary.
    let records = [
        boundary,
        json!({"type":"user","uuid":"next","sessionId":"s1","message":{"content":"go on"}}),
    ];
    let out = run(&parser, &mut ParserState::default(), &jsonl(&records), 0)
        .unwrap()
        .output;
    let events: Vec<_> = out
        .events
        .iter()
        .filter(|event| event.kind == "compaction")
        .collect();
    assert_eq!(events.len(), 1);
    let recorded: polylogue_model::CompactionBoundary =
        serde_json::from_value(events[0].data.clone()).unwrap();
    assert_eq!(recorded.summary, None);

    // A summary with no boundary record before it is a boundary of its own.
    let out = run(&parser, &mut ParserState::default(), &jsonl(&[summary]), 0)
        .unwrap()
        .output;
    let recorded: Vec<polylogue_model::CompactionBoundary> = out
        .events
        .iter()
        .filter(|event| event.kind == "compaction")
        .map(|event| serde_json::from_value(event.data.clone()).unwrap())
        .collect();
    assert_eq!(recorded.len(), 1);
    assert_eq!(
        recorded[0]
            .summary
            .as_ref()
            .map(ToString::to_string)
            .as_deref(),
        Some("claude-code:summary")
    );
}

#[test]
fn a_sidechain_with_a_missing_parent_starts_its_own_session() {
    let records = vec![
        json!({"type":"user","uuid":"first","sessionId":"s1","isSidechain":true,
            "parentUuid":"absent-one","message":{"role":"user","content":"First task."}}),
        json!({"type":"user","uuid":"second","sessionId":"s1","isSidechain":true,
            "parentUuid":"absent-two","message":{"role":"user","content":"Second task."}}),
        json!({"type":"assistant","uuid":"answer","sessionId":"s1","isSidechain":true,
            "parentUuid":"second","message":{"role":"assistant","content":"Second answer."}}),
    ];
    let piece = run(
        &main_file("s1"),
        &mut ParserState::default(),
        &jsonl(&records),
        0,
    )
    .unwrap();
    let messages = final_messages(&[piece]);
    assert_eq!(
        messages
            .iter()
            .map(|message| message.session.native().as_str())
            .collect::<Vec<_>>(),
        ["s1~side-first", "s1~side-second", "s1~side-second"]
    );
    assert_eq!(messages[0].parent, message_id("absent-one"));
    assert_eq!(messages[1].parent, message_id("absent-two"));
}

#[test]
fn morrows_older_sidechain_uses_its_first_record_uuid() {
    let bytes = include_bytes!(
        "../../../../examples/morrow/claude-code-only/claude-code/projects/-home-op-morrow/7f3a2c10-0101-4000-8000-000000000101.jsonl"
    );
    let piece = run(
        &main_file("7f3a2c10-0101-4000-8000-000000000101"),
        &mut ParserState::default(),
        bytes,
        0,
    )
    .unwrap();
    let pieces = crate::chain_names::tests::settled(&[piece]);
    let out = &pieces[0].output;
    let side = "7f3a2c10-0101-4000-8000-000000000101~side-d1500000-0000-4000-8000-000000000001";
    assert_eq!(
        out.sessions
            .iter()
            .filter(|session| session.kind == SessionKind::Sidechain)
            .map(|session| session.id.native().as_str())
            .collect::<Vec<_>>(),
        [side]
    );
    assert_eq!(
        out.messages
            .iter()
            .filter(|row| row.message.session.native().as_str() == side)
            .map(|row| row.message.id.native().as_str())
            .collect::<Vec<_>>(),
        [
            "d1500000-0000-4000-8000-000000000001",
            "d1500000-0000-4000-8000-000000000002"
        ]
    );
    assert_eq!(
        out.links
            .iter()
            .filter(|link| link.kind == LinkKind::Spawn && link.child.native().as_str() == side)
            .map(|link| link.call_id.as_deref())
            .collect::<Vec<_>>(),
        [Some("toolu_1102")]
    );
    let prompt = &out
        .messages
        .iter()
        .find(|row| row.message.session.native().as_str() == side)
        .unwrap()
        .message;
    assert_eq!(
        polylogue_model::WrittenBy::of(prompt),
        polylogue_model::WrittenBy::Model
    );
}

#[test]
fn a_parent_before_the_latest_sidechain_record_starts_a_separate_sidechain() {
    let records = vec![
        json!({"type":"user","uuid":"first","sessionId":"s1","isSidechain":true,
            "parentUuid":null,"message":{"role":"user","content":"Review the file."}}),
        json!({"type":"assistant","uuid":"second","sessionId":"s1","isSidechain":true,
            "parentUuid":"first","message":{"role":"assistant","content":"File reviewed."}}),
        json!({"type":"user","uuid":"third","sessionId":"s1","isSidechain":true,
            "parentUuid":"first","message":{"role":"user","content":"Check the test."}}),
        json!({"type":"assistant","uuid":"fourth","sessionId":"s1","isSidechain":true,
            "parentUuid":"second","message":{"role":"assistant","content":"Review finished."}}),
    ];
    let piece = run(
        &main_file("s1"),
        &mut ParserState::default(),
        &jsonl(&records),
        0,
    )
    .unwrap();
    let messages = final_messages(&[piece]);
    assert_eq!(
        messages
            .iter()
            .map(|message| message.session.native().as_str())
            .collect::<Vec<_>>(),
        [
            "s1~side-first",
            "s1~side-first",
            "s1~side-third",
            "s1~side-first"
        ]
    );
}

fn check_appended(parser: &ClaudeCode, records: &[Json], check: impl Fn(&[crate::parse::Step])) {
    let bytes = jsonl(records);
    check(&[run(parser, &mut ParserState::default(), &bytes, 0).unwrap()]);
    for cut in bytes
        .iter()
        .enumerate()
        .filter_map(|(i, byte)| (*byte == b'\n' && i + 1 < bytes.len()).then_some(i + 1))
    {
        let mut state = ParserState::default();
        let first = run(parser, &mut state, &bytes[..cut], 0).unwrap();
        let mut state = serde_json::from_value(serde_json::to_value(state).unwrap()).unwrap();
        let second = run(parser, &mut state, &bytes[cut..], cut as u64).unwrap();
        check(&[first, second]);
    }
}

#[test]
fn fork_records_copy_the_named_parent_messages() {
    let records = [
        json!({"type":"user","uuid":"copy-u","sessionId":"fork","parentUuid":null,
               "forkedFrom":{"sessionId":"root","messageUuid":"root-u"},
               "message":{"role":"user","content":"Check the table."}}),
        json!({"type":"assistant","uuid":"copy-a","sessionId":"fork","parentUuid":"copy-u",
               "forkedFrom":{"sessionId":"root","messageUuid":"root-a"},
               "message":{"id":"answer","role":"assistant","content":"The table checks out."}}),
        json!({"type":"user","uuid":"fork-u","sessionId":"fork","parentUuid":"copy-a",
               "message":{"role":"user","content":"Check the dates too."}}),
    ];
    check_appended(&main_file("fork"), &records, |steps| {
        let messages = final_messages(steps);
        let facts: Vec<_> = messages
            .iter()
            .map(|m| {
                (
                    m.session.native().as_str(),
                    m.id.native().as_str(),
                    m.parent.as_ref().map(|id| id.native().as_str()),
                )
            })
            .collect();
        assert_eq!(
            facts,
            [
                ("root", "root-u", None),
                ("root", "root-a", Some("root-u")),
                ("fork", "fork-u", Some("root-a")),
            ]
        );
        let copies: Vec<_> = steps
            .iter()
            .flat_map(|step| &step.output.prefix_copies)
            .collect();
        let copy = polylogue_model::PrefixCopy {
            parent: session_id("root").unwrap(),
            child: session_id("fork").unwrap(),
            explicit_members: true,
        };
        assert!(!copies.is_empty());
        assert!(copies.into_iter().all(|stated| *stated == copy));
        let members: Vec<_> = steps
            .iter()
            .flat_map(|step| &step.output.prefix_members)
            .map(|member| {
                (
                    &member.parent,
                    &member.child,
                    member.message.native().as_str(),
                )
            })
            .collect();
        assert_eq!(
            members,
            [
                (&copy.parent, &copy.child, "root-u"),
                (&copy.parent, &copy.child, "root-a"),
            ]
        );
    });
}

fn task(uuid: &str, id: &str, prompt: &str) -> Json {
    assistant(
        uuid,
        uuid,
        json!({"type":"tool_use","name":"Task","id":id,"input":{"prompt":prompt}}),
    )
}

fn sidechain(uuid: &str, content: Json) -> Json {
    json!({"type":"user","uuid":uuid,"sessionId":"s1","isSidechain":true,"parentUuid":null,
           "message":{"role":"user","content":content}})
}

fn check_sidechain_calls(records: &[Json], expected: &[(&str, Option<&str>)]) {
    check_appended(&main_file("s1"), records, |steps| {
        let calls: Vec<_> = steps
            .iter()
            .flat_map(|step| &step.output.links)
            .map(|link| {
                assert_eq!(link.parent, session_id("s1").unwrap());
                assert_eq!(link.kind, LinkKind::Spawn);
                (link.child.native().as_str(), link.call_id.as_deref())
            })
            .collect();
        assert_eq!(calls, expected);
    });
}

#[test]
fn sidechain_names_its_only_open_task() {
    let records = [
        task("a1", "call-one", "Review the index."),
        sidechain("review", json!("Review the index.")),
        json!({"type":"user","uuid":"result","sessionId":"s1",
               "message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"call-one","content":"Done."}]}}),
        sidechain("unmatched", json!("Review the index.")),
    ];
    check_sidechain_calls(
        &records,
        &[
            ("s1~side-review", Some("call-one")),
            ("s1~side-unmatched", None),
        ],
    );
}

#[test]
fn interleaved_sidechains_match_their_own_open_tasks() {
    let records = [
        task("a1", "call-one", "Review the index."),
        task("a2", "call-two", "Review the appendix."),
        sidechain("index", json!("Review the index.")),
        sidechain(
            "appendix",
            json!([{"type":"text","text":"Your task: Review the appendix. Report any issues."}]),
        ),
        json!({"type":"assistant","uuid":"index-answer","sessionId":"s1","isSidechain":true,
               "parentUuid":"index","message":{"role":"assistant","content":"The index is correct."}}),
        json!({"type":"assistant","uuid":"appendix-answer","sessionId":"s1","isSidechain":true,
               "parentUuid":"appendix","message":{"role":"assistant","content":"The appendix is correct."}}),
    ];
    check_sidechain_calls(
        &records,
        &[
            ("s1~side-index", Some("call-one")),
            ("s1~side-appendix", Some("call-two")),
        ],
    );
}

#[test]
fn sidechain_names_no_call_when_several_prompts_match() {
    let records = [
        task("a1", "call-one", "Review the index."),
        task("a2", "call-two", "Review the appendix."),
        sidechain("both", json!("Review the index. Review the appendix.")),
    ];
    check_sidechain_calls(&records, &[("s1~side-both", None)]);
}

#[test]
fn opening_sidechain_prompt_is_written_by_a_model() {
    use polylogue_model::WrittenBy;
    let mut follow_up = sidechain("follow-up", json!("Now check the appendix."));
    follow_up["parentUuid"] = json!("opening");
    follow_up["origin"] = json!({"kind":"human"});
    let records = [sidechain("opening", json!("Check the table.")), follow_up];
    check_appended(&main_file("s1"), &records, |steps| {
        let messages = final_messages(steps);
        assert_eq!(
            messages.iter().map(WrittenBy::of).collect::<Vec<_>>(),
            [WrittenBy::Model, WrittenBy::Human]
        );
        assert_eq!(messages[0].prompt_kind, Some(PromptKind::Model));
    });
}

#[test]
fn first_subagent_prompt_after_copied_messages_is_written_by_a_model() {
    use polylogue_model::WrittenBy;
    let agent = ClaudeCode {
        file: File::Subagent {
            stem: "agent-next".into(),
        },
    };
    let records = [
        json!({"type":"user","uuid":"original","sessionId":"s1","agentId":"first","parentUuid":null,
               "message":{"role":"user","content":"Check the table."}}),
        json!({"type":"assistant","uuid":"answer","sessionId":"s1","agentId":"first","parentUuid":"original",
               "message":{"role":"assistant","content":"The table is correct."}}),
        json!({"type":"user","uuid":"delegated","sessionId":"s1","agentId":"next","parentUuid":"answer",
               "message":{"role":"user","content":"Check the appendix."}}),
        json!({"type":"user","uuid":"later","sessionId":"s1","agentId":"next","parentUuid":"delegated",
               "message":{"role":"user","content":"That is enough."}}),
    ];
    check_appended(&agent, &records, |steps| {
        let messages = final_messages(steps);
        let writers: Vec<_> = messages
            .iter()
            .map(|m| {
                (
                    m.session.native().as_str(),
                    m.id.native().as_str(),
                    WrittenBy::of(m),
                )
            })
            .collect();
        assert_eq!(
            writers,
            [
                ("agent-first", "original", WrittenBy::Model),
                ("agent-first", "answer", WrittenBy::Model),
                ("agent-next", "delegated", WrittenBy::Model),
                ("agent-next", "later", WrittenBy::Human),
            ]
        );
        assert_eq!(messages[2].prompt_kind, Some(PromptKind::Model));
    });
}

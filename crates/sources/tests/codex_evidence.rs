//! Raw source evidence and unknown ownership, independent of writer finalization.
use polylogue_model::{BlockContent, BlockSlot, ToolOutcome};
use polylogue_sources::{
    codex::keyed::{Cell, EvidenceCursor, EvidenceHead, Key, KeyedRecords, MemoryReadView, Resume},
    frame::drive_jsonl,
};
use serde_json::{Value, json};
fn line(kind: &str, payload: Value) -> String {
    json!({"type":kind,"payload":payload}).to_string() + "\n"
}
fn header() -> String {
    line(
        "session_meta",
        json!({"id":"session:opaque","source":"cli"}),
    )
}
fn run(view: MemoryReadView, resume: Resume, bytes: &str) -> (MemoryReadView, Resume) {
    let start = resume.parsed_upto;
    let framed = drive_jsonl(KeyedRecords::new(&view, resume), bytes.as_bytes(), start);
    let mut delta = framed.output.unwrap();
    delta.resume.parsed_upto = framed.parsed_upto;
    let mut view = view;
    view.apply(&delta);
    (view, delta.resume)
}
fn head(view: &MemoryReadView, message: usize) -> EvidenceHead {
    match view.cells.get(&Key::EvidenceHead(message)).unwrap() {
        Cell::EvidenceHead(head) => head.clone(),
        _ => panic!("shape"),
    }
}
fn units(view: &MemoryReadView, message: usize) -> Vec<(String, Vec<u8>)> {
    let head = head(view, message);
    let mut cursor = EvidenceCursor::default();
    let mut out = Vec::new();
    while let Some(key) = head.next_slot(message, &mut cursor) {
        if let Some(Cell::Evidence(unit)) = view.cells.get(&key) {
            out.push((unit.tag.clone(), unit.canonical.clone()));
        }
    }
    assert_eq!(out.len() as u64, head.units);
    out
}
#[test]
fn reasoning_reemission_replaces_one_slot_and_reanchor_keeps_reasoning_identity() {
    let r = line(
        "response_item",
        json!({"type":"reasoning","id":"r:1","summary":[{"text":"first"}]}),
    );
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &r),
    );
    let original = head(&view, 0);
    let original_units = units(&view, 0);
    let (view, resume) = run(view, resume, &r);
    assert_eq!(head(&view, 0), original);
    assert_eq!(units(&view, 0), original_units);
    let replacement = line(
        "response_item",
        json!({"type":"reasoning","id":"r:1","summary":[{"text":"longer"}]}),
    );
    let (view, resume) = run(view, resume, &replacement);
    assert_eq!(head(&view, 0).items, 1);
    assert!(head(&view, 0).revision > original.revision);
    let anchor = line(
        "response_item",
        json!({"type":"function_call","id":"a:1","call_id":"c:1","name":"exec","arguments":"{}"}),
    );
    let (view, resume) = run(view, resume, &anchor);
    let message = view.message(0).unwrap().unwrap();
    assert_eq!(message.id.session().as_str(), "codex:session:opaque");
    assert_eq!(message.id.as_str(), "codex:session:opaque:a:1");
    assert!(matches!(
        message.blocks[0].id.slot(),
        BlockSlot::Reasoning(0)
    ));
    assert!(matches!(message.blocks[1].id.slot(), BlockSlot::Item(0)));
    assert!(message.version_hash.is_none());
    assert_eq!(resume.message_count(), 1);
    assert_eq!(head(&view, 0).items, 2);
}
#[test]
fn repeat_usage_null_close_and_rate_limit_do_not_change_effective_membership() {
    let call = line(
        "response_item",
        json!({"type":"function_call","id":"a","call_id":"c","name":"tool","arguments":"{}"}),
    );
    let info =
        json!({"total_token_usage":{"input_tokens":2},"last_token_usage":{"input_tokens":2}});
    let usage = line("event_msg", json!({"type":"token_count","info":info}));
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &call + &usage),
    );
    let before = units(&view, 0);
    let before_head = head(&view, 0);
    let repeated = line(
        "event_msg",
        json!({"type":"token_count","info":info,"rate_limits":{"remaining":5}}),
    );
    let null = line("event_msg", json!({"type":"token_count","info":null}));
    let (view, _) = run(view, resume, &(repeated + &null));
    assert_eq!(units(&view, 0), before);
    assert_eq!(head(&view, 0), before_head);
    assert_eq!(before.iter().filter(|(tag, _)| tag == "usage").count(), 1);
}
#[test]
fn unowned_child_never_enters_latest_wrapper_message() {
    let w1 = line(
        "response_item",
        json!({"type":"function_call","id":"a1","call_id":"w1","name":"exec","arguments":"{}"}),
    );
    let close = line("event_msg", json!({"type":"token_count","info":null}));
    let w2 = line(
        "response_item",
        json!({"type":"function_call","id":"a2","call_id":"w2","name":"exec","arguments":"{}"}),
    );
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &w1 + &close + &w2),
    );
    let before = [units(&view, 0), units(&view, 1)];
    let child = line(
        "event_msg",
        json!({"type":"item_completed","item":{"type":"CommandExecution","id":"child","source":"agent","status":"failed","exit_code":1,"aggregated_output":""}}),
    );
    let framed = drive_jsonl(KeyedRecords::new(&view, resume), child.as_bytes(), 0);
    let delta = framed.output.unwrap();
    assert!(delta.resume.normalization_incomplete());
    assert_eq!(delta.events[0].kind, "unowned_tool_item");
    assert!(delta.pending_versions.is_empty());
    let mut after = view;
    after.apply(&delta);
    assert_eq!([units(&after, 0), units(&after, 1)], before);
    for i in 0..2 {
        assert_eq!(after.message(i).unwrap().unwrap().blocks.len(), 1);
    }
}
#[test]
fn exact_numeric_fallback_and_structured_error_are_not_normalized_away() {
    let a = header()
        + "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"text\":\"hello\"}],\"n\":1E2}}\n";
    let b = a.replace("1E2", "1e2");
    let (a, _) = run(MemoryReadView::default(), Resume::default(), &a);
    let (b, _) = run(MemoryReadView::default(), Resume::default(), &b);
    assert_ne!(
        a.message(0).unwrap().unwrap().id,
        b.message(0).unwrap().unwrap().id
    );
    assert!(units(&a, 0)[0].1.windows(3).any(|s| s == b"1E2"));
    let output = line(
        "response_item",
        json!({"type":"function_call_output","call_id":"unpaired","output":{"is_error":true,"output":""}}),
    );
    let (view, _) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &output),
    );
    let message = view.message(0).unwrap().unwrap();
    assert!(
        matches!(&message.blocks[0].content,BlockContent::ToolResult(result) if matches!(result.outcome,ToolOutcome::Error{..}))
    );
}

#[test]
fn assistant_snapshot_replaces_its_block_and_retains_missing_raw_fields() {
    let first = line(
        "response_item",
        json!({"type":"message","id":"answer","role":"assistant","phase":"final_answer","content":[{"text":"first"}]}),
    );
    let close = line("event_msg", json!({"type":"token_count","info":null}));
    let later = line(
        "response_item",
        json!({"id":"answer","content":[{"text":"last"}]}),
    );
    let bytes = header() + &first + &first + &close + &later;
    let (whole, resume) = run(MemoryReadView::default(), Resume::default(), &bytes);
    assert_eq!(resume.message_count(), 1);
    let message = whole.message(0).unwrap().unwrap();
    assert_eq!(message.blocks.len(), 1);
    assert!(matches!(&message.blocks[0].content,BlockContent::Text{text} if text=="last"));
    assert_eq!(message.attrs.unwrap()["phase"], "final_answer");
    let head = head(&whole, 0);
    assert_eq!(head.items, 1);
    assert!(!head.incomplete);
    assert!(
        units(&whole, 0)[0]
            .1
            .windows(12)
            .any(|bytes| bytes == b"final_answer")
    );
    let mut view = MemoryReadView::default();
    let mut state = Resume::default();
    for record in bytes.split_inclusive('\n') {
        let (next, resume) = run(view, state, record);
        view = next;
        state = resume;
    }
    assert_eq!(view.message(0).unwrap(), whole.message(0).unwrap());
    assert_eq!(units(&view, 0), units(&whole, 0));
}
#[test]
fn call_and_owned_child_snapshots_replace_without_reopening_or_double_counting() {
    let wrapper = line(
        "response_item",
        json!({"type":"function_call","id":"wrapper-item","call_id":"w","name":"exec","arguments":"{\"cmd\":\"first\"}"}),
    );
    let replacement = line(
        "response_item",
        json!({"type":"function_call","id":"wrapper-item","arguments":"{\"cmd\":\"last\"}"}),
    );
    let child = line(
        "event_msg",
        json!({"type":"item_completed","item":{"type":"CommandExecution","id":"child","parent_call_id":"w","status":"completed","aggregated_output":""}}),
    );
    let done = line(
        "response_item",
        json!({"type":"function_call_output","id":"wrapper-result","call_id":"w","output":"Script completed"}),
    );
    let changed = line(
        "event_msg",
        json!({"type":"item_completed","item":{"type":"CommandExecution","id":"child","status":"failed","exit_code":1}}),
    );
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &wrapper + &wrapper + &replacement + &child + &done + &changed),
    );
    assert_eq!(resume.message_count(), 2);
    assert!(matches!(
        view.cells.get(&Key::ChildCount("w".into())),
        Some(Cell::Number(1))
    ));
    let message = view.message(0).unwrap().unwrap();
    assert_eq!(message.blocks.len(), 3);
    assert!(
        matches!(&message.blocks[0].content,BlockContent::ToolCall(call) if call.input.as_ref().unwrap()["cmd"]=="last")
    );
    assert!(
        matches!(&message.blocks[2].content,BlockContent::ToolResult(result) if matches!(result.outcome,ToolOutcome::Error{..})&&result.exit_code==Some(1)&&result.output==Some(polylogue_model::ToolOutput::Text(String::new())))
    );
}
#[test]
fn native_result_snapshot_replaces_the_output_chunk_without_a_second_message() {
    let call = line(
        "response_item",
        json!({"type":"function_call","id":"a","call_id":"c","name":"tool","arguments":"{}"}),
    );
    let first = line(
        "response_item",
        json!({"type":"function_call_output","id":"out","call_id":"c","output":"first"}),
    );
    let last = line("response_item", json!({"id":"out","output":"last"}));
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &call + &first + &last),
    );
    assert_eq!(resume.message_count(), 2);
    assert_eq!(head(&view, 1).items, 1);
    let message = view.message(1).unwrap().unwrap();
    assert_eq!(message.blocks.len(), 1);
    assert!(
        matches!(&message.blocks[0].content,BlockContent::ToolResult(result) if result.output==Some(polylogue_model::ToolOutput::Text("last".into())))
    );
    assert!(view.cells.contains_key(&Key::PendingResult(1, 0)));
}
#[test]
fn replacing_an_old_continuation_preserves_distinct_equal_chunks_without_prefix_copying() {
    let call = line(
        "response_item",
        json!({"type":"function_call","id":"a","call_id":"c","name":"exec_command","arguments":"{}"}),
    );
    let start = line(
        "response_item",
        json!({"type":"function_call_output","id":"start","call_id":"c","output":"Process running with session ID 7"}),
    );
    let mut bytes = header() + &call + &start;
    for n in 1..=2 {
        bytes += &line(
            "response_item",
            json!({"type":"function_call","id":format!("stdin{n}"),"call_id":format!("s{n}"),"name":"write_stdin","arguments":"{\"session_id\":7}"}),
        );
        bytes += &line(
            "response_item",
            json!({"type":"function_call_output","id":format!("chunk{n}"),"call_id":format!("s{n}"),"output":"tick"}),
        );
    }
    let (view, resume) = run(MemoryReadView::default(), Resume::default(), &bytes);
    let changed = line("response_item", json!({"id":"chunk1","output":"TACK"}));
    let (view, _) = run(view, resume, &changed);
    assert_eq!(head(&view, 1).items, 3);
    let message = view.message(1).unwrap().unwrap();
    assert!(
        matches!(&message.blocks[0].content,BlockContent::ToolResult(result) if result.output==Some(polylogue_model::ToolOutput::Text("Process running with session ID 7TACKtick".into())))
    );
    let raw = match view.cells.get(&Key::Block(1, 0)).unwrap() {
        Cell::Block(block) => block,
        _ => panic!("shape"),
    };
    assert!(
        matches!(&raw.content,BlockContent::ToolResult(result) if result.output==Some(polylogue_model::ToolOutput::Text("Process running with session ID 7".into()))),
        "interpretation did not rebuild the growing result body"
    );
}

#[test]
fn distinct_native_calls_with_duplicate_call_ids_keep_both_wrapper_occurrences() {
    let mut bytes = header();
    for id in ["one", "two"] {
        bytes += &line(
            "response_item",
            json!({"type":"function_call","id":id,"call_id":"duplicate","name":"exec","arguments":"{}"}),
        );
    }
    let (view, _) = run(MemoryReadView::default(), Resume::default(), &bytes);
    assert_eq!(
        view.cells
            .keys()
            .filter(|key| matches!(key, Key::Wrapper(_)))
            .count(),
        2
    );
    assert_eq!(view.message(0).unwrap().unwrap().blocks.len(), 2);
    assert_eq!(head(&view, 0).items, 2);
}

#[test]
fn unmapped_message_content_is_preserved_without_completed_empty_text() {
    let item = json!({"type":"message","id":"image-prompt","role":"user","content":[{"type":"input_image","image_url":"synthetic://image","text":"caption"}]});
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &line("response_item", item.clone())),
    );
    assert!(resume.normalization_incomplete());
    let message = view.message(0).unwrap().unwrap();
    assert!(matches!(&message.blocks[0].content,BlockContent::Other{data,..} if data==&item));
}
#[test]
fn explicitly_unreadable_model_context_does_not_reuse_a_stale_model() {
    let known = line("turn_context", json!({"model":"model-before"}));
    let unknown = line("turn_context", json!({"model":null}));
    let answer = line(
        "response_item",
        json!({"type":"message","id":"answer","role":"assistant","content":[{"text":"hello"}]}),
    );
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &known + &unknown + &answer),
    );
    assert!(resume.normalization_incomplete());
    assert!(view.message(0).unwrap().unwrap().model.is_none());
    assert!(units(&view, 0).contains(&("context:model".into(), b"null".to_vec())));
}

#[test]
fn a_promised_running_result_with_wrong_content_fails_the_attempt() {
    let first = line(
        "response_item",
        json!({"type":"function_call","id":"a","call_id":"c","name":"exec_command","arguments":"{}"}),
    );
    let running = line(
        "response_item",
        json!({"type":"function_call_output","id":"start","call_id":"c","output":"Process running with session ID 7"}),
    );
    let stdin = line(
        "response_item",
        json!({"type":"function_call","id":"stdin","call_id":"s","name":"write_stdin","arguments":"{\"session_id\":7}"}),
    );
    let (mut view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(header() + &first + &running + &stdin),
    );
    if let Some(Cell::Block(block)) = view.cells.get_mut(&Key::Block(1, 0)) {
        block.content = BlockContent::Text {
            text: "wrong state shape".into(),
        };
    }
    let output = line(
        "response_item",
        json!({"type":"function_call_output","id":"end","call_id":"s","output":"still running"}),
    );
    let framed = drive_jsonl(KeyedRecords::new(&view, resume), output.as_bytes(), 0);
    assert!(
        framed
            .output
            .unwrap_err()
            .error
            .contains("promised tool result")
    );
    assert_eq!(framed.counts.used, 0);
    assert!(framed.counts.unrecognized.is_empty());
}

#[test]
fn external_native_status_snapshots_overlay_missing_fields_and_keep_duration() {
    let call = line(
        "response_item",
        json!({"type":"function_call","id":"call-item","call_id":"c","name":"exec_command","arguments":"{}"}),
    );
    let native = line(
        "event_msg",
        json!({"type":"item_completed","turn_id":"turn","item":{"type":"CommandExecution","id":"c","status":"failed","exit_code":1,"duration":{"secs":1,"nanos":2000000}}}),
    );
    let output = line(
        "response_item",
        json!({"type":"function_call_output","id":"output","call_id":"c","output":"Exit code: 0"}),
    );
    let updated = line(
        "event_msg",
        json!({"type":"item_completed","turn_id":"turn","item":{"type":"CommandExecution","id":"c","status":"completed"}}),
    );
    let output_changed = line(
        "response_item",
        json!({"id":"output","output":"different text"}),
    );
    for bytes in [
        header() + &call + &native + &output + &updated + &output_changed,
        header() + &call + &native + &updated + &output + &output_changed,
    ] {
        let (view, resume) = run(MemoryReadView::default(), Resume::default(), &bytes);
        assert_eq!(resume.message_count(), 2);
        let message = view.message(1).unwrap().unwrap();
        assert!(
            matches!(&message.blocks[0].content,BlockContent::ToolResult(result) if matches!(result.outcome,ToolOutcome::Error{..})&&result.exit_code==Some(1)&&result.duration_ms==Some(1002))
        );
        let status = units(&view, 1)
            .into_iter()
            .find(|(tag, _)| tag == "status")
            .unwrap()
            .1;
        assert!(status.windows(13).any(|bytes| bytes == b"\"exit_code\":1"));
        assert!(status.windows(6).any(|bytes| bytes == b"\"secs\""));
    }
}
#[test]
fn fork_lineage_does_not_claim_completed_inheritance_correspondence() {
    let meta = line(
        "session_meta",
        json!({"id":"child","source":"cli","forked_from_id":"parent"}),
    );
    let framed = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        meta.as_bytes(),
        0,
    );
    let delta = framed.output.unwrap();
    assert!(delta.resume.normalization_incomplete());
    assert_eq!(delta.links.len(), 1);
}

#[test]
fn evidence_validation_owns_roles_required_slots_and_count_bounds() {
    use polylogue_sources::codex::keyed::EvidenceUnit;
    let mut head = EvidenceHead {
        items: 1,
        units: 1,
        revision: 1,
        incomplete: false,
    };
    assert!(head.validate().is_ok());
    assert!(head.required_slot(&Key::Evidence(0, 0, 0)));
    assert!(!head.required_slot(&Key::Evidence(0, 0, 1)));
    let mut unit = EvidenceUnit {
        tag: "item".into(),
        canonical: br#"{"n":1E2}"#.to_vec(),
        whole_record: true,
        origins: Default::default(),
    };
    assert!(unit.validate_at(&Key::Evidence(0, 0, 0)).is_ok());
    assert!(unit.validate_at(&Key::Evidence(0, 0, 1)).is_err());
    unit.tag = "context:source".into();
    assert!(unit.validate_at(&Key::Evidence(0, 1, 1)).is_err());
    unit.whole_record = false;
    assert!(unit.validate_at(&Key::Evidence(0, 1, 1)).is_ok());
    unit.canonical = b"bad".to_vec();
    assert!(unit.validate_at(&Key::Evidence(0, 1, 1)).is_err());
    head.items = usize::MAX;
    assert!(head.validate().is_err());
    head.items = 1;
    head.units = 9;
    assert!(head.validate().is_err());
    head.units = 1;
    head.incomplete = true;
    assert!(head.validate().is_err());
}

#[test]
fn pinned_unowned_command_keeps_wrapper_messages_and_raw_evidence_unchanged() {
    let w = |id: &str| {
        line(
            "response_item",
            json!({"type":"function_call","id":format!("message-{id}"),"call_id":id,"name":"exec","arguments":"{}"}),
        )
    };
    let bytes = header()
        + &line("turn_context", json!({"turn_id":"T"}))
        + &w("W1")
        + &line("event_msg", json!({"type":"token_count","info":null}))
        + &w("W2");
    let (view, resume) = run(MemoryReadView::default(), Resume::default(), &bytes);
    let before = [
        (
            view.message(0).unwrap().unwrap().blocks,
            head(&view, 0),
            units(&view, 0),
        ),
        (
            view.message(1).unwrap().unwrap().blocks,
            head(&view, 1),
            units(&view, 1),
        ),
    ];
    let completion = line(
        "event_msg",
        json!({"type":"item_completed","thread_id":"session:opaque","turn_id":"T","started_at_ms":1,"completed_at_ms":2,"item":{"type":"CommandExecution","id":"C","source":"agent","command":["printf",""],"parsed_cmd":[],"cwd":"/synthetic","status":"failed","exit_code":1,"aggregated_output":""}}),
    );
    let (view, resume) = run(view, resume, &completion);
    assert_eq!(resume.message_count(), 2);
    for (i, (blocks, h, u)) in before.iter().enumerate() {
        assert_eq!(&view.message(i).unwrap().unwrap().blocks, blocks);
        assert_eq!(&head(&view, i), h);
        assert_eq!(&units(&view, i), u);
    }
    let operations: Vec<_> = view
        .cells
        .values()
        .filter_map(|cell| match cell {
            Cell::Operation(op) => Some(op),
            _ => None,
        })
        .collect();
    let child = operations.iter().find(|op| op.invocation == "C").unwrap();
    assert!(child.message.is_none() && child.wrapper.is_none());
    assert_eq!(child.wrapper_overlaps.len(), 2);
    assert!(
        operations
            .iter()
            .filter(|op| op.kind == "CodeExecution")
            .all(|op| !op.native && op.provisional_anchor.is_some())
    );
    assert_eq!(child.origins.iter().filter(|o|matches!(o,polylogue_model::tool_item::ToolOrigin::Rollout{tag,..} if tag.starts_with("wrapper_overlap:"))).count(),2);
    assert!(
        child
            .wrapper_overlaps
            .iter()
            .all(|id| id.starts_with("operation:"))
    );
    assert!(child.counted() && child.result_present);
    assert_eq!(child.output.as_deref(), Some(""));
    assert_eq!(child.exit_code, Some(1));
    assert!(
        operations
            .iter()
            .filter(|op| op.kind == "CodeExecution")
            .all(|op| op.id().unwrap() != child.id().unwrap())
    );
}

#[test]
fn provisional_call_uses_its_real_block_and_only_an_exact_session_witness_refines_it() {
    let bytes = header()
        + &line(
            "response_item",
            json!({"type":"reasoning","id":"reason","summary":[{"text":"synthetic thought"}]}),
        )
        + &line(
            "response_item",
            json!({"type":"function_call","id":"request","call_id":"C","name":"exec_command","arguments":"{\"cmd\":\"same\"}"}),
        );
    let (view, resume) = run(MemoryReadView::default(), Resume::default(), &bytes);
    let call = view
        .message(0)
        .unwrap()
        .unwrap()
        .blocks
        .into_iter()
        .find(|b| matches!(b.content, BlockContent::ToolCall(_)))
        .unwrap();
    let provisional = view
        .cells
        .values()
        .find_map(|c| match c {
            Cell::Operation(op) if op.invocation == "C" => Some(op.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        provisional.provisional_anchor.as_deref(),
        Some(call.id.as_str())
    );
    assert!(!provisional.native && !provisional.counted());
    let completion = |thread: &str| {
        line(
            "event_msg",
            json!({"type":"item_completed","thread_id":thread,"turn_id":"T","started_at_ms":1,"completed_at_ms":2,"item":{"type":"CommandExecution","id":"C","source":"agent","command":["same"],"parsed_cmd":[],"cwd":"/synthetic","status":"failed","exit_code":1,"aggregated_output":""}}),
        )
    };
    let (view, resume) = run(view, resume, &completion("foreign-session"));
    assert!(
        !view
            .cells
            .values()
            .any(|c| matches!(c,Cell::Operation(op) if op.native))
    );
    let (view, _) = run(view, resume, &completion("session:opaque"));
    let native = view
        .cells
        .values()
        .find_map(|c| match c {
            Cell::Operation(op) if op.native => Some(op),
            _ => None,
        })
        .unwrap();
    assert_eq!(native.turn.as_deref(), Some("T"));
    assert_eq!(native.direct_anchors, vec![provisional.id().unwrap()]);
    assert_ne!(native.id().unwrap(), provisional.id().unwrap());
}

#[test]
fn native_completion_without_recorded_turn_never_borrows_current_turn() {
    let bytes = header()
        + &line("turn_context", json!({"turn_id":"newer-turn"}))
        + &line(
            "event_msg",
            json!({"type":"item_completed","thread_id":"session:opaque","item":{"type":"CommandExecution","id":"late-C","source":"agent","command":["same"],"status":"completed","exit_code":0}}),
        );
    let (view, resume) = run(MemoryReadView::default(), Resume::default(), &bytes);
    assert_eq!(resume.message_count(), 0);
    let operation = view
        .cells
        .values()
        .find_map(|c| match c {
            Cell::Operation(op) => Some(op),
            _ => None,
        })
        .unwrap();
    assert_eq!(operation.turn, None);
    assert_eq!(operation.correspondence, "unresolved");
    assert!(!operation.native && !operation.counted());
    assert!(
        operation
            .id()
            .unwrap()
            .starts_with("operation:provisional-v1:")
    );
}

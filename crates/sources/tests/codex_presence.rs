//! Presence comes from effective raw mappings, never populated DTO defaults.
use polylogue_model::{
    BlockSlot,
    merge::{BlockField as B, MessageField as M, Slot, UsageField as U},
};
use polylogue_sources::codex::{
    keyed::{Cell, Key, KeyedRecords, MemoryReadView, Resume},
    projection::{Cursor, Mode, Step, Unit},
};
use polylogue_sources::frame::drive_jsonl;
use serde_json::{Value, json};
fn line(kind: &str, payload: Value) -> String {
    json!({"timestamp":"2026-10-09T14:30:00Z","type":kind,"payload":payload}).to_string() + "\n"
}
fn header() -> String {
    line(
        "session_meta",
        json!({"id":"presence:synthetic","source":"cli"}),
    )
}
fn parse(bytes: &str) -> (MemoryReadView, Resume) {
    let framed = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        bytes.as_bytes(),
        0,
    );
    let delta = framed.output.unwrap();
    let mut view = MemoryReadView::default();
    view.apply(&delta);
    (view, delta.resume)
}
fn slots(view: &MemoryReadView, message: usize) -> std::collections::BTreeMap<Slot, Vec<u8>> {
    let mut cursor = Cursor::new(message, Mode::Annotated);
    let mut fields = std::collections::BTreeMap::new();
    loop {
        match cursor.advance(view).unwrap() {
            Step::Done => break,
            Step::Unit(Unit::MessageWitness(w)) => {
                assert!(fields.insert(w.slot, w.canonical).is_none());
            }
            Step::Unit(Unit::BlockPresence {
                group: Some(group), ..
            }) => {
                for w in group.fields {
                    assert!(fields.insert(w.slot, w.canonical).is_none());
                }
            }
            _ => {}
        }
    }
    fields
}
fn block(n: u32, field: B) -> Slot {
    Slot::Block {
        block: BlockSlot::Item(n),
        field,
    }
}
#[test]
fn direct_false_zero_empty_payloads_and_extras_count_but_inferred_fields_do_not() {
    let bytes = header()
        + &line("turn_context", json!({"model":""}))
        + &line(
            "response_item",
            json!({"type":"function_call","id":"call","name":"exec_command","call_id":"c","arguments":"{\"cmd\":\"\"}","extra_empty":{},"extra_false":false}),
        )
        + &line(
            "response_item",
            json!({"type":"function_call_output","id":"result","call_id":"c","output":{"output":"","metadata":{"is_error":false}}}),
        )
        + &line(
            "event_msg",
            json!({"type":"item_completed","turn_id":"t","item":{"type":"CommandExecution","id":"c","source":"agent","status":"completed","exit_code":0,"duration_ms":0}}),
        )
        + &line(
            "event_msg",
            json!({"type":"token_count","info":{"last_token_usage":{"input_tokens":0,"output_tokens":0,"cached_input_tokens":0},"total_token_usage":{"input_tokens":0}}}),
        );
    let (view, _) = parse(&bytes);
    let call = slots(&view, 0);
    let result = slots(&view, 1);
    assert_eq!(call[&Slot::Message(M::Model)], br#""""#.to_vec());
    assert_eq!(call[&block(0, B::Command)], br#""""#.to_vec());
    assert_eq!(
        call[&Slot::Extra {
            block: Some(BlockSlot::Item(0)),
            key: "extra_empty".into()
        }],
        b"{}"
    );
    assert_eq!(
        call[&Slot::Extra {
            block: Some(BlockSlot::Item(0)),
            key: "extra_false".into()
        }],
        b"false"
    );
    assert_eq!(result[&block(0, B::ExitCode)], b"0");
    assert_eq!(result[&block(0, B::DurationMs)], b"0");
    assert_eq!(result[&block(0, B::Outcome)], br#""completed""#);
    assert!(result.contains_key(&block(0, B::Output)));
    assert_eq!(call[&Slot::Usage(U::Input)], b"0");
    assert_eq!(call[&Slot::Usage(U::Output)], b"0");
    assert!(!call.contains_key(&Slot::Usage(U::ProviderCostNusd)));
    assert!(!call.contains_key(&Slot::Message(M::PromptKind)));
    assert!(call.contains_key(&Slot::Message(M::At)));
    assert!(call.keys().all(
        |slot| !matches!(slot,Slot::Extra{key,..} if key=="id"||key=="type"||key=="timestamp")
    ));
}
#[test]
fn empty_text_excludes_its_metadata_while_known_empty_nontext_payloads_survive() {
    let bytes = header()
        + &line(
            "response_item",
            json!({"type":"message","id":"empty","role":"user","content":[],"extra":{},"phase":"final"}),
        )
        + &line(
            "response_item",
            json!({"type":"reasoning","id":"reason","summary":[],"encrypted_content":""}),
        )
        + &line(
            "response_item",
            json!({"type":"custom_tool_call","id":"custom","name":"tool","call_id":"c","input":null}),
        );
    let (view, _) = parse(&bytes);
    let user = slots(&view, 0);
    assert!(
        user.keys()
            .all(|slot| !matches!(slot, Slot::Block { .. } | Slot::Extra { .. }))
    );
    assert!(
        matches!(view.cells.get(&Key::PresenceBlock(0,0)),Some(Cell::Presence(g)) if !g.nonempty && g.fields.is_empty())
    );
    let assistant = slots(&view, 1);
    assert_eq!(
        assistant[&Slot::Block {
            block: BlockSlot::Reasoning(0),
            field: B::ThinkingText
        }],
        b"[]"
    );
    assert!(!assistant.contains_key(&Slot::Block {
        block: BlockSlot::Reasoning(0),
        field: B::Encrypted
    }));
    assert_eq!(assistant[&block(0, B::Input)], b"null");
    assert!(!assistant.keys().any(
        |slot| matches!(slot,Slot::Extra{key,..} if key=="continuation_calls"||key=="code_children")
    ));
}
#[test]
fn effective_overlay_and_line_steps_keep_exact_fields_once() {
    let bytes = header()
        + &line(
            "response_item",
            json!({"type":"function_call","id":"call","name":"tool","call_id":"c","arguments":"{}","extra":[]}),
        )
        + &line(
            "response_item",
            json!({"type":"function_call","id":"call","arguments":"null"}),
        );
    let (whole, resume) = parse(&bytes);
    let expected = slots(&whole, 0);
    assert!(expected.contains_key(&block(0, B::Tool)));
    assert_eq!(expected[&block(0, B::Input)], br#""null""#);
    assert_eq!(
        expected[&Slot::Extra {
            block: Some(BlockSlot::Item(0)),
            key: "extra".into()
        }],
        b"[]"
    );
    let mut view = MemoryReadView::default();
    let mut state = Resume::default();
    for record in bytes.as_bytes().split_inclusive(|b| *b == b'\n') {
        let framed = drive_jsonl(KeyedRecords::new(&view, state), record, 0);
        let delta = framed.output.unwrap();
        view.apply(&delta);
        state = delta.resume;
    }
    assert_eq!(state.message_count(), resume.message_count());
    assert_eq!(slots(&view, 0), expected);
}
#[test]
fn textual_exit_is_not_a_supplied_outcome_and_unowned_native_facts_stay_outside_messages() {
    let bytes = header()
        + &line(
            "response_item",
            json!({"type":"function_call","id":"call","name":"exec_command","call_id":"c","arguments":"{}"}),
        )
        + &line(
            "response_item",
            json!({"type":"function_call_output","id":"result","call_id":"c","output":"Exit code: 0"}),
        )
        + &line(
            "event_msg",
            json!({"type":"item_completed","item":{"type":"CommandExecution","id":"unowned","status":"completed","exit_code":0}}),
        );
    let (view, resume) = parse(&bytes);
    let result = slots(&view, 1);
    assert!(!result.contains_key(&block(0, B::Outcome)));
    assert!(!result.contains_key(&block(0, B::ExitCode)));
    assert_eq!(resume.message_count(), 2);
    assert!(resume.normalization_incomplete());
}

#[test]
fn intrinsic_time_provenance_is_separate_from_volatile_free_version_units() {
    let bytes=header()+&line("response_item",json!({"type":"function_call","id":"call","name":"tool","call_id":"c","arguments":"{}"}))
        +&json!({"timestamp":"2026-10-09T14:31:00Z","type":"response_item","payload":{"type":"message","id":"answer","role":"assistant","content":[{"text":"hi"}]}}).to_string()+"\n"
        +&json!({"type":"response_item","payload":{"type":"message","id":"no-time","role":"user","content":[{"text":"next"}]}}).to_string()+"\n"
        +&json!({"timestamp":"unreadable","type":"response_item","payload":{"type":"message","id":"bad-time","role":"assistant","content":[{"text":"last"}]}}).to_string()+"\n";
    let (view, _) = parse(&bytes);
    let first = slots(&view, 0);
    assert_eq!(first[&Slot::Message(M::At)], br#""2026-10-09T14:30:00Z""#);
    assert_eq!(
        first[&Slot::Message(M::CompletedAt)],
        br#""2026-10-09T14:31:00Z""#
    );
    for message in [1, 2] {
        assert!(!slots(&view, message).contains_key(&Slot::Message(M::At)));
    }
    for (key, cell) in &view.cells {
        if matches!(key, Key::Evidence(_, _, _)) {
            if let Cell::Evidence(unit) = cell {
                if unit.whole_record {
                    let value: Value = serde_json::from_slice(&unit.canonical).unwrap();
                    assert!(value.get("timestamp").is_none());
                }
            }
        }
    }
}

#[test]
fn native_children_and_legacy_array_results_keep_only_direct_fields() {
    let bytes = header()
        + &line(
            "response_item",
            json!({"type":"custom_tool_call","id":"wrapper","name":"exec","call_id":"w","input":""}),
        )
        + &line(
            "event_msg",
            json!({"type":"item_completed","turn_id":"t","item":{"type":"CommandExecution","id":"child","parent_call_id":"w","source":"agent","command":"","aggregated_output":"","status":"completed","exit_code":0,"duration":{"secs":0,"nanos":0},"extra":{}}}),
        );
    let (view, _) = parse(&bytes);
    let fields = slots(&view, 0);
    assert!(fields.contains_key(&block(1, B::Command)));
    assert!(!fields.contains_key(&block(1, B::Tool)));
    assert!(!fields.contains_key(&block(1, B::CallId)));
    assert!(!fields.contains_key(&block(1, B::Input)));
    assert_eq!(fields[&block(2, B::DurationMs)], br#"{"nanos":0,"secs":0}"#);
    assert_eq!(fields[&block(2, B::Output)], br#""""#);
    assert!(fields.contains_key(&Slot::Extra {
        block: Some(BlockSlot::Item(1)),
        key: "extra".into()
    }));
    assert!(!fields.contains_key(&Slot::Extra {
        block: Some(BlockSlot::Item(2)),
        key: "extra".into()
    }));
    let legacy=[json!({"id":"legacy:synthetic","timestamp":"2025-08-11T12:00:00Z"}),json!({"type":"function_call","id":"call","name":"tool","call_id":"c","arguments":"{}"}),json!({"type":"function_call_output","id":"result","call_id":"c","output":[{"text":"{\"is_error\":false}"}]})].iter().map(|v|v.to_string()+"\n").collect::<String>();
    let (view, _) = parse(&legacy);
    assert_eq!(slots(&view, 1)[&block(0, B::Outcome)], b"false");
}

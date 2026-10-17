//! Source-local correspondence laws; no private captures or product qualification.
use polylogue_sources::{
    codex::keyed::{Cell, Key, KeyedRecords, Limits, MemoryReadView, Resume},
    frame::{JsonlDriver, StepBudget, drive_jsonl},
};
use serde_json::{Value, json};
fn line(kind: &str, payload: Value) -> String {
    json!({"type":kind,"payload":payload}).to_string() + "\n"
}
fn start() -> String {
    line("session_meta", json!({"id":"neutral","source":"cli"})) + &turn("T")
}
fn turn(id: &str) -> String {
    line("turn_context", json!({"turn_id":id}))
}
fn response(kind: &str, id: &str, text: &str) -> String {
    let p = match kind {
        "reasoning" => {
            json!({"type":"reasoning","id":id,"summary":[],"encrypted_content":"opaque encrypted value","internal_chat_message_metadata_passthrough":{"turn_id":"T"}})
        }
        role => {
            json!({"type":"message","id":id,"role":role,"content":[{"type":if role=="user"{"input_text"}else{"output_text"},"text":text}],"phase":"final_answer","internal_chat_message_metadata_passthrough":{"turn_id":"T"}})
        }
    };
    line("response_item", p)
}
fn completed(kind: &str, id: &str, text: &str) -> String {
    let item = match kind {
        "Reasoning" => json!({"type":kind,"id":id,"summary_text":[],"raw_content":[]}),
        "UserMessage" => {
            json!({"type":kind,"id":id,"content":[{"type":"text","text":text,"text_elements":[]}]})
        }
        _ => {
            json!({"type":kind,"id":id,"content":[{"type":"Text","text":text}],"phase":"final_answer"})
        }
    };
    line(
        "event_msg",
        json!({"type":"item_completed","thread_id":"neutral","turn_id":"T","item":item}),
    )
}
fn edited(record: &str, f: impl FnOnce(&mut Value)) -> String {
    let mut v: Value = serde_json::from_str(record).unwrap();
    f(&mut v);
    v.to_string() + "\n"
}
fn run(view: MemoryReadView, resume: Resume, record: &str) -> (MemoryReadView, Resume) {
    let framed = drive_jsonl(
        KeyedRecords::new(&view, resume.clone()),
        record.as_bytes(),
        resume.parsed_upto,
    );
    assert!(framed.counts.balanced());
    let mut d = framed.output.unwrap();
    d.resume.parsed_upto = framed.parsed_upto;
    let mut view = view;
    view.apply(&d);
    view.cells = view
        .cells
        .into_iter()
        .map(|(key, cell)| {
            let k = Key::from_ordered_bytes(&key.ordered_bytes()).unwrap();
            let c: Cell = serde_json::from_slice(&serde_json::to_vec(&cell).unwrap()).unwrap();
            assert!(c.valid_for(&k));
            (k, c)
        })
        .collect();
    let state = serde_json::from_slice(&serde_json::to_vec(&d.resume).unwrap()).unwrap();
    (view, state)
}
fn messages(v: &MemoryReadView, s: &Resume) -> Value {
    json!(
        (0..s.message_count())
            .map(|i| v.message(i).unwrap().unwrap())
            .collect::<Vec<_>>()
    )
}
#[test]
fn native_and_response_orders_preserve_one_exact_message_and_encrypted_reasoning() {
    for (native, source) in [("AgentMessage", "assistant"), ("Reasoning", "reasoning")] {
        for native_first in [false, true] {
            let (v, s) = run(MemoryReadView::default(), Resume::default(), &start());
            let expected = response(source, "I", "neutral text");
            let (baseline, bs) = run(v.clone(), s.clone(), &expected);
            let (v, s) = if native_first {
                let (v, s) = run(v, s, &completed(native, "I", "neutral text"));
                assert!(s.normalization_incomplete());
                assert_eq!(s.message_count(), 0);
                run(v, s, &expected)
            } else {
                let (v, s) = run(v, s, &expected);
                run(v, s, &completed(native, "I", "neutral text"))
            };
            assert!(
                !s.normalization_incomplete(),
                "{native} first={native_first}"
            );
            assert_eq!(s.message_count(), 1);
            assert_eq!(messages(&v, &s), messages(&baseline, &bs));
        }
    }
}
#[test]
fn unmatched_native_eof_is_explicit_without_fabricating_messages() {
    for native in ["AgentMessage", "Reasoning", "UserMessage"] {
        let (_, s) = run(
            MemoryReadView::default(),
            Resume::default(),
            &(start() + &completed(native, "I", "text")),
        );
        assert!(s.normalization_incomplete());
        assert_eq!(s.message_count(), 0);
    }
}
#[test]
fn invalid_scope_unknown_members_and_unimplemented_constructors_stay_incomplete() {
    let base = completed("AgentMessage", "I", "text");
    for bad in [
        edited(&base, |v| v["payload"]["thread_id"] = json!("other")),
        edited(&base, |v| v["payload"]["turn_id"] = Value::Null),
        edited(&base, |v| {
            v["payload"].as_object_mut().unwrap().remove("turn_id");
        }),
        edited(&base, |v| v["payload"]["item"]["future"] = json!(true)),
        edited(&base, |v| v["payload"]["future"] = json!(true)),
        edited(&base, |v| {
            v["payload"]["item"]["content"][0]["future"] = json!(1)
        }),
        completed("FutureMessage", "I", "text"),
    ] {
        for first in [false, true] {
            let input = if first {
                start() + &bad + &response("assistant", "I", "text")
            } else {
                start() + &response("assistant", "I", "text") + &bad
            };
            let (_, s) = run(MemoryReadView::default(), Resume::default(), &input);
            assert!(s.normalization_incomplete());
        }
    }
}
#[test]
fn mismatched_id_turn_text_phase_and_role_never_prove_correspondence() {
    let native = completed("AgentMessage", "I", "text");
    for bad in [
        response("assistant", "other", "text"),
        response("assistant", "I", "other"),
        response("user", "I", "text"),
        edited(&response("assistant", "I", "text"), |v| {
            v["payload"]["phase"] = json!("commentary")
        }),
        edited(&response("assistant", "I", "text"), |v| {
            v["payload"]["internal_chat_message_metadata_passthrough"]["turn_id"] = json!("other")
        }),
    ] {
        for first in [false, true] {
            let input = if first {
                start() + &native + &bad
            } else {
                start() + &bad + &native
            };
            let (_, s) = run(MemoryReadView::default(), Resume::default(), &input);
            assert!(s.normalization_incomplete());
        }
    }
}
#[test]
fn native_reasoning_does_not_claim_unrepresented_plaintext_or_unknown_summary() {
    for bad in [
        edited(&completed("Reasoning", "I", ""), |v| {
            v["payload"]["item"]["raw_content"] = json!(["private plaintext"])
        }),
        edited(&completed("Reasoning", "I", ""), |v| {
            v["payload"]["item"]["summary_text"] = json!(["not present"])
        }),
        edited(&completed("Reasoning", "I", ""), |v| {
            v["payload"]["item"]["summary_text"] = Value::Null
        }),
    ] {
        let (v, s) = run(
            MemoryReadView::default(),
            Resume::default(),
            &(start() + &response("reasoning", "I", "") + &bad),
        );
        assert!(s.normalization_incomplete());
        let doc = messages(&v, &s).to_string();
        assert!(doc.contains("opaque encrypted value"));
        assert!(!doc.contains("private plaintext"));
    }
}
#[test]
fn exact_summary_can_correspond_without_replacing_its_source_evidence() {
    let native = edited(&completed("Reasoning", "I", ""), |v| {
        v["payload"]["item"]["summary_text"] = json!(["brief"])
    });
    let source = edited(&response("reasoning", "I", ""), |v| {
        v["payload"]["summary"] = json!([{"type":"summary_text","text":"brief"}])
    });
    let (v, s) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &native + &source),
    );
    assert!(!s.normalization_incomplete());
    assert!(messages(&v, &s).to_string().contains("brief"));
}
#[test]
fn later_corrections_and_constructor_changes_invalidate_exact_current_proof() {
    let (v, s) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &completed("AgentMessage", "I", "text") + &response("assistant", "I", "text")),
    );
    assert!(!s.normalization_incomplete());
    for correction in [
        completed("AgentMessage", "I", "changed"),
        response("assistant", "I", "changed"),
        edited(&response("assistant", "I", "text"), |v| {
            v["payload"]["internal_chat_message_metadata_passthrough"]["turn_id"] = json!("other")
        }),
        completed("Reasoning", "I", ""),
        completed("FutureMessage", "I", "text"),
    ] {
        let (_, s) = run(v.clone(), s.clone(), &correction);
        assert!(s.normalization_incomplete());
    }
}
#[test]
fn resolving_only_the_matching_pending_component_cannot_clear_unrelated_failures() {
    let (v, s) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start()
            + &completed("FutureMessage", "bad", "")
            + &completed("AgentMessage", "I", "text")),
    );
    let (_, s) = run(v, s, &response("assistant", "I", "text"));
    assert!(s.normalization_incomplete());
    let (v, s) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &completed("AgentMessage", "I", "text") + &completed("Reasoning", "R", "")),
    );
    let (_, s) = run(v, s, &response("assistant", "I", "text"));
    assert!(s.normalization_incomplete());
}
#[test]
fn same_id_in_another_turn_cannot_borrow_response_scope() {
    let (v, s) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &response("assistant", "I", "text") + &turn("other")),
    );
    let native = edited(&completed("AgentMessage", "I", "text"), |v| {
        v["payload"]["turn_id"] = json!("other")
    });
    let (_, s) = run(v, s, &native);
    assert!(s.normalization_incomplete());
}
#[test]
fn user_echo_is_first_unpaired_same_turn_and_keeps_original_message() {
    let (v, s) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &response("user", "U", "prompt")),
    );
    let before = messages(&v, &s);
    let (v, s) = run(v, s, &completed("UserMessage", "echo-1", "prompt"));
    assert!(!s.normalization_incomplete());
    assert_eq!(messages(&v, &s), before);
    let (v, s) = run(v, s, &completed("UserMessage", "echo-1", "prompt"));
    assert!(!s.normalization_incomplete());
    assert_eq!(messages(&v, &s), before);
    let (_, s) = run(v, s, &completed("UserMessage", "echo-2", "prompt"));
    assert!(s.normalization_incomplete());
}
#[test]
fn user_echo_before_prompt_wrong_turn_or_next_prompt_cannot_pair_by_text() {
    let native = completed("UserMessage", "echo", "prompt");
    for input in [
        start() + &native + &response("user", "U", "prompt"),
        start()
            + &response("user", "U", "prompt")
            + &edited(&native, |v| v["payload"]["turn_id"] = json!("other")),
        start() + &response("user", "U", "prompt") + &response("user", "V", "next") + &native,
    ] {
        let (_, s) = run(MemoryReadView::default(), Resume::default(), &input);
        assert!(s.normalization_incomplete());
    }
}
#[test]
fn equal_prompts_in_separate_turns_remain_distinct_with_separate_echoes() {
    let second = edited(&response("user", "V", "prompt"), |v| {
        v["payload"]["internal_chat_message_metadata_passthrough"]["turn_id"] = json!("other")
    });
    let echo = edited(&completed("UserMessage", "echo-2", "prompt"), |v| {
        v["payload"]["turn_id"] = json!("other")
    });
    let (_, s) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start()
            + &response("user", "U", "prompt")
            + &completed("UserMessage", "echo-1", "prompt")
            + &turn("other")
            + &second
            + &echo),
    );
    assert!(!s.normalization_incomplete());
    assert_eq!(s.message_count(), 2);
}
#[test]
fn editing_the_paired_user_response_invalidates_its_echo_correspondence() {
    let (v, s) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &response("user", "U", "prompt") + &completed("UserMessage", "echo", "prompt")),
    );
    assert!(!s.normalization_incomplete());
    let (_, s) = run(v, s, &response("user", "U", "changed"));
    assert!(s.normalization_incomplete());
}
#[test]
fn per_record_stop_and_serialized_restart_preserve_the_pending_obligation() {
    let (v, s) = run(MemoryReadView::default(), Resume::default(), &start());
    let input = completed("AgentMessage", "I", "text") + &response("assistant", "I", "text");
    let mut driver = JsonlDriver::new(KeyedRecords::new(&v, s.clone()), s.parsed_upto);
    let step = driver
        .advance(
            input.as_bytes(),
            StepBudget {
                bytes: input.len(),
                records: 1,
                record_bytes: 1024 * 1024,
            },
        )
        .unwrap();
    let framed = driver.finish();
    let mut d = framed.output.unwrap();
    d.resume.parsed_upto = framed.parsed_upto;
    assert!(d.resume.normalization_incomplete());
    let mut v = v;
    v.apply(&d);
    let state: Resume = serde_json::from_slice(&serde_json::to_vec(&d.resume).unwrap()).unwrap();
    let (_, s) = run(v, state, &input[step.consumed..]);
    assert!(!s.normalization_incomplete());
}
#[test]
fn pending_claims_respect_existing_cell_and_encoding_admission() {
    let (v, s) = run(MemoryReadView::default(), Resume::default(), &start());
    for limits in [
        Limits {
            changed_cells: 0,
            encoded_bytes: usize::MAX,
        },
        Limits {
            changed_cells: usize::MAX,
            encoded_bytes: 0,
        },
    ] {
        let f = drive_jsonl(
            KeyedRecords::with_limits(&v, s.clone(), limits),
            completed("AgentMessage", "I", "text").as_bytes(),
            s.parsed_upto,
        );
        assert!(f.output.is_err());
    }
    assert!(
        v.cells
            .keys()
            .all(|k| !matches!(k, Key::CompletedMessage(..)))
    );
}

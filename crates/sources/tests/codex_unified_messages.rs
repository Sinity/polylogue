use polylogue_model::tool_item::Operation;
use polylogue_sources::{
    codex::keyed::{Cell, Key, KeyedRecords, Limits, MemoryReadView, Resume},
    frame::drive_jsonl,
};
use serde_json::{Value, json};

const OUTPUT: &str = "Chunk ID: 123abc\nWall time: 0.001 seconds\nProcess exited with code 0\nFinal output:\nneutral\n";

fn line(kind: &str, payload: Value) -> String {
    json!({"type":kind,"payload":payload}).to_string() + "\n"
}
fn start() -> String {
    line("session_meta", json!({"id":"neutral","source":"cli"}))
        + &line("turn_context", json!({"turn_id":"T"}))
}
fn call(output: bool) -> String {
    let mut records = line(
        "response_item",
        json!({"type":"function_call","id":"call-message","call_id":"C","name":"exec_command","arguments":"{\"cmd\":\"printf neutral\"}"}),
    );
    if output {
        records += &line(
            "response_item",
            json!({"type":"function_call_output","call_id":"C","output":"Chunk ID: 123abc\nWall time: 0.001 seconds\nProcess exited with code 0\nFinal output:\nneutral\n"}),
        );
    }
    records
}
fn completed(source: &str, thread: &str, turn: &str, id: &str, duration: Value) -> String {
    line(
        "event_msg",
        json!({"type":"item_completed","thread_id":thread,"turn_id":turn,"started_at_ms":1,"completed_at_ms":2,"item":{"type":"CommandExecution","id":id,"source":source,"command":["printf","neutral"],"status":"completed","exit_code":0,"aggregated_output":"native output is not a Message result","duration":duration}}),
    )
}
fn completion(source: &str) -> String {
    completed(
        source,
        "neutral",
        "T",
        "C",
        json!({"secs":0,"nanos":1000000}),
    )
}
fn run(view: MemoryReadView, resume: Resume, records: &str) -> (MemoryReadView, Resume) {
    let framed = drive_jsonl(
        KeyedRecords::new(&view, resume.clone()),
        records.as_bytes(),
        resume.parsed_upto,
    );
    let mut delta = framed.output.unwrap();
    delta.resume.parsed_upto = framed.parsed_upto;
    let mut view = view;
    view.apply(&delta);
    view.cells = view
        .cells
        .into_iter()
        .map(|(key, cell)| {
            let key = Key::from_ordered_bytes(&key.ordered_bytes()).unwrap();
            let cell: Cell = serde_json::from_slice(&serde_json::to_vec(&cell).unwrap()).unwrap();
            assert!(cell.valid_for(&key));
            (key, cell)
        })
        .collect();
    // Reconstruct through the same serialized continuation/cells used on restart.
    let resume = serde_json::from_slice(&serde_json::to_vec(&delta.resume).unwrap()).unwrap();
    (view, resume)
}
fn operation(view: &MemoryReadView, turn: &str, id: &str) -> Operation {
    view.cells
        .values()
        .find_map(|cell| match cell {
            Cell::Operation(op) if op.turn.as_deref() == Some(turn) && op.invocation == id => {
                Some(op.clone())
            }
            _ => None,
        })
        .unwrap()
}
fn messages(view: &MemoryReadView, resume: &Resume) -> Vec<serde_json::Value> {
    (0..resume.message_count())
        .map(|i| serde_json::to_value(view.message(i).unwrap().unwrap()).unwrap())
        .collect()
}

#[test]
fn unified_attribution_keeps_independent_messages_and_no_native_result_in_both_orders() {
    for source in [
        "unified_exec_startup",
        "unifiedExecStartup",
        "unified_exec_interaction",
        "unifiedExecInteraction",
    ] {
        for native_first in [false, true] {
            let (view, resume) = run(MemoryReadView::default(), Resume::default(), &start());
            let (view, resume) = if native_first {
                let (view, resume) = run(view, resume, &completion(source));
                run(view, resume, &call(true))
            } else {
                let (view, resume) = run(view, resume, &call(true));
                let before = messages(&view, &resume);
                let (view, resume) = run(view, resume, &completion(source));
                assert_eq!(messages(&view, &resume), before);
                (view, resume)
            };
            assert!(
                !resume.normalization_incomplete(),
                "{source} first={native_first}"
            );
            assert_eq!(resume.message_count(), 2);
            let op = operation(&view, "T", "C");
            assert_eq!(op.agent_class, "unknown");
            assert!(!op.counted() && op.result_present);
            assert_eq!(op.output.as_deref(), Some(OUTPUT));
            assert_eq!(op.outcome.as_deref(), Some("ok"));
            assert_eq!(op.exit_code, Some(0));
            assert_eq!(op.duration_ms, Some(1));
            assert!(op.origins.iter().any(|o| matches!(o, polylogue_model::tool_item::ToolOrigin::Rollout { tag, .. } if tag=="result")));
            assert!(op.message.is_some());
            assert_eq!(op.tool.as_deref(), Some("exec_command"));
            assert!(
                view.cells
                    .contains_key(&Key::UnifiedOperation(op.id().unwrap()))
            );
            assert!(op.origins.iter().any(|o| matches!(o, polylogue_model::tool_item::ToolOrigin::Rollout { tag, .. } if tag=="status")));
            assert!(op.origins.iter().any(|o| matches!(o, polylogue_model::tool_item::ToolOrigin::Rollout { tag, .. } if tag=="call")));
        }
    }
}

#[test]
fn absent_output_and_unowned_items_do_not_gain_a_result_or_latest_wrapper_parent() {
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &call(false)),
    );
    let before = messages(&view, &resume);
    let (view, resume) = run(view, resume, &completion("unified_exec_startup"));
    assert_eq!(messages(&view, &resume), before);
    assert_eq!(resume.message_count(), 1);
    assert!(!operation(&view, "T", "C").result_present);
    let (view, resume) = run(
        view,
        resume,
        &completed(
            "unified_exec_startup",
            "neutral",
            "T",
            "unowned",
            json!({"secs":0,"nanos":0}),
        ),
    );
    assert_eq!(messages(&view, &resume), before);
    let op = operation(&view, "T", "unowned");
    assert!(op.message.is_none() && op.wrapper.is_none());
    assert!(!op.counted() && !op.result_present);
}

#[test]
fn bad_scope_source_or_duration_never_links_a_unified_operation() {
    for record in [
        completion("user_shell"),
        completion("future_source"),
        completed(
            "unified_exec_startup",
            "foreign",
            "T",
            "C",
            json!({"secs":0,"nanos":0}),
        ),
        completed(
            "unified_exec_startup",
            "neutral",
            "T",
            "C",
            json!({"secs":0,"nanos":1000000000_u64}),
        ),
    ] {
        let (view, resume) = run(
            MemoryReadView::default(),
            Resume::default(),
            &(start() + &call(true)),
        );
        let (view, resume) = run(view, resume, &record);
        assert!(resume.normalization_incomplete());
        assert!(
            !view
                .cells
                .keys()
                .any(|k| matches!(k, Key::UnifiedOperation(_)))
        );
    }
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &call(true)),
    );
    let before = messages(&view, &resume);
    let (view, resume) = run(
        view,
        resume,
        &completed(
            "unified_exec_startup",
            "neutral",
            "other-turn",
            "C",
            json!({"secs":0,"nanos":0}),
        ),
    );
    assert_eq!(messages(&view, &resume), before);
    assert!(operation(&view, "other-turn", "C").message.is_none());
}

#[test]
fn a_new_constructor_replaces_the_previous_record_in_both_orders() {
    for native_first in [false, true] {
        for replacement in [
            completion("agent"),
            completion("future_source"),
            completed(
                "unified_exec_startup",
                "neutral",
                "T",
                "C",
                json!({"secs":"invalid","nanos":0}),
            ),
        ] {
            let prefix = if native_first {
                start()
            } else {
                start() + &call(true)
            };
            let (view, resume) = run(
                MemoryReadView::default(),
                Resume::default(),
                &(prefix + &completion("unified_exec_startup")),
            );
            let (view, resume) = run(view, resume, &replacement);
            assert!(
                !view
                    .cells
                    .keys()
                    .any(|k| matches!(k, Key::UnifiedOperation(_)))
            );
            let (view, _) = if native_first {
                run(view, resume, &call(true))
            } else {
                (view, resume)
            };
            assert!(
                !view
                    .cells
                    .keys()
                    .any(|k| matches!(k, Key::UnifiedOperation(_)))
            );
            let op = operation(&view, "T", "C");
            if op.agent_class == "unknown" {
                assert!(!op.counted());
                if op.result_present {
                    assert_eq!(op.output.as_deref(), Some(OUTPUT));
                }
            }
        }
    }
}

#[test]
fn a_different_status_record_cannot_associate_a_later_call() {
    let (mut view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &completion("unified_exec_startup")),
    );
    for cell in view.cells.values_mut() {
        if let Cell::UnifiedOperation { raw_hash, .. } = cell {
            raw_hash[0] ^= 1;
        }
    }
    let (view, _) = run(view, resume, &call(true));
    let op = operation(&view, "T", "C");
    assert!(op.message.is_none());
    assert_eq!(op.agent_class, "unknown");
    assert!(!op.counted());
    assert_eq!(op.output.as_deref(), Some(OUTPUT));
}

#[test]
fn unified_operation_writes_respect_cell_limits() {
    let (view, resume) = run(MemoryReadView::default(), Resume::default(), &start());
    let record = completion("unified_exec_startup");
    let framed = drive_jsonl(
        KeyedRecords::with_limits(
            &view,
            resume.clone(),
            Limits {
                changed_cells: 0,
                encoded_bytes: 0,
            },
        ),
        record.as_bytes(),
        resume.parsed_upto,
    );
    assert!(framed.output.is_err());
    assert!(
        !view
            .cells
            .keys()
            .any(|k| matches!(k, Key::UnifiedOperation(_)))
    );
}

#[test]
fn supported_evidence_never_clears_an_earlier_normalization_failure() {
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &call(true) + &completion("future_source")),
    );
    assert!(resume.normalization_incomplete());
    let (_, resume) = run(view, resume, &completion("unified_exec_startup"));
    assert!(resume.normalization_incomplete());
}

#[test]
fn unified_native_items_do_not_borrow_an_exec_wrapper_parent() {
    let wrapper = |id: &str| {
        line(
            "response_item",
            json!({"type":"function_call","id":format!("wrapper-{id}"),"call_id":id,"name":"exec","arguments":"{}"}),
        )
    };
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &wrapper("W1") + &wrapper("W2")),
    );
    let before = messages(&view, &resume);
    let (view, resume) = run(view, resume, &completion("unified_exec_startup"));
    assert_eq!(messages(&view, &resume), before);
    let op = operation(&view, "T", "C");
    assert!(op.message.is_none() && op.wrapper.is_none());
    assert_eq!(op.wrapper_overlaps.len(), 2);
    assert!(!op.result_present && !op.counted());
}

#[test]
fn conflicting_constructor_drops_the_link_without_changing_messages() {
    let (view, resume) = run(
        MemoryReadView::default(),
        Resume::default(),
        &(start() + &call(true) + &completion("unified_exec_startup")),
    );
    let before = messages(&view, &resume);
    let conflict = line(
        "event_msg",
        json!({"type":"item_completed","thread_id":"neutral","turn_id":"T","item":{"type":"FileChange","id":"C","status":"completed","changes":[]}}),
    );
    let (view, resume) = run(view, resume, &conflict);
    assert!(resume.normalization_incomplete());
    assert_eq!(messages(&view, &resume), before);
    assert!(
        !view
            .cells
            .keys()
            .any(|k| matches!(k, Key::UnifiedOperation(_)))
    );
}

#[test]
fn an_unscoped_call_cannot_borrow_a_later_native_turn_for_read_permission() {
    let prefix = line("session_meta", json!({"id":"neutral","source":"cli"})) + &call(true);
    let (view, resume) = run(MemoryReadView::default(), Resume::default(), &prefix);
    let (_, resume) = run(view, resume, &completion("unified_exec_startup"));
    assert!(resume.normalization_incomplete());
}

#[test]
fn unsupported_native_first_records_cannot_gain_later_call_readability_or_classification() {
    for record in [
        completion("user_shell"),
        completion("future_source"),
        completed(
            "unified_exec_startup",
            "neutral",
            "T",
            "C",
            json!({"secs":"invalid","nanos":0}),
        ),
    ] {
        let (view, resume) = run(
            MemoryReadView::default(),
            Resume::default(),
            &(start() + &record),
        );
        let (view, resume) = run(view, resume, &call(true));
        assert!(resume.normalization_incomplete());
        assert!(
            !view
                .cells
                .keys()
                .any(|k| matches!(k, Key::UnifiedOperation(_)))
        );
        let op = operation(&view, "T", "C");
        assert_ne!(op.agent_class, "agent");
        assert!(!op.counted());
        assert!(op.message.is_none());
        if op.result_present {
            assert_eq!(op.output.as_deref(), Some(OUTPUT));
        }
    }
}

#[test]
fn a_completion_without_its_own_turn_cannot_reuse_an_old_scope() {
    for source in ["unified_exec_startup", "future_source"] {
        for turn in [Value::Null, json!(123)] {
            let (view, resume) = run(
                MemoryReadView::default(),
                Resume::default(),
                &(start() + &call(true) + &completion("unified_exec_startup")),
            );
            assert!(!resume.normalization_incomplete());
            let before = messages(&view, &resume);
            let mut record: Value = serde_json::from_str(&completion(source)).unwrap();
            if turn.is_null() {
                record["payload"].as_object_mut().unwrap().remove("turn_id");
            } else {
                record["payload"]["turn_id"] = turn;
            }
            let (view, resume) = run(view, resume, &(record.to_string() + "\n"));
            assert!(resume.normalization_incomplete());
            assert_eq!(messages(&view, &resume), before);
        }
    }
}

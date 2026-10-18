//! Codex usage descriptors recompute values from the original numbers.
use polylogue_model::{merge::Slot, source_leaf};
use polylogue_sources::{
    codex::{
        keyed::{Cell, Key, KeyedRecords, MemoryReadView, Resume},
        selection::Transform,
    },
    frame::drive_jsonl,
};
use serde_json::{Value, json};

fn usage_view(usage: Value) -> (String, MemoryReadView) {
    let records = [
        json!({"type":"session_meta","payload":{"id":"neutral-usage-session"}}),
        json!({"type":"response_item","payload":{"type":"message","id":"answer","role":"assistant","content":[{"type":"output_text","text":"Neutral answer"}]}}),
        json!({"type":"event_msg","payload":{"type":"token_count","info":{"last_token_usage":usage}}}),
    ];
    let bytes = records
        .iter()
        .map(|r| r.to_string() + "\n")
        .collect::<String>();
    let delta = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        bytes.as_bytes(),
        0,
    )
    .output
    .unwrap();
    let mut view = MemoryReadView::default();
    view.apply(&delta);
    (bytes, view)
}

#[test]
fn supplied_usage_fields_reproduce_normalized_leaf_and_preserve_zero() {
    for (total, cached, expected) in [
        (10, 6, 4),
        (0, 0, 0),
        (2, 7, 0),
        (u64::MAX, 1, u64::MAX - 1),
    ] {
        let (bytes, view) = usage_view(json!({
            "input_tokens":total,"cached_input_tokens":cached,"output_tokens":0,
            "cache_write_input_tokens":3,"reasoning_output_tokens":1
        }));
        let message = view.projected_message(0).unwrap().unwrap();
        assert_eq!(message.usage.as_ref().unwrap().input, Some(expected));
        for name in [
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "cache_write_input_tokens",
            "reasoning_output_tokens",
        ] {
            let Cell::SlotValue(witness) = &view.cells[&Key::Presence(0, format!("usage:{name}"))]
            else {
                panic!("usage witness")
            };
            assert!(matches!(witness.slot, Slot::Usage(_)));
            let selection = witness
                .selection
                .as_ref()
                .expect("numeric usage has source selection");
            let mut output = Vec::new();
            selection
                .write_value(
                    |range| {
                        Ok::<_, ()>(
                            bytes.as_bytes()[range.start as usize..range.end as usize].to_vec(),
                        )
                    },
                    bytes.len(),
                    &mut output,
                )
                .unwrap();
            let expected =
                source_leaf::encode(source_leaf::Subject::Message(&message), &witness.slot, 128)
                    .unwrap();
            assert_eq!(source_leaf::Projection::Encoded(output), expected, "{name}");
            if name == "input_tokens" {
                assert_eq!(selection.transform, Transform::UncachedInputV1);
                assert_eq!(selection.inputs.len(), 2);
                assert_eq!(selection.inputs[0].record, selection.inputs[1].record);
                let mut missing = selection.clone();
                missing.inputs.pop();
                assert!(missing.validate().is_err());
            } else {
                assert_eq!(selection.transform, Transform::JsonValueV1);
            }
        }
    }
}

#[test]
fn absent_null_and_invalid_counts_do_not_become_supplied_zero() {
    for cached in [None, Some(Value::Null), Some(json!("6")), Some(json!(-1))] {
        let mut usage = json!({"input_tokens":10});
        if let Some(cached) = cached {
            usage["cached_input_tokens"] = cached;
        }
        let (bytes, view) = usage_view(usage);
        let message = view.projected_message(0).unwrap().unwrap();
        assert_eq!(message.usage.as_ref().unwrap().input, Some(10));
        assert_eq!(message.usage.as_ref().unwrap().cache_read, None);
        assert!(
            !view
                .cells
                .contains_key(&Key::Presence(0, "usage:cached_input_tokens".into()))
        );
        let Cell::SlotValue(w) = &view.cells[&Key::Presence(0, "usage:input_tokens".into())] else {
            panic!("input witness")
        };
        let selection = w.selection.as_ref().unwrap();
        assert_eq!(selection.transform, Transform::JsonValueV1);
        let mut output = Vec::new();
        selection
            .write_value(
                |r| Ok::<_, ()>(bytes.as_bytes()[r.start as usize..r.end as usize].to_vec()),
                bytes.len(),
                &mut output,
            )
            .unwrap();
        assert_eq!(output, b"10");
    }
    let (_, view) =
        usage_view(json!({"input_tokens":null,"cached_input_tokens":0,"output_tokens":"0"}));
    let usage = view.projected_message(0).unwrap().unwrap().usage.unwrap();
    assert_eq!(usage.input, None);
    assert_eq!(usage.cache_read, Some(0));
    assert_eq!(usage.output, None);
    assert!(
        !view
            .cells
            .contains_key(&Key::Presence(0, "usage:input_tokens".into()))
    );
    assert!(
        !view
            .cells
            .contains_key(&Key::Presence(0, "usage:output_tokens".into()))
    );
}

#[test]
fn uncached_input_reads_both_original_operands_and_rejects_wrong_types() {
    let (bytes, view) = usage_view(json!({"input_tokens":10,"cached_input_tokens":6}));
    let Cell::SlotValue(w) = &view.cells[&Key::Presence(0, "usage:input_tokens".into())] else {
        panic!("input witness")
    };
    let selection = w.selection.as_ref().unwrap();
    let resolve = |replacement: &str| {
        let mut output = Vec::new();
        let changed = bytes.replace("\"cached_input_tokens\":6", replacement);
        selection
            .write_value(
                |r| Ok::<_, ()>(changed.as_bytes()[r.start as usize..r.end as usize].to_vec()),
                bytes.len(),
                &mut output,
            )
            .map(|()| output)
    };
    assert_eq!(resolve("\"cached_input_tokens\":7").unwrap(), b"3");
    // A same-size invalid numeric source cannot be mistaken for an absent cache.
    assert!(resolve("\"cached_input_tokens\":x").is_err());
    let mut swapped = selection.clone();
    swapped.inputs.swap(0, 1);
    let mut output = Vec::new();
    swapped
        .write_value(
            |r| Ok::<_, ()>(bytes.as_bytes()[r.start as usize..r.end as usize].to_vec()),
            bytes.len(),
            &mut output,
        )
        .unwrap();
    assert_eq!(output, b"0");
}

#[test]
fn stored_usage_slot_value_rejects_wrong_slot_record_operand_order_or_field() {
    let (_, view) = usage_view(json!({"input_tokens":10,"cached_input_tokens":6}));
    let key = Key::Presence(0, "usage:input_tokens".into());
    let Cell::SlotValue(witness) = &view.cells[&key] else {
        panic!("input witness")
    };
    assert!(Cell::SlotValue(witness.clone()).valid_for(&key));
    for damage in 0..5 {
        let mut changed = witness.clone();
        let selection = changed.selection.as_mut().unwrap();
        match damage {
            0 => selection.inputs.swap(0, 1),
            1 => selection.inputs[1].record.start += 1,
            2 => selection.inputs[1]
                .path
                .last_mut()
                .map(|p| *p = polylogue_sources::selection::PathPart::Key("output_tokens".into()))
                .unwrap(),
            3 => changed.slot = Slot::Usage(polylogue_model::merge::UsageField::Output),
            _ => selection.inputs[0].path.clear(),
        }
        assert!(!Cell::SlotValue(changed).valid_for(&key), "damage {damage}");
    }
}

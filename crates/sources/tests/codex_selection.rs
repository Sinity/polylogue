//! Source-owned field selectors are captured with mapping, not reconstructed by readers.
use polylogue_model::{
    BlockSlot,
    merge::{BlockField, Slot},
};
use polylogue_sources::{
    codex::{
        keyed::{Cell, Key, KeyedRecords, MemoryReadView, Resume},
        selection::{ADAPTER_RECIPE_VERSION, Binding, Error, FactSelection, PathPart, Transform},
    },
    frame::drive_jsonl,
};
use serde_json::{Value, json};
use std::io::{self, Write};
fn line(kind: &str, payload: Value) -> String {
    json!({"timestamp":"2026-10-09T14:30:00.123456Z","type":kind,"payload":payload}).to_string()
        + "\n"
}
fn parse(bytes: &str) -> MemoryReadView {
    let output = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        bytes.as_bytes(),
        0,
    )
    .output
    .unwrap();
    let mut view = MemoryReadView::default();
    view.apply(&output);
    view
}
fn selection(
    view: &MemoryReadView,
    message: usize,
    block: usize,
    field: BlockField,
) -> FactSelection {
    let Cell::Presence(group) = view.cells.get(&Key::PresenceBlock(message, block)).unwrap() else {
        panic!("shape")
    };
    group
        .fields
        .iter()
        .find(|w| {
            w.slot
                == Slot::Block {
                    block: BlockSlot::Item(0),
                    field,
                }
                || w.slot
                    == Slot::Block {
                        block: BlockSlot::Reasoning(0),
                        field,
                    }
        })
        .unwrap()
        .selection
        .clone()
        .unwrap()
}
fn extract(bytes: &str, selection: &FactSelection) -> Vec<Value> {
    selection
        .inputs
        .iter()
        .map(|input| {
            let mut value: Value = serde_json::from_slice(
                &bytes.as_bytes()[input.record.start as usize..input.record.end as usize],
            )
            .unwrap();
            for part in &input.path {
                value = match part {
                    PathPart::Key(key) => value.get(key).unwrap().clone(),
                    PathPart::Index(index) => value.get(*index as usize).unwrap().clone(),
                };
            }
            value
        })
        .collect()
}
#[test]
fn text_and_thinking_use_exact_original_leaf_indices_and_one_normalization_iterator() {
    let bytes = line("session_meta", json!({"id":"selector-test"}))
        + &line("turn_context", json!({"model":"selected-model"}))
        + &line(
            "response_item",
            json!({"type":"message","id":"a","role":"assistant","content":["first",{"text":"second"},""],"opaque":{"empty":[]}}),
        )
        + &line(
            "response_item",
            json!({"type":"reasoning","id":"r","summary":[{"text":"think"},{"ignored":true},"again"],"encrypted_content":"cipher"}),
        );
    let view = parse(&bytes);
    let text = selection(&view, 0, 0, BlockField::Text);
    assert_eq!(text.transform, Transform::TextJoinLfV1);
    assert_eq!(
        extract(&bytes, &text),
        vec![json!("first"), json!("second"), json!("")]
    );
    assert_eq!(text.inputs[0].path.last(), Some(&PathPart::Index(0)));
    assert_eq!(
        text.inputs[1].path[text.inputs[1].path.len() - 2..],
        [PathPart::Index(1), PathPart::Key("text".into())]
    );
    let message = view.projected_message(0).unwrap().unwrap();
    let polylogue_model::BlockContent::Text { text: normalized } = &message.blocks[0].content
    else {
        panic!("shape")
    };
    assert_eq!(
        normalized,
        &extract(&bytes, &text)
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    );
    let Cell::Presence(group) = view.cells.get(&Key::PresenceBlock(0, 0)).unwrap() else {
        panic!("shape")
    };
    let opaque = group
        .fields
        .iter()
        .find(|w| matches!(&w.slot,Slot::Extra { key,.. } if key=="opaque"))
        .unwrap();
    assert_eq!(
        opaque.selection.as_ref().unwrap().transform,
        Transform::JsonValueV1
    );
    assert_eq!(
        extract(&bytes, opaque.selection.as_ref().unwrap()),
        vec![json!({"empty":[]})]
    );
    let thinking = selection(&view, 0, 1, BlockField::ThinkingText);
    assert_eq!(
        extract(&bytes, &thinking),
        vec![json!("think"), json!("again")]
    );
    let data = selection(&view, 0, 1, BlockField::ThinkingData);
    assert_eq!(data.transform, Transform::JsonStringV1);
    assert_eq!(extract(&bytes, &data), vec![json!("cipher")]);
    let Cell::Witness(model) = view.cells.get(&Key::Presence(0, "model".into())).unwrap() else {
        panic!("shape")
    };
    assert_eq!(
        extract(&bytes, model.selection.as_ref().unwrap()),
        vec![json!("selected-model")]
    );
    let Cell::Witness(at) = view.cells.get(&Key::Presence(0, "at".into())).unwrap() else {
        panic!("shape")
    };
    assert_eq!(
        at.selection.as_ref().unwrap().transform,
        Transform::ProviderTimeV1
    );
    assert_eq!(
        extract(&bytes, at.selection.as_ref().unwrap()),
        vec![json!("2026-10-09T14:30:00.123456Z")]
    );
}
#[test]
fn overlay_selection_retains_original_content_and_replacement_uses_new_record() {
    let header = line("session_meta", json!({"id":"selector-overlay"}));
    let first = line(
        "response_item",
        json!({"type":"message","id":"a","role":"assistant","content":[{"text":"old"}]}),
    );
    let update = line("response_item", json!({"id":"a","phase":"commentary"}));
    let bytes = header + &first + &update;
    let view = parse(&bytes);
    let selected = selection(&view, 0, 0, BlockField::Text);
    assert_eq!(extract(&bytes, &selected), vec![json!("old")]);
    let Cell::Witness(phase) = view
        .cells
        .get(&Key::Presence(0, "extra:phase".into()))
        .unwrap()
    else {
        panic!("shape")
    };
    assert_eq!(
        extract(&bytes, phase.selection.as_ref().unwrap()),
        vec![json!("commentary")]
    );
    let replacement = line(
        "response_item",
        json!({"type":"message","id":"a","role":"assistant","content":"new"}),
    );
    let changed = bytes + &replacement;
    assert_eq!(
        extract(
            &changed,
            &selection(&parse(&changed), 0, 0, BlockField::Text)
        ),
        vec![json!("new")]
    );
    // An earlier admitted descriptor remains bound to its earlier original.
    assert_eq!(extract(&changed, &selected), vec![json!("old")]);
}
#[test]
fn identical_reemission_preserves_frozen_selection_and_direct_tools_have_proved_selectors() {
    let header = line("session_meta", json!({"id":"selector-replay"}));
    let item = line(
        "response_item",
        json!({"type":"message","id":"a","role":"assistant","content":"same"}),
    );
    let before = parse(&(header.clone() + &item));
    let after = parse(&(header + &item + &item));
    assert_eq!(
        selection(&before, 0, 0, BlockField::Text),
        selection(&after, 0, 0, BlockField::Text)
    );
    let bytes = line("session_meta", json!({"id":"unsupported-selector"}))
        + &line(
            "response_item",
            json!({"type":"function_call","id":"c","call_id":"c","name":"exec_command","arguments":"{\"cmd\":\"x\"}"}),
        )
        + &line(
            "response_item",
            json!({"type":"function_call_output","call_id":"c","output":"result"}),
        );
    let view = parse(&bytes);
    for cell in view.cells.values() {
        if let Cell::Presence(group) = cell {
            for w in &group.fields {
                if matches!(
                    w.slot,
                    Slot::Block {
                        field: BlockField::Command | BlockField::Output,
                        ..
                    }
                ) {
                    let selected = w
                        .selection
                        .as_ref()
                        .expect("supported direct tool selector");
                    let mut emitted = Vec::new();
                    selected
                        .write_value(
                            |range| {
                                Ok::<_, ()>(
                                    bytes.as_bytes()[range.start as usize..range.end as usize]
                                        .to_vec(),
                                )
                            },
                            65536,
                            &mut emitted,
                        )
                        .unwrap();
                    let actual: Value = serde_json::from_slice(&emitted).unwrap();
                    let expected = if matches!(
                        w.slot,
                        Slot::Block {
                            field: BlockField::Command,
                            ..
                        }
                    ) {
                        json!("x")
                    } else {
                        json!({"Text":"result"})
                    };
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
struct Chunks {
    bytes: Vec<u8>,
    calls: usize,
    maximum: usize,
    fail: Option<usize>,
}
impl Write for Chunks {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        if self.fail == Some(self.calls) {
            return Err(io::Error::from_raw_os_error(28));
        }
        let n = bytes.len().min(self.maximum);
        self.bytes.extend_from_slice(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn bound_descriptor_streams_to_fixed_chunks_preserving_original_io_errors_and_cursor() {
    let bytes = line("session_meta", json!({"id":"selector-stream"}))
        + &line(
            "response_item",
            json!({"type":"message","id":"a","role":"assistant","content":"text"}),
        );
    let mut selected = selection(&parse(&bytes), 0, 0, BlockField::Text);
    selected.inputs[0]
        .path
        .push(PathPart::Key("large-key".repeat(5000)));
    let mut malformed = selected.clone();
    malformed.codec_version = 99;
    let mut bad_cursor = malformed.cursor();
    assert!(malformed.advance(&mut bad_cursor).is_err());
    assert!(malformed.advance(&mut bad_cursor).is_err());
    assert_eq!(selected.adapter_version, ADAPTER_RECIPE_VERSION);
    assert_eq!(selected.adapter_version, 1);
    let mut unsupported = selected.clone();
    unsupported.adapter_version = ADAPTER_RECIPE_VERSION + 1;
    assert!(unsupported.validate().is_err());
    let mut unsupported_cursor = unsupported.cursor();
    assert!(unsupported.advance(&mut unsupported_cursor).is_err());
    assert!(unsupported.advance(&mut unsupported_cursor).is_err());
    for chain in [0, -1] {
        let mut bytes = Vec::new();
        assert!(
            selected
                .write_bound(
                    Binding {
                        input: "immutable-input",
                        chain
                    },
                    &mut bytes
                )
                .is_err()
        );
        assert!(bytes.is_empty());
    }
    let binding = Binding {
        input: "immutable-input",
        chain: 7,
    };
    let mut expected = Vec::new();
    selected.write_bound(binding, &mut expected).unwrap();
    let mut chunks = Chunks {
        bytes: Vec::new(),
        calls: 0,
        maximum: 31,
        fail: None,
    };
    selected.write_bound(binding, &mut chunks).unwrap();
    assert_eq!(chunks.bytes, expected);
    assert!(chunks.calls > 100);
    let mut cursor = selected.cursor();
    assert_eq!(
        selected.advance(&mut cursor).unwrap(),
        Some(selected.inputs[0].record)
    );
    let mut cursor = serde_json::from_slice(&serde_json::to_vec(&cursor).unwrap()).unwrap();
    assert_eq!(selected.advance(&mut cursor).unwrap(), None);
    for fail in [1, 15] {
        let mut chunks = Chunks {
            bytes: Vec::new(),
            calls: 0,
            maximum: 31,
            fail: Some(fail),
        };
        let Err(Error::Io(error)) = selected.write_bound(binding, &mut chunks) else {
            panic!("expected original IO error")
        };
        assert_eq!(error.raw_os_error(), Some(28));
        assert_eq!(chunks.calls, fail);
    }
    let value: Value = serde_json::from_slice(&expected).unwrap();
    assert_eq!(value["binding"]["chain"], 7);
}

fn recipe(
    transform: Transform,
    records: &[(&[u8], Vec<PathPart>)],
) -> (
    FactSelection,
    Vec<(polylogue_sources::codex::origins::RecordRange, Vec<u8>)>,
) {
    use polylogue_sources::codex::{
        origins::RecordRange,
        selection::{CODEC_VERSION, SelectedRaw},
    };
    let mut offset = 0;
    let mut stored = Vec::new();
    let inputs = records
        .iter()
        .map(|(bytes, path)| {
            let record = RecordRange::new(offset, offset + bytes.len() as u64).unwrap();
            offset = record.end + 1;
            stored.push((record, bytes.to_vec()));
            SelectedRaw {
                record,
                path: path.clone(),
            }
        })
        .collect();
    (
        FactSelection {
            codec_version: CODEC_VERSION,
            adapter_version: ADAPTER_RECIPE_VERSION,
            transform,
            inputs,
        },
        stored,
    )
}
#[test]
fn evaluator_matches_normalized_json_null_strings_opaque_values_and_time() {
    use PathPart::Key;
    use std::convert::Infallible;
    let expected_string = serde_json::to_vec("quote\"\n\\\0λ").unwrap();
    let cases = [
        (
            Transform::JsonValueV1,
            br#"{"v":null}"#.as_slice(),
            b"null".as_slice(),
        ),
        (
            Transform::JsonValueV1,
            br#"{"v":{"z":[],"a":false}}"#.as_slice(),
            br#"{"a":false,"z":[]}"#.as_slice(),
        ),
        (
            Transform::JsonStringV1,
            br#"{"v":"quote\"\n\\\u0000\u03bb"}"#.as_slice(),
            expected_string.as_slice(),
        ),
        (
            Transform::ProviderTimeV1,
            br#"{"v":"1970-01-01T00:00:00.123456Z"}"#.as_slice(),
            br#"{"at":123,"precision":"instant","time_quality":"intrinsic","until":null}"#
                .as_slice(),
        ),
    ];
    for (transform, raw, expected) in cases {
        let (selected, records) = recipe(transform, &[(raw, vec![Key("v".into())])]);
        let mut output = Vec::new();
        let mut reads = 0;
        selected
            .write_value(
                |range| {
                    reads += 1;
                    assert_eq!(range, records[0].0);
                    Ok::<_, Infallible>(records[0].1.clone())
                },
                raw.len(),
                &mut output,
            )
            .unwrap();
        assert_eq!(output, expected);
        assert_eq!(reads, 1);
    }
}
#[test]
fn evaluator_joins_selected_string_leaves_using_serde_escaping_and_adjacent_record_cache() {
    use PathPart::{Index, Key};
    use std::convert::Infallible;
    let raw = serde_json::to_vec(&json!({"v":["quote\"",{"text":"slash\\\n\0λ"},""]})).unwrap();
    let (mut selected, records) = recipe(
        Transform::TextJoinLfV1,
        &[(&raw, vec![Key("v".into()), Index(0)])],
    );
    for path in [
        vec![Key("v".into()), Index(1), Key("text".into())],
        vec![Key("v".into()), Index(2)],
    ] {
        let mut next = selected.inputs[0].clone();
        next.path = path;
        selected.inputs.push(next);
    }
    let mut reads = 0;
    let mut sink = Chunks {
        bytes: Vec::new(),
        calls: 0,
        maximum: 3,
        fail: None,
    };
    selected
        .write_value(
            |_| {
                reads += 1;
                Ok::<_, Infallible>(records[0].1.clone())
            },
            raw.len(),
            &mut sink,
        )
        .unwrap();
    assert_eq!(reads, 1);
    assert_eq!(
        sink.bytes,
        serde_json::to_vec("quote\"\nslash\\\n\0λ\n").unwrap()
    );
    assert!(sink.calls > 5);
}
#[test]
fn evaluator_read_admission_byte_count_and_original_resolver_errors_are_distinct() {
    use PathPart::Key;
    use polylogue_sources::codex::selection::ValueError;
    let raw = br#"{"v":"ok"}"#;
    let (selected, records) = recipe(Transform::JsonStringV1, &[(raw, vec![Key("v".into())])]);
    let mut output = Vec::new();
    let err = selected
        .write_value(
            |_| -> Result<Vec<u8>, ()> { panic!("resolver called before admission") },
            raw.len() - 1,
            &mut output,
        )
        .unwrap_err();
    assert!(
        matches!(err,ValueError::Admission{required,allowance} if required==raw.len() as u64 && allowance==raw.len()-1)
    );
    assert!(output.is_empty());
    let marker = Box::new(42u64);
    let pointer = &*marker as *const u64;
    let mut marker = Some(marker);
    let err = selected
        .write_value(
            |_| Err::<Vec<u8>, _>(marker.take().unwrap()),
            raw.len(),
            &mut output,
        )
        .unwrap_err();
    let ValueError::Resolve(marker) = err else {
        panic!("lost resolver error")
    };
    assert_eq!(&*marker as *const u64, pointer);
    let err = selected
        .write_value(|_| Ok::<_, ()>(vec![b'{']), raw.len(), &mut output)
        .unwrap_err();
    assert!(matches!(err,ValueError::ByteCount{expected,actual:1} if expected==raw.len() as u64));
    let mut calls = 0;
    selected
        .write_value(
            |range| {
                calls += 1;
                assert_eq!(range, records[0].0);
                Ok::<_, ()>(records[0].1.clone())
            },
            raw.len(),
            &mut output,
        )
        .unwrap();
    assert_eq!(calls, 1);
}
#[test]
fn evaluator_refuses_missing_typed_paths_wrong_types_bad_json_and_bad_time() {
    use PathPart::{Index, Key};
    use polylogue_sources::codex::selection::ValueError;
    let raw = br#"{"v":null}"#;
    for path in [
        vec![Key("missing".into())],
        vec![Index(0)],
        vec![Key("v".into()), Key("nested".into())],
    ] {
        let (selected, _) = recipe(Transform::JsonValueV1, &[(raw, path)]);
        let err = selected
            .write_value(|_| Ok::<_, ()>(raw.to_vec()), raw.len(), &mut Vec::new())
            .unwrap_err();
        assert!(matches!(err, ValueError::Path { .. }));
    }
    for transform in [
        Transform::JsonStringV1,
        Transform::TextJoinLfV1,
        Transform::ProviderTimeV1,
    ] {
        let (selected, _) = recipe(transform, &[(raw, vec![Key("v".into())])]);
        assert!(matches!(
            selected.write_value(|_| Ok::<_, ()>(raw.to_vec()), raw.len(), &mut Vec::new()),
            Err(ValueError::Type { input: 0 })
        ));
    }
    let invalid = br#"{"v":oops}"#;
    let (selected, _) = recipe(Transform::JsonValueV1, &[(invalid, vec![Key("v".into())])]);
    assert!(matches!(
        selected.write_value(
            |_| Ok::<_, ()>(invalid.to_vec()),
            invalid.len(),
            &mut Vec::new()
        ),
        Err(ValueError::Json(_))
    ));
    let invalid = br#"{"v":"yesterday"}"#;
    let (selected, _) = recipe(
        Transform::ProviderTimeV1,
        &[(invalid, vec![Key("v".into())])],
    );
    assert!(matches!(
        selected.write_value(
            |_| Ok::<_, ()>(invalid.to_vec()),
            invalid.len(),
            &mut Vec::new()
        ),
        Err(ValueError::Time)
    ));
}
#[test]
fn evaluator_first_and_middle_sink_failure_keep_original_error_and_stop_future_reads() {
    use PathPart::Key;
    use polylogue_sources::codex::selection::ValueError;
    let raw = serde_json::to_vec(&json!({"v":"a\"\n\\".repeat(100)})).unwrap();
    let (mut selected, records) = recipe(
        Transform::TextJoinLfV1,
        &[(&raw, vec![Key("v".into())]), (&raw, vec![Key("v".into())])],
    );
    for fail in [1, 5] {
        let mut reads = 0;
        let mut sink = Chunks {
            bytes: Vec::new(),
            calls: 0,
            maximum: 3,
            fail: Some(fail),
        };
        let err = selected
            .write_value(
                |range| {
                    reads += 1;
                    Ok::<_, ()>(records.iter().find(|(r, _)| *r == range).unwrap().1.clone())
                },
                raw.len(),
                &mut sink,
            )
            .unwrap_err();
        let ValueError::Io(error) = err else {
            panic!("lost sink error")
        };
        assert_eq!(error.raw_os_error(), Some(28));
        assert_eq!(sink.calls, fail);
        assert!(reads <= 1);
    }
    // A nonadjacent return to an earlier record must reload it, not retain all
    // decoded inputs in a growing cache.
    selected.inputs.push(selected.inputs[0].clone());
    let mut reads = 0;
    selected
        .write_value(
            |range| {
                reads += 1;
                Ok::<_, ()>(records.iter().find(|(r, _)| *r == range).unwrap().1.clone())
            },
            raw.len(),
            &mut Vec::new(),
        )
        .unwrap();
    assert_eq!(reads, 3);
}

#[test]
fn evaluator_sink_error_object_survives_json_fragment_and_typed_time_wrappers() {
    use polylogue_sources::codex::selection::ValueError;
    use std::sync::Arc;
    #[derive(Debug)]
    struct Sentinel(Arc<()>);
    impl std::fmt::Display for Sentinel {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "sentinel")
        }
    }
    impl std::error::Error for Sentinel {}
    struct Fail {
        error: Option<io::Error>,
        at: usize,
        calls: usize,
    }
    impl Write for Fail {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            if self.calls == self.at {
                Err(self.error.take().unwrap())
            } else {
                Ok(bytes.len().min(2))
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let raw = br#"{"v":"1970-01-01T00:00:00.123456Z"}"#;
    for transform in [
        Transform::JsonValueV1,
        Transform::JsonStringV1,
        Transform::TextJoinLfV1,
        Transform::ProviderTimeV1,
    ] {
        let (selected, _) = recipe(transform, &[(raw, vec![PathPart::Key("v".into())])]);
        for at in [1, 3] {
            let marker = Arc::new(());
            let mut sink = Fail {
                error: Some(io::Error::other(Sentinel(marker.clone()))),
                at,
                calls: 0,
            };
            let err = selected
                .write_value(|_| Ok::<_, ()>(raw.to_vec()), raw.len(), &mut sink)
                .unwrap_err();
            let ValueError::Io(error) = err else {
                panic!("lost original sink error")
            };
            assert!(Arc::ptr_eq(
                &marker,
                &error
                    .get_ref()
                    .unwrap()
                    .downcast_ref::<Sentinel>()
                    .unwrap()
                    .0
            ));
            assert_eq!(sink.calls, at);
        }
    }
}
#[test]
fn evaluator_captured_text_thinking_and_message_witnesses_match_owning_leaf_encoder() {
    use polylogue_model::source_leaf::{self, PresenceStatus, Subject};
    use std::convert::Infallible;
    let bytes = line("session_meta", json!({"id":"leaf-evaluation"}))
        + &line("turn_context", json!({"model":"model-α"}))
        + &line(
            "response_item",
            json!({"type":"message","id":"a","role":"assistant","content":["first",{"text":"second\n\""}]}),
        )
        + &line(
            "response_item",
            json!({"type":"reasoning","id":"r","summary":["summary","again"],"encrypted_content":"cipher"}),
        );
    let view = parse(&bytes);
    let message = view.projected_message(0).unwrap().unwrap();
    let mut count = 0;
    for cell in view.cells.values() {
        let witnesses: Vec<_> = match cell {
            Cell::Presence(group) => group.fields.iter().collect(),
            Cell::Witness(w) => vec![w],
            _ => continue,
        };
        for witness in witnesses {
            let Some(selection) = &witness.selection else {
                continue;
            };
            let subject = match &witness.slot {
                Slot::Message(_) => Subject::Message(&message),
                Slot::Block { block, .. } => {
                    let content = &message
                        .blocks
                        .iter()
                        .find(|b| b.id.slot() == block)
                        .unwrap()
                        .content;
                    Subject::Block {
                        slot: block,
                        content,
                    }
                }
                _ => continue,
            };
            let mut expected = Vec::new();
            assert_eq!(
                source_leaf::write(subject, &witness.slot, &mut expected).unwrap(),
                PresenceStatus::Written
            );
            let mut actual = Vec::new();
            selection
                .write_value(
                    |range| {
                        Ok::<_, Infallible>(
                            bytes.as_bytes()[range.start as usize..range.end as usize].to_vec(),
                        )
                    },
                    bytes.len(),
                    &mut actual,
                )
                .unwrap();
            assert_eq!(actual, expected, "{:?}", witness.slot);
            count += 1;
        }
    }
    assert!(count >= 5);
}

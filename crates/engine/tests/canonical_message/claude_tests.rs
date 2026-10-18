//! Claude Messages through the same action/state/value/origin code as Codex.
use super::*;
use polylogue_engine::state::claude as ca;
use polylogue_sources::claude_code::{self as cc, FileCtx, KeyedRecords, keyed::PendingMessage};
use polylogue_store::contributions::{FieldRead, MessageView};
const SID: &str = "7f3a2c10-0001-4000-8000-000000000001";
const MORROW: &[u8] = include_bytes!("../../../sources/tests/fixtures/morrow-main.jsonl");
fn make_world(retained: &[u8], parsed: &[u8]) -> (World, Vec<PendingMessage>) {
    make_world_with(retained, parsed, |_| {})
}
fn make_world_with(
    retained: &[u8],
    parsed: &[u8],
    mutate: impl FnOnce(&mut cc::StepOutput),
) -> (World, Vec<PendingMessage>) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let mut writer = Writer::open(&root).unwrap();
    let blob = writer.put_file_blob(retained).unwrap();
    let input = writer
        .store_immutable_input(
            blob,
            InputFormat::ClaudeCodeJsonl,
            &["claude-main", SID],
            &ImportObservation {
                id: "synthetic-claude-materializer",
                machine: "synthetic",
                host: "synthetic",
                path: "synthetic/main.jsonl",
                seen_at: 0,
            },
            4096,
        )
        .unwrap();
    let head = writer
        .begin_parser(&Basis {
            input_pk: input.input_pk,
            chain: input.chain,
            parser_version: cc::PARSER_VERSION,
            digest: [9; 32],
            acquired_upto: parsed.len() as u64,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    let snapshot = writer.parser_snapshot(&head.token).unwrap();
    let view = ca::View::new(&snapshot, budget()).unwrap();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main(SID), &view).unwrap(),
        parsed,
        0,
    );
    let mut output = framed.output.unwrap();
    mutate(&mut output);
    assert!(view.take_error().is_none());
    let pending = output.delta.pending_messages.clone();
    let changes = ca::encode_step(
        &mut output,
        &head,
        framed.parsed_upto,
        &framed.counts,
        budget(),
    )
    .unwrap();
    drop(view);
    drop(snapshot);
    let mut prepared = writer
        .prepare_parser(
            &head,
            Progress {
                inspected_upto: framed.parsed_upto,
                parsed_upto: framed.parsed_upto,
                phase: Phase::Finalizing,
                finalization_pending: true,
                counts: Accounting {
                    framed: framed.counts.framed,
                    used: framed.counts.used,
                    ignored: framed.counts.ignored.values().sum(),
                    unrecognized: framed.counts.unrecognized.len() as u64,
                    undecodable: framed.counts.undecodable.len() as u64,
                },
            },
        )
        .unwrap();
    prepared = writer
        .stage_parser(&prepared.token, &changes, budget())
        .unwrap();
    prepared = writer.seal_parser(&prepared.token).unwrap();
    while prepared.token.applied < prepared.token.staged {
        prepared = writer.apply_parser(&prepared.token, budget()).unwrap()
    }
    let head = writer.publish_parser(&prepared.token).unwrap();
    (
        World {
            dir,
            root,
            writer,
            head,
            source_binding: blob,
        },
        pending,
    )
}
fn request(world: &World, pending: &PendingMessage) -> m::ClaudeRequest {
    let mut key = [0; 32];
    key[..8].copy_from_slice(&pending.ordinal.to_be_bytes());
    m::ClaudeRequest {
        pending: pending.clone(),
        source_binding: world.source_binding,
        context: vec!["claude-main".into(), SID.into()],
        revision: 1,
        logical_key: key,
    }
}
fn begin(world: &mut World, request: &m::ClaudeRequest) -> m::Handle {
    let current = world
        .writer
        .current_source_projection(&world.head.token, &request.logical_key, 65536)
        .unwrap();
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let action = m::start_claude(&snapshot, request, current, budget(), 65536).unwrap();
    drop(snapshot);
    match m::apply(&mut world.writer, action).unwrap() {
        m::Applied::Prepared(h) => h,
        _ => panic!("expected Claude build"),
    }
}
fn selected(world: &mut World, source: &SourceProjection) -> MessageView {
    world
        .writer
        .select_compatible_source_message_for_test(source, 65536)
        .unwrap();
    world
        .writer
        .read_handle()
        .unwrap()
        .selected_message(source.message())
        .unwrap()
        .unwrap()
}
fn value(view: &MessageView, slot: &Slot) -> (Value, bool, Value) {
    let FieldRead::Value(field) = view.field(slot, 65536).unwrap().unwrap() else {
        panic!("conflict")
    };
    let data = serde_json::from_slice(&field.range(0, 65536).unwrap().bytes).unwrap();
    let origin =
        serde_json::from_slice(&field.origin_range(0, 65536).unwrap().unwrap().bytes).unwrap();
    (data, field.supplied, origin)
}
#[test]
fn background_completion_cites_its_own_record_in_the_source_set() {
    let bytes =
        include_bytes!("../../../sources/tests/fixtures/donor/claude-normalization-main.jsonl");
    let (mut world, pending) = make_world(bytes, bytes);
    let pending = pending
        .iter()
        .find(|p| p.message.as_str().ends_with(":main-bg-start"))
        .unwrap();
    let request = request(&world, pending);
    let handle = begin(&mut world, &request);
    // Byte ranges in the fixed synthetic fixture, excluding LF.
    let ranges = [(942, 1287), (3018, 3446)];
    for _ in 0..4000 {
        let action = world
            .compute_with(&handle, work_budget())
            .unwrap_or_else(|error| {
                let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
                let membership: Vec<_> = ranges
                    .iter()
                    .map(|&(start, end)| {
                        let present = snapshot
                            .has_message_source(
                                handle.build(),
                                polylogue_store::contributions::SourceRange { start, end },
                            )
                            .unwrap();
                        (start, end, present)
                    })
                    .collect();
                panic!(
                    "completion origin: {error}; exact candidate source membership {membership:?}"
                );
            });
        if let m::Applied::Source(source) = m::apply(&mut world.writer, action).unwrap() {
            let view = selected(&mut world, &source);
            let mut cursor = None;
            let mut selected_ranges = Vec::new();
            loop {
                use polylogue_store::contributions::SourceStep;
                match view.next_source(cursor.as_ref(), 65536).unwrap() {
                    SourceStep::Span {
                        cursor: next,
                        source,
                    } => {
                        selected_ranges.push((source.start, source.end));
                        cursor = Some(next);
                    }
                    SourceStep::Continue { cursor: next } => cursor = Some(next),
                    SourceStep::End => break,
                }
            }
            assert_eq!(selected_ranges, ranges);
            let (exit, supplied, origin) = value(
                &view,
                &Slot::Block {
                    block: BlockSlot::Item(0),
                    field: BlockField::ExitCode,
                },
            );
            assert_eq!(exit, 7);
            assert!(supplied);
            assert_eq!(
                origin["selection"]["inputs"][0]["record"],
                json!({"start":3018,"end":3446})
            );
            return;
        }
    }
    panic!("background completion did not materialize");
}
#[test]
fn real_morrow_claude_bodies_use_native_ids_flat_order_and_verified_origins() {
    let (mut world, pending) = make_world(MORROW, MORROW);
    assert_eq!(pending.len(), 12);
    for p in &pending {
        let request = request(&world, p);
        let handle = begin(&mut world, &request);
        let source = world.complete(&handle);
        assert_eq!(source.message(), &p.message);
        assert_eq!(source.evidence_revision(), p.revision);
        let view = selected(&mut world, &source);
        let (at, supplied, origin) = value(&view, &Slot::Message(MessageField::At));
        assert!(at["at"].is_i64());
        assert!(supplied);
        assert_eq!(origin["format"], "claude-code-field-origin");
        assert_eq!(
            origin["binding"]["input"],
            polylogue_model::InputId::imported(
                &world.source_binding,
                "claude-code-jsonl",
                &["claude-main", SID]
            )
            .as_str()
        );
        if p.ordinal == 0 {
            assert_eq!(
                view.header().unwrap().written_by,
                polylogue_model::WrittenBy::Human
            );
            let (prompt, supplied, _) = value(&view, &Slot::Message(MessageField::PromptKind));
            assert_eq!(prompt, "prose");
            assert!(!supplied)
        }
        let mut index = 0;
        while let Some(block) = view.block_at(index).unwrap() {
            assert_eq!(block.slot, BlockSlot::Item(index as u32));
            assert_eq!(block.group, 0);
            if block.kind == "thinking" {
                if let Some(FieldRead::Value(signature)) = block
                    .field(
                        &Slot::Block {
                            block: block.slot.clone(),
                            field: BlockField::ThinkingSignature,
                        },
                        65536,
                    )
                    .unwrap()
                {
                    assert!(!signature.supplied);
                    assert!(
                        serde_json::from_slice::<Value>(&signature.range(0, 65536).unwrap().bytes)
                            .unwrap()
                            .is_string()
                    );
                    assert!(signature.origin_range(0, 65536).unwrap().is_some())
                }
            }
            index += 1;
        }
        assert!(index > 0);
        if view.header().unwrap().role == polylogue_model::Role::Assistant {
            let (stop, supplied, origin) = value(
                &view,
                &Slot::Extra {
                    block: None,
                    key: "stop_sequence".into(),
                },
            );
            assert_eq!(stop, Value::Null);
            assert!(supplied);
            assert_eq!(origin["format"], "claude-code-field-origin");
            let (tier, supplied, origin) = value(
                &view,
                &Slot::Usage(polylogue_model::merge::UsageField::ServiceTier),
            );
            assert_eq!(tier, "standard");
            assert!(supplied);
            assert_eq!(
                origin["selection"]["inputs"][0]["path"]
                    .as_array()
                    .unwrap()
                    .len(),
                3
            );
        }
        drop(view);
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let action = m::start_claude(&snapshot, &request, Some(source), budget(), 65536).unwrap();
        drop(snapshot);
        assert_eq!(action.operation(), "existing");
        assert!(matches!(
            m::apply(&mut world.writer, action).unwrap(),
            m::Applied::Existing(_)
        ));
    }
    assert!(world.head.finalization_pending);
}
#[test]
fn same_file_sidechain_keeps_raw_main_parent_as_an_extra_without_a_message_parent() {
    let bytes =
        include_bytes!("../../../sources/tests/fixtures/donor/claude-normalization-main.jsonl");
    let (mut world, pending) = make_world(bytes, bytes);
    let main = format!("claude-code:{SID}");
    let pending = pending
        .iter()
        .find(|p| p.message.session().as_str() != main)
        .unwrap();
    let request = request(&world, pending);
    let handle = begin(&mut world, &request);
    let source = world.complete(&handle);
    let view = selected(&mut world, &source);
    assert!(
        view.field(&Slot::Message(MessageField::Parent), 65536)
            .unwrap()
            .is_none()
    );
    let (parent, _, origin) = value(
        &view,
        &Slot::Extra {
            block: None,
            key: "parentUuid".into(),
        },
    );
    assert_eq!(parent, "main-fg-result");
    assert_eq!(
        origin["selection"]["inputs"][0]["path"],
        json!([{"Key":"parentUuid"}])
    );
    assert_eq!(
        origin["selection"]["inputs"][0]["record"],
        json!({"start":3682,"end":4088})
    );
}
#[test]
fn a_native_parent_uuid_from_a_later_response_fragment_resolves_its_owning_message() {
    for native_parent in ["answer-second", "missing-parent"] {
        let records = [
            json!({"type":"user","uuid":"root","sessionId":SID,"message":{"role":"user","content":"Root prompt."}}),
            json!({"type":"assistant","uuid":"answer-first","parentUuid":"root","sessionId":SID,"requestId":"provider-request","message":{"id":"provider-response","role":"assistant","content":[{"type":"text","text":"First fragment."}]}}),
            json!({"type":"assistant","uuid":"answer-second","parentUuid":"answer-first","sessionId":SID,"requestId":"provider-request","message":{"id":"provider-response","role":"assistant","content":[{"type":"text","text":"Second fragment."}]}}),
            json!({"type":"user","uuid":"after","parentUuid":native_parent,"sessionId":SID,"message":{"role":"user","content":"Follow-up prompt."}}),
        ];
        let bytes = records
            .iter()
            .map(|record| record.to_string() + "\n")
            .collect::<String>()
            .into_bytes();
        let (mut world, pending) = make_world(&bytes, &bytes);
        let owner = pending
            .iter()
            .find(|p| p.items == 2)
            .unwrap()
            .message
            .clone();
        let child = pending
            .iter()
            .find(|p| p.message.as_str().ends_with(":after"))
            .unwrap();
        let request = request(&world, child);
        let handle = begin(&mut world, &request);
        if native_parent == "missing-parent" {
            assert!(matches!(
                world.compute_with(&handle, work_budget()),
                Err(m::Error::Unsupported(
                    "Claude parent record unavailable"
                ))
            ));
            continue;
        }
        let source = world.complete(&handle);
        let view = selected(&mut world, &source);
        let (parent, supplied, origin) = value(&view, &Slot::Message(MessageField::Parent));
        assert!(supplied);
        assert_eq!(parent["id"], owner.as_str());
        assert_eq!(parent["session"], owner.session().as_str());
        assert_eq!(
            origin["selection"]["transform"]["ParentReferenceV1"]["native_uuid"],
            "answer-second"
        );
        assert_eq!(
            origin["selection"]["inputs"][0]["path"],
            json!([{"Key":"parentUuid"}])
        );
    }
}
fn input(text: &str) -> Vec<u8> {
    json!({"type":"assistant","uuid":"opaque:message","requestId":"request","sessionId":SID,"timestamp":"2026-10-01T10:00:00Z","message":{"id":"response","content":[{"type":"thinking","thinking":"reason","signature":"retained-signature"},{"type":"text","text":text}],"stop_reason":"end_turn"}}).to_string().into_bytes().into_iter().chain([b'\n']).collect()
}
#[test]
fn claude_reopens_partial_normalized_and_origin_values_and_finish_before_publication() {
    let text = "multibyte λ value ".repeat(9000);
    let raw = input(&text);
    let (mut world, pending) = make_world(&raw, &raw);
    let req = request(&world, &pending[0]);
    let mut handle = begin(&mut world, &req);
    let mut value_restart = false;
    let mut origin_restart = false;
    let mut finish_restart = false;
    let mut peer = false;
    let mut turns = 0;
    let source = loop {
        turns += 1;
        assert!(turns < 12000, "Claude restart fixture did not converge");
        let state = m::inspect(&mut world.writer, &handle, 65536).unwrap();
        let partial_value = !value_restart
            && state
                .active_value()
                .is_some_and(|v| v.appended > 0 && v.appended < v.length);
        let partial_origin = !origin_restart
            && state
                .active_origin_value()
                .is_some_and(|v| v.appended > 0 && v.appended < v.length);
        let finished = !finish_restart && state.projection_ready();
        let origin_active = state.active_origin_value().is_some();
        drop(state);
        if partial_value || partial_origin || finished {
            world = world.reopen();
            handle = begin(&mut world, &req);
            value_restart |= partial_value;
            origin_restart |= partial_origin;
            finish_restart |= finished;
            continue;
        }
        let mut b = work_budget();
        if origin_active {
            b.value_chunk = 128;
        }
        let action = world.compute_with(&handle, b).unwrap();
        if action.operation() == "block_install" && !peer {
            world.writer.put_file_blob(&b"synthetic peer"[..]).unwrap();
            peer = true
        }
        if let m::Applied::Source(source) = m::apply(&mut world.writer, action).unwrap() {
            break source;
        }
    };
    assert!(value_restart && origin_restart && finish_restart && peer);
    let view = selected(&mut world, &source);
    let thinking = view.block_at(0).unwrap().unwrap();
    assert_eq!(thinking.kind, "thinking");
    assert_eq!(thinking.encrypted, None);
    assert_eq!(view.block_at(1).unwrap().unwrap().kind, "text");
    assert!(world.head.finalization_pending);
}
#[test]
fn claude_wrong_original_hash_or_selector_type_mismatch_gets_no_source_certificate() {
    let raw = input("original");
    let (mut world, pending) = make_world(&raw, &raw);
    let mut req = request(&world, &pending[0]);
    req.source_binding = polylogue_model::BlobHash::from_bytes([77; 32]);
    let handle = begin(&mut world, &req);
    assert!(matches!(
        world.compute_with(&handle, work_budget()),
        Err(m::Error::Invariant(_))
    ));
    let changed = input("modified");
    let (mut world, pending) = make_world(&raw, &changed);
    let req = request(&world, &pending[0]);
    let handle = begin(&mut world, &req);
    let mut refused = false;
    for _ in 0..4000 {
        match world.compute_with(&handle, work_budget()) {
            Ok(action) => {
                assert!(!matches!(
                    m::apply(&mut world.writer, action).unwrap(),
                    m::Applied::Source(_)
                ))
            }
            Err(_) => {
                refused = true;
                break;
            }
        }
    }
    assert!(refused);
    assert!(
        world
            .writer
            .current_source_projection(&world.head.token, &req.logical_key, 65536)
            .unwrap()
            .is_none()
    );
}

#[test]
fn claude_tool_input_known_null_and_missing_parent_remain_distinct() {
    let raw=json!({"type":"assistant","uuid":"call-message","requestId":"request","sessionId":SID,"message":{"id":"response","content":[{"type":"tool_use","id":"call:opaque","name":"synthetic","input":null}],"stop_reason":"end_turn"}}).to_string()+"\n";
    let (mut world, pending) = make_world(raw.as_bytes(), raw.as_bytes());
    let req = request(&world, &pending[0]);
    let handle = begin(&mut world, &req);
    let source = world.complete(&handle);
    let view = selected(&mut world, &source);
    let (input, supplied, origin) = value(
        &view,
        &Slot::Block {
            block: BlockSlot::Item(0),
            field: BlockField::Input,
        },
    );
    assert_eq!(input, Value::Null);
    assert!(supplied);
    assert_eq!(origin["format"], "claude-code-field-origin");
    drop(view);
    drop(world);
    let raw=json!({"type":"user","uuid":"unresolved","sessionId":SID,"parentUuid":"missing","message":{"content":"retained"}}).to_string()+"\n";
    let (mut world, pending) = make_world(raw.as_bytes(), raw.as_bytes());
    let req = request(&world, &pending[0]);
    let handle = begin(&mut world, &req);
    assert!(matches!(
        world.compute_with(&handle, work_budget()),
        Err(m::Error::Unsupported(_))
    ));
    assert!(
        world
            .writer
            .current_source_projection(&world.head.token, &req.logical_key, 65536)
            .unwrap()
            .is_none()
    );
}

#[test]
fn nonadjacent_claude_response_keeps_contribution_order_and_stale_turn_refuses() {
    let record = |uuid: &str, response: &str, content: Value| {
        json!({"type":"assistant","uuid":uuid,"requestId":"request","sessionId":SID,"timestamp":"2026-10-01T10:00:00Z","message":{"id":response,"content":content}}).to_string()+"\n"
    };
    let raw = record("first", "response-a", json!([{"type":"text","text":"a"}]))
        + &record("other", "response-b", json!([{"type":"text","text":"b"}]))
        + &record(
            "continued",
            "response-a",
            json!([{"type":"thinking","thinking":"reason"},{"type":"text","text":"a2"}]),
        );
    let (mut world, pending) = make_world(raw.as_bytes(), raw.as_bytes());
    let p = pending.iter().find(|p| p.ordinal == 0).unwrap();
    assert_eq!(p.items, 2);
    let req = request(&world, p);
    let handle = begin(&mut world, &req);
    let first = world.compute(&handle);
    let stale = world.compute(&handle);
    m::apply(&mut world.writer, first).unwrap();
    assert!(m::apply(&mut world.writer, stale).is_err());
    let source = world.complete(&handle);
    let view = selected(&mut world, &source);
    for (index, group, local, kind) in [(0, 0, 0, "text"), (1, 1, 0, "thinking"), (2, 1, 1, "text")]
    {
        let block = view.block_at(index).unwrap().unwrap();
        assert_eq!(block.group, group);
        assert_eq!(block.local, local);
        assert_eq!(block.kind, kind);
        assert_eq!(block.slot, BlockSlot::Item(index as u32));
    }
    assert!(view.block_at(3).unwrap().is_none());
}

#[test]
fn claude_immediate_opaque_extra_retains_utf16_numbers_null_and_own_origin() {
    let raw = br#"{"type":"user","uuid":"u","sessionId":"synthetic","message":{"role":"user","content":"text"},"uninterpreted":{"\ud800":[9007199254740993,1e+02,-0,null,"\udfff"]}}
"#;
    let (mut world, pending) = make_world(raw, raw);
    let req = request(&world, &pending[0]);
    let handle = begin(&mut world, &req);
    let source = world.complete(&handle);
    let view = selected(&mut world, &source);
    let slot = Slot::Extra {
        block: None,
        key: "uninterpreted".into(),
    };
    let FieldRead::Value(field) = view.field(&slot, 65536).unwrap().unwrap() else {
        panic!("opaque extra conflict")
    };
    let bytes = field.range(0, 65536).unwrap().bytes;
    assert_eq!(
        bytes,
        br#"{"\ud800":[9007199254740993,1e+02,-0,null,"\udfff"]}"#
    );
    assert!(field.supplied);
    let origin: Value =
        serde_json::from_slice(&field.origin_range(0, 65536).unwrap().unwrap().bytes).unwrap();
    assert_eq!(origin["format"], "claude-code-field-origin");
    assert_eq!(
        origin["selection"]["inputs"][0]["path"],
        json!([{"Key":"uninterpreted"}])
    );
    assert_eq!(
        origin["binding"]["input"],
        polylogue_model::InputId::imported(
            &world.source_binding,
            "claude-code-jsonl",
            &["claude-main", SID]
        )
        .as_str()
    );
}

#[test]
fn claude_nested_extra_and_selected_key_collision_get_no_source_certificate() {
    let cases = [
        r#"{"type":"user","uuid":"u","message":{"role":"user","content":"text","x":1},"message.x":2}"#,
        r#"{"type":"assistant","uuid":"u","requestId":"r","stop_sequence":1,"message":{"id":"a","role":"assistant","content":[{"type":"text","text":"text"}],"stop_sequence":null}}"#,
    ];
    for raw in cases {
        let bytes = format!("{raw}\n");
        let (mut world, pending) = make_world(bytes.as_bytes(), bytes.as_bytes());
        let req = request(&world, &pending[0]);
        let handle = begin(&mut world, &req);
        let mut refused = false;
        for _ in 0..4000 {
            match world.compute_with(&handle, work_budget()) {
                Ok(action) => assert!(!matches!(
                    m::apply(&mut world.writer, action).unwrap(),
                    m::Applied::Source(_)
                )),
                Err(m::Error::UnsupportedField { .. }) => {
                    refused = true;
                    break;
                }
                Err(e) => panic!("unexpected refusal: {e}"),
            }
        }
        assert!(refused, "ambiguous attrs key must not certify: {raw}");
        assert!(
            world
                .writer
                .current_source_projection(&world.head.token, &req.logical_key, 65536)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn claude_opaque_extra_must_match_original_selected_source() {
    let original = br#"{"type":"user","uuid":"u","message":{"role":"user","content":"text"},"uninterpreted":{"v":9007199254740993}}
"#;
    let parsed = br#"{"type":"user","uuid":"u","message":{"role":"user","content":"text"},"uninterpreted":{"v":9007199254740992}}
"#;
    let (mut world, pending) = make_world(original, parsed);
    let req = request(&world, &pending[0]);
    let handle = begin(&mut world, &req);
    let mut refused = false;
    for _ in 0..4000 {
        match world.compute_with(&handle, work_budget()) {
            Ok(action) => assert!(!matches!(
                m::apply(&mut world.writer, action).unwrap(),
                m::Applied::Source(_)
            )),
            Err(_) => {
                refused = true;
                break;
            }
        }
    }
    assert!(
        refused,
        "opaque extra must independently verify against retained input"
    );
    assert!(
        world
            .writer
            .current_source_projection(&world.head.token, &req.logical_key, 65536)
            .unwrap()
            .is_none()
    );
}

#[path = "claude_stop_details.rs"]
mod stop_details;

#[path = "claude_block_extras.rs"]
mod block_extras;

#[path = "claude_provider_usage.rs"]
mod provider_usage;

#[path = "claude_record_extras.rs"]
mod record_extras;

#[path = "claude_parents.rs"]
mod parents;
#[path = "claude_result_metadata.rs"]
mod result_metadata;

#[path = "claude_small_fields.rs"]
mod small_fields;

#[path = "claude_bookkeeping.rs"]
mod bookkeeping;

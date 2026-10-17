//! Source prerequisites over the real synthetic Morrow input, not client/M1 proof.
use polylogue_model::{
    MessageId, Origin, SessionId, WrittenBy,
    merge::{BlockField, MessageField},
};
use polylogue_sources::{
    claude_code::{
        self, FileCtx, KeyedRecords,
        keyed::{MemoryReadView, ReadView, ResponseKey},
        projection::{CountingObligation, Field, RecordCursor, Unit},
        selection::{AuthorLeaf, Transform},
    },
    frame::drive_jsonl,
    selection::{PathPart, Recipe},
};
use serde_json::{Value, json};
const SID: &str = "7f3a2c10-0001-4000-8000-000000000001";
const MORROW: &[u8] = include_bytes!("fixtures/morrow-main.jsonl");
fn run(memory: &mut MemoryReadView, bytes: &[u8], base: u64) -> claude_code::StepOutput {
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main(SID), &*memory).unwrap(),
        bytes,
        base,
    );
    let mut output = framed.output.unwrap();
    output.header.parsed_upto = framed.parsed_upto;
    memory.apply(output.header.clone(), output.delta.clone());
    output
}
fn evaluate(recipe: &impl Recipe<Error = claude_code::selection::Error>, bytes: &[u8]) -> Value {
    let mut output = Vec::new();
    polylogue_sources::selection::write_value(
        recipe,
        |range| -> Result<Vec<u8>, std::io::Error> {
            Ok(bytes[range.start as usize..range.end as usize].to_vec())
        },
        1 << 20,
        &mut output,
    )
    .unwrap();
    serde_json::from_slice(&output).unwrap()
}
fn units(c: &claude_code::keyed::Contribution, item: u64) -> Vec<Unit> {
    let mut cursor = RecordCursor::default();
    let mut units = Vec::new();
    while let Some(unit) = c.next_output(item, &mut cursor).unwrap() {
        units.push(unit);
    }
    units
}
#[test]
fn morrow_has_exact_native_message_identity_source_order_and_replayable_typed_leaves() {
    let mut memory = MemoryReadView::default();
    let output = run(&mut memory, MORROW, 0);
    assert_eq!(output.delta.responses.len(), 12);
    assert_eq!(output.delta.sessions.len(), 1);
    assert_eq!(output.delta.sessions[0].next_message_ordinal, 12);
    assert_eq!(
        output
            .delta
            .pending_messages
            .iter()
            .map(|m| m.ordinal)
            .collect::<Vec<_>>(),
        (0..12).collect::<Vec<_>>()
    );
    let raw: Vec<Value> = MORROW
        .split(|b| *b == b'\n')
        .filter(|b| !b.is_empty())
        .map(|b| serde_json::from_slice(b).unwrap())
        .filter(|v: &Value| matches!(v["type"].as_str(), Some("user" | "assistant")))
        .collect();
    let session = SessionId::new(&Origin::new("claude-code").unwrap(), SID).unwrap();
    let mut prompts = 0;
    let mut signatures = 0;
    let mut calls = 0;
    let mut results = 0;
    for ((_, _, response), raw) in output.delta.responses.iter().zip(&raw) {
        let uuid = raw["uuid"].as_str().unwrap();
        assert_eq!(
            response.message.id,
            MessageId::new(session.clone(), uuid).unwrap()
        );
        assert_eq!(response.items, 1);
        let header = response.output_header().unwrap();
        assert_eq!(header.ordinal, response.ordinal);
        let author = response.author_selection.as_ref().unwrap();
        assert_eq!(
            evaluate(author, MORROW),
            serde_json::to_value(response.message.written_by).unwrap()
        );
        if response.message.written_by == WrittenBy::Human {
            prompts += 1;
            assert_eq!(
                evaluate(&author.for_leaf(AuthorLeaf::PromptKind), MORROW),
                "prose"
            );
        }
        let c = memory
            .contribution(&response.message.id, 0)
            .unwrap()
            .unwrap();
        for unit in units(&c, 0) {
            match unit {
                Unit::Fact { fact, .. } => match (&fact.field, &fact.selection) {
                    (
                        Field::Block {
                            field: BlockField::Tool,
                            ..
                        },
                        Some(recipe),
                    ) => {
                        assert!(evaluate(recipe, MORROW).is_string());
                        calls += 1;
                    }
                    (
                        Field::Block {
                            field: BlockField::Outcome,
                            ..
                        },
                        Some(recipe),
                    ) => {
                        assert!(matches!(
                            evaluate(recipe, MORROW).as_str(),
                            Some("ok" | "error")
                        ));
                        results += 1;
                    }
                    (Field::FirstMessage(MessageField::At), Some(recipe)) => {
                        let value = evaluate(recipe, MORROW);
                        assert_eq!(value["time_quality"], "intrinsic");
                        assert!(value["at"].is_i64());
                    }
                    (_, Some(recipe)) => {
                        evaluate(recipe, MORROW);
                    }
                    (field, None) => panic!("Morrow field lacks selector: {field:?}"),
                },
                Unit::Signature {
                    value,
                    selection: Some(recipe),
                    ..
                } => {
                    assert_eq!(evaluate(&recipe, MORROW), value);
                    signatures += 1;
                }
                Unit::Counting { obligation, .. } => {
                    panic!("unexpected Morrow counting obligation: {obligation:?}")
                }
                _ => {}
            }
        }
        for index in 0..4 {
            for fact in response.selected_unit(index).unwrap().facts {
                evaluate(
                    fact.selection
                        .as_ref()
                        .expect("selected Morrow field selector"),
                    MORROW,
                );
            }
        }
    }
    assert_eq!(prompts, 3);
    assert!(signatures > 0);
    assert!(calls > 0);
    assert!(results > 0);
}
fn line(uuid: &str, response: &str, content: Value) -> String {
    json!({"type":"assistant","uuid":uuid,"requestId":"request","sessionId":SID,"timestamp":"2026-10-01T10:00:00Z","message":{"id":response,"model":"original-model","content":content}}).to_string()+"\n"
}
#[test]
fn uuid_overlay_retains_original_nested_donors_and_first_response_ordinal() {
    let first = line(
        "z-first",
        "response-z",
        json!([{"type":"text","text":"retained"}]),
    );
    let other = line(
        "a-second",
        "response-a",
        json!([{"type":"text","text":"other"}]),
    );
    let tail = line("tail", "response-z", json!([{"type":"text","text":"tail"}]));
    let update=json!({"type":"assistant","uuid":"z-first","message":{"usage":{"input_tokens":0,"output_tokens":2}}}).to_string()+"\n";
    let text = format!("{first}{other}{tail}{update}");
    let mut memory = MemoryReadView::default();
    let output = run(&mut memory, text.as_bytes(), 0);
    assert_eq!(
        output
            .delta
            .responses
            .iter()
            .map(|(_, _, r)| r.ordinal)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    let response = memory
        .response(SID, &ResponseKey::new("response-z", "request"))
        .unwrap()
        .unwrap();
    assert_eq!(response.ordinal, 0);
    assert_eq!(response.items, 2);
    let c = memory
        .contribution(&response.message.id, 0)
        .unwrap()
        .unwrap();
    let fact = c
        .projection
        .facts
        .iter()
        .find(|f| {
            matches!(
                f.field,
                Field::Block {
                    field: BlockField::Text,
                    ..
                }
            )
        })
        .unwrap();
    let recipe = fact.selection.as_ref().unwrap();
    assert_eq!(recipe.inputs[0].record.start, 0);
    assert_eq!(recipe.inputs[0].record.end, (first.len() - 1) as u64);
    assert_eq!(evaluate(recipe, text.as_bytes()), "retained");
    let selected = response.selected_unit(0).unwrap();
    assert_eq!(
        response.selected_origins[0].as_ref().unwrap().record.start,
        (first.len() + other.len() + tail.len()) as u64
    );
    for fact in selected.facts {
        let recipe = fact.selection.unwrap();
        assert_eq!(recipe.transform, Transform::JsonValueV1);
        assert!(evaluate(&recipe, text.as_bytes()).is_u64());
    }
    let (_, key, second) = &output.delta.responses[1];
    assert_eq!(
        key.response,
        serde_json::to_string(&polylogue_model::NativeId::unicode("response-a")).unwrap()
    );
    assert_eq!(
        second.message.id,
        MessageId::new(second.message.id.session().clone(), "a-second").unwrap()
    );
    // A coordinate shift changes donor references, never semantic evidence bytes.
    let mut shifted = MemoryReadView::default();
    run(&mut shifted, text.as_bytes(), 1000);
    let moved = shifted
        .response(SID, &ResponseKey::new("response-z", "request"))
        .unwrap()
        .unwrap();
    for item in 0..response.items {
        assert_eq!(
            memory
                .contribution(&response.message.id, item)
                .unwrap()
                .unwrap()
                .record_unit(item),
            shifted
                .contribution(&moved.message.id, item)
                .unwrap()
                .unwrap()
                .record_unit(item)
        );
    }
}
#[test]
fn signature_is_an_auxiliary_unit_and_author_correction_stays_unresolved() {
    let first = line(
        "a",
        "response",
        json!([{"type":"thinking","thinking":"","signature":"original-signature"}]),
    );
    let mut memory = MemoryReadView::default();
    run(&mut memory, first.as_bytes(), 0);
    let response = memory
        .response(SID, &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap();
    let c = memory
        .contribution(&response.message.id, 0)
        .unwrap()
        .unwrap();
    assert!(
        !c.projection
            .facts
            .iter()
            .any(|f| f.source_path.last().map(String::as_str) == Some("signature"))
    );
    assert!(
        units(&c, 0)
            .iter()
            .any(|u| matches!(u,Unit::Signature {value,..} if value=="original-signature"))
    );
    let user=json!({"type":"user","uuid":"u","sessionId":SID,"promptSource":"typed","message":{"content":"human"}}).to_string()+"\n";
    let change = json!({"type":"user","uuid":"u","promptSource":"sdk"}).to_string() + "\n";
    let mut memory = MemoryReadView::default();
    run(&mut memory, format!("{user}{change}").as_bytes(), 0);
    let response = memory
        .response(SID, &ResponseKey::user("u"))
        .unwrap()
        .unwrap();
    let c = memory
        .contribution(&response.message.id, 0)
        .unwrap()
        .unwrap();
    assert!(
        c.projection
            .obligations
            .contains(&CountingObligation::FirstAuthorUpdate)
    );
    assert_eq!(response.message.written_by, WrittenBy::Human);
    assert_eq!(
        response
            .author_selection
            .as_ref()
            .unwrap()
            .input
            .record
            .start,
        0
    );
}
#[test]
fn direct_paths_distinguish_numeric_object_keys_and_literal_dotted_names() {
    let text = line(
        "a",
        "response",
        json!([{"type":"text","text":"x","0":false,"a.b":{"":0}}]),
    );
    let mut memory = MemoryReadView::default();
    run(&mut memory, text.as_bytes(), 0);
    let response = memory
        .response(SID, &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap();
    let c = memory
        .contribution(&response.message.id, 0)
        .unwrap()
        .unwrap();
    let fact = c
        .projection
        .facts
        .iter()
        .find(|f| f.source_path.last().map(String::as_str) == Some("0"))
        .unwrap();
    let recipe = fact.selection.as_ref().unwrap();
    assert_eq!(
        recipe.inputs[0].path.last(),
        Some(&PathPart::Key("0".into()))
    );
    assert_eq!(evaluate(recipe, text.as_bytes()), false);
    let fact = c
        .projection
        .facts
        .iter()
        .find(|f| f.source_path.last().map(String::as_str) == Some("a.b"))
        .unwrap();
    assert_eq!(
        evaluate(fact.selection.as_ref().unwrap(), text.as_bytes()),
        json!({"":0})
    );
}

#[test]
fn inline_output_and_named_channels_remain_distinct_with_known_empty_stderr() {
    let raw=json!({"type":"user","uuid":"result","sessionId":SID,"timestamp":"2026-10-01T00:00:00Z","message":{"content":[{"type":"tool_result","tool_use_id":"call","content":"inline wins","is_error":false}]},"toolUseResult":{"stdout":"different stdout","stderr":"","interrupted":false,"exitCode":0}}).to_string()+"\n";
    let mut memory = MemoryReadView::default();
    run(&mut memory, raw.as_bytes(), 0);
    let response = memory
        .response(SID, &ResponseKey::user("result"))
        .unwrap()
        .unwrap();
    let c = memory
        .contribution(&response.message.id, 0)
        .unwrap()
        .unwrap();
    let polylogue_model::BlockContent::ToolResult(result) = &c.blocks[0] else {
        panic!("result")
    };
    assert_eq!(
        result.output,
        Some(polylogue_model::ToolOutput::Text("inline wins".into()))
    );
    assert_eq!(
        result.data.as_ref().unwrap()["stdout"].as_str(),
        Some("different stdout")
    );
    assert_eq!(result.data.as_ref().unwrap()["stderr"].as_str(), Some(""));
    assert_eq!(result.exit_code, Some(0));
    assert!(c.projection.obligations.is_empty());
    assert_eq!(
        c.projection
            .facts
            .iter()
            .filter(|f| matches!(
                f.field,
                Field::Block {
                    field: BlockField::Outcome,
                    ..
                }
            ))
            .count(),
        1
    );
    for (key, expected) in [
        ("stdout", json!("different stdout")),
        ("stderr", json!("")),
        ("interrupted", json!(false)),
    ] {
        let fact = c
            .projection
            .facts
            .iter()
            .find(|f| matches!(&f.field,Field::ResultData {key:found,..} if found==key))
            .unwrap();
        assert_eq!(
            evaluate(fact.selection.as_ref().unwrap(), raw.as_bytes()),
            expected
        );
    }
    for fact in c.projection.facts.iter().filter(|f| {
        matches!(
            f.field,
            Field::Block {
                field: BlockField::Outcome | BlockField::ExitCode,
                ..
            }
        )
    }) {
        evaluate(fact.selection.as_ref().unwrap(), raw.as_bytes());
    }
}
#[test]
fn supplied_status_mapping_and_selector_share_the_classifier_without_double_counting() {
    for (meta, reason) in [
        (json!({"interrupted":true}), "interrupted"),
        (json!({"timedOutAfterMs":0}), "timed_out"),
    ] {
        let raw=json!({"type":"user","uuid":"result","sessionId":SID,"message":{"content":[{"type":"tool_result","tool_use_id":"call","content":"","is_error":false}]},"toolUseResult":meta}).to_string()+"\n";
        let mut memory = MemoryReadView::default();
        run(&mut memory, raw.as_bytes(), 0);
        let response = memory
            .response(SID, &ResponseKey::user("result"))
            .unwrap()
            .unwrap();
        let c = memory
            .contribution(&response.message.id, 0)
            .unwrap()
            .unwrap();
        assert_eq!(
            c.projection
                .facts
                .iter()
                .filter(|f| matches!(
                    f.field,
                    Field::Block {
                        field: BlockField::Outcome,
                        ..
                    }
                ))
                .count(),
            1
        );
        let outcome = c
            .projection
            .facts
            .iter()
            .find(|f| {
                matches!(
                    f.field,
                    Field::Block {
                        field: BlockField::Outcome,
                        ..
                    }
                )
            })
            .unwrap();
        assert_eq!(
            evaluate(outcome.selection.as_ref().unwrap(), raw.as_bytes()),
            "error"
        );
        let why = c
            .projection
            .retained
            .iter()
            .find(|f| {
                matches!(
                    f.field,
                    Field::Block {
                        field: BlockField::OutcomeReason,
                        ..
                    }
                )
            })
            .unwrap();
        assert_eq!(
            evaluate(why.selection.as_ref().unwrap(), raw.as_bytes()),
            reason
        );
        assert!(!c.projection.facts.iter().any(|fact| matches!(
            fact.field,
            Field::Block {
                field: BlockField::OutcomeReason,
                ..
            }
        )));
        assert!(units(&c,0).iter().any(|unit| matches!(unit,Unit::Retained{fact,..} if matches!(fact.field,Field::Block{field:BlockField::OutcomeReason,..}))));
    }
}

#[test]
fn channels_never_fabricate_missing_inline_output_or_a_wrapper_count() {
    let raw=json!({"type":"user","uuid":"result","sessionId":SID,"message":{"content":[{"type":"tool_result","tool_use_id":"call"}]},"toolUseResult":{"stdout":"channel","stderr":""}}).to_string()+"\n";
    let mut memory = MemoryReadView::default();
    run(&mut memory, raw.as_bytes(), 0);
    let response = memory
        .response(SID, &ResponseKey::user("result"))
        .unwrap()
        .unwrap();
    let c = memory
        .contribution(&response.message.id, 0)
        .unwrap()
        .unwrap();
    let polylogue_model::BlockContent::ToolResult(result) = &c.blocks[0] else {
        panic!("result")
    };
    assert_eq!(result.output, None);
    assert_eq!(result.data.as_ref().unwrap()["stderr"].as_str(), Some(""));
    assert!(!c.projection.facts.iter().any(|f| matches!(
        f.field,
        Field::Block {
            field: BlockField::Output | BlockField::Outcome,
            ..
        }
    )));
    assert_eq!(
        c.projection
            .facts
            .iter()
            .filter(|f| matches!(f.field, Field::ResultData { .. }))
            .count(),
        2
    );
}

#[test]
fn false_interrupted_is_its_own_supplied_status_not_a_default_outcome_witness() {
    for is_error in [None, Some(false)] {
        let mut block = json!({"type":"tool_result","tool_use_id":"call","content":""});
        if let Some(value) = is_error {
            block["is_error"] = json!(value);
        }
        let raw = json!({"type":"user","uuid":"result","sessionId":SID,
            "message":{"content":[block]},"toolUseResult":{"interrupted":false}})
        .to_string()
            + "\n";
        let mut memory = MemoryReadView::default();
        run(&mut memory, raw.as_bytes(), 0);
        let response = memory
            .response(SID, &ResponseKey::user("result"))
            .unwrap()
            .unwrap();
        let c = memory
            .contribution(&response.message.id, 0)
            .unwrap()
            .unwrap();
        let polylogue_model::BlockContent::ToolResult(result) = &c.blocks[0] else {
            panic!("result")
        };
        assert_eq!(result.outcome, polylogue_model::ToolOutcome::Ok);
        assert_eq!(
            result.data.as_ref().unwrap()["interrupted"].as_bool(),
            Some(false)
        );
        assert!(c.projection.obligations.is_empty());
        assert_eq!(
            c.projection
                .facts
                .iter()
                .filter(|fact| matches!(
                    fact.field,
                    Field::Block {
                        field: BlockField::Outcome,
                        ..
                    }
                ))
                .count(),
            usize::from(is_error.is_some())
        );
        let flags: Vec<_> = c
            .projection
            .facts
            .iter()
            .filter(|fact| {
                matches!(&fact.field,
            Field::ResultData{key,..} if key == "interrupted")
            })
            .collect();
        assert_eq!(flags.len(), 1);
        let recipe = flags[0].selection.as_ref().unwrap();
        assert_eq!(recipe.inputs[0].record.start, 0);
        assert_eq!(
            recipe.inputs[0].record.end,
            raw.trim_end_matches('\n').len() as u64
        );
        assert_eq!(
            recipe.inputs[0].path,
            vec![
                PathPart::Key("toolUseResult".into()),
                PathPart::Key("interrupted".into())
            ]
        );
        assert_eq!(evaluate(recipe, raw.as_bytes()), false);
        assert!(!c.projection.facts.iter().any(|fact| matches!(
            fact.field,
            Field::Block {
                field: BlockField::OutcomeReason,
                ..
            }
        )));
    }
}

#[test]
fn parent_reference_uses_session_scoped_record_address_and_original_donor() {
    let first = line(
        "parent:opaque",
        "parent-response",
        json!([{"type":"text","text":"parent"}]),
    );
    let child = json!({"type":"user","uuid":"child","sessionId":SID,"parentUuid":"parent:opaque","message":{"content":"child"}}).to_string()+"\n";
    let raw = first.clone() + &child;
    let mut memory = MemoryReadView::default();
    run(&mut memory, raw.as_bytes(), 0);
    let parent = memory
        .response(SID, &ResponseKey::new("parent-response", "request"))
        .unwrap()
        .unwrap();
    let response = memory
        .response(SID, &ResponseKey::user("child"))
        .unwrap()
        .unwrap();
    assert_eq!(response.message.parent, Some(parent.message.id.clone()));
    let c = memory
        .contribution(&response.message.id, 0)
        .unwrap()
        .unwrap();
    assert!(c.projection.obligations.is_empty());
    let fact = c
        .projection
        .facts
        .iter()
        .find(|fact| matches!(fact.field, Field::FirstMessage(MessageField::Parent)))
        .unwrap();
    let recipe = fact.selection.as_ref().unwrap();
    assert_eq!(recipe.inputs[0].record.start, first.len() as u64);
    let value = evaluate(recipe, raw.as_bytes());
    assert_eq!(value["id"], parent.message.id.as_str());
    assert_eq!(value["session"], parent.message.id.session().as_str());
    let mut changed = raw.clone().into_bytes();
    let start = first.len();
    let changed_child = child.replace("parent:opaque", "wrong!:opaque");
    changed.truncate(start);
    changed.extend_from_slice(changed_child.as_bytes());
    let mut output = Vec::new();
    assert!(
        recipe
            .write_value(
                |range| -> Result<Vec<u8>, std::io::Error> {
                    Ok(changed[range.start as usize..range.end as usize].to_vec())
                },
                1 << 20,
                &mut output
            )
            .is_err()
    );
}
#[test]
fn unresolved_forward_and_grouped_parents_do_not_guess_global_message_ids() {
    for grouped in [false, true] {
        let first = line("first", "response", json!([{"type":"text","text":"one"}]));
        let second = line("grouped", "response", json!([{"type":"text","text":"two"}]));
        let parent_uuid = if grouped { "grouped" } else { "not-seen" };
        let child = json!({"type":"user","uuid":"child","sessionId":SID,"parentUuid":parent_uuid,"message":{"content":"child"}}).to_string()+"\n";
        let mut memory = MemoryReadView::default();
        run(&mut memory, (first + &second + &child).as_bytes(), 0);
        let response = memory
            .response(SID, &ResponseKey::user("child"))
            .unwrap()
            .unwrap();
        assert!(response.message.parent.is_none());
        let c = memory
            .contribution(&response.message.id, 0)
            .unwrap()
            .unwrap();
        assert!(
            c.projection
                .obligations
                .contains(&CountingObligation::ParentResolution)
        );
        assert!(
            c.projection
                .facts
                .iter()
                .find(|fact| matches!(fact.field, Field::FirstMessage(MessageField::Parent)))
                .unwrap()
                .selection
                .is_none()
        );
    }
}

#[test]
fn contribution_origins_roundtrip_as_typed_entries_and_reject_duplicate_paths() {
    let first = line(
        "first",
        "response",
        json!([{"type":"text","text":"retained"}]),
    );
    let mut live = MemoryReadView::default();
    let output = run(&mut live, first.as_bytes(), 0);
    let encoded = serde_json::to_vec(&output.delta).unwrap();
    let decoded = serde_json::from_slice(&encoded).unwrap();
    let mut reopened = MemoryReadView::default();
    reopened.apply(output.header, decoded);
    let message = live
        .response(SID, &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap()
        .message
        .id;
    let contribution = live.contribution(&message, 0).unwrap().unwrap();
    let encoded_contribution = serde_json::to_vec(&contribution).unwrap();
    let restored: claude_code::keyed::Contribution =
        serde_json::from_slice(&encoded_contribution).unwrap();
    assert_eq!(contribution, restored);
    let append = json!({"type":"assistant","uuid":"first","message":{"usage":{"input_tokens":0}}})
        .to_string()
        + "\n";
    run(&mut live, append.as_bytes(), first.len() as u64);
    run(&mut reopened, append.as_bytes(), first.len() as u64);
    assert_eq!(
        live.contribution(&message, 0).unwrap(),
        reopened.contribution(&message, 0).unwrap()
    );
    assert_eq!(
        live.response(SID, &ResponseKey::new("response", "request"))
            .unwrap(),
        reopened
            .response(SID, &ResponseKey::new("response", "request"))
            .unwrap()
    );
    let mut invalid: Value = serde_json::from_slice(&encoded_contribution).unwrap();
    let fields = invalid["origins"]["fields"].as_array_mut().unwrap();
    assert!(!fields.is_empty());
    fields.push(fields[0].clone());
    let error = serde_json::from_value::<claude_code::keyed::Contribution>(invalid).unwrap_err();
    assert!(error.to_string().contains("duplicate Claude origin path"));
}

#[test]
fn signature_point_unit_matches_the_source_cursor_without_becoming_a_fact() {
    let raw = line(
        "thinking",
        "response",
        json!([{"type":"text","text":"before"},{"type":"thinking","thinking":"reason","signature":"auxiliary"}]),
    );
    let mut memory = MemoryReadView::default();
    run(&mut memory, raw.as_bytes(), 0);
    let response = memory
        .response(SID, &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap();
    let c = memory
        .contribution(&response.message.id, 0)
        .unwrap()
        .unwrap();
    let point = c.signature_unit(0, 1).unwrap().unwrap();
    assert!(units(&c, 0).contains(&point));
    assert!(c.signature_unit(0, 0).unwrap().is_none());
    assert!(c.signature_unit(0, 2).unwrap().is_none());
    assert!(!c.projection.facts.iter().any(|fact| matches!(
        fact.field,
        Field::Block {
            field: BlockField::ThinkingSignature,
            ..
        }
    )));
}

fn correction_units(
    memory: &MemoryReadView,
    response: &claude_code::keyed::ResponseCell,
) -> Vec<claude_code::keyed::EvidenceUnit> {
    let mut baseline: [Option<Vec<u8>>; 5] = Default::default();
    for item in 0..response.items {
        let c = memory
            .contribution(&response.message.id, item)
            .unwrap()
            .unwrap();
        for (previous, current) in baseline.iter_mut().zip(c.final_fields().unwrap()) {
            if current.is_some() {
                *previous = current
            }
        }
    }
    response.metadata_units(&baseline)
}
#[test]
fn stop_sequence_is_one_normalized_selected_extra_with_original_donor_and_raw_correction() {
    let a=json!({"type":"assistant","uuid":"a","requestId":"request","sessionId":SID,"message":{"id":"response","content":[{"type":"text","text":"a"}],"stop_sequence":"A"}}).to_string()+"\n";
    let b=json!({"type":"assistant","uuid":"b","requestId":"request","sessionId":SID,"message":{"id":"response","content":[{"type":"text","text":"b"}],"stop_sequence":"B"}}).to_string()+"\n";
    let changed_a=json!({"type":"assistant","uuid":"a","requestId":"request","sessionId":SID,"message":{"id":"response","content":[{"type":"text","text":"changed"}],"stop_sequence":"A"}}).to_string()+"\n";
    let mut duplicate = MemoryReadView::default();
    let duplicate_raw = a.clone() + &b + &a;
    run(&mut duplicate, duplicate_raw.as_bytes(), 0);
    let r = duplicate
        .response(SID, &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap();
    assert_eq!(
        r.message.attrs.as_ref().unwrap()["stop_sequence"].as_str(),
        Some("B")
    );
    assert!(correction_units(&duplicate, &r).is_empty());
    let omitted_a=json!({"type":"assistant","uuid":"a","requestId":"request","sessionId":SID,"message":{"id":"response","content":[{"type":"text","text":"omitted-stop-update"}]}}).to_string()+"\n";
    let mut omitted_update = MemoryReadView::default();
    run(
        &mut omitted_update,
        (a.clone() + &b + &omitted_a).as_bytes(),
        0,
    );
    let omitted_response = omitted_update
        .response(SID, &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap();
    assert_eq!(
        omitted_response.message.attrs.as_ref().unwrap()["stop_sequence"].as_str(),
        Some("B")
    );
    assert_eq!(
        omitted_response
            .stop_sequence
            .as_ref()
            .unwrap()
            .selection
            .as_ref()
            .unwrap()
            .inputs[0]
            .record
            .start,
        a.len() as u64
    );
    assert!(correction_units(&omitted_update, &omitted_response).is_empty());
    let mut changed = MemoryReadView::default();
    let changed_raw = a.clone() + &b + &changed_a;
    run(&mut changed, changed_raw.as_bytes(), 0);
    let r = changed
        .response(SID, &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap();
    assert_eq!(
        r.message.attrs.as_ref().unwrap()["stop_sequence"].as_str(),
        Some("A")
    );
    let selected = r.stop_sequence.as_ref().unwrap();
    let recipe = selected.selection.as_ref().unwrap();
    assert_eq!(recipe.inputs[0].record.start, (a.len() + b.len()) as u64);
    assert_eq!(evaluate(recipe, changed_raw.as_bytes()), "A");
    assert!(
        matches!(correction_units(&changed,&r).as_slice(),[claude_code::keyed::EvidenceUnit::Json{tag,canonical,whole_record:false}] if tag=="response/stop_sequence"&&canonical==br#""A""#)
    );
    let mut base_order = MemoryReadView::default();
    run(&mut base_order, (changed_a.clone() + &b).as_bytes(), 0);
    let base = base_order
        .response(SID, &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap();
    assert_eq!(
        base.message.attrs.as_ref().unwrap()["stop_sequence"].as_str(),
        Some("B")
    );
    assert!(correction_units(&base_order, &base).is_empty());
    for item in 0..2 {
        assert_eq!(
            changed
                .contribution(&r.message.id, item)
                .unwrap()
                .unwrap()
                .record_unit(item),
            base_order
                .contribution(&base.message.id, item)
                .unwrap()
                .unwrap()
                .record_unit(item)
        )
    }
    let null_a=json!({"type":"assistant","uuid":"a","requestId":"request","sessionId":SID,"message":{"id":"response","content":[{"type":"text","text":"null-update"}],"stop_sequence":null}}).to_string()+"\n";
    let mut null_selected = MemoryReadView::default();
    run(&mut null_selected, (a + &b + &null_a).as_bytes(), 0);
    let r = null_selected
        .response(SID, &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap();
    assert!(
        matches!(correction_units(&null_selected,&r).as_slice(),[claude_code::keyed::EvidenceUnit::Json{tag,canonical,whole_record:false}] if tag=="response/stop_sequence"&&canonical==b"null")
    );
}
#[test]
fn stop_sequence_null_is_supplied_and_omission_preserves_its_original_donor() {
    let a=json!({"type":"assistant","uuid":"a","requestId":"request","sessionId":SID,"message":{"id":"response","content":[{"type":"text","text":"a"}],"stop_sequence":null}}).to_string()+"\n";
    let omitted=json!({"type":"assistant","uuid":"b","requestId":"request","sessionId":SID,"message":{"id":"response","content":[{"type":"text","text":"b"}]}}).to_string()+"\n";
    let mut memory = MemoryReadView::default();
    run(&mut memory, a.as_bytes(), 0);
    let single = memory
        .response(SID, &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap();
    assert!(correction_units(&memory, &single).is_empty());
    assert_eq!(single.stop_sequence_raw.as_deref(), Some(&b"null"[..]));
    run(&mut memory, omitted.as_bytes(), a.len() as u64);
    let r = memory
        .response(SID, &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap();
    assert_eq!(
        r.message.attrs.as_ref().unwrap().get("stop_sequence"),
        Some(&polylogue_model::RetainedJson::Null)
    );
    let fact = r.stop_sequence.as_ref().unwrap();
    assert_eq!(fact.selection.as_ref().unwrap().inputs[0].record.start, 0);
    assert_eq!(
        evaluate(fact.selection.as_ref().unwrap(), (a + &omitted).as_bytes()),
        Value::Null
    );
    assert!(correction_units(&memory, &r).is_empty());
    let missing=json!({"type":"assistant","uuid":"missing","requestId":"request","sessionId":SID,"message":{"id":"other","content":[{"type":"text","text":"none"}]}}).to_string()+"\n";
    let mut empty = MemoryReadView::default();
    run(&mut empty, missing.as_bytes(), 0);
    let r = empty
        .response(SID, &ResponseKey::new("other", "request"))
        .unwrap()
        .unwrap();
    assert!(r.stop_sequence.is_none());
    assert!(r.stop_sequence_raw.is_none());
}

#[test]
fn morrow_preflight_enumerates_every_extra_and_checks_normalized_source_carriers() {
    use polylogue_model::{
        merge::Slot,
        source_leaf::{self, Projection, Subject},
    };
    use std::collections::BTreeSet;
    let mut memory = MemoryReadView::default();
    let output = run(&mut memory, MORROW, 0);
    let mut observed = BTreeSet::new();
    let mut unsupported = BTreeSet::new();
    let mut typed_tier_paths = BTreeSet::new();
    let mut typed_tiers = 0;
    for (_, _, r) in &output.delta.responses {
        let mut inspect = |fact: &claude_code::projection::Fact| match &fact.field {
            Field::Extra { .. } => {
                observed.insert(fact.source_path.clone());
                unsupported.insert(fact.source_path.clone());
            }
            Field::SelectedExtra { key } => {
                observed.insert(fact.source_path.clone());
                let projected = source_leaf::encode(
                    Subject::Message(&r.message),
                    &Slot::Extra {
                        block: None,
                        key: key.clone(),
                    },
                    65536,
                )
                .unwrap();
                let Projection::Encoded(bytes) = projected else {
                    panic!("missing normalized carrier for {:?}", fact.source_path)
                };
                assert_eq!(
                    serde_json::from_slice::<Value>(&bytes).unwrap(),
                    evaluate(fact.selection.as_ref().unwrap(), MORROW),
                    "normalized donor differs at {:?}",
                    fact.source_path
                );
            }
            Field::Usage(polylogue_model::merge::UsageField::ServiceTier) => {
                typed_tier_paths.insert(fact.source_path.clone());
                typed_tiers += 1;
                let Projection::Encoded(bytes) = source_leaf::encode(
                    Subject::Message(&r.message),
                    &Slot::Usage(polylogue_model::merge::UsageField::ServiceTier),
                    65536,
                )
                .unwrap() else {
                    panic!("missing typed service tier carrier")
                };
                assert_eq!(
                    serde_json::from_slice::<Value>(&bytes).unwrap(),
                    evaluate(fact.selection.as_ref().unwrap(), MORROW)
                );
            }
            _ => {}
        };
        for item in 0..r.items {
            let c = memory.contribution(&r.message.id, item).unwrap().unwrap();
            for fact in c.projection.facts.iter().chain(&c.projection.retained) {
                inspect(fact)
            }
        }
        for index in 0..4 {
            for fact in &r.selected_unit(index).unwrap().facts {
                inspect(fact)
            }
        }
        if let Some(fact) = &r.stop_sequence {
            inspect(fact)
        }
    }
    eprintln!(
        "Morrow source extra paths: {observed:?}; typed usage tiers: {typed_tier_paths:?}; unsupported: {unsupported:?}"
    );
    assert_eq!(typed_tiers, 6);
    assert_eq!(
        typed_tier_paths,
        BTreeSet::from([vec![
            "message".into(),
            "usage".into(),
            "service_tier".into()
        ]])
    );
    assert!(
        unsupported.is_empty(),
        "all unsupported Morrow extras: {unsupported:?}"
    );
    assert_eq!(
        observed,
        BTreeSet::from([vec!["message".into(), "stop_sequence".into()]])
    );
}
#[test]
fn selected_usage_tier_is_typed_string_with_null_unknown_and_replacement_omission() {
    use polylogue_model::merge::UsageField;
    for tier in [Value::Null, json!("standard"), json!("")] {
        let first=json!({"type":"assistant","uuid":"a","requestId":"request","sessionId":SID,"message":{"id":"response","content":[{"type":"text","text":"a"}],"usage":{"input_tokens":0,"service_tier":tier},"stop_sequence":null}}).to_string()+"\n";
        let second = json!({"type":"assistant","uuid":"a","message":{"usage":{"input_tokens":1}}})
            .to_string()
            + "\n";
        let mut memory = MemoryReadView::default();
        run(&mut memory, first.as_bytes(), 0);
        let r = memory
            .response(SID, &ResponseKey::new("response", "request"))
            .unwrap()
            .unwrap();
        assert_eq!(
            r.message.usage.as_ref().unwrap().service_tier.as_deref(),
            tier.as_str()
        );
        assert!(
            r.message
                .attrs
                .as_ref()
                .unwrap()
                .get("service_tier")
                .is_none()
        );
        let selected = r.selected_unit(0).unwrap();
        let fact = selected
            .facts
            .iter()
            .find(|fact| matches!(fact.field, Field::Usage(UsageField::ServiceTier)));
        if tier.is_string() {
            let fact = fact.unwrap();
            assert_eq!(
                evaluate(fact.selection.as_ref().unwrap(), first.as_bytes()),
                tier
            );
            assert_eq!(fact.selection.as_ref().unwrap().inputs[0].record.start, 0)
        } else {
            assert!(fact.is_none());
            assert!(
                String::from_utf8(r.selected[0].clone().unwrap())
                    .unwrap()
                    .contains("null")
            )
        }
        run(&mut memory, second.as_bytes(), first.len() as u64);
        let r = memory
            .response(SID, &ResponseKey::new("response", "request"))
            .unwrap()
            .unwrap();
        assert!(r.message.usage.as_ref().unwrap().service_tier.is_none());
        assert_eq!(
            r.message.attrs.as_ref().unwrap().get("stop_sequence"),
            Some(&polylogue_model::RetainedJson::Null)
        );
        assert!(
            !r.selected_unit(0)
                .unwrap()
                .facts
                .iter()
                .any(|fact| matches!(fact.field, Field::Usage(UsageField::ServiceTier)))
        );
    }
}

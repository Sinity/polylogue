//! Source-local projection/presence, not canonical publication qualification.
use polylogue_model::merge::{BlockField, FinishState, UsageField};
use polylogue_model::{BlockContent, Thinking};
use polylogue_sources::claude_code::keyed::{
    BodyPayload, Contribution, MemoryReadView, ReadView, ResponseCell, ResponseKey,
};
use polylogue_sources::claude_code::projection::{CountingObligation, Field, RecordCursor, Unit};
use polylogue_sources::claude_code::{FileCtx, KeyedRecords};
use polylogue_sources::frame::drive_jsonl;
use serde_json::{Value, json};
fn line(v: Value) -> String {
    v.to_string() + "\n"
}
fn record(uuid: &str, content: Value) -> Value {
    json!({"type":"assistant","uuid":uuid,"requestId":"request","sessionId":"s","timestamp":"2026-10-01T00:00:00Z","message":{"id":"response","content":content}})
}
fn step(memory: &mut MemoryReadView, value: Value) {
    let text = line(value);
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &*memory).unwrap(),
        text.as_bytes(),
        0,
    );
    let out = framed.output.unwrap();
    memory.apply(out.header, out.delta);
}
fn response(memory: &MemoryReadView) -> ResponseCell {
    memory
        .response("s", &ResponseKey::new("response", "request"))
        .unwrap()
        .unwrap()
}
fn contribution(memory: &MemoryReadView, item: u64) -> Contribution {
    memory
        .contribution(&response(memory).message.id, item)
        .unwrap()
        .unwrap()
}
fn units(c: &Contribution, item: u64) -> Vec<Unit> {
    let mut cursor = RecordCursor::default();
    let mut out = Vec::new();
    while let Some(unit) = c.next_output(item, &mut cursor).unwrap() {
        out.push(unit);
        cursor = serde_json::from_slice(&serde_json::to_vec(&cursor).unwrap()).unwrap();
    }
    out
}

#[test]
fn local_units_preserve_empty_blocks_and_exclude_their_metadata_from_counts() {
    let mut memory = MemoryReadView::default();
    let mut value = record(
        "a",
        json!([
        {"type":"text","text":"","hidden":false},
        {"type":"text","text":"visible","":false},
        {"type":"thinking","thinking":"","encrypted":false,"signature":"signature"},
        {"type":"tool_use","name":"","id":"call","input":{"command":"","array":[]}},
        {"type":"tool_use","input":null},
        {"type":"tool_result","content":"","is_error":false},
        {"type":"tool_result","content":[]} ]),
    );
    value[""] = json!([]);
    step(&mut memory, value);
    let c = contribution(&memory, 0);
    let out = units(&c, 0);
    assert!(c.projection.obligations.is_empty());
    let blocks: Vec<_> = out
        .iter()
        .filter_map(|u| {
            if let Unit::Block {
                local,
                nonempty,
                nonempty_text,
                ..
            } = u
            {
                Some((*local, *nonempty, *nonempty_text))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        blocks,
        vec![
            (0, false, false),
            (1, true, true),
            (2, true, false),
            (3, true, false),
            (4, true, false),
            (5, true, false),
            (6, true, false)
        ]
    );
    let facts: Vec<_> = out
        .iter()
        .filter_map(|u| {
            if let Unit::Fact { fact, value, .. } = u {
                Some((&fact.field, value))
            } else {
                None
            }
        })
        .collect();
    assert!(
        !facts
            .iter()
            .any(|(f, _)| matches!(f, Field::Extra { local: Some(0), .. }))
    );
    assert!(facts.iter().any(|(f,v)|matches!(f,Field::Extra{local:Some(1),path} if path.last().is_some_and(String::is_empty)) && **v==json!(false)));
    assert!(facts.iter().any(
        |(f, v)| matches!(f,Field::Extra{local:None,path} if path==&vec![String::new()])
            && **v == json!([])
    ));
    assert!(facts.iter().any(|(f, v)| matches!(
        f,
        Field::Block {
            local: 2,
            field: BlockField::Encrypted
        }
    ) && **v == json!(false)));
    assert!(!facts.iter().any(|(f, _)| matches!(
        f,
        Field::Block {
            field: BlockField::Command,
            ..
        }
    )));
    assert!(facts.iter().any(|(f, v)| matches!(
        f,
        Field::Block {
            local: 4,
            field: BlockField::Input
        }
    ) && **v == Value::Null));
    assert_eq!(
        facts
            .iter()
            .filter(|(f, _)| matches!(
                f,
                Field::Block {
                    field: BlockField::Outcome,
                    ..
                }
            ))
            .count(),
        1,
        "format fallback success is not a supplied status"
    );
    assert!(matches!(
        &c.blocks[2],
        BlockContent::Thinking(Thinking {
            encrypted: Some(false),
            ..
        })
    ));
}

#[test]
fn selected_usage_counts_leaves_once_and_preserves_zero_empty_extra_and_atomic_replacement() {
    let mut memory = MemoryReadView::default();
    let mut a = record("a", json!([{"type":"text","text":"a"}]));
    a["message"]["usage"] = json!({"input_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":1},"output_tokens_details":{"thinking_tokens":0},"":[],"service_tier":""});
    step(&mut memory, a);
    let cell = response(&memory);
    let usage = cell.selected_unit(0).unwrap();
    assert_eq!(usage.facts.len(), 6);
    assert!(
        usage
            .facts
            .iter()
            .any(|f| f.field == Field::Usage(UsageField::Input)
                && usage.fact_value(f).unwrap() == json!(0))
    );
    assert!(usage.facts.iter().any(
        |f| matches!(&f.field,Field::Extra{path,..} if path.last().is_some_and(String::is_empty))
    ));
    assert_eq!(cell.message.usage.as_ref().unwrap().cache_write_5m, Some(0));
    assert_eq!(cell.message.usage.as_ref().unwrap().reasoning, Some(0));
    assert!(
        cell.selected_unit(3).unwrap().facts.is_empty(),
        "one raw timestamp is not counted twice for at/completed_at"
    );
    step(
        &mut memory,
        json!({"type":"assistant","uuid":"a","message":{"usage":{"output_tokens":0}}}),
    );
    let cell = response(&memory);
    let usage = cell.selected_unit(0).unwrap();
    assert_eq!(usage.facts.len(), 1);
    assert_eq!(usage.facts[0].field, Field::Usage(UsageField::Output));
    assert_eq!(cell.message.usage.unwrap().input, None);
}

#[test]
fn replacements_keep_group_local_addresses_and_sparse_updates_keep_fact_provenance() {
    let mut memory = MemoryReadView::default();
    let mut a = record("a", json!([{"type":"text","text":"one"}]));
    a["extra"] = json!(false);
    step(&mut memory, a);
    let first = contribution(&memory, 0);
    let stamp = first
        .projection
        .facts
        .iter()
        .find(|f| matches!(&f.field,Field::Extra{path,..} if path==&vec!["extra".to_owned()]))
        .unwrap()
        .supplied_revision;
    step(
        &mut memory,
        record("b", json!([{"type":"text","text":"later"}])),
    );
    step(
        &mut memory,
        json!({"type":"assistant","uuid":"a","message":{"content":[{"type":"text","text":"changed"},{"type":"text","text":"added"}]}}),
    );
    let cell = response(&memory);
    assert_eq!(cell.items, 2);
    let changed = contribution(&memory, 0);
    assert_eq!(
        changed
            .projection
            .facts
            .iter()
            .find(|f| matches!(&f.field,Field::Extra{path,..} if path==&vec!["extra".to_owned()]))
            .unwrap()
            .supplied_revision,
        stamp
    );
    assert!(units(&changed, 0).iter().any(|u| matches!(
        u,
        Unit::Block {
            item: 0,
            local: 1,
            ..
        }
    )));
    assert!(
        units(&contribution(&memory, 1), 1).iter().any(|u| matches!(
            u,
            Unit::Block {
                item: 1,
                local: 0,
                ..
            }
        )),
        "later contribution is not prefix-reflowed"
    );
    assert!(
        !units(&contribution(&memory, 1), 1).iter().any(
            |u| matches!(u,Unit::Fact{fact,..} if matches!(fact.field,Field::FirstMessage(_)))
        )
    );
}

#[test]
fn unresolved_mapping_and_required_body_never_become_completed_empty_projection() {
    let mut memory = MemoryReadView::default();
    let mut a = record(
        "a",
        json!([{"type":"text"},{"type":"image","source":{"data":"opaque"}}]),
    );
    a["costUSD"] = json!(1);
    step(&mut memory, a);
    let c = contribution(&memory, 0);
    assert!(
        c.projection
            .obligations
            .iter()
            .any(|o| matches!(o, CountingObligation::MissingText { local: 0 }))
    );
    assert!(units(&c, 0).iter().any(|u| matches!(
        u,
        Unit::RecordHeader {
            counting_complete: false,
            ..
        }
    )));
    assert_eq!(
        response(&memory).output_header().unwrap().finish,
        FinishState::Unknown
    );
    let result = json!({"type":"user","uuid":"result","toolUseResult":{"persistedOutputPath":"body.txt"},"message":{"content":[{"type":"tool_result","content":"preview"}]}});
    step(&mut memory, result.clone());
    let cell = memory
        .response("s", &ResponseKey::user("result"))
        .unwrap()
        .unwrap();
    let c = memory.contribution(&cell.message.id, 0).unwrap().unwrap();
    assert!(
        units(&c, 0)
            .iter()
            .any(|u| matches!(u, Unit::RequiredBodyUnavailable { .. }))
    );
    memory.set_body(
        "body.txt",
        Some(BodyPayload {
            blob: polylogue_model::BlobHash::of(b"body"),
            length: 4,
        }),
    );
    step(&mut memory, result);
    let c = memory.contribution(&cell.message.id, 0).unwrap().unwrap();
    assert!(units(&c, 0).iter().any(|u| matches!(
        u,
        Unit::Evidence(polylogue_sources::claude_code::keyed::EvidenceUnit::Body { .. })
    )));
}

#[test]
fn completion_proof_is_explicit_and_record_witness_disappears_on_replacement() {
    let mut memory = MemoryReadView::default();
    let mut a = record("a", json!([]));
    a["isAbortedMidStream"] = json!(true);
    step(&mut memory, a);
    assert!(units(&contribution(&memory, 0), 0).iter().any(|u| matches!(
        u,
        Unit::RecordHeader {
            finished_witness: true,
            ..
        }
    )));
    step(
        &mut memory,
        json!({"type":"assistant","uuid":"a","isAbortedMidStream":false}),
    );
    assert!(
        !units(&contribution(&memory, 0), 0).iter().any(|u| matches!(
            u,
            Unit::RecordHeader {
                finished_witness: true,
                ..
            }
        ))
    );
    step(
        &mut memory,
        json!({"type":"assistant","uuid":"a","message":{"stop_reason":"end_turn"}}),
    );
    assert_eq!(
        response(&memory).output_header().unwrap().finish,
        FinishState::Finished
    );
}

#[test]
fn projection_is_restart_equivalent_and_one_group_work_does_not_depend_on_prefix() {
    let records = vec![
        record("a", json!([{"type":"text","text":"a"}])),
        record("b", json!([{"type":"text","text":"b"}])),
        json!({"type":"assistant","uuid":"a","message":{"content":[{"type":"thinking","thinking":"replacement"},{"type":"text","text":"a2"}],"usage":{"input_tokens":0}}}),
    ];
    let text: String = records.iter().cloned().map(line).collect();
    let empty = MemoryReadView::default();
    let whole = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &empty).unwrap(),
        text.as_bytes(),
        0,
    )
    .output
    .unwrap();
    let mut all = MemoryReadView::default();
    all.apply(whole.header, whole.delta);
    let mut restarted = MemoryReadView::default();
    for record in records {
        step(&mut restarted, record);
    }
    assert_eq!(response(&all), response(&restarted));
    for item in 0..2 {
        assert_eq!(
            units(&contribution(&all, item), item),
            units(&contribution(&restarted, item), item)
        );
    }
    let mut counts = Vec::new();
    for prefix in [1, 200] {
        let mut memory = MemoryReadView::default();
        for item in 0..prefix {
            step(
                &mut memory,
                record(
                    &format!("prefix-{item}"),
                    json!([{"type":"text","text":"p"}]),
                ),
            );
        }
        step(
            &mut memory,
            record("append", json!([{"type":"text","text":"x"}])),
        );
        let c = contribution(&memory, prefix);
        let output = units(&c, prefix);
        assert!(
            output
                .iter()
                .all(|u| !matches!(u,Unit::Block{local,..} if *local!=0))
        );
        counts.push(output.len());
    }
    assert_eq!(
        counts[0], counts[1],
        "projection admits only the selected contribution, not its prefix"
    );
}

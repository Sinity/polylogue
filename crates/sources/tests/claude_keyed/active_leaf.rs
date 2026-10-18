use super::*;
fn leaf(memory: &MemoryReadView) -> ActiveLeaf {
    memory.session("s").unwrap().unwrap().active_leaf.unwrap()
}
#[test]
fn markers_follow_source_order_across_steps_including_invalid_summary_and_reapplication() {
    let mut memory = MemoryReadView::default();
    let mut at = 0;
    for (kind, value, scoped) in [
        ("last-prompt", json!("first"), true),
        ("summary", json!("second"), true),
        ("summary", Value::Null, true),
        ("last-prompt", json!(false), true),
        ("summary", json!("restored"), true),
    ] {
        let bytes = line(json!({"type":kind,"sessionId":"s","leafUuid":value}));
        step(&mut memory, FileCtx::main("s"), bytes.as_bytes(), at);
        let got = leaf(&memory);
        assert_eq!(got.kind, kind);
        assert_eq!(
            got.value,
            polylogue_model::RetainedJson::parse(value.to_string().as_bytes()).unwrap()
        );
        assert_eq!(got.scoped, scoped);
        assert_eq!(got.record.start, at);
        at += bytes.len() as u64;
        // A decoded checkpoint carries the exact marker, not a remembered flag.
        let serialized = serde_json::to_vec(&memory.session("s").unwrap().unwrap()).unwrap();
        let restored: SessionCell = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(restored.active_leaf, Some(got));
    }
}
#[test]
fn explicit_bad_scope_never_borrows_the_declared_main_owner() {
    for claim in [json!("foreign"), Value::Null, json!(17), json!({})] {
        let mut memory = MemoryReadView::default();
        let first = line(json!({"type":"last-prompt","leafUuid":"good"}));
        step(&mut memory, FileCtx::main("s"), first.as_bytes(), 0);
        let later = line(json!({"type":"summary","leafUuid":"same","sessionId":claim}));
        step(
            &mut memory,
            FileCtx::main("s"),
            later.as_bytes(),
            first.len() as u64,
        );
        assert!(!leaf(&memory).scoped);
        assert_eq!(leaf(&memory).kind, "summary");
    }
}
#[test]
fn omitted_leaf_overwrites_prior_marker_and_partial_marker_does_not_parse() {
    let mut memory = MemoryReadView::default();
    let first = line(json!({"type":"last-prompt","leafUuid":"first"}));
    step(&mut memory, FileCtx::main("s"), first.as_bytes(), 0);
    let partial = b"{\"type\":\"summary\",\"leafUuid\":";
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &memory).unwrap(),
        partial,
        first.len() as u64,
    );
    assert_eq!(framed.parsed_upto, first.len() as u64);
    assert_eq!(leaf(&memory).kind, "last-prompt");
    let absent = line(json!({"type":"summary"}));
    step(
        &mut memory,
        FileCtx::main("s"),
        absent.as_bytes(),
        first.len() as u64,
    );
    assert!(leaf(&memory).value.is_null());
}

#[test]
fn point_parent_relation_retains_omitted_donor_and_replaces_supplied_correction_without_body_hydration()
 {
    let mut memory = MemoryReadView::default();
    let mut first = user("large");
    first["parentUuid"] = json!("before");
    first["message"]["content"] = json!("x".repeat(128 * 1024));
    let bytes = line(first);
    step(&mut memory, FileCtx::main("s"), bytes.as_bytes(), 0);
    let key = serde_json::to_string(&NativeId::unicode("large")).unwrap();
    let address = memory.record("s", &key).unwrap().unwrap();
    assert!(serde_json::to_vec(&address).unwrap().len() < 1024);
    let relation = address.relation.unwrap();
    assert_eq!(relation.parent.unwrap().value, RetainedJson::from("before"));
    let patch = line(json!({"type":"user","uuid":"large","sessionId":"s","parentUuid":"after"}));
    step(
        &mut memory,
        FileCtx::main("s"),
        patch.as_bytes(),
        bytes.len() as u64,
    );
    let next = line(json!({"type":"user","uuid":"large","sessionId":"s","cwd":"/synthetic"}));
    step(
        &mut memory,
        FileCtx::main("s"),
        next.as_bytes(),
        (bytes.len() + patch.len()) as u64,
    );
    let relation = memory.record("s", &key).unwrap().unwrap().relation.unwrap();
    let link = relation.parent.unwrap();
    assert_eq!(link.value, RetainedJson::from("after"));
    assert_eq!(link.donor.record.start, bytes.len() as u64);
    assert_eq!(relation.record.start, (bytes.len() + patch.len()) as u64);
}

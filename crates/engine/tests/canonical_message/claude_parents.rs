//! Retained non-Message relations are proved against the pinned SQLite snapshot.
use super::*;
fn message(uuid: &str, parent: Option<&str>) -> String {
    (json!({"type":"assistant","uuid":uuid,"sessionId":SID,"requestId":uuid,"message":{"id":uuid,"role":"assistant","content":[{"type":"text","text":uuid}]}}).as_object().map(|o|{let mut o=o.clone();if let Some(p)=parent{o.insert("parentUuid".into(),json!(p));} Value::Object(o)}).unwrap()).to_string()+"\n"
}
fn event(uuid: &str, parent: Value, kind: &str, extra: Value) -> String {
    let mut v = json!({"type":kind,"uuid":uuid,"sessionId":SID,"parentUuid":parent});
    v.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    v.to_string() + "\n"
}
fn materialize(
    raw: &str,
    index: usize,
) -> (World, SourceProjection, MessageView, Vec<PendingMessage>) {
    let (mut w, p) = make_world(raw.as_bytes(), raw.as_bytes());
    let req = request(&w, &p[index]);
    let h = begin(&mut w, &req);
    w = w.reopen();
    let s = w.complete(&h);
    let v = selected(&mut w, &s);
    (w, s, v, p)
}
#[test]
fn event_attachment_system_progress_chain_resolves_nearest_message_with_exact_proof_and_citations()
{
    let a = message("a", None);
    let x = event(
        "x",
        json!("a"),
        "attachment",
        json!({"attachment":{"type":"synthetic"},"data":{"opaque":null},"subtype":"opaque subtype"}),
    );
    let y = event("y", json!("x"), "system", json!({"subtype":"api_error"}));
    let z = event(
        "z",
        json!("y"),
        "progress",
        json!({"data":{"type":"mcp_progress"}}),
    );
    let b = message("b", Some("z"));
    let raw = a.clone() + &x + &y + &z + &b;
    let (_w, s, v, p) = materialize(&raw, 1);
    let (parent, supplied, origin) = value(&v, &Slot::Message(MessageField::Parent));
    assert!(supplied);
    assert_eq!(parent["id"], p[0].message.as_str());
    let proof = &origin["selection"]["transform"]["ParentViaRecordsV1"]["relations"];
    assert_eq!(
        proof
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["uuid"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["z", "y", "x"]
    );
    assert_eq!(
        origin["selection"]["inputs"][0]["record"]["start"],
        raw.len() - b.len()
    );
    let (_reordered, changed, _, _) = materialize(&(a.clone() + &x + &z + &y + &b), 1);
    assert_eq!(
        s.version_hash(),
        changed.version_hash(),
        "storage order of independent retained relation nodes is not Message version authority"
    );
}
#[test]
fn logical_compaction_parent_summary_and_late_record_order_are_connected() {
    let a = message("a", None);
    let boundary = event(
        "boundary",
        Value::Null,
        "system",
        json!({"subtype":"compact_boundary","logicalParentUuid":"a","compactMetadata":{"trigger":"manual"}}),
    );
    let summary=(json!({"type":"user","uuid":"summary","sessionId":SID,"isCompactSummary":true,"parentUuid":"boundary","message":{"content":"summary"}})).to_string()+"\n";
    let b = message("b", Some("summary"));
    let raw = b.clone() + &summary + &boundary + &a;
    let (_w, _, v, p) = materialize(&raw, 0);
    assert_eq!(
        value(&v, &Slot::Message(MessageField::Parent)).0["id"],
        p[1].message.as_str()
    );
    let root = event("root", Value::Null, "attachment", json!({}));
    let (_w, _, v, _) = materialize(&(root + &message("b", Some("root"))), 0);
    assert!(
        v.field(&Slot::Message(MessageField::Parent), 65536)
            .unwrap()
            .is_none()
    );
}
fn refuse(raw: &str, index: usize) {
    let (mut w, p) = make_world(raw.as_bytes(), raw.as_bytes());
    let req = request(&w, &p[index]);
    let h = begin(&mut w, &req);
    for _ in 0..4000 {
        match w.compute_with(&h, work_budget()) {
            Ok(a) => assert!(!matches!(
                m::apply(&mut w.writer, a).unwrap(),
                m::Applied::Source(_)
            )),
            Err(m::Error::Unsupported(_) | m::Error::UnsupportedField { .. }) => return,
            Err(e) => panic!("{e}"),
        }
    }
    panic!("unresolved parent certified")
}
#[test]
fn absent_wrong_type_cross_scope_cycles_and_compaction_without_logical_parent_stay_unavailable() {
    let b = message("b", Some("x"));
    refuse(&b, 0);
    refuse(&(event("x", json!(7), "attachment", json!({})) + &b), 0);
    refuse(
        &(event("x", json!("y"), "attachment", json!({}))
            + &event("y", json!("x"), "system", json!({"subtype":"api_error"}))
            + &b),
        0,
    );
    refuse(&message("b", Some("b")), 0);
    refuse(
        &(event(
            "x",
            Value::Null,
            "system",
            json!({"subtype":"compact_boundary"}),
        ) + &b),
        0,
    );
    refuse(
        &(event("x", json!("a"), "attachment", json!({}))
            + &event("x", json!("a"), "system", json!({"subtype":"api_error"}))
            + &message("a", None)
            + &b),
        1,
    );
}
#[test]
fn event_link_correction_and_omission_select_exact_supplier_and_change_relation_evidence() {
    let a = message("a", None);
    let c = message("c", None);
    let first = event("x", json!("a"), "attachment", json!({}));
    let correction = event("x", json!("c"), "attachment", json!({}));
    let omitted = (json!({"type":"attachment","uuid":"x","sessionId":SID,"attachment":null}))
        .to_string()
        + "\n";
    let b = message("b", Some("x"));
    let (_before, s_before, _, _) = materialize(&(a.clone() + &c + &first + &b), 2);
    let (_after, s_after, v, p) =
        materialize(&(a.clone() + &c + &first + &b + &correction + &omitted), 2);
    assert_ne!(s_before.version_hash(), s_after.version_hash());
    let (parent, _, origin) = value(&v, &Slot::Message(MessageField::Parent));
    assert_eq!(parent["id"], p[1].message.as_str());
    let node = &origin["selection"]["transform"]["ParentViaRecordsV1"]["relations"][0];
    assert_eq!(
        node["parent"]["donor"]["record"]["start"],
        a.len() + c.len() + first.len() + b.len()
    );
    let (_duplicate, s_duplicate, _, _) = materialize(
        &(a + &c + &first + &b + &correction + &omitted + &correction),
        2,
    );
    assert_eq!(s_after.version_hash(), s_duplicate.version_hash());
}
#[test]
fn parent_relation_work_refuses_small_point_and_scratch_allowances_without_installing() {
    let raw = message("a", None)
        + &event("x", json!("a"), "attachment", json!({}))
        + &message("b", Some("x"));
    let (mut w, p) = make_world(raw.as_bytes(), raw.as_bytes());
    let req = request(&w, &p[1]);
    let h = begin(&mut w, &req);
    let mut b = work_budget();
    b.read.cells = 3;
    assert!(matches!(
        w.compute_with(&h, b),
        Err(m::Error::Store(
            polylogue_store::Error::ParserAdmission { .. }
        ))
    ));
    let mut b = work_budget();
    b.metadata = 512;
    assert!(matches!(
        w.compute_with(&h, b),
        Err(m::Error::Store(
            polylogue_store::Error::ParserAdmission { .. }
        ))
    ));
    assert!(
        w.writer
            .current_source_projection(&w.head.token, &req.logical_key, 65536)
            .unwrap()
            .is_none()
    );
    w = w.reopen();
    let s = w.complete(&h);
    assert_eq!(
        selected(&mut w, &s)
            .field(&Slot::Message(MessageField::Parent), 65536)
            .unwrap()
            .is_some(),
        true
    );
}
#[test]
fn relation_value_or_supplier_must_match_the_original_record_before_certification() {
    let raw = message("a", None)
        + &message("c", None)
        + &event("x", json!("a"), "attachment", json!({}))
        + &message("b", Some("x"));
    for tamper in 0..3 {
        let (mut w, p) = make_world_with(raw.as_bytes(), raw.as_bytes(), |out| {
            let node = &mut out.delta.parent_nodes[0].2;
            if tamper == 0 {
                node.parent.as_mut().unwrap().value =
                    polylogue_model::RawJson::parse(br#""c""#).unwrap();
            } else if tamper == 1 {
                node.parent.as_mut().unwrap().donor.path =
                    vec![polylogue_sources::selection::PathPart::Key("wrong".into())];
            } else {
                node.parent.as_mut().unwrap().donor.record =
                    polylogue_sources::selection::RecordRange {
                        start: 0,
                        end: message("a", None).len() as u64 - 1,
                    };
            }
        });
        let req = request(&w, &p[2]);
        let h = begin(&mut w, &req);
        let mut refused = false;
        for _ in 0..4000 {
            match w.compute_with(&h, work_budget()) {
                Ok(a) => assert!(!matches!(
                    m::apply(&mut w.writer, a).unwrap(),
                    m::Applied::Source(_)
                )),
                Err(m::Error::Unsupported(
                    "Claude retained parent proof differs from original source",
                )) => {
                    refused = true;
                    break;
                }
                Err(e) => panic!("{e}"),
            }
        }
        assert!(refused);
    }
}
#[test]
fn a_long_valid_tree_keeps_direct_parent_point_read_count_independent_of_ancestry() {
    for count in [2, 128] {
        let mut raw = String::new();
        for n in 0..count {
            let parent = (n > 0).then(|| format!("neutral-{:03}", n - 1));
            raw += &message(&format!("neutral-{n:03}"), parent.as_deref());
        }
        let (mut w, p) = make_world(raw.as_bytes(), raw.as_bytes());
        let req = request(&w, &p[count - 1]);
        let h = begin(&mut w, &req);
        let mut budget = work_budget();
        budget.metadata = 8192;
        budget.read.cells = 2;
        assert!(matches!(
            w.compute_with(&h, budget),
            Err(m::Error::Store(polylogue_store::Error::ParserAdmission {
                resource: "Claude state point attempts",
                required: 3
            }))
        ));
        budget.read.cells = 3;
        let source = loop {
            let action = w.compute_with(&h, budget).unwrap();
            if let m::Applied::Source(source) = m::apply(&mut w.writer, action).unwrap() {
                break source;
            }
        };
        assert_eq!(
            value(
                &selected(&mut w, &source),
                &Slot::Message(MessageField::Parent)
            )
            .0["id"],
            p[count - 2].message.as_str()
        );
    }
}

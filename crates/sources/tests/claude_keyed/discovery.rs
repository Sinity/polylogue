use super::*;

#[test]
fn configured_claude_subagent_folder_never_substitutes_for_missing_or_conflicting_owner() {
    let context = FileCtx::Subagent {
        folder_owner: "folder-lineage-only".into(),
        agent: "a".into(),
        meta: None,
    };
    let mut missing = user("missing");
    missing.as_object_mut().unwrap().remove("sessionId");
    let mut actual = user("actual");
    actual["sessionId"] = json!("real-owner");
    let mut conflict = user("conflict");
    conflict["sessionId"] = json!("different-owner");
    let mut inherited = user("established");
    inherited.as_object_mut().unwrap().remove("sessionId");
    let records = [
        missing,
        json!({"type":"system","subtype":"compact_boundary"}),
        actual,
        conflict,
        inherited,
    ];
    let bytes: String = records.iter().cloned().map(line).collect();
    let mut whole = MemoryReadView::default();
    let (out, counts) = step(&mut whole, context.clone(), bytes.as_bytes(), 0);
    assert_eq!(counts.used, 2);
    assert_eq!(counts.unrecognized.len(), 3);
    assert_eq!(out.header.owner.as_deref(), Some("real-owner"));
    assert!(whole.session("folder-lineage-only/a").unwrap().is_none());
    assert!(whole.session("different-owner/a").unwrap().is_none());
    let session = whole.session("real-owner/a").unwrap().unwrap();
    assert_eq!(session.next_message_ordinal, 2);
    assert_eq!(out.links.len(), 1);
    assert_eq!(
        out.links[0].parent.as_str(),
        "claude-code:folder-lineage-only"
    );
    assert_eq!(out.links[0].child.as_str(), session.session.id.as_str());
    let mut incremental = MemoryReadView::default();
    let mut offset = 0;
    let mut used = 0;
    let mut unresolved = 0;
    for record in records {
        let text = line(record);
        let (_, count) = step(&mut incremental, context.clone(), text.as_bytes(), offset);
        offset += text.len() as u64;
        used += count.used;
        unresolved += count.unrecognized.len();
    }
    assert_eq!((used, unresolved), (2, 3));
    assert_eq!(incremental.session("real-owner/a").unwrap(), Some(session));
    assert!(
        incremental
            .session("folder-lineage-only/a")
            .unwrap()
            .is_none()
    );
}

#[test]
fn configured_claude_explicit_author_evidence_precedes_first_agent_fallback() {
    for (label, extra, text, expected) in [
        (
            "typed",
            json!({"promptSource":"typed"}),
            "Explicit human input",
            WrittenBy::Human,
        ),
        (
            "coordinator",
            json!({"promptSource":"typed","origin":{"kind":"coordinator"}}),
            "Parent model instruction",
            WrittenBy::Model,
        ),
        (
            "human_origin",
            json!({"promptSource":"system","origin":{"kind":"human"}}),
            "Explicit human origin",
            WrittenBy::Human,
        ),
        (
            "sdk",
            json!({"promptSource":"sdk"}),
            "SDK opening input",
            WrittenBy::Harness,
        ),
        (
            "meta",
            json!({"isMeta":true}),
            "Harness context",
            WrittenBy::Harness,
        ),
        (
            "wrapper",
            json!({}),
            "<system-reminder>Harness context</system-reminder>",
            WrittenBy::Harness,
        ),
        (
            "fallback",
            json!({}),
            "Parent instruction without explicit author fields",
            WrittenBy::Model,
        ),
    ] {
        let mut value =
            json!({"type":"user","uuid":"u","sessionId":"s","message":{"content":text}});
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let raw = line(value);
        let mut memory = MemoryReadView::default();
        let (_, counts) = step(
            &mut memory,
            FileCtx::Subagent {
                folder_owner: "folder-lineage".into(),
                agent: "a".into(),
                meta: None,
            },
            raw.as_bytes(),
            0,
        );
        assert_eq!(counts.unrecognized.len(), 0, "{label}");
        let response = memory
            .response("s/a", &ResponseKey::user("u"))
            .unwrap()
            .unwrap();
        assert_eq!(response.message.written_by, expected, "{label}");
        let recipe = response.author_selection.unwrap();
        let mut encoded = Vec::new();
        polylogue_sources::selection::write_value(
            &recipe,
            |range| -> Result<Vec<u8>, std::io::Error> {
                Ok(raw.as_bytes()[range.start as usize..range.end as usize].to_vec())
            },
            1 << 20,
            &mut encoded,
        )
        .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&encoded).unwrap(),
            serde_json::to_value(expected).unwrap(),
            "replayed author {label}"
        );
    }
}

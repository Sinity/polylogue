use super::*;
#[test]
fn cloud_tasks_keep_native_turns_metadata_and_unknown_times() {
    let bytes=br#"[{"id":"task_e_sample","title":"Repair example","archived":false,"turns":[{"id":"task_e_sample~usertrn_e_one","role":"user","custom_instructions":null,"input_items":[{"type":"message","role":"user","content":[{"content_type":"text","text":"Repair it."}]}]},{"id":"task_e_sample~assttrn_e_two","role":"assistant","previous_turn_id":"task_e_sample~usertrn_e_one","branch":"feature/example","external_pull_request_id":"None","turn_status":"completed","output_items":[{"content":[{"content_type":"text","text":"Repaired."},{"content_type":"repo_file_citation","path":"src/example.rs","line_range_start":4,"line_range_end":9}]}]}]}]"#;
    let step =
        crate::export::fixture::parse(&mut open(Path::new("codex.json")).unwrap(), bytes).unwrap();
    assert_eq!(step.output.sessions.len(), 1);
    let session = &step.output.sessions[0];
    assert_eq!(session.id.to_string(), "codex-cloud:task_e_sample");
    assert_eq!(session.git_branch.as_deref(), Some("feature/example"));
    assert!(session.started.is_none());
    let messages: &Vec<_> = &step.output.messages;
    assert_eq!(
        messages
            .iter()
            .map(|row| row.message.text())
            .collect::<Vec<_>>(),
        ["Repair it.", "Repaired."]
    );
    assert_eq!(
        messages[1].message.parent.as_ref(),
        Some(&messages[0].message.id)
    );
    assert!(messages.iter().all(|row| row.message.at.is_none()));
    assert_eq!(messages[0].message.prompt_kind, Some(PromptKind::Prose));
    assert!(
        !messages[1]
            .message
            .attrs
            .contains_key("external_pull_request_id")
    );
    assert!(
        matches!(&messages[1].message.blocks[1],Block::Other{type_name,data} if type_name=="repo_file_citation" && data["path"]=="src/example.rs" && data["line_range_end"]==9)
    );
    assert!(
        step.counts.unread_fields.is_empty(),
        "{:?}",
        step.counts.unread_fields
    );
}
#[test]
fn cloud_shape_rejects_local_rollouts_and_chatgpt_conversations() {
    assert!(!task_shape(
        &serde_json::json!({"id":"local","turns":[{"id":"m","role":"user"}]})
    ));
    assert!(!task_shape(
        &serde_json::json!({"id":"task_e_x","mapping":{},"turns":[{"id":"m","role":"user"}]})
    ));
    assert!(!task_shape(
        &serde_json::json!({"id":"task_e_x","turns":[]})
    ));
    assert!(!task_shape(
        &serde_json::json!({"id":"task_e_x","turns":[{"id":"m","role":"system"}]})
    ));
}

#[test]
fn a_cloud_browser_capture_and_codex_export_share_provider_task_and_turn_ids() {
    let task = serde_json::json!({"id":"task_e_shared","turns":[{"id":"task_e_shared~usertrn_e_one","role":"user","input_items":[{"content":[{"content_type":"text","text":"same request"}]}]}]});
    let exported = crate::export::fixture::parse(
        &mut open(Path::new("codex.json")).unwrap(),
        &serde_json::to_vec(&vec![&task]).unwrap(),
    )
    .unwrap();
    let capture = serde_json::json!({"polylogue_capture_kind":"browser_llm_session","schema_version":1,"session":{"provider":"codex-cloud","provider_session_id":"task_e_shared"},"raw_provider_payload":task});
    let captured = crate::export::fixture::parse(
        &mut crate::browser_capture::open(Path::new("capture.json")).unwrap(),
        &serde_json::to_vec(&capture).unwrap(),
    )
    .unwrap();
    assert_eq!(
        captured.output.sessions[0].id,
        exported.output.sessions[0].id
    );
    assert_eq!(
        captured.output.messages[0].message.id,
        exported.output.messages[0].message.id
    );
    assert_eq!(captured.output.messages[0].message.text(), "same request");
}

#[test]
fn native_capture_accounts_the_payload_once_and_leaves_unused_dom_turns_visible() {
    let capture = serde_json::json!({"polylogue_capture_kind":"browser_llm_session","schema_version":1,"provenance":{"captured_at":"2026-01-01T00:00:00Z","transport":"native"},"provider_meta":{"adapter":"cloud"},"session":{"provider":"codex-cloud","provider_session_id":"task_e_accounting","turns":[{"role":"user","text":"ignored DOM copy"}]},"raw_provider_payload":{"id":"task_e_accounting","turns":[{"id":"task_e_accounting~usertrn_e_one","role":"user","input_items":[{"content":[{"content_type":"text","text":"native request"}]}]}]}});
    let step = crate::export::fixture::parse(
        &mut crate::browser_capture::open(Path::new("capture.json")).unwrap(),
        &serde_json::to_vec(&capture).unwrap(),
    )
    .unwrap();
    assert_eq!(step.output.messages[0].message.text(), "native request");
    assert!(
        !step
            .counts
            .unread_fields
            .keys()
            .any(|key| key.starts_with("capture.raw_provider_payload.")),
        "{:?}",
        step.counts.unread_fields
    );
    assert!(
        step.counts
            .unread_fields
            .keys()
            .any(|key| key.contains("turns") && key.ends_with("text")),
        "{:?}",
        step.counts.unread_fields
    );
    assert_eq!(
        step.output.events.last().unwrap().data["provider_meta"]["adapter"],
        "cloud"
    );
}

#[test]
fn a_literal_branch_named_none_is_a_branch_not_a_missing_value() {
    let bytes=br#"[{"id":"task_e_branch","turns":[{"id":"task_e_branch~assttrn_e_one","role":"assistant","branch":"None","external_pull_request_id":"None","output_items":[]}]}]"#;
    let step =
        crate::export::fixture::parse(&mut open(Path::new("codex.json")).unwrap(), bytes).unwrap();
    assert_eq!(step.output.sessions[0].git_branch.as_deref(), Some("None"));
    assert!(
        !step.output.messages[0]
            .message
            .attrs
            .contains_key("external_pull_request_id")
    );
}

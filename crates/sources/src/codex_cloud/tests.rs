use super::*;
#[test]
fn cloud_tasks_keep_turns_metadata_and_unknown_times() {
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
fn capture_counts_payload_once_and_shows_dom_turns() {
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

#[test]
fn custom_instructions_become_session_instructions() {
    let bytes=br#"[{"id":"task_e_instr","turns":[{"id":"task_e_instr~usertrn_e_one","role":"user","custom_instructions":"Answer briefly.","input_items":[]}]}]"#;
    let step =
        crate::export::fixture::parse(&mut open(Path::new("codex.json")).unwrap(), bytes).unwrap();
    assert_eq!(
        step.output.sessions[0].instructions.as_deref(),
        Some("Answer briefly.")
    );
}

#[test]
fn an_empty_branch_falls_through_to_branch_name() {
    let bytes=br#"[{"id":"task_e_alt","turns":[{"id":"task_e_alt~assttrn_e_one","role":"assistant","branch":"","branch_name":"feature/alt","output_items":[]}]}]"#;
    let step =
        crate::export::fixture::parse(&mut open(Path::new("codex.json")).unwrap(), bytes).unwrap();
    assert_eq!(
        step.output.sessions[0].git_branch.as_deref(),
        Some("feature/alt")
    );
}

fn parse_task(turns: serde_json::Value, extra: serde_json::Value) -> crate::parse::Step {
    let mut task = serde_json::json!({"id":"task_e_meta","turns":turns});
    for (key, value) in extra.as_object().unwrap() {
        task[key] = value.clone();
    }
    crate::export::fixture::parse(
        &mut open(Path::new("codex.json")).unwrap(),
        &serde_json::to_vec(&vec![task]).unwrap(),
    )
    .unwrap()
}

#[test]
fn task_and_turn_metadata_are_kept_as_events_and_attributes() {
    let step = parse_task(
        serde_json::json!([
            {"id":"t1","role":"user","input_items":[{"content":[{"content_type":"text","text":"Go."}]}]},
            {"id":"t2","role":"assistant","previous_turn_id":"t1","external_pull_request_id":"pr_42","turn_status":"completed","pull_request_status":"open","output_items":[{"content":[{"content_type":"text","text":"Done."}]}]}
        ]),
        serde_json::json!({"archived":false}),
    );
    let messages = &step.output.messages;
    assert_eq!(
        messages[1].message.attrs["external_pull_request_id"],
        "pr_42"
    );
    assert_eq!(messages[1].message.attrs["turn_status"], "completed");
    let task: Vec<_> = step
        .output
        .events
        .iter()
        .filter(|e| e.kind == "codex_cloud_task")
        .collect();
    assert_eq!(task.len(), 1);
    assert_eq!(task[0].data["archived"], false);
    let turns: Vec<_> = step
        .output
        .events
        .iter()
        .filter(|e| e.kind == "codex_cloud_turn")
        .collect();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].after.as_ref(), Some(&messages[1].message.id));
    assert_eq!(turns[0].data["pull_request_status"], "open");
}

#[test]
fn only_the_none_null_and_empty_pull_request_spellings_mean_no_reference() {
    for spelling in ["None", "null", "", " NULL "] {
        let step = parse_task(
            serde_json::json!([{"id":"t1","role":"user","external_pull_request_id":spelling,"input_items":[{"content":[{"content_type":"text","text":"Go."}]}]}]),
            serde_json::json!({}),
        );
        assert!(
            !step.output.messages[0]
                .message
                .attrs
                .contains_key("external_pull_request_id"),
            "{spelling:?}"
        );
    }
    let step = parse_task(
        serde_json::json!([{"id":"t1","role":"user","external_pull_request_id":"pr_7","input_items":[{"content":[{"content_type":"text","text":"Go."}]}]}]),
        serde_json::json!({}),
    );
    assert_eq!(
        step.output.messages[0].message.attrs["external_pull_request_id"],
        "pr_7"
    );
}

#[test]
fn cloud_shape_needs_nonempty_task_and_turn_ids() {
    assert!(!task_shape(
        &serde_json::json!({"id":"","turns":[{"id":"m","role":"user"}]})
    ));
    assert!(!task_shape(
        &serde_json::json!({"id":"task_e_x","turns":[{"id":"","role":"user"}]})
    ));
    assert!(!task_shape(
        &serde_json::json!({"id":"task_e_x","turns":[{"id":"m","role":""}]})
    ));
    assert!(task_shape(
        &serde_json::json!({"id":"task_e_x","turns":[{"id":"m","role":"assistant"}]})
    ));
}

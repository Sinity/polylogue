use super::*;
use serde_json::json;

fn parse(value: Json) -> Step {
    let mut parser = open(Path::new("renamed.json")).unwrap();
    crate::export::fixture::parse(&mut parser, &serde_json::to_vec(&value).unwrap()).unwrap()
}

fn conversation() -> Json {
    json!({"conversation":{"title":"Synthetic account chat", "create_time":{"$date":{"$numberLong":"1712000000000"}}}, "responses":[
        {"response":{"sender":"human", "message":"# AGENTS.md\nExplain this synthetic document.", "create_time":{"$date":{"$numberLong":"1712000001000"}}}},
        {"response":{"sender":"assistant", "message":"Synthetic answer λ🙂", "create_time":"2024-04-01T19:33:22Z"}}
    ]})
}

#[test]
fn nested_flat_and_equivalent_declared_time_forms_keep_ids_and_supplied_content() {
    let original = conversation();
    let first = parse(json!({"conversations":[original.clone()]}));
    assert_eq!(first.output.messages.len(), 2);
    assert_eq!(first.output.messages[0].message.role, Role::User);
    assert_eq!(
        first.output.messages[0].message.prompt_kind,
        Some(PromptKind::Prose)
    );
    assert_eq!(
        first.output.messages[0].message.text(),
        "# AGENTS.md\nExplain this synthetic document."
    );
    assert_eq!(
        first.output.messages[0].message.at.unwrap().to_string(),
        "2024-04-01T19:33:21Z"
    );
    assert!(
        first.output.sessions[0]
            .id
            .native()
            .as_str()
            .starts_with('~')
    );
    for time in [
        json!(1712000001),
        json!({"$date":1712000001000i64}),
        json!("2024-04-01T21:33:21+02:00"),
    ] {
        let mut flat = original.clone();
        for entry in flat["responses"].as_array_mut().unwrap() {
            *entry = entry["response"].take();
        }
        flat["responses"][0]["create_time"] = time;
        let next = parse(json!({"conversations":[flat]}));
        assert_eq!(next.output.sessions[0].id, first.output.sessions[0].id);
        assert_eq!(
            next.output
                .messages
                .iter()
                .map(|row| &row.message)
                .collect::<Vec<_>>(),
            first
                .output
                .messages
                .iter()
                .map(|row| &row.message)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn same_opening_does_not_merge_different_idless_conversations_and_order_is_not_identity() {
    let first = conversation();
    let mut different = first.clone();
    different["responses"][1]["response"]["message"] =
        json!("Another independently exported answer");
    let parsed = parse(json!({"conversations":[first.clone(), different.clone()]}));
    assert_ne!(parsed.output.sessions[0].id, parsed.output.sessions[1].id);
    let mut reordered = first;
    reordered["responses"].as_array_mut().unwrap().reverse();
    let again = parse(json!({"conversations":[different, reordered]}));
    assert_eq!(again.output.sessions[1].id, parsed.output.sessions[0].id);
    assert_eq!(
        again.output.messages[2].message.id,
        parsed.output.messages[1].message.id
    );
    assert_eq!(
        again.output.messages[2].message.text(),
        "Synthetic answer λ🙂"
    );
}

#[test]
fn equal_idless_turns_keep_occurrences_and_report_the_correspondence_limit() {
    let mut value = conversation();
    let repeated = value["responses"][1].clone();
    value["responses"].as_array_mut().unwrap().push(repeated);
    let parsed = parse(json!({"conversations":[value]}));
    assert_eq!(parsed.output.messages.len(), 3);
    assert_ne!(
        parsed.output.messages[1].message.id,
        parsed.output.messages[2].message.id
    );
    assert_eq!(
        parsed.output.messages[1].message.text(),
        parsed.output.messages[2].message.text()
    );
    assert!(
        parsed.output.messages[1]
            .message
            .id
            .native()
            .as_str()
            .contains(":~")
    );
    assert_eq!(parsed.output.events.len(), 1);
    assert!(
        parsed.output.events[0]
            .data
            .to_string()
            .contains("physical correspondence is ambiguous")
    );
}

#[test]
fn native_ids_blank_text_and_only_supplied_models_survive() {
    let value = json!({"conversations":[{"conversation":{"conversationId":"native-chat", "title":"", "model":"not-inherited"}, "responses":[
        {"sender":"user", "responseId":"user-turn", "message":""},
        {"sender":"assistant", "responseId":"model-turn", "parentResponseId":"user-turn", "message":"answer"},
        {"sender":"grok", "responseId":"empty-model", "message":"answer", "model":""}
    ]}]});
    let parsed = parse(value);
    assert_eq!(parsed.output.sessions[0].id.to_string(), "grok:native-chat");
    assert!(
        parsed.output.sessions[0].title.is_none(),
        "empty title does not win the preferred-title selection"
    );
    assert_eq!(parsed.output.messages.len(), 3);
    assert_eq!(
        parsed.output.messages[0].message.id.to_string(),
        "grok:user-turn"
    );
    assert_eq!(
        parsed.output.messages[0].message.blocks,
        vec![Block::Text { text: "".into() }]
    );
    assert_eq!(
        parsed.output.messages[1]
            .message
            .parent
            .as_ref()
            .unwrap()
            .to_string(),
        "grok:user-turn"
    );
    assert!(parsed.output.messages[1].message.model.is_none());
    assert_eq!(parsed.output.messages[2].message.model.as_deref(), Some(""));
    assert!(
        parsed
            .output
            .messages
            .iter()
            .all(|row| row.message.at.is_none())
    );
}

#[test]
fn malformed_and_unknown_fields_refuse_interpretation_without_inventing_facts() {
    let parsed = parse(json!({"conversations":[
        {"not":"a conversation"},
        {"conversation":{"id":"diagnostics"}, "responses":[
            null, {"sender":"not-a-known-author", "message":"kept in original material"},
            {"sender":"human"}, {"sender":"human", "message":null},
            {"sender":"assistant", "message":"valid", "create_time":1712000001000i64, "model":{}},
            {"sender":"assistant", "message":"valid", "create_time":{"$date":{"$numberLong":"bad"}}}
        ]}
    ]}));
    assert_eq!(parsed.output.sessions.len(), 1);
    assert_eq!(parsed.output.messages.len(), 2);
    assert!(
        parsed
            .output
            .messages
            .iter()
            .all(|row| row.message.at.is_none() && row.message.model.is_none())
    );
    assert!(parsed.accounting.refused.values().sum::<u64>() >= 3);
    assert_eq!(parsed.accounting.unknown.values().sum::<u64>(), 1);
    assert!(
        parsed.output.events[0]
            .data
            .to_string()
            .contains("no time was inferred")
    );
}

#[test]
fn detection_is_content_based_across_filenames_and_rejects_cli_history() {
    let bytes = serde_json::to_vec(&json!({"conversations":[conversation()]})).unwrap();
    let head = &bytes[..bytes.len().min(64)];
    assert!(recognize(Path::new("prod-grok-backend.json"), head));
    assert!(recognize(Path::new("renamed.json"), head));
    let metadata_first = br#"{"account":"uninterpreted metadata before the conversation signature"#;
    assert!(recognize(
        Path::new("prod-grok-backend.json"),
        metadata_first
    ));
    assert!(!recognize(Path::new("renamed.json"), metadata_first));
    assert!(!recognize(
        Path::new("chat_history.jsonl"),
        br#"{"type":"user","content":"hello"}"#
    ));
    let bad = parse(json!({"conversations":"not an array"}));
    assert!(!bad.accounting.refused.is_empty());
    assert!(bad.output.sessions.is_empty());
}

#[test]
fn conflicting_native_ids_and_times_do_not_choose_an_interpretation() {
    let parsed = parse(json!({"conversations":[
        {"conversation":{"id":"first", "conversationId":"second"}, "responses":[]},
        {"conversation":{"id":"valid"}, "responses":[
            {"sender":"human", "id":"a", "responseId":"b", "message":"unresolved identity"},
            {"sender":"assistant", "message":"unresolved time", "create_time":1712000001,
             "createTime":"2024-04-01T19:33:22Z"}
        ]}
    ]}));
    assert_eq!(parsed.output.sessions.len(), 1);
    assert_eq!(parsed.output.messages.len(), 1);
    assert!(parsed.output.messages[0].message.at.is_none());
    assert!(
        parsed.output.events[0]
            .data
            .to_string()
            .contains("conflicting Grok creation times")
    );
    assert_eq!(parsed.accounting.refused.values().sum::<u64>(), 2);
}

#[test]
fn identity_sorts_object_members_but_keeps_unknown_values_and_null_presence() {
    let a: Json = serde_json::from_str(r#"{"conversation":{"title":"same","extension":{"z":1,"a":2}},"responses":[{"sender":"human","message":"question"}]}"#).unwrap();
    let b: Json = serde_json::from_str(r#"{"responses":[{"message":"question","sender":"human"}],"conversation":{"extension":{"a":2,"z":1},"title":"same"}}"#).unwrap();
    let first = parse(json!({"conversations":[a.clone()]}));
    let same = parse(json!({"conversations":[b]}));
    assert_eq!(first.output.sessions[0].id, same.output.sessions[0].id);
    let mut changed = a;
    changed["responses"][0]["create_time"] = Json::Null;
    let different = parse(json!({"conversations":[changed.clone()]}));
    assert_ne!(first.output.sessions[0].id, different.output.sessions[0].id);
    changed["conversation"]["extension"]["z"] = json!(3);
    let other = parse(json!({"conversations":[changed]}));
    assert_ne!(different.output.sessions[0].id, other.output.sessions[0].id);
}

#[test]
fn placeholder_title_does_not_replace_a_preferred_title() {
    let mut value = conversation();
    value["conversation"]["title"] = json!("New chat");
    let parsed = parse(json!({"conversations":[value]}));
    assert!(parsed.output.sessions[0].title.is_none());
}

#[test]
fn unknown_time_object_fields_stay_uninterpreted_identity_evidence() {
    let mut first = conversation();
    first["responses"][0]["response"]["create_time"] =
        json!({"$date":1712000001000i64,"unknown":"first"});
    let a = parse(json!({"conversations":[first.clone()]}));
    first["responses"][0]["response"]["create_time"]["unknown"] = json!("second");
    let b = parse(json!({"conversations":[first]}));
    assert_ne!(a.output.sessions[0].id, b.output.sessions[0].id);
    assert!(a.output.messages[0].message.at.is_none());
    assert!(
        a.output.events[0]
            .data
            .to_string()
            .contains("unrecognized Grok creation time")
    );
}

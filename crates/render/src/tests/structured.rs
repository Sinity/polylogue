use super::*;
use polylogue_model::export::{Document, SessionExport};
use serde_json::json;

#[test]
fn structured_read_keeps_message_metadata_and_original_tool_fields() {
    let mut messages = conversation();
    messages[1].production.effort = Some("high".into());
    if let Block::Thinking { signature, .. } = &mut messages[1].blocks[0] {
        *signature = Some("synthetic signature".into());
    }
    messages[1].duration_ms = Some(0);
    messages[1].blocks.push(Block::Other {
        type_name: "provider-extension".into(),
        data: json!({"empty": "", "nested": [0, false, null]}),
    });
    let session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    let value = crate::session(&session, None, messages.clone(), View::Full, Format::Json).unwrap();
    assert_eq!(value["messages"], serde_json::to_value(&messages).unwrap());
    assert_eq!(value["session"], serde_json::to_value(&session).unwrap());
    let actions = crate::session(
        &session,
        None,
        messages.clone(),
        View::Actions,
        Format::Json,
    )
    .unwrap();
    assert_eq!(actions["messages"][0]["production"]["effort"], "high");
    assert_eq!(actions["messages"][0]["duration_ms"], 0);
    assert!(actions["messages"][0]["blocks"][0]["input"].is_object());
    assert_eq!(actions["messages"][0]["blocks"][0]["type"], "tool_call");
}

#[test]
fn structured_export_uses_the_whole_importable_document_without_reinterpreting_blocks() {
    let mut session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    session.repo = Some("https://example.invalid/synthetic/project".into());
    session.mode = Some("bg".into());
    let doc = Document::new(vec![SessionExport {
        session,
        messages: conversation(),
        events: vec![serde_json::from_value(json!({"session": "claude-code:s", "kind": "synthetic", "data": {"empty": "", "nested": [false, null]}})).unwrap()],
        links: vec![serde_json::from_value(json!({"kind": "spawn", "parent": "claude-code:parent", "child": "claude-code:s", "evidence": "synthetic marker"})).unwrap()],
    }]);
    let value = crate::export(&doc, Format::Json).unwrap();
    assert_eq!(value, serde_json::to_value(&doc).unwrap());
    assert_eq!(Document::from_json(&value.to_string()).unwrap(), doc);
}

#[test]
fn full_json_keeps_a_genuinely_empty_message_and_empty_text_block() {
    let empty = message("empty", Role::User, Some(PromptKind::Prose), vec![]);
    let blank = message(
        "blank",
        Role::User,
        Some(PromptKind::Prose),
        vec![Block::Text {
            text: String::new(),
        }],
    );
    let session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    let full = crate::session(
        &session,
        None,
        vec![empty, blank.clone()],
        View::Full,
        Format::Json,
    )
    .unwrap();
    assert_eq!(full["messages"].as_array().unwrap().len(), 2);
    assert_eq!(full["messages"][0]["blocks"], json!([]));
    let prompts = crate::session(&session, None, vec![blank], View::Prompts, Format::Json).unwrap();
    assert_eq!(prompts["messages"][0]["blocks"][0]["text"], "");
}

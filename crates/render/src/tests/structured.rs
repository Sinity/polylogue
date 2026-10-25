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
    let value = crate::session(
        &session,
        None,
        messages.clone(),
        View::Full,
        Format::Json,
        &[],
    )
    .unwrap();
    let mut expected = serde_json::to_value(&messages).unwrap();
    crate::mask_json(&mut expected);
    assert_eq!(value["messages"], expected);
    assert_eq!(value["session"], serde_json::to_value(&session).unwrap());
    let actions = crate::session(
        &session,
        None,
        messages.clone(),
        View::Actions,
        Format::Json,
        &[],
    )
    .unwrap();
    assert_eq!(actions["messages"][0]["production"]["effort"], "high");
    assert_eq!(actions["messages"][0]["duration_ms"], 0);
    assert!(actions["messages"][0]["blocks"][0]["input"].is_object());
    assert_eq!(actions["messages"][0]["blocks"][0]["type"], "tool_call");
}

#[test]
fn structured_export_uses_the_whole_importable_document_without_rebuilding_blocks() {
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
    let value = crate::export(doc.clone(), Format::Json).unwrap();
    let mut expected = serde_json::to_value(&doc).unwrap();
    crate::mask_json(&mut expected);
    assert_eq!(value, expected);
    assert_eq!(
        Document::from_json(&value.to_string()).unwrap(),
        serde_json::from_value::<Document>(expected).unwrap()
    );
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
        &[],
    )
    .unwrap();
    assert_eq!(full["messages"].as_array().unwrap().len(), 2);
    assert_eq!(full["messages"][0]["blocks"], json!([]));
    let prompts = crate::session(
        &session,
        None,
        vec![blank],
        View::Prompts,
        Format::Json,
        &[],
    )
    .unwrap();
    assert_eq!(prompts["messages"][0]["blocks"][0]["text"], "");
}

#[test]
fn explicit_harness_authorship_keeps_the_recorded_assistant_role() {
    let message = super::message(
        "harness",
        Role::Assistant,
        Some(PromptKind::Harness),
        vec![Block::Text {
            text: "synthetic harness context".into(),
        }],
    );
    assert!(message.model.is_none());
    let session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        let rendered = crate::session(
            &session,
            None,
            vec![message.clone()],
            View::Full,
            format,
            &[],
        )
        .unwrap();
        let text = rendered["content"].as_str().unwrap();
        assert!(text.contains("harness"), "{text}");
        assert!(!text.contains("assistant"), "{text}");
    }
    let json =
        crate::session(&session, None, vec![message], View::Full, Format::Json, &[]).unwrap();
    assert_eq!(json["messages"][0]["role"], "assistant");
    assert_eq!(json["messages"][0]["prompt_kind"], "harness");
    assert!(json["messages"][0]["model"].is_null());
}

#[test]
fn incomplete_parents_notices_are_visible_and_html_escaped() {
    let session = Session::new(
        SessionId::parse("claude-code:s").unwrap(),
        SessionKind::Main,
    );
    let notices = ["Parents incomplete: parent <unknown> & its anchor are unavailable".to_owned()];
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        let rendered =
            crate::session(&session, None, vec![], View::Prompts, format, &notices).unwrap();
        let text = rendered["content"].as_str().unwrap();
        assert!(text.contains("Parents incomplete"));
        if format == Format::Html {
            assert!(text.contains("&lt;unknown&gt; &amp;"));
            assert!(!text.contains("<unknown>"));
        }
    }
}

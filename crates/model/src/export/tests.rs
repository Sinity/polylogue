use super::Document;
use super::*;
use crate::*;

fn sample() -> Document {
    let sid = SessionId::parse("claude-code:s1").unwrap();
    let mut session = Session::new(sid.clone(), SessionKind::Main);
    session.offer_title("Fix the build", TitleSource::Custom);
    let message = Message {
        id: MessageId::parse("claude-code:m1").unwrap(),
        session: sid.clone(),
        parent: None,
        role: Role::User,
        at: Timestamp::from_unix_ms(1_700_000_000_000),
        model: None,
        prompt_kind: Some(PromptKind::Prose),
        blocks: vec![Block::Text {
            text: "hello: 12345678901234567890".into(),
        }],
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
    };
    Document::new(vec![SessionExport {
        session,
        messages: vec![message],
        events: vec![],
        links: vec![],
    }])
}

#[test]
fn round_trips() {
    let doc = sample();
    let text = doc.to_json().unwrap();
    assert_eq!(Document::from_json(&text).unwrap(), doc);
}

#[test]
fn refuses_newer_versions_and_other_formats() {
    let text = r#"{"format":"polylogue-export","version":99,"sessions":[]}"#;
    assert!(matches!(
        Document::from_json(text),
        Err(ExportError::TooNew { found: 99 })
    ));
    let text = r#"{"format":"other","version":1,"sessions":[]}"#;
    assert!(matches!(
        Document::from_json(text),
        Err(ExportError::NotAnExport(_))
    ));
}

#[test]
fn large_numbers_in_opaque_json_are_lossless() {
    let v: Json = serde_json::from_str(r#"{"n":123456789012345678901234567890}"#).unwrap();
    assert_eq!(
        serde_json::to_string(&v).unwrap(),
        r#"{"n":123456789012345678901234567890}"#
    );
}

#[test]
fn document_exports_require_wire_two_and_keep_empty_text_and_unknown_authorship() {
    let mut document = crate::Document::new(
        crate::DocumentId::parse("claude-ai:project:a:doc:b").unwrap(),
        crate::DocumentPurpose::Instruction,
    );
    document.blocks.push(crate::Block::Text {
        text: String::new(),
    });
    let export = Document::new(Vec::new()).with_documents(vec![document.clone()]);
    assert_eq!(export.version, 2);
    assert_eq!(
        Document::from_json(&export.to_json().unwrap())
            .unwrap()
            .documents,
        [document]
    );
    let mut old = export;
    old.version = 1;
    assert!(matches!(
        Document::from_json(&old.to_json().unwrap()),
        Err(ExportError::DocumentsNeedVersion2)
    ));
    assert_eq!(sample().version, 1);
}

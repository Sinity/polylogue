use super::*;
use crate::parse::run_document;
use polylogue_model::export::SessionExport;
use polylogue_model::*;

fn doc() -> Document {
    let sid = SessionId::parse("claude-code:s1").unwrap();
    let mut session = Session::new(sid.clone(), SessionKind::Main);
    session.offer_title("Fix the build", TitleSource::Custom);
    let message = |id: &str, text: &str, role| Message {
        id: MessageId::parse(id).unwrap(),
        session: sid.clone(),
        parent: None,
        role,
        at: Timestamp::from_unix_ms(1_700_000_000_000),
        model: None,
        prompt_kind: None,
        blocks: vec![Block::Text { text: text.into() }],
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
    };
    Document::new(vec![SessionExport {
        session,
        messages: vec![
            message("claude-code:m1", "hello", Role::User),
            message("claude-code:m2", "hi", Role::Assistant),
        ],
        events: vec![Event {
            session: sid.clone(),
            kind: "compaction".into(),
            at: None,
            after: Some(MessageId::parse("claude-code:m1").unwrap()),
            range: Some((10, 20)),
            data: serde_json::json!({"x": 1}),
        }],
        links: vec![],
    }])
}

#[test]
fn recognizes_only_its_own_format() {
    let text = doc().to_json().unwrap();
    assert!(recognize(Path::new("x.json"), text.as_bytes()));
    assert!(!recognize(Path::new("x.json"), br#"{"format": "other"}"#));
    assert!(!recognize(Path::new("x.json"), br#"[{"uuid": "1"}]"#));
}

#[test]
fn reads_back_the_same_ids_in_document_order() {
    let text = doc().to_json().unwrap();
    let step = run_document(&Export, text.as_bytes()).unwrap();
    let ids: Vec<String> = step
        .output
        .messages
        .iter()
        .map(|m| m.message.id.to_string())
        .collect();
    assert_eq!(ids, ["claude-code:m1", "claude-code:m2"]);
    assert!(step.output.messages[0].offset < step.output.messages[1].offset);
    assert_eq!(step.output.sessions[0].id.to_string(), "claude-code:s1");
    assert_eq!(step.output.events[0].range, None);
    assert_eq!(step.accounting.framed, 1);
}

#[test]
fn refuses_a_newer_version_as_a_record() {
    let text = r#"{"format":"polylogue-export","version":99,"sessions":[]}"#;
    let step = run_document(&Export, text.as_bytes()).unwrap();
    assert!(step.output.sessions.is_empty());
    assert_eq!(step.accounting.framed, 1);
}

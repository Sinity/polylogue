use super::*;
use crate::parse::Step;
use polylogue_model::export::Document;
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

fn steps(bytes: &[u8]) -> Vec<Step> {
    struct NoOwners;
    impl ExportState for NoOwners {
        fn claim(&mut self, _: &str, _: &SessionId) -> Result<Option<SessionId>, String> {
            panic!("Polylogue export must preserve its authored message ownership")
        }
        fn last_message(
            &mut self,
            _: &SessionId,
        ) -> Result<Option<polylogue_model::MessageId>, String> {
            panic!("Polylogue export must preserve its authored parent links")
        }
        fn set_last_message(
            &mut self,
            _: &SessionId,
            _: &polylogue_model::MessageId,
        ) -> Result<(), String> {
            panic!("Polylogue export must preserve its authored parent links")
        }
    }
    let mut parser = Export::default();
    let mut items = Vec::new();
    parser
        .scan(
            "synthetic-input",
            &mut std::io::Cursor::new(bytes),
            &mut |range: std::ops::Range<u64>, _: Vec<u8>, _: Option<&str>, raw: &[u8]| {
                items.push((range, String::from_utf8(raw.to_vec()).unwrap()));
                Ok(())
            },
        )
        .unwrap();
    let mut steps = Vec::new();
    if let Some(step) = parser.begin(bytes.len() as u64, None).unwrap() {
        steps.push(step);
    }
    for (range, raw) in items {
        steps.push(
            crate::export::parse_item(&mut parser, raw.as_bytes(), range, &mut NoOwners).unwrap(),
        );
    }
    steps
}

#[test]
fn reads_back_the_same_ids_with_exact_export_session_spans() {
    let doc = doc();
    let text = doc.to_json().unwrap();
    let steps = steps(text.as_bytes());
    let step = &steps[0];
    assert_eq!(step.output.sessions, [doc.sessions[0].session.clone()]);
    assert_eq!(
        step.output
            .messages
            .iter()
            .map(|m| m.message.clone())
            .collect::<Vec<_>>(),
        doc.sessions[0].messages
    );
    let start = step.output.messages[0].offset as usize;
    let end = step.output.messages[0].end as usize;
    assert_eq!(
        serde_json::from_slice::<SessionExport>(&text.as_bytes()[start..end]).unwrap(),
        doc.sessions[0]
    );
    assert_eq!(step.output.messages[1].offset, start as u64);
    assert_eq!(
        step.output.events[0].range,
        Some((start as u64, end as u64))
    );
    assert_eq!(step.output.events[0].data, doc.sessions[0].events[0].data);
    assert_eq!(step.accounting.framed, 1);
}

#[test]
fn refuses_newer_headers_and_late_invalid_sessions_without_publishing_a_prefix() {
    let good = serde_json::to_string(&doc().sessions[0]).unwrap();
    for text in [
        format!("{{\"format\":\"polylogue-export\",\"sessions\":[{good}],\"version\":99}}"),
        format!(
            "{{\"format\":\"polylogue-export\",\"sessions\":[{good},{{\"session\":false}}],\"version\":1}}"
        ),
    ] {
        let steps = steps(text.as_bytes());
        assert!(
            steps
                .iter()
                .all(|s| s.output.sessions.is_empty() && s.output.messages.is_empty())
        );
        assert_eq!(steps.iter().map(|s| s.accounting.framed).sum::<u64>(), 1);
        assert_eq!(
            steps
                .iter()
                .map(|s| s.accounting.refused.values().sum::<u64>())
                .sum::<u64>(),
            1
        );
    }
}

#[test]
fn many_sessions_scan_individually_and_version_after_them_still_applies() {
    let one = serde_json::to_string(&doc().sessions[0]).unwrap();
    let bytes = format!(
        "{{\"format\":\"polylogue-export\",\"sessions\":[{}],\"version\":1}}",
        vec![one.as_str(); 100].join(",")
    );
    let mut parser = Export::default();
    let mut count = 0;
    let stats = parser
        .scan(
            "synthetic-input",
            &mut std::io::Cursor::new(bytes.as_bytes()),
            &mut |range: std::ops::Range<u64>, _: Vec<u8>, _: Option<&str>, raw: &[u8]| {
                count += 1;
                assert_eq!(
                    &bytes.as_bytes()[range.start as usize..range.end as usize],
                    raw
                );
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(count, 100);
    assert_eq!(stats.elements, 100);
    assert!(stats.peak_element_bytes < one.len() * 3);
    assert!(parser.begin(bytes.len() as u64, None).unwrap().is_none());
}

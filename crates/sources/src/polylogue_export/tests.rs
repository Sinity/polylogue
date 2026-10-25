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
            session: Some(sid.clone()),
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
    let public = serde_json::to_vec(&serde_json::to_value(doc()).unwrap()).unwrap();
    assert!(recognize(Path::new("x.json"), &public));
    assert!(recognize(
        Path::new("x.json"),
        br#"{"events":[],"format":"polylogue-export","version":"#
    ));
    assert!(!recognize(
        Path::new("x.json"),
        br#"{"events":[{"format":"polylogue-export"}],"format":"other"}"#
    ));
    assert!(!recognize(Path::new("x.json"), br#"{"format": "other"}"#));
    assert!(!recognize(Path::new("x.json"), br#"[{"uuid": "1"}]"#));
}

fn steps(bytes: &[u8]) -> Vec<Step> {
    struct NoOwners;
    impl ExportState for NoOwners {
        fn record_first_session(
            &mut self,
            _: &str,
            _: &SessionId,
        ) -> Result<Option<SessionId>, String> {
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
    assert_eq!(step.counts.records, 1);
}

#[test]
fn refuses_newer_headers_and_late_invalid_sessions_without_emitting_a_prefix() {
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
        assert_eq!(steps.iter().map(|s| s.counts.records).sum::<u64>(), 1);
        assert_eq!(
            steps
                .iter()
                .map(|s| s.counts.unreadable.values().sum::<u64>())
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

#[test]
fn typed_export_collections_refuse_null_instead_of_becoming_empty_arrays() {
    for raw in [
        r#"{"format":"polylogue-export","version":2,"sessions":null,"documents":[]}"#,
        r#"{"format":"polylogue-export","version":2,"sessions":[],"documents":null}"#,
    ] {
        let steps = steps(raw.as_bytes());
        assert!(
            steps
                .iter()
                .all(|s| s.output.documents.is_empty() && s.output.sessions.is_empty())
        );
        assert!(steps.iter().any(|s| !s.counts.unreadable.is_empty()));
    }
}

#[test]
fn version3_events_rebind_only_to_the_local_export_record_and_keep_sender_claims_in_stored_bytes() {
    let native = serde_json::json!({"approved":false,"value":null,"text":" exact λ "});
    let fact = polylogue_model::export::EventExport {
        id: "sender-observation".into(),
        input: "sender-input".into(),
        source: "hermes".into(),
        source_chain: 7,
        source_start: Some(200),
        source_end: Some(300),
        event: Event {
            session: None,
            kind: "hermes_observer_mark".into(),
            at: None,
            after: None,
            range: Some((200, 300)),
            data: native.clone(),
        },
    };
    let document = Document::new(vec![]).with_events(vec![fact.clone(), fact]);
    let raw = document.to_json().unwrap();
    let parsed = steps(raw.as_bytes());
    let facts: Vec<_> = parsed.iter().flat_map(|s| &s.output.events).collect();
    assert_eq!(facts.len(), 2);
    assert_ne!(facts[0].range, facts[1].range);
    for fact in facts {
        assert!(fact.session.is_none());
        assert_eq!(fact.data, native);
        let (start, end) = fact.range.unwrap();
        let record: serde_json::Value =
            serde_json::from_str(&raw[start as usize..end as usize]).unwrap();
        assert_eq!(record["input"], "sender-input");
        assert_eq!(record["source_chain"], 7);
        assert_eq!(record["id"], "sender-observation");
        assert_ne!((start, end), (200, 300));
    }
    assert!(
        parsed
            .iter()
            .all(|s| s.output.sessions.is_empty() && s.output.messages.is_empty())
    );
    let mut incompatible: serde_json::Value = serde_json::from_str(&raw).unwrap();
    incompatible["version"] = serde_json::json!(2);
    let refused = steps(serde_json::to_string(&incompatible).unwrap().as_bytes());
    assert!(
        refused
            .iter()
            .all(|s| s.output.events.is_empty() && s.output.sessions.is_empty())
    );
    assert!(refused.iter().any(|s| !s.counts.unreadable.is_empty()));
}

#[test]
fn discriminator_after_a_large_value_is_found_without_claiming_foreign_objects() {
    let large = "neutral λ ".repeat(20_000);
    let leading = serde_json::to_string(&large).unwrap();
    let own = format!("{{\"documents\":[{{\"text\":{leading}}}],\"format\":\"polylogue-export\"}}");
    assert!(!recognize(Path::new("x.json"), &own.as_bytes()[..64]));
    assert!(recognize_reader(std::io::Cursor::new(own)).unwrap());
    for foreign in [
        format!("{{\"events\":[{{\"text\":{leading}}}]}}"),
        format!(
            "{{\"documents\":[{{\"text\":{leading},\"format\":\"polylogue-export\"}}],\"format\":\"foreign\"}}"
        ),
    ] {
        assert!(!recognize_reader(std::io::Cursor::new(foreign)).unwrap());
    }
    struct Failed;
    impl std::io::Read for Failed {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("synthetic storage failure"))
        }
    }
    assert!(recognize_reader(Failed).is_err());
}

use super::*;

struct EmptyDocument;

impl DocumentParser for EmptyDocument {
    fn document(&self, _: &[u8], _: &mut Output, _: &mut Records) {}
}

#[test]
fn document_parser_requires_the_whole_input() {
    let parser = Parse::Document(Box::new(EmptyDocument));
    assert!(parser.is_document());
    assert!(matches!(
        parser.step(None, b"{}", 12),
        Err(ParseError::NotWhole(12))
    ));
    let (step, parser_state) = parser.step(None, b"{}", 0).unwrap();
    assert_eq!(step.consumed, 2);
    assert_eq!(parser_state, Json::Null);
}

struct ClassifiedLines;

impl Parser for ClassifiedLines {
    type ParserState = Vec<(u64, String)>;

    fn kind(&self, value: &Json) -> String {
        value["kind"].as_str().unwrap_or("?").into()
    }

    fn parse_record(
        &self,
        parser_state: &mut Self::ParserState,
        record: Record<'_>,
        value: &Json,
        _: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        let kind = self.kind(value);
        parser_state.push((record.offset, kind.clone()));
        static FIELDS: crate::fields::Fields = crate::fields::Fields::new(&[(
            "kind",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        )])
        .for_kind("record");
        records.fields(&kind, value, &FIELDS);
        match kind.as_str() {
            "message" => RecordOutcome::Used,
            "metadata" => RecordOutcome::skipped("not parsed"),
            _ => RecordOutcome::unreadable("not a supported record"),
        }
    }

    fn flush(&self, _: &mut Self::ParserState, _: &mut Output) {}
}

#[test]
fn append_boundaries_keep_outcomes_fields_offsets_and_parser_state() {
    let bytes = concat!(
        "\r\n",
        r#"{"kind":"message","extra":1}"#,
        "\r\n  \n",
        "{broken}\n",
        r#"{"kind":"metadata"}"#,
        "\n",
        r#"{"kind":"wrong"}"#,
        "\n",
        r#"{"kind":"message"}"#,
    )
    .as_bytes();
    let mut whole_carry = Vec::new();
    let whole = run(&ClassifiedLines, &mut whole_carry, bytes, 17).unwrap();
    assert_eq!(whole.counts.records, 4);
    assert_eq!(whole.counts.used["message"], 1);
    assert_eq!(whole.counts.skipped["metadata: not parsed"], 1);
    assert_eq!(whole.counts.unreadable.values().sum::<u64>(), 2);
    assert_eq!(whole.counts.unread_fields["message.extra"], 1);
    assert!(whole.consumed < bytes.len() as u64);
    for split in 0..=bytes.len() {
        let mut parser_state = Vec::new();
        let first = run(&ClassifiedLines, &mut parser_state, &bytes[..split], 17).unwrap();
        // The caller keeps a partial trailing record, not just bytes after
        // its most recent read boundary.
        let start = first.consumed as usize;
        let second = run(
            &ClassifiedLines,
            &mut parser_state,
            &bytes[start..],
            17 + first.consumed,
        )
        .unwrap();
        let mut counts = first.counts;
        counts.merge(&second.counts);
        assert_eq!(counts, whole.counts, "split {split}");
        assert_eq!(parser_state, whole_carry, "split {split}");
        assert_eq!(
            first.consumed + second.consumed,
            whole.consumed,
            "split {split}"
        );
    }
}

#[test]
fn capture_logs_verify_complete_captures_and_wait_on_incomplete_suffixes() {
    use polylogue_model::capture_log::{Header, MAGIC};
    let parser = Parse::CaptureLog(Box::new(EmptyDocument));
    let body = b"{}";
    let mut bytes = MAGIC.to_vec();
    bytes.extend(Header::of(body).encode());
    bytes.extend(body);
    let (steps, _) = parser.steps(None, &bytes, 0).unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].consumed, bytes.len() as u64);
    assert!(steps[0].repeats_whole_conversations);
    let end = bytes.len();
    bytes.extend(&Header::of(body).encode()[..10]);
    let (steps, _) = parser.steps(None, &bytes, 0).unwrap();
    assert_eq!(steps[0].consumed, end as u64);
    let (waiting, _) = parser.steps(None, &bytes[end..], end as u64).unwrap();
    assert_eq!(waiting[0].consumed, 0);
    bytes[end - 1] = b'[';
    assert!(matches!(
        parser.steps(None, &bytes, 0),
        Err(ParseError::CaptureLog(_))
    ));
}

struct DocumentEvent;
impl DocumentParser for DocumentEvent {
    fn document(&self, _: &[u8], out: &mut Output, records: &mut Records) {
        out.events.push(polylogue_model::Event {
            session: Some(polylogue_model::SessionId::parse("example:session").unwrap()),
            kind: "metadata".into(),
            at: None,
            after: None,
            range: None,
            data: serde_json::json!({"kept":true}),
        });
        records.record("metadata", &RecordOutcome::Used);
    }
}

#[test]
fn capture_event_ranges_name_the_capture_in_whole_and_incremental_reads() {
    use polylogue_model::capture_log::{Header, MAGIC};
    let body = br#"{"kept":true}"#;
    let mut bytes = MAGIC.to_vec();
    let mut expected = Vec::new();
    for _ in 0..2 {
        bytes.extend(Header::of(body).encode());
        let start = bytes.len() as u64;
        bytes.extend(body);
        expected.push((start, bytes.len() as u64));
    }
    let parser = Parse::CaptureLog(Box::new(DocumentEvent));
    let (steps, _) = parser.steps(None, &bytes, 0).unwrap();
    assert_eq!(steps.len(), 2);
    for (step, &range) in steps.iter().zip(&expected) {
        assert_eq!(step.output.events[0].range, Some(range));
        assert_eq!(&bytes[range.0 as usize..range.1 as usize], body);
    }
    let start = steps[0].consumed as usize;
    let (later, _) = parser.steps(None, &bytes[start..], start as u64).unwrap();
    assert_eq!(later[0].output.events, steps[1].output.events);
}

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
        static FIELDS: crate::fields::Fields = crate::fields::Fields::new(
            "record",
            &[(
                "kind",
                crate::fields::Rule::Read(crate::fields::ExpectedType::String),
            )],
        );
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
    assert_eq!(whole.counts.unreadable.len(), 2);
    assert_eq!(whole.counts.unreadable["wrong: not a supported record"], 1);
    let malformed: Vec<_> = whole
        .counts
        .unreadable
        .iter()
        .filter(|(key, _)| key.starts_with("?: malformed JSON: "))
        .collect();
    assert_eq!(malformed.len(), 1);
    assert_eq!(*malformed[0].1, 1);
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
    let (steps, _) = parser.steps(None, &bytes, 0, false).unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].consumed, bytes.len() as u64);
    assert!(steps[0].repeats_whole_conversations);
    let end = bytes.len();
    bytes.extend(&Header::of(body).encode()[..10]);
    let (steps, _) = parser.steps(None, &bytes, 0, false).unwrap();
    assert_eq!(steps[0].consumed, end as u64);
    let (waiting, _) = parser
        .steps(None, &bytes[end..], end as u64, false)
        .unwrap();
    assert_eq!(waiting[0].consumed, 0);
}

#[test]
fn a_damaged_capture_log_keeps_the_captures_around_it_and_reports_the_damage() {
    use polylogue_model::capture_log::{Header, MAGIC};
    let parser = Parse::CaptureLog(Box::new(EmptyDocument));
    let frame = |body: &[u8]| {
        let mut frame = Header::of(body).encode();
        frame.extend(body);
        frame
    };
    let unreadable = |step: &Step| step.counts.unreadable.clone();
    let reason = |text: &str| {
        [(format!("capture: invalid capture log: {text}"), 1)]
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    // A frame whose body fails its digest is counted unreadable; the frames
    // around it are read.
    let mut bytes = MAGIC.to_vec();
    bytes.extend(frame(b"{}"));
    let second = bytes.len();
    let mut damaged = frame(b"{}");
    *damaged.last_mut().unwrap() = b']';
    bytes.extend(damaged);
    let third = bytes.len();
    bytes.extend(frame(b"{}"));
    let (steps, _) = parser.steps(None, &bytes, 0, false).unwrap();
    assert_eq!(steps.len(), 3);
    assert!(unreadable(&steps[0]).is_empty());
    assert_eq!(
        (steps[1].start, steps[1].consumed),
        (second as u64, third as u64)
    );
    assert_eq!(unreadable(&steps[1]), reason("capture digest mismatch"));
    assert!(steps[1].output.events.is_empty() && steps[1].output.messages.is_empty());
    assert!(unreadable(&steps[2]).is_empty());
    assert_eq!(steps[2].consumed, bytes.len() as u64);

    // An input that will not grow and ends inside a frame reports it after
    // the complete captures; one that may still grow waits for the rest.
    let complete = bytes.len();
    bytes.extend(&frame(b"{}")[..30]);
    let (waiting, _) = parser.steps(None, &bytes, 0, false).unwrap();
    assert_eq!(waiting.last().unwrap().consumed, complete as u64);
    let (ended, _) = parser.steps(None, &bytes, 0, true).unwrap();
    assert_eq!(ended.len(), 4);
    let last = ended.last().unwrap();
    assert_eq!(
        (last.start, last.consumed),
        (complete as u64, bytes.len() as u64)
    );
    assert_eq!(
        unreadable(last),
        reason("the capture log ends inside a frame")
    );

    // A header that does not decode ends the framing: the rest is one
    // unreadable record.
    let mut bytes = MAGIC.to_vec();
    bytes.extend(frame(b"{}"));
    let broken = bytes.len();
    bytes.extend(vec![b'x'; 200]);
    let (steps, _) = parser.steps(None, &bytes, 0, false).unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(
        (steps[1].start, steps[1].consumed),
        (broken as u64, bytes.len() as u64)
    );
    assert_eq!(
        unreadable(&steps[1]),
        reason("invalid capture frame header")
    );
}

#[test]
fn a_capture_frame_has_the_documented_literal_layout() {
    use polylogue_model::capture_log::{Header, MAGIC};
    // A 20-digit zero-padded length, a space, the blake3 hex digest of the
    // body (computed outside this code) and a newline.
    let header =
        b"00000000000000000002 6e46dd10defc9b56c29a6ec56b508c21f54c08192194e4df25bf36f0c9c3c279\n";
    assert_eq!(Header::of(b"{}").encode(), header);
    let mut bytes = MAGIC.to_vec();
    bytes.extend(header);
    bytes.extend(b"{}");
    let parser = Parse::CaptureLog(Box::new(EmptyDocument));
    let (steps, _) = parser.steps(None, &bytes, 0, false).unwrap();
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].consumed, bytes.len() as u64);
}

struct DocumentEvent;
impl DocumentParser for DocumentEvent {
    fn document(&self, _: &[u8], out: &mut Output, records: &mut Records) {
        out.events.push(polylogue_model::Event {
            session: Some(polylogue_model::SessionId::parse("example:session").unwrap()),
            kind: "metadata".into(),
            at: None,
            at_quality: polylogue_model::TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
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
    let (steps, _) = parser.steps(None, &bytes, 0, false).unwrap();
    assert_eq!(steps.len(), 2);
    for (step, &range) in steps.iter().zip(&expected) {
        assert_eq!(step.output.events[0].range, Some(range));
        assert_eq!(&bytes[range.0 as usize..range.1 as usize], body);
    }
    let start = steps[0].consumed as usize;
    let (later, _) = parser
        .steps(None, &bytes[start..], start as u64, false)
        .unwrap();
    assert_eq!(later[0].output.events, steps[1].output.events);
}

struct FirstByte;
impl DocumentParser for FirstByte {
    fn document(&self, bytes: &[u8], out: &mut Output, records: &mut Records) {
        out.events.push(polylogue_model::Event {
            session: Some(polylogue_model::SessionId::parse("example:session").unwrap()),
            kind: "metadata".into(),
            at: None,
            at_quality: polylogue_model::TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            after: None,
            range: None,
            data: serde_json::json!({"first": bytes[0]}),
        });
        records.record("metadata", &RecordOutcome::Used);
    }
}

#[test]
fn a_document_starts_after_a_byte_order_mark_and_spans_the_whole_input() {
    let step = run_document(&FirstByte, b"\xEF\xBB\xBF{}").unwrap();
    assert_eq!(step.output.events[0].data["first"], b'{');
    assert_eq!(step.output.events[0].range, Some((0, 5)));
    assert_eq!(step.consumed, 5);
}

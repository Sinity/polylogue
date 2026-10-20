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
    let (step, carry) = parser.step(None, b"{}", 0).unwrap();
    assert_eq!(step.consumed, 2);
    assert_eq!(carry, Json::Null);
}

struct ClassifiedLines;

impl Parser for ClassifiedLines {
    type Carry = Vec<(u64, String)>;

    fn kind(&self, value: &Json) -> String {
        value["kind"].as_str().unwrap_or("?").into()
    }

    fn parse_record(
        &self,
        carry: &mut Self::Carry,
        record: Record<'_>,
        value: &Json,
        _: &mut Output,
        records: &mut Records,
    ) -> Disposition {
        let kind = self.kind(value);
        carry.push((record.offset, kind.clone()));
        records.fields(&kind, value, &[("kind", crate::fields::Rule::Read)]);
        match kind.as_str() {
            "message" => Disposition::Materialized,
            "metadata" => Disposition::unknown("not interpreted"),
            _ => Disposition::refused("not a supported record"),
        }
    }

    fn flush(&self, _: &mut Self::Carry, _: &mut Output) {}
}

#[test]
fn append_boundaries_keep_dispositions_fields_offsets_and_carry() {
    let bytes=b"\r\n{\"kind\":\"message\",\"extra\":1}\r\n  \n{broken}\n{\"kind\":\"metadata\"}\n{\"kind\":\"wrong\"}\n{\"kind\":\"message\"}";
    let mut whole_carry = Vec::new();
    let whole = run(&ClassifiedLines, &mut whole_carry, bytes, 17).unwrap();
    assert_eq!(whole.accounting.framed, 4);
    assert_eq!(whole.accounting.materialized["message"], 1);
    assert_eq!(whole.accounting.unknown["metadata: not interpreted"], 1);
    assert_eq!(whole.accounting.refused.values().sum::<u64>(), 2);
    assert_eq!(whole.accounting.unread_fields["message.extra"], 1);
    assert!(whole.consumed < bytes.len() as u64);
    for split in 0..=bytes.len() {
        let mut carry = Vec::new();
        let first = run(&ClassifiedLines, &mut carry, &bytes[..split], 17).unwrap();
        // The caller retains a partial trailing record, not just bytes after
        // its most recent read boundary.
        let start = first.consumed as usize;
        let second = run(
            &ClassifiedLines,
            &mut carry,
            &bytes[start..],
            17 + first.consumed,
        )
        .unwrap();
        let mut counts = first.accounting;
        counts.merge(&second.accounting);
        assert_eq!(counts, whole.accounting, "split {split}");
        assert_eq!(carry, whole_carry, "split {split}");
        assert_eq!(
            first.consumed + second.consumed,
            whole.consumed,
            "split {split}"
        );
    }
}

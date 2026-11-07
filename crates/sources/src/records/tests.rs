use super::*;
use crate::fields::ExpectedType as T;
use crate::fields::Rule;
use serde_json::json;

static KNOWN: Fields = Fields::new(&[("known", Rule::Read(T::String))]).for_kind("known");
static EMPTY: Fields = Fields::new(&[]).for_kind("empty");

#[test]
fn line_and_document_records_keep_the_same_classifications_and_field_counts() {
    let mut lines = Records::lines(4);
    let mut document = Records::default();
    for records in [&mut lines, &mut document] {
        records.record("message", &RecordOutcome::Used);
        records.record("message", &RecordOutcome::Used);
        records.record("metadata", &RecordOutcome::skipped("kept only as bytes"));
        records.record("message", &RecordOutcome::unreadable("missing identity"));
        records.fields(
            "message",
            &json!({"known":1,"extra":2,"nested":{"extra":3}}),
            &KNOWN,
        );
        records.fields("message", &json!({"extra":4}), &EMPTY);
    }
    let lines = lines.finish().unwrap();
    assert_eq!(lines, document.finish().unwrap());
    assert_eq!(lines.records, 4);
    assert_eq!(lines.used["message"], 2);
    assert_eq!(lines.skipped["metadata: kept only as bytes"], 1);
    assert_eq!(lines.unreadable["message: missing identity"], 1);
    assert_eq!(lines.unread_fields["message.extra"], 2);
    assert_eq!(lines.unread_fields["message.nested.extra"], 1);
}

#[test]
fn independent_line_framing_refuses_missing_and_extra_classifications() {
    for count in [1, 3] {
        let mut records = Records::lines(2);
        for _ in 0..count {
            records.record("row", &RecordOutcome::Used);
        }
        assert_eq!(
            records.finish(),
            Err(CountMismatch {
                records: 2,
                counted: count
            })
        );
    }
}

#[test]
fn unused_leaf_counted_without_changing_record_outcomes() {
    let mut records = Records::default();
    let value = json!({
        "read": 1,
        "kept": {"extra": 2},
        "excluded": 3,
        "items": [{"known": 1, "extra": 2}, {"known": 2, "extra": 3}],
    });
    static FIELDS: Fields = Fields::new(&[
        ("read", Rule::Read(T::String)),
        ("kept", Rule::Kept("kept as event data")),
        ("excluded", Rule::Excluded("not parsed")),
        ("items[].known", Rule::Read(T::String)),
    ])
    .for_kind("record");
    records.fields("event", &value, &FIELDS);
    let counts = records.finish().unwrap();
    assert_eq!(counts.records, 0);
    assert!(counts.used.is_empty());
    assert_eq!(
        counts.unread_fields,
        std::collections::BTreeMap::from([
            ("event.items[].extra".into(), 2),
            ("event.excluded".into(), 1)
        ])
    );
}

#[test]
fn public_shape_kind_uses_declarations_instead_of_wire_discriminator_values() {
    static FIELDS: Fields = Fields::new(&[("type", Rule::Read(T::String))]).shape(
        &[],
        &[crate::fields::KnownKinds {
            path: "type",
            values: &["message"],
        }],
    );
    let mut records = Records::default();
    records.fields(
        "response_item/distinctive-private-kind",
        &json!({"type":"distinctive-private-kind"}),
        &FIELDS,
    );
    let got = records.finish().unwrap();
    let shape = &got.wire_shapes[&serde_json::to_string(&("*", "/type")).unwrap()];
    assert_eq!(shape.record_kind, "*");
    assert_eq!(shape.unknown_kind_count, 1);
    assert!(
        !serde_json::to_string(&got.wire_shapes)
            .unwrap()
            .contains("distinctive-private-kind")
    );
}

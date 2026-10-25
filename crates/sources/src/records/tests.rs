use super::*;
use crate::fields::Rule;
use serde_json::json;

static KNOWN: Fields = Fields::new(&[("known", Rule::Read)]);
static EMPTY: Fields = Fields::new(&[]);

#[test]
fn line_and_document_records_keep_the_same_classifications_and_field_counts() {
    let mut lines = Records::lines(4);
    let mut document = Records::default();
    for records in [&mut lines, &mut document] {
        records.record("message", &Disposition::Materialized);
        records.record("message", &Disposition::Materialized);
        records.record("metadata", &Disposition::unknown("kept only as bytes"));
        records.record("message", &Disposition::refused("missing identity"));
        records.fields(
            "message",
            &json!({"known":1,"extra":2,"nested":{"extra":3}}),
            &KNOWN,
        );
        records.fields("message", &json!({"extra":4}), &EMPTY);
    }
    let lines = lines.finish().unwrap();
    assert_eq!(lines, document.finish().unwrap());
    assert_eq!(lines.framed, 4);
    assert_eq!(lines.materialized["message"], 2);
    assert_eq!(lines.unknown["metadata: kept only as bytes"], 1);
    assert_eq!(lines.refused["message: missing identity"], 1);
    assert_eq!(lines.unread_fields["message.extra"], 2);
    assert_eq!(lines.unread_fields["message.nested"], 1);
}

#[test]
fn independent_line_framing_refuses_missing_and_extra_classifications() {
    for count in [1, 3] {
        let mut records = Records::lines(2);
        for _ in 0..count {
            records.record("row", &Disposition::Materialized);
        }
        assert_eq!(
            records.finish(),
            Err(CountMismatch {
                framed: 2,
                accounted: count
            })
        );
    }
}

#[test]
fn field_checks_do_not_frame_records_or_change_declared_exclusions() {
    let mut records = Records::default();
    let value = json!({
        "read": 1,
        "kept": {"extra": 2},
        "excluded": 3,
        "items": [{"known": 1, "extra": 2}, {"known": 2, "extra": 3}],
    });
    static FIELDS: Fields = Fields::new(&[
        ("read", Rule::Read),
        ("kept", Rule::Kept("kept as event data")),
        ("excluded", Rule::Excluded("not parsed")),
        ("items[].known", Rule::Read),
    ]);
    records.fields("event", &value, &FIELDS);
    let counts = records.finish().unwrap();
    assert_eq!(counts.framed, 0);
    assert!(counts.materialized.is_empty());
    assert_eq!(
        counts.unread_fields,
        std::collections::BTreeMap::from([("event.items[].extra".into(), 2)])
    );
}

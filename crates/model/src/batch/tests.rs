use super::*;

#[test]
fn counts_must_add_up() {
    let mut a = RecordCounts {
        records: 3,
        ..RecordCounts::default()
    };
    a.add("user", &RecordOutcome::Used);
    a.add("progress", &RecordOutcome::skipped("tick"));
    assert_eq!(
        a.check(),
        Err(CountMismatch {
            records: 3,
            counted: 2
        })
    );
    a.add("?", &RecordOutcome::unreadable("malformed JSON"));
    assert_eq!(a.check(), Ok(()));
}

#[test]
fn merge_adds_every_map() {
    let mut a = RecordCounts {
        records: 1,
        ..RecordCounts::default()
    };
    a.add("user", &RecordOutcome::Used);
    let mut b = a.clone();
    b.unread("user", "x");
    a.merge(&b);
    assert_eq!(a.records, 2);
    assert_eq!(a.used["user"], 2);
    assert_eq!(a.unread_fields["user.x"], 1);
}

#[test]
fn order_key_sorts_by_input_then_offset() {
    let mut keys = [
        OrderKey {
            input: 2,
            offset: 0,
        },
        OrderKey {
            input: 1,
            offset: 90,
        },
        OrderKey {
            input: 1,
            offset: 10,
        },
    ];
    keys.sort();
    assert_eq!(
        keys[0],
        OrderKey {
            input: 1,
            offset: 10
        }
    );
    assert_eq!(
        keys[2],
        OrderKey {
            input: 2,
            offset: 0
        }
    );
}

#[test]
fn wire_shapes_survive_scratch_serialization_and_count_merge() {
    let key = serde_json::to_string(&("record.kind", "/a.b/[]/~1")).unwrap();
    let mut counts = RecordCounts::default();
    counts.wire_shapes.insert(
        key.clone(),
        WireShape {
            record_kind: "record.kind".into(),
            path: "/a.b/[]/~1".into(),
            types: BTreeMap::from([("string".into(), 2)]),
            expected: vec!["string".into()],
            dynamic_map: true,
            known_kinds: vec!["message".into()],
            unknown_kind_count: 0,
            table: None,
            unknown_table_count: 0,
        },
    );
    let decoded: RecordCounts =
        serde_json::from_slice(&serde_json::to_vec(&counts).unwrap()).unwrap();
    assert_eq!(decoded, counts);
    counts.merge(&decoded);
    assert_eq!(counts.wire_shapes[&key].types["string"], 4);
    assert_eq!(counts.wire_shapes[&key].record_kind, "record.kind");
    assert_eq!(counts.wire_shapes[&key].path, "/a.b/[]/~1");
}

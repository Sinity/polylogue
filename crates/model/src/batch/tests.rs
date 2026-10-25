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

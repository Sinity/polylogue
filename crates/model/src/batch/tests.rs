use super::*;

#[test]
fn counts_must_add_up() {
    let mut a = Accounting {
        framed: 3,
        ..Accounting::default()
    };
    a.add("user", &Disposition::Materialized);
    a.add("progress", &Disposition::unknown("tick"));
    assert_eq!(
        a.check(),
        Err(CountMismatch {
            framed: 3,
            accounted: 2
        })
    );
    a.add("?", &Disposition::refused("malformed JSON"));
    assert_eq!(a.check(), Ok(()));
}

#[test]
fn merge_adds_every_map() {
    let mut a = Accounting {
        framed: 1,
        ..Accounting::default()
    };
    a.add("user", &Disposition::Materialized);
    let mut b = a.clone();
    b.unread("user", "x");
    a.merge(&b);
    assert_eq!(a.framed, 2);
    assert_eq!(a.materialized["user"], 2);
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

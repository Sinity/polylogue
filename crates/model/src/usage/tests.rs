use super::*;

#[test]
fn unknown_token_kinds_stay_unknown() {
    let u = Usage::default();
    assert!(u.is_empty());
    assert_eq!(u.total_input(), None);
    // A reported zero is a known count, not an absent one.
    assert!(
        !Usage {
            input: Some(0),
            ..Usage::default()
        }
        .is_empty()
    );
    assert!(
        !Usage {
            speed: Some("fast".into()),
            ..Usage::default()
        }
        .is_empty()
    );
    assert!(
        !Usage {
            cache_write_1h: Some(1),
            ..Usage::default()
        }
        .is_empty()
    );
    assert_eq!(
        Usage {
            input: Some(0),
            ..Usage::default()
        }
        .total_input(),
        Some(0)
    );
    let u = Usage {
        input: Some(3),
        cached_input: Some(10),
        ..Usage::default()
    };
    assert_eq!(u.total_input(), Some(13));
}

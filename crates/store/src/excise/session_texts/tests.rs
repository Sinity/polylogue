use super::*;
use std::collections::BTreeSet;

#[test]
fn texts_deduplicate_order_and_count_by_exact_bytes() {
    let mut texts = SessionTexts::default();
    assert_eq!(texts.missing().unwrap(), 0);
    texts
        .visit(0, |_| panic!("empty set visited a text"))
        .unwrap();
    let values = [
        "",
        "short",
        "éééééé",
        "\0embedded NUL\0text",
        "b long text follows",
        "a long text first",
        "a long text first",
    ];
    for value in values {
        texts.insert(value).unwrap();
    }
    let expected: BTreeSet<_> = values.into_iter().collect();
    let mut actual = Vec::new();
    texts
        .visit(12, |value| {
            actual.push(value.to_owned());
            texts.found(value)
        })
        .unwrap();
    assert_eq!(
        actual,
        expected
            .iter()
            .filter(|v| v.len() >= 12)
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        texts.missing().unwrap(),
        expected.iter().filter(|v| v.len() < 12).count() as u64
    );
    texts.reset_found().unwrap();
    assert_eq!(texts.missing().unwrap(), expected.len() as u64);
    let error = texts
        .visit(0, |_| {
            Err(crate::StoreError::Invalid("callback failed".into()))
        })
        .unwrap_err();
    assert!(error.to_string().contains("callback failed"));
}

#[test]
fn a_many_text_session_visits_one_value_at_a_time_in_lexical_order() {
    let mut texts = SessionTexts::default();
    let value = |n| format!("session text {n:05} {}", "neutral payload ".repeat(32));
    for n in (0..10000).rev() {
        texts.insert(&value(n)).unwrap();
    }
    texts.insert(&value(5000)).unwrap();
    assert_eq!(texts.missing().unwrap(), 10000);
    let mut actual = blake3::Hasher::new();
    let mut count = 0;
    texts
        .visit(12, |text| {
            actual.update(text.as_bytes());
            count += 1;
            texts.found(text)
        })
        .unwrap();
    let mut expected = blake3::Hasher::new();
    for n in 0..10000 {
        expected.update(value(n).as_bytes());
    }
    assert_eq!(actual.finalize(), expected.finalize());
    assert_eq!(count, 10000);
    assert_eq!(texts.missing().unwrap(), 0);
    texts.reset_found().unwrap();
    assert_eq!(texts.missing().unwrap(), 10000);
}

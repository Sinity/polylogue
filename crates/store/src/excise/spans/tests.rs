use super::*;

#[test]
fn many_citations_keep_multiplicity_order_and_whole_input_precedence_on_disk() {
    let mut spans = Spans::default();
    assert!(spans.0.is_none());
    assert!(!spans.has_chain("absent", 1).unwrap());
    for n in (0..100_000).rev() {
        spans
            .insert(
                &format!("input:{}", n % 100),
                Span {
                    chain: n % 7,
                    start: n * 10,
                    end: n * 10 + 5,
                },
            )
            .unwrap();
    }
    let duplicate = Span {
        chain: 9,
        start: 1,
        end: 3,
    };
    for input in ["input:opaque:\0", "input:☺", "input:colon:two"] {
        spans.insert(input, duplicate).unwrap();
        spans.insert(input, duplicate).unwrap();
    }
    let conn = spans.0.as_ref().unwrap();
    let bytes: i64 = conn
        .query_row(
            "SELECT page_count*page_size FROM pragma_page_count(),pragma_page_size()",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(bytes > 1 << 20);
    assert_eq!(conn.path(), Some(""));
    let mut total = 0;
    let mut previous: Option<String> = None;
    spans
        .visit_inputs(|input, count| {
            if let Some(previous) = &previous {
                assert!(previous.as_str() < input);
            }
            previous = Some(input.to_owned());
            total += count;
            Ok(())
        })
        .unwrap();
    assert_eq!(total, 100_006);
    let mut count = 0;
    spans
        .visit("input:opaque:\0", |span| {
            assert_eq!(span, duplicate);
            count += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(count, 2);
    spans
        .remove_whole(&BTreeSet::from([
            "input:opaque:\0".into(),
            "input:0".into(),
        ]))
        .unwrap();
    assert!(!spans.has_chain("input:opaque:\0", 9).unwrap());
    let mut chains = Vec::new();
    spans
        .visit_chains("input:1", |chain| {
            chains.push(chain);
            Ok(())
        })
        .unwrap();
    assert_eq!(chains, (0..7).collect::<Vec<_>>());
    assert!(!spans.has_chain("input:1", 8).unwrap());
    let mut seen = 0;
    let mut last: Option<Span> = None;
    spans
        .visit("input:1", |span| {
            if let Some(last) = last {
                assert!(last <= span);
            }
            last = Some(span);
            seen += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, 1000);
    let mut remaining = 0;
    spans
        .visit_inputs(|_, count| {
            remaining += count;
            Ok(())
        })
        .unwrap();
    assert_eq!(remaining, 99_004);

    drop(spans);
}

#[test]
fn invalid_bounds_and_query_errors_do_not_become_empty_selections() {
    let mut spans = Spans::default();
    for (chain, start, end) in [(1, 0, 5), (2, -1, 5), (3, 4, 3), (4, 0, 11)] {
        spans.insert("input", Span { chain, start, end }).unwrap();
    }
    assert!(!spans.invalid_bounds("input", 1, 10).unwrap());
    for chain in [2, 3, 4] {
        assert!(spans.invalid_bounds("input", chain, 10).unwrap());
    }
    assert!(
        spans
            .visit("input", |_| Err(crate::StoreError::Invalid("stop".into())))
            .is_err()
    );
    spans
        .0
        .as_ref()
        .unwrap()
        .execute("DROP TABLE span", [])
        .unwrap();
    assert!(spans.visit("input", |_| Ok(())).is_err());
    assert!(spans.visit_inputs(|_, _| Ok(())).is_err());
    assert!(spans.visit_chains("input", |_| Ok(())).is_err());
    assert!(spans.has_chain("input", 1).is_err());
    assert!(spans.invalid_bounds("input", 1, 10).is_err());
}

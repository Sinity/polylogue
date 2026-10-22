use super::*;

#[test]
fn requests_and_checked_plans_are_exact_ordered_and_operation_local() {
    let plan = Erasures::new().unwrap();
    let file = plan._file.path().to_owned();
    for n in (0..10_000_u32).rev() {
        let hash = blake3::hash(&n.to_le_bytes());
        for (start, end) in [(12, 24), (1, 9), (12, 24)] {
            plan.request(hash.as_bytes(), start, end).unwrap();
        }
        assert_eq!(plan.requested(hash.as_bytes()).unwrap(), [(1, 9), (12, 24)]);
        plan.checked(
            hash.as_bytes(),
            &Erasure {
                ranges: if n % 2 == 0 {
                    vec![(1, 10), (12, 24)]
                } else {
                    vec![]
                },
                documents: 3,
                erased_documents: usize::from(n % 2 == 0),
            },
        )
        .unwrap();
        assert_eq!(
            plan.counts(hash.as_bytes()).unwrap(),
            Some((3, usize::from(n % 2 == 0)))
        );
    }
    assert!(!plan.has_unchecked().unwrap());
    let mut count = 0;
    let mut previous: Option<Vec<u8>> = None;
    plan.visit(|hash, ranges| {
        assert!(previous.as_deref().is_none_or(|p| p < hash));
        previous = Some(hash.to_vec());
        assert_eq!(ranges, [(1, 10), (12, 24)]);
        count += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 5000);
    assert!(
        plan.visit(|_, _| Err(crate::StoreError::Invalid("stop".into())))
            .is_err()
    );
    assert!(file.metadata().unwrap().len() > 1 << 20);
    drop(plan);
    assert!(!file.exists());
}

#[test]
fn a_checked_unrequested_piece_does_not_hide_an_unmatched_citation() {
    let plan = Erasures::new().unwrap();
    plan.request(b"missing", 1, 4).unwrap();
    plan.checked(
        b"unchanged",
        &Erasure {
            ranges: vec![],
            documents: 2,
            erased_documents: 0,
        },
    )
    .unwrap();
    assert_eq!(plan.counts(b"unchanged").unwrap(), Some((2, 0)));
    assert_eq!(plan.counts(b"missing").unwrap(), None);
    assert!(plan.has_unchecked().unwrap());
    assert!(
        plan.checked(
            b"unchanged",
            &Erasure {
                ranges: vec![],
                documents: 3,
                erased_documents: 0
            }
        )
        .is_err()
    );
    let mut visits = 0;
    plan.visit(|_, _| {
        visits += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(visits, 0);
}

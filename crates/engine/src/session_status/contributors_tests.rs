use super::*;
fn basis(i: usize) -> SourceRowBasis {
    SourceRowBasis {
        input: format!("synthetic-{i}"),
        original_hash: None,
        chain: 1,
        revision: 1,
        length: 64,
        parser_version: 10,
        interpretation: polylogue_model::BlobHash::of(b"synthetic recipe"),
    }
}
#[test]
fn contributor_sample_reports_partial_listing_and_digest_covers_unsampled_inputs() {
    let mut left = SessionStatus::default();
    let mut right = SessionStatus::default();
    for i in 0..5000 {
        left.add_contributor(basis(i), false, 8000).unwrap();
        let mut changed = basis(i);
        if i == 4999 {
            changed.revision = 2;
        }
        right.add_contributor(changed, false, 8000).unwrap();
    }
    assert_eq!(left.contributors, right.contributors);
    assert!(left.contributors.len() <= 8);
    assert_eq!(left.contributor_digest.count, 5000);
    assert_ne!(
        left.contributor_digest.digest(),
        right.contributor_digest.digest()
    );
    left.exact = true;
    left.add_contributor(basis(5000), true, 8000).unwrap();
    assert!(
        left.contributors
            .iter()
            .any(|b| b.input == "synthetic-5000")
    );
    assert!(left.add_contributor(basis(5001), true, 1).is_err());
}

#[test]
fn configured_claude_observed_member_permission_uses_full_count_not_display_sample() {
    let mut q = SessionStatus::default();
    q.exact = true;
    q.known_rows_readable = false;
    q.observed_member_verified = true;
    q.prefix_effects_supported = true;
    q.add_contributor(basis(0), true, 8000).unwrap();
    assert!(q.member_rows_readable());
    q.add_contributor(basis(1), true, 8000).unwrap();
    q.contributors.truncate(1);
    assert_eq!(
        q.contributors.len(),
        1,
        "the second contributor is not a display entry"
    );
    assert_eq!(q.contributor_digest.count, 2);
    assert!(
        !q.member_rows_readable(),
        "sampling must not grant member permission"
    );
}

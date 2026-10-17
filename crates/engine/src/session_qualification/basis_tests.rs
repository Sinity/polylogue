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
fn configured_claude_population_sample_never_claims_full_enumeration_or_drops_unsampled_basis() {
    let mut left = Qualification::default();
    let mut right = Qualification::default();
    for i in 0..5000 {
        left.include_basis(basis(i), false, 8000).unwrap();
        let mut changed = basis(i);
        if i == 4999 {
            changed.revision = 2;
        }
        right.include_basis(changed, false, 8000).unwrap();
    }
    assert_eq!(left.basis, right.basis);
    assert!(left.basis.len() <= 8);
    assert_eq!(left.basis_witness.count, 5000);
    assert_ne!(left.basis_witness.digest(), right.basis_witness.digest());
    left.exact = true;
    left.include_basis(basis(5000), true, 8000).unwrap();
    assert!(left.basis.iter().any(|b| b.input == "synthetic-5000"));
    assert!(left.include_basis(basis(5001), true, 1).is_err());
}

#[test]
fn configured_claude_observed_member_permission_uses_full_count_not_display_sample() {
    let mut q = Qualification::default();
    q.exact = true;
    q.known_rows_readable = false;
    q.certified_observed_member = true;
    q.prefix_effects_supported = true;
    q.include_basis(basis(0), true, 8000).unwrap();
    assert!(q.member_rows_readable());
    q.include_basis(basis(1), false, 1).unwrap();
    assert_eq!(
        q.basis.len(),
        1,
        "the second contributor is not a display entry"
    );
    assert_eq!(q.basis_witness.count, 2);
    assert!(
        !q.member_rows_readable(),
        "sampling must not grant member permission"
    );
}

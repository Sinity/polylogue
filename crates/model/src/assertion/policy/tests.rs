use super::*;
fn r(text: &str) -> Reference {
    Reference::parse(text).unwrap()
}
fn check(kind: AssertionKind, key: Option<&str>, links: &[(LinkRole, &str)]) -> Result<()> {
    let links: Vec<_> = links
        .iter()
        .map(|(role, reference)| (*role, r(reference)))
        .collect();
    validate_links(
        kind,
        key,
        links.iter().map(|(role, reference)| (*role, reference)),
    )
}

#[test]
fn saved_names_do_not_occupy_the_query_hash_namespace() {
    for key in ["deadbeef", "ABCDEF", "0123456789"] {
        assert!(
            check(AssertionKind::SavedQuery, Some(key), &[]).is_err(),
            "{key}"
        );
    }
    for key in ["research", "cost-30d", "a saved query"] {
        check(AssertionKind::SavedQuery, Some(key), &[]).unwrap();
    }
}

#[test]
fn a_new_saved_query_name_is_one_query_name_can_say() {
    for name in ["research", "cost-30d", "v1.2_final", "A-b.c_9"] {
        assert!(sayable_query_name(name), "{name}");
    }
    for name in [
        "",
        "two words",
        "a(b)",
        "a,b",
        "a:b",
        "caf\u{e9}",
        "a|b",
        "\"quoted\"",
    ] {
        assert!(!sayable_query_name(name), "{name:?}");
    }
}

#[test]
fn authored_target_shapes_preserve_exact_judgments_and_page_membership() {
    assert!(
        check(
            AssertionKind::Note,
            None,
            &[(LinkRole::About, "message:codex:m")]
        )
        .is_ok()
    );
    assert!(check(AssertionKind::Note, None, &[]).is_err());
    assert!(
        check(
            AssertionKind::Highlight,
            Some("yellow"),
            &[(LinkRole::About, "span:message:codex:m#b=0-4")]
        )
        .is_ok()
    );
    assert!(
        check(
            AssertionKind::Highlight,
            Some("yellow"),
            &[(LinkRole::About, "message:codex:m")]
        )
        .is_err()
    );
    assert!(
        check(
            AssertionKind::Title,
            None,
            &[(LinkRole::About, "session:codex:s")]
        )
        .is_ok()
    );
    assert!(
        check(
            AssertionKind::Title,
            None,
            &[(LinkRole::About, "message:codex:m")]
        )
        .is_err()
    );
    assert!(
        check(
            AssertionKind::Judgment,
            Some("accepted"),
            &[(LinkRole::About, "assertion:a@r1")]
        )
        .is_ok()
    );
    assert!(
        check(
            AssertionKind::Judgment,
            Some("accepted"),
            &[(LinkRole::About, "assertion:a")]
        )
        .is_err()
    );
    assert!(
        check(
            AssertionKind::Member,
            Some("reviewer"),
            &[
                (LinkRole::About, "session:codex:s"),
                (LinkRole::MemberOf, "page:p"),
                (LinkRole::RoleOf, "role:reviewer")
            ]
        )
        .is_ok()
    );
    assert!(
        check(
            AssertionKind::Member,
            Some("reviewer"),
            &[(LinkRole::About, "message:codex:m")]
        )
        .is_err()
    );
    assert!(
        check(
            AssertionKind::Page,
            Some("project"),
            &[(LinkRole::About, "session:codex:s")]
        )
        .is_err()
    );
    assert!(check(AssertionKind::Slot, None, &[(LinkRole::SlotOf, "page:p")]).is_ok());
    assert!(
        check(
            AssertionKind::Comparison,
            Some("clarity"),
            &[
                (LinkRole::About, "message:codex:a"),
                (LinkRole::About, "message:codex:b")
            ]
        )
        .is_ok()
    );
    assert!(
        check(
            AssertionKind::Comparison,
            Some("clarity"),
            &[(LinkRole::About, "message:codex:a")]
        )
        .is_err()
    );
}
#[test]
fn relation_shapes_allow_query_evidence_but_hold_authored_revisions_exact() {
    let links = [
        (LinkRole::About, "message:codex:m"),
        (LinkRole::DerivedFrom, "result:population"),
        (LinkRole::Evidence, "query:abc"),
    ];
    assert!(check(AssertionKind::Finding, None, &links).is_ok());
    let links = [
        (LinkRole::About, "message:codex:m"),
        (LinkRole::DerivedFrom, "assertion:rubric"),
    ];
    assert!(matches!(
        check(AssertionKind::Annotation, Some("clarity"), &links),
        Err(InvalidAssertion::Link {
            role: LinkRole::DerivedFrom,
            ..
        })
    ));
    assert!(
        check(
            AssertionKind::Annotation,
            Some("clarity"),
            &[
                (LinkRole::About, "message:codex:m"),
                (LinkRole::DerivedFrom, "assertion:rubric@r1")
            ]
        )
        .is_ok()
    );
    for role in [
        LinkRole::Answers,
        LinkRole::Challenges,
        LinkRole::Supports,
        LinkRole::Qualifies,
        LinkRole::Corrects,
        LinkRole::Blocks,
    ] {
        assert!(
            check(
                AssertionKind::Claim,
                None,
                &[
                    (LinkRole::About, "message:codex:m"),
                    (LinkRole::About, "assertion:claim@r1"),
                    (role, "assertion:claim@r1")
                ]
            )
            .is_ok()
        );
        assert!(
            check(
                AssertionKind::Claim,
                None,
                &[
                    (LinkRole::About, "message:codex:m"),
                    (role, "assertion:claim")
                ]
            )
            .is_err()
        );
    }
}
#[test]
fn revision_and_withdrawal_policy_rejects_mixed_roots_without_resolving_targets() {
    assert!(
        check(
            AssertionKind::Claim,
            None,
            &[
                (LinkRole::About, "message:codex:future"),
                (LinkRole::Revises, "assertion:a@left"),
                (LinkRole::Revises, "assertion:a@right")
            ]
        )
        .is_ok()
    );
    assert_eq!(
        check(
            AssertionKind::Claim,
            None,
            &[
                (LinkRole::About, "message:codex:m"),
                (LinkRole::Revises, "assertion:a@left"),
                (LinkRole::Revises, "assertion:b@right")
            ]
        ),
        Err(InvalidAssertion::RevisionFamilies)
    );
    assert!(
        check(
            AssertionKind::Withdrawal,
            None,
            &[
                (LinkRole::Withdraws, "assertion:family"),
                (LinkRole::Withdraws, "variant:translation@r1")
            ]
        )
        .is_ok()
    );
    assert_eq!(
        check(AssertionKind::Withdrawal, None, &[]),
        Err(InvalidAssertion::NoWithdrawalTarget)
    );
    assert!(
        check(
            AssertionKind::Withdrawal,
            None,
            &[(LinkRole::Withdraws, "variant:translation@r1#0")]
        )
        .is_err()
    );
    assert!(
        check(
            AssertionKind::Claim,
            None,
            &[
                (LinkRole::About, "message:codex:m"),
                (LinkRole::Withdraws, "assertion:a")
            ]
        )
        .is_err()
    );
    assert!(matches!(
        check(
            AssertionKind::Claim,
            None,
            &[
                (LinkRole::About, "message:codex:m"),
                (LinkRole::About, "message:codex:m")
            ]
        ),
        Err(InvalidAssertion::DuplicateLink { .. })
    ));
    assert!(check(AssertionKind::SavedQuery, None, &[]).is_err());
    assert!(check(AssertionKind::SavedQuery, Some(""), &[]).is_err());
    assert!(
        check(
            AssertionKind::Title,
            Some("unexpected"),
            &[(LinkRole::About, "session:codex:s")]
        )
        .is_err()
    );
}

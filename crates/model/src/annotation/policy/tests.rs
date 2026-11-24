use super::*;
fn r(text: &str) -> Reference {
    Reference::parse(text).unwrap()
}
fn check(kind: AnnotationKind, key: Option<&str>, links: &[(LinkRole, &str)]) -> Result<()> {
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
            check(AnnotationKind::SavedQuery, Some(key), &[]).is_err(),
            "{key}"
        );
    }
    for key in ["research", "cost-30d", "a saved query"] {
        check(AnnotationKind::SavedQuery, Some(key), &[]).unwrap();
    }
}

#[test]
fn a_new_saved_query_name_is_one_query_name_can_say() {
    for name in [
        "research",
        "cost-30d",
        "v1-2-final",
        "a-b-c9",
        "a",
        "a0",
        "a-",
    ] {
        assert!(valid_saved_name(name), "{name}");
    }
    for name in [
        "",
        "two words",
        "v1.2_final",
        "A-b.c_9",
        "Research",
        "-leading",
        "1query",
        "a_underscore",
        "a(b)",
        "a,b",
        "a:b",
        "caf\u{e9}",
        "a|b",
        "\"quoted\"",
    ] {
        assert!(!valid_saved_name(name), "{name:?}");
    }
}

#[test]
fn authored_target_shapes_preserve_judgments_and_page_membership() {
    assert!(
        check(
            AnnotationKind::Note,
            None,
            &[(LinkRole::About, "message:codex:m")]
        )
        .is_ok()
    );
    assert!(check(AnnotationKind::Note, None, &[]).is_err());
    assert!(
        check(
            AnnotationKind::Highlight,
            Some("yellow"),
            &[(LinkRole::About, "span:message:codex:m#b=0-4")]
        )
        .is_ok()
    );
    assert!(
        check(
            AnnotationKind::Highlight,
            Some("yellow"),
            &[(LinkRole::About, "message:codex:m")]
        )
        .is_err()
    );
    assert!(
        check(
            AnnotationKind::Title,
            None,
            &[(LinkRole::About, "session:codex:s")]
        )
        .is_ok()
    );
    assert!(
        check(
            AnnotationKind::Title,
            None,
            &[(LinkRole::About, "message:codex:m")]
        )
        .is_err()
    );
    assert!(
        check(
            AnnotationKind::Judgment,
            Some("accepted"),
            &[(LinkRole::About, "annotation:a@r1")]
        )
        .is_ok()
    );
    assert!(
        check(
            AnnotationKind::Judgment,
            Some("accepted"),
            &[(LinkRole::About, "annotation:a")]
        )
        .is_err()
    );
    assert!(
        check(
            AnnotationKind::Member,
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
            AnnotationKind::Member,
            Some("reviewer"),
            &[(LinkRole::About, "message:codex:m")]
        )
        .is_err()
    );
    assert!(
        check(
            AnnotationKind::Page,
            Some("project"),
            &[(LinkRole::About, "session:codex:s")]
        )
        .is_err()
    );
    assert!(check(AnnotationKind::Slot, None, &[(LinkRole::SlotOf, "page:p")]).is_ok());
    assert!(
        check(
            AnnotationKind::Comparison,
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
            AnnotationKind::Comparison,
            Some("clarity"),
            &[(LinkRole::About, "message:codex:a")]
        )
        .is_err()
    );
}
#[test]
fn relation_shapes_allow_query_evidence_but_pin_authored_revisions() {
    let links = [
        (LinkRole::About, "message:codex:m"),
        (LinkRole::DerivedFrom, "result:population"),
        (LinkRole::Evidence, "query:abc"),
    ];
    assert!(check(AnnotationKind::Finding, None, &links).is_ok());
    let links = [
        (LinkRole::About, "message:codex:m"),
        (LinkRole::DerivedFrom, "annotation:rubric"),
    ];
    assert!(matches!(
        check(AnnotationKind::RubricValue, Some("clarity"), &links),
        Err(InvalidAnnotation::Link {
            role: LinkRole::DerivedFrom,
            ..
        })
    ));
    assert!(
        check(
            AnnotationKind::RubricValue,
            Some("clarity"),
            &[
                (LinkRole::About, "message:codex:m"),
                (LinkRole::DerivedFrom, "annotation:rubric@r1")
            ]
        )
        .is_ok()
    );
    for role in [
        LinkRole::RepliesTo,
        LinkRole::Challenges,
        LinkRole::Supports,
        LinkRole::Qualifies,
        LinkRole::Corrects,
        LinkRole::Blocks,
    ] {
        assert!(
            check(
                AnnotationKind::Claim,
                None,
                &[
                    (LinkRole::About, "message:codex:m"),
                    (LinkRole::About, "annotation:claim@r1"),
                    (role, "annotation:claim@r1")
                ]
            )
            .is_ok()
        );
        assert!(
            matches!(
                check(
                    AnnotationKind::Claim,
                    None,
                    &[
                        (LinkRole::About, "message:codex:m"),
                        (role, "annotation:claim")
                    ]
                ),
                Err(InvalidAnnotation::Link { role: refused, .. }) if refused == role
            ),
            "{role:?}"
        );
    }
}
#[test]
fn revisions_and_withdrawals_refuse_mixed_roots() {
    assert!(
        check(
            AnnotationKind::Claim,
            None,
            &[
                (LinkRole::About, "message:codex:future"),
                (LinkRole::Revises, "annotation:a@left"),
                (LinkRole::Revises, "annotation:a@right")
            ]
        )
        .is_ok()
    );
    assert_eq!(
        check(
            AnnotationKind::Claim,
            None,
            &[
                (LinkRole::About, "message:codex:m"),
                (LinkRole::Revises, "annotation:a@left"),
                (LinkRole::Revises, "annotation:b@right")
            ]
        ),
        Err(InvalidAnnotation::RevisionFamilies)
    );
    assert!(
        check(
            AnnotationKind::Withdrawal,
            None,
            &[
                (LinkRole::Withdraws, "annotation:family"),
                (LinkRole::Withdraws, "alternate:translation@r1")
            ]
        )
        .is_ok()
    );
    assert_eq!(
        check(AnnotationKind::Withdrawal, None, &[]),
        Err(InvalidAnnotation::NoWithdrawalTarget)
    );
    assert!(
        check(
            AnnotationKind::Withdrawal,
            None,
            &[(LinkRole::Withdraws, "alternate:translation@r1#0")]
        )
        .is_err()
    );
    assert!(
        check(
            AnnotationKind::Claim,
            None,
            &[
                (LinkRole::About, "message:codex:m"),
                (LinkRole::Withdraws, "annotation:a")
            ]
        )
        .is_err()
    );
    assert!(matches!(
        check(
            AnnotationKind::Claim,
            None,
            &[
                (LinkRole::About, "message:codex:m"),
                (LinkRole::About, "message:codex:m")
            ]
        ),
        Err(InvalidAnnotation::DuplicateLink { .. })
    ));
    assert!(check(AnnotationKind::SavedQuery, None, &[]).is_err());
    assert!(check(AnnotationKind::SavedQuery, Some(""), &[]).is_err());
    assert!(
        check(
            AnnotationKind::Title,
            Some("unexpected"),
            &[(LinkRole::About, "session:codex:s")]
        )
        .is_err()
    );
}

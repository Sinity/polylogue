use super::*;
use crate::LinkKind;

fn id(value: &str) -> SessionId {
    SessionId::parse(&format!("fixture:{value}")).unwrap()
}
fn link(event: &str, parent: &str, child: &str, kind: LinkKind) -> ConfirmedLink {
    ConfirmedLink {
        event: event.into(),
        facts: Link {
            kind,
            parent: id(parent),
            child: id(child),
            at: None,
            call_id: None,
            evidence: "explicit native link".into(),
        },
    }
}

#[test]
fn membership_follows_all_explicit_kinds_and_keeps_physical_origins() {
    let links = vec![
        link("1", "root", "spawn", LinkKind::Spawn),
        link("2", "spawn", "fork", LinkKind::Fork),
        link("3", "root", "resume", LinkKind::Copy),
        link("4", "resume", "continue", LinkKind::Continuation),
    ];
    let actual = derive([id("unrelated")], links.clone());
    let group = actual
        .iter()
        .find(|group| group.root == Some(id("root")))
        .unwrap();
    assert_eq!(
        group.physical,
        [
            id("continue"),
            id("fork"),
            id("resume"),
            id("root"),
            id("spawn")
        ]
    );
    assert_eq!(group.links.len(), 4);
    assert!(group.unavailable.is_empty());
    let reversed = derive([id("unrelated")], links.into_iter().rev());
    assert_eq!(actual, reversed);
    assert_eq!(actual[0].identity(), reversed[0].identity());
    assert!(
        actual
            .iter()
            .any(|group| group.physical == [id("unrelated")])
    );
}

#[test]
fn cycles_and_competing_parents_do_not_select_a_root_or_drop_evidence() {
    let ambiguous = derive(
        [],
        [
            link("a", "parent-a", "child", LinkKind::Fork),
            link("b", "parent-b", "child", LinkKind::Copy),
        ],
    );
    assert_eq!(ambiguous.len(), 1);
    assert!(ambiguous[0].root.is_none());
    assert_eq!(ambiguous[0].links.len(), 2);
    assert!(
        ambiguous[0]
            .unavailable
            .iter()
            .any(|reason| reason == "logical_membership_multiple_parents")
    );
    let cyclic = derive(
        [],
        [
            link("a", "one", "two", LinkKind::Spawn),
            link("b", "two", "one", LinkKind::Continuation),
        ],
    );
    assert!(cyclic[0].root.is_none());
    assert!(
        cyclic[0]
            .unavailable
            .iter()
            .any(|reason| reason == "logical_membership_cycle")
    );
}

#[test]
fn late_membership_merges_and_splits_deterministically_and_replay_is_idempotent() {
    let separate = derive([id("parent"), id("child")], []);
    assert_eq!(separate.len(), 2);
    let fact = link("native-link", "parent", "child", LinkKind::Copy);
    let merged = derive([], [fact.clone()]);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].root, Some(id("parent")));
    assert_eq!(merged, derive([], [fact.clone(), fact]));
    assert_eq!(derive([id("parent"), id("child")], []), separate);
    assert_ne!(merged[0].identity(), separate[0].identity());
}

use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

#[test]
fn pearson_keeps_translation_scale_and_order_and_undefined_cases() {
    for scale in [1.0, 1e120, 1e200] {
        let pairs: Vec<_> = [(1.0, 1.0), (2.0, 2.0), (3.0, 4.0), (4.0, 3.0)]
            .map(|(x, y)| ((100_000_000.0 + x) * scale, y))
            .into();
        let value = pearson(&pairs).0.unwrap();
        assert!((value - 0.8).abs() < 1e-7);
        let reverse: Vec<_> = pairs.into_iter().rev().collect();
        assert!((pearson(&reverse).0.unwrap() - 0.8).abs() < 1e-7);
    }
    assert_eq!(pearson(&[(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]).0, Some(-1.0));
    assert_eq!(pearson(&[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]).0, Some(1.0));
    assert_eq!(
        pearson(&[(1.0, 1.0), (2.0, 2.0)]),
        (None, Some("insufficient_pairs"))
    );
    assert_eq!(
        pearson(&[(1e8, 1.0), (1e8, 2.0), (1e8, 3.0)]),
        (None, Some("zero_variance"))
    );
}

#[test]
fn comparisons_diff_only_varying_fields_and_similarity_excludes_self_and_zero_scores() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (id, text) in [
        ("codex:a", "one"),
        ("codex:b", "one two"),
        ("chatgpt:c", "one two three"),
    ] {
        let pk = f
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
        f.message(
            pk,
            MessageFacts {
                id: &format!("{id}:m"),
                role: "user",
                written_by: "human",
                text: Some(text),
                ..Default::default()
            },
        )
        .unwrap();
        f.note("tag", None, Some("shared"), &format!("session:{id}"))
            .unwrap();
    }
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let reference_pk: i64 = r
        .conn()
        .unwrap()
        .query_row("SELECT pk FROM session WHERE id='codex:a'", [], |r| {
            r.get(0)
        })
        .unwrap();
    let matches = similar(
        &r,
        &Selection {
            sessions: Some(vec![reference_pk]),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].session, "codex:b");
    assert_eq!(matches[0].score, 5);
    assert!(
        matches
            .iter()
            .all(|m| m.reference == "codex:a" && m.session != "codex:a")
    );
    assert!(similar(&r, &Selection::default()).is_err());
    assert!(
        similar(
            &r,
            &Selection {
                sessions: Some(vec![]),
                ..Default::default()
            }
        )
        .is_err()
    );
    let x = compare(&r, &Selection::default()).unwrap();
    assert!(x.differences.contains_key("word_count"));
    assert!(!x.differences.contains_key("workflow_shape"));
    assert!(!x.differences.contains_key("session"));
    let a = x.profiles.iter().find(|p| p.session == "codex:a").unwrap();
    let b = x.profiles.iter().find(|p| p.session == "codex:b").unwrap();
    assert!(similarity(a, a).is_none());
    assert_eq!(similarity(a, b).unwrap().score, 5);
    let mut b = reader_profile(&r, "codex:b");
    b.origin = "other".into();
    b.facts.workflow_shape = "unknown".into();
    b.tags.clear();
    assert!(similarity(a, &b).is_none());
    let mut reference = reader_profile(&r, "codex:a");
    reference.facts.first_at = Some(0);
    b.facts.first_at = Some(3 * 86_400_000);
    assert_eq!(similarity(&reference, &b).unwrap().score, 2);
    b.facts.first_at = Some(14 * 86_400_000);
    assert_eq!(similarity(&reference, &b).unwrap().score, 1);
    b.facts.first_at = Some(15 * 86_400_000);
    assert!(similarity(&reference, &b).is_none());
    let c = correlation(
        &r,
        &Selection::default(),
        "message_count:median_tool_call_ms",
    )
    .unwrap();
    assert_eq!(c.sample_count, 0);
    assert_eq!(c.missing_pairs, 3);
    assert!(correlation(&r, &Selection::default(), "word_count:unknown").is_err());
}
fn reader_profile(r: &Reader<'_>, id: &str) -> Profile {
    r.session_profiles(&Selection::default())
        .unwrap()
        .into_iter()
        .find(|p| p.session == id)
        .unwrap()
}

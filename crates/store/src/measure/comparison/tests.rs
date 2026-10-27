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

fn matches(r: &Reader<'_>, reference: &str) -> Vec<serde_json::Value> {
    r.query(
        &format!(
            "{} ORDER BY score DESC,session",
            similarity_query("SELECT pk AS _pk FROM session WHERE id=?1")
        ),
        &[crate::SqlValue::Text(reference.into())],
    )
    .unwrap()
}

#[test]
fn comparisons_and_similarity_read_facts_without_message_usage_or_parent_walks() {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
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
        f.note("tag", None, Some("Shared"), &format!("session:{id}"))
            .unwrap();
    }
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    reader
        .conn()
        .unwrap()
        .authorizer(Some(|context: AuthContext<'_>| match context.action {
            AuthAction::Read {
                table_name: "message" | "link" | "block" | "tool_call",
                ..
            } => Authorization::Deny,
            _ => Authorization::Allow,
        }))
        .unwrap();
    let rows = matches(&reader, "codex:a");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["session"], "codex:b");
    assert_eq!(rows[0]["score"], 5);
    assert!(
        rows.iter()
            .all(|r| r["reference"] == "codex:a" && r["session"] != "codex:a")
    );
    let x = compare(&reader, &Selection::default(), "").unwrap();
    assert!(x.differences.contains_key("word_count"));
    assert!(!x.differences.contains_key("workflow_shape"));
    assert!(!x.differences.contains_key("cost_usd"));
    assert_eq!(x.underived_sessions, 0);
    let c = correlation(
        &reader,
        &Selection::default(),
        "message_count:median_tool_call_ms",
    )
    .unwrap();
    assert_eq!(c.sample_count, 0);
    assert_eq!(c.missing_pairs, 3);
    assert!(correlation(&reader, &Selection::default(), "word_count:unknown").is_err());
}

#[test]
fn similarity_time_weights_include_boundaries_and_exclude_zero_scores() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (id, day) in [
        ("codex:ref", 0),
        ("chatgpt:three", 3),
        ("chatgpt:fourteen", 14),
        ("chatgpt:fifteen", 15),
    ] {
        f.session(SessionFacts {
            id,
            started: Some(day * 86_400_000),
            ..Default::default()
        })
        .unwrap();
    }
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let rows = matches(&reader, "codex:ref");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["session"], "chatgpt:three");
    assert_eq!(rows[0]["score"], 2);
    assert_eq!(rows[1]["session"], "chatgpt:fourteen");
    assert_eq!(rows[1]["score"], 1);
}

#[test]
fn underived_sessions_keep_null_facts_and_do_not_fail_aggregates_or_similarity() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    f.session(SessionFacts {
        id: "codex:new",
        ..Default::default()
    })
    .unwrap();
    f.session(SessionFacts {
        id: "codex:other",
        ..Default::default()
    })
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let p = reader.session_profiles(&Selection::default()).unwrap();
    assert!(p.iter().all(|p| p.facts.is_none()));
    let c = compare(&reader, &Selection::default(), "").unwrap();
    assert_eq!(c.underived_sessions, 2);
    let d = distribution(&reader, &Selection::default(), "message_count").unwrap();
    assert_eq!(d.underived_sessions, 2);
    assert_eq!(d.missing_sessions, 2);
    assert_eq!(d.median, None);
    let c = correlation(&reader, &Selection::default(), "message_count:word_count").unwrap();
    assert_eq!(c.underived_sessions, 2);
    assert_eq!(c.sample_count, 0);
    let rows = matches(&reader, "codex:new");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["score"], 1);
}

#[test]
fn explicit_cost_comparison_reads_own_usage_without_parent_history() {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let pk = f
        .session(SessionFacts {
            id: "codex:cost",
            ..Default::default()
        })
        .unwrap();
    let message = f
        .message(
            pk,
            MessageFacts {
                id: "reply",
                role: "assistant",
                written_by: "model",
                model: Some("gpt-6.1-sol"),
                ..Default::default()
            },
        )
        .unwrap();
    f.usage(
        message,
        &polylogue_model::Usage {
            input: Some(100),
            cached_input: Some(0),
            cache_write: Some(0),
            output: Some(10),
            reasoning: Some(0),
            ..Default::default()
        },
        None,
    )
    .unwrap();
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    reader
        .conn()
        .unwrap()
        .authorizer(Some(|context: AuthContext<'_>| match context.action {
            AuthAction::Read {
                table_name: "link" | "block",
                ..
            } => Authorization::Deny,
            _ => Authorization::Allow,
        }))
        .unwrap();
    let compared = compare(&reader, &Selection::default(), "cost_usd").unwrap();
    let metric = compared.profiles[0].metric.as_ref().unwrap();
    assert_eq!(metric.name, "cost_usd");
    assert!(metric.value.unwrap() > 0.0);
    assert!(compare(&reader, &Selection::default(), "unknown").is_err());
}

use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

#[test]
fn comparisons_use_recorded_fields_and_keep_missing_values() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (id, origin, text, at) in [
        ("codex:a", "model", "one", Some(0)),
        ("codex:b", "model", "one two", Some(60000)),
        ("chatgpt:c", "human", "one two three", None),
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
                id: &format!("{id}-m"),
                role: if origin == "model" {
                    "assistant"
                } else {
                    "user"
                },
                written_by: origin,
                text: Some(text),
                at,
                ..Default::default()
            },
        )
        .unwrap();
    }
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let compared = compare(&reader, &Selection::default(), "origin,words,wall_span").unwrap();
    assert_eq!(compared.sessions.len(), 3);
    assert!(compared.varies.contains(&"origin".into()));
    assert!(compared.varies.contains(&"words".into()));
    assert_eq!(compared.unknown["wall_span"], 1);
    assert_eq!(compared.pending, 0);
    let (varies, unknown, pending) =
        summary(&reader, &Selection::default(), "origin,words,wall_span").unwrap();
    assert_eq!(varies, compared.varies);
    assert_eq!(unknown, compared.unknown);
    assert_eq!(pending, 0);
    assert!(compare(&reader, &Selection::default(), "unsupported").is_err());
}

#[test]
fn structural_comparison_does_not_read_bodies_usage_or_parents() {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for id in ["codex:a", "codex:b"] {
        let s = f
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
        f.message(
            s,
            MessageFacts {
                id: &format!("{id}-m"),
                role: "assistant",
                written_by: "model",
                text: Some(id),
                at: Some(0),
                ..Default::default()
            },
        )
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
                table_name: "message" | "block" | "tool_call" | "link" | "text",
                ..
            } => Authorization::Deny,
            _ => Authorization::Allow,
        }))
        .unwrap();
    let compared = compare(&reader, &Selection::default(), "").unwrap();
    assert_eq!(compared.pending, 0);
    assert_eq!(compared.sessions.len(), 2);
    assert!(summary(&reader, &Selection::default(), "").is_ok());
}

#[test]
fn pending_populations_do_not_serve_stale_values_and_single_reads_compute() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let mut keys = vec![];
    for id in ["codex:a", "codex:b"] {
        let s = f
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
        keys.push(s);
        f.message(
            s,
            MessageFacts {
                id: &format!("{id}-m"),
                role: "user",
                written_by: "human",
                text: Some("one"),
                at: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
    }
    f.derive().unwrap();
    for (i, &s) in keys.iter().enumerate() {
        f.message(
            s,
            MessageFacts {
                id: &format!("new-{i}"),
                role: "assistant",
                written_by: "model",
                text: Some("one two"),
                at: Some(60000),
                ..Default::default()
            },
        )
        .unwrap();
    }
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let population = compare(&reader, &Selection::default(), "words,wall_span").unwrap();
    assert_eq!(population.pending, 2);
    assert!(
        population
            .sessions
            .iter()
            .all(|s| s.values.values().all(Value::is_null))
    );
    let one = compare(
        &reader,
        &Selection {
            sessions: Some(vec![keys[0]]),
            ..Default::default()
        },
        "words,wall_span",
    )
    .unwrap();
    assert_eq!(one.pending, 0);
    assert_eq!(one.sessions[0].values["words"], 3);
    assert_eq!(one.sessions[0].values["wall_span"], 60000);
}

#[test]
fn cost_comparison_keeps_integer_nano_usd_and_unknown_totals() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (id, cost) in [
        ("claude-code:known", Some(9_007_199_254_740_993_i64)),
        ("claude-code:zero", Some(0)),
        ("chatgpt:unknown", None),
    ] {
        let s = f
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
        f.message(
            s,
            MessageFacts {
                id: &format!("{id}-m"),
                role: "assistant",
                written_by: "model",
                cost_nusd: cost,
                ..Default::default()
            },
        )
        .unwrap();
    }
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let c = compare(&reader, &Selection::default(), "cost_nusd").unwrap();
    assert_eq!(c.unknown["cost_nusd"], 1);
    assert!(
        c.sessions
            .iter()
            .any(|s| s.values["cost_nusd"] == 9_007_199_254_740_993_i64)
    );
    assert!(c.sessions.iter().any(|s| s.values["cost_nusd"] == 0));
}

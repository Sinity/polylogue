use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
use serde_json::{Value, json};

fn fixture() -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    for (id, input, output, cost, repo) in [
        ("a", Some(1), Some(4), Some(1_000_000_000), "east"),
        ("b", Some(3), Some(6), Some(3_000_000_000), "east"),
        ("c", Some(5), Some(5), Some(5_000_000_000), "west"),
        ("u", None, None, None, "west"),
    ] {
        let session = fixture
            .session(SessionFacts {
                id: &format!("claude-code:{id}"),
                repo: Some(repo),
                started: Some(1000),
                ..Default::default()
            })
            .unwrap();
        let message = fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("claude-code:message-{id}"),
                    role: "assistant",
                    written_by: "model",
                    at: Some(3000),
                    cost_nusd: cost,
                    ..Default::default()
                },
            )
            .unwrap();
        fixture
            .usage(
                message,
                &polylogue_model::Usage {
                    input,
                    output,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
    }
    fixture.derive().unwrap();
    (dir, fixture)
}
fn rows(fixture: &Fixture, text: &str) -> Vec<Value> {
    let mut options = opts(&[]);
    options.fields = vec!["id".into()];
    let program = compile(text, &options).unwrap_or_else(|error| panic!("{text}: {error}"));
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut rows = reader
        .query_rows(&program)
        .unwrap_or_else(|error| panic!("{text}: {error}"));
    for row in &mut rows {
        crate::query::execute::present(&program, row).unwrap();
    }
    rows
}

#[test]
fn computed_conditions_keep_unknowns_and_native_money_and_duration_units() {
    let (_dir, fixture) = fixture();
    let ids = |text| {
        rows(&fixture, text)
            .into_iter()
            .map(|row| row["id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        ids("sessions where repo = east | sort by id"),
        ["claude-code:a", "claude-code:b"]
    );
    assert!(ids("sessions where repo = EAST").is_empty());
    assert_eq!(
        ids("sessions where repo:EAST | sort by id"),
        ["claude-code:a", "claude-code:b"]
    );
    assert_eq!(
        ids("sessions where input_tokens >= output_tokens"),
        ["claude-code:c"]
    );
    assert_eq!(
        ids("sessions where cost / 2 > 1 | sort by id"),
        ["claude-code:b", "claude-code:c"]
    );
    assert_eq!(
        ids("sessions where last_active - started >= 1s | sort by id").len(),
        4
    );
    assert!(ids("sessions where NOT (input_tokens / output_tokens > 0)").is_empty());
    assert!(ids("sessions where input_tokens / (output_tokens - output_tokens) > 0").is_empty());
    assert_eq!(
        ids("sessions | sort by input_tokens / output_tokens desc"),
        [
            "claude-code:c",
            "claude-code:b",
            "claude-code:a",
            "claude-code:u"
        ]
    );
}

#[test]
fn numeric_aggregates_execute_expressions_intervals_and_complete_pairs() {
    let (_dir, fixture) = fixture();
    let answer = &rows(
        &fixture,
        "sessions | agg sum(input_tokens + output_tokens) as total, covered(input_tokens,output_tokens) as occupied, corr(input_tokens,output_tokens) as correlation, sum(cost / 2) as half_cost",
    )[0];
    assert_eq!(answer["total"], 24.0);
    assert_eq!(answer["occupied"], 5.0);
    assert!(
        (answer["correlation"].as_f64().unwrap() - 0.5).abs() < 1e-12,
        "{answer}"
    );
    assert_eq!(answer["half_cost"], 4.5);
    assert_eq!(answer["unknown_input_tokens"], 1);
    assert_eq!(answer["unknown_output_tokens"], 1);
    let bins = &rows(&fixture, "sessions | agg histogram(repo) as repositories")[0]["repositories"];
    assert_eq!(
        *bins,
        json!([{"value":"east","count":2},{"value":"west","count":2}])
    );
    let groups = rows(
        &fixture,
        "sessions | group by input_tokens + output_tokens as total | count | sort by total",
    );
    assert_eq!(
        groups
            .iter()
            .map(|row| row["total"].clone())
            .collect::<Vec<_>>(),
        [json!(5), json!(9), json!(10), Value::Null]
    );
}

#[test]
fn conditional_expression_is_lazy_and_keeps_the_selected_native_kind() {
    let (_dir, fixture) = fixture();
    let options = opts(&[]);
    let mut compiler = Compiler::new("", &options);
    let expression = Expression::Function {
        name: "case_eq".into(),
        arguments: vec![
            Expression::field("repo"),
            Expression::value(crate::query::ast::Value::scalar("east")),
            Expression::field("cost"),
        ],
    };
    let scope = RowContext::plain(Unit::Sessions);
    let (sql, kind) = compiler.sql_expression(&expression, &scope, 0).unwrap();
    assert_eq!(kind, ColKind::Money);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let rows = reader
        .query(
            &format!("SELECT {sql} AS amount FROM session s ORDER BY s.id"),
            &crate::selection::values(&compiler.params),
        )
        .unwrap();
    assert_eq!(
        rows.iter()
            .map(|row| row["amount"].clone())
            .collect::<Vec<_>>(),
        [
            json!(1_000_000_000),
            json!(3_000_000_000),
            Value::Null,
            Value::Null
        ]
    );
    let guarded = Expression::Function {
        name: "case_eq".into(),
        arguments: vec![
            Expression::value(crate::query::ast::Value::scalar("1")),
            Expression::value(crate::query::ast::Value::scalar("2")),
            Expression::Arithmetic {
                operator: Arithmetic::Multiply,
                left: Box::new(Expression::value(crate::query::ast::Value::scalar("1e308"))),
                right: Box::new(Expression::value(crate::query::ast::Value::scalar("1e308"))),
            },
        ],
    };
    let mut compiler = Compiler::new("", &options);
    let (sql, kind) = compiler.sql_expression(&guarded, &scope, 0).unwrap();
    assert_eq!(kind, ColKind::Number);
    let rows = reader
        .query(
            &format!("SELECT {sql} AS guarded"),
            &crate::selection::values(&compiler.params),
        )
        .unwrap();
    assert_eq!(rows[0]["guarded"], Value::Null);
}

#[test]
fn expression_calendar_values_use_the_programs_logical_day_and_zone() {
    let (_dir, fixture) = fixture();
    let answer = &rows(
        &fixture,
        "sessions | group by year(started),hour(started),weekday(started),month_day(started) | count",
    )[0];
    assert_eq!(answer["year"], 1969);
    assert_eq!(answer["hour"], 1);
    assert_eq!(answer["weekday"], "wed");
    assert_eq!(answer["month_day"], "12-31");
    assert_eq!(answer["count"], 4);
}

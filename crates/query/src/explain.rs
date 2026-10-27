//! Grammar and execution shape, without opening or querying a store.

use crate::{Options, QueryError, ast, compile};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Explanation {
    pub parsed: ast::Query,
    pub units: Vec<ast::Unit>,
    pub program: crate::Program,
}

/// Uses the same validation and compilation as execution.
pub fn query(expression: &str, options: &Options) -> Result<Explanation, QueryError> {
    let parsed = crate::parse(expression)?;
    let program = compile(expression, options)?;
    Ok(Explanation {
        units: selected_units(&parsed, ast::Unit::Sessions),
        parsed,
        program,
    })
}

fn selected_units(query: &ast::Query, implicit_unit: ast::Unit) -> Vec<ast::Unit> {
    fn add(out: &mut Vec<ast::Unit>, unit: ast::Unit) {
        if !out.contains(&unit) {
            out.push(unit);
        }
    }
    fn condition(cond: &ast::Cond, out: &mut Vec<ast::Unit>) {
        use ast::CondKind;
        match &cond.kind {
            CondKind::And { all } => {
                for c in all {
                    condition(c, out);
                }
            }
            CondKind::Or { any } => {
                for c in any {
                    condition(c, out);
                }
            }
            CondKind::Not { cond } => condition(cond, out),
            CondKind::Exists { unit, cond } => {
                add(out, *unit);
                condition(cond, out);
            }
            CondKind::Count { unit, cond, .. } => {
                add(out, *unit);
                if let Some(c) = cond {
                    condition(c, out);
                }
            }
            CondKind::In { query, .. } => {
                for u in selected_units(query, ast::Unit::Sessions) {
                    add(out, u);
                }
            }
            CondKind::Seq { steps, .. } => {
                add(out, ast::Unit::ToolCalls);
                for c in steps {
                    condition(c, out);
                }
            }
            CondKind::Match { .. }
            | CondKind::Compare { .. }
            | CondKind::Between { .. }
            | CondKind::Text { .. }
            | CondKind::Url { .. }
            | CondKind::Position { .. }
            | CondKind::Saved { .. } => {}
        }
    }
    let mut out = Vec::new();
    let mut current_unit = match &query.source {
        ast::Source::Unit { unit, .. } => *unit,
        ast::Source::Sessions { .. } | ast::Source::Empty => implicit_unit,
    };
    match &query.source {
        ast::Source::Unit { unit, cond } => {
            add(&mut out, *unit);
            if let Some(c) = cond {
                condition(c, &mut out);
            }
        }
        ast::Source::Sessions { cond } => {
            add(&mut out, implicit_unit);
            condition(cond, &mut out);
        }
        ast::Source::Empty => add(&mut out, ast::Unit::Sessions),
    }
    for child in &query.attach {
        add(&mut out, child.unit);
        for window in &child.windows {
            if let ast::Window::Cond { cond } = window {
                condition(cond, &mut out);
            }
        }
    }
    for stage in &query.stages {
        match &stage.kind {
            ast::StageKind::Where { cond } => condition(cond, &mut out),
            ast::StageKind::Scope { unit, cond } => {
                current_unit = *unit;
                add(&mut out, *unit);
                if let Some(c) = cond {
                    condition(c, &mut out);
                }
            }
            ast::StageKind::Combine { query, .. } => {
                for u in selected_units(query, current_unit) {
                    add(&mut out, u);
                }
            }
            ast::StageKind::SortBy { .. }
            | ast::StageKind::GroupBy { .. }
            | ast::StageKind::Agg { .. }
            | ast::StageKind::Count
            | ast::StageKind::Limit { .. }
            | ast::StageKind::Offset { .. }
            | ast::StageKind::Distinct
            | ast::StageKind::Matches
            | ast::StageKind::Measure { .. } => {}
        }
    }
    out
}

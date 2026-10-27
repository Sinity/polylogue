//! Grammar and execution shape, without opening or querying a store.

use crate::{Options, QueryError, ast, compile};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Explanation {
    pub expression: String,
    pub parsed: ast::Query,
    pub selected_units: Vec<ast::Unit>,
    pub unit: ast::Unit,
    pub shape: crate::Shape,
    pub columns: Vec<crate::Column>,
    pub ranked: bool,
    pub page_size: u64,
    pub order: Vec<crate::OrderKey>,
    pub measure: Option<crate::Measure>,
    pub notes: Vec<String>,
    pub unsupported: Vec<String>,
    pub sql: String,
    pub parameters: Vec<crate::Param>,
    pub now: i64,
    pub zone: String,
    pub day_start_minutes: u16,
}

/// Uses the same validation and compilation as execution. Options supply
/// the clock and declarations; this function performs no I/O.
pub fn query(expression: &str, options: &Options) -> Result<Explanation, QueryError> {
    let parsed = crate::parse(expression)?;
    let program = compile(expression, options)?;
    Ok(Explanation {
        expression: expression.to_owned(),
        selected_units: selected_units(&parsed, ast::Unit::Sessions),
        parsed,
        unit: program.unit,
        shape: program.shape,
        columns: program.columns,
        ranked: program.ranked,
        page_size: program.page_size,
        order: program.order,
        measure: program.measure,
        notes: program.notes,
        unsupported: program.incomplete,
        sql: program.sql,
        parameters: program.params,
        now: program.now,
        zone: program.zone,
        day_start_minutes: program.day_start_minutes,
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
            _ => {}
        }
    }
    let mut out = Vec::new();
    let mut current_unit = match &query.source {
        ast::Source::Unit { unit, .. } => *unit,
        _ => implicit_unit,
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
            _ => {}
        }
    }
    out
}

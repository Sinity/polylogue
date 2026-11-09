//! Grammar and execution shape, without opening or querying a store.

use crate::query::{Options, QueryError, ast};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Explanation {
    pub parsed: ast::Query,
    pub units: Vec<ast::Unit>,
    pub program: crate::query::Program,
}

/// Uses the same validation and compilation as execution.
pub fn query(expression: &str, options: &Options) -> Result<Explanation, QueryError> {
    query_language::depth::deeper(|| {
        let plan = polylogue_query::compile(expression, options)?;
        let parsed = plan.bound_query().query().clone();
        let program = crate::query::lower(&plan, options)?;
        let mut units = selected_units(plan.expanded_query().query());
        if !units.contains(&program.unit) {
            units.insert(0, program.unit);
        }
        Ok(Explanation {
            units,
            parsed,
            program,
        })
    })
}

fn selected_units(query: &ast::Query) -> Vec<ast::Unit> {
    let mut units = Vec::new();
    query.visit_units(&mut |name| {
        if let Some(unit) = ast::Unit::from_name(name.strip_prefix("pl.").unwrap_or(name))
            && !units.contains(&unit)
        {
            units.push(unit);
        }
    });
    units
}

#[cfg(test)]
mod tests;

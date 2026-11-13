//! Grammar and execution shape, without opening or querying a store.

use crate::query::{Options, QueryError, ast};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Explanation {
    pub parsed: ast::Query,
    pub units: Vec<ast::Unit>,
    /// The ranking in effect: none, relevance, similarity or fused, or
    /// replaced by a later stage; `None` for a kept result's rows.
    pub order: Option<polylogue_query::selection::Order>,
    pub program: crate::query::Program,
}

/// Uses the same validation and compilation as execution.
pub fn query(expression: &str, options: &Options) -> Result<Explanation, QueryError> {
    query_language::depth::deeper(|| {
        let plan = polylogue_query::compile(expression, options)?;
        let parsed = plan.bound_query().query().clone();
        let program = crate::query::lower(&plan, options)?;
        let order = plan
            .selection(
                &options.fields,
                &options.clock,
                options.provider_catalog.as_deref(),
            )?
            .map(|selection| selection.order);
        let mut units = selected_units(plan.expanded_query().query());
        if !units.contains(&program.unit) {
            units.insert(0, program.unit);
        }
        Ok(Explanation {
            units,
            parsed,
            order,
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

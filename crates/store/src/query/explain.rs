//! Grammar and execution shape, without opening or querying a store.

use crate::query::{Options, QueryError, ast, compile};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Explanation {
    pub parsed: ast::Query,
    pub units: Vec<ast::Unit>,
    pub program: crate::query::Program,
}

/// Uses the same validation and compilation as execution.
pub fn query(expression: &str, options: &Options) -> Result<Explanation, QueryError> {
    let parsed = crate::query::parse::with_catalog(
        expression,
        &crate::query::catalog::catalog(&options.origins, &options.command_words),
    )?;
    let program = compile(expression, options)?;
    Ok(Explanation {
        units: selected_units(&parsed, ast::Unit::Sessions),
        parsed,
        program,
    })
}

fn selected_units(query: &ast::Query, _implicit_unit: ast::Unit) -> Vec<ast::Unit> {
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

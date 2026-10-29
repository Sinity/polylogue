//! Pure query shape and source dependencies.
use crate::{Options, QueryError};
use serde::Serialize;
#[derive(Debug, Serialize)]
pub struct Explanation {
    pub parsed: crate::ast::Query,
    pub output: query_language::catalog::OutputSchema,
    pub definitions: Vec<crate::plan::ResolvedDefinition>,
}
pub fn query(text: &str, options: &Options) -> Result<Explanation, QueryError> {
    let plan = crate::compile(text, options)?;
    Ok(Explanation {
        parsed: plan.bound_query().query().clone(),
        output: options.output_schema(text)?,
        definitions: plan.definitions().to_vec(),
    })
}

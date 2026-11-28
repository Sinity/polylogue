//! Read continuations resume a selection by a row's order values, never by
//! a row number: a row added before the page start never shifts a page.
use super::*;

impl Reader<'_> {
    /// The order values of the row just before `at`; only order metadata is
    /// read, never an earlier body.
    pub(super) fn row_before(
        &self,
        selection: &PreparedSelection<'_>,
        at: &OrderPosition,
    ) -> Result<Option<Vec<serde_json::Value>>, SelectionError> {
        let program = &selection.program;
        if program.order.is_empty() {
            return Err(SelectionError::invalid(
                polylogue_model::Code::StaleCursor,
                "the selected row is no longer available; run the query again without its cursor",
            ));
        }
        if at.values.len() != program.order.len() {
            return Err(SelectionError::invalid(
                polylogue_model::Code::InvalidQuery,
                "selection order position has the wrong arity",
            ));
        }
        let mut params = values(&program.params);
        params.extend(at.values.iter().map(sql_value));
        params.push(SqlValue::Int(1));
        let mut found = None;
        let page = program.statements.page(
            &program.order,
            program.params.len(),
            Some(crate::query::compile::in_second_part(
                &program.order,
                &at.values,
            )),
            true,
            crate::query::compile::PageColumns::Order,
        );
        self.query_each(&page, &params, |row| {
            selection.scope.check()?;
            found = Some(
                program
                    .order
                    .iter()
                    .map(|key| row[&key.column].clone())
                    .collect(),
            );
            Ok::<_, SelectionError>(())
        })?;
        Ok(found)
    }
}

#[cfg(test)]
mod tests;

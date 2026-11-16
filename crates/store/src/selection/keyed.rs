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
    ) -> Result<Option<OrderPosition>, SelectionError> {
        let program = &selection.program;
        if program.order.is_empty() {
            return Err(SelectionError::invalid(
                polylogue_model::Code::StaleCursor,
                "the selected row position is unavailable",
            ));
        }
        if at.values.len() != program.order.len() {
            return Err(SelectionError::invalid(
                polylogue_model::Code::InvalidCursor,
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
            true,
            true,
            crate::query::compile::PageColumns::Order,
        );
        self.query_each(&page, &params, |row| {
            selection.scope.check()?;
            found = Some(OrderPosition {
                values: program
                    .order
                    .iter()
                    .map(|key| row[&key.column].clone())
                    .collect(),
                row_identity: String::new(),
            });
            Ok::<_, SelectionError>(())
        })?;
        Ok(found)
    }
}

#[cfg(test)]
mod tests;

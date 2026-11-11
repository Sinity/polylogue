//! Read continuations resume a selection by a row's order values, never by
//! a row number: a row added before the anchor never shifts a page.
use super::*;

impl Reader<'_> {
    /// Visit rows from the row at `from` (inclusive) in selection order, or
    /// from the first row when `from` is `None`.
    pub fn visit_selection_from_row<E>(
        &self,
        selection: &PreparedSelection<'_>,
        from: Option<&OrderPosition>,
        near: Option<&NearFn<'_>>,
        mut visit: impl FnMut(SelectionRow) -> Result<ControlFlow<()>, E>,
    ) -> Result<(), E>
    where
        E: From<SelectionError>,
    {
        self.in_selection_context(selection, near, || {
            // A direct reference selects one row; its continuation starts there.
            let after = match from {
                Some(at) if selection.direct.is_none() => {
                    self.row_before(selection, at).map_err(E::from)?
                }
                _ => None,
            };
            self.visit_selection_scoped(selection, after.as_ref(), &mut visit)
        })
    }

    /// Visit rows strictly before the row at `before`, in reverse order.
    pub fn visit_selection_before_row<E>(
        &self,
        selection: &PreparedSelection<'_>,
        before: &OrderPosition,
        near: Option<&NearFn<'_>>,
        mut visit: impl FnMut(SelectionRow) -> Result<ControlFlow<()>, E>,
    ) -> Result<(), E>
    where
        E: From<SelectionError>,
    {
        if selection.direct.is_some() {
            return Ok(());
        }
        self.in_selection_context(selection, near, || {
            self.visit_selection_scoped_in_direction(selection, Some(before), true, &mut visit)
        })
    }

    /// The order values of the row just before `at`; only order metadata is
    /// read, never an earlier body.
    fn row_before(
        &self,
        selection: &PreparedSelection<'_>,
        at: &OrderPosition,
    ) -> Result<Option<OrderPosition>, SelectionError> {
        let program = &selection.program;
        if program.order.is_empty() {
            return Err(invalid(
                "stale_cursor",
                "the selected row position is unavailable",
            ));
        }
        if at.values.len() != program.order.len() {
            return Err(invalid(
                "invalid_cursor",
                "selection order position has the wrong arity",
            ));
        }
        let mut params = values(&program.params);
        params.extend(at.values.iter().map(sql_value));
        params.push(SqlValue::Int(1));
        let mut found = None;
        self.query_each(&program.order_page_sql(true, true), &params, |row| {
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

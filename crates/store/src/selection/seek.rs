//! Compact read coordinates resolve to private keys in the prepared metadata order.
use super::*;

impl Reader<'_> {
    /// Resume a strict read after its execution, content and version bindings
    /// were authenticated. The row number is not an ordinary query page key.
    /// Seeking decodes only private order metadata before the requested row.
    pub fn visit_selection_from<E>(
        &self,
        selection: &PreparedSelection<'_>,
        start_row: u64,
        near: Option<&NearFn<'_>>,
        mut visit: impl FnMut(u64, SelectionRow) -> Result<ControlFlow<()>, E>,
    ) -> Result<(), E>
    where
        E: From<SelectionError>,
    {
        self.in_selection_context(selection, near, || {
            let after = start_row
                .checked_sub(1)
                .map(|ordinal| self.selection_order_at(selection, ordinal))
                .transpose()
                .map_err(E::from)?;
            let mut ordinal = start_row;
            self.visit_selection_scoped(selection, after.as_ref(), &mut |row| {
                let result = visit(ordinal, row)?;
                ordinal = ordinal
                    .checked_add(1)
                    .ok_or_else(|| E::from(invalid("store", "selected row ordinal overflow")))?;
                Ok(result)
            })
        })
    }

    /// Visit rows strictly before `end_row`, in reverse prepared order. The
    /// anchor row's record/item boundary is handled by the read layout caller.
    /// Only private order metadata is inspected while resolving the anchor.
    pub fn visit_selection_before<E>(
        &self,
        selection: &PreparedSelection<'_>,
        end_row: u64,
        near: Option<&NearFn<'_>>,
        mut visit: impl FnMut(u64, SelectionRow) -> Result<ControlFlow<()>, E>,
    ) -> Result<(), E>
    where
        E: From<SelectionError>,
    {
        self.in_selection_context(selection, near, || {
            if end_row == 0 {
                return Ok(());
            }
            let before = self
                .selection_order_at(selection, end_row)
                .map_err(E::from)?;
            let mut ordinal = end_row;
            self.visit_selection_scoped_in_direction(selection, Some(&before), true, &mut |row| {
                ordinal = ordinal
                    .checked_sub(1)
                    .ok_or_else(|| E::from(invalid("store", "selected row ordinal underflow")))?;
                visit(ordinal, row)
            })
        })
    }

    fn selection_order_at(
        &self,
        selection: &PreparedSelection<'_>,
        ordinal: u64,
    ) -> Result<OrderPosition, SelectionError> {
        let program = &selection.program;
        if program.order.is_empty() || selection.direct.is_some() {
            return Err(invalid(
                "stale_cursor",
                "the selected row position is unavailable",
            ));
        }
        let ordinal = i64::try_from(ordinal)
            .map_err(|_| invalid("stale_cursor", "the selected row position is unavailable"))?;
        let projection = program
            .order
            .iter()
            .map(|key| format!("q.\"{}\"", key.column.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(",");
        let order = program
            .order
            .iter()
            .map(|key| {
                format!(
                    "q.\"{}\"{}",
                    key.column.replace('"', "\"\""),
                    if key.descending { " DESC" } else { "" }
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let mut params = values(&program.params);
        params.push(SqlValue::Int(-1));
        params.push(SqlValue::Int(ordinal));
        let sql = format!(
            "SELECT {projection} FROM ({}) q ORDER BY {order} LIMIT 1 OFFSET ?{}",
            program.page_sql(false),
            params.len()
        );
        let mut found = None;
        self.query_each(&sql, &params, |row| {
            selection.scope.check()?;
            found = Some(OrderPosition {
                values: program
                    .order
                    .iter()
                    .map(|key| row[&key.column].clone())
                    .collect(),
                // The execution locator carries no native identity. Keyset
                // comparison uses the private values resolved above.
                row_identity: String::new(),
            });
            Ok::<_, SelectionError>(())
        })?;
        found.ok_or_else(|| invalid("stale_cursor", "the selected row position is unavailable"))
    }
}

#[cfg(test)]
mod tests;

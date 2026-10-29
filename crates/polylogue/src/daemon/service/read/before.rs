//! Descending metadata coordinates hydrate only the adjacent canonical record.
use super::*;
use polylogue_store::selection::SelectionRow;

pub(super) fn visit(
    reader: &Reader<'_>,
    selection: &PreparedSelection,
    options: &ReadOptions,
    anchor: ReadPosition,
    near: Option<&polylogue_store::selection::NearFn<'_>>,
    footer: &mut ReadFooter,
    emit: &mut dyn FnMut(ReadRecord, SelectionLocator) -> Result<bool, Response>,
) -> Result<(), Response> {
    let mut current = None;
    reader.visit_selection_from::<Response>(
        selection,
        anchor.selection.row,
        near,
        |ordinal, row| {
            if ordinal != anchor.selection.row {
                return Err(error("stale_cursor", "the backward row coordinate changed"));
            }
            current = Some(row);
            Ok(ControlFlow::Break(()))
        },
    )?;
    let current =
        current.ok_or_else(|| error("stale_cursor", "the backward row is no longer selected"))?;
    let mut walk = Walk {
        reader,
        selection,
        options,
        footer,
        emit,
    };
    if !walk.row(anchor.selection.row, current, Some(anchor.selection.record))? {
        return Ok(());
    }
    reader.visit_selection_before::<Response>(
        selection,
        anchor.selection.row,
        near,
        |ordinal, row| {
            Ok(if walk.row(ordinal, row, None)? {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            })
        },
    )?;
    Ok(())
}

struct Walk<'a, 'pool, 'read> {
    reader: &'a Reader<'pool>,
    selection: &'a PreparedSelection<'read>,
    options: &'a ReadOptions,
    footer: &'a mut ReadFooter,
    emit: &'a mut dyn FnMut(ReadRecord, SelectionLocator) -> Result<bool, Response>,
}
impl Walk<'_, '_, '_> {
    fn row(
        &mut self,
        ordinal: u64,
        row: SelectionRow,
        start: Option<ReadRecordLocator>,
    ) -> Result<bool, Response> {
        let Some(reference) = row.reference else {
            let record = ReadRecordLocator {
                kind: ReadRecordKind::Presentation,
                ordinal: 0,
            };
            if start.is_some_and(|start| start != record) {
                return Err(error("stale_cursor", "the backward result row changed"));
            }
            return (self.emit)(
                presentation(self.selection, &row.values),
                SelectionLocator {
                    row: ordinal,
                    record,
                },
            );
        };
        let mut current = match start {
            Some(start) => Some(start),
            None => backward::last_record(self.reader, &reference, self.options.inherited)?,
        };
        let mut required_anchor = start.is_some();
        while let Some(record_position) = current {
            let mut observed = false;
            let mut keep = true;
            records::reference(
                self.reader,
                &reference,
                self.options.inherited,
                Some(&record_position),
                &mut |record, evidence, actual| {
                    if observed || actual != record_position {
                        return Err(error(
                            "stale_cursor",
                            "the backward record coordinate changed",
                        ));
                    }
                    observed = true;
                    let projected = output::project(
                        record,
                        self.options,
                        &evidence,
                        self.selection.direct_reference().is_none(),
                        self.footer,
                    )?;
                    if let Some(record) = projected {
                        keep = (self.emit)(
                            record,
                            SelectionLocator {
                                row: ordinal,
                                record: actual,
                            },
                        )?;
                    } else if required_anchor {
                        return Err(error(
                            "stale_cursor",
                            "the backward anchor is no longer visible",
                        ));
                    }
                    // The metadata adapter names exactly one record, never its suffix.
                    Ok(false)
                },
            )?;
            if !observed {
                return Err(error(
                    "stale_cursor",
                    "the backward record is no longer selected",
                ));
            }
            if !keep {
                return Ok(false);
            }
            current = backward::previous_record(
                self.reader,
                &reference,
                self.options.inherited,
                &record_position,
            )?;
            required_anchor = false;
        }
        Ok(true)
    }
}

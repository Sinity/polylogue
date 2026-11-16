//! Descending metadata coordinates hydrate only the adjacent source record.
use super::*;
use polylogue_store::selection::SelectionRow;

#[allow(clippy::too_many_arguments)]
pub(super) fn visit(
    reader: &Reader<'_>,
    selection: &PreparedSelection,
    rendering: &Rendering<'_>,
    page_start: ItemPlace,
    start_row: Option<&OrderPosition>,
    rows: &mut RowKeys,
    footer: &mut ReadFooter,
    emit: &mut Emit<'_>,
) -> Result<(), Failure> {
    let near = rendering.near;
    let mut current = None;
    reader.visit_selection_from_row::<Failure>(selection, start_row, near, |row| {
        current = Some(row);
        Ok(ControlFlow::Break(()))
    })?;
    let current = current
        .ok_or_else(|| Failure::new("stale_cursor", "the backward row is no longer selected"))?;
    if start_row.is_some_and(|page_start| page_start.values != current.position.values) {
        return Err(Failure::new(
            "stale_cursor",
            "the backward row is no longer selected",
        ));
    }
    let start_row_position = current.position.clone();
    rows.keys
        .insert(page_start.selection.row, start_row_position.clone());
    let mut walk = Walk {
        reader,
        selection,
        rendering,
        footer,
        emit,
    };
    if !walk.row(page_start.selection.row, current, Some(page_start), near)? {
        return Ok(());
    }
    let mut ordinal = page_start.selection.row;
    reader.visit_selection_before_row::<Failure>(selection, &start_row_position, near, |row| {
        ordinal = ordinal
            .checked_sub(1)
            .ok_or_else(|| Failure::new("stale_cursor", "the backward row number underflowed"))?;
        rows.keys.insert(ordinal, row.position.clone());
        Ok(if walk.row(ordinal, row, None, near)? {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        })
    })?;
    Ok(())
}

struct Walk<'a, 'pool, 'read, 'emit> {
    reader: &'a Reader<'pool>,
    selection: &'a PreparedSelection<'read>,
    rendering: &'a Rendering<'a>,
    footer: &'a mut ReadFooter,
    emit: &'a mut Emit<'emit>,
}
impl Walk<'_, '_, '_, '_> {
    fn row(
        &mut self,
        ordinal: u64,
        row: SelectionRow,
        position: Option<ItemPlace>,
        near: Option<&polylogue_store::selection::NearFn<'_>>,
    ) -> Result<bool, Failure> {
        let start = position.map(|position| position.selection.record);
        let start_item = position.map(|position| position.item);
        let Some(target) = row.target else {
            let record = ReadRecordLocator {
                kind: ReadRecordKind::Presentation,
                ordinal: 0,
            };
            if start.is_some_and(|start| start != record) {
                return Err(Failure::new(
                    "stale_cursor",
                    "the backward result row changed",
                ));
            }
            return (self.emit)(
                &presentation(self.selection, &row.values),
                SelectionLocator {
                    row: ordinal,
                    record,
                },
                Some(0),
            );
        };
        if let ReadTarget::Document(id) = &target {
            let selected = |_: &Source| Ok(false);
            let facts = LayoutFacts::default();
            let layout = layout_facts::Layout {
                options: self.rendering.options,
                facts: &facts,
                selected: &selected,
                ready_attachment: self.rendering.ready_attachment,
                backwards: true,
                page_start: None,
            };
            if let Some(record) = output::document_omission(
                &target.to_string(),
                self.rendering.options,
                self.selection.direct_target().is_none(),
                self.footer,
            )? {
                let at = SelectionLocator {
                    row: ordinal,
                    record: ReadRecordLocator {
                        kind: ReadRecordKind::Presentation,
                        ordinal: 0,
                    },
                };
                if position.is_some_and(|position| position.selection != at || position.item != 0) {
                    return Err(Failure::new(
                        "stale_cursor",
                        "the backward document omission changed",
                    ));
                }
                return (self.emit)(&record, at, Some(0));
            }
            return documents::visit(self.reader, id, ordinal, position, &layout, self.emit);
        }
        let mut current = match start {
            Some(start) => Some(start),
            None => backward::last_record(self.reader, &target, self.rendering.options.inherited)?,
        };
        let mut start_required = start.is_some();
        while let Some(record_position) = current {
            let mut observed = false;
            let mut keep = true;
            records::reference(
                self.reader,
                &target,
                self.rendering.options.inherited,
                Some(&record_position),
                &mut |record, facts, actual| {
                    if observed || actual != record_position {
                        return Err(Failure::new(
                            "stale_cursor",
                            "the backward record coordinate changed",
                        ));
                    }
                    observed = true;
                    let projected = output::project(
                        record,
                        self.rendering.options,
                        &facts,
                        self.selection.direct_target().is_none(),
                        self.footer,
                    )?;
                    if let Some(record) = projected {
                        let at = SelectionLocator {
                            row: ordinal,
                            record: actual,
                        };
                        let selected = |source: &Source| {
                            selected_source(
                                self.reader,
                                self.selection,
                                Some(&target),
                                self.rendering.options,
                                near,
                                source,
                            )
                        };
                        let layout = layout_facts::Layout {
                            options: self.rendering.options,
                            facts: &facts,
                            selected: &selected,
                            ready_attachment: self.rendering.ready_attachment,
                            backwards: true,
                            page_start: start_required.then_some(start_item).flatten(),
                        };
                        keep = output::emit_record(self.reader, &record, at, &layout, self.emit)?;
                    } else if start_required {
                        return Err(Failure::new(
                            "stale_cursor",
                            "the backward page start is no longer visible",
                        ));
                    }
                    // The metadata adapter names exactly one record, never its suffix.
                    Ok(false)
                },
            )?;
            if !observed {
                return Err(Failure::new(
                    "stale_cursor",
                    "the backward record is no longer selected",
                ));
            }
            if !keep {
                return Ok(false);
            }
            current = backward::previous_record(
                self.reader,
                &target,
                self.rendering.options.inherited,
                &record_position,
            )?;
            start_required = false;
        }
        Ok(true)
    }
}

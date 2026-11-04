//! Descending metadata coordinates hydrate only the adjacent canonical record.
use super::*;
use polylogue_store::selection::SelectionRow;

#[allow(clippy::too_many_arguments)]
pub(super) fn visit(
    reader: &Reader<'_>,
    selection: &PreparedSelection,
    rendering: &Rendering<'_>,
    anchor: ItemPlace,
    anchor_row: Option<&OrderPosition>,
    rows: &mut RowKeys,
    footer: &mut ReadFooter,
    emit: &mut Emit<'_>,
) -> Result<(), Failure> {
    let near = rendering.near;
    let mut current = None;
    reader.visit_selection_from_row::<Failure>(selection, anchor_row, near, |row| {
        current = Some(row);
        Ok(ControlFlow::Break(()))
    })?;
    let current = current
        .ok_or_else(|| Failure::new("stale_cursor", "the backward row is no longer selected"))?;
    if anchor_row.is_some_and(|anchor| anchor.values != current.position.values) {
        return Err(Failure::new(
            "stale_cursor",
            "the backward row is no longer selected",
        ));
    }
    let anchor_position = current.position.clone();
    rows.keys
        .insert(anchor.selection.row, anchor_position.clone());
    let mut walk = Walk {
        reader,
        selection,
        rendering,
        footer,
        emit,
    };
    if !walk.row(anchor.selection.row, current, Some(anchor), near)? {
        return Ok(());
    }
    let mut ordinal = anchor.selection.row;
    reader.visit_selection_before_row::<Failure>(selection, &anchor_position, near, |row| {
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
        let anchor_item = position.map(|position| position.item);
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
            let evidence = ProjectionEvidence::default();
            let layout = evidence::Layout {
                options: self.rendering.options,
                evidence: &evidence,
                selected: &selected,
                ready_attachment: self.rendering.ready_attachment,
                backwards: true,
                anchor: None,
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
        let mut required_anchor = start.is_some();
        while let Some(record_position) = current {
            let mut observed = false;
            let mut keep = true;
            records::reference(
                self.reader,
                &target,
                self.rendering.options.inherited,
                Some(&record_position),
                &mut |record, evidence, actual| {
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
                        &evidence,
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
                        let layout = evidence::Layout {
                            options: self.rendering.options,
                            evidence: &evidence,
                            selected: &selected,
                            ready_attachment: self.rendering.ready_attachment,
                            backwards: true,
                            anchor: required_anchor.then_some(anchor_item).flatten(),
                        };
                        keep = output::emit_record(self.reader, &record, at, &layout, self.emit)?;
                    } else if required_anchor {
                        return Err(Failure::new(
                            "stale_cursor",
                            "the backward anchor is no longer visible",
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
            required_anchor = false;
        }
        Ok(true)
    }
}

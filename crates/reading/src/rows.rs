//! One row walk in either direction, with record coordinates independent of layout.
use super::*;
use polylogue_store::selection::SelectionRow;

pub(crate) enum Rows<'a> {
    Selection(&'a PreparedSelection<'a>),
    SavedQuery(
        &'a PreparedSelection<'a>,
        &'a crate::saved_query::SavedQuery,
    ),
    Object(&'a ReadRecord, ReadRecordKind),
}

impl Rows<'_> {
    pub(crate) fn history(&self) -> Option<&crate::saved_query::SavedQuery> {
        match self {
            Self::SavedQuery(_, history) => Some(history),
            _ => None,
        }
    }
}

pub(crate) struct RowStart {
    pub row: Option<OrderPosition>,
    pub place: Option<ItemPlace>,
    pub direction: ReadDirection,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn visit_rows(
    reader: &Reader<'_>,
    rows: &Rows<'_>,
    rendering: &Rendering<'_>,
    start: &RowStart,
    keys: &mut RowKeys,
    footer: &mut ReadFooter,
    emit: &mut Emit<'_>,
) -> Result<(), Error> {
    if start.direction == ReadDirection::Past {
        return Err(Error::invalid_query(
            "a row reading continues forward or backward only",
        ));
    }
    let backwards = start.direction == ReadDirection::Before;
    let (selection, history) = match rows {
        Rows::Selection(selection) => (*selection, None),
        Rows::SavedQuery(selection, history) => (*selection, Some(*history)),
        Rows::Object(record, kind) => {
            let at = SelectionLocator {
                row: start.place.map_or(0, |place| place.selection.row),
                record: ReadRecordLocator {
                    kind: *kind,
                    ordinal: 0,
                },
            };
            if start.place.is_some_and(|place| place.selection != at) {
                return Err(Error::stale_cursor(
                    "the stored record this page continues from changed",
                ));
            }
            if !rendering.page {
                emit(record, at, None)?;
                return Ok(());
            }
            let selected = |_: &Source| Ok(false);
            let empty_facts = LayoutFacts::default();
            let layout = layout_facts::Layout {
                options: rendering.options,
                facts: rendering.object_facts.unwrap_or(&empty_facts),
                selected: &selected,
                ready_attachment: rendering.ready_attachment,
                per_item_source: false,
                backwards,
                page_start: start.place.map(|place| place.item),
            };
            output::emit_record(reader, record, at, &layout, emit)?;
            return Ok(());
        }
    };
    let prefix_start = history.is_some() && start.row.is_none();
    if prefix_start {
        let history = history.expect("a saved-query prefix");
        let row = start.place.map_or(0, |place| place.selection.row);
        if !visit_history(
            reader,
            history,
            rendering,
            start.place,
            row,
            backwards,
            emit,
        )? || backwards
        {
            return Ok(());
        }
    }
    let data_start = RowStart {
        row: start.row.clone(),
        place: start.place.filter(|_| !prefix_start),
        direction: start.direction,
    };
    let start = &data_start;
    let mut walk = Walk {
        reader,
        selection,
        rendering,
        footer,
        emit,
    };
    let mut first = true;
    let mut ordinal = start
        .place
        .map_or(u64::from(prefix_start), |place| place.selection.row);
    if backwards {
        let mut current = None;
        reader.visit_selection::<Error>(
            selection,
            start.row.as_ref().map_or(
                polylogue_store::read::VisitFrom::First,
                polylogue_store::read::VisitFrom::At,
            ),
            rendering.near,
            |row| {
                current = Some(row);
                Ok(ControlFlow::Break(()))
            },
        )?;
        let row = current.ok_or_else(|| missing_row(rendering.target))?;
        if start
            .row
            .as_ref()
            .is_some_and(|at| at.values != row.position.values)
        {
            return Err(missing_row(rendering.target));
        }
        let row_position = row.position.clone();
        keys.keys.insert(ordinal, row_position.clone());
        if !walk.visit_row(ordinal, row, start.place, start.direction)? {
            return Ok(());
        }
        let mut full = false;
        reader.visit_selection::<Error>(
            selection,
            polylogue_store::read::VisitFrom::Before(&row_position),
            rendering.near,
            |row| {
                ordinal = ordinal.checked_sub(1).ok_or_else(|| {
                    Error::Internal(
                        "a backward page counted more than 2^32 rows before its start".into(),
                    )
                })?;
                keys.keys.insert(ordinal, row.position.clone());
                Ok(if walk.visit_row(ordinal, row, None, start.direction)? {
                    ControlFlow::Continue(())
                } else {
                    full = true;
                    ControlFlow::Break(())
                })
            },
        )?;
        if !full && let Some(history) = history {
            let row = ordinal.checked_sub(1).ok_or_else(|| {
                Error::internal("saved history lies before the row coordinate range")
            })?;
            visit_history(reader, history, rendering, None, row, true, emit)?;
        }
    } else {
        reader.visit_selection::<Error>(
            selection,
            start.row.as_ref().map_or(
                polylogue_store::read::VisitFrom::First,
                polylogue_store::read::VisitFrom::At,
            ),
            rendering.near,
            |row| {
                if first
                    && start
                        .row
                        .as_ref()
                        .is_some_and(|at| at.values != row.position.values)
                {
                    return Err(missing_row(rendering.target));
                }
                let place = if first { start.place } else { None };
                first = false;
                keys.keys.insert(ordinal, row.position.clone());
                let keep = walk.visit_row(ordinal, row, place, start.direction)?;
                ordinal = ordinal
                    .checked_add(1)
                    .ok_or_else(|| Error::Internal("a page counted more than 2^64 rows".into()))?;
                Ok(if keep {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                })
            },
        )?;
        if first && start.place.is_some() {
            return Err(missing_row(rendering.target));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn visit_history(
    reader: &Reader<'_>,
    history: &crate::saved_query::SavedQuery,
    rendering: &Rendering<'_>,
    from: Option<ItemPlace>,
    row: u64,
    backwards: bool,
    emit: &mut Emit<'_>,
) -> Result<bool, Error> {
    let seed = rendering
        .history_seed
        .ok_or_else(|| Error::internal("saved-query history has no reading identity"))?;
    let first = from.map_or_else(
        || {
            if backwards {
                history.item_count() - 1
            } else {
                0
            }
        },
        |place| place.selection.record.ordinal as usize,
    );
    if first >= history.item_count()
        || from.is_some_and(|place| place.selection.record.kind != ReadRecordKind::Presentation)
    {
        return Err(Error::stale_cursor(
            "the saved-query history position changed",
        ));
    }
    let selected = |_: &Source| Ok(false);
    let facts = LayoutFacts::default();
    let mut index = first;
    loop {
        reader.check()?;
        let record = ReadRecord::Presentation {
            parts: vec![ReadPart::Item(history.item(index, seed)?)],
        };
        let at = SelectionLocator {
            row,
            record: ReadRecordLocator {
                kind: ReadRecordKind::Presentation,
                ordinal: index as u64,
            },
        };
        let keep = if rendering.page {
            let layout = layout_facts::Layout {
                options: rendering.options,
                facts: &facts,
                selected: &selected,
                ready_attachment: rendering.ready_attachment,
                per_item_source: false,
                backwards,
                page_start: from.filter(|_| index == first).map(|place| place.item),
            };
            output::emit_record(reader, &record, at, &layout, emit)?
        } else {
            emit(&record, at, None)?
        };
        if !keep {
            return Ok(false);
        }
        if backwards {
            let Some(previous) = index.checked_sub(1) else {
                break;
            };
            index = previous;
        } else {
            index += 1;
            if index == history.item_count() {
                break;
            }
        }
    }
    Ok(true)
}

fn missing_row(target: &str) -> Error {
    Error::stale_cursor(format!(
        "the record this page continues from no longer matches `{}`",
        target
    ))
}

struct Walk<'a, 'pool, 'read, 'emit> {
    reader: &'a Reader<'pool>,
    selection: &'a PreparedSelection<'read>,
    rendering: &'a Rendering<'a>,
    footer: &'a mut ReadFooter,
    emit: &'a mut Emit<'emit>,
}
impl Walk<'_, '_, '_, '_> {
    fn visit_row(
        &mut self,
        ordinal: u64,
        row: SelectionRow,
        place: Option<ItemPlace>,
        direction: ReadDirection,
    ) -> Result<bool, Error> {
        let backwards = direction == ReadDirection::Before;
        let start = place.map(|place| place.selection.record);
        let Some(target) = row.target else {
            let at = SelectionLocator {
                row: ordinal,
                record: ReadRecordLocator {
                    kind: ReadRecordKind::Presentation,
                    ordinal: 0,
                },
            };
            if place.is_some_and(|place| place.selection != at) {
                return Err(Error::stale_cursor("a result row on this page changed"));
            }
            return (self.emit)(&presentation(self.selection, &row.values), at, Some(0));
        };
        let options = self.rendering.options;
        let mixed = self.selection.direct_target().is_none();
        if self.rendering.page
            && let ReadTarget::Document(id) = &target
        {
            let selected = |_: &Source| Ok(false);
            let facts = LayoutFacts::default();
            let layout = layout_facts::Layout {
                options,
                facts: &facts,
                selected: &selected,
                ready_attachment: self.rendering.ready_attachment,
                per_item_source: false,
                backwards,
                page_start: None,
            };
            if let Some(record) =
                output::document_omission(&target.to_string(), options, mixed, self.footer)?
            {
                let at = SelectionLocator {
                    row: ordinal,
                    record: ReadRecordLocator {
                        kind: ReadRecordKind::Presentation,
                        ordinal: 0,
                    },
                };
                if place.is_some_and(|place| place.selection != at || place.item != 0) {
                    return Err(Error::stale_cursor(
                        "the document omitted on this page changed",
                    ));
                }
                return (self.emit)(&record, at, Some(0));
            }
            return documents::visit(self.reader, id, ordinal, place, &layout, self.emit);
        }
        let reader = self.reader;
        let selection = self.selection;
        let rendering = self.rendering;
        let footer = &mut self.footer;
        let emit = &mut self.emit;
        let mut pending = start;
        let mut render = |record: records::SelectedRecord, mut facts: LayoutFacts, position| {
            if let Some(expected) = pending.take()
                && expected != position
            {
                return Err(Error::stale_cursor(
                    "a record on this page changed since it was issued",
                ));
            }
            if options.view == View::Live
                && let records::SelectedRecord::Original(original) = &record
                && let polylogue_model::bundle::Record::Session(row) = original.as_ref()
            {
                facts.cards.push(live::card(
                    reader,
                    &row.session.id.to_string(),
                    selection.cursor_context().clock.now.as_millisecond(),
                    selection.cursor_context().clock.day.zone(),
                )?);
            }
            if options.annotations()
                && let records::SelectedRecord::Original(original) = &record
            {
                layout_facts::annotate(
                    reader,
                    original,
                    &mut facts,
                    rendering.at,
                    options.annotation_at(),
                )?;
            }
            polylogue_store::read::limit::ReadLimit::set_stage("read layout");
            let Some(record) = output::project(record, options, &facts, mixed, footer)? else {
                if place.is_some_and(|place| place.selection.record == position) {
                    return Err(Error::stale_cursor(
                        "the record this page continues from is no longer shown",
                    ));
                }
                return Ok(true);
            };
            let at = SelectionLocator {
                row: ordinal,
                record: position,
            };
            let selected = |source: &Source| {
                selected_source(
                    reader,
                    selection,
                    Some(&target),
                    options,
                    rendering.near,
                    source,
                )
            };
            let layout = layout_facts::Layout {
                options,
                facts: &facts,
                selected: &selected,
                ready_attachment: rendering.ready_attachment,
                per_item_source: false,
                backwards,
                page_start: place
                    .filter(|place| place.selection == at)
                    .map(|place| place.item),
            };
            output::emit_record(reader, &record, at, &layout, *emit)
        };
        if !backwards {
            let keep = records::reference(
                reader,
                &target,
                options.inherited(),
                start.as_ref(),
                &mut render,
            )?;
            if pending.is_some() {
                return Err(missing_row(rendering.target));
            }
            return Ok(keep);
        }
        let mut current = match start {
            Some(start) => Some(start),
            None => records::step(reader, &target, options.inherited(), None, direction)?,
        };
        while let Some(position) = current {
            let mut observed = false;
            let mut keep = true;
            records::reference(
                reader,
                &target,
                options.inherited(),
                Some(&position),
                &mut |record, facts, actual| {
                    if observed || actual != position {
                        return Err(Error::stale_cursor(
                            "a record on this page changed since it was issued",
                        ));
                    }
                    observed = true;
                    keep = render(record, facts, actual)?;
                    Ok(false)
                },
            )?;
            if !observed {
                return Err(missing_row(rendering.target));
            }
            if !keep {
                return Ok(false);
            }
            current = records::step(
                reader,
                &target,
                options.inherited(),
                Some(&position),
                direction,
            )?;
        }
        Ok(true)
    }
}

fn presentation(selection: &PreparedSelection<'_>, values: &serde_json::Value) -> ReadRecord {
    ReadRecord::Presentation {
        parts: vec![ReadPart::Item(ReadItem {
            source: Source::record(format!("query:{}", selection.question_id())),
            kind: ItemKind::Card {
                card: Box::new(Card::Fields {
                    title: "Query result".into(),
                    figures: figures(values),
                }),
            },
        })],
    }
}

/// One figure per column of a query row. A SQL NULL is an unknown value, not
/// a known `null`.
fn figures(values: &serde_json::Value) -> Vec<Figure> {
    values
        .as_object()
        .into_iter()
        .flat_map(|object| object.iter())
        .map(|(name, value)| Figure {
            name: name.clone(),
            value: if value.is_null() {
                ValueState::Unknown { reason: None }
            } else {
                ValueState::known(value.clone())
            },
            unit: None,
        })
        .collect()
}

#[cfg(test)]
mod tests;

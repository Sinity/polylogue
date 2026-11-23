//! Page presentation seeks original document blocks without making a partial
//! stored Document. Full record transfer keeps the strict record adapter.
use super::{Emit, Error, attachment_facts, layout_facts};
use polylogue_model::{
    Block, DocumentId,
    read::{
        ItemPlace, ReadItem, ReadPart, ReadRecord, ReadRecordKind, ReadRecordLocator,
        SelectionLocator, Source, SourceBlock, View,
    },
};
use polylogue_store::Reader;
use std::ops::ControlFlow;

fn stale() -> Error {
    Error::stale_cursor("the block this page continues from is no longer in this document")
}

fn check(reader: &Reader<'_>) -> Result<(), Error> {
    reader.check()?;
    if let Some(limit) = polylogue_store::read::limit::ReadLimit::current() {
        limit.check()?;
    }
    Ok(())
}

fn position(row: u64, ordinal: u64) -> SelectionLocator {
    SelectionLocator {
        row,
        record: ReadRecordLocator {
            kind: ReadRecordKind::Presentation,
            ordinal,
        },
    }
}

fn emit_items(
    reader: &Reader<'_>,
    row: u64,
    ordinal: u64,
    items: Vec<ReadItem>,
    start: Option<ItemPlace>,
    layout: &layout_facts::Layout<'_>,
    emit: &mut Emit<'_>,
) -> Result<bool, Error> {
    let at = position(row, ordinal);
    let page_start = start.filter(|start| start.selection == at);
    if page_start.is_some_and(|start| {
        usize::try_from(start.item)
            .ok()
            .is_none_or(|item| item >= items.len())
    }) {
        return Err(stale());
    }
    let mut items: Vec<_> = items.into_iter().enumerate().collect();
    if layout.backwards {
        items.reverse();
    }
    for (item, value) in items {
        let item = u64::try_from(item)
            .map_err(|_| Error::internal("a document has more items than the supported count"))?;
        if page_start.is_some_and(|page_start| {
            if layout.backwards {
                item > page_start.item
            } else {
                item < page_start.item
            }
        }) {
            continue;
        }
        check(reader)?;
        if !emit(
            &ReadRecord::Presentation {
                parts: vec![ReadPart::Item(value)],
            },
            at,
            Some(item),
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn block_items(
    reader: &Reader<'_>,
    reference: &str,
    ordinal: u64,
    block: Block,
    layout: &layout_facts::Layout<'_>,
) -> Result<Vec<ReadItem>, Error> {
    check(reader)?;
    let source = Source {
        block: Some(ordinal),
        ..Source::record(reference)
    };
    let mut items = Vec::new();
    polylogue_render::visit_source_block(
        &SourceBlock { source, block },
        layout.options,
        layout.facts,
        |item| {
            items.push(item);
            true
        },
    )
    .map_err(Error::from)?;
    let mut attachment = 0u64;
    for item in &mut items {
        if matches!(
            item.kind,
            polylogue_model::read::ItemKind::Attachment { .. }
        ) {
            *item = attachment_facts::prepare(
                reader,
                item.clone(),
                attachment,
                layout.ready_attachment,
            )?;
            attachment = attachment.checked_add(1).ok_or_else(|| {
                Error::internal("a block has more attachments than the supported count")
            })?;
        }
    }
    Ok(items)
}

pub(super) fn visit(
    reader: &Reader<'_>,
    id: &DocumentId,
    row: u64,
    start: Option<ItemPlace>,
    layout: &layout_facts::Layout<'_>,
    emit: &mut Emit<'_>,
) -> Result<bool, Error> {
    check(reader)?;
    let selected = reader
        .select_document(id)?
        .ok_or_else(|| Error::not_found(format!("no document {id}")))?;
    let reference = format!("document:{id}");
    if reader.document_commit(id)?.is_none() {
        return Err(Error::not_yet_available(format!(
            "{reference} has no stored commit position for its source"
        )));
    }
    if start.is_some_and(|start| {
        start.selection.row != row || start.selection.record.kind != ReadRecordKind::Presentation
    }) {
        return Err(stale());
    }
    let from = start.map(|start| start.selection.record.ordinal);
    if let Some(ordinal) = from.filter(|ordinal| *ordinal > 0)
        && !selected.has_block(ordinal - 1)?
    {
        return Err(stale());
    }
    if matches!(
        layout.options.view,
        View::Summary | View::Links | View::Changes | View::Live
    ) {
        return if start.is_some() {
            Err(stale())
        } else {
            Ok(true)
        };
    }
    let root = || -> Result<Vec<ReadItem>, Error> {
        let mut items = Vec::new();
        polylogue_render::visit_document_root(&reference, layout.options, layout.facts, |item| {
            items.push(item);
            true
        })
        .map_err(Error::from)?;
        Ok(items)
    };
    if layout.options.view == View::Raw {
        if from.is_some_and(|ordinal| ordinal != 0) {
            return Err(stale());
        }
        return emit_items(reader, row, 0, root()?, start, layout, emit);
    }
    if !layout.backwards
        && from.is_none_or(|ordinal| ordinal == 0)
        && !emit_items(reader, row, 0, root()?, start, layout, emit)?
    {
        return Ok(false);
    }
    if layout.backwards && from == Some(0) {
        return emit_items(reader, row, 0, root()?, start, layout, emit);
    }
    let mut failure = None;
    let mut continuing = true;
    let mut visit_block = |ordinal: u64, block| {
        let result = (|| {
            let locator = ordinal.checked_add(1).ok_or_else(stale)?;
            emit_items(
                reader,
                row,
                locator,
                block_items(reader, &reference, ordinal, block, layout)?,
                start,
                layout,
                emit,
            )
        })();
        match result {
            Ok(keep) => {
                continuing = keep;
                if keep {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                }
            }
            Err(error) => {
                failure = Some(error);
                ControlFlow::Break(())
            }
        }
    };
    let result = if layout.backwards {
        selected.visit_blocks(
            from.map_or(
                polylogue_store::read::VisitFrom::Last,
                polylogue_store::read::VisitFrom::Before,
            ),
            &mut visit_block,
        )
    } else {
        selected.visit_blocks(
            polylogue_store::read::VisitFrom::At(
                from.map_or(0, |ordinal| ordinal.saturating_sub(1)),
            ),
            &mut visit_block,
        )
    };
    check(reader)?;
    result?;
    if let Some(error) = failure {
        return Err(error);
    }
    if !continuing {
        return Ok(false);
    }
    if layout.backwards {
        return emit_items(reader, row, 0, root()?, None, layout, emit);
    }
    Ok(true)
}

#[cfg(test)]
mod tests;

//! Page presentation seeks original document blocks without making a partial
//! canonical Document. Full record transfer retains the strict record adapter.
use super::{Emit, Failure, attachment_evidence, evidence, output};
use polylogue_model::{
    Block, DocumentId,
    read::{
        ItemPlace, ReadItem, ReadPart, ReadRecord, ReadRecordKind, ReadRecordLocator,
        SelectionLocator, Source, SourceBlock, View,
    },
};
use polylogue_store::Reader;
use std::ops::ControlFlow;

fn stale() -> Failure {
    Failure::new(
        "stale_cursor",
        "the continued document block is no longer selected",
    )
}

fn check(reader: &Reader<'_>) -> Result<(), Failure> {
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
    layout: &evidence::Layout<'_>,
    emit: &mut Emit<'_>,
) -> Result<bool, Failure> {
    let at = position(row, ordinal);
    let anchor = start.filter(|start| start.selection == at);
    if anchor.is_some_and(|start| {
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
        let item = u64::try_from(item).map_err(|_| stale())?;
        if anchor.is_some_and(|anchor| {
            if layout.backwards {
                item > anchor.item
            } else {
                item < anchor.item
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
    layout: &evidence::Layout<'_>,
) -> Result<Vec<ReadItem>, Failure> {
    check(reader)?;
    let source = Source {
        block: Some(ordinal),
        ..Source::record(reference)
    };
    let mut items = Vec::new();
    polylogue_render::visit_source_block(
        &SourceBlock { source, block },
        layout.options,
        layout.evidence,
        |item| {
            items.push(item);
            true
        },
    )
    .map_err(output::read_error)?;
    let mut attachment = 0u64;
    for item in &mut items {
        if matches!(
            item.kind,
            polylogue_model::read::ItemKind::Attachment { .. }
        ) {
            *item = attachment_evidence::prepare(
                reader,
                item.clone(),
                attachment,
                layout.ready_attachment,
            )?;
            attachment = attachment.checked_add(1).ok_or_else(stale)?;
        }
    }
    Ok(items)
}

pub(super) fn visit(
    reader: &Reader<'_>,
    id: &DocumentId,
    row: u64,
    start: Option<ItemPlace>,
    layout: &evidence::Layout<'_>,
    emit: &mut Emit<'_>,
) -> Result<bool, Failure> {
    check(reader)?;
    let selected = reader
        .select_document(id)?
        .ok_or_else(|| Failure::new("not_found", format!("no document {id}")))?;
    let reference = format!("document:{id}");
    if reader.document_commit(id)?.is_none() {
        return Err(Failure::new(
            "not_yet_available",
            format!("{reference} has no retained source admission evidence"),
        ));
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
    let root = || -> Result<Vec<ReadItem>, Failure> {
        let mut items = Vec::new();
        polylogue_render::visit_document_root(
            &reference,
            layout.options,
            layout.evidence,
            |item| {
                items.push(item);
                true
            },
        )
        .map_err(output::read_error)?;
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
    let mut block = |ordinal: u64, block| {
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
        selected.visit_blocks_before_with_ordinals(from, &mut block)
    } else {
        selected.visit_blocks_from(
            from.map_or(0, |ordinal| ordinal.saturating_sub(1)),
            &mut block,
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

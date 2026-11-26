use super::{BranchState, Expansion, LayoutFacts, ReadError, ReadOptions, View};
use crate::{Block, bundle::Record, is_harness_message};

/// Whether the context view selected this record. Missing prepared membership
/// is an error, not an empty selection.
pub(super) fn context_selected(
    facts: &LayoutFacts,
    view: View,
    reference: &str,
) -> Result<bool, ReadError> {
    facts
        .context
        .as_ref()
        .map(|context| context.selected)
        .ok_or_else(|| ReadError::MissingPreparedData {
            view,
            facts: "context membership".into(),
            reference: reference.into(),
        })
}

/// A document reads as `transcript` or `raw` only; every other view refuses it
/// and names the two that fit.
pub fn document_view(view: View, reference: &str) -> Result<(), ReadError> {
    if matches!(
        view,
        View::Dialogue | View::Overview | View::Tree | View::Changes | View::Live
    ) {
        return Err(ReadError::UnsupportedTarget {
            view,
            target: reference.into(),
            accepted_views: vec![View::Transcript, View::Raw],
        });
    }
    Ok(())
}

/// Project a record already selected by the store. Source identity, original
/// block positions, provider metadata and time accuracy are never reconstructed
/// from rendered text. Alternate substitutions belong only to layout.
pub fn project_record(
    mut record: Record,
    options: &ReadOptions,
    facts: &LayoutFacts,
) -> Result<Option<Record>, ReadError> {
    options.validate()?;
    match &mut record {
        Record::Message(row) => {
            row.validate()
                .map_err(|error| ReadError::invalid(error.to_string()))?;
            let message = &row.message;
            if !options.view.keeps_message(message) {
                return Ok(None);
            }
            if options.view == View::Context {
                if !context_selected(facts, options.view, &row.reference)? {
                    return Ok(None);
                }
            } else {
                if facts.inherited && !options.inherited() {
                    return Ok(None);
                }
                if matches!(facts.branch, BranchState::Abandoned { .. }) && !options.branches() {
                    return Ok(None);
                }
            }
            if options.harness() == Expansion::None
                && is_harness_message(message.role, message.prompt_kind)
            {
                return Ok(None);
            }
            let old = std::mem::take(&mut row.message.blocks);
            let ordinals = std::mem::take(&mut row.block_ordinals);
            for (ordinal, block) in ordinals.into_iter().zip(old) {
                let keep = options.view.block_kinds().contains(&block.kind())
                    && !matches!(
                        &block,
                        Block::Thinking {
                            text: None,
                            signature: Some(_),
                            redacted: false
                        }
                    )
                    && (!matches!(block, Block::ToolCall { .. } | Block::ToolResult { .. })
                        || options.tools() != Expansion::None)
                    && (!matches!(block, Block::Thinking { .. })
                        || options.thinking() != Expansion::None);
                if keep {
                    row.block_ordinals.push(ordinal);
                    row.message.blocks.push(block);
                }
            }
            row.injections
                .retain(|injection| row.block_ordinals.binary_search(&injection.block).is_ok());
            row.bindings
                .retain(|binding| row.block_ordinals.binary_search(&binding.block).is_ok());
            if options.view == View::Dialogue && row.message.blocks.is_empty() {
                return Ok(None);
            }
            row.validate()
                .map_err(|error| ReadError::invalid(error.to_string()))?;
        }
        Record::Event(_) => {
            if matches!(
                options.view,
                View::Dialogue | View::Overview | View::Tree | View::Live
            ) {
                return Ok(None);
            }
            if options.view == View::Changes && facts.cards.is_empty() {
                return Ok(None);
            }
            if options.view == View::Context
                && !context_selected(facts, options.view, record.reference().unwrap_or_default())?
            {
                return Ok(None);
            }
        }
        Record::Document(document) => {
            document_view(options.view, &document.reference)?;
        }
        Record::TextFragment(row) => {
            super::fragments::project_fragment(row, options, facts)?;
        }
        Record::Authored(_) | Record::Annotation(_) => {
            if !options.annotations() {
                return Ok(None);
            }
        }
        Record::Manifest(_)
        | Record::Session(_)
        | Record::Link(_)
        | Record::Alternate(_)
        | Record::AlternatePart(_)
        | Record::Blob(_) => {}
    }
    Ok(Some(record))
}

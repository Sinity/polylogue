use super::{BranchState, Expansion, ProjectionEvidence, ReadError, ReadOptions, View};
use crate::{Block, PromptKind, Role, bundle::Record};

/// Project a record already selected by the store. Source identity, original
/// block positions, provider metadata and time evidence are never reconstructed
/// from rendered text. Variant substitutions belong only to layout.
pub fn project_record(
    mut record: Record,
    options: &ReadOptions,
    evidence: &ProjectionEvidence,
) -> Result<Option<Record>, ReadError> {
    options.validate()?;
    match &mut record {
        Record::Message(row) => {
            row.validate()
                .map_err(|error| ReadError::InvalidEvidence(error.to_string()))?;
            let message = &row.message;
            if !options.view.keeps_message(message) {
                return Ok(None);
            }
            if options.view == View::Context {
                let context =
                    evidence
                        .context
                        .as_ref()
                        .ok_or_else(|| ReadError::MissingEvidence {
                            view: options.view,
                            evidence: "context membership".into(),
                            reference: row.reference.clone(),
                        })?;
                if !context.selected {
                    return Ok(None);
                }
            } else {
                if evidence.inherited && !options.inherited {
                    return Ok(None);
                }
                if matches!(evidence.branch, BranchState::Abandoned { .. }) && !options.branches {
                    return Ok(None);
                }
            }
            let harness =
                message.role == Role::System || message.prompt_kind == Some(PromptKind::Harness);
            if options.harness() == Expansion::None && harness {
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
            if options.view == View::Dialogue && row.message.blocks.is_empty() {
                return Ok(None);
            }
            row.validate()
                .map_err(|error| ReadError::InvalidEvidence(error.to_string()))?;
        }
        Record::Event(_) => {
            if matches!(
                options.view,
                View::Dialogue | View::Summary | View::Links | View::Live
            ) {
                return Ok(None);
            }
            if options.view == View::Changes && evidence.cards.is_empty() {
                return Ok(None);
            }
            if options.view == View::Context {
                let context =
                    evidence
                        .context
                        .as_ref()
                        .ok_or_else(|| ReadError::MissingEvidence {
                            view: options.view,
                            evidence: "context membership".into(),
                            reference: record.reference().unwrap_or_default().into(),
                        })?;
                if !context.selected {
                    return Ok(None);
                }
            }
        }
        Record::Document(document) => {
            if options.view == View::Dialogue {
                return Err(ReadError::UnsupportedTarget {
                    view: options.view,
                    target: document.reference.clone(),
                    accepted_views: vec![View::Transcript, View::Raw],
                });
            }
            if matches!(
                options.view,
                View::Summary | View::Links | View::Changes | View::Live
            ) {
                return Ok(None);
            }
        }
        Record::Authored(_) => {
            if !options.assertions {
                return Ok(None);
            }
        }
        Record::Manifest(_)
        | Record::Session(_)
        | Record::Link(_)
        | Record::Variant(_)
        | Record::VariantPart(_)
        | Record::Blob(_) => {}
    }
    Ok(Some(record))
}

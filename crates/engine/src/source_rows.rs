//! Checks that an input's parsed rows are complete: provider receipts and
//! selected source certificates. A restart reruns the check from the start.
use super::owner::Result;
use crate::{
    canonical_message as m,
    owner::{self, CanonicalPosition},
    state::{claude, codex},
};
use polylogue_model::{
    BlobHash,
    source_rows::{ParsedFrom, SourceRowScope, SourceRowState, SourceRows},
};
use polylogue_store::parser_state::Phase;
use polylogue_store::{
    Error, InputFormat, StoredSource,
    parser_state::{Budget, Snapshot, Token},
};
use serde::{Deserialize, Serialize};
pub const RECIPE: u32 = 2;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub predecessor: Token,
    pub dispositions: [u64; 5],
    pub unsupported_effects: bool,
    pub rows: SourceRows,
}
impl Receipt {
    pub fn validate(&self, token: &Token, basis: &ParsedFrom) -> Result<()> {
        if &self.predecessor != token
            || &self.rows.parsed_from != basis
            || self.rows.scope
                != if basis.original_hash.is_some() {
                    SourceRowScope::ImportedSourceRows
                } else {
                    SourceRowScope::ObservedSourceRows
                }
            || matches!(self.rows.state, SourceRowState::Pending { .. })
        {
            return Err(Error::StaleParserState.into());
        }
        Ok(())
    }
    fn validate_published(
        &self,
        head: &polylogue_store::parser_state::Head,
        basis: &ParsedFrom,
    ) -> Result<()> {
        self.validate(&head.token, basis)?;
        let counts = [
            head.counts.framed,
            head.counts.used,
            head.counts.ignored,
            head.counts.unrecognized,
            head.counts.undecodable,
        ];
        let missing =
            self.unsupported_effects || head.counts.unrecognized > 0 || head.counts.undecodable > 0;
        let state = match &self.rows.state {
            SourceRowState::Complete => {
                head.status == Phase::Done && !head.finalization_pending && !missing
            }
            SourceRowState::Unavailable {
                notes,
                record_errors,
                ..
            } => {
                head.status == Phase::Finalizing
                    && head.finalization_pending
                    && missing
                    && !notes.is_empty()
                    && *record_errors == head.counts.undecodable
            }
            SourceRowState::Pending { .. } => false,
        };
        if self.dispositions != counts
            || !state
            || head.parsed_upto != basis.length
            || head.acquired_upto != basis.length
        {
            return Err(Error::Invariant(
                "source row receipt disagrees with the stored completion state".into(),
            )
            .into());
        }
        Ok(())
    }
}
pub fn computed_from(input: &StoredSource) -> ParsedFrom {
    ParsedFrom {
        input: input.input.id.clone(),
        original_hash: input.original_hash(),
        chain: input.input.chain,
        revision: input.revision,
        length: input.input.length,
        parser_version: owner::parser_version(input.format),
        interpretation: BlobHash::from_bytes(owner::parser_identity(input)),
    }
}
pub fn pending(input: &StoredSource, obligation: &str) -> SourceRows {
    SourceRows {
        scope: if input.observed() {
            SourceRowScope::ObservedSourceRows
        } else {
            SourceRowScope::ImportedSourceRows
        },
        parsed_from: computed_from(input),
        state: SourceRowState::Pending {
            pending_work: vec![obligation.into()],
        },
    }
}
pub fn read(snapshot: &Snapshot, input: &StoredSource) -> Result<Option<Receipt>> {
    let value = snapshot.cell(
        polylogue_store::parser_state::SOURCE_ROWS_NAMESPACE,
        b"",
        16384,
    )?;
    let Some(value) = value else {
        if snapshot.head.status == Phase::Done {
            return Err(Error::Invariant("done source lost its source-row receipt".into()).into());
        }
        return Ok(None);
    };
    Ok(Some(decode(&value, &snapshot.head, input)?))
}
pub(crate) fn decode(
    value: &[u8],
    head: &polylogue_store::parser_state::Head,
    input: &StoredSource,
) -> Result<Receipt> {
    let receipt: Receipt = serde_json::from_slice(value)
        .map_err(|_| Error::Invariant("invalid stored source-row receipt".into()))?;
    receipt.validate_published(head, &computed_from(input))?;
    Ok(receipt)
}

#[derive(Clone)]
pub struct Audit {
    step: i64,
    effect: Option<codex::EffectsCursor>,
    position: Option<CanonicalPosition>,
    unsupported: bool,
    usable: bool,
    ignored: u64,
    unrecognized: u64,
    undecodable: u64,
}
pub enum Turn {
    Progress(Audit),
    Complete(Receipt),
}
impl Audit {
    pub fn new() -> Self {
        Self {
            step: 1,
            effect: None,
            position: None,
            unsupported: false,
            usable: false,
            ignored: 0,
            unrecognized: 0,
            undecodable: 0,
        }
    }
    pub fn advance(
        mut self,
        snapshot: &Snapshot,
        input: &StoredSource,
        reads: Budget,
    ) -> Result<Turn> {
        let head = &snapshot.head;
        if head.status != Phase::Finalizing
            || !head.finalization_pending
            || head.parsed_upto != input.input.length
            || head.acquired_upto != input.input.length
            || head.token.basis != owner::parser_identity(input)
        {
            return Err(Error::StaleParserState.into());
        }
        if self.step <= head.token.step {
            match input.format {
                InputFormat::CodexPromptHistory => {
                    return Err(Error::Invariant(
                        "prompt references cannot complete conversation source rows".into(),
                    )
                    .into());
                }
                InputFormat::ClaudeCodeToolResult => {
                    return Err(
                        Error::Invariant("companion is not independently parsed".into()).into(),
                    );
                }
                InputFormat::CodexHistorySqlite => {
                    self.step = head.token.step + 1;
                }
                InputFormat::ClaudeCodeJsonl => {
                    let view = claude::View::new(snapshot, reads)?;
                    let effects = view.effects(self.step)?.ok_or_else(|| {
                        Error::Invariant("missing Claude step disposition receipt".into())
                    })?;
                    self.unsupported |= !effects.events.is_empty() || !effects.links.is_empty();
                    self.ignored = self
                        .ignored
                        .checked_add(
                            effects
                                .ignored
                                .values()
                                .try_fold(0u64, |n, v| n.checked_add(*v))
                                .ok_or(Error::Range)?,
                        )
                        .ok_or(Error::Range)?;
                    self.unrecognized = self
                        .unrecognized
                        .checked_add(effects.unrecognized.len() as u64)
                        .ok_or(Error::Range)?;
                    self.undecodable = self
                        .undecodable
                        .checked_add(effects.undecodable.len() as u64)
                        .ok_or(Error::Range)?;
                    // Old scheduling cuts must name a current candidate, not an
                    // obsolete intermediate body.
                    for pending in &effects.pending_messages {
                        if view.pending(&pending.message)?.is_none() {
                            return Err(Error::Invariant(
                                "lost Claude required Message obligation".into(),
                            )
                            .into());
                        }
                    }
                    self.step = self.step.checked_add(1).ok_or(Error::Range)?;
                }
                InputFormat::CodexRollout => {
                    let attempt = codex::Attempt::new(snapshot, reads);
                    let result = (|| {
                        let reader = attempt.effects(self.step as u64)?;
                        let mut cursor = self.effect.take().unwrap_or_else(|| reader.start());
                        let effect = reader.advance(&mut cursor)?;
                        self.effect = Some(cursor);
                        match effect {
                            codex::EffectStep::Item(
                                codex::Effect::Event(_)
                                | codex::Effect::Link(_)
                                | codex::Effect::Conflict(_)
                                | codex::Effect::CallImpact { .. },
                            ) => self.unsupported = true,
                            codex::EffectStep::Item(
                                codex::Effect::PendingVersion { message, .. }
                                | codex::Effect::MessageChanged { message }
                                | codex::Effect::PendingResult { message, .. },
                            ) => {
                                if message >= attempt.resume()?.message_count() {
                                    return Err(
                                        "lost Codex required Message obligation".to_string()
                                    );
                                }
                            }
                            codex::EffectStep::Done => {
                                self.effect = None;
                                self.step = self
                                    .step
                                    .checked_add(1)
                                    .ok_or_else(|| "source effect step overflow".to_string())?;
                            }
                            _ => {}
                        }
                        Ok::<(), String>(())
                    })();
                    if let Some(error) = attempt.take_failure() {
                        return Err(error.into());
                    }
                    result.map_err(owner::Error::InvalidEvidence)?;
                }
            }
            return Ok(Turn::Progress(self));
        }
        // A ready Session operation counts without any Message; read its
        // publication from the same snapshot.
        self.usable |= snapshot.source_has_tool_items()?;
        let position = self
            .position
            .take()
            .unwrap_or_else(|| CanonicalPosition::first(input.format));
        match position {
            CanonicalPosition::ToolsAfter(_) | CanonicalPosition::ToolsBeforeFailure(_, _) => {
                self.usable |= snapshot.source_has_tool_items()?;
            }
            CanonicalPosition::Audit(_) => {
                return Err(owner::Error::InvalidRequest("nested source audit cursor"));
            }
            CanonicalPosition::ClaudeAfter(after) => {
                let view = claude::View::new(snapshot, reads)?;
                if let Some((cursor, pending)) = view.next_pending(after.as_ref())? {
                    self.position = Some(CanonicalPosition::ClaudeMessage(cursor, pending));
                    return Ok(Turn::Progress(self));
                }
            }
            CanonicalPosition::ClaudeMessage(cursor, pending) => {
                let position = CanonicalPosition::ClaudeMessage(cursor.clone(), pending.clone());
                let key = position.logical_key()?.ok_or(Error::Range)?;
                let source = snapshot
                    .selected_source_projection(&key, 65536)?
                    .ok_or_else(|| {
                        Error::Invariant("required Claude source certificate absent".into())
                    })?;
                let action = m::start_claude(
                    snapshot,
                    &m::ClaudeRequest {
                        pending,
                        source_binding: input.binding,
                        context: input.context.clone(),
                        revision: input.revision,
                        logical_key: key,
                    },
                    Some(source),
                    reads,
                    65536,
                )
                .map_err(owner::Error::Canonical)?;
                if action.operation() != "existing" {
                    return Err(Error::StaleMessage.into());
                }
                self.usable = true;
                self.position = Some(CanonicalPosition::ClaudeAfter(Some(cursor)));
                return Ok(Turn::Progress(self));
            }
            CanonicalPosition::Codex(ordinal) => {
                let attempt = codex::Attempt::new(snapshot, reads);
                let resume = attempt.resume();
                if let Some(e) = attempt.take_failure() {
                    return Err(e.into());
                }
                let resume = resume.map_err(owner::Error::InvalidEvidence)?;
                self.unsupported |= resume.normalization_incomplete();
                if ordinal < resume.message_count() {
                    let key = owner::canonical_key(ordinal)?;
                    let source = snapshot
                        .selected_source_projection(&key, 65536)?
                        .ok_or_else(|| {
                            Error::Invariant("required Codex source certificate absent".into())
                        })?;
                    let action = m::start(
                        snapshot,
                        &m::Request {
                            message: ordinal,
                            source_binding: input.binding,
                            revision: input.revision,
                            logical_key: key,
                        },
                        Some(source),
                        reads,
                        65536,
                    )
                    .map_err(owner::Error::Canonical)?;
                    if action.operation() != "existing" {
                        return Err(Error::StaleMessage.into());
                    }
                    self.usable = true;
                    self.position = Some(CanonicalPosition::Codex(ordinal + 1));
                    return Ok(Turn::Progress(self));
                }
            }
        }
        if input.format == InputFormat::ClaudeCodeJsonl
            && (self.ignored != head.counts.ignored
                || self.unrecognized != head.counts.unrecognized
                || self.undecodable != head.counts.undecodable)
        {
            return Err(Error::Invariant(
                "source disposition receipts disagree with accounting".into(),
            )
            .into());
        }
        let mut notes = Vec::new();
        if self.unsupported {
            notes.push(
                "source contains unsupported event, link, operation or normalization effects"
                    .into(),
            );
        }
        if head.counts.unrecognized > 0 {
            notes.push("source has unrecognized records with unresolved required effects".into());
        }
        if head.counts.undecodable > 0 {
            notes.push("source has undecodable complete records".into());
        }
        let state = if notes.is_empty() {
            SourceRowState::Complete
        } else {
            SourceRowState::Unavailable {
                usable_result: self.usable,
                notes,
                record_errors: head.counts.undecodable,
            }
        };
        Ok(Turn::Complete(Receipt {
            predecessor: head.token.clone(),
            unsupported_effects: self.unsupported,
            dispositions: [
                head.counts.framed,
                head.counts.used,
                head.counts.ignored,
                head.counts.unrecognized,
                head.counts.undecodable,
            ],
            rows: SourceRows {
                scope: if input.observed() {
                    SourceRowScope::ObservedSourceRows
                } else {
                    SourceRowScope::ImportedSourceRows
                },
                parsed_from: computed_from(input),
                state,
            },
        }))
    }
}

#[cfg(test)]
mod tests;

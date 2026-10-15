//! Read-side audit of existing provider receipts and selected source certificates.
//! Restart replays this scalar audit; no second work queue or archive ledger.
use super::owner::Result;
use crate::{
    canonical_message as m,
    owner::{self, CanonicalPosition},
    state::{claude, codex},
};
use polylogue_model::{
    BlobHash,
    settlement::{SourceRowBasis, SourceRowScope, SourceRowState, SourceRows},
};
use polylogue_store::{
    Error, ImmutableImport, InputFormat,
    parser_state::{Budget, Snapshot, Token},
};
use serde::{Deserialize, Serialize};
pub const RECIPE: u32 = 1;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub generation: i64,
    pub step: i64,
    pub basis: BlobHash,
    pub rows: SourceRows,
}
impl Receipt {
    pub fn validate(&self, token: &Token, basis: &SourceRowBasis) -> Result<()> {
        if self.generation != token.generation
            || self.step != token.step
            || self.basis.bytes() != &token.basis
            || &self.rows.basis != basis
            || self.rows.scope != SourceRowScope::ImportedSourceRows
            || matches!(self.rows.state, SourceRowState::Pending { .. })
        {
            return Err(Error::ParserPredecessorChanged.into());
        }
        Ok(())
    }
}
pub fn basis(input: &ImmutableImport) -> SourceRowBasis {
    SourceRowBasis {
        input: input.input.id.clone(),
        original_hash: input.hash,
        chain: input.input.chain,
        revision: 1,
        length: input.input.length,
        parser_version: owner::parser_version(input.format),
        interpretation: BlobHash::from_bytes(owner::parser_basis(input)),
    }
}
pub fn pending(input: &ImmutableImport, obligation: &str) -> SourceRows {
    SourceRows {
        scope: SourceRowScope::ImportedSourceRows,
        basis: basis(input),
        state: SourceRowState::Pending {
            obligations: vec![obligation.into()],
        },
    }
}
pub fn read(snapshot: &Snapshot, input: &ImmutableImport) -> Result<Option<Receipt>> {
    let value = snapshot.cell(
        polylogue_store::parser_state::SOURCE_ROWS_NAMESPACE,
        b"",
        16384,
    )?;
    let Some(value) = value else {
        if snapshot.head.status == "done" {
            return Err(
                Error::Invariant("done source lost its row settlement authority".into()).into(),
            );
        }
        return Ok(None);
    };
    let receipt: Receipt = serde_json::from_slice(&value)
        .map_err(|_| Error::Invariant("invalid source row settlement receipt".into()))?;
    receipt.validate(&snapshot.head.token, &basis(input))?;
    Ok(Some(receipt))
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
        input: &ImmutableImport,
        reads: Budget,
    ) -> Result<Turn> {
        let head = &snapshot.head;
        if head.status != "finalizing"
            || !head.finalization_pending
            || head.parsed_upto != input.input.length
            || head.acquired_upto != input.input.length
            || head.token.basis != owner::parser_basis(input)
        {
            return Err(Error::ParserPredecessorChanged.into());
        }
        if self.step <= head.token.step {
            match input.format {
                InputFormat::ClaudeCodeJsonl => {
                    let view = claude::View::new(snapshot, reads)?;
                    let effects = view.effects(self.step)?.ok_or_else(|| {
                        Error::Invariant("missing Claude step disposition receipt".into())
                    })?;
                    self.unsupported |= !effects.events.is_empty() || !effects.links.is_empty();
                    self.ignored = self
                        .ignored
                        .checked_add(effects.ignored.values().sum::<u64>())
                        .ok_or(Error::Range)?;
                    self.unrecognized = self
                        .unrecognized
                        .checked_add(effects.unrecognized.len() as u64)
                        .ok_or(Error::Range)?;
                    self.undecodable = self
                        .undecodable
                        .checked_add(effects.undecodable.len() as u64)
                        .ok_or(Error::Range)?;
                    // Historical scheduling cuts must name an extant effective
                    // candidate, not resurrect their obsolete intermediate body.
                    for pending in &effects.pending_messages {
                        if view.pending(&pending.message)?.is_none() {
                            return Err(Error::Invariant(
                                "lost Claude required Message obligation".into(),
                            )
                            .into());
                        }
                    }
                    self.step += 1;
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
                                self.step += 1;
                            }
                            _ => {}
                        }
                        Ok::<(), String>(())
                    })();
                    if let Some(error) = attempt.take_failure() {
                        return Err(error.into());
                    }
                    result.map_err(|e| owner::Error::InvalidEvidence(e))?;
                }
            }
            return Ok(Turn::Progress(self));
        }
        let position = self
            .position
            .take()
            .unwrap_or_else(|| CanonicalPosition::first(input.format));
        match position {
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
                        source_hash: input.hash,
                        context: input.context.clone(),
                        revision: 1,
                        logical_key: key,
                    },
                    Some(source),
                    reads,
                    65536,
                )
                .map_err(owner::Error::Canonical)?;
                if action.operation() != "existing" {
                    return Err(Error::MessagePredecessorChanged.into());
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
                            source_hash: input.hash,
                            revision: 1,
                            logical_key: key,
                        },
                        Some(source),
                        reads,
                        65536,
                    )
                    .map_err(owner::Error::Canonical)?;
                    if action.operation() != "existing" {
                        return Err(Error::MessagePredecessorChanged.into());
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
            notes.push(
                "retained source has unrecognized records with unresolved required effects".into(),
            );
        }
        if head.counts.undecodable > 0 {
            notes.push("retained source has undecodable complete records".into());
        }
        let state = if notes.is_empty() {
            SourceRowState::Settled
        } else {
            SourceRowState::Unavailable {
                usable_result: self.usable,
                notes,
                record_errors: head.counts.undecodable,
            }
        };
        Ok(Turn::Complete(Receipt {
            generation: head.token.generation,
            step: head.token.step,
            basis: BlobHash::from_bytes(head.token.basis),
            rows: SourceRows {
                scope: SourceRowScope::ImportedSourceRows,
                basis: basis(input),
                state,
            },
        }))
    }
}

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
    Error, InputFormat, RetainedSource,
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
    pub fn validate(&self, token: &Token, basis: &SourceRowBasis) -> Result<()> {
        if &self.predecessor != token
            || &self.rows.basis != basis
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
        basis: &SourceRowBasis,
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
            SourceRowState::Settled => {
                head.status == "done" && !head.finalization_pending && !missing
            }
            SourceRowState::Unavailable {
                notes,
                record_errors,
                ..
            } => {
                head.status == "finalizing"
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
                "source row receipt disagrees with its retained completion authority".into(),
            )
            .into());
        }
        Ok(())
    }
}
pub fn basis(input: &RetainedSource) -> SourceRowBasis {
    SourceRowBasis {
        input: input.input.id.clone(),
        original_hash: input.original_hash(),
        chain: input.input.chain,
        revision: input.revision,
        length: input.input.length,
        parser_version: owner::parser_version(input.format),
        interpretation: BlobHash::from_bytes(owner::parser_basis(input)),
    }
}
pub fn pending(input: &RetainedSource, obligation: &str) -> SourceRows {
    SourceRows {
        scope: if input.observed() {
            SourceRowScope::ObservedSourceRows
        } else {
            SourceRowScope::ImportedSourceRows
        },
        basis: basis(input),
        state: SourceRowState::Pending {
            obligations: vec![obligation.into()],
        },
    }
}
pub fn read(snapshot: &Snapshot, input: &RetainedSource) -> Result<Option<Receipt>> {
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
    Ok(Some(decode(&value, &snapshot.head, input)?))
}
pub(crate) fn decode(
    value: &[u8],
    head: &polylogue_store::parser_state::Head,
    input: &RetainedSource,
) -> Result<Receipt> {
    let receipt: Receipt = serde_json::from_slice(value)
        .map_err(|_| Error::Invariant("invalid source row settlement receipt".into()))?;
    receipt.validate_published(head, &basis(input))?;
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
        input: &RetainedSource,
        reads: Budget,
    ) -> Result<Turn> {
        let head = &snapshot.head;
        if head.status != "finalizing"
            || !head.finalization_pending
            || head.parsed_upto != input.input.length
            || head.acquired_upto != input.input.length
            || head.token.basis != owner::parser_basis(input)
        {
            return Err(Error::StaleParserState.into());
        }
        if self.step <= head.token.step {
            match input.format {
                InputFormat::CodexPromptHistory => {
                    return Err(Error::Invariant(
                        "prompt references cannot certify conversation row settlement".into(),
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
                    result.map_err(|e| owner::Error::InvalidEvidence(e))?;
                }
            }
            return Ok(Turn::Progress(self));
        }
        // A ready Session-owned operation is usable without any Message. This
        // reads the actual publication in the same pinned source snapshot.
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
                basis: basis(input),
                state,
            },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polylogue_store::parser_state::{Accounting, Head};
    fn fixture() -> (Receipt, Head) {
        let token = Token {
            index_generation: "synthetic-index".into(),
            input_pk: 1,
            chain: 1,
            generation: 2,
            step: 3,
            basis: [4; 32],
        };
        let basis = SourceRowBasis {
            input: "synthetic-input".into(),
            original_hash: Some(BlobHash::of(b"raw")),
            chain: 1,
            revision: 1,
            length: 10,
            parser_version: 1,
            interpretation: BlobHash::of(b"interpretation"),
        };
        let head = Head {
            token: token.clone(),
            parser_version: 1,
            acquired_upto: 10,
            inspected_upto: 10,
            parsed_upto: 10,
            status: "done".into(),
            finalization_pending: false,
            counts: Accounting::default(),
        };
        (
            Receipt {
                predecessor: token,
                dispositions: [0; 5],
                unsupported_effects: false,
                rows: SourceRows {
                    scope: SourceRowScope::ImportedSourceRows,
                    basis,
                    state: SourceRowState::Settled,
                },
            },
            head,
        )
    }
    #[test]
    fn receipt_binds_every_predecessor_coordinate_and_source_basis() {
        let (receipt, head) = fixture();
        receipt
            .validate_published(&head, &receipt.rows.basis)
            .unwrap();
        for alter in [
            |t: &mut Token| t.index_generation.push('x'),
            |t: &mut Token| t.input_pk += 1,
            |t: &mut Token| t.chain += 1,
            |t: &mut Token| t.generation += 1,
            |t: &mut Token| t.step += 1,
            |t: &mut Token| t.basis[0] ^= 1,
        ] {
            let mut wrong = head.clone();
            alter(&mut wrong.token);
            assert!(
                receipt
                    .validate_published(&wrong, &receipt.rows.basis)
                    .is_err()
            );
        }
        let mut wrong = receipt.rows.basis.clone();
        wrong.interpretation = BlobHash::of(b"new recipe");
        assert!(receipt.validate_published(&head, &wrong).is_err());
        let mut wrong = receipt.rows.basis.clone();
        wrong.original_hash = Some(BlobHash::of(b"different raw"));
        assert!(receipt.validate_published(&head, &wrong).is_err());
    }
    #[test]
    fn same_generation_damaged_status_counts_and_missing_effects_do_not_become_settled() {
        let (receipt, head) = fixture();
        let mut wrong = head.clone();
        wrong.finalization_pending = true;
        assert!(
            receipt
                .validate_published(&wrong, &receipt.rows.basis)
                .is_err()
        );
        let mut wrong = head.clone();
        wrong.counts.framed = 1;
        wrong.counts.undecodable = 1;
        assert!(
            receipt
                .validate_published(&wrong, &receipt.rows.basis)
                .is_err()
        );
        let mut wrong = receipt.clone();
        wrong.unsupported_effects = true;
        assert!(wrong.validate_published(&head, &wrong.rows.basis).is_err());
        let mut wrong = head.clone();
        wrong.parsed_upto = 9;
        assert!(
            receipt
                .validate_published(&wrong, &receipt.rows.basis)
                .is_err()
        );
        let encoded = serde_json::to_vec(&receipt).unwrap();
        let decoded: Receipt = serde_json::from_slice(&encoded).unwrap();
        decoded
            .validate_published(&head, &decoded.rows.basis)
            .unwrap();
    }
}

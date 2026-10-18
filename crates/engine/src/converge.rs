//! Writes one parsed provider-state delta, one transaction per owner turn.
use polylogue_store::parser_state::prepared::{Work, WorkPhase};
use polylogue_store::parser_state::{Budget, CellChange, Head, Phase, Progress, Token};
use polylogue_store::{Error, Writer};

type Result<T> = std::result::Result<T, Error>;

pub enum Turn {
    Yielded,
    Published(Head),
}

/// One encoded provider delta being written to parser state.
pub struct PendingInstall {
    work: Work,
    changes: Vec<CellChange>,
}
impl PendingInstall {
    pub fn begin(
        writer: &mut Writer,
        head: &Head,
        progress: Progress,
        changes: Vec<CellChange>,
    ) -> Result<Self> {
        // Only Message finalization may mark an input done, never a state install.
        if matches!(progress.phase, Phase::Done) || !progress.finalization_pending {
            return Err(Error::Invariant(
                "provider-state install cannot finish an input".into(),
            ));
        }
        let work = writer.prepare_parser(head, progress)?;
        Ok(Self { work, changes })
    }
    pub fn predecessor(&self) -> &Token {
        &self.work.token.head
    }
    /// Runs at most one transaction: stage, seal, apply or publish. Staging and
    /// applying stop at the budget; cells that do not fit wait for the next
    /// call, and a first cell that does not fit fails before writing.
    pub fn turn(&mut self, writer: &mut Writer, budget: Budget) -> Result<Turn> {
        if self.work.phase != WorkPhase::Applying {
            let staged = usize::try_from(self.work.token.staged).map_err(|_| Error::Range)?;
            if staged < self.changes.len() {
                let count = self
                    .work
                    .token
                    .staging_prefix(&self.changes[staged..], budget)?;
                self.work = writer.stage_parser(
                    &self.work.token,
                    &self.changes[staged..staged + count],
                    budget,
                )?;
            } else {
                self.work = writer.seal_parser(&self.work.token)?;
            }
            return Ok(Turn::Yielded);
        }
        if self.work.token.applied < self.work.token.staged {
            self.work = writer.apply_parser(&self.work.token, budget)?;
            return Ok(Turn::Yielded);
        }
        let head = writer.publish_parser(&self.work.token)?;
        self.changes.clear();
        Ok(Turn::Published(head))
    }
}

pub enum Recovery {
    Yielded,
    Reparse,
    Published(Head),
}

/// After a crash, an unsealed delta is incomplete: discard it in batches and
/// parse again from the unchanged state. A sealed delta is complete: finish
/// applying it without parsing against partly changed state.
pub fn recover_turn(writer: &mut Writer, predecessor: &Token, budget: Budget) -> Result<Recovery> {
    let Some(work) = writer.parser_work(predecessor)? else {
        return Ok(Recovery::Reparse);
    };
    if matches!(work.progress.phase, Phase::Done) || !work.progress.finalization_pending {
        return Err(Error::Invariant(
            "provider-state recovery cannot finish an input".into(),
        ));
    }
    if work.phase != WorkPhase::Applying {
        return Ok(
            match writer.discard_parser_preparation(&work.token, budget)? {
                Some(_) => Recovery::Yielded,
                None => Recovery::Reparse,
            },
        );
    }
    if work.token.applied < work.token.staged {
        writer.apply_parser(&work.token, budget)?;
        return Ok(Recovery::Yielded);
    }
    writer.publish_parser(&work.token).map(Recovery::Published)
}

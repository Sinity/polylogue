//! One admitted provider-state installation transaction per owner turn.
//! Canonical output/finalization remains separate; EOF never earns Done here.
use polylogue_store::parser_state::prepared::Work;
use polylogue_store::parser_state::{Budget, CellChange, Head, Phase, Progress, Token};
use polylogue_store::{Error, Writer};

type Result<T> = std::result::Result<T, Error>;

pub enum Turn {
    Yielded,
    Published(Head),
}

/// Owns one already admitted encoded provider delta. Its buffers, pinned source
/// state and decoder allocations have separate caller lifetime charges. This
/// type does not claim that a single huge cell is physically bounded.
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
        // Only a canonical finalizer can earn completion. Installing parser
        // state/effect receipts proves no public Message or FTS completeness.
        if matches!(progress.phase, Phase::Done) || !progress.finalization_pending {
            return Err(Error::Invariant(
                "provider installation cannot certify finalization".into(),
            ));
        }
        let work = writer.prepare_parser(head, progress)?;
        Ok(Self { work, changes })
    }
    pub fn predecessor(&self) -> &Token {
        &self.work.token.predecessor
    }
    /// Each call performs at most one staging/application/control transaction.
    /// Staging and application honor the supplied cell/byte budget. A later
    /// unfit member stays owed; an unfit first member fails without mutation.
    pub fn turn(&mut self, writer: &mut Writer, budget: Budget) -> Result<Turn> {
        if !self.work.applying {
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

/// After process loss, an unsealed delta is incomplete and is discarded in
/// admitted keysets before reinterpreting its unchanged predecessor. A sealed
/// delta is complete: finish applying its durable members without replaying a
/// callback against partly changed state. Failure never manufactures EOF.
pub fn recover_turn(writer: &mut Writer, predecessor: &Token, budget: Budget) -> Result<Recovery> {
    let Some(work) = writer.parser_work(predecessor)? else {
        return Ok(Recovery::Reparse);
    };
    if matches!(work.progress.phase, Phase::Done) || !work.progress.finalization_pending {
        return Err(Error::Invariant(
            "provider recovery cannot certify finalization".into(),
        ));
    }
    if !work.applying {
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

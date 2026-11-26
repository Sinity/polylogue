//! Requests a bundle carried are restored with their recorded effects; nothing
//! in them is resolved or run again.
use super::*;
use polylogue_model::authored::PreparedRequest;
use polylogue_model::user::UserOutcome;

pub(super) struct Restoration {
    pub prepared: PreparedRequest,
}

impl WriterSender {
    /// Restores a request a bundle carried without resolving its intent again.
    pub fn restore_authored(&self, prepared: PreparedRequest) -> Result<UserOutcome> {
        self.0
            .ask(|reply| Msg::RestoreAuthored(Box::new(Restoration { prepared }), reply))
    }
}

/// Whether the request is already stored; a different request under its ID
/// is a conflict.
fn stored(conn: &Connection, restoration: &Restoration) -> Result<bool> {
    let Some(known) =
        crate::authored_records::stored_request(conn, &restoration.prepared.request.id)?
    else {
        return Ok(false);
    };
    if known != restoration.prepared {
        return Err(StoreError::Conflict(known.request.id));
    }
    Ok(true)
}

impl Inner {
    pub(super) fn project_alternate_text(&mut self) -> Result<()> {
        let before = self.index.total_changes();
        let tx = self.index.transaction()?;
        crate::alternates::project(&tx)?;
        let changed = tx.total_changes() != before;
        if changed {
            bump_counter(&tx)?;
        }
        tx.commit()?;
        if changed {
            self.commits
                .committed(crate::completion::Committed::AUTHORED);
        }
        Ok(())
    }

    pub(super) fn restore_authored(&mut self, restoration: Restoration) -> Result<UserOutcome> {
        if stored(&self.archive, &restoration)? {
            self.project_alternate_text()?;
            let mut outcome = restoration.prepared.outcome;
            outcome.repeat = true;
            return Ok(outcome);
        }
        let tx = self.archive.transaction()?;
        let counter = bump_counter(&tx)?;
        let outcome = crate::authored_records::restore(&tx, &restoration.prepared, counter)?;
        crate::write_lock::commit(tx)?;
        self.commits
            .committed(crate::completion::Committed::AUTHORED);
        self.project_alternate_text()?;
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests;

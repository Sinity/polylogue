//! Requests a bundle carried are restored with their frozen effects; nothing
//! in them is resolved or run again.
use super::*;
use polylogue_model::authored::PreparedRequest;
use polylogue_model::user::UserOutcome;

pub(super) struct Restoration {
    pub prepared: PreparedRequest,
}

pub(super) fn restore_via(queue: &WriterQueue, restoration: Restoration) -> Result<UserOutcome> {
    let (reply, wait) = sync_channel(1);
    queue
        .send(Msg::RestoreAuthored(Box::new(restoration), reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

/// Whether the request is already stored; a different request under its ID
/// is a conflict.
fn stored(conn: &Connection, restoration: &Restoration) -> Result<bool> {
    let Some(known) =
        crate::primary::authored::frozen_request(conn, &restoration.prepared.request.id)?
    else {
        return Ok(false);
    };
    if known != restoration.prepared {
        return Err(StoreError::Conflict(known.request.id));
    }
    Ok(true)
}

impl Inner {
    pub(super) fn project_variant_text(&mut self) -> Result<()> {
        let before = self.index.total_changes();
        let tx = self.index.transaction()?;
        crate::variants::project(&tx)?;
        let changed = tx.total_changes() != before;
        if changed {
            bump_counter(&tx)?;
        }
        tx.commit()?;
        if changed {
            self.commits.committed();
        }
        Ok(())
    }

    pub(super) fn restore_authored(&mut self, restoration: Restoration) -> Result<UserOutcome> {
        if let Some(sinex) = self.sinex.as_ref().map(Arc::clone) {
            self.primary_cache_ready()?;
            let mut outcome = restoration.prepared.outcome.clone();
            if stored(&self.index, &restoration)? {
                outcome.repeat = true;
                return Ok(outcome);
            }
            let product = crate::primary::Product::RestoredRequest {
                prepared: restoration.prepared.clone(),
            };
            let writes = self.product_writes(vec![product])?;
            let identity = format!(
                "authored-restore:{}",
                blake3::hash(&serde_json::to_vec(&restoration.prepared)?).to_hex()
            );
            sinex.prepare_named_products(&identity, &writes)?;
            outcome.pending = true;
            return Ok(outcome);
        }
        if stored(&self.archive, &restoration)? {
            self.project_variant_text()?;
            let mut outcome = restoration.prepared.outcome;
            outcome.repeat = true;
            return Ok(outcome);
        }
        let tx = self.archive.transaction()?;
        let counter = bump_counter(&tx)?;
        let outcome = crate::primary::authored::restore(&tx, &restoration.prepared, counter)?;
        crate::write_lock::commit(tx)?;
        self.commits.committed();
        self.project_variant_text()?;
        self.derive_due = true;
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests;

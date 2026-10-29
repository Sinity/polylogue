//! Historical authored copies preserve frozen effects and computed local ingress.
use super::*;
use polylogue_model::authored::{ImportAttribution, ImportHistory, PreparedRequest};
use polylogue_model::user::UserOutcome;

pub(super) struct Restoration {
    pub prepared: PreparedRequest,
    pub claims: Vec<ImportAttribution>,
    pub current: ImportAttribution,
}

pub(super) fn restore_via(queue: &WriterQueue, restoration: Restoration) -> Result<UserOutcome> {
    let (reply, wait) = sync_channel(1);
    queue
        .send(Msg::RestoreAuthored(Box::new(restoration), reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

fn observed(conn: &Connection, restoration: &Restoration) -> Result<bool> {
    let Some(known) = crate::primary::authored::known_request(conn, &restoration.prepared.request)?
    else {
        return Ok(false);
    };
    if known != restoration.prepared {
        return Err(StoreError::Conflict(known.request.id));
    }
    let source = crate::source::Source::of(conn)?;
    let imports: String = conn.query_row(
        &source.sql("SELECT imports FROM @source.client_request WHERE id=?1"),
        [&known.request.id],
        |row| row.get(0),
    )?;
    let imports: ImportHistory = serde_json::from_str(&imports)?;
    let current = &restoration.current;
    Ok(imports.observed.iter().any(|prior| {
        prior.input == current.input
            && prior.chain == current.chain
            && prior.machine == current.machine
            && prior.actor == current.actor
    }) && restoration
        .claims
        .iter()
        .all(|claim| imports.declared.contains(claim)))
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

    pub(super) fn restore_authored(&mut self, mut restoration: Restoration) -> Result<UserOutcome> {
        if let Some(sinex) = self.sinex.as_ref().map(Arc::clone) {
            let mut outcome = restoration.prepared.outcome.clone();
            if observed(&self.index, &restoration)? {
                outcome.repeat = true;
                return Ok(outcome);
            }
            let mut original = None;
            sinex.visit_pending_requests(&mut |pending| {
                if let polylogue_model::authored::PendingAuthored::Restore {
                    prepared,
                    claims,
                    current,
                } = pending
                    && prepared == &restoration.prepared
                    && claims == &restoration.claims
                    && current.input == restoration.current.input
                    && current.chain == restoration.current.chain
                    && current.machine == restoration.current.machine
                    && current.actor == restoration.current.actor
                {
                    original = Some(current.clone());
                }
                Ok(())
            })?;
            if let Some(original) = original {
                restoration.current = original;
            }
            let product = crate::primary::Product::RestoredRequest {
                prepared: restoration.prepared.clone(),
                claims: restoration.claims.clone(),
                current: restoration.current.clone(),
            };
            let writes = self.product_writes(vec![product])?;
            let identity = format!(
                "authored-restore:{}",
                blake3::hash(&serde_json::to_vec(&(
                    &restoration.prepared,
                    &restoration.claims,
                    &restoration.current.input,
                    restoration.current.chain,
                    &restoration.current.machine,
                    &restoration.current.actor
                ))?)
                .to_hex()
            );
            sinex.prepare_named_products(&identity, &writes)?;
            match self.poll_primary() {
                Ok(_) => outcome.pending = !observed(&self.index, &restoration)?,
                Err(StoreError::Sinex(_)) => outcome.pending = true,
                Err(error) => return Err(error),
            }
            return Ok(outcome);
        }
        if observed(&self.archive, &restoration)? {
            self.project_variant_text()?;
            let mut outcome = restoration.prepared.outcome;
            outcome.repeat = true;
            return Ok(outcome);
        }
        let tx = self.archive.transaction()?;
        let counter = bump_counter(&tx)?;
        let outcome = crate::primary::authored::restore(
            &tx,
            &restoration.prepared,
            &restoration.claims,
            &restoration.current,
            counter,
        )?;
        // Repeated content can still add a newly observed import provenance.
        crate::write_lock::commit(tx)?;
        self.commits.committed();
        self.project_variant_text()?;
        self.derive_due = true;
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests;

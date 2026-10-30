//! Independent historical assertions use the existing immutable producer spool.
use super::*;
use polylogue_model::{assertion::AssertionFact, authored::ImportAttribution, user::UserOutcome};

pub(super) struct Restoration {
    pub fact: AssertionFact,
    pub claims: Vec<ImportAttribution>,
    pub current: ImportAttribution,
}

pub(super) fn restore_via(queue: &WriterQueue, restoration: Restoration) -> Result<UserOutcome> {
    let (reply, wait) = sync_channel(1);
    queue
        .send(Msg::RestoreAssertion(Box::new(restoration), reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

fn observed(conn: &Connection, restoration: &Restoration) -> Result<bool> {
    let Some(known) = crate::assertions::get(conn, &restoration.fact.id)? else {
        return Ok(false);
    };
    if known.fact != restoration.fact {
        return Err(StoreError::Conflict(restoration.fact.id.clone()));
    }
    let current = &restoration.current;
    Ok(known.imports.observed.iter().any(|prior| {
        prior.input == current.input
            && prior.chain == current.chain
            && prior.machine == current.machine
            && prior.actor == current.actor
    }) && restoration
        .claims
        .iter()
        .all(|claim| known.imports.declared.contains(claim)))
}

impl Inner {
    pub(super) fn restore_assertion(
        &mut self,
        mut restoration: Restoration,
    ) -> Result<UserOutcome> {
        restoration.fact = crate::assertions::canonical_fact(&restoration.fact)?;
        let mut outcome = UserOutcome {
            id: restoration.fact.id.clone(),
            repeat: false,
            pending: false,
        };
        if let Some(sinex) = self.sinex.as_ref().map(Arc::clone) {
            self.primary_cache_ready()?;
            if observed(&self.index, &restoration)? {
                outcome.repeat = true;
                return Ok(outcome);
            }
            // Retry the original admitted observation, including its first
            // ingress time. A later caller clock cannot change a frozen body.
            sinex.visit_pending_requests(&mut |pending| {
                if let polylogue_model::authored::PendingAuthored::RestoreAssertion {
                    fact,
                    claims,
                    current,
                } = pending
                    && fact == &restoration.fact
                    && claims == &restoration.claims
                    && current.input == restoration.current.input
                    && current.chain == restoration.current.chain
                    && current.machine == restoration.current.machine
                    && current.actor == restoration.current.actor
                {
                    restoration.current = current.clone();
                }
                Ok(())
            })?;
            let identity = format!(
                "assertion-restore:{}",
                blake3::hash(&serde_json::to_vec(&(
                    &restoration.fact,
                    &restoration.claims,
                    &restoration.current.input,
                    restoration.current.chain,
                    &restoration.current.machine,
                    &restoration.current.actor,
                ))?)
                .to_hex()
            );
            let writes = self.product_writes(vec![crate::primary::Product::RestoredAssertion {
                fact: restoration.fact.clone(),
                claims: restoration.claims.clone(),
                current: restoration.current.clone(),
            }])?;
            sinex.prepare_named_products(&identity, &writes)?;
            match self.poll_primary() {
                Ok(_) => outcome.pending = !observed(&self.index, &restoration)?,
                Err(StoreError::Sinex(_)) => outcome.pending = true,
                Err(error) => return Err(error),
            }
            return Ok(outcome);
        }
        if observed(&self.archive, &restoration)? {
            outcome.repeat = true;
            return Ok(outcome);
        }
        let tx = self.archive.transaction()?;
        let counter = bump_counter(&tx)?;
        crate::assertions::restore(
            &tx,
            &restoration.fact,
            &restoration.claims,
            &restoration.current,
            counter,
        )?;
        crate::write_lock::commit(tx)?;
        self.commits.committed();
        self.derive_due = true;
        Ok(outcome)
    }
}

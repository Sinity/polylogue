//! Assertions a bundle carried, restored unchanged through the same writer
//! queue as every other write.
use super::*;
use polylogue_model::{assertion::AssertionFact, user::UserOutcome};

pub(super) struct Restoration {
    pub fact: AssertionFact,
}

pub(super) fn restore_via(queue: &WriterQueue, restoration: Restoration) -> Result<UserOutcome> {
    let (reply, wait) = sync_channel(1);
    queue
        .send(Msg::RestoreAssertion(Box::new(restoration), reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

/// Whether the fact is already stored; a different fact under its ID is a
/// conflict.
fn stored(conn: &Connection, fact: &AssertionFact) -> Result<bool> {
    let Some(known) = crate::assertions::get(conn, &fact.id)? else {
        return Ok(false);
    };
    if known.fact != *fact {
        return Err(StoreError::Conflict(fact.id.clone()));
    }
    Ok(true)
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
            if stored(&self.index, &restoration.fact)? {
                outcome.repeat = true;
                return Ok(outcome);
            }
            let identity = format!(
                "assertion-restore:{}",
                blake3::hash(&serde_json::to_vec(&restoration.fact)?).to_hex()
            );
            let writes = self.product_writes(vec![crate::primary::Product::RestoredAssertion {
                fact: restoration.fact.clone(),
            }])?;
            sinex.prepare_named_products(&identity, &writes)?;
            outcome.pending = true;
            return Ok(outcome);
        }
        if stored(&self.archive, &restoration.fact)? {
            outcome.repeat = true;
            return Ok(outcome);
        }
        let tx = self.archive.transaction()?;
        let counter = bump_counter(&tx)?;
        crate::assertions::apply(&tx, std::slice::from_ref(&restoration.fact), counter)?;
        crate::write_lock::commit(tx)?;
        self.commits
            .committed(crate::completion::Committed::AUTHORED);
        Ok(outcome)
    }
}

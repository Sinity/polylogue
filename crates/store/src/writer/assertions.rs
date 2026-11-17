//! Assertions a bundle carried, restored unchanged through the same writer
//! queue as every other write.
use super::*;
use polylogue_model::{assertion::AssertionFact, user::UserOutcome};

pub(super) struct Restoration {
    pub fact: AssertionFact,
    pub import: Option<polylogue_model::authored::ImportAttribution>,
}

impl WriterSender {
    /// Restores an assertion and its separately observed import unchanged.
    pub fn restore_assertion(
        &self,
        fact: AssertionFact,
        import: Option<polylogue_model::authored::ImportAttribution>,
    ) -> Result<UserOutcome> {
        self.0
            .ask(|reply| Msg::RestoreAssertion(Box::new(Restoration { fact, import }), reply))
    }
}

/// Whether the assertion is already stored; a different assertion under its ID is a
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
            outcome.repeat = stored(&self.index, &restoration.fact)?;
            if outcome.repeat
                && match &restoration.import {
                    None => true,
                    Some(import) => crate::assertions::imports::contains(
                        &self.index,
                        &restoration.fact.id,
                        import,
                    )?,
                }
            {
                return Ok(outcome);
            }
            let identity = format!(
                "assertion-restore:{}",
                blake3::hash(&serde_json::to_vec(&(
                    &restoration.fact,
                    restoration
                        .import
                        .as_ref()
                        .map(|i| (&i.input, i.chain, &i.machine, &i.actor))
                ))?)
                .to_hex()
            );
            let writes = self.product_writes(vec![crate::primary::Product::RestoredAssertion {
                fact: restoration.fact.clone(),
                import: restoration.import.clone(),
            }])?;
            sinex.prepare_named_products(&identity, &writes)?;
            outcome.pending = true;
            return Ok(outcome);
        }
        outcome.repeat = stored(&self.archive, &restoration.fact)?;
        if outcome.repeat
            && match &restoration.import {
                None => true,
                Some(import) => crate::assertions::imports::contains(
                    &self.archive,
                    &restoration.fact.id,
                    import,
                )?,
            }
        {
            return Ok(outcome);
        }
        let tx = self.archive.transaction()?;
        let counter = bump_counter(&tx)?;
        if !outcome.repeat {
            crate::assertions::apply(&tx, std::slice::from_ref(&restoration.fact), counter)?;
        }
        if let Some(import) = &restoration.import {
            crate::assertions::imports::record(&tx, &restoration.fact.id, import, counter)?;
        }
        crate::write_lock::commit(tx)?;
        self.commits
            .committed(crate::completion::Committed::AUTHORED);
        Ok(outcome)
    }
}

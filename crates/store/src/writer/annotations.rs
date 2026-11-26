//! Assertions a bundle carried, restored unchanged through the same writer
//! queue as every other write.
use super::*;
use polylogue_model::{annotation::AnnotationFact, user::UserOutcome};

pub(super) struct Restoration {
    pub fact: AnnotationFact,
    pub import: Option<polylogue_model::authored::ImportAttribution>,
}

impl WriterSender {
    /// Restores an assertion and its separately observed import unchanged.
    pub fn restore_annotation(
        &self,
        fact: AnnotationFact,
        import: Option<polylogue_model::authored::ImportAttribution>,
    ) -> Result<UserOutcome> {
        self.0
            .ask(|reply| Msg::RestoreAnnotation(Box::new(Restoration { fact, import }), reply))
    }
}

/// Whether the assertion is already stored; a different assertion under its ID is a
/// conflict.
fn stored(conn: &Connection, fact: &AnnotationFact) -> Result<bool> {
    let Some(known) = crate::annotations::get(conn, &fact.id)? else {
        return Ok(false);
    };
    if known.fact != *fact {
        return Err(StoreError::Conflict(fact.id.clone()));
    }
    Ok(true)
}

impl Inner {
    pub(super) fn restore_annotation(
        &mut self,
        mut restoration: Restoration,
    ) -> Result<UserOutcome> {
        restoration.fact = crate::annotations::canonical_fact(&restoration.fact)?;
        let mut outcome = UserOutcome {
            id: restoration.fact.id.clone(),
            repeat: false,
            pending: false,
        };
        outcome.repeat = stored(&self.archive, &restoration.fact)?;
        if outcome.repeat
            && match &restoration.import {
                None => true,
                Some(import) => crate::annotations::imports::contains(
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
            crate::annotations::apply(&tx, std::slice::from_ref(&restoration.fact), counter)?;
        }
        if let Some(import) = &restoration.import {
            crate::annotations::imports::record(&tx, &restoration.fact.id, import, counter)?;
        }
        crate::write_lock::commit(tx)?;
        self.commits
            .committed(crate::completion::Committed::AUTHORED);
        Ok(outcome)
    }
}

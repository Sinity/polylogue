//! Requests and assertions a bundle carried, restored with their recorded
//! effects through the same writer as every other write; nothing in them is
//! resolved or run again.
use super::*;
use polylogue_model::annotation::AnnotationFact;
use polylogue_model::authored::{ImportAttribution, PreparedRequest};

impl ArchiveSender {
    /// Restores a request a bundle carried without resolving its intent again.
    pub fn restore_authored(&self, prepared: PreparedRequest) -> Result<UserOutcome> {
        restored(self.commit(Group::RestoredRequest(prepared))?)
    }

    /// Restores an assertion and its separately observed import unchanged.
    pub fn restore_annotation(
        &self,
        fact: AnnotationFact,
        import: Option<ImportAttribution>,
    ) -> Result<UserOutcome> {
        restored(self.commit(Group::RestoredAnnotation { fact, import })?)
    }
}

fn restored(committed: Committed<Outcome>) -> Result<UserOutcome> {
    match committed.outcome {
        Outcome::Restored(outcome) => Ok(outcome),
        Outcome::Stored(_) => Err(StoreError::internal("a restoration stored inputs")),
    }
}

/// Whether the request is already stored; a different request under its ID
/// is a conflict.
fn stored_request(conn: &Connection, prepared: &PreparedRequest) -> Result<bool> {
    let Some(known) = crate::authored_records::stored_request(conn, &prepared.request.id)? else {
        return Ok(false);
    };
    if known != *prepared {
        return Err(StoreError::Conflict(known.request.id));
    }
    Ok(true)
}

fn stored_annotation(conn: &Connection, fact: &AnnotationFact) -> Result<bool> {
    let Some(known) = crate::annotations::get(conn, &fact.id)? else {
        return Ok(false);
    };
    if known.fact != *fact {
        return Err(StoreError::Conflict(fact.id.clone()));
    }
    Ok(true)
}

impl Inner {
    pub(super) fn restore_request(
        &mut self,
        prepared: PreparedRequest,
    ) -> Result<Committed<UserOutcome>> {
        if stored_request(&self.conn, &prepared)? {
            let mut outcome = prepared.outcome;
            outcome.repeat = true;
            return Ok(Committed {
                position: self.position()?,
                outcome,
            });
        }
        let tx = self.begin()?;
        let counter = bump_counter(&tx)?;
        let outcome = crate::authored_records::restore(&tx, &prepared, counter)?;
        let position = self.commit(tx, counter, crate::completion::Committed::AUTHORED)?;
        Ok(Committed { position, outcome })
    }

    pub(super) fn restore_annotation(
        &mut self,
        fact: AnnotationFact,
        import: Option<ImportAttribution>,
    ) -> Result<Committed<UserOutcome>> {
        let fact = crate::annotations::canonical_fact(&fact)?;
        let mut outcome = UserOutcome {
            id: fact.id.clone(),
            repeat: false,
            pending: false,
        };
        outcome.repeat = stored_annotation(&self.conn, &fact)?;
        if outcome.repeat
            && match &import {
                None => true,
                Some(import) => {
                    crate::annotations::imports::contains(&self.conn, &fact.id, import)?
                }
            }
        {
            return Ok(Committed {
                position: self.position()?,
                outcome,
            });
        }
        let tx = self.begin()?;
        let counter = bump_counter(&tx)?;
        if !outcome.repeat {
            crate::annotations::apply(&tx, std::slice::from_ref(&fact), counter)?;
        }
        if let Some(import) = &import {
            crate::annotations::imports::record(&tx, &fact.id, import, counter)?;
        }
        let position = self.commit(tx, counter, crate::completion::Committed::AUTHORED)?;
        Ok(Committed { position, outcome })
    }
}

//! Portable predecessor evidence uses the existing restoration queue and custody.
use super::*;
use polylogue_model::{
    authored::{ImportAttribution, ImportHistory, PendingAuthored},
    historical::{AssertionEvidence, HistoricalRecord},
    user::UserOutcome,
};

pub(super) struct Restoration {
    pub evidence: HistoricalRecord,
    pub claims: Vec<ImportAttribution>,
    pub current: ImportAttribution,
}

pub(super) fn restore_via(queue: &WriterQueue, restoration: Restoration) -> Result<UserOutcome> {
    let (reply, wait) = sync_channel(1);
    queue
        .send(Msg::RestoreHistory(Box::new(restoration), reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

fn same_ingress(a: &ImportAttribution, b: &ImportAttribution) -> bool {
    a.input == b.input && a.chain == b.chain && a.machine == b.machine && a.actor == b.actor
}

fn observed(conn: &Connection, restoration: &Restoration) -> Result<bool> {
    let imports: ImportHistory = match &restoration.evidence {
        HistoricalRecord::Accepted(record) => {
            let Some(known) = crate::primary::authored::historical::get(conn, &record.id)? else {
                let source = crate::source::Source::reading(conn)?;
                if conn
                    .prepare_cached(
                        &source.sql("SELECT 1 FROM @source.client_request WHERE id=?1"),
                    )?
                    .exists([&record.id])?
                {
                    return Err(StoreError::Conflict(record.id.clone()));
                }
                return Ok(false);
            };
            if !crate::primary::authored::same_historical(&known, record) {
                return Err(StoreError::Conflict(record.id.clone()));
            }
            // Original metadata may add evidence even under an existing ingress.
            let mut merged = known.clone();
            if let (Some(old), Some(new)) = (&mut merged.original, &record.original) {
                crate::primary::authored::merge_original(old, new);
            }
            merged.native.declare(
                record
                    .native
                    .observed
                    .iter()
                    .chain(&record.native.declared)
                    .cloned(),
            );
            if merged != known {
                return Ok(false);
            }
            known.imports
        }
        HistoricalRecord::Assertion(fact) => {
            let (known, imports) = match crate::assertions::get_evidence(conn, &fact.id)? {
                Some(AssertionEvidence::Historical { fact, imports, .. }) => (fact, imports),
                Some(_) => return Err(StoreError::Conflict(fact.id.clone())),
                None => return Ok(false),
            };
            if known != **fact {
                return Err(StoreError::Conflict(fact.id.clone()));
            }
            imports
        }
        HistoricalRecord::Run(run) => {
            let Some((known, _)) = crate::query_objects::historical_run_on(conn, &run.id)? else {
                return Ok(false);
            };
            let merged = crate::query_objects::prepare_historical_run_restore(
                conn,
                run,
                &restoration.claims,
                &restoration.current,
            )?;
            return Ok(merged == known);
        }
    };
    if let HistoricalRecord::Accepted(record) = &restoration.evidence
        && record
            .imports
            .observed
            .iter()
            .chain(&record.imports.declared)
            .any(|claim| !imports.declared.contains(claim))
    {
        return Ok(false);
    }
    Ok(imports
        .observed
        .iter()
        .any(|entry| same_ingress(entry, &restoration.current))
        && restoration
            .claims
            .iter()
            .all(|entry| imports.declared.contains(entry)))
}

impl Inner {
    pub(super) fn restore_history(&mut self, mut restoration: Restoration) -> Result<UserOutcome> {
        restoration
            .evidence
            .validate()
            .map_err(|error| StoreError::Invalid(error.to_string()))?;
        let mut outcome = UserOutcome {
            id: restoration.evidence.id().into(),
            repeat: false,
            pending: false,
        };
        if let Some(sinex) = self.sinex.as_ref().map(Arc::clone) {
            self.primary_cache_ready()?;
            if observed(&self.index, &restoration)? {
                outcome.repeat = true;
                return Ok(outcome);
            }
            sinex.visit_pending_requests(&mut |pending| {
                if let PendingAuthored::RestoreHistory {
                    evidence,
                    claims,
                    current,
                } = pending
                    && evidence == &restoration.evidence
                    && claims == &restoration.claims
                    && same_ingress(current, &restoration.current)
                {
                    restoration.current = current.clone();
                }
                Ok(())
            })?;
            let identity = format!(
                "historical-restore:{}",
                BlobHash::of(&serde_json::to_vec(&(
                    &restoration.evidence,
                    &restoration.claims,
                    &restoration.current.input,
                    restoration.current.chain,
                    &restoration.current.machine,
                    &restoration.current.actor
                ))?)
            );
            let writes = self.product_writes(vec![crate::primary::Product::RestoredHistory {
                evidence: restoration.evidence.clone(),
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
        crate::primary::authored::historical::restore_record(
            &tx,
            &restoration.evidence,
            &restoration.claims,
            &restoration.current,
            counter,
        )?;
        crate::write_lock::commit(tx)?;
        self.commits.committed();
        self.project_variant_text()?;
        self.derive_due = true;
        Ok(outcome)
    }
}

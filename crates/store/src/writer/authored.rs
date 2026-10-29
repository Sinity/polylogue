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
    pub(super) fn restore_authored(&mut self, restoration: Restoration) -> Result<UserOutcome> {
        if let Some(sinex) = self.sinex.as_ref().map(Arc::clone) {
            let mut outcome = restoration.prepared.outcome.clone();
            if observed(&self.index, &restoration)? {
                outcome.repeat = true;
                return Ok(outcome);
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
        let tx = self.archive.transaction()?;
        let known =
            crate::primary::authored::known_request(&tx, &restoration.prepared.request)?.is_some();
        let counter = if known {
            tx.query_row(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'",
                [],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or(0)
        } else {
            bump_counter(&tx)?
        };
        let outcome = crate::primary::authored::restore(
            &tx,
            &restoration.prepared,
            &restoration.claims,
            &restoration.current,
            counter,
        )?;
        // Repeated content can still add a newly observed import provenance.
        crate::write_lock::commit(tx)?;
        let tx = self.index.transaction()?;
        crate::variants::project(&tx)?;
        tx.commit()?;
        self.derive_due = true;
        Ok(outcome)
    }
}

//! Current producer observations; the confirmed position remains owned by the cache.
use crate::{Result, Sinex};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Default)]
pub(crate) struct Health {
    pub context: crate::context::Cache,
    pub reachable: Option<bool>,
    pub family: Option<String>,
    /// The current read or connection outcome.
    pub error: Option<String>,
    /// The last complete publication retry, independent of reads.
    pub publication_error: Option<String>,
    /// This run's last acknowledgement on the monotonic clock, which
    /// retirement compares with when the owner head was observed. Status
    /// reads the spool's record instead, which survives a restart.
    pub last_ack: Option<Instant>,
    pub owner_position: Option<i64>,
    pub position_observed: Option<Instant>,
    pub owner_scope: Option<String>,
}

impl Health {
    fn error(&self) -> Option<String> {
        match (&self.publication_error, &self.error) {
            (Some(publication), Some(read)) if publication == read => {
                Some(format!("publication and read: {publication}"))
            }
            (Some(publication), Some(read)) => {
                Some(format!("publication: {publication}; read: {read}"))
            }
            (Some(publication), None) => Some(format!("publication: {publication}")),
            (None, read) => read.clone(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Status {
    pub endpoint: String,
    pub family: Option<String>,
    pub producer: String,
    pub reachable: Option<bool>,
    pub error: Option<String>,
    pub last_ack_seconds_ago: Option<f64>,
    pub owner_position: Option<i64>,
    pub owner_position_seconds_ago: Option<f64>,
    pub prepared_materials: u64,
    pub prepared_writes: u64,
    pub prepared_bytes: u64,
    pub refused_operations: u64,
    pub last_refusal: Option<String>,
}

impl Sinex {
    pub fn currency(
        &self,
        applied_position: Option<i64>,
        applied_scope: Option<String>,
    ) -> Result<polylogue_model::currency::IndexCurrency> {
        let health = self
            .health
            .lock()
            .map_err(|_| crate::Error::Invalid("producer health panicked".into()))?;
        let same_scope = applied_scope
            .as_ref()
            .is_some_and(|scope| health.owner_scope.as_ref() == Some(scope));
        Ok(polylogue_model::currency::IndexCurrency::primary(
            applied_position,
            same_scope.then_some(health.owner_position).flatten(),
            health
                .position_observed
                .map(|at| at.elapsed().as_secs_f64()),
            health.reachable,
            health.error().or_else(|| {
                (!same_scope)
                    .then(|| "the applied read scope has not been confirmed by Sinex".into())
            }),
        )
        .with_scope(applied_scope))
    }
    pub fn status(&self) -> Result<Status> {
        let (reachable, error, owner_position, owner_position_seconds_ago, family) = {
            let health = self
                .health
                .lock()
                .map_err(|_| crate::Error::Invalid("producer health panicked".into()))?;
            (
                health.reachable,
                health.error(),
                health.owner_position,
                health
                    .position_observed
                    .map(|at| at.elapsed().as_secs_f64()),
                health.family.clone(),
            )
        };
        let queue = self
            .spool
            .lock()
            .map_err(|_| crate::Error::Invalid("producer spool panicked".into()))?;
        // From the spool, so a restart during an outage still says how long
        // Sinex has been silent.
        let last_ack_seconds_ago = queue.last_acknowledged()?.map(|at| {
            std::time::SystemTime::now()
                .duration_since(at)
                .unwrap_or_default()
                .as_secs_f64()
        });
        let counts = queue.counts()?;
        let mut after = None;
        let mut refused_operations = 0;
        let mut last = None;
        loop {
            let page = queue.pending_after(after, 64)?;
            if page.is_empty() {
                break;
            }
            for entry in page {
                after = Some(entry.order);
                if entry.phase.is_refused() {
                    refused_operations += 1;
                    last = Some(entry.record.identity);
                }
            }
        }
        let last_refusal = last
            .map(|identity| queue.acknowledgment(&identity))
            .transpose()?
            .flatten()
            .and_then(|ack| ack.refusal);
        Ok(Status {
            endpoint: self.config.nats.clone(),
            family,
            producer: self.config.producer.clone(),
            reachable,
            error,
            last_ack_seconds_ago,
            owner_position,
            owner_position_seconds_ago,
            prepared_materials: counts.0,
            prepared_writes: counts.1,
            prepared_bytes: counts.2,
            refused_operations,
            last_refusal,
        })
    }
}

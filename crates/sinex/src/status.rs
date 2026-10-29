//! Current producer observations; the confirmed position remains owned by the cache.
use crate::{Result, Sinex};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Default)]
pub(crate) struct Health {
    pub reachable: Option<bool>,
    pub family: Option<String>,
    pub canonical_publisher: bool,
    pub error: Option<String>,
    pub last_ack: Option<Instant>,
    pub owner_position: Option<i64>,
    pub position_observed: Option<Instant>,
    pub owner_scope: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Status {
    pub endpoint: String,
    pub family: Option<String>,
    pub canonical_publisher: bool,
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
            health.error.clone().or_else(|| {
                (!same_scope)
                    .then(|| "the applied read scope has not been confirmed by Sinex".into())
            }),
        )
        .with_scope(applied_scope))
    }
    pub fn status(&self) -> Result<Status> {
        let (
            reachable,
            error,
            last_ack_seconds_ago,
            owner_position,
            owner_position_seconds_ago,
            family,
            canonical_publisher,
        ) = {
            let health = self
                .health
                .lock()
                .map_err(|_| crate::Error::Invalid("producer health panicked".into()))?;
            (
                health.reachable,
                health.error.clone(),
                health.last_ack.map(|at| at.elapsed().as_secs_f64()),
                health.owner_position,
                health
                    .position_observed
                    .map(|at| at.elapsed().as_secs_f64()),
                health.family.clone(),
                health.canonical_publisher,
            )
        };
        let queue = self
            .spool
            .lock()
            .map_err(|_| crate::Error::Invalid("producer spool panicked".into()))?;
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
                if entry.refused {
                    refused_operations += 1;
                    last = Some(entry.identity);
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
            canonical_publisher,
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

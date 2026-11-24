//! Current producer observations; the cache keeps the confirmed position.
use crate::{Result, Sinex};
use polylogue_model::LockExt;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Default)]
pub(crate) struct Health {
    pub context: crate::context::Cache,
    pub reachable: Option<bool>,
    pub family: Option<String>,
    /// The current read or connection outcome.
    pub error: Option<String>,
    /// The last complete send retry, independent of reads.
    pub send_error: Option<String>,
    /// This run's last acknowledgment on the monotonic clock, which
    /// confirmed removal compares with when the latest Sinex position was observed. Status
    /// reads the spool's record instead, which survives a restart.
    pub last_ack: Option<Instant>,
    pub sinex_position: Option<i64>,
    pub position_observed: Option<Instant>,
    pub sinex_scope: Option<String>,
}

impl Health {
    fn error(&self) -> Option<String> {
        match (&self.send_error, &self.error) {
            (Some(send), Some(read)) if send == read => Some(format!("send and read: {send}")),
            (Some(send), Some(read)) => Some(format!("send: {send}; read: {read}")),
            (Some(send), None) => Some(format!("send: {send}")),
            (None, read) => read.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Status {
    pub endpoint: String,
    pub family: Option<String>,
    pub producer: String,
    pub reachable: Option<bool>,
    pub error: Option<String>,
    pub last_ack_seconds_ago: Option<f64>,
    pub sinex_position: Option<i64>,
    pub sinex_position_seconds_ago: Option<f64>,
    pub prepared_materials: u64,
    pub prepared_writes: u64,
    pub prepared_bytes: u64,
    pub refused_operations: u64,
    pub last_refusal: Option<String>,
}

impl Sinex {
    pub fn freshness(
        &self,
        applied_position: Option<i64>,
        applied_scope: Option<String>,
    ) -> Result<polylogue_model::freshness::IndexFreshness> {
        let health = self.health.locked();
        let same_scope = applied_scope
            .as_ref()
            .is_some_and(|scope| health.sinex_scope.as_ref() == Some(scope));
        Ok(polylogue_model::freshness::IndexFreshness::primary(
            applied_position,
            same_scope.then_some(health.sinex_position).flatten(),
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
    /// The producer's observations: the spool's counts and refusals, read
    /// from the spool, and the connection's health ([`Self::observe`]).
    pub fn status(&self) -> Result<Status> {
        let queue = self.spool.locked();
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
            .and_then(|ack| ack.refusal().map(|(_, reason)| reason.to_owned()));
        let mut status = Status {
            endpoint: self.config.nats.clone(),
            family: None,
            producer: self.config.producer.clone(),
            reachable: None,
            error: None,
            last_ack_seconds_ago,
            sinex_position: None,
            sinex_position_seconds_ago: None,
            prepared_materials: counts.0,
            prepared_writes: counts.1,
            prepared_bytes: counts.2,
            refused_operations,
            last_refusal,
        };
        self.observe(&mut status);
        Ok(status)
    }

    /// Brings `status`'s connection health up to date from memory, without
    /// reading the spool: a status answer reads the spool's counts from its
    /// snapshot and the health as it is now.
    pub fn observe(&self, status: &mut Status) {
        let health = self.health.locked();
        status.reachable = health.reachable;
        status.error = health.error();
        status.sinex_position = health.sinex_position;
        status.sinex_position_seconds_ago = health
            .position_observed
            .map(|at| at.elapsed().as_secs_f64());
        status.family = health.family.clone();
    }
}

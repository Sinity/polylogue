//! Grant-checked payload reads and one durable confirmed-stream subscription.
use super::Client;
use crate::{Error, Result};
use async_nats::jetstream;
use futures_util::StreamExt;
use polylogue_store::primary::{
    self as local, Confirmed, ConfirmedChange, ReplayControl, ReplayState,
};
use serde::de::DeserializeOwned;
use sinex_protocol::{
    ConfirmedConsumer, ConfirmedConsumerReply, FetchEvent, FetchEventReply, FetchMaterial,
    FetchMaterialReply,
};
use std::time::Duration;

impl Client {
    pub(crate) async fn request<T: DeserializeOwned + Send + 'static>(
        &self,
        subject: String,
        body: impl serde::Serialize + Sync,
    ) -> Result<T> {
        sinex_transport::replies::request_json(
            &self.client,
            subject,
            &self.producer,
            &body,
            Duration::from_secs(120),
        )
        .await
        .map_err(|error| Error::Unreachable(error.to_string()))
    }

    pub(crate) async fn fetch_material(
        &self,
        material: sinex_model::MaterialId,
    ) -> Result<FetchMaterialReply> {
        let reply: FetchMaterialReply = self
            .request(
                self.subjects.fetch_material(&self.producer),
                FetchMaterial { material },
            )
            .await?;
        if let Some(reason) = &reply.refusal {
            return Err(Error::Refused {
                name: format!("material {material}"),
                reason: reason.clone(),
            });
        }
        let metadata = reply
            .material
            .as_ref()
            .ok_or_else(|| Error::Invalid("material fetch supplied no material".into()))?;
        if metadata.id != material {
            return Err(Error::Invalid("material fetch changed its identity".into()));
        }
        let hash = metadata
            .blob
            .strip_prefix("blake3:")
            .and_then(|hash| polylogue_model::BlobHash::parse(hash).ok())
            .ok_or_else(|| Error::Invalid("material has no BLAKE3 digest".into()))?;
        crate::material::validate(&polylogue_store::primary::Material {
            id: material.to_string(),
            hash,
            length: metadata.length,
            extents: reply.extents.clone(),
        })?;
        Ok(reply)
    }

    async fn fetch_event(
        &self,
        id: sinex_model::EventId,
        original_position: i64,
    ) -> Result<sinex_protocol::ConfirmedEvent> {
        let reply: FetchEventReply = self
            .request(
                self.subjects.fetch_event(&self.producer),
                FetchEvent { event: id },
            )
            .await?;
        if let Some(reason) = reply.refusal {
            return Err(Error::Refused {
                name: format!("event {id}"),
                reason,
            });
        }
        if let Some(erased) = reply.erased {
            if reply.event.is_some()
                || erased.event != id
                || erased.original_commit_seq != original_position
                || erased.commit_seq <= original_position
            {
                return Err(Error::Invalid(
                    "event erasure differs from its confirmed reference".into(),
                ));
            }
            return Err(Error::Erased {
                event: id.to_string(),
                position: erased.commit_seq,
            });
        }
        let event = reply
            .event
            .ok_or_else(|| Error::Invalid("event fetch supplied no event".into()))?;
        if event.id != id || !event.projection.verifies(&event.payload) {
            return Err(Error::Invalid(
                "event fetch changed identity or projected payload hash".into(),
            ));
        }
        Ok(event)
    }

    fn decode_event(&self, event: &sinex_protocol::ConfirmedEvent) -> Result<Confirmed> {
        let declaration = crate::kinds::declarations("polylogue")
            .into_iter()
            .find(|declaration| {
                declaration.kind.name == event.event_type
                    && declaration.kind.version == event.kind_version
            })
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "unsupported indexed kind {}@{}",
                    event.event_type, event.kind_version
                ))
            })?;
        if declaration.definition_hash() != event.kind_definition_hash {
            return Err(Error::Invalid(
                "fetched indexed kind differs from the supported immutable schema".into(),
            ));
        }
        let product = crate::products::decode(&event.event_type, event.payload.clone())?;
        if matches!(
            &product,
            local::Product::Session { .. }
                | local::Product::Message { .. }
                | local::Product::Event { .. }
                | local::Product::Link { .. }
                | local::Product::Background { .. }
        ) && (event.source_ref.as_ref() != Some(&crate::source_reference())
            || event.source_fingerprint.as_deref()
                != Some(crate::kinds::source_declaration().fingerprint().as_str())
            || event.producer_family.as_deref() != Some("polylogue")
            || event
                .producer_identity
                .as_ref()
                .is_none_or(String::is_empty))
        {
            return Err(Error::Invalid(
                "raw membership input has no supported authenticated native origin".into(),
            ));
        }

        Ok(Confirmed {
            position: event.commit_seq,
            event: event.id.to_string(),
            output_index: event.output_index,
            native: Some(local::NativeFactContext {
                kind: local::NativeKind {
                    name: event.event_type.clone(),
                    version: event.kind_version,
                },
                kind_definition_hash: event.kind_definition_hash.clone(),
                projection: local::ProjectionIdentity {
                    policy_epoch: event.projection.policy_epoch,
                    payload_hash: event.projection.payload_hash.clone(),
                },
                origin: local::ConfirmedOrigin {
                    source: event.source_ref.as_ref().map(|source| local::NativeSource {
                        name: source.name.clone(),
                        version: source.version,
                    }),
                    source_fingerprint: event.source_fingerprint.clone(),
                    producer_family: event.producer_family.clone(),
                    producer_identity: event.producer_identity.clone(),
                },
            }),
            change: ConfirmedChange::Live(Box::new(product)),
        })
    }

    pub(crate) async fn confirmed(
        &mut self,
        state: &ReplayState,
        cache: &str,
    ) -> Result<Vec<Confirmed>> {
        if !self.pending.is_empty() {
            return Err(Error::Invalid(
                "the previous native page has not been committed locally".into(),
            ));
        }
        self.publication_scope = None;
        self.publication_epoch = None;
        let result = if state.coverage.is_none()
            || state.availability != local::ProjectionAvailability::Certified
        {
            self.snapshot_page(state, cache).await
        } else {
            self.native_page(state, cache).await
        };
        match result {
            Ok(page) => {
                if page.iter().any(|event| matches!(&event.change,ConfirmedChange::Control(control)
                    if matches!(control.as_ref(),ReplayControl::Reset{..}
                        |ReplayControl::BasisStatus{status:local::ReplayBasisStatus::Invalidated{..}}
                        |ReplayControl::Coverage{coverage:local::RawInputCoverage::Incomplete{basis:local::ReplayBasisStatus::Invalidated{..},..}}))) {
                    self.spool.lock().map_err(|_|Error::Invalid("producer spool panicked".into()))?.revoke_canonical()?;
                    self.fence_publication();
                }
                Ok(page)
            }
            Err(error) => {
                for message in &self.pending {
                    let _ = message
                        .ack_with(jetstream::AckKind::Nak(Some(Duration::from_secs(1))))
                        .await;
                }
                self.pending.clear();
                if let Ok(mut health) = self.health.lock() {
                    health.error = Some(error.to_string());
                }
                let control = match error {
                    Error::Erased { position, .. } => ReplayControl::ErasedThrough { position },
                    _ => ReplayControl::BasisStatus {
                        status: local::ReplayBasisStatus::Unverified,
                    },
                };
                Ok(vec![control_fact(control)])
            }
        }
    }

    async fn snapshot_page(&mut self, state: &ReplayState, cache: &str) -> Result<Vec<Confirmed>> {
        self.consumer = None;
        self.consumer_cache = None;
        let reply: sinex_protocol::ReplayReply = self
            .request(
                self.subjects.replay_scope(&self.producer),
                sinex_protocol::ReplayQuery::Inputs {
                    cache: cache.into(),
                    cursor: state.cursor.as_ref().map(|cursor| cursor.cursor.clone()),
                },
            )
            .await?;
        let sinex_protocol::ReplayReply::Inputs { result } = reply else {
            return Err(Error::Invalid(
                "indexed replay returned another replay mode".into(),
            ));
        };
        let sinex_protocol::InputReplayReply::Page {
            position,
            policy_epoch,
            scope,
            inputs,
            continuation,
        } = result
        else {
            let sinex_protocol::InputReplayReply::Incomplete { basis, .. } = result else {
                unreachable!()
            };
            return Ok(vec![control_fact(ReplayControl::BasisStatus {
                status: basis_status(basis),
            })]);
        };
        if position < 0 || policy_epoch < 0 || scope.is_empty() {
            return Err(Error::Invalid(
                "indexed replay has no actual owner identity".into(),
            ));
        }
        if state.scope.as_deref() != Some(scope.as_str())
            || state.policy_epoch != Some(policy_epoch)
            || (state.coverage.is_some() && state.cursor.is_none())
        {
            return Ok(vec![control_fact(ReplayControl::Reset {
                scope,
                policy_epoch,
            })]);
        }
        let mut page = Vec::new();
        for reference in inputs {
            if reference.commit_seq > position || reference.projection.policy_epoch != policy_epoch
            {
                return Err(Error::Invalid(
                    "indexed replay reference lies outside its pinned view".into(),
                ));
            }
            let event = self
                .fetch_event(reference.event, reference.commit_seq)
                .await?;
            if event.commit_seq != reference.commit_seq
                || event.projection != reference.projection
                || event.event_type != reference.kind.name
                || event.kind_version != reference.kind.version
                || event.kind_definition_hash != reference.kind_definition_hash
                || event.source_ref != reference.source
                || event.source_fingerprint != reference.source_fingerprint
            {
                return Err(Error::Invalid(
                    "fetched indexed record differs from its exact replay reference".into(),
                ));
            }
            page.push(self.decode_event(&event)?);
            if reference.archived {
                let archived = reference
                    .lifecycle_position
                    .filter(|value| *value > 0 && *value <= position)
                    .ok_or_else(|| {
                        Error::Invalid(
                            "archived replay record has no actual lifecycle position".into(),
                        )
                    })?;
                page.push(Confirmed {
                    position: archived,
                    event: reference.event.to_string(),
                    output_index: 0,
                    native: None,
                    change: ConfirmedChange::Archived,
                });
            }
        }
        let continuation = match continuation {
            sinex_protocol::ReplayContinuation::Next { cursor } => {
                local::ReplayContinuation::Next { cursor }
            }
            sinex_protocol::ReplayContinuation::Complete {
                projection,
                raw_inputs,
            } => {
                let catalog = crate::kinds::projection_catalog()?;
                if projection.catalog != catalog.fingerprint() {
                    return Err(Error::Invalid(
                        "owner full projection catalog differs from the supported local projectors"
                            .into(),
                    ));
                }
                local::ReplayContinuation::Complete {
                    projection: local::FullProjectionComplete {
                        stamp: projection.stamp,
                        position: projection.position,
                        scope: projection.scope,
                        catalog: projection.catalog,
                    },
                    raw_inputs: raw_coverage(raw_inputs),
                }
            }
        };
        page.push(control_fact(ReplayControl::Page {
            page: local::ReplayPage {
                position,
                policy_epoch,
                scope,
                continuation,
            },
        }));
        Ok(page)
    }

    async fn coverage(
        &mut self,
        state: &ReplayState,
        cache: &str,
        position: i64,
        completion: Option<local::DeliveredCompletion>,
    ) -> Result<local::RawInputCoverage> {
        let basis = state
            .coverage
            .as_ref()
            .and_then(|coverage| match &coverage.raw_inputs {
                local::RawInputCoverage::Complete { proof } => {
                    Some(sinex_protocol::CoverageProof {
                        stamp: proof.stamp.clone(),
                        position: proof.position,
                        policy_epoch: proof.policy_epoch,
                        scope: proof.scope.clone(),
                    })
                }
                _ => None,
            });
        let reply: sinex_protocol::PublicContext = self
            .request(
                self.subjects.public_context(&self.producer),
                sinex_protocol::PublicContextRequest {
                    coverage: Some(sinex_protocol::CoverageRequest {
                        basis,
                        cache: cache.into(),
                        completed_position: position,
                        completion: completion.map(|receipt| sinex_protocol::DeliveredCompletion {
                            stream_instance: receipt.stream_instance,
                            stream_sequence: receipt.stream_sequence,
                        }),
                    }),
                },
            )
            .await?;
        let coverage = match &reply {
            sinex_protocol::PublicContext::Available { coverage, .. } => {
                raw_coverage(coverage.clone())
            }
            _ => local::RawInputCoverage::Incomplete {
                reason: "owner publication context is unavailable".into(),
                basis: local::ReplayBasisStatus::Unverified,
            },
        };
        if let Ok(mut health) = self.health.lock() {
            health.context.record(Ok(reply));
        }
        Ok(coverage)
    }

    async fn native_page(&mut self, state: &ReplayState, cache: &str) -> Result<Vec<Confirmed>> {
        let after = state.applied_position.ok_or_else(|| {
            Error::Invalid("native replay requires an applied full snapshot".into())
        })?;
        let coverage = state
            .coverage
            .as_ref()
            .ok_or_else(|| Error::Invalid("native replay has no full projection basis".into()))?;
        let reply: ConfirmedConsumerReply = self
            .request(
                self.subjects.confirmed_consumer(&self.producer),
                ConfirmedConsumer {
                    basis: Some(sinex_protocol::FullProjectionComplete {
                        stamp: coverage.projection.stamp.clone(),
                        position: coverage.projection.position,
                        scope: coverage.projection.scope.clone(),
                        catalog: coverage.projection.catalog.clone(),
                    }),
                    after,
                    cache: cache.into(),
                    status_only: false,
                },
            )
            .await?;
        if reply.refusal.is_some() || reply.gap.is_some() {
            return Err(Error::Refused {
                name: "confirmed consumer".into(),
                reason: reply.refusal.or(reply.gap).unwrap_or_default(),
            });
        }
        if reply.stream != self.subjects.confirmed_stream()
            || reply.consumer.is_empty()
            || reply.scope.is_empty()
        {
            return Err(Error::Invalid(
                "owner confirmed consumer identity is incomplete".into(),
            ));
        }
        if let Ok(mut health) = self.health.lock() {
            health.owner_position = reply.position;
            health.position_observed = Some(std::time::Instant::now());
            health.owner_scope = Some(reply.scope.clone());
        }
        if state.scope.as_deref() != Some(reply.scope.as_str()) {
            self.consumer = None;
            self.consumer_cache = None;
            return Ok(vec![control_fact(ReplayControl::BasisStatus {
                status: local::ReplayBasisStatus::Invalidated {
                    policy_epoch: None,
                    scope: Some(reply.scope),
                },
            })]);
        }
        let instance = reply
            .stream_instance
            .filter(|instance| !instance.is_empty())
            .ok_or_else(|| Error::Invalid("confirmed stream incarnation is unavailable".into()))?;
        let current = self
            .coverage(state, cache, after, state.completion.clone())
            .await?;
        match &current {
            local::RawInputCoverage::Complete { proof }
                if proof.position == after
                    && Some(proof.policy_epoch) == state.policy_epoch
                    && Some(&proof.scope) == state.scope.as_ref() =>
            {
                self.publication_scope = Some(proof.scope.clone());
                self.publication_epoch = Some(proof.policy_epoch);
                if !state.publication.as_ref().is_some_and(|old| {
                    old.position == proof.position
                        && old.policy_epoch == proof.policy_epoch
                        && old.scope == proof.scope
                }) {
                    return Ok(vec![control_fact(ReplayControl::Coverage {
                        coverage: current,
                    })]);
                }
            }
            _ => {
                return Ok(vec![control_fact(ReplayControl::Coverage {
                    coverage: current,
                })]);
            }
        }
        if self.consumer.is_none()
            || self.consumer_cache.as_deref() != Some(cache)
            || self.consumer_instance.as_deref() != Some(instance.as_str())
        {
            self.consumer = Some(
                jetstream::new(self.client.clone())
                    .get_consumer_from_stream(reply.consumer, reply.stream)
                    .await
                    .map_err(|error| Error::Unreachable(error.to_string()))?,
            );
            self.consumer_cache = Some(cache.into());
            self.consumer_scope = state.scope.clone();
            self.consumer_instance = Some(instance.clone());
        }
        let mut messages = self
            .consumer
            .as_ref()
            .expect("owner consumer")
            .fetch()
            .max_messages(256)
            .expires(Duration::from_millis(50))
            .messages()
            .await
            .map_err(|error| Error::Unreachable(error.to_string()))?;
        let mut page = Vec::new();
        let mut last = None;
        while let Some(message) = messages.next().await {
            let message = message.map_err(|error| Error::Unreachable(error.to_string()))?;
            let info = message
                .info()
                .map_err(|error| Error::Invalid(error.to_string()))?;
            if info.stream != self.subjects.confirmed_stream() {
                return Err(Error::Invalid(
                    "native delivery came from another stream".into(),
                ));
            }
            let sequence = info.stream_sequence;
            self.pending.push(message.clone());
            last = Some(local::NativeDeliveryCursor {
                stream_instance: instance.clone(),
                stream_sequence: sequence,
            });
            let change: sinex_protocol::Confirmed = serde_json::from_slice(&message.payload)?;
            let item = match change {
                sinex_protocol::Confirmed::EventReference { commit_seq, event }
                    if commit_seq > after =>
                {
                    let fetched = self.fetch_event(event, commit_seq).await?;
                    if fetched.commit_seq != commit_seq
                        || Some(fetched.projection.policy_epoch) != state.policy_epoch
                    {
                        return Err(Error::Invalid(
                            "native event differs from its confirmed position or policy epoch"
                                .into(),
                        ));
                    }
                    self.decode_event(&fetched)?
                }
                sinex_protocol::Confirmed::EventReference { .. } => continue,
                sinex_protocol::Confirmed::Archived(change) => Confirmed {
                    position: change.archived_seq,
                    event: change.event.to_string(),
                    output_index: 0,
                    native: None,
                    change: ConfirmedChange::Archived,
                },
                sinex_protocol::Confirmed::Unhidden(change) => Confirmed {
                    position: change.unhidden_seq,
                    event: change.event.to_string(),
                    output_index: 0,
                    native: None,
                    change: ConfirmedChange::Restored,
                },
                sinex_protocol::Confirmed::Excised {
                    event, commit_seq, ..
                } => Confirmed {
                    position: commit_seq.ok_or_else(|| {
                        Error::Invalid("native tombstone has no actual position".into())
                    })?,
                    event: event.to_string(),
                    output_index: 0,
                    native: None,
                    change: ConfirmedChange::Tombstoned,
                },
                sinex_protocol::Confirmed::Gap { reason } => {
                    return Err(Error::Invalid(format!("confirmed history gap: {reason}")));
                }
                sinex_protocol::Confirmed::PositionComplete { commit_seq } => {
                    if commit_seq < after {
                        continue;
                    }
                    let delivered = local::DeliveredCompletion {
                        stream_instance: instance.clone(),
                        stream_sequence: sequence,
                    };
                    page.push(control_fact(ReplayControl::Completion {
                        position: commit_seq,
                        delivered: delivered.clone(),
                    }));
                    let coverage = self
                        .coverage(state, cache, commit_seq, Some(delivered))
                        .await?;
                    page.push(control_fact(ReplayControl::Coverage { coverage }));
                    continue;
                }
            };
            if item.position > after {
                page.push(item);
            }
        }
        if let Some(delivery) = last {
            page.push(control_fact(ReplayControl::Delivered { delivery }));
        }
        Ok(page)
    }

    pub(crate) async fn acknowledge(&mut self, state: &ReplayState) -> Result<()> {
        for message in &self.pending {
            let info = message
                .info()
                .map_err(|error| Error::Invalid(error.to_string()))?;
            let committed = state.staged_delivery.as_ref().is_some_and(|delivery| {
                Some(delivery.stream_instance.as_str()) == self.consumer_instance.as_deref()
                    && delivery.stream_sequence >= info.stream_sequence
            });
            if !committed {
                return Err(Error::Invalid(
                    "native delivery has not been durably staged".into(),
                ));
            }
            message
                .ack()
                .await
                .map_err(|error| Error::Unreachable(error.to_string()))?;
        }
        self.client
            .flush()
            .await
            .map_err(|error| Error::Unreachable(error.to_string()))?;
        self.pending.clear();
        if let Some(position) = state.applied_position {
            self.spool
                .lock()
                .map_err(|_| Error::Invalid("producer spool panicked".into()))?
                .retire(position)?;
        }
        Ok(())
    }
}

fn control_fact(control: ReplayControl) -> Confirmed {
    Confirmed {
        position: 0,
        event: String::new(),
        output_index: 0,
        native: None,
        change: ConfirmedChange::Control(Box::new(control)),
    }
}
fn basis_status(status: sinex_protocol::ReplayBasisStatus) -> local::ReplayBasisStatus {
    match status {
        sinex_protocol::ReplayBasisStatus::Invalidated {
            policy_epoch,
            scope,
        } => local::ReplayBasisStatus::Invalidated {
            policy_epoch,
            scope,
        },
        sinex_protocol::ReplayBasisStatus::Unverified => local::ReplayBasisStatus::Unverified,
    }
}
fn raw_coverage(coverage: sinex_protocol::CoverageStatus) -> local::RawInputCoverage {
    match coverage {
        sinex_protocol::CoverageStatus::Complete { proof } => local::RawInputCoverage::Complete {
            proof: local::RawInputProof {
                stamp: proof.stamp,
                position: proof.position,
                policy_epoch: proof.policy_epoch,
                scope: proof.scope,
            },
        },
        sinex_protocol::CoverageStatus::Incomplete { reason, basis } => {
            local::RawInputCoverage::Incomplete {
                reason,
                basis: basis_status(basis),
            }
        }
        sinex_protocol::CoverageStatus::NotRequested => local::RawInputCoverage::NotRequested,
    }
}

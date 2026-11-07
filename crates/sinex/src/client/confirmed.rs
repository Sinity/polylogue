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

/// An event Polylogue cannot read (an unknown kind, a refused or
/// inconsistent fetch) is refused at its position so the stream continues;
/// only a lost connection stops the page.
pub(crate) fn readable(
    position: i64,
    event: &str,
    kind: String,
    definition_hash: Option<&str>,
    read: Result<Confirmed>,
) -> Result<Confirmed> {
    match read {
        Err(Error::Unreachable(reason)) => Err(Error::Unreachable(reason)),
        Err(error) => {
            // Only an exact known non-input declaration can exclude a hidden input.
            let known = kind.rsplit_once('@').and_then(|(name, version)| {
                crate::kinds::kind_reference(name).ok().filter(|kind| {
                    version.parse::<u32>() == Ok(kind.version)
                        && crate::kinds::declared_definition(kind).map(String::as_str)
                            == definition_hash
                })
            });
            let unreadable_input = known.is_none_or(|kind| kind.name == crate::kinds::INPUT_KIND);
            Ok(Confirmed {
                position,
                event: event.into(),
                output_index: 0,
                native: None,
                change: ConfirmedChange::Refused {
                    kind,
                    reason: error.to_string(),
                    unreadable_input,
                },
            })
        }
        read => read,
    }
}

fn decode_event(event: &sinex_protocol::ConfirmedEvent) -> Result<Confirmed> {
    let product = crate::products::decode_source(
        &sinex_model::KindRef {
            name: event.event_type.clone(),
            version: event.kind_version,
        },
        &event.kind_definition_hash,
        event.payload.clone(),
    )?;
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

    /// The event a stream reference names, or `None` when Sinex erased it
    /// after the reference was committed.
    async fn fetch_event(
        &self,
        id: sinex_model::EventId,
        original_position: i64,
    ) -> Result<Option<sinex_protocol::ConfirmedEvent>> {
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
            return Ok(None);
        }
        let event = reply
            .event
            .ok_or_else(|| Error::Invalid("event fetch supplied no event".into()))?;
        if event.id != id || !event.projection.verifies(&event.payload) {
            return Err(Error::Invalid(
                "event fetch changed identity or projected payload hash".into(),
            ));
        }
        Ok(Some(event))
    }

    /// One page from the durable consumer, read after the applied position
    /// (0 for a rebuild). A changed read scope is answered with a reset.
    pub(crate) async fn confirmed(
        &mut self,
        state: &ReplayState,
        cache: &str,
    ) -> Result<Vec<Confirmed>> {
        // A page that was not saved (its application failed) is delivered again.
        self.release_pending().await;
        let result = self.page(state, cache).await;
        if result.is_err() {
            self.release_pending().await;
        }
        result
    }

    async fn release_pending(&mut self) {
        for message in &self.pending {
            let _ = message
                .ack_with(jetstream::AckKind::Nak(Some(Duration::from_secs(1))))
                .await;
        }
        self.pending.clear();
    }

    async fn page(&mut self, state: &ReplayState, cache: &str) -> Result<Vec<Confirmed>> {
        let after = state.applied_position.unwrap_or(0);
        // The durable consumer is provisioned once per connection; later pages
        // only ask for the head and scope. Provisioning again would restart a
        // consumer whose acknowledgements are not at a completed position.
        let provisioned = self.consumer.is_some() && self.consumer_cache.as_deref() == Some(cache);
        let reply: ConfirmedConsumerReply = self
            .request(
                self.subjects.confirmed_consumer(&self.producer),
                ConfirmedConsumer {
                    basis: None,
                    after,
                    cache: cache.into(),
                    status_only: provisioned,
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
                "the confirmed consumer reply has no stream, consumer or scope".into(),
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
            return Ok(vec![control_fact(ReplayControl::Reset {
                scope: reply.scope,
            })]);
        }
        if !provisioned {
            let instance = reply
                .stream_instance
                .filter(|instance| !instance.is_empty())
                .ok_or_else(|| {
                    Error::Invalid("confirmed stream incarnation is unavailable".into())
                })?;
            self.consumer = Some(
                jetstream::new(self.client.clone())
                    .get_consumer_from_stream(reply.consumer, reply.stream)
                    .await
                    .map_err(|error| Error::Unreachable(error.to_string()))?,
            );
            self.consumer_cache = Some(cache.into());
            self.consumer_instance = Some(instance);
        }
        let instance = self
            .consumer_instance
            .clone()
            .expect("a provisioned consumer has its stream incarnation");
        let mut messages = self
            .consumer
            .as_ref()
            .expect("consumer was just provisioned")
            .fetch()
            .max_messages(256)
            .expires(Duration::from_millis(50))
            .messages()
            .await
            .map_err(|error| Error::Unreachable(error.to_string()))?;
        let mut page = Vec::new();
        if let Some(position) = reply.position {
            page.push(control_fact(ReplayControl::Head { position }));
        }
        let mut last = None;
        while let Some(message) = messages.next().await {
            let message = message.map_err(|error| Error::Unreachable(error.to_string()))?;
            let info = message
                .info()
                .map_err(|error| Error::Invalid(error.to_string()))?;
            if info.stream != self.subjects.confirmed_stream() {
                return Err(Error::Invalid(
                    "a stream delivery came from another stream".into(),
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
                    match self.fetch_event(event, commit_seq).await {
                        Ok(Some(fetched)) => {
                            let same_position = fetched.commit_seq == commit_seq;
                            let read = if same_position {
                                decode_event(&fetched)
                            } else {
                                Err(Error::Invalid(
                                    "a fetched event differs from its confirmed position".into(),
                                ))
                            };
                            readable(
                                commit_seq,
                                &event.to_string(),
                                format!("{}@{}", fetched.event_type, fetched.kind_version),
                                same_position.then_some(fetched.kind_definition_hash.as_str()),
                                read,
                            )?
                        }
                        Ok(None) => Confirmed {
                            position: commit_seq,
                            event: event.to_string(),
                            output_index: 0,
                            native: None,
                            change: ConfirmedChange::Tombstoned,
                        },
                        Err(error) => readable(
                            commit_seq,
                            &event.to_string(),
                            String::new(),
                            None,
                            Err(error),
                        )?,
                    }
                }
                sinex_protocol::Confirmed::EventReference { .. } => continue,
                sinex_protocol::Confirmed::Archived(change) => Confirmed {
                    position: change.archived_seq,
                    event: change.event.to_string(),
                    output_index: 0,
                    native: None,
                    change: ConfirmedChange::Archived,
                },
                sinex_protocol::Confirmed::Restored(change) => Confirmed {
                    position: change.restored_seq,
                    event: change.event.to_string(),
                    output_index: 0,
                    native: None,
                    change: ConfirmedChange::Restored,
                },
                sinex_protocol::Confirmed::Tombstoned {
                    event, commit_seq, ..
                } => Confirmed {
                    position: commit_seq
                        .ok_or_else(|| Error::Invalid("a tombstone has no position".into()))?,
                    event: event.to_string(),
                    output_index: 0,
                    native: None,
                    change: ConfirmedChange::Tombstoned,
                },
                sinex_protocol::Confirmed::Gap { reason } => {
                    return Err(Error::Invalid(format!("confirmed history gap: {reason}")));
                }
                sinex_protocol::Confirmed::PositionComplete { commit_seq } => {
                    if commit_seq >= after {
                        page.push(control_fact(ReplayControl::Completion {
                            position: commit_seq,
                        }));
                    }
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
        // A confirmed delivery may retire original producer bytes. Freeze must
        // cover that retirement until its immutable queue cut is acquired.
        let _publication = self.publication.enter().await;
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

#[cfg(test)]
mod tests;

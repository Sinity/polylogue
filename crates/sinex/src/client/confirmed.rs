//! One durable confirmed-stream subscription, and grant-checked reads of the
//! events the stream carries withheld.
use super::Client;
use crate::{Error, Result};
use async_nats::jetstream;
use futures_util::{StreamExt, stream};
use polylogue_model::LockExt;
use polylogue_store::primary::{
    self as local, Confirmed, ConfirmedChange, ReplayControl, ReplayState,
};
use serde::de::DeserializeOwned;
use sinex_protocol::{
    ConfirmedConsumer, ConfirmedConsumerReply, FetchEvent, FetchEventReply, FetchMaterial,
    FetchMaterialReply,
};
use std::time::Duration;
mod snapshot;

/// Fetches of withheld events in flight; results keep delivery order.
const FETCH_CONCURRENCY: usize = 16;

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
                    version.parse::<u64>() == Ok(kind.version)
                        && crate::kinds::declared_definition(kind) == definition_hash
                })
            });
            let unreadable_input = known.is_none_or(|kind| kind.name == crate::kinds::INPUT_KIND);
            Ok(Confirmed {
                position,
                event: event.into(),
                output_index: 0,
                sinex_context: None,
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

/// A complete event as the index stages it; an unreadable one is refused at
/// its position.
fn confirmed_event(event: &sinex_protocol::ConfirmedEvent) -> Result<Confirmed> {
    let mut fact = readable(
        event.commit_seq,
        &event.id.to_string(),
        format!("{}@{}", event.event_type, event.kind_version),
        Some(&event.kind_definition_hash),
        decode_event(event),
    )?;
    fact.sinex_context = Some(sinex_context(event));
    Ok(fact)
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
        sinex_context: Some(sinex_context(event)),
        change: ConfirmedChange::Live(Box::new(product)),
    })
}

fn sinex_context(event: &sinex_protocol::ConfirmedEvent) -> local::SinexEventContext {
    local::SinexEventContext {
        kind: local::SinexKind {
            name: event.event_type.clone(),
            version: event.kind_version,
        },
        kind_definition_hash: event.kind_definition_hash.clone(),
        record_hash: Some(event.record_hash.clone()),
        location: event.location.as_ref().map(|location| {
            polylogue_model::sinex_event::SinexLocation {
                material: location.material.to_string(),
                anchor: location.record_offset,
            }
        }),
        payload_identity: local::PayloadIdentity {
            payload_hash: event.payload_identity.payload_hash.clone(),
        },
        origin: local::ConfirmedOrigin {
            source: event.source_ref.as_ref().map(|source| local::SinexSource {
                name: source.name.clone(),
                version: source.version,
            }),
            source_fingerprint: event.source_fingerprint.clone(),
            producer_family: event.producer_family.clone(),
            producer_identity: event.producer_identity.clone(),
        },
    }
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
    ) -> Result<sinex_protocol::MaterialFrame> {
        let mut request = FetchMaterial::new(material);
        request.byte_budget = 1;
        Ok(self.fetch_material_page(request).await?.material)
    }

    pub(crate) async fn fetch_material_page(
        &self,
        request: FetchMaterial,
    ) -> Result<crate::MaterialPage> {
        let reply: FetchMaterialReply = self
            .request(
                self.subjects.fetch_material(&self.producer),
                request.clone(),
            )
            .await?;
        let (metadata, offset, bytes, next_offset) = match reply {
            FetchMaterialReply::Refused(refusal) => {
                return Err(Error::Refused {
                    name: format!("material {}", request.material),
                    reason: refusal.reason,
                });
            }
            FetchMaterialReply::Material {
                material,
                offset,
                bytes,
                next_offset,
            } => (*material, offset, bytes, next_offset),
        };
        let end = request.end.unwrap_or(metadata.length);
        let length = end
            .checked_sub(request.offset)
            .ok_or_else(|| Error::invalid("material fetch has a reversed range"))?;
        let expected = length.min(request.byte_budget);
        let through = offset
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::invalid("material page offset overflow"))?;
        if metadata.id != request.material
            || offset != request.offset
            || end > metadata.length
            || request.byte_budget == 0
            || bytes.len() as u64 != expected
            || next_offset != (through < end).then_some(through)
        {
            return Err(Error::invalid(
                "material fetch changed its identity or byte range",
            ));
        }
        metadata
            .blob
            .strip_prefix("blake3:")
            .and_then(|hash| polylogue_model::BlobHash::parse(hash).ok())
            .ok_or_else(|| Error::invalid("material has no BLAKE3 digest"))?;
        Ok(crate::MaterialPage {
            material: metadata,
            offset,
            bytes,
            next_offset,
        })
    }

    /// The event a withheld message names, or `None` when Sinex erased it
    /// after it was committed.
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
        let event = match reply {
            FetchEventReply::Refused(refusal) => {
                return Err(Error::Refused {
                    name: format!("event {id}"),
                    reason: refusal.reason,
                });
            }
            FetchEventReply::Erased(erased) => {
                if erased.event != id
                    || erased.original_commit_seq != original_position
                    || erased.commit_seq <= original_position
                {
                    return Err(Error::invalid(
                        "event erasure differs from its confirmed position",
                    ));
                }
                return Ok(None);
            }
            FetchEventReply::Event(event) => *event,
        };
        if event.id != id || !event.payload_identity.verifies(&event.payload) {
            return Err(Error::invalid(
                "event fetch changed identity or projected payload hash",
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
        // only ask for the latest position and scope. Provisioning again would restart a
        // consumer whose acknowledgments are not at a completed position.
        let provisioned = self.consumer.is_some() && self.consumer_cache.as_deref() == Some(cache);
        let reply: ConfirmedConsumerReply = self
            .request(
                self.subjects.confirmed_consumer(&self.producer),
                ConfirmedConsumer {
                    after,
                    completion: state.completed_delivery.as_ref().map(|marker| {
                        sinex_protocol::DeliveredCompletion {
                            stream_instance: marker.stream_instance.clone(),
                            stream_sequence: marker.stream_sequence,
                            fingerprint: marker.fingerprint.clone(),
                        }
                    }),
                    // Until its first completion marker, the index names
                    // where its applied snapshot's read stopped.
                    start: state
                        .snapshot
                        .as_ref()
                        .filter(|_| state.completed_delivery.is_none())
                        .map(|snapshot| sinex_protocol::SnapshotStart {
                            stream_instance: snapshot.stream_instance.clone(),
                            last: snapshot.last.as_ref().map(|last| {
                                sinex_protocol::LastStreamMessage {
                                    sequence: last.sequence,
                                    fingerprint: last.fingerprint.clone(),
                                }
                            }),
                        }),
                    cache: cache.into(),
                    status_only: provisioned,
                },
            )
            .await?;
        let (position, stream, stream_instance, consumer, scope) = match reply {
            ConfirmedConsumerReply::SnapshotRequired {
                origin,
                position,
                scope,
                stream_instance,
            } => {
                return self
                    .snapshot_page(state, origin, position, scope, stream_instance)
                    .await;
            }
            ConfirmedConsumerReply::Refused { reason }
            | ConfirmedConsumerReply::Gap { reason, .. } => {
                return Err(Error::Refused {
                    name: "confirmed consumer".into(),
                    reason,
                });
            }
            ConfirmedConsumerReply::Provisioned {
                position,
                stream,
                stream_instance,
                consumer,
                scope,
            } => (position, stream, stream_instance, consumer, scope),
        };
        if stream != self.subjects.confirmed_stream() || consumer.is_empty() || scope.is_empty() {
            return Err(Error::invalid(
                "the confirmed consumer reply has no stream, consumer or scope",
            ));
        }
        if state.snapshot_progress.is_some() {
            self.consumer = None;
            self.consumer_cache = None;
            self.consumer_instance = None;
            return Ok(vec![control_event(ReplayControl::Reset { scope })]);
        }
        {
            let mut health = self.health.locked();
            health.sinex_position = Some(position);
            health.position_observed = Some(std::time::Instant::now());
            health.sinex_scope = Some(scope.clone());
        }
        if state.scope.as_deref() != Some(scope.as_str()) {
            self.consumer = None;
            self.consumer_cache = None;
            return Ok(vec![control_event(ReplayControl::Reset { scope })]);
        }
        if !provisioned {
            let instance = stream_instance
                .filter(|instance| !instance.is_empty())
                .ok_or_else(|| Error::invalid("confirmed stream instance is unavailable"))?;
            self.consumer = Some(
                jetstream::new(self.client.clone())
                    .get_consumer_from_stream(consumer, stream)
                    .await
                    .map_err(|error| Error::Unreachable(error.to_string()))?,
            );
            self.consumer_cache = Some(cache.into());
            self.consumer_instance = Some(instance);
        }
        let instance = self
            .consumer_instance
            .clone()
            .expect("a provisioned consumer has its stream instance");
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
        let mut changes = Vec::new();
        let mut last = None;
        while let Some(message) = messages.next().await {
            let message = message.map_err(|error| Error::Unreachable(error.to_string()))?;
            let info = message
                .info()
                .map_err(|error| Error::invalid(error.to_string()))?;
            if info.stream != self.subjects.confirmed_stream() {
                return Err(Error::invalid("a stream delivery came from another stream"));
            }
            let sequence = info.stream_sequence;
            self.pending.push(message.clone());
            last = Some(local::StreamCursor {
                stream_instance: instance.clone(),
                stream_sequence: sequence,
            });
            let marker = local::CompletedDelivery {
                stream_instance: instance.clone(),
                stream_sequence: sequence,
                fingerprint: sinex_protocol::DeliveredCompletion::fingerprint(
                    sequence,
                    info.published.unix_timestamp_nanos(),
                    message.subject.as_str(),
                    &message.payload,
                ),
            };
            changes.push((
                serde_json::from_slice::<sinex_protocol::Confirmed>(&message.payload)?,
                marker,
            ));
        }
        let mut page = vec![control_event(ReplayControl::LatestPosition { position })];
        let reader = &*self;
        let mut reads = stream::iter(changes.into_iter().map(|(change, marker)| async move {
            let mut fact = reader.read_change(change, after).await?;
            if let Some(Confirmed {
                change: ConfirmedChange::Control(control),
                ..
            }) = &mut fact
                && let ReplayControl::Completion { delivery, .. } = control.as_mut()
            {
                *delivery = Some(marker);
            }
            Ok::<_, Error>(fact)
        }))
        .buffered(FETCH_CONCURRENCY);
        while let Some(read) = reads.next().await {
            page.extend(read?);
        }
        drop(reads);
        if let Some(delivery) = last {
            page.push(control_event(ReplayControl::Delivered { delivery }));
        }
        Ok(page)
    }

    async fn read_change(
        &self,
        change: sinex_protocol::Confirmed,
        after: i64,
    ) -> Result<Option<Confirmed>> {
        let item = match change {
            sinex_protocol::Confirmed::Event(event) if event.commit_seq > after => {
                let event = *event;
                if !event.payload_identity.verifies(&event.payload) {
                    return Err(Error::invalid(
                        "a confirmed event's payload differs from its hash",
                    ));
                }
                confirmed_event(&event)?
            }
            sinex_protocol::Confirmed::Withheld(withheld) if withheld.commit_seq > after => {
                let (commit_seq, event) = (withheld.commit_seq, withheld.event);
                match self.fetch_event(event, commit_seq).await {
                    Ok(Some(fetched)) if fetched.commit_seq == commit_seq => {
                        confirmed_event(&fetched)?
                    }
                    Ok(Some(fetched)) => readable(
                        commit_seq,
                        &event.to_string(),
                        format!("{}@{}", fetched.event_type, fetched.kind_version),
                        None,
                        Err(Error::invalid(
                            "a fetched event differs from its confirmed position",
                        )),
                    )?,
                    Ok(None) => Confirmed {
                        position: commit_seq,
                        event: event.to_string(),
                        output_index: 0,
                        sinex_context: None,
                        change: ConfirmedChange::Tombstoned,
                    },
                    Err(error) => readable(
                        commit_seq,
                        &event.to_string(),
                        format!("{}@?", withheld.event_type),
                        None,
                        Err(error),
                    )?,
                }
            }
            sinex_protocol::Confirmed::Event(_) | sinex_protocol::Confirmed::Withheld(_) => {
                return Ok(None);
            }
            sinex_protocol::Confirmed::Archived(change) => Confirmed {
                position: change.archived_seq,
                event: change.event.to_string(),
                output_index: 0,
                sinex_context: None,
                change: ConfirmedChange::Archived,
            },
            sinex_protocol::Confirmed::Restored(change) => Confirmed {
                position: change.restored_seq,
                event: change.event.to_string(),
                output_index: 0,
                sinex_context: None,
                change: ConfirmedChange::Restored,
            },
            sinex_protocol::Confirmed::Tombstoned {
                event, commit_seq, ..
            } => Confirmed {
                position: commit_seq
                    .ok_or_else(|| Error::invalid("a tombstone has no position"))?,
                event: event.to_string(),
                output_index: 0,
                sinex_context: None,
                change: ConfirmedChange::Tombstoned,
            },
            sinex_protocol::Confirmed::Gap { reason } => {
                return Err(Error::invalid(format!("confirmed history gap: {reason}")));
            }
            sinex_protocol::Confirmed::PositionComplete { commit_seq } => {
                if commit_seq >= after {
                    return Ok(Some(control_event(ReplayControl::Completion {
                        position: commit_seq,
                        delivery: None,
                    })));
                }
                return Ok(None);
            }
        };
        Ok((item.position > after).then_some(item))
    }

    pub(crate) async fn acknowledge(
        &mut self,
        state: &ReplayState,
        applied: &dyn Fn(&str) -> Result<bool>,
    ) -> Result<()> {
        // A confirmed delivery may remove original producer bytes. Freeze must
        // cover that confirmed removal until its immutable queue position is acquired.
        let _publication = self.publication.enter().await;
        for message in &self.pending {
            let info = message
                .info()
                .map_err(|error| Error::invalid(error.to_string()))?;
            let committed = state.staged_delivery.as_ref().is_some_and(|delivery| {
                Some(delivery.stream_instance.as_str()) == self.consumer_instance.as_deref()
                    && delivery.stream_sequence >= info.stream_sequence
            });
            if !committed {
                return Err(Error::invalid(
                    "confirmed delivery has not been staged in the index",
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
        let confirmed_removal = {
            let health = self.health.locked();
            confirmed_removal_state(state, &health)
        };
        self.spool
            .locked()
            .remove_confirmed(&confirmed_removal, applied)?;
        Ok(())
    }
}

/// An acknowledgment without an admitted-event position needs a completed prefix
/// whose latest Sinex position was observed after the acknowledgment. Sinex
/// reads that position from its committed counter, not from the delivery broker.
fn confirmed_removal_state(state: &ReplayState, health: &crate::status::Health) -> ReplayState {
    let mut confirmed_removal = state.clone();
    let fresh_position = health.sinex_scope.as_ref() == state.scope.as_ref()
        && health.sinex_position == state.latest_position
        && health.position_observed.is_some_and(|observed| {
            health
                .last_ack
                .is_none_or(|acknowledged| observed >= acknowledged)
        });
    if !fresh_position {
        confirmed_removal.latest_position = None;
    }
    confirmed_removal
}

fn control_event(control: ReplayControl) -> Confirmed {
    Confirmed {
        position: 0,
        event: String::new(),
        output_index: 0,
        sinex_context: None,
        change: ConfirmedChange::Control(Box::new(control)),
    }
}

#[cfg(test)]
mod tests;

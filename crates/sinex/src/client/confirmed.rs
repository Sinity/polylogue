//! Grant-checked payload reads and one durable confirmed-stream subscription.
use super::Client;
use crate::{Error, Result};
use async_nats::jetstream;
use futures_util::StreamExt;
use polylogue_store::primary::{Confirmed, ConfirmedChange};
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
        let event = reply
            .event
            .ok_or_else(|| Error::Invalid("event fetch supplied no event".into()))?;
        if event.id != id || event.payload.is_none() {
            return Err(Error::Invalid(
                "event fetch changed identity or withheld complete facts".into(),
            ));
        }
        Ok(event)
    }

    async fn full_event(
        &self,
        event: &sinex_protocol::ConfirmedEvent,
    ) -> Result<sinex_protocol::ConfirmedEvent> {
        if event.payload.is_some() {
            return Ok(event.clone());
        }
        let fetched = self.fetch_event(event.id).await?;
        if fetched.commit_seq != event.commit_seq
            || fetched.kind_version != event.kind_version
            || fetched.event_type != event.event_type
            || fetched.record_hash != event.record_hash
        {
            return Err(Error::Invalid(
                "fetched event differs from its confirmed identity".into(),
            ));
        }
        Ok(fetched)
    }

    async fn decode_event(
        &self,
        event: &sinex_protocol::ConfirmedEvent,
        after: i64,
    ) -> Result<Confirmed> {
        let change = if event.commit_seq > after && event.event_type.starts_with("polylogue.") {
            let expected = crate::kinds::kind_reference(&event.event_type)?;
            if expected.version != event.kind_version {
                return Err(Error::Invalid(format!(
                    "unsupported confirmed {} version {}, expected {}",
                    event.event_type, event.kind_version, expected.version
                )));
            }
            let event = self.full_event(event).await?;
            ConfirmedChange::Live(Box::new(crate::products::decode(
                &event.event_type,
                event.payload.expect("complete event"),
            )?))
        } else {
            ConfirmedChange::Ignored
        };
        Ok(Confirmed {
            position: event.commit_seq,
            event: event.id.to_string(),
            output_index: event.output_index,
            change,
        })
    }

    pub(crate) async fn confirmed(
        &mut self,
        after: i64,
        cache: &str,
        scope: Option<&str>,
    ) -> Result<Vec<Confirmed>> {
        let result = self.confirmed_page(after, cache, scope).await;
        if result.is_err() {
            for message in &self.pending {
                let _ = message
                    .ack_with(jetstream::AckKind::Nak(Some(Duration::from_secs(1))))
                    .await;
            }
            self.pending.clear();
        }
        result
    }

    async fn confirmed_page(
        &mut self,
        after: i64,
        cache: &str,
        scope: Option<&str>,
    ) -> Result<Vec<Confirmed>> {
        if !self.pending.is_empty() {
            return Err(Error::Invalid(
                "the previous confirmed page has not been committed locally".into(),
            ));
        }
        if self.consumer.is_none() || self.consumer_cache.as_deref() != Some(cache) {
            let reply: ConfirmedConsumerReply = self
                .request(
                    self.subjects.confirmed_consumer(&self.producer),
                    ConfirmedConsumer {
                        after,
                        cache: cache.into(),
                        status_only: false,
                    },
                )
                .await?;
            if let Some(reason) = reply.gap {
                return Err(Error::Invalid(format!("confirmed history gap: {reason}")));
            }
            if let Some(reason) = reply.refusal {
                return Err(Error::Refused {
                    name: "confirmed subscription".into(),
                    reason,
                });
            }
            if reply.stream != self.subjects.confirmed_stream()
                || reply.consumer.is_empty()
                || reply.scope.is_empty()
            {
                return Err(Error::Invalid(
                    "Sinex supplied another consumer identity or an absent scope".into(),
                ));
            }
            if let Ok(mut health) = self.health.lock() {
                health.owner_position = reply.position;
                health.position_observed = Some(std::time::Instant::now());
                health.owner_scope = Some(reply.scope.clone());
            }
            self.consumer = Some(
                jetstream::new(self.client.clone())
                    .get_consumer_from_stream(reply.consumer, reply.stream)
                    .await
                    .map_err(|error| {
                        Error::Unreachable(format!("confirmed subscription: {error}"))
                    })?,
            );
            self.consumer_cache = Some(cache.into());
            self.consumer_scope = Some(reply.scope);
        }
        if self.consumer_scope.as_deref() != scope {
            self.publication_scope = None;
            return Ok(scope_reset(
                self.consumer_scope.clone().expect("provisioned scope"),
            ));
        }
        let refresh = self
            .health
            .lock()
            .map_err(|_| Error::Invalid("producer health panicked".into()))?
            .position_observed
            .is_none_or(|at| at.elapsed() >= Duration::from_secs(1));
        if refresh {
            let reply: ConfirmedConsumerReply = self
                .request(
                    self.subjects.confirmed_consumer(&self.producer),
                    ConfirmedConsumer {
                        after,
                        cache: cache.into(),
                        status_only: true,
                    },
                )
                .await?;
            if let Some(reason) = reply.refusal {
                return Err(Error::Refused {
                    name: "confirmed status".into(),
                    reason,
                });
            }
            if let Some(reason) = reply.gap {
                return Err(Error::Invalid(format!("confirmed history gap: {reason}")));
            }
            if reply.scope.is_empty() {
                return Err(Error::Invalid(
                    "Sinex supplied an absent confirmed scope".into(),
                ));
            }
            if let Ok(mut health) = self.health.lock() {
                health.owner_position = reply.position;
                health.position_observed = Some(std::time::Instant::now());
                health.owner_scope = Some(reply.scope.clone());
            }
            if self.consumer_scope.as_deref() != Some(reply.scope.as_str()) {
                self.consumer = None;
                self.consumer_cache = None;
                self.consumer_scope = Some(reply.scope.clone());
                self.publication_scope = None;
                return Ok(scope_reset(reply.scope));
            }
        }
        self.publication_scope = self.consumer_scope.clone();
        let mut messages = self
            .consumer
            .as_ref()
            .expect("confirmed consumer")
            .fetch()
            .max_messages(256)
            .expires(Duration::from_millis(50))
            .messages()
            .await
            .map_err(|error| Error::Unreachable(format!("confirmed read: {error}")))?;
        let mut page = Vec::new();
        while let Some(message) = messages.next().await {
            let message = message
                .map_err(|error| Error::Unreachable(format!("confirmed message: {error}")))?;
            self.pending.push(message.clone());
            let change: sinex_protocol::Confirmed = serde_json::from_slice(&message.payload)?;
            let confirmed = match change {
                sinex_protocol::Confirmed::Event(event) => self.decode_event(&event, after).await?,
                sinex_protocol::Confirmed::EventReference { commit_seq, event } => {
                    if commit_seq <= after {
                        Confirmed {
                            position: commit_seq,
                            event: event.to_string(),
                            output_index: 0,
                            change: ConfirmedChange::Ignored,
                        }
                    } else {
                        let fetched = self.fetch_event(event).await?;
                        if fetched.commit_seq != commit_seq {
                            return Err(Error::Invalid(
                                "event reference changed its confirmed position".into(),
                            ));
                        }
                        self.decode_event(&fetched, after).await?
                    }
                }
                sinex_protocol::Confirmed::Archived(change) => Confirmed {
                    position: change.archived_seq,
                    event: change.event.to_string(),
                    output_index: 0,
                    change: ConfirmedChange::Archived,
                },
                sinex_protocol::Confirmed::Unhidden(change) => Confirmed {
                    position: change.unhidden_seq,
                    event: change.event.to_string(),
                    output_index: 0,
                    change: ConfirmedChange::Restored,
                },
                sinex_protocol::Confirmed::Excised {
                    event, commit_seq, ..
                } => Confirmed {
                    position: commit_seq.ok_or_else(|| {
                        Error::Invalid(format!(
                            "confirmed history gap: tombstone {event} has no commit position"
                        ))
                    })?,
                    event: event.to_string(),
                    output_index: 0,
                    change: ConfirmedChange::Tombstoned,
                },
                sinex_protocol::Confirmed::Gap { reason } => {
                    return Err(Error::Invalid(format!("confirmed history gap: {reason}")));
                }
                sinex_protocol::Confirmed::PositionComplete { commit_seq } => Confirmed {
                    position: commit_seq,
                    event: String::new(),
                    output_index: 0,
                    change: ConfirmedChange::PositionComplete,
                },
            };
            page.push(
                if confirmed.position <= after
                    && !matches!(confirmed.change, ConfirmedChange::PositionComplete)
                {
                    Confirmed {
                        change: ConfirmedChange::Ignored,
                        ..confirmed
                    }
                } else {
                    confirmed
                },
            );
        }
        Ok(page)
    }

    /// The caller committed this page to durable staging or the read index.
    /// Only the completed scalar position authorizes producer-spool retirement.
    pub(crate) async fn acknowledge(&mut self, position: i64) -> Result<()> {
        for message in &self.pending {
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
        self.spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?
            .retire(position)?;
        Ok(())
    }
}

fn scope_reset(scope: String) -> Vec<Confirmed> {
    vec![Confirmed {
        position: 0,
        event: String::new(),
        output_index: 0,
        change: ConfirmedChange::ScopeReset { scope },
    }]
}

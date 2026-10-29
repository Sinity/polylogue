//! Grant-checked payload reads and one durable confirmed-stream subscription.
use super::Client;
use crate::{Error, Result};
use async_nats::jetstream::{self, consumer};
use base64::Engine;
use futures_util::StreamExt;
use polylogue_store::primary::{Confirmed, ConfirmedChange};
use serde::de::DeserializeOwned;
use sinex_protocol::{
    FetchEvent, FetchEventReply, FetchPayload, FetchPayloadReply, PayloadReference,
};
use std::io::{Seek, Write};
use std::time::Duration;

impl Client {
    async fn request<T: DeserializeOwned>(
        &self,
        subject: String,
        body: impl serde::Serialize,
    ) -> Result<T> {
        let reply = self
            .client
            .request(subject, serde_json::to_vec(&body)?.into())
            .await
            .map_err(|error| Error::Unreachable(error.to_string()))?;
        Ok(serde_json::from_slice(&reply.payload)?)
    }

    /// A referenced value is transferred in bounded units and checked before
    /// any of it reaches the read index. A privacy change invalidates the hash.
    async fn fetch_json<T: DeserializeOwned>(&self, reference: &PayloadReference) -> Result<T> {
        let mut file = tempfile::tempfile()?;
        let mut digest = blake3::Hasher::new();
        let mut offset = 0u64;
        while offset < reference.length {
            let length = (reference.length - offset)
                .min((self.max_message.saturating_sub(4096) * 3 / 4).max(1) as u64);
            let reply: FetchPayloadReply = self
                .request(
                    self.subjects.fetch_payload(&self.producer),
                    FetchPayload {
                        event: reference.event,
                        offset,
                        length,
                        hash: reference.hash.clone(),
                    },
                )
                .await?;
            if let Some(reason) = reply.refusal {
                return Err(Error::Refused {
                    name: format!("payload {}", reference.event),
                    reason,
                });
            }
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(reply.data)
                .map_err(|error| Error::Invalid(format!("payload transfer: {error}")))?;
            if reply.offset != offset
                || reply.total != reference.length
                || reply.hash != reference.hash
                || bytes.is_empty()
                || bytes.len() as u64 > reference.length - offset
            {
                return Err(Error::Invalid(
                    "payload transfer changed its range or identity".into(),
                ));
            }
            file.write_all(&bytes)?;
            digest.update(&bytes);
            offset += bytes.len() as u64;
        }
        if reference.hash != format!("blake3:{}", digest.finalize().to_hex()) {
            return Err(Error::Invalid(
                "payload transfer failed its digest check".into(),
            ));
        }
        file.rewind()?;
        let mut deserializer = serde_json::Deserializer::from_reader(file);
        deserializer.disable_recursion_limit();
        let value = T::deserialize(&mut deserializer)?;
        deserializer.end()?;
        Ok(value)
    }

    async fn full_event(
        &self,
        event: &sinex_protocol::ConfirmedEvent,
    ) -> Result<sinex_protocol::ConfirmedEvent> {
        if event.payload.is_some() {
            return Ok(event.clone());
        }
        let reply: FetchEventReply = self
            .request(
                self.subjects.fetch_event(&self.producer),
                FetchEvent { event: event.id },
            )
            .await?;
        if let Some(reason) = reply.refusal {
            return Err(Error::Refused {
                name: format!("event {}", event.id),
                reason,
            });
        }
        let mut fetched = if let Some(reference) = &reply.event_ref {
            self.fetch_json::<sinex_protocol::ConfirmedEvent>(reference)
                .await?
        } else {
            reply
                .event
                .ok_or_else(|| Error::Invalid("event fetch supplied no event".into()))?
        };
        if let Some(reference) = &reply.payload_ref {
            fetched.payload = Some(self.fetch_json(reference).await?);
            fetched.withheld = None;
        }
        if fetched.id != event.id
            || fetched.commit_seq != event.commit_seq
            || fetched.kind_version != event.kind_version
            || fetched.event_type != event.event_type
            || fetched.record_hash != event.record_hash
            || fetched.payload.is_none()
        {
            return Err(Error::Invalid(
                "fetched event differs from its confirmed identity or has withheld facts".into(),
            ));
        }
        Ok(fetched)
    }

    pub(crate) async fn confirmed(&mut self, after: i64) -> Result<Vec<Confirmed>> {
        let result = self.confirmed_page(after).await;
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

    async fn confirmed_page(&mut self, after: i64) -> Result<Vec<Confirmed>> {
        if !self.pending.is_empty() {
            return Err(Error::Invalid(
                "the previous confirmed page has not been committed locally".into(),
            ));
        }
        if self.consumer.is_none() {
            let stream = jetstream::new(self.client.clone())
                .get_stream(self.subjects.confirmed_stream())
                .await
                .map_err(|error| Error::Unreachable(format!("confirmed stream: {error}")))?;
            // A missing index starts from the retained history even if the
            // server remembers an older copy's acknowledged subscription.
            if after == 0 {
                match stream
                    .get_consumer::<consumer::pull::Config>(&self.consumer_name)
                    .await
                {
                    Ok(_) => {
                        stream
                            .delete_consumer(&self.consumer_name)
                            .await
                            .map_err(|error| {
                                Error::Unreachable(format!("reset confirmed subscription: {error}"))
                            })?;
                    }
                    Err(error) if matches!(error.kind(),jetstream::stream::ConsumerErrorKind::JetStream(error) if error.error_code()==jetstream::errors::ErrorCode::CONSUMER_NOT_FOUND || error.error_code()==jetstream::errors::ErrorCode::CONSUMER_DOES_NOT_EXIST) =>
                        {}
                    Err(error) => {
                        return Err(Error::Unreachable(format!(
                            "confirmed subscription: {error}"
                        )));
                    }
                }
            }
            self.consumer = Some(
                stream
                    .get_or_create_consumer(
                        &self.consumer_name,
                        consumer::pull::Config {
                            durable_name: Some(self.consumer_name.clone()),
                            deliver_policy: consumer::DeliverPolicy::All,
                            ack_policy: consumer::AckPolicy::Explicit,
                            max_deliver: -1,
                            ack_wait: Duration::from_secs(120),
                            filter_subject: self.subjects.confirmed_all(),
                            ..Default::default()
                        },
                    )
                    .await
                    .map_err(|error| {
                        Error::Unreachable(format!("confirmed subscription: {error}"))
                    })?,
            );
        }
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
                sinex_protocol::Confirmed::Event(event) => {
                    let position = event.commit_seq;
                    let id = event.id.to_string();
                    let change = if event.event_type.starts_with("polylogue.")
                        && !matches!(
                            event.event_type.as_str(),
                            "polylogue.tool_call" | "polylogue.turn"
                        ) {
                        let event = self.full_event(&event).await?;
                        ConfirmedChange::Live(crate::products::decode(
                            &event.event_type,
                            event.payload.expect("complete event"),
                        )?)
                    } else {
                        ConfirmedChange::Ignored
                    };
                    Confirmed {
                        position,
                        event: id,
                        change,
                    }
                }
                sinex_protocol::Confirmed::Archived(change) => Confirmed {
                    position: change.archived_seq,
                    event: change.event.to_string(),
                    change: ConfirmedChange::Archived,
                },
                sinex_protocol::Confirmed::Unhidden(change) => Confirmed {
                    position: change.unhidden_seq,
                    event: change.event.to_string(),
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
                    change: ConfirmedChange::Tombstoned,
                },
                sinex_protocol::Confirmed::Gap { reason } => {
                    return Err(Error::Invalid(format!("confirmed history gap: {reason}")));
                }
                sinex_protocol::Confirmed::PositionComplete { commit_seq } => Confirmed {
                    position: commit_seq,
                    event: String::new(),
                    change: ConfirmedChange::PositionComplete,
                },
            };
            // Replaying a retained stream after NATS loss uses the saved
            // product position, independent of the stream's new sequences.
            if confirmed.position <= after {
                message
                    .ack()
                    .await
                    .map_err(|error| Error::Unreachable(error.to_string()))?;
                self.pending.pop();
            } else {
                page.push(confirmed);
            }
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

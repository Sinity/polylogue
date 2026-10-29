//! The producer protocol over NATS: frames to the ingress, their
//! acknowledgments, which frames Sinex recorded, and the materials Sinex holds from this producer.
//! A material's bytes, and a frame too large for one message, are written to
//! the frame's directory under Sinex's store root on this host; Sinex
//! stores them before it records the frame.
use crate::{Config, Error, Result};
mod confirmed;
use async_nats::jetstream;
use futures_util::StreamExt;
use sinex_protocol::{
    Acknowledgment, Ingress, ProducerHello, ProducerWelcome, RecordedQuery, Subjects,
};
use sinex_transport::frame_bytes;
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

/// How long a frame may wait for its acknowledgment.
const ACKNOWLEDGMENT_WAIT: Duration = Duration::from_secs(120);

/// Room left in a NATS message for its subject and headers.
const MESSAGE_OVERHEAD: usize = 4096;

pub(crate) struct Client {
    client: async_nats::Client,
    subjects: Subjects,
    producer: String,
    max_message: usize,
    pub(crate) store_root: PathBuf,
    acknowledgments: async_nats::Subscriber,
    consumer_name: String,
    consumer: Option<async_nats::jetstream::consumer::PullConsumer>,
    pending: Vec<async_nats::jetstream::Message>,
    spool: std::sync::Arc<std::sync::Mutex<crate::spool::Spool>>,
}

impl Client {
    pub(crate) fn producer(&self) -> &str {
        &self.producer
    }

    pub(crate) async fn replay_scope(
        &self,
        query: &sinex_protocol::ReplayScopeQuery,
    ) -> Result<Vec<sinex_protocol::ReplaySelection>> {
        let reply = self
            .client
            .request(
                self.subjects.replay_scope(&self.producer),
                serde_json::to_vec(query)?.into(),
            )
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let value: serde_json::Value = serde_json::from_slice(&reply.payload)?;
        if let Some(reason) = value.get("refusal").and_then(serde_json::Value::as_str) {
            return Err(Error::Invalid(reason.into()));
        }
        serde_json::from_value(value).map_err(Error::Json)
    }

    pub(crate) async fn connect(
        config: &Config,
        spool: std::sync::Arc<std::sync::Mutex<crate::spool::Spool>>,
        consumer_name: String,
    ) -> Result<Self> {
        let options = match config.credentials()? {
            Some((user, password)) => {
                async_nats::ConnectOptions::with_user_and_password(user, password)
            }
            None => async_nats::ConnectOptions::new(),
        };
        let client = options
            .connect(config.nats.as_str())
            .await
            .map_err(|e| Error::Unreachable(format!("NATS at {}: {e}", config.nats)))?;
        let subjects = Subjects::new(config.subject_root.clone());
        let acknowledgments = client
            .subscribe(subjects.ack(&config.producer))
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        client
            .flush()
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let welcome = client
            .request(
                subjects.connect(&config.producer),
                serde_json::to_vec(&ProducerHello::default())?.into(),
            )
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let welcome: ProducerWelcome = serde_json::from_slice(&welcome.payload)?;
        if let Some(reason) = welcome.refusal {
            return Err(Error::Refused {
                name: "connect".into(),
                reason,
            });
        }
        let max_message = client.server_info().max_payload;
        let mut connected = Self {
            client,
            subjects,
            producer: config.producer.clone(),
            max_message,
            store_root: config.store.clone(),
            acknowledgments,
            consumer_name,
            consumer: None,
            pending: Vec::new(),
            spool,
        };
        let declarations = crate::kinds::declarations(&config.producer);
        let hashes: Vec<_> = declarations
            .iter()
            .map(|d| (d.kind.clone(), d.definition_hash()))
            .collect();
        let body = serde_json::to_vec(&hashes)?;
        let name = format!("kinds:{}", blake3::hash(&body).to_hex());
        let registration = Ingress::RegisterKinds {
            frame: name.clone(),
            declarations,
        };
        let bytes = serde_json::to_vec(&registration)?;
        let blob = format!("blake3:{}", blake3::hash(&bytes).to_hex());
        connected.write_bytes(&name, &blob, bytes.len() as u64, bytes.as_slice())?;
        let ack = connected.send(registration).await?;
        let registered: Vec<(sinex_model::KindRef, String)> =
            serde_json::from_value(ack.outcome["registered_kinds"].clone())?;
        if registered != hashes {
            return Err(Error::Invalid(
                "Sinex acknowledged different kind definitions".into(),
            ));
        }
        Ok(connected)
    }

    /// What Sinex recorded under these frame names.
    pub(crate) async fn recorded(
        &self,
        names: &[String],
    ) -> Result<HashMap<String, Acknowledgment>> {
        let mut out = HashMap::new();
        for chunk in names.chunks(500) {
            let request = RecordedQuery {
                frames: chunk.to_vec(),
            };
            let reply = self
                .client
                .request(
                    self.subjects.recorded(&self.producer),
                    serde_json::to_vec(&request)?.into(),
                )
                .await
                .map_err(|e| Error::Unreachable(e.to_string()))?;
            let acknowledgments: Vec<Acknowledgment> = serde_json::from_slice(&reply.payload)?;
            out.extend(acknowledgments.into_iter().map(|a| (a.frame.clone(), a)));
        }
        Ok(out)
    }

    /// Writes `bytes` (`length` long, named `blob`) to the directory of
    /// the frame `name`.
    pub(crate) fn write_bytes(
        &self,
        name: &str,
        blob: &str,
        length: u64,
        bytes: impl Read,
    ) -> Result<()> {
        frame_bytes::write(&self.store_root, &self.producer, name, blob, length, bytes)
            .map_err(|error| Error::Invalid(error.to_string()))
    }

    /// Sends `frame` and waits for its acknowledgment. A frame too large for
    /// one message travels indirectly. A frame Sinex recorded before is
    /// answered from its record.
    pub(crate) async fn send(&mut self, frame: Ingress) -> Result<Acknowledgment> {
        let prepared = self
            .spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?
            .prepare(frame)?;
        if let Some(acknowledgment) = prepared.acknowledgment {
            return checked(acknowledgment);
        }
        let acknowledgment = self.send_prepared(prepared.frame).await?;
        self.spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?
            .acknowledge(acknowledgment.clone())?;
        checked(acknowledgment)
    }

    /// Resume every original frame after reconnect, in preparation order.
    pub(crate) async fn retry_prepared(&mut self) -> Result<()> {
        let prepared = self
            .spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?
            .prepared()?;
        for prepared in prepared {
            if prepared.acknowledgment.is_some() {
                continue;
            }
            let acknowledgment = self.send_prepared(prepared.frame).await?;
            let refused = acknowledgment.refusal.clone();
            let frame = acknowledgment.frame.clone();
            self.spool
                .lock()
                .map_err(|_| Error::Invalid("producer spool panicked".into()))?
                .acknowledge(acknowledgment)?;
            if let Some(reason) = refused {
                return Err(Error::Refused {
                    name: frame,
                    reason,
                });
            }
        }
        Ok(())
    }

    async fn send_prepared(&mut self, frame: Ingress) -> Result<Acknowledgment> {
        if let Ingress::Material { material, .. } = &frame {
            let bytes = {
                let spool = self
                    .spool
                    .lock()
                    .map_err(|_| Error::Invalid("producer spool panicked".into()))?;
                if spool
                    .get(frame.frame())?
                    .is_some_and(|prepared| prepared.body)
                {
                    Some(spool.body(frame.frame())?)
                } else {
                    None
                }
            };
            if let Some(bytes) = bytes {
                self.write_bytes(frame.frame(), &material.blob, material.length, bytes)?;
            }
        }
        let body = serde_json::to_vec(&frame)?;
        if body.len().saturating_add(MESSAGE_OVERHEAD) <= self.max_message {
            return self.transmit(frame.frame(), body).await;
        }
        let name = frame.frame().to_owned();
        let blob = format!("blake3:{}", blake3::hash(&body).to_hex());
        let length = body.len() as u64;
        self.write_bytes(&name, &blob, length, body.as_slice())?;
        let indirect = serde_json::to_vec(&Ingress::LargeFrame {
            frame: name.clone(),
            blob,
            length,
        })?;
        self.transmit(&name, indirect).await
    }

    /// Publishes one frame's JSON and waits for its acknowledgment. Accepted
    /// by NATS is not confirmed: only the acknowledgment, which Sinex commits
    /// with the frame's contents, is. A lost acknowledgment is an error here;
    /// sending the frame again is answered from Sinex's record of it.
    async fn transmit(&mut self, name: &str, body: Vec<u8>) -> Result<Acknowledgment> {
        let js = jetstream::new(self.client.clone());
        js.publish(self.subjects.ingress(&self.producer), body.into())
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        loop {
            let message = tokio::time::timeout(ACKNOWLEDGMENT_WAIT, self.acknowledgments.next())
                .await
                .map_err(|_| Error::Unreachable(format!("no acknowledgment of {name}")))?
                .ok_or_else(|| Error::Unreachable("acknowledgments closed".into()))?;
            let acknowledgment: Acknowledgment = serde_json::from_slice(&message.payload)?;
            if acknowledgment.frame == name {
                return Ok(acknowledgment);
            }
            // A body Sinex could not read as a frame is refused under no
            // name; one frame is in flight at a time, so it is this one.
            if acknowledgment.frame.is_empty()
                && let Some(reason) = acknowledgment.refusal
            {
                return Err(Error::Refused {
                    name: name.to_owned(),
                    reason,
                });
            }
        }
    }
}

pub(crate) fn checked(acknowledgment: Acknowledgment) -> Result<Acknowledgment> {
    match &acknowledgment.refusal {
        Some(reason) => Err(Error::Refused {
            name: acknowledgment.frame.clone(),
            reason: reason.clone(),
        }),
        None => Ok(acknowledgment),
    }
}

#[cfg(test)]
mod tests;

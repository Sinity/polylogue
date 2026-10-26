//! The producer protocol over NATS: frames to the ingress, their
//! acknowledgments, which frames Sinex recorded, and the materials Sinex holds from this producer.
//! A material's bytes, and a frame too large for one message, are written to
//! the frame's directory under Sinex's store root on this host; Sinex
//! stores them before it records the frame.
use crate::{Config, Error, Result};
use async_nats::jetstream;
use futures_util::StreamExt;
use sinex_protocol::{
    Acknowledgment, Excise, ExciseReply, Ingress, Materials, ProducerMaterial, RecordedQuery,
    Subjects,
};
use sinex_transport::frame_bytes;
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

/// How long a frame may wait for its acknowledgment.
const ACKNOWLEDGMENT_WAIT: Duration = Duration::from_secs(120);

/// How long an excision may wait for Sinex's answer. Sinex plans it over
/// every event it holds, which takes far longer than NATS's default request
/// timeout on a large store; giving up early would leave Sinex excising
/// what Polylogue then keeps.
const EXCISION_WAIT: Duration = Duration::from_secs(600);

/// Room left in a NATS message for its subject and headers.
const MESSAGE_OVERHEAD: usize = 4096;

/// How many held materials one page asks for.
const PAGE: u32 = 1000;

pub(crate) struct Client {
    client: async_nats::Client,
    subjects: Subjects,
    producer: String,
    max_message: usize,
    pub(crate) store_root: PathBuf,
    acknowledgments: async_nats::Subscriber,
}

impl Client {
    pub(crate) async fn connect(config: &Config) -> Result<Self> {
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
        let max_message = client.server_info().max_payload;
        Ok(Self {
            client,
            subjects,
            producer: config.producer.clone(),
            max_message,
            store_root: config.store.clone(),
            acknowledgments,
        })
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

    /// One page of the materials Sinex holds from this producer, after the
    /// frame numbered `after`.
    pub(crate) async fn materials(&self, after: i64) -> Result<Vec<ProducerMaterial>> {
        let reply = self
            .client
            .request(
                self.subjects.materials(&self.producer),
                serde_json::to_vec(&Materials { after, limit: PAGE })?.into(),
            )
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let value: serde_json::Value = serde_json::from_slice(&reply.payload)?;
        if let Some(refusal) = value.get("refusal") {
            return Err(Error::Refused {
                name: "materials".into(),
                reason: refusal.to_string(),
            });
        }
        Ok(serde_json::from_value(value)?)
    }

    /// Asks Sinex to excise these materials this producer registered.
    pub(crate) async fn excise(&self, materials: &[String]) -> Result<()> {
        let materials = materials
            .iter()
            .map(|id| {
                id.parse()
                    .map(sinex_model::MaterialId)
                    .map_err(|e| Error::Invalid(format!("material {id}: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let request = async_nats::Request::new()
            .payload(serde_json::to_vec(&Excise { materials })?.into())
            .timeout(Some(EXCISION_WAIT));
        let reply = self
            .client
            .send_request(self.subjects.excise(&self.producer), request)
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let reply: ExciseReply = serde_json::from_slice(&reply.payload)?;
        match reply.refusal {
            Some(reason) => Err(Error::Refused {
                name: "excision".into(),
                reason,
            }),
            None => Ok(()),
        }
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
            .map_err(Error::Invalid)
    }

    /// Sends `frame` and waits for its acknowledgment. A frame too large for
    /// one message travels indirectly. A frame Sinex recorded before is
    /// answered from its record.
    pub(crate) async fn send(&mut self, frame: Ingress) -> Result<Acknowledgment> {
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
                return checked(acknowledgment);
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

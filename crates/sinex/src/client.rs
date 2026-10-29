//! The producer protocol over NATS: frames to the ingress, their
//! acknowledgments, which frames Sinex recorded, and the materials Sinex holds from this producer.
//! A material's bytes, and a frame too large for one message, are written to
//! the frame's directory under Sinex's store root on this host; Sinex
//! stores them before it records the frame.
use crate::{Authentication, Config, Error, Result};
mod confirmed;
use sinex_protocol::{Acknowledgment, Ingress, ProducerHello, ProducerWelcome, Subjects};
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
    family: String,
    max_message: usize,
    pub(crate) store_root: PathBuf,
    acknowledgments: async_nats::Subscriber,
    consumer_cache: Option<String>,
    consumer_scope: Option<String>,
    publication_scope: Option<String>,
    health: std::sync::Arc<std::sync::Mutex<crate::status::Health>>,
    consumer: Option<async_nats::jetstream::consumer::PullConsumer>,
    pending: Vec<async_nats::jetstream::Message>,
    spool: std::sync::Arc<std::sync::Mutex<crate::spool::Spool>>,
}

impl Client {
    pub(crate) fn family(&self) -> &str {
        &self.family
    }

    pub(crate) async fn replay_scope(
        &self,
        query: &sinex_protocol::ReplayScopeQuery,
    ) -> Result<Vec<sinex_protocol::ReplaySelection>> {
        let mut query = query.clone();
        let mut sequences = Vec::new();
        loop {
            let page: sinex_protocol::ReplayScopePage = self
                .request(self.subjects.replay_scope(&self.producer), &query)
                .await?;
            sequences.extend(page.sequences);
            match page.next {
                Some(next) if query.after.is_none_or(|after| next > after) => {
                    query.after = Some(next)
                }
                Some(_) => return Err(Error::Invalid("replay scope page did not advance".into())),
                None => return Ok(sequences),
            }
        }
    }

    pub(crate) async fn connect(
        config: &Config,
        authentication: &Authentication,
        spool: std::sync::Arc<std::sync::Mutex<crate::spool::Spool>>,
        health: std::sync::Arc<std::sync::Mutex<crate::status::Health>>,
    ) -> Result<Self> {
        let subjects = Subjects::new(config.subject_root.clone());
        let options = match authentication {
            Authentication::Password(password) => {
                async_nats::ConnectOptions::with_user_and_password(
                    config.producer.clone(),
                    password.clone(),
                )
            }
            Authentication::Anonymous => async_nats::ConnectOptions::new(),
            Authentication::Unavailable { reason } => {
                return Err(Error::Invalid(format!(
                    "producer authentication is unavailable: {reason}"
                )));
            }
        };
        let client = options
            .custom_inbox_prefix(subjects.inbox_prefix(&config.producer))
            .connect(config.nats.as_str())
            .await
            .map_err(|e| Error::Unreachable(format!("NATS at {}: {e}", config.nats)))?;
        let acknowledgments = client
            .subscribe(subjects.ack(&config.producer))
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        client
            .flush()
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let welcome: ProducerWelcome = sinex_transport::replies::request_json(
            &client,
            subjects.connect(&config.producer),
            &config.producer,
            &ProducerHello::default(),
            ACKNOWLEDGMENT_WAIT,
        )
        .await
        .map_err(|error| Error::Unreachable(error.to_string()))?;
        if let Some(reason) = welcome.refusal {
            return Err(Error::Refused {
                name: "connect".into(),
                reason,
            });
        }
        let family = welcome
            .family
            .filter(|family| family == "polylogue")
            .ok_or_else(|| Error::Refused {
                name: "connect".into(),
                reason: "this identity has no configured Polylogue write family".into(),
            })?;
        if let Ok(mut health) = health.lock() {
            health.family = Some(family.clone());
            health.canonical_publisher = welcome.canonical_publisher;
        }
        let max_message = client.server_info().max_payload;
        let mut connected = Self {
            client,
            subjects,
            producer: config.producer.clone(),
            family: family.clone(),
            max_message,
            store_root: config.store.clone(),
            acknowledgments,
            consumer_cache: None,
            consumer_scope: None,
            publication_scope: None,
            health,
            consumer: None,
            pending: Vec::new(),
            spool,
        };
        let declarations = crate::kinds::declarations(&family);
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
        Ok(sinex_transport::producer::recorded(
            &self.client,
            &self.subjects,
            &self.producer,
            names,
            &[],
            ACKNOWLEDGMENT_WAIT,
        )
        .await
        .map_err(|error| Error::Unreachable(error.to_string()))?
        .into_iter()
        .map(|ack| (ack.frame.clone(), ack))
        .collect())
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
        self.publish_prepared(prepared).await
    }

    /// Resume original immutable operations in bounded preparation-order pages.
    pub(crate) async fn retry_prepared(&mut self) -> Result<()> {
        let mut after = None;
        loop {
            let page = self
                .spool
                .lock()
                .map_err(|_| Error::Invalid("producer spool panicked".into()))?
                .pending_after(after, 64)?;
            if page.is_empty() {
                return Ok(());
            }
            for prepared in page {
                after = Some(prepared.order);
                if prepared.acknowledged {
                    continue;
                }
                if !self.publication_allowed(&prepared.identity) {
                    continue;
                }
                self.publish_prepared(prepared).await?;
            }
        }
    }

    async fn publish_prepared(
        &mut self,
        prepared: crate::spool::Prepared,
    ) -> Result<Acknowledgment> {
        let (receipt, frame, body) = {
            let spool = self
                .spool
                .lock()
                .map_err(|_| Error::Invalid("producer spool panicked".into()))?;
            (
                spool.acknowledgment(&prepared.identity)?,
                spool.frame(&prepared.identity)?,
                if prepared.body.is_some() {
                    Some(spool.body(&prepared.identity)?)
                } else {
                    None
                },
            )
        };
        if let Some(receipt) = receipt {
            return checked(receipt);
        }
        if !self.publication_allowed(&prepared.identity) {
            return Err(Error::Refused {
                name: prepared.identity,
                reason: "canonical output belongs to another or unknown confirmed read scope"
                    .into(),
            });
        }
        if let (Some(material), Some(body)) = (&prepared.material, body) {
            self.write_bytes(&prepared.identity, &material.blob, material.length, body)?;
        }
        let body = if prepared
            .frame_length
            .saturating_add(MESSAGE_OVERHEAD as u64)
            <= self.max_message as u64
        {
            let mut bytes = Vec::new();
            std::io::BufReader::new(frame).read_to_end(&mut bytes)?;
            bytes
        } else {
            self.write_bytes(
                &prepared.identity,
                &prepared.frame_hash,
                prepared.frame_length,
                frame,
            )?;
            serde_json::to_vec(&Ingress::LargeFrame {
                frame: prepared.identity.clone(),
                blob: prepared.frame_hash,
                length: prepared.frame_length,
            })?
        };
        let acknowledgment = self.transmit(&prepared.identity, body).await?;
        self.spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?
            .acknowledge(acknowledgment.clone())?;
        if let Ok(mut health) = self.health.lock() {
            health.last_ack = Some(std::time::Instant::now());
        }
        checked(acknowledgment)
    }

    fn publication_allowed(&self, identity: &str) -> bool {
        !identity.starts_with("work-snapshot:")
            || self.publication_scope.as_ref().is_some_and(|scope| {
                identity.starts_with(&format!(
                    "work-snapshot:{}:",
                    blake3::hash(scope.as_bytes()).to_hex()
                ))
            })
    }

    /// Publishes one frame's JSON and waits for its acknowledgment. Accepted
    /// by NATS is not confirmed: only the acknowledgment, which Sinex commits
    /// with the frame's contents, is. A lost acknowledgment is an error here;
    /// sending the frame again is answered from Sinex's record of it.
    async fn transmit(&mut self, name: &str, body: Vec<u8>) -> Result<Acknowledgment> {
        let frame: Ingress = serde_json::from_slice(&body)?;
        if frame.frame() != name {
            return Err(Error::Invalid(
                "prepared transport changed frame identity".into(),
            ));
        }
        sinex_transport::producer::submit_frame(
            &self.client,
            &self.subjects,
            &self.producer,
            &mut self.acknowledgments,
            &frame,
            ACKNOWLEDGMENT_WAIT,
        )
        .await
        .map_err(|error| Error::Unreachable(error.to_string()))
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

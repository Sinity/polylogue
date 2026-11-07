//! The producer protocol over NATS: frames to the ingress, their
//! acknowledgments, which frames Sinex recorded, and the materials Sinex holds from this producer.
//! A material's bytes, and a frame too large for one message, are written to
//! the frame's directory under Sinex's store root on this host; Sinex
//! stores them before it records the frame.
use crate::{Authentication, Config, Error, Result};
pub(crate) mod confirmed;
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

/// Acquire the producer-owned artifact. Pinning holds the queue mutex only
/// while retaining immutable files; byte copying and hashing happen after it.
fn capture_backup(
    spool: &std::sync::Mutex<crate::spool::Spool>,
    cancelled: &std::sync::atomic::AtomicBool,
) -> std::result::Result<sinex_transport::backup::Captured, String> {
    let pinned = spool
        .lock()
        .map_err(|_| "producer spool panicked".to_string())?
        .pin_backup(cancelled)
        .map_err(|error| error.to_string())?;
    let directory = tempfile::Builder::new()
        .prefix(".backup-acquired-")
        .tempdir_in(pinned.path().parent().expect("snapshot parent"))
        .map_err(|error| error.to_string())?;
    let files = sinex_transport::backup::capture_tree(pinned.path(), directory.path(), cancelled)?;
    Ok(sinex_transport::backup::Captured { directory, files })
}

pub(crate) struct Client {
    client: async_nats::Client,
    subjects: Subjects,
    producer: String,
    family: String,
    max_message: usize,
    pub(crate) store_root: PathBuf,
    acknowledgments: async_nats::Subscriber,
    consumer_cache: Option<String>,
    consumer_instance: Option<String>,
    health: std::sync::Arc<std::sync::Mutex<crate::status::Health>>,
    consumer: Option<async_nats::jetstream::consumer::PullConsumer>,
    pending: Vec<async_nats::jetstream::Message>,
    spool: std::sync::Arc<std::sync::Mutex<crate::spool::Spool>>,
    publication: sinex_transport::backup::Publication,
    _backup: sinex_transport::backup::Participant,
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
            let reply: sinex_protocol::ReplayReply = self
                .request(
                    self.subjects.replay_scope(&self.producer),
                    sinex_protocol::ReplayQuery::Materials {
                        query: query.clone(),
                    },
                )
                .await?;
            let sinex_protocol::ReplayReply::Materials { page } = reply else {
                return Err(Error::Invalid(
                    "material replay returned another mode".into(),
                ));
            };
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
        let backup_spool = std::sync::Arc::clone(&spool);
        Self::connect_with_backup_capture(config, authentication, spool, health, move |cancelled| {
            capture_backup(&backup_spool, &cancelled)
        })
        .await
    }

    async fn connect_with_backup_capture(
        config: &Config,
        authentication: &Authentication,
        spool: std::sync::Arc<std::sync::Mutex<crate::spool::Spool>>,
        health: std::sync::Arc<std::sync::Mutex<crate::status::Health>>,
        capture: impl Fn(
            std::sync::Arc<std::sync::atomic::AtomicBool>,
        ) -> std::result::Result<sinex_transport::backup::Captured, String>
        + Send
        + Sync
        + 'static,
    ) -> Result<Self> {
        let (client, subjects, welcome) = transport(config, authentication).await?;
        let acknowledgments = client
            .subscribe(subjects.ack(&config.producer))
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        client
            .flush()
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let family = welcome
            .family
            .filter(|family| family == "polylogue")
            .ok_or_else(|| Error::Refused {
                name: "connect".into(),
                reason: "this identity has no configured Polylogue write family".into(),
            })?;
        if let Ok(mut health) = health.lock() {
            health.family = Some(family.clone());
        }
        let max_message = client.server_info().max_payload;
        let publication = sinex_transport::backup::Publication::default();
        let backup = sinex_transport::backup::Participant::start(
            client.clone(),
            subjects.clone(),
            config.producer.clone(),
            publication.clone(),
            std::sync::Arc::new(capture),
        )
        .await
        .map_err(Error::Invalid)?;
        let mut connected = Self {
            client,
            subjects,
            producer: config.producer.clone(),
            family: family.clone(),
            max_message,
            store_root: config.store.clone(),
            acknowledgments,
            consumer_cache: None,
            consumer_instance: None,
            health,
            consumer: None,
            pending: Vec::new(),
            spool,
            publication,
            _backup: backup,
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
        connected.refresh_context().await;
        Ok(connected)
    }

    async fn refresh_context(&self) {
        let due = self
            .health
            .lock()
            .map(|health| health.context.refresh_due())
            .unwrap_or(false);
        if !due {
            return;
        }
        let result = sinex_transport::replies::request_json::<sinex_protocol::PublicContext>(
            &self.client,
            self.subjects.public_context(&self.producer),
            &self.producer,
            &sinex_protocol::PublicContextRequest::default(),
            Duration::from_secs(2),
        )
        .await
        .map_err(|error| error.to_string());
        if let Ok(mut health) = self.health.lock() {
            health.context.record(result);
        }
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
                if !prepared.phase.is_pending() || prepared.unresolved {
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
        // Freeze waits for transmission and durable receipt publication as one
        // unit. Capture may append new spool operations while this waits.
        let _publication = self.publication.enter().await;
        let (frame, body) = {
            let spool = self
                .spool
                .lock()
                .map_err(|_| Error::Invalid("producer spool panicked".into()))?;
            let current = spool
                .get(&prepared.identity)?
                .ok_or_else(|| Error::Invalid("prepared operation disappeared".into()))?;
            if let Some(receipt) = spool.acknowledgment(&prepared.identity)? {
                return checked(verified_receipt(&prepared, receipt)?);
            }
            if current.unresolved || current.phase.is_refused() {
                return Err(Error::Refused {
                    name: prepared.identity.clone(),
                    reason: "the original operation is not publishable".into(),
                });
            }
            (
                spool.prepared_frame(&prepared.identity)?,
                if prepared.body().is_some() {
                    Some(spool.prepared_body(&prepared.identity)?)
                } else {
                    None
                },
            )
        };
        if let Some(body) = body {
            frame_bytes::write_prepared(&self.store_root, &self.producer, &prepared.identity, body)
                .map_err(|error| Error::Invalid(error.to_string()))?;
        }
        let body = if prepared
            .frame
            .length
            .saturating_add(MESSAGE_OVERHEAD as u64)
            <= self.max_message as u64
        {
            let mut bytes = Vec::new();
            std::io::BufReader::new(frame.reader(None, hashed_blobs::ReadLimits::default())?)
                .read_to_end(&mut bytes)?;
            bytes
        } else {
            frame_bytes::write_prepared(
                &self.store_root,
                &self.producer,
                &prepared.identity,
                frame,
            )
            .map_err(|error| Error::Invalid(error.to_string()))?;
            serde_json::to_vec(&Ingress::LargeFrame {
                frame: prepared.identity.clone(),
                blob: prepared.frame.blob.clone(),
                length: prepared.frame.length,
            })?
        };
        let acknowledgment = match self.transmit(&prepared.identity, body).await {
            Ok(receipt) => receipt,
            Err(Error::ReceiptConflict {
                frame,
                offered_hash,
                receipt,
            }) => {
                self.spool
                    .lock()
                    .map_err(|_| Error::Invalid("producer spool panicked".into()))?
                    .record_conflict(&frame, &offered_hash, &receipt)?;
                return Err(Error::Refused {
                    name: frame,
                    reason: "the owner recorded different immutable contents".into(),
                });
            }
            Err(error) => return Err(error),
        };
        self.spool
            .lock()
            .map_err(|_| Error::Invalid("producer spool panicked".into()))?
            .acknowledge(acknowledgment.clone())?;
        if let Ok(mut health) = self.health.lock() {
            health.last_ack = Some(std::time::Instant::now());
        }
        checked(acknowledgment)
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
        sinex_transport::producer::transfer_frame(
            &self.client,
            &self.subjects,
            &self.producer,
            &mut self.acknowledgments,
            &self.store_root,
            &frame,
            ACKNOWLEDGMENT_WAIT,
        )
        .await
        .map_err(|error| Error::Unreachable(error.to_string()))?;
        sinex_transport::producer::submit_frame(
            &self.client,
            &self.subjects,
            &self.producer,
            &mut self.acknowledgments,
            &frame,
            ACKNOWLEDGMENT_WAIT,
        )
        .await
        .map_err(submit_error)
    }
}

/// Connects, authenticates and checks the producer welcome.
async fn transport(
    config: &Config,
    authentication: &Authentication,
) -> Result<(async_nats::Client, Subjects, ProducerWelcome)> {
    let subjects = Subjects::new(config.subject_root.clone());
    let options = match authentication {
        Authentication::Password(password) => async_nats::ConnectOptions::with_user_and_password(
            config.producer.clone(),
            password.clone(),
        ),
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
    if let Some(reason) = &welcome.refusal {
        return Err(Error::Refused {
            name: "connect".into(),
            reason: reason.clone(),
        });
    }
    Ok((client, subjects, welcome))
}

pub(crate) fn submit_error(error: sinex_transport::producer::SubmitError) -> Error {
    match error {
        sinex_transport::producer::SubmitError::ReceiptConflict {
            frame,
            offered_hash,
            receipt,
        } => Error::ReceiptConflict {
            frame,
            offered_hash,
            receipt,
        },
        other => Error::Unreachable(other.to_string()),
    }
}

/// A LargeFrame envelope names the original logical frame hash. Receipt
/// adoption needs that identity, independently of the encoded custody layout.
pub(crate) fn verified_receipt(
    prepared: &crate::spool::Prepared,
    receipt: Acknowledgment,
) -> Result<Acknowledgment> {
    let frame = Ingress::LargeFrame {
        frame: prepared.identity.clone(),
        blob: prepared.frame.blob.clone(),
        length: prepared.frame.length,
    };
    if receipt.frame != prepared.identity {
        return Err(Error::Invalid(
            "recorded receipt changed its operation identity".into(),
        ));
    }
    sinex_transport::producer::check_receipt(&frame, receipt).map_err(submit_error)
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

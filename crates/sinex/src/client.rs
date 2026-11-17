//! The producer protocol over NATS: frames to the ingress, their
//! acknowledgments, which frames Sinex recorded, and the materials Sinex holds from this producer.
//! Immutable spool encodings pass through local frame directories. Shared-host
//! peers use hardlinks; remote peers receive encoded chunks before the logical
//! frame. Sinex stores the bytes before it records that frame.
use crate::{Authentication, Config, Error, Result};
use polylogue_model::LockExt;
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
        .locked()
        .pin_backup(cancelled)
        .map_err(|error| error.to_string())?;
    let pinned = pinned
        .finish(cancelled)
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
    pub(crate) frame_root: PathBuf,
    remote_upload: bool,
    consumer_cache: Option<String>,
    consumer_instance: Option<String>,
    health: std::sync::Arc<std::sync::Mutex<crate::status::Health>>,
    consumer: Option<async_nats::jetstream::consumer::PullConsumer>,
    pending: Vec<async_nats::jetstream::Message>,
    spool: std::sync::Arc<std::sync::Mutex<crate::spool::Spool>>,
    publication: sinex_transport::backup::Publication,
    _backup: Option<sinex_transport::backup::BackupResponder>,
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
                return Err(Error::invalid("material replay returned another mode"));
            };
            sequences.extend(page.sequences);
            match page.next {
                Some(next) if query.after.is_none_or(|after| next > after) => {
                    query.after = Some(next)
                }
                Some(_) => return Err(Error::invalid("replay scope page did not advance")),
                None => return Ok(sequences),
            }
        }
    }

    pub(crate) async fn connect_shared(
        config: &Config,
        authentication: &Authentication,
        spool: std::sync::Arc<std::sync::Mutex<crate::spool::Spool>>,
        health: std::sync::Arc<std::sync::Mutex<crate::status::Health>>,
        publication: sinex_transport::backup::Publication,
        start_backup: bool,
    ) -> Result<Self> {
        let backup_spool = spool.clone();
        Self::connect_with_context(
            config,
            authentication,
            spool,
            health,
            move |cancelled| capture_backup(&backup_spool, &cancelled),
            publication,
            start_backup,
        )
        .await
    }

    async fn connect_with_context(
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
        publication: sinex_transport::backup::Publication,
        start_backup: bool,
    ) -> Result<Self> {
        let (client, subjects, welcome) = transport(config, authentication).await?;
        client
            .flush()
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let family = welcome
            .filter(|family| family == "polylogue")
            .ok_or_else(|| Error::Refused {
                name: "connect".into(),
                reason: "this identity has no configured Polylogue write family".into(),
            })?;
        {
            let mut health = health.locked();
            health.family = Some(family.clone());
        }
        let max_message = client.server_info().max_payload;
        let backup = if start_backup {
            Some(
                sinex_transport::backup::BackupResponder::start(
                    client.clone(),
                    subjects.clone(),
                    config.producer.clone(),
                    publication.clone(),
                    std::sync::Arc::new(capture),
                )
                .await
                .map_err(Error::Invalid)?,
            )
        } else {
            None
        };
        let mut connected = Self {
            client,
            subjects,
            producer: config.producer.clone(),
            family: family.clone(),
            max_message,
            frame_root: config.upload_root.as_ref().unwrap_or(&config.store).clone(),
            remote_upload: config.upload_root.is_some(),
            consumer_cache: None,
            consumer_instance: None,
            health,
            consumer: None,
            pending: Vec::new(),
            spool,
            publication,
            _backup: backup,
        };
        // The sending connection handles registration and the backup callback.
        // A confirmed reader never sends while it connects.
        if start_backup {
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
            let sinex_protocol::FrameResult::KindsRegistered { kinds: registered } = ack.result
            else {
                return Err(Error::invalid(
                    "Sinex answered the kind registration with another result",
                ));
            };
            if registered != hashes {
                return Err(Error::invalid(
                    "Sinex acknowledged different kind definitions",
                ));
            }
        }
        connected.refresh_context().await;
        Ok(connected)
    }

    async fn refresh_context(&self) {
        let due = self.health.locked().context.refresh_due();
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
        {
            let mut health = self.health.locked();
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
        let prepared = self.spool.locked().prepare(frame)?;
        self.send_prepared(prepared).await
    }

    /// Resume a bounded turn of immutable operations, leaving completion
    /// preparation a turn even while capture continually adds new material.
    pub(crate) async fn retry_prepared(&mut self) -> Result<()> {
        use futures_util::{StreamExt, stream::FuturesUnordered};
        const IN_FLIGHT: usize = 4;
        const PER_TURN: usize = 64;
        let mut selected = 0;
        let mut sending = FuturesUnordered::new();
        let mut identities = Vec::new();
        let mut refused = None;
        loop {
            let ready = self.spool.locked().ready(
                &identities,
                (IN_FLIGHT - identities.len()).min(PER_TURN - selected),
            )?;
            selected += ready.len();
            for prepared in ready {
                let sender = self.frame_sender();
                identities.push(prepared.identity.clone());
                sending.push(async move {
                    let identity = prepared.identity.clone();
                    (identity, sender.send_prepared(prepared).await)
                });
            }
            let Some((identity, result)) = sending.next().await else {
                return refused.map_or(Ok(()), Err);
            };
            identities.retain(|active| active != &identity);
            match result {
                Ok(_) => {}
                Err(error @ Error::Refused { .. }) => {
                    refused = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
    }

    fn frame_sender(&self) -> FrameSender {
        FrameSender {
            client: self.client.clone(),
            subjects: self.subjects.clone(),
            producer: self.producer.clone(),
            frame_root: self.frame_root.clone(),
            remote_upload: self.remote_upload,
            max_message: self.max_message,
            spool: self.spool.clone(),
            publication: self.publication.clone(),
            health: self.health.clone(),
        }
    }
    async fn send_prepared(&self, prepared: crate::spool::Prepared) -> Result<Acknowledgment> {
        self.frame_sender().send_prepared(prepared).await
    }
}

/// Connects, authenticates and checks the producer welcome, returning the
/// write family it configures.
async fn transport(
    config: &Config,
    authentication: &Authentication,
) -> Result<(async_nats::Client, Subjects, Option<String>)> {
    let subjects = Subjects::new(config.subject_root.clone());
    let mut options = match authentication {
        Authentication::Password(password) => async_nats::ConnectOptions::with_user_and_password(
            config.producer.clone(),
            password.clone(),
        ),
        Authentication::Anonymous => async_nats::ConnectOptions::new(),
        Authentication::Unavailable { reason } => {
            return Err(Error::invalid(format!(
                "producer authentication is unavailable: {reason}"
            )));
        }
    };
    let address: async_nats::ServerAddr = config
        .nats
        .parse()
        .map_err(|error| Error::invalid(format!("invalid NATS endpoint: {error}")))?;
    let loopback = address.host().eq_ignore_ascii_case("localhost")
        || address
            .host()
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    if loopback {
        // A local forwarder owns its upstream. Broker discovery must not move
        // the plaintext loopback exception onto another host.
        options = options.ignore_discovered_servers();
    } else {
        // The NATS client's built-in TLS handshake runs and verifies before AUTH is sent.
        options = options.require_tls(true);
    }
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
    match welcome {
        ProducerWelcome::Refused { reason } => Err(Error::Refused {
            name: "connect".into(),
            reason,
        }),
        ProducerWelcome::Welcome { family, .. } => Ok((client, subjects, family)),
    }
}

pub(crate) fn submit_error(error: sinex_transport::producer::SubmitError) -> Error {
    match error {
        sinex_transport::producer::SubmitError::AckConflict {
            frame,
            offered_hash,
            ack: acknowledgment,
        } => Error::AcknowledgmentConflict {
            frame,
            offered_hash,
            acknowledgment,
        },
        other => Error::Unreachable(other.to_string()),
    }
}

/// A LargeFrame names the original frame's hash. Accepting a recorded
/// acknowledgment needs that identity, whatever the encoded spool layout.
pub(crate) fn verified_acknowledgment(
    prepared: &crate::spool::Prepared,
    acknowledgment: Acknowledgment,
) -> Result<Acknowledgment> {
    let frame = Ingress::LargeFrame {
        frame: prepared.identity.clone(),
        blob: prepared.frame.blob.clone(),
        length: prepared.frame.length,
    };
    if acknowledgment.frame != prepared.identity {
        return Err(Error::invalid(
            "recorded acknowledgment changed its operation identity",
        ));
    }
    sinex_transport::producer::check_ack(&frame, acknowledgment).map_err(submit_error)
}

pub(crate) fn checked(acknowledgment: Acknowledgment) -> Result<Acknowledgment> {
    match acknowledgment.result {
        sinex_protocol::FrameResult::Refused { reason, .. } => Err(Error::Refused {
            name: acknowledgment.frame,
            reason,
        }),
        _ => Ok(acknowledgment),
    }
}

#[cfg(test)]
mod tests;

#[derive(Clone)]
struct FrameSender {
    client: async_nats::Client,
    subjects: Subjects,
    producer: String,
    frame_root: PathBuf,
    remote_upload: bool,
    max_message: usize,
    spool: std::sync::Arc<std::sync::Mutex<crate::spool::Spool>>,
    publication: sinex_transport::backup::Publication,
    health: std::sync::Arc<std::sync::Mutex<crate::status::Health>>,
}
fn frame_error(error: frame_bytes::Error) -> Error {
    match error {
        frame_bytes::Error::Storage(error) => Error::Io(error),
        frame_bytes::Error::Invalid(reason)
        | frame_bytes::Error::Mismatch(reason)
        | frame_bytes::Error::Permissions(reason) => Error::invalid(reason),
    }
}

impl FrameSender {
    async fn send_prepared(&self, prepared: crate::spool::Prepared) -> Result<Acknowledgment> {
        // A backup freeze waits for transmission and the saved acknowledgment as one
        // unit. Capture may append new spool operations while this waits.
        let _publication = self.publication.enter().await;
        let (frame, body) = {
            let spool = self.spool.locked();
            let current = spool
                .get(&prepared.identity)?
                .ok_or_else(|| Error::invalid("prepared operation disappeared"))?;
            if let Some(acknowledgment) = spool.acknowledgment(&prepared.identity)? {
                spool.acknowledge(acknowledgment.clone())?;
                return checked(verified_acknowledgment(&prepared, acknowledgment)?);
            }
            if current.unresolved || current.phase.is_refused() {
                return Err(Error::Refused {
                    name: prepared.identity.clone(),
                    reason: "the original operation is not sendable".into(),
                });
            }
            (
                spool.pin_frame(&prepared.identity)?,
                if prepared.body().is_some() {
                    Some(spool.pin_body(&prepared.identity)?)
                } else {
                    None
                },
            )
        };
        // Spool paths stay local; the wire names the frame's bytes.
        for part in body.into_iter().flatten() {
            frame_bytes::link_file(
                &self.frame_root,
                &self.producer,
                &prepared.identity,
                &part.descriptor,
                &part.path,
                part.file,
            )
            .map_err(frame_error)?;
        }
        let body = if prepared
            .frame
            .length
            .saturating_add(MESSAGE_OVERHEAD as u64)
            <= self.max_message as u64
        {
            let mut bytes = Vec::new();
            let frame = hashed_blobs::Prepared::from_encoded(
                frame.file,
                frame.descriptor,
                None,
                hashed_blobs::ReadLimits::default(),
            )?;
            std::io::BufReader::new(frame.reader(None, hashed_blobs::ReadLimits::default())?)
                .read_to_end(&mut bytes)?;
            bytes
        } else {
            frame_bytes::link_file(
                &self.frame_root,
                &self.producer,
                &prepared.identity,
                &frame.descriptor,
                &frame.path,
                frame.file,
            )
            .map_err(frame_error)?;
            serde_json::to_vec(&Ingress::LargeFrame {
                frame: prepared.identity.clone(),
                blob: prepared.frame.blob.clone(),
                length: prepared.frame.length,
            })?
        };
        let acknowledgment = match self.transmit(&prepared.identity, body).await {
            Ok(acknowledgment) => acknowledgment,
            Err(Error::AcknowledgmentConflict {
                frame,
                offered_hash,
                acknowledgment,
            }) => {
                self.spool
                    .locked()
                    .record_conflict(&frame, &offered_hash, &acknowledgment)?;
                return Err(Error::Refused {
                    name: frame,
                    reason: "Sinex recorded different immutable contents".into(),
                });
            }
            Err(error) => return Err(error),
        };
        self.spool.locked().acknowledge(acknowledgment.clone())?;
        {
            let mut health = self.health.locked();
            health.last_ack = Some(std::time::Instant::now());
        }
        checked(acknowledgment)
    }

    /// Sends one frame's JSON and waits for its acknowledgment. Accepted
    /// by NATS is not confirmed: only the acknowledgment, which Sinex commits
    /// with the frame's contents, is. A lost acknowledgment is an error here;
    /// sending the frame again is answered from Sinex's record of it.
    async fn transmit(&self, name: &str, body: Vec<u8>) -> Result<Acknowledgment> {
        let mut acknowledgments = self
            .client
            .subscribe(self.subjects.ack(&self.producer))
            .await
            .map_err(|error| Error::Unreachable(error.to_string()))?;
        self.client
            .flush()
            .await
            .map_err(|error| Error::Unreachable(error.to_string()))?;
        let frame: Ingress = serde_json::from_slice(&body)?;
        if frame.frame() != name {
            return Err(Error::invalid("prepared transport changed frame identity"));
        }
        if self.remote_upload {
            sinex_transport::producer::transfer_frame(
                &self.client,
                &self.subjects,
                &self.producer,
                &mut acknowledgments,
                &self.frame_root,
                &frame,
                ACKNOWLEDGMENT_WAIT,
            )
            .await
            .map_err(|error| Error::Unreachable(error.to_string()))?;
        }
        sinex_transport::producer::submit_frame(
            &self.client,
            &self.subjects,
            &self.producer,
            &mut acknowledgments,
            &frame,
            ACKNOWLEDGMENT_WAIT,
        )
        .await
        .map_err(submit_error)
    }
}

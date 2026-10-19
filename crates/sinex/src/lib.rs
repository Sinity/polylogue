//! Polylogue on Sinex (architecture 7.1). Two modes: `standalone` (the
//! default; this crate is not used) and `primary`, where Sinex holds the
//! archive's evidence and Polylogue's databases are copies that can be
//! restored from it.
//!
//! What crosses: every input blob the archive stored, placed into Sinex's
//! content store on the same host and registered as a material; and every
//! unit of `archive.db` rows ([`rows`]) as a record-batch material with one
//! `polylogue.rows` event anchored at its start. Frames are named by what
//! they carry (`blob:<hex>`, `batch:<archive>:<position>`,
//! `rows:<archive>:<position>`, where a unit's position is its place in the
//! archive's write order), so a resend is answered from Sinex's record
//! and nothing local records what was sent: the publish step asks Sinex
//! (`have`) and sends the rest.
//!
//! Restoring is the reverse: read the archive's `polylogue.rows` events from
//! the confirmed stream in order, insert their rows into an empty
//! `archive.db`, and let the loop build `index.db` from the stored bytes,
//! which it reads from Sinex's store (an extra `hashed-blobs` root).
use async_nats::jetstream;
use futures_util::StreamExt;
use polylogue_store::rows::{self, Published as ArchiveRows, Restoring};
use serde::Deserialize;
use sinex_model::kinds::polylogue::PolylogueRows;
use sinex_model::{Class, InputRef, MaterialId, RecordLocation};
use sinex_protocol::{Acknowledgment, Have, Ingress, MaterialFrame, ProducerEvent, Subjects};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Sinex is unreachable: {0}")]
    Unreachable(String),
    #[error("Sinex refused {frame}: {reason}")]
    Refused { frame: String, reason: String },
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Store(#[from] polylogue_store::StoreError),
    #[error(transparent)]
    Blobs(#[from] hashed_blobs::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Standalone,
    Primary,
}

/// `$XDG_CONFIG_HOME/polylogue/sinex.toml`. It lives outside the archive,
/// so a restore can start from an empty one.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub mode: Mode,
    /// NATS server URL, with the producer's credentials when required.
    pub nats: String,
    /// Subject root (`sinex` when deployed).
    #[serde(default = "default_subjects")]
    pub subjects: String,
    /// This producer's name on the ingress.
    #[serde(default = "default_producer")]
    pub producer: String,
    /// Sinex's content store directory, written by placement and read
    /// directly (same host).
    pub store: PathBuf,
    /// The archive Sinex holds for this machine: restored from Sinex when
    /// the local archive is empty.
    #[serde(default)]
    pub archive: Option<String>,
}

fn default_subjects() -> String {
    "sinex".into()
}

fn default_producer() -> String {
    "polylogue".into()
}

impl Config {
    pub fn path() -> PathBuf {
        let config = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .unwrap_or_else(|| PathBuf::from("."));
        config.join("polylogue/sinex.toml")
    }

    /// The configuration, or `None` when there is no file (standalone). A
    /// file that does not parse is an error the caller shows; it never
    /// switches the mode silently.
    pub fn load() -> Result<Option<Self>> {
        let path = Self::path();
        match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str(&text)
                .map(Some)
                .map_err(|e| Error::Invalid(format!("{}: {e}", path.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// What one publish pass did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Published {
    /// Units Sinex already held before this pass.
    pub held: u64,
    pub units: u64,
    pub rows: u64,
    pub blobs: u64,
}

/// What a restore did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Restored {
    pub units: u64,
    pub rows: u64,
    /// The archive's write counter after the restore.
    pub local_seq: i64,
}

/// A connection to Sinex as the producer `config.producer`.
pub struct Sinex {
    client: async_nats::Client,
    subjects: Subjects,
    producer: String,
    store: hashed_blobs::Store,
    store_root: PathBuf,
    acks: async_nats::Subscriber,
}

const ACK_WAIT: Duration = Duration::from_secs(120);

impl Sinex {
    pub async fn connect(config: &Config) -> Result<Self> {
        let client = async_nats::connect(&config.nats)
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let subjects = Subjects::new(config.subjects.clone());
        let acks = client
            .subscribe(subjects.ack(&config.producer))
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        client
            .flush()
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        Ok(Self {
            client,
            subjects,
            producer: config.producer.clone(),
            store: hashed_blobs::Store::open(config.store.clone(), vec![], 0o640)?,
            store_root: config.store.clone(),
            acks,
        })
    }

    /// Sinex's content store, as an extra read root for Polylogue's blobs.
    pub fn store_root(&self) -> &Path {
        &self.store_root
    }

    /// What Sinex recorded for these frames.
    pub async fn have(&self, frames: &[String]) -> Result<HashMap<String, Acknowledgment>> {
        let mut out = HashMap::new();
        for chunk in frames.chunks(500) {
            let request = Have {
                frames: chunk.to_vec(),
            };
            let reply = self
                .client
                .request(
                    self.subjects.have(&self.producer),
                    serde_json::to_vec(&request)?.into(),
                )
                .await
                .map_err(|e| Error::Unreachable(e.to_string()))?;
            let acks: Vec<Acknowledgment> = serde_json::from_slice(&reply.payload)?;
            out.extend(acks.into_iter().map(|a| (a.frame.clone(), a)));
        }
        Ok(out)
    }

    /// Sends one frame and waits for its acknowledgment. Accepted by NATS is
    /// not confirmed: only the acknowledgment, which Sinex commits with the
    /// frame's events, is.
    async fn send(&mut self, frame: &Ingress) -> Result<Acknowledgment> {
        let js = jetstream::new(self.client.clone());
        js.publish(
            self.subjects.ingress(&self.producer),
            serde_json::to_vec(frame)?.into(),
        )
        .await
        .map_err(|e| Error::Unreachable(e.to_string()))?
        .await
        .map_err(|e| Error::Unreachable(e.to_string()))?;
        let name = frame.frame();
        loop {
            let message = tokio::time::timeout(ACK_WAIT, self.acks.next())
                .await
                .map_err(|_| Error::Unreachable(format!("no acknowledgment of {name}")))?
                .ok_or_else(|| Error::Unreachable("acknowledgments closed".into()))?;
            let ack: Acknowledgment = serde_json::from_slice(&message.payload)?;
            if ack.frame == name {
                return checked(ack);
            }
        }
    }

    /// Registers bytes already placed in the store; returns the material.
    async fn material(
        &mut self,
        frame: String,
        known: Option<&Acknowledgment>,
        blob: &hashed_blobs::Hash,
        length: u64,
        locator: String,
        manifest: serde_json::Value,
    ) -> Result<MaterialId> {
        let ack = match known {
            Some(a) => checked(a.clone())?,
            None => {
                let m = MaterialFrame {
                    id: MaterialId::new(),
                    input: InputRef {
                        kind: "polylogue".into(),
                        locator,
                        host: String::new(),
                    },
                    blob: format!("blake3:{}", blob.hex()),
                    length,
                    parser: "polylogue".into(),
                    parser_version: 1,
                    part_index: 0,
                    previous_part: None,
                    manifest,
                };
                self.send(&Ingress::Material {
                    frame,
                    material: Box::new(m),
                })
                .await?
            }
        };
        ack.material
            .ok_or_else(|| Error::Invalid(format!("{} registered no material", ack.frame)))
    }

    /// Publishes every unit of the archive at `root` that Sinex does not hold
    /// yet, in write order: its input blobs, its record batch, then its
    /// `polylogue.rows` event.
    pub async fn publish(&mut self, root: &Path) -> Result<Published> {
        let archive = ArchiveRows::open(root)?;
        let id = archive
            .archive_id()
            .map_err(|e| Error::Invalid(format!("the archive has no identity: {e}")))?;
        let units = archive.units(0)?;
        let rows_frame = |position: usize| format!("rows:{id}:{position}");
        // Units are sent in order and each waits for its acknowledgment, so
        // Sinex holds a prefix: find its end by bisection.
        let (mut lo, mut hi) = (0usize, units.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            let f = rows_frame(mid + 1);
            if self.have(std::slice::from_ref(&f)).await?.contains_key(&f) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut done = Published {
            held: lo as u64,
            ..Published::default()
        };
        for (index, unit) in units.iter().enumerate().skip(lo) {
            let position = index + 1;
            let batch = archive.read(unit)?;
            let blob_frames: Vec<String> = batch
                .blobs
                .iter()
                .map(|h| format!("blob:{}", rows::hex(h)))
                .collect();
            let known = self.have(&blob_frames).await?;
            for (h, frame) in batch.blobs.iter().zip(blob_frames) {
                if known.contains_key(&frame) {
                    continue;
                }
                let hash = hashed_blobs::Hash(*h);
                let bytes = archive.blob(h)?;
                if self.store.put(&bytes[..], 3)? != hash {
                    return Err(Error::Invalid(format!(
                        "blob {} changed while placed",
                        hash.hex()
                    )));
                }
                self.material(
                    frame,
                    None,
                    &hash,
                    bytes.len() as u64,
                    format!("blob:{}", hash.hex()),
                    serde_json::json!({ "archive": id }),
                )
                .await?;
                done.blobs += 1;
            }
            let batch_hash = self.store.put(&batch.lines[..], 3)?;
            let batch_frame = format!("batch:{id}:{position}");
            let known = self.have(std::slice::from_ref(&batch_frame)).await?;
            let material = self
                .material(
                    batch_frame.clone(),
                    known.get(&batch_frame),
                    &batch_hash,
                    batch.lines.len() as u64,
                    batch_frame.clone(),
                    serde_json::json!({ "archive": id, "media_type": "application/jsonl" }),
                )
                .await?;
            let event = PolylogueRows {
                archive: id.clone(),
                position: position as u64,
                local_seq: unit.local_seq,
                unit: unit.kind.into(),
                key: unit.key.clone(),
                batch: format!("blake3:{}", batch_hash.hex()),
                rows: batch.rows,
                blobs: batch
                    .blobs
                    .iter()
                    .map(|h| format!("blake3:{}", rows::hex(h)))
                    .collect(),
            };
            self.send(&Ingress::Events {
                frame: rows_frame(position),
                events: vec![ProducerEvent {
                    event_type: "polylogue.rows".into(),
                    payload: serde_json::to_value(&event)?,
                    location: Some(RecordLocation {
                        material,
                        anchor: 0,
                    }),
                    ordinal: 0,
                    parents: vec![],
                    reading_order: position as i64,
                    class: Class::Open,
                }],
            })
            .await?;
            done.units += 1;
            done.rows += batch.rows;
        }
        Ok(done)
    }

    /// Restores the archive `archive` into the empty, locked archive from
    /// Sinex: every `polylogue.rows` unit in write order, each batch read
    /// from Sinex's store and checked whole. Afterwards the loop builds
    /// `index.db` from the stored bytes, read from Sinex's store.
    pub async fn restore(
        &self,
        store_archive: &polylogue_store::Archive,
        archive: &str,
    ) -> Result<Restored> {
        let mut restoring = Restoring::open(store_archive)?;
        let js = jetstream::new(self.client.clone());
        let stream = js
            .get_stream(self.subjects.confirmed_stream())
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let consumer = stream
            .create_consumer(jetstream::consumer::pull::OrderedConfig {
                filter_subject: self.subjects.confirmed_event("polylogue.rows"),
                deliver_policy: jetstream::consumer::DeliverPolicy::All,
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        let mut units: Vec<PolylogueRows> = Vec::new();
        // Read up to the last unit on the stream now; later ones arrive with
        // the next pass.
        let last = match stream
            .get_last_raw_message_by_subject(&self.subjects.confirmed_event("polylogue.rows"))
            .await
        {
            Ok(m) => m.sequence,
            Err(e) if e.kind() == jetstream::stream::LastRawMessageErrorKind::NoMessageFound => 0,
            Err(e) => return Err(Error::Unreachable(e.to_string())),
        };
        let mut messages = consumer
            .messages()
            .await
            .map_err(|e| Error::Unreachable(e.to_string()))?;
        while last > 0 {
            let message = tokio::time::timeout(ACK_WAIT, messages.next())
                .await
                .map_err(|_| Error::Unreachable("the confirmed stream stopped".into()))?
                .ok_or_else(|| Error::Unreachable("the confirmed stream closed".into()))?
                .map_err(|e| Error::Unreachable(e.to_string()))?;
            let at = message
                .info()
                .map_err(|e| Error::Unreachable(e.to_string()))?
                .stream_sequence;
            let confirmed: sinex_protocol::Confirmed = serde_json::from_slice(&message.payload)?;
            let sinex_protocol::Confirmed::Event(event) = confirmed else {
                continue;
            };
            let payload = event.payload.ok_or_else(|| {
                Error::Invalid(format!("{} arrived without its payload", event.id))
            })?;
            let unit: PolylogueRows = serde_json::from_value(payload)?;
            if unit.archive == archive {
                units.push(unit);
            }
            if at >= last {
                break;
            }
        }
        units.sort_by_key(|u| u.position);
        units.dedup_by_key(|u| u.position);
        let reader = hashed_blobs::Reader::new(self.store_root.clone(), vec![]);
        let mut done = Restored::default();
        for (index, unit) in units.iter().enumerate() {
            if unit.position != index as u64 + 1 {
                return Err(Error::Invalid(format!(
                    "Sinex holds unit {} but not unit {}: a unit is missing",
                    unit.position,
                    index + 1
                )));
            }
            let name = unit
                .batch
                .strip_prefix("blake3:")
                .ok_or_else(|| Error::Invalid(format!("bad batch name {}", unit.batch)))?;
            let hash = hashed_blobs::Hash(
                rows::unhex(name)?
                    .try_into()
                    .map_err(|_| Error::Invalid(format!("bad batch name {}", unit.batch)))?,
            );
            let mut lines = Vec::new();
            std::io::Read::read_to_end(&mut reader.read(&hash)?, &mut lines)?;
            let count = lines
                .split(|b| *b == b'\n')
                .filter(|l| !l.is_empty())
                .count() as u64;
            if count != unit.rows {
                return Err(Error::Invalid(format!(
                    "unit {} holds {count} rows, not the {} its event names",
                    unit.position, unit.rows
                )));
            }
            done.rows += restoring.apply(&lines)?;
            done.units += 1;
            done.local_seq = done.local_seq.max(unit.local_seq);
        }
        restoring.finish(archive, done.local_seq)?;
        Ok(done)
    }
}

fn checked(ack: Acknowledgment) -> Result<Acknowledgment> {
    match &ack.refusal {
        Some(reason) => Err(Error::Refused {
            frame: ack.frame.clone(),
            reason: reason.clone(),
        }),
        None => Ok(ack),
    }
}

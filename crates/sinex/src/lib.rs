//! Polylogue on Sinex (architecture 7.1). Two modes: `standalone` (the
//! default; this crate is not used) and `primary`, where Sinex holds the
//! archive's evidence and Polylogue's databases are copies that can be
//! restored from it.
//!
//! Input blobs and units of archive rows cross as content-addressed materials.
//! A unit keeps its write counter, kind and key even after deletion. An excision
//! publishes hashes of replacement batches or tombstones, never old row bodies.
//! Restoring first folds the confirmed metadata, then fetches only current
//! batches. Older intentionally retired payloads are not needed.
//!
//! Material frames include their content hash; row frames keep archive/position
//! identity. Publication asks Sinex which frames it holds, places current payloads,
//! then sends the highest missing position first. An interrupted suffix has an
//! explicit gap and cannot masquerade as a complete pre-excision archive.
use async_nats::jetstream;
use futures_util::StreamExt;
use polylogue_store::sinex_rows::{self, Published as ArchiveRows, RestoreUnit, Restoring, Unit};
use serde::Deserialize;
use sinex_model::event_types::polylogue::PolylogueRows;
use sinex_model::{Class, InputRef, MaterialId, RecordLocation};
use sinex_protocol::{Acknowledgment, Have, Ingress, MaterialFrame, ProducerEvent, Subjects};
use std::collections::{BTreeMap, HashMap};
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

/// Logical publication does not confirm physical material removal.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalRetirement {
    #[default]
    NotRequested,
    /// Excision mappings exist, but no native retirement receipt was obtained.
    Unverified,
}

/// What one publish pass did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Published {
    /// Units Sinex already held before this pass.
    pub held: u64,
    pub units: u64,
    pub rows: u64,
    pub blobs: u64,
    pub physical_retirement: PhysicalRetirement,
}

/// What a restore did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Restored {
    pub units: u64,
    pub rows: u64,
    /// The archive's write counter after the restore.
    pub commit_seq: i64,
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

    async fn place_blobs(
        &mut self,
        archive: &ArchiveRows,
        id: &str,
        hashes: &[[u8; 32]],
        done: &mut Published,
    ) -> Result<()> {
        let frames: Vec<_> = hashes
            .iter()
            .map(|h| format!("blob:{}", sinex_rows::hex(h)))
            .collect();
        let known = self.have(&frames).await?;
        for (hash, frame) in hashes.iter().zip(frames) {
            if let Some(ack) = known.get(&frame) {
                checked(ack.clone())?;
                continue;
            }
            let length = archive.copy_blob_to(hash, &self.store)?;
            let expected = hashed_blobs::Hash(*hash);
            self.material(
                frame,
                None,
                &expected,
                length,
                format!("blob:{}", expected.hex()),
                serde_json::json!({"archive":id}),
            )
            .await?;
            done.blobs += 1;
        }
        Ok(())
    }

    async fn place_batch(&mut self, id: &str, bytes: &[u8]) -> Result<MaterialId> {
        let hash = self.store.put(bytes, 3)?;
        // A material acknowledgment is keyed by frame, not by incoming bytes.
        // Include the content address so an unknown outcome can be retried safely.
        let frame = format!("batch:{id}:{}", hash.hex());
        let known = self.have(std::slice::from_ref(&frame)).await?;
        self.material(
            frame.clone(),
            known.get(&frame),
            &hash,
            bytes.len() as u64,
            frame,
            serde_json::json!({"archive":id,"media_type":"application/jsonl"}),
        )
        .await
    }

    /// Publishes every unit of the archive at `root` that Sinex does not hold
    /// yet. Places current input blobs and batches before confirming row events;
    /// an interrupted suffix remains explicitly incomplete.
    pub async fn publish(&mut self, root: &Path) -> Result<Published> {
        let archive = ArchiveRows::open(root)?;
        let id = archive
            .archive_id()
            .map_err(|e| Error::Invalid(format!("the archive has no identity: {e}")))?;
        let units = archive.units(0)?;
        let frames: Vec<_> = (1..=units.len())
            .map(|position| format!("rows:{id}:{position}"))
            .collect();
        let known = self.have(&frames).await?;
        for ack in known.values() {
            checked(ack.clone())?;
        }
        let mut missing: Vec<_> = (0..units.len())
            .filter(|i| !known.contains_key(&frames[*i]))
            .collect();
        // Publish the suffix boundary first. A stopped pass then has an explicit
        // gap, never a complete-looking prefix with retired intermediate bytes.
        if let Some(last) = missing.pop() {
            missing.insert(0, last);
        }
        // Freeze/validate the complete mapping before placing replacement bytes.
        // Only current replacements need payloads: older ones may be retired.
        let mut current = BTreeMap::new();
        let mut publications = BTreeMap::new();
        for unit in &units {
            if let Some(publication) = archive.publication(unit)? {
                for replacement in &publication.replacements {
                    current.insert(replacement.unit.clone(), replacement.clone());
                }
                publications.insert(unit.clone(), publication);
            }
        }
        let mut done = Published {
            held: known.len() as u64,
            physical_retirement: if publications.is_empty() {
                PhysicalRetirement::NotRequested
            } else {
                PhysicalRetirement::Unverified
            },
            ..Published::default()
        };
        if !missing.is_empty() {
            for replacement in current.values() {
                let Some(name) = &replacement.batch else {
                    continue;
                };
                let bytes = archive.replacement(name)?;
                if bytes.len() as u64 != replacement.bytes {
                    return Err(Error::Invalid(
                        "replacement length differs from its manifest".into(),
                    ));
                }
                let hashes: Vec<_> = replacement
                    .blobs
                    .iter()
                    .map(|name| parse_hash(name).map(|h| h.0))
                    .collect::<Result<_>>()?;
                self.place_blobs(&archive, &id, &hashes, &mut done).await?;
                self.place_batch(&id, &bytes).await?;
            }
        }
        for index in missing {
            let unit = &units[index];
            self.publish_unit(
                &archive,
                &id,
                index + 1,
                unit,
                publications.get(unit),
                &mut done,
            )
            .await?;
        }
        Ok(done)
    }

    async fn publish_unit(
        &mut self,
        archive: &ArchiveRows,
        id: &str,
        position: usize,
        unit: &Unit,
        publication: Option<&sinex_rows::ExcisionPublication>,
        done: &mut Published,
    ) -> Result<()> {
        let (frame, rows) = self
            .unit_frame(archive, id, position, unit, publication, done)
            .await?;
        self.send(&frame).await?;
        done.units += 1;
        done.rows += rows;
        Ok(())
    }

    async fn unit_frame(
        &mut self,
        archive: &ArchiveRows,
        id: &str,
        position: usize,
        unit: &Unit,
        publication: Option<&sinex_rows::ExcisionPublication>,
        done: &mut Published,
    ) -> Result<(Ingress, u64)> {
        let batch = archive.read(unit)?;
        self.place_blobs(archive, id, &batch.blobs, done).await?;
        let batch_hash = self.store.put(&batch.lines[..], 3)?;
        let material = self.place_batch(id, &batch.lines).await?;
        let event = PolylogueRows {
            archive: id.to_owned(),
            position: position as u64,
            // Sinex names this field `local_seq` until its own rename.
            local_seq: unit.commit_seq,
            unit: unit.kind.clone(),
            key: unit.key.clone(),
            batch: format!("blake3:{}", batch_hash.hex()),
            rows: batch.rows,
            blobs: batch
                .blobs
                .iter()
                .map(|h| format!("blake3:{}", sinex_rows::hex(h)))
                .collect(),
            excision: publication
                .map(|p| serde_json::to_value(p).and_then(serde_json::from_value))
                .transpose()?,
        };
        let payload = serde_json::to_value(&event)?;
        Ok((
            Ingress::Events {
                frame: format!("rows:{id}:{position}"),
                failures: vec![],
                events: vec![ProducerEvent {
                    event_type: "polylogue.rows".into(),
                    payload,
                    location: Some(RecordLocation {
                        material,
                        anchor: 0,
                    }),
                    ordinal: 0,
                    parents: vec![],
                    reading_order: position as i64,
                    class: Class::Open,
                }],
            },
            batch.rows,
        ))
    }

    /// Restores into an empty locked target, or updates the same archive, from
    /// Sinex. Selects current batches from confirmed metadata before payload IO,
    /// then checks and installs their rows atomically. Afterwards the loop builds
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
        let mut catalog = sinex_rows::Catalog::new()?;
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
        let mut done = last == 0;
        while !done {
            let message = tokio::time::timeout(ACK_WAIT, messages.next())
                .await
                .map_err(|_| Error::Unreachable("the confirmed stream stopped".into()))?
                .ok_or_else(|| Error::Unreachable("the confirmed stream closed".into()))?
                .map_err(|e| Error::Unreachable(e.to_string()))?;
            let at = message
                .info()
                .map_err(|e| Error::Unreachable(e.to_string()))?
                .stream_sequence;
            done = at >= last;
            let confirmed: sinex_protocol::Confirmed = serde_json::from_slice(&message.payload)?;
            let sinex_protocol::Confirmed::Event(event) = confirmed else {
                continue;
            };
            let payload = event.payload.ok_or_else(|| {
                Error::Invalid(format!("{} arrived without its payload", event.id))
            })?;
            let unit: PolylogueRows = serde_json::from_value(payload.clone())?;
            if unit.archive == archive {
                let excision = payload
                    .get("excision")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()?;
                catalog.push(
                    unit.position,
                    &RestoreUnit {
                        unit: Unit {
                            commit_seq: unit.local_seq,
                            kind: unit.unit,
                            key: unit.key,
                        },
                        batch: Some(unit.batch),
                        bytes: None,
                        rows: unit.rows,
                        blobs: unit.blobs,
                        excision,
                    },
                )?;
            }
        }
        let selected = catalog.finish()?;
        let mut done = Restored {
            commit_seq: selected.commit_seq(),
            ..Restored::default()
        };
        let reader = hashed_blobs::Reader::new(self.store_root.clone(), vec![]);
        done.rows = restoring.install(archive, &selected, |unit| {
            // Required current input bytes are checked too; an unretired missing
            // source blob is corruption, even if the row batch itself survives.
            for name in &unit.blobs {
                let hash = parse_hash(name)
                    .map_err(|e| polylogue_store::StoreError::Invalid(e.to_string()))?;
                let mut plain = reader.read_plain(&hash, None, || Ok(()))?;
                std::io::copy(&mut plain, &mut std::io::sink())?;
            }
            let Some(name) = &unit.batch else {
                return Ok(None);
            };
            let hash = hashed_blobs::Hash(
                sinex_rows::unhex(name.strip_prefix("blake3:").ok_or_else(|| {
                    polylogue_store::StoreError::Invalid("missing batch hash".into())
                })?)?
                .try_into()
                .map_err(|_| {
                    polylogue_store::StoreError::Invalid("invalid batch hash length".into())
                })?,
            );
            Ok(Some(reader.read_plain(&hash, unit.bytes, || Ok(()))?))
        })?;
        done.units = selected.len();
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

fn parse_hash(name: &str) -> Result<hashed_blobs::Hash> {
    let hex = name
        .strip_prefix("blake3:")
        .ok_or_else(|| Error::Invalid("missing BLAKE3 hash".into()))?;
    Ok(hashed_blobs::Hash(
        sinex_rows::unhex(hex)?
            .try_into()
            .map_err(|_| Error::Invalid("invalid BLAKE3 hash length".into()))?,
    ))
}

#[cfg(test)]
mod tests;

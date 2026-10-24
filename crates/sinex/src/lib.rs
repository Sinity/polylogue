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
use sinex_protocol::{
    Acknowledgment, Have, Ingress, MaterialFrame, ProducerEvent, RetireBatches, Retirement,
    Subjects,
};
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
        Self::load_from(&Self::path())
    }

    /// Reads one explicitly selected configuration file.
    pub fn load_from(path: &Path) -> Result<Option<Self>> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text)
                .map(Some)
                .map_err(|e| Error::Invalid(format!("{}: {e}", path.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(std::io::Error::new(e.kind(), format!("{}: {e}", path.display())).into()),
        }
    }
}

/// Logical publication does not confirm physical material removal.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalRetirement {
    #[default]
    NotRequested,
    Completed,
    /// Superseded bytes still serve unrelated references.
    Shared {
        files: u64,
    },
    Pending {
        files: u64,
        shared: u64,
    },
    /// The server has no row-batch retirement request handler.
    Unsupported,
}

impl std::fmt::Display for PhysicalRetirement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRequested => f.write_str("not requested"),
            Self::Completed => f.write_str("completed"),
            Self::Shared { files } => write!(f, "retained {files} shared files"),
            Self::Pending { files, shared } => {
                write!(f, "{files} files pending, {shared} shared files retained")
            }
            Self::Unsupported => f.write_str("unsupported by this server"),
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
    machine_memo: Option<machine_history::Memo>,
}

const ACK_WAIT: Duration = Duration::from_secs(120);

mod confirmed;
mod machine_history;

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
            machine_memo: None,
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
        for hashes in hashes.chunks(500) {
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
        }
        Ok(())
    }

    async fn place_named_blobs(
        &mut self,
        archive: &ArchiveRows,
        id: &str,
        names: &[String],
        done: &mut Published,
    ) -> Result<()> {
        // Preserve validation-before-placement even for an invalid late name.
        for name in names {
            parse_hash(name)?;
        }
        for names in names.chunks(500) {
            let hashes: Vec<_> = names
                .iter()
                .map(|name| parse_hash(name).map(|h| h.0))
                .collect::<Result<_>>()?;
            self.place_blobs(archive, id, &hashes, done).await?;
        }
        Ok(())
    }

    async fn place_batch(
        &mut self,
        id: &str,
        bytes: &mut std::fs::File,
    ) -> Result<(MaterialId, hashed_blobs::Hash)> {
        use std::io::Seek;
        bytes.rewind()?;
        let length = bytes.metadata()?.len();
        let hash = self.store.put_plain_checked::<Error>(bytes, 3, || Ok(()))?;
        // A material acknowledgment is keyed by frame, not by incoming bytes.
        // Include the content address so an unknown outcome can be retried safely.
        let frame = format!("batch:{id}:{}", hash.hex());
        let known = self.have(std::slice::from_ref(&frame)).await?;
        let material = self
            .material(
                frame.clone(),
                known.get(&frame),
                &hash,
                length,
                frame,
                serde_json::json!({"archive":id,"media_type":"application/jsonl"}),
            )
            .await?;
        Ok((material, hash))
    }

    /// Publishes every unit of the archive at `root` that Sinex does not hold
    /// yet. Places current input blobs and batches before confirming row events;
    /// an interrupted suffix remains explicitly incomplete.
    pub async fn publish(&mut self, root: &Path) -> Result<Published> {
        let archive = ArchiveRows::open(root)?;
        let id = archive
            .archive_id()
            .map_err(|e| Error::Invalid(format!("the archive has no identity: {e}")))?;
        let machine_identity = archive
            .machine_history_state()?
            .map(|state| self.machine_identity(&id, state))
            .transpose()?;
        let mut inventory = archive.inventory()?;
        let machine_catalog = if let Some(identity) = machine_identity {
            inventory.enable_machine_history()?;
            if self
                .machine_memo
                .as_ref()
                .is_some_and(|memo| memo.identity == identity)
            {
                None
            } else {
                Some(sinex_rows::Catalog::new()?)
            }
        } else {
            None
        };
        let mut machine_acks = blake3::Hasher::new();
        let mut machine_frames = 0;

        let mut after = 0;
        while after < inventory.len() {
            let end = (after + 500).min(inventory.len());
            let frames: Vec<_> = ((after + 1)..=end)
                .map(|position| format!("rows:{id}:{position}"))
                .collect();
            let known = self.have(&frames).await?;
            let mut held = Vec::with_capacity(frames.len());
            for (offset, frame) in frames.iter().enumerate() {
                if let Some(ack) = known.get(frame) {
                    checked(ack.clone())?;
                    held.push(after + offset as u64 + 1);
                }
            }
            if machine_identity.is_some() {
                machine_frames += machine_history::record(
                    &id,
                    machine_catalog.as_ref(),
                    &mut machine_acks,
                    inventory.machine_candidates(after, end)?,
                    &known,
                )?;
            }
            inventory.mark_held(&held)?;
            after = end;
        }
        if let Some(identity) = machine_identity
            && machine_frames > 0
        {
            self.check_machine_history(
                &archive,
                &id,
                identity,
                *machine_acks.finalize().as_bytes(),
                machine_catalog,
                &inventory,
            )
            .await?;
        }
        let mut done = Published {
            held: inventory.held(),
            ..Published::default()
        };
        if let Some(last) = inventory.last_missing()? {
            // Complete replacement selection was checked before opening payloads.
            let mut cursor = None;
            while let Some(replacement) = inventory.next_replacement(cursor.as_ref())? {
                if let Some(name) = &replacement.batch {
                    let mut bytes = archive.replacement(name)?;
                    if bytes.metadata()?.len() != replacement.bytes {
                        return Err(Error::Invalid(
                            "replacement length differs from its manifest".into(),
                        ));
                    }
                    self.place_named_blobs(&archive, &id, &replacement.blobs, &mut done)
                        .await?;
                    self.place_batch(&id, &mut bytes).await?;
                }
                cursor = Some(replacement.unit);
            }
            // Publish the suffix boundary first, then fill every earlier gap.
            let unit = inventory.unit(last)?;
            let publication = archive.publication(&unit)?;
            self.publish_unit(&archive, &id, last, &unit, publication.as_ref(), &mut done)
                .await?;
            inventory.mark_held(&[last])?;
            let mut cursor = 0;
            while let Some((position, unit)) = inventory.next_missing(cursor)? {
                let publication = archive.publication(&unit)?;
                self.publish_unit(
                    &archive,
                    &id,
                    position,
                    &unit,
                    publication.as_ref(),
                    &mut done,
                )
                .await?;
                inventory.mark_held(&[position])?;
                cursor = position;
            }
        }
        if done.held + done.units != inventory.len() {
            return Err(Error::Invalid(
                "publication did not account for every unit".into(),
            ));
        }
        if let Some(owner) = inventory.latest_excision()? {
            let publication = archive
                .publication(&owner)?
                .ok_or_else(|| Error::Invalid("excision has no publication proof".into()))?;
            let request = RetireBatches {
                archive: id.clone(),
                owner: sinex_model::event_types::polylogue::ArchiveUnit {
                    commit_seq: owner.commit_seq,
                    kind: owner.kind,
                    key: owner.key,
                },
                digest: publication.digest,
            };
            done.physical_retirement = self.retire(&request).await?;
        }
        Ok(done)
    }

    async fn retire(&self, proof: &RetireBatches) -> Result<PhysicalRetirement> {
        let response = self
            .client
            .request(
                self.subjects.retire(&self.producer),
                serde_json::to_vec(proof)?.into(),
            )
            .await;
        let response = match response {
            Ok(response) => response,
            Err(error) if error.kind() == async_nats::RequestErrorKind::NoResponders => {
                return Ok(PhysicalRetirement::Unsupported);
            }
            Err(error) => return Err(Error::Unreachable(format!("row-batch retirement: {error}"))),
        };
        let result: Retirement = serde_json::from_slice(&response.payload)?;
        if let Some(reason) = result.refusal {
            return Err(Error::Refused {
                frame: "row-batch retirement".into(),
                reason,
            });
        }
        Ok(if result.pending_files > 0 {
            PhysicalRetirement::Pending {
                files: result.pending_files,
                shared: result.retained_files,
            }
        } else if result.retained_files > 0 {
            PhysicalRetirement::Shared {
                files: result.retained_files,
            }
        } else {
            PhysicalRetirement::Completed
        })
    }

    async fn publish_unit(
        &mut self,
        archive: &ArchiveRows,
        id: &str,
        position: u64,
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
        position: u64,
        unit: &Unit,
        publication: Option<&sinex_rows::ExcisionPublication>,
        done: &mut Published,
    ) -> Result<(Ingress, u64)> {
        let mut batch = archive.read(unit)?;
        self.place_blobs(archive, id, &batch.blobs, done).await?;
        let (material, batch_hash) = self.place_batch(id, &mut batch.file).await?;
        let event = PolylogueRows {
            archive: id.to_owned(),
            position,
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
                    reading_order: i64::try_from(position).map_err(|_| {
                        Error::Invalid("publication position exceeds the wire order range".into())
                    })?,
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
        let mut restoring = Restoring::with_roots(store_archive, vec![self.store_root.clone()])?;
        let selected = self.confirmed(archive, sinex_rows::Catalog::new()?).await?;
        let mut done = Restored {
            commit_seq: selected.commit_seq(),
            ..Restored::default()
        };
        let reader = hashed_blobs::Reader::new(self.store_root.clone(), vec![]);
        // A complete selected view determines required bytes. Shared source
        // blobs need one verification in this restore; no result survives it.
        selected.verify_source_blobs(|name| {
            let hash = parse_hash(name)
                .map_err(|e| polylogue_store::StoreError::Invalid(e.to_string()))?;
            let mut plain = reader.read_plain(&hash, None, || Ok(()))?;
            Ok(std::io::copy(&mut plain, &mut std::io::sink())?)
        })?;
        done.rows = restoring.install(archive, &selected, |unit| {
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

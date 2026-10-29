//! Prepared producer frames. A frame keeps its original bytes and identity
//! across every retry, including a process restart. Acknowledged frames stay
//! until the confirmed consumer has committed their position.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use sinex_protocol::{Acknowledgment, Ingress};
use std::cell::Cell;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Prepared {
    pub order: u64,
    pub frame: Ingress,
    pub acknowledgment: Option<Acknowledgment>,
    pub body: bool,
    pub dependency: Option<String>,
}

/// The producer owns this directory under Sinex's store root. It contains
/// prepared writes, never a second ledger of committed inputs.
pub(crate) struct Spool {
    root: PathBuf,
    next: Cell<u64>,
    _lock: File,
}

impl Spool {
    pub fn open(store: &Path, producer: &str) -> Result<Self> {
        let identity = blake3::hash(producer.as_bytes()).to_hex();
        let root = store.join("producer-spool").join(identity.as_str());
        fs::create_dir_all(&root)?;
        File::open(store)?.sync_all()?;
        File::open(root.parent().expect("spool parent"))?.sync_all()?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join("lock"))?;
        lock.try_lock()
            .map_err(|error| Error::Invalid(format!("producer spool is already open: {error}")))?;
        let spool = Self {
            root,
            next: Cell::new(1),
            _lock: lock,
        };
        let last = spool
            .prepared()?
            .iter()
            .map(|prepared| prepared.order)
            .max()
            .unwrap_or(0);
        spool.next.set(
            last.checked_add(1)
                .ok_or_else(|| Error::Invalid("producer spool order exhausted".into()))?,
        );
        Ok(spool)
    }

    fn path(&self, identity: &str) -> PathBuf {
        self.root.join(format!(
            "{}.json",
            blake3::hash(identity.as_bytes()).to_hex()
        ))
    }

    /// An identity can be prepared once. Equal repeats preserve the first
    /// serialization; changed contents under that identity are a defect.
    pub fn prepare(&self, frame: Ingress) -> Result<Prepared> {
        self.prepare_frame(frame, false)
    }

    fn prepare_frame(&self, frame: Ingress, body: bool) -> Result<Prepared> {
        let path = self.path(frame.frame());
        if let Some(prepared) = self.read(&path)? {
            if serde_json::to_value(&prepared.frame)? != serde_json::to_value(&frame)? {
                return Err(Error::Invalid(format!(
                    "prepared frame {} changed contents",
                    frame.frame()
                )));
            }
            return Ok(prepared);
        }
        let order = self.next.get();
        self.next.set(
            order
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("producer spool order exhausted".into()))?,
        );
        let prepared = Prepared {
            order,
            frame,
            acknowledgment: None,
            body,
            dependency: None,
        };
        self.write(&path, &prepared)?;
        Ok(prepared)
    }

    pub fn prepare_material(&self, frame: Ingress, mut bytes: impl Read) -> Result<Prepared> {
        let Ingress::Material { material, .. } = &frame else {
            return Err(Error::Invalid(
                "only a material frame has captured bytes".into(),
            ));
        };
        let expected_hash = material
            .blob
            .strip_prefix("blake3:")
            .ok_or_else(|| Error::Invalid("material digest must be BLAKE3".into()))?;
        let path = self.body_path(frame.frame());
        if !path.exists() {
            let temporary = self.root.join("body-preparing");
            let mut file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&temporary)?;
            let mut digest = blake3::Hasher::new();
            let mut length = 0u64;
            let mut buffer = vec![0u8; 1 << 16];
            loop {
                let count = bytes.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                file.write_all(&buffer[..count])?;
                digest.update(&buffer[..count]);
                length = length
                    .checked_add(count as u64)
                    .ok_or_else(|| Error::Invalid("material length overflow".into()))?;
            }
            if length != material.length || digest.finalize().to_hex().as_str() != expected_hash {
                return Err(Error::Invalid(
                    "captured bytes differ from their prepared material".into(),
                ));
            }
            file.sync_all()?;
            fs::rename(&temporary, &path)?;
            File::open(&self.root)?.sync_all()?;
        }
        self.prepare_frame(frame, true)
    }

    pub fn body(&self, identity: &str) -> Result<File> {
        Ok(File::open(self.body_path(identity))?)
    }

    fn body_path(&self, identity: &str) -> PathBuf {
        self.root.join(format!(
            "{}.body",
            blake3::hash(identity.as_bytes()).to_hex()
        ))
    }

    pub fn get(&self, identity: &str) -> Result<Option<Prepared>> {
        self.read(&self.path(identity))
    }

    /// The canonical product frame is frozen once the source coordinates are
    /// assigned. Retrying a capture keeps that exact dependency.
    pub fn depend(&self, identity: &str, dependency: &str) -> Result<()> {
        let path = self.path(identity);
        let mut prepared = self
            .read(&path)?
            .ok_or_else(|| Error::Invalid("capture is not prepared".into()))?;
        if prepared
            .dependency
            .as_deref()
            .is_some_and(|previous| previous != dependency)
        {
            return Err(Error::Invalid(
                "capture's canonical products changed".into(),
            ));
        }
        prepared.dependency = Some(dependency.into());
        self.write(&path, &prepared)
    }

    pub fn retire(&self, applied: i64) -> Result<()> {
        let entries = self.prepared()?;
        let acknowledgments: std::collections::HashMap<_, _> = entries
            .iter()
            .filter_map(|entry| {
                entry
                    .acknowledgment
                    .as_ref()
                    .map(|ack| (entry.frame.frame(), ack))
            })
            .collect();
        for prepared in &entries {
            let settled = if let Some(dependency) = &prepared.dependency {
                acknowledgments.get(dependency.as_str()).is_some_and(|ack| {
                    ack.refusal.is_none()
                        && ack.commit_seq.is_some_and(|position| position <= applied)
                })
            } else {
                prepared.acknowledgment.as_ref().is_some_and(|ack| {
                    ack.refusal.is_none()
                        && (ack.commit_seq.is_some_and(|position| position <= applied)
                            || matches!(prepared.frame, Ingress::RegisterKinds { .. }))
                })
            };
            if settled {
                fs::remove_file(self.path(prepared.frame.frame()))?;
                if prepared.body {
                    fs::remove_file(self.body_path(prepared.frame.frame()))?;
                }
            }
        }
        File::open(&self.root)?.sync_all()?;
        Ok(())
    }

    pub fn acknowledge(&self, acknowledgment: Acknowledgment) -> Result<()> {
        let path = self.path(&acknowledgment.frame);
        let mut prepared = self.read(&path)?.ok_or_else(|| {
            Error::Invalid(format!(
                "acknowledgment has no prepared frame {}",
                acknowledgment.frame
            ))
        })?;
        if let Some(previous) = &prepared.acknowledgment {
            if serde_json::to_value(previous)? != serde_json::to_value(&acknowledgment)? {
                return Err(Error::Invalid(format!(
                    "acknowledgment changed for {}",
                    acknowledgment.frame
                )));
            }
            return Ok(());
        }
        prepared.acknowledgment = Some(acknowledgment);
        self.write(&path, &prepared)
    }

    /// Includes acknowledged writes until their effects are applied locally.
    pub fn prepared(&self) -> Result<Vec<Prepared>> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let path = entry?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json")
            {
                entries.push(self.read(&path)?.ok_or_else(|| {
                    Error::Invalid(format!("prepared frame disappeared: {}", path.display()))
                })?);
            }
        }
        entries.sort_by_key(|prepared| prepared.order);
        Ok(entries)
    }

    /// Only committed confirmed positions authorize retirement. Frames with
    /// no event position (material and registration frames) belong to a
    /// dependent write and are retired explicitly with that write.
    pub fn settle(&self, identity: &str, applied: i64) -> Result<()> {
        let path = self.path(identity);
        let Some(prepared) = self.read(&path)? else {
            return Ok(());
        };
        let acknowledgment = prepared.acknowledgment.as_ref().ok_or_else(|| {
            Error::Invalid(format!("cannot retire unacknowledged frame {identity}"))
        })?;
        if acknowledgment.refusal.is_some()
            || acknowledgment
                .commit_seq
                .is_none_or(|position| position > applied)
        {
            return Err(Error::Invalid(format!(
                "frame {identity} is not applied at confirmed position {applied}"
            )));
        }
        fs::remove_file(&path)?;
        File::open(&self.root)?.sync_all()?;
        Ok(())
    }

    fn read(&self, path: &Path) -> Result<Option<Prepared>> {
        match fs::read(path) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn write(&self, path: &Path, prepared: &Prepared) -> Result<()> {
        let bytes = serde_json::to_vec(prepared)?;
        let temporary = self.root.join("preparing");
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        File::open(&self.root)?.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

//! Pending producer operations. Immutable frame and body files become visible
//! together; small queue metadata never contains event or request payloads.
use crate::{Error, Result};
use polylogue_store::primary::Capture;
use serde::{Deserialize, Serialize};
use sinex_protocol::{Acknowledgment, Ingress};
use std::cell::Cell;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Kind {
    Material,
    Events,
    ReplaySequence,
    RegisterKinds,
    Chunk,
    LargeFrame,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Bytes {
    pub blob: String,
    pub length: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Material {
    pub id: String,
    pub blob: String,
    pub length: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Group {
    pub captures_hash: String,
    pub captures_count: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Prepared {
    pub order: u64,
    pub identity: String,
    pub kind: Kind,
    pub frame_hash: String,
    pub frame_length: u64,
    pub body: Option<Bytes>,
    pub material: Option<Material>,
    pub capture: Option<Capture>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captured: Option<Bytes>,
    pub request: bool,
    pub group: Option<Group>,
    #[serde(skip)]
    pub acknowledged: bool,
    #[serde(skip)]
    pub commit_seq: Option<i64>,
    #[serde(skip)]
    pub refused: bool,
}

#[derive(Serialize, Deserialize)]
struct Checked<T> {
    value: T,
    digest: String,
}

#[derive(Deserialize)]
struct AckHeader {
    frame: String,
    material: Option<String>,
    commit_seq: Option<i64>,
    refusal: Option<serde::de::IgnoredAny>,
}

#[derive(Serialize, Deserialize)]
struct Receipt {
    blob: String,
    length: u64,
    commit_seq: Option<i64>,
    refused: bool,
}

/// Only pending operations live here. Ordered and capture-membership indexes
/// are rebuildable from operation directories, never source progress.
pub(crate) struct Spool {
    root: PathBuf,
    first: Cell<u64>,
    next: Cell<u64>,
    index_dirty: Cell<bool>,
    _lock: File,
}

impl Spool {
    pub fn open(store: &Path, producer: &str) -> Result<Self> {
        let root = store.join("producer-spool").join(hash(producer.as_bytes()));
        fs::create_dir_all(root.join("operations"))?;
        fs::create_dir_all(root.join("retired"))?;
        sync(&root)?;
        sync(store)?;
        sync(root.parent().expect("spool parent"))?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join("lock"))?;
        lock.try_lock()
            .map_err(|error| Error::Invalid(format!("producer spool is already open: {error}")))?;
        for entry in fs::read_dir(&root)? {
            let path = entry?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json" || extension == "body")
            {
                return invalid(
                    "producer spool contains an earlier pending-frame format; preserve it before an explicit transition",
                );
            }
        }
        let spool = Self {
            root,
            first: Cell::new(1),
            next: Cell::new(1),
            index_dirty: Cell::new(true),
            _lock: lock,
        };
        spool.rebuild_indexes()?;
        for entry in fs::read_dir(spool.root.join("retired"))? {
            let path = entry?.path();
            if fs::symlink_metadata(&path)?.is_dir() {
                fs::remove_dir_all(path)?;
            }
        }
        sync(&spool.root.join("retired"))?;
        Ok(spool)
    }

    fn operation(&self, identity: &str) -> PathBuf {
        self.root.join("operations").join(hash(identity.as_bytes()))
    }
    fn index(&self, kind: &str, key: &str) -> PathBuf {
        self.root.join("index").join(kind).join(key)
    }

    pub fn get(&self, identity: &str) -> Result<Option<Prepared>> {
        let Some(mut prepared) =
            read_checked::<Prepared>(&self.operation(identity).join("metadata.json"))?
        else {
            return Ok(None);
        };
        if prepared.identity != identity {
            return invalid("spool operation identity differs from its directory");
        }
        let directory = self.operation(identity);
        check_length(&directory.join("frame.json"), prepared.frame_length)?;
        if let Some(body) = &prepared.body {
            check_length(&directory.join("body"), body.length)?;
        }
        if let Some(captured) = &prepared.captured {
            check_length(&directory.join("captured.json"), captured.length)?;
        }
        if let Some(receipt) = read_checked::<Receipt>(&directory.join("receipt.json"))? {
            check_length(&directory.join("ack.json"), receipt.length)?;
            prepared.acknowledged = true;
            prepared.commit_seq = receipt.commit_seq;
            prepared.refused = receipt.refused;
        } else if directory.join("ack.json").try_exists()? {
            let acknowledgment: AckHeader = decode_json(regular(&directory.join("ack.json"))?)?;
            validate_ack_header(
                &prepared,
                &acknowledgment.frame,
                acknowledgment.material.as_deref(),
                acknowledgment.commit_seq,
                acknowledgment.refusal.is_some(),
            )?;
            self.write_receipt(
                &prepared,
                acknowledgment.commit_seq,
                acknowledgment.refusal.is_some(),
            )?;
            prepared.acknowledged = true;
            prepared.commit_seq = acknowledgment.commit_seq;
            prepared.refused = acknowledgment.refusal.is_some();
        }
        Ok(Some(prepared))
    }

    /// The immutable serialization used on every send. Large frames can be
    /// copied from this file without decoding their event vectors.
    pub fn frame(&self, identity: &str) -> Result<File> {
        let prepared = self.required(identity)?;
        verified(
            &self.operation(identity).join("frame.json"),
            &prepared.frame_hash,
            prepared.frame_length,
        )
    }

    pub fn decode_frame(&self, identity: &str) -> Result<Ingress> {
        decode_json(self.frame(identity)?)
    }

    pub fn body(&self, identity: &str) -> Result<File> {
        let prepared = self.required(identity)?;
        let body = prepared
            .body
            .ok_or_else(|| Error::Invalid("operation has no material body".into()))?;
        verified(
            &self.operation(identity).join("body"),
            &body.blob,
            body.length,
        )
    }

    pub fn acknowledgment(&self, identity: &str) -> Result<Option<Acknowledgment>> {
        let directory = self.operation(identity);
        let path = directory.join("ack.json");
        if !path.try_exists()? {
            return Ok(None);
        }
        let file = regular(&path)?;
        let acknowledgment: Acknowledgment = decode_json(file)?;
        let prepared = read_checked::<Prepared>(&directory.join("metadata.json"))?
            .ok_or_else(|| Error::Invalid("acknowledgment has no operation".into()))?;
        validate_ack(&prepared, &acknowledgment)?;
        if let Some(receipt) = read_checked::<Receipt>(&directory.join("receipt.json"))? {
            verified(&path, &receipt.blob, receipt.length)?;
            if receipt.commit_seq != acknowledgment.commit_seq
                || receipt.refused != acknowledgment.refusal.is_some()
            {
                return invalid("acknowledgment summary differs from its immutable receipt");
            }
        }
        Ok(Some(acknowledgment))
    }

    pub fn prepare(&self, frame: Ingress) -> Result<Prepared> {
        if let Some(prepared) = self.get(frame.frame())? {
            let desired = encoded_hash(&frame)?;
            if desired != (prepared.frame_hash.clone(), prepared.frame_length) {
                return invalid("prepared identity changed immutable frame contents");
            }
            if prepared.kind == Kind::Material && prepared.body.is_none() {
                return invalid("material must have its immutable body prepared before sending");
            }
            return Ok(prepared);
        }
        if matches!(&frame, Ingress::Material { .. }) {
            return invalid("material must have its immutable body prepared before sending");
        }
        self.prepare_operation(frame, None::<std::io::Empty>, None, None)
    }

    pub fn prepare_material(&self, frame: Ingress, bytes: impl Read) -> Result<Prepared> {
        if !matches!(&frame, Ingress::Material { .. }) {
            return invalid("only a material frame has captured bytes");
        }
        self.prepare_operation(frame, Some(bytes), None, None)
    }

    /// Complete parser facts share the material's atomic publication, but
    /// remain outside queue metadata and the raw provider manifest.
    pub fn prepare_capture(
        &self,
        frame: Ingress,
        bytes: impl Read,
        capture: &Capture,
    ) -> Result<Prepared> {
        if !matches!(&frame, Ingress::Material { .. }) {
            return invalid("only a material frame can own a capture");
        }
        self.prepare_operation(frame, Some(bytes), None, Some(capture))
    }

    pub fn capture(&self, identity: &str) -> Result<Capture> {
        let prepared = self.required(identity)?;
        if let Some(captured) = prepared.captured {
            let capture: Capture = decode_json(verified(
                &self.operation(identity).join("captured.json"),
                &captured.blob,
                captured.length,
            )?)?;
            if prepared.capture.as_ref() != Some(&capture_header(&capture)) {
                return invalid("capture sidecar differs from its queue header");
            }
            Ok(capture)
        } else {
            prepared
                .capture
                .ok_or_else(|| Error::Invalid("operation has no captured facts".into()))
        }
    }

    /// One publication owns the complete immutable capture set. Per-capture
    /// indexes merely accelerate lookups and can be rebuilt after a crash.
    pub fn prepare_group(&self, frame: Ingress, captures: &[String]) -> Result<Prepared> {
        if !matches!(
            &frame,
            Ingress::Events { .. } | Ingress::ReplaySequence { .. }
        ) {
            return invalid("capture products need one Events or ReplaySequence frame");
        }
        if captures.is_empty() {
            return invalid("capture group is empty");
        }
        self.ensure_indexes()?;
        let captures = canonical(captures)?;
        let group_hash = captures_hash(&captures)?;
        if let Some(existing) = self.group_for_sorted(&group_hash)? {
            if existing.identity != frame.frame() {
                return invalid("capture group already has another immutable frame identity");
            }
        }
        for identity in &captures {
            let raw = self.required(identity)?;
            if raw.kind != Kind::Material || raw.body.is_none() {
                return invalid("capture group member is not a durable material");
            }
            if let Some(existing) = self.capture_group(identity)? {
                if existing.group.as_ref().map(|group| &group.captures_hash) != Some(&group_hash) {
                    return invalid("capture already belongs to another canonical product group");
                }
            }
        }
        self.prepare_operation(
            frame,
            None::<std::io::Empty>,
            Some((&captures, group_hash)),
            None,
        )
    }

    pub fn group_for(&self, captures: &[String]) -> Result<Option<Prepared>> {
        self.ensure_indexes()?;
        self.group_for_sorted(&captures_hash(&canonical(captures)?)?)
    }

    fn group_for_sorted(&self, captures_hash: &str) -> Result<Option<Prepared>> {
        let Some(identity) = read_json::<String>(&self.index("groups", captures_hash))? else {
            return Ok(None);
        };
        let prepared = self.get(&identity)?.ok_or_else(|| {
            Error::Invalid("capture group index names a missing operation".into())
        })?;
        if prepared
            .group
            .as_ref()
            .map(|group| group.captures_hash.as_str())
            != Some(captures_hash)
        {
            return invalid("capture group index differs from its authoritative group");
        }
        Ok(Some(prepared))
    }

    pub fn capture_group(&self, identity: &str) -> Result<Option<Prepared>> {
        self.ensure_indexes()?;
        let Some(group) = read_json::<String>(&self.index("members", &hash(identity.as_bytes())))?
        else {
            return Ok(None);
        };
        self.get(&group)?
            .ok_or_else(|| Error::Invalid("capture membership names a missing group".into()))
            .map(Some)
    }

    /// Page size bounds returned metadata only. It never bounds the source,
    /// frame size, or number of operations that may be queued.
    pub fn pending_after(&self, after: Option<u64>, limit: usize) -> Result<Vec<Prepared>> {
        self.ensure_indexes()?;
        let mut entries = Vec::new();
        if limit == 0 {
            return Ok(entries);
        }
        let start = after.map_or(self.first.get(), |order| {
            order.saturating_add(1).max(self.first.get())
        });
        for order in start..self.next.get() {
            let Some(identity) =
                read_json::<String>(&self.index("order", &format!("{order:020}")))?
            else {
                continue;
            };
            let entry = self.required(&identity)?;
            if entry.order != order {
                return invalid("pending queue order differs from its operation");
            }
            entries.push(entry);
            if entries.len() == limit {
                break;
            }
        }
        Ok(entries)
    }

    pub fn acknowledge(&self, acknowledgment: Acknowledgment) -> Result<()> {
        let prepared = self.required(&acknowledgment.frame)?;
        validate_ack(&prepared, &acknowledgment)?;
        let directory = self.operation(&prepared.identity);
        let encoded = encoded_hash(&acknowledgment)?;
        if let Some(previous) = self.acknowledgment(&prepared.identity)? {
            if encoded_hash(&previous)? != encoded {
                return invalid("acknowledgment changed for an immutable operation");
            }
            return Ok(());
        }
        atomic_json(&directory.join("ack.json"), &acknowledgment)?;
        boundary("ack_published");
        self.store_receipt(&prepared, &acknowledgment)
    }

    fn store_receipt(&self, prepared: &Prepared, acknowledgment: &Acknowledgment) -> Result<()> {
        validate_ack(prepared, acknowledgment)?;
        self.write_receipt(
            prepared,
            acknowledgment.commit_seq,
            acknowledgment.refusal.is_some(),
        )
    }

    fn write_receipt(
        &self,
        prepared: &Prepared,
        commit_seq: Option<i64>,
        refused: bool,
    ) -> Result<()> {
        let (blob, length) = file_hash(&self.operation(&prepared.identity).join("ack.json"))?;
        atomic_checked(
            &self.operation(&prepared.identity).join("receipt.json"),
            &Receipt {
                blob,
                length,
                commit_seq,
                refused,
            },
        )
    }

    pub fn retire(&self, applied: i64) -> Result<()> {
        if applied < 0 {
            return invalid("confirmed position is negative");
        }
        self.ensure_indexes()?;
        // Indexes stay unchanged during this pass. Rebuilding after every
        // removed page would turn queue retirement into a quadratic scan.
        let through = self.next.get();
        for order in self.first.get()..through {
            let Some(identity) =
                read_json::<String>(&self.index("order", &format!("{order:020}")))?
            else {
                continue;
            };
            let Some(prepared) = self.get(&identity)? else {
                continue;
            };
            if prepared.order != order {
                return invalid("pending queue order differs from its operation");
            }
            if prepared.refused
                || !prepared.acknowledged
                || prepared
                    .commit_seq
                    .is_none_or(|position| position > applied)
            {
                continue;
            }
            if prepared.group.is_some() {
                self.visit_captures(&prepared.identity, |identity| {
                    self.remove(identity)?;
                    boundary("group_member_retired");
                    Ok(())
                })?;
            } else if prepared.kind == Kind::Material {
                continue;
            }
            self.remove(&prepared.identity)?;
        }
        self.index_dirty.set(true);
        self.ensure_indexes()
    }

    /// Exact identities come from the confirmed tombstone transaction's
    /// existing outbox. Target references of authored records are not causal
    /// parents and must never be selected merely because their target died.
    pub fn discard_confirmed(
        &self,
        identities: &[String],
        tombstone_position: i64,
        applied: i64,
    ) -> Result<()> {
        if tombstone_position < 0 || tombstone_position > applied {
            return invalid(
                "tombstone cleanup is not authorized by the applied confirmed position",
            );
        }
        self.ensure_indexes()?;
        let identities = canonical(identities)?;
        for identity in &identities {
            if let Some(group) = self.capture_group(identity)? {
                if identities.binary_search(&group.identity.as_str()).is_err() {
                    return invalid("tombstone cleanup omits the canonical capture group");
                }
            }
            if let Some(prepared) = self.get(identity)? {
                if prepared.group.is_some() {
                    self.visit_captures(identity, |member| {
                        if identities.binary_search(&member).is_err() {
                            return invalid("tombstone cleanup omits a capture group member");
                        }
                        Ok(())
                    })?;
                }
            }
        }
        let authorized: std::collections::BTreeSet<_> = identities
            .iter()
            .map(|identity| hash(identity.as_bytes()))
            .collect();
        let mut staged = Vec::new();
        for entry in fs::read_dir(self.root.join("operations"))? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(suffix) = name
                .to_str()
                .and_then(|name| name.strip_prefix(".preparing-"))
            else {
                continue;
            };
            if suffix.len() > 65
                && suffix.as_bytes().get(64) == Some(&b'-')
                && suffix.get(..64).is_some_and(|id| authorized.contains(id))
            {
                if !entry.file_type()?.is_dir() {
                    return invalid("prepared payload staging is not a directory");
                }
                staged.push(entry.path());
            }
        }
        // Keep the authoritative group until all of its authorized raw and
        // request copies are gone. The owner's outbox retries after a crash.
        for groups in [false, true] {
            for identity in &identities {
                if let Some(prepared) = self.get(identity)? {
                    if prepared.group.is_some() == groups {
                        self.remove(identity)?;
                        boundary("discard_member_retired");
                    }
                }
            }
        }
        for path in staged {
            self.retire_path(&path)?;
        }
        self.index_dirty.set(true);
        self.ensure_indexes()
    }

    pub fn counts(&self) -> Result<(u64, u64, u64)> {
        let (mut materials, mut writes, mut bytes) = (0, 0, 0u64);
        let mut after = None;
        loop {
            let page = self.pending_after(after, 128)?;
            if page.is_empty() {
                break;
            }
            after = page.last().map(|entry| entry.order);
            for prepared in page {
                match prepared.kind {
                    Kind::Material => materials += 1,
                    Kind::Events | Kind::ReplaySequence => writes += 1,
                    _ => {}
                }
                for file in fs::read_dir(self.operation(&prepared.identity))? {
                    let file = file?;
                    if file.file_type()?.is_file() {
                        bytes = bytes.saturating_add(file.metadata()?.len());
                    }
                }
            }
        }
        Ok((materials, writes, bytes))
    }

    pub fn visit_captures(
        &self,
        identity: &str,
        mut visit: impl FnMut(&str) -> Result<()>,
    ) -> Result<()> {
        let prepared = self.required(identity)?;
        let group = prepared
            .group
            .ok_or_else(|| Error::Invalid("operation is not a capture group".into()))?;
        let path = self.operation(identity).join("captures.jsonl");
        let mut file = verified(
            &path,
            &group.captures_hash,
            fs::symlink_metadata(&path)?.len(),
        )?;
        let mut count = 0u64;
        for line in BufReader::new(&mut file).lines() {
            let _: String = serde_json::from_str(&line?)?;
            count = count
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("capture count overflow".into()))?;
        }
        if count != group.captures_count {
            return invalid("capture group differs from its immutable member set");
        }
        file.rewind()?;
        for line in BufReader::new(file).lines() {
            let capture: String = serde_json::from_str(&line?)?;
            visit(&capture)?;
        }
        Ok(())
    }

    fn required(&self, identity: &str) -> Result<Prepared> {
        self.get(identity)?
            .ok_or_else(|| Error::Invalid(format!("operation is not prepared: {identity}")))
    }

    fn prepare_operation<R: Read>(
        &self,
        frame: Ingress,
        mut body: Option<R>,
        captures: Option<(&[&str], String)>,
        full_capture: Option<&Capture>,
    ) -> Result<Prepared> {
        self.ensure_indexes()?;
        let identity = frame.frame().to_owned();
        if let Some(previous) = self.get(&identity)? {
            let desired = encoded_hash(&frame)?;
            if (previous.frame_hash.as_str(), previous.frame_length)
                != (desired.0.as_str(), desired.1)
            {
                return invalid("prepared identity changed immutable frame contents");
            }
            if previous.body.is_some() != body.is_some() {
                return invalid("prepared identity changed its body contract");
            }
            if previous
                .group
                .as_ref()
                .map(|group| group.captures_hash.as_str())
                != captures.as_ref().map(|(_, digest)| digest.as_str())
            {
                return invalid("prepared identity changed its capture group");
            }
            if let Some(input) = body.as_mut() {
                let mut output = HashWriter::new(std::io::sink());
                std::io::copy(input, &mut output)?;
                let (blob, length) = output.finish();
                if previous.body.as_ref() != Some(&Bytes { blob, length }) {
                    return invalid("prepared identity changed immutable body contents");
                }
            }
            let desired_capture = full_capture.map(encoded_hash).transpose()?;
            if previous
                .captured
                .as_ref()
                .map(|bytes| (bytes.blob.as_str(), bytes.length))
                != desired_capture
                    .as_ref()
                    .map(|(blob, length)| (blob.as_str(), *length))
            {
                return invalid("prepared identity changed immutable captured facts");
            }
            return Ok(previous);
        }
        let prefix = format!(".preparing-{}-", hash(identity.as_bytes()));
        let temporary = tempfile::Builder::new()
            .prefix(&prefix)
            .tempdir_in(self.root.join("operations"))?;
        let (frame_hash, frame_length) = write_json(&temporary.path().join("frame.json"), &frame)?;
        let (kind, material, mut capture, request) = describe(&frame)?;
        let captured = full_capture
            .map(|full| -> Result<Bytes> {
                let header = capture_header(full);
                if capture.as_ref() != Some(&header) {
                    return invalid("capture header differs from its material manifest");
                }
                capture = Some(header);
                let (blob, length) = write_json(&temporary.path().join("captured.json"), full)?;
                boundary("capture_synced");
                Ok(Bytes { blob, length })
            })
            .transpose()?;
        let body = if let Some(mut input) = body {
            let descriptor = material
                .as_ref()
                .ok_or_else(|| Error::Invalid("body has no material descriptor".into()))?;
            let path = temporary.path().join("body");
            let mut output = HashWriter::new(
                OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&path)?,
            );
            std::io::copy(&mut input, &mut output)?;
            output.inner.sync_all()?;
            let (blob, length) = output.finish();
            if blob != descriptor.blob || length != descriptor.length {
                return invalid("captured bytes differ from their prepared material");
            }
            boundary("body_synced");
            Some(Bytes { blob, length })
        } else {
            None
        };
        let group = captures
            .map(|(captures, captures_hash)| -> Result<Group> {
                let mut file = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(temporary.path().join("captures.jsonl"))?;
                for capture in captures {
                    serde_json::to_writer(&mut file, capture)?;
                    file.write_all(b"\n")?;
                }
                file.sync_all()?;
                Ok(Group {
                    captures_hash,
                    captures_count: captures.len() as u64,
                })
            })
            .transpose()?;
        let order = self.next.get();
        let prepared = Prepared {
            order,
            identity: identity.clone(),
            kind,
            frame_hash,
            frame_length,
            body,
            material,
            capture,
            captured,
            request,
            group,
            acknowledged: false,
            commit_seq: None,
            refused: false,
        };
        write_checked(&temporary.path().join("metadata.json"), &prepared)?;
        sync(temporary.path())?;
        boundary("operation_synced");
        if prepared.group.is_some() {
            boundary("group_synced");
        }
        self.index_dirty.set(true);
        fs::rename(temporary.path(), self.operation(&identity))?;
        sync(&self.root.join("operations"))?;
        boundary("operation_published");
        if prepared.group.is_some() {
            boundary("group_published");
        }
        self.next.set(
            order
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("producer queue order exhausted".into()))?,
        );
        self.index_entry(&prepared)?;
        self.index_dirty.set(false);
        Ok(prepared)
    }

    fn remove(&self, identity: &str) -> Result<()> {
        self.retire_path(&self.operation(identity))
    }

    fn retire_path(&self, directory: &Path) -> Result<()> {
        if !directory.try_exists()? {
            return Ok(());
        }
        let retired = self
            .root
            .join("retired")
            .join(directory.file_name().expect("owned spool directory"));
        self.index_dirty.set(true);
        if retired.try_exists()? {
            fs::remove_dir_all(&retired)?;
            sync(&self.root.join("retired"))?;
        }
        fs::rename(directory, &retired)?;
        sync(&self.root.join("operations"))?;
        sync(&self.root.join("retired"))?;
        boundary("operation_retired");
        fs::remove_dir_all(retired)?;
        Ok(())
    }

    fn ensure_indexes(&self) -> Result<()> {
        if self.index_dirty.get() {
            self.rebuild_indexes()?;
        }
        Ok(())
    }

    fn rebuild_indexes(&self) -> Result<()> {
        let index = self.root.join("index");
        if index.try_exists()? {
            fs::remove_dir_all(&index)?;
        }
        for name in ["order", "groups", "members"] {
            fs::create_dir_all(index.join(name))?;
        }
        let (mut first, mut last) = (u64::MAX, 0);
        for entry in fs::read_dir(self.root.join("operations"))? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str().filter(|name| {
                name.len() == 64 && name.bytes().all(|byte| byte.is_ascii_hexdigit())
            }) else {
                continue;
            };
            if !entry.file_type()?.is_dir() {
                return invalid("spool operation is not a directory");
            }
            let metadata = read_checked::<Prepared>(&entry.path().join("metadata.json"))?
                .ok_or_else(|| Error::Invalid("published operation has no metadata".into()))?;
            if hash(metadata.identity.as_bytes()) != name {
                return invalid("spool directory has another operation identity");
            }
            let prepared = self.required(&metadata.identity)?;
            first = first.min(prepared.order);
            last = last.max(prepared.order);
            self.index_entry(&prepared)?;
        }
        self.first.set(if first == u64::MAX { 1 } else { first });
        self.next.set(
            last.checked_add(1)
                .ok_or_else(|| Error::Invalid("producer queue order exhausted".into()))?,
        );
        self.index_dirty.set(false);
        sync(&index)?;
        Ok(())
    }

    fn index_entry(&self, prepared: &Prepared) -> Result<()> {
        let order = self.index("order", &format!("{:020}", prepared.order));
        if read_json::<String>(&order)?.is_some_and(|identity| identity != prepared.identity) {
            return invalid("two pending operations have the same queue order");
        }
        write_index(&order, &prepared.identity)?;
        if let Some(group) = &prepared.group {
            let index = self.index("groups", &group.captures_hash);
            if read_json::<String>(&index)?.is_some_and(|identity| identity != prepared.identity) {
                return invalid("two canonical frames claim the same capture group");
            }
            write_index(&index, &prepared.identity)?;
            self.visit_captures(&prepared.identity, |capture| {
                let index = self.index("members", &hash(capture.as_bytes()));
                if read_json::<String>(&index)?
                    .is_some_and(|identity| identity != prepared.identity)
                {
                    return invalid("capture belongs to two canonical frames");
                }
                write_index(&index, &prepared.identity)?;
                boundary("group_indexed_member");
                Ok(())
            })?;
        }
        Ok(())
    }
}

fn capture_header(capture: &Capture) -> Capture {
    match capture {
        Capture::Input(input) => Capture::Input(polylogue_store::primary::CapturedInput {
            input: input.input.clone(),
            rows: input.rows.clone(),
            index: input.index,
            previous: input.previous.clone(),
            hash: input.hash,
            length: input.length,
            group: input.group.clone(),
            arrival_rank: input.arrival_rank,
            parsed: Vec::new(),
        }),
        Capture::Blob { hash, length } => Capture::Blob {
            hash: *hash,
            length: *length,
        },
    }
}

fn describe(frame: &Ingress) -> Result<(Kind, Option<Material>, Option<Capture>, bool)> {
    if let Ingress::Material { material, .. } = frame {
        if let Some(value) = material.manifest.details.get("polylogue_capture") {
            let capture: Capture = serde_json::from_value(value.clone())?;
            if capture != capture_header(&capture) {
                return invalid(
                    "raw material manifest must keep parser facts in the private capture sidecar",
                );
            }
        }
    }
    Ok(match frame {
        Ingress::Material { material, .. } => (
            Kind::Material,
            Some(Material {
                id: material.id.to_string(),
                blob: material.blob.clone(),
                length: material.length,
            }),
            material
                .manifest
                .details
                .get("polylogue_capture")
                .map(|capture| serde_json::from_value(capture.clone()))
                .transpose()?,
            material.parser == "polylogue-request",
        ),
        Ingress::Events { .. } => (Kind::Events, None, None, false),
        Ingress::ReplaySequence { .. } => (Kind::ReplaySequence, None, None, false),
        Ingress::RegisterKinds { .. } => (Kind::RegisterKinds, None, None, false),
        Ingress::Chunk { .. } => (Kind::Chunk, None, None, false),
        Ingress::LargeFrame { .. } => (Kind::LargeFrame, None, None, false),
    })
}

fn validate_ack(prepared: &Prepared, acknowledgment: &Acknowledgment) -> Result<()> {
    validate_ack_header(
        prepared,
        &acknowledgment.frame,
        acknowledgment
            .material
            .as_ref()
            .map(ToString::to_string)
            .as_deref(),
        acknowledgment.commit_seq,
        acknowledgment.refusal.is_some(),
    )
}
fn validate_ack_header(
    prepared: &Prepared,
    frame: &str,
    material: Option<&str>,
    position: Option<i64>,
    refused: bool,
) -> Result<()> {
    if frame != prepared.identity {
        return invalid("acknowledgment names another prepared operation");
    }
    if refused && position.is_some() {
        return invalid("refused frame claims a confirmed position");
    }
    if position.is_some_and(|position| position < 0) {
        return invalid("acknowledgment has a negative confirmed position");
    }
    if !refused {
        if matches!(prepared.kind, Kind::Events | Kind::ReplaySequence) && position.is_none() {
            return invalid("canonical frame acknowledgment has no confirmed position");
        }
        if let Some(expected) = &prepared.material {
            if material != Some(expected.id.as_str()) {
                return invalid("material acknowledgment has another material identity");
            }
        }
    }
    Ok(())
}

fn canonical(captures: &[String]) -> Result<Vec<&str>> {
    let mut sorted: Vec<_> = captures.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return invalid("canonical capture group repeats a member");
    }
    Ok(sorted)
}
fn captures_hash(captures: &[&str]) -> Result<String> {
    let mut writer = HashWriter::new(std::io::sink());
    for capture in captures {
        serde_json::to_writer(&mut writer, capture)?;
        writer.write_all(b"\n")?;
    }
    Ok(writer.finish().0)
}
fn hash(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}
fn hash_name(digest: blake3::Hash) -> String {
    format!("blake3:{digest}")
}
fn invalid<T>(reason: &str) -> Result<T> {
    Err(Error::Invalid(reason.into()))
}
fn sync(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}
fn regular(path: &Path) -> Result<File> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return invalid("spool payload is not a regular file");
    }
    Ok(File::open(path)?)
}
fn check_length(path: &Path, length: u64) -> Result<()> {
    if regular(path)?.metadata()?.len() != length {
        return invalid("spool payload length differs from its metadata");
    }
    Ok(())
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    match regular(path) {
        Ok(file) => Ok(Some(decode_json(file)?)),
        Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}
fn read_checked<T: serde::de::DeserializeOwned + Serialize>(path: &Path) -> Result<Option<T>> {
    let Some(checked) = read_json::<Checked<T>>(path)? else {
        return Ok(None);
    };
    if encoded_hash(&checked.value)?.0 != checked.digest {
        return invalid("spool metadata checksum differs from its contents");
    }
    Ok(Some(checked.value))
}
fn atomic_checked(path: &Path, value: &impl Serialize) -> Result<()> {
    atomic_json(
        path,
        &Checked {
            value,
            digest: encoded_hash(value)?.0,
        },
    )
}
fn write_checked(path: &Path, value: &impl Serialize) -> Result<()> {
    write_json(
        path,
        &Checked {
            value,
            digest: encoded_hash(value)?.0,
        },
    )?;
    Ok(())
}

fn write_index(path: &Path, identity: &str) -> Result<()> {
    // An interrupted derived index is rebuilt before it is read again; only
    // the immutable operation publication needs a durability barrier.
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    serde_json::to_writer(file, identity)?;
    Ok(())
}

fn decode_json<T: serde::de::DeserializeOwned>(reader: impl Read) -> Result<T> {
    let mut decoder = serde_json::Deserializer::from_reader(reader);
    decoder.disable_recursion_limit();
    let value = T::deserialize(&mut decoder)?;
    decoder.end()?;
    Ok(value)
}

fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().expect("spool path parent");
    let temporary = tempfile::NamedTempFile::new_in(parent)?;
    write_json_file(temporary.as_file(), value)?;
    temporary
        .persist(path)
        .map_err(|error| Error::Io(error.error))?;
    sync(parent)
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<(String, u64)> {
    let file = OpenOptions::new().create_new(true).write(true).open(path)?;
    write_json_file(&file, value)
}
fn write_json_file(file: &File, value: &impl Serialize) -> Result<(String, u64)> {
    let mut writer = HashWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    writer.inner.sync_all()?;
    Ok(writer.finish())
}
fn encoded_hash(value: &impl Serialize) -> Result<(String, u64)> {
    let mut writer = HashWriter::new(std::io::sink());
    serde_json::to_writer(&mut writer, value)?;
    Ok(writer.finish())
}
fn file_hash(path: &Path) -> Result<(String, u64)> {
    let mut input = regular(path)?;
    let mut writer = HashWriter::new(std::io::sink());
    std::io::copy(&mut input, &mut writer)?;
    Ok(writer.finish())
}
fn verified(path: &Path, blob: &str, length: u64) -> Result<File> {
    let mut file = regular(path)?;
    let mut writer = HashWriter::new(std::io::sink());
    std::io::copy(&mut file, &mut writer)?;
    if writer.finish() != (blob.to_owned(), length) {
        return invalid("spool payload differs from its immutable digest");
    }
    file.rewind()?;
    Ok(file)
}
struct HashWriter<W> {
    inner: W,
    digest: blake3::Hasher,
    length: u64,
}
impl<W> HashWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            digest: blake3::Hasher::new(),
            length: 0,
        }
    }
    fn finish(self) -> (String, u64) {
        (hash_name(self.digest.finalize()), self.length)
    }
}
impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.digest.update(&bytes[..written]);
        self.length = self
            .length
            .checked_add(written as u64)
            .ok_or_else(|| std::io::Error::other("spool byte length overflow"))?;
        Ok(written)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
fn boundary(name: &str) {
    #[cfg(test)]
    tests::boundary(name);
    #[cfg(not(test))]
    let _ = name;
}

#[cfg(test)]
mod tests;

//! Pending producer operations. Immutable frame and body files become visible
//! together; small queue metadata never contains event or request payloads.
use crate::{Error, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use hashed_blobs::{BlobReader, EncodedBlob, Preparation, ReadLimits};
use polylogue_store::primary::Capture;
use serde::{Deserialize, Serialize};
use sinex_protocol::{Acknowledgment, Ingress};
use std::cell::Cell;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
mod exports;
mod names;
mod raw_ranges;
mod sources;

/// A durable body or sidecar file: its decoded identity and the encoding that
/// reads it back.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Bytes {
    pub blob: String,
    pub length: u64,
    encoding: Encoding,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
struct Encoding(String);
impl Encoding {
    fn new(descriptor: &EncodedBlob) -> Self {
        Self(BASE64.encode(descriptor.encode()))
    }
    fn descriptor(&self) -> Result<EncodedBlob> {
        let bytes = BASE64
            .decode(&self.0)
            .map_err(|error| Error::Invalid(format!("invalid encoded descriptor: {error}")))?;
        Ok(EncodedBlob::decode(&bytes)?)
    }
}
impl Bytes {
    fn new(descriptor: &EncodedBlob) -> Self {
        Self {
            blob: descriptor.hash.to_string(),
            length: descriptor.decoded_len,
            encoding: Encoding::new(descriptor),
        }
    }
    fn descriptor(&self) -> Result<EncodedBlob> {
        let descriptor = self.encoding.descriptor()?;
        if descriptor.hash.to_string() != self.blob || descriptor.decoded_len != self.length {
            return invalid("encoded descriptor differs from its decoded identity");
        }
        Ok(descriptor)
    }
    fn identity(&self) -> (&str, u64) {
        (&self.blob, self.length)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Group {
    pub captures_hash: String,
    pub captures_count: u64,
    encoding: Encoding,
}

/// A captured material and the facts the queue keeps about its custody.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Material {
    pub id: String,
    pub blob: String,
    pub length: u64,
    pub body: Bytes,
    pub capture: Option<Capture>,
    pub captured: Option<Bytes>,
    pub control: Option<Control>,
    pub exports: Vec<exports::ExportBinding>,
    /// Its observation is found by logical name.
    pub named: bool,
    /// Carries an authored request.
    pub request: bool,
}

/// What an operation is. Each variant owns exactly the facts that variant has.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "fields", rename_all = "snake_case")]
pub(crate) enum Payload {
    Material(Box<Material>),
    Events { group: Option<Group> },
    ReplaySequence { group: Option<Group> },
    Signals,
    RegisterKinds,
    Chunk,
    LargeFrame,
}

/// Where an operation stands with its receipt. The only legal transition is
/// from pending to one settled phase; a settled phase never changes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub(crate) enum Phase {
    /// No receipt yet.
    Pending,
    /// The owner accepted the operation; writes carry the position that
    /// confirmed them.
    Acknowledged { commit_seq: Option<i64> },
    /// The owner refused the operation.
    Refused,
    /// The owner already holds different contents under this identity; the
    /// offered frame hash is kept as evidence.
    Conflicted { offered: String },
}

impl Phase {
    /// The phase an acknowledgment from the owner settles.
    fn settled_by(acknowledgment: &Acknowledgment) -> Self {
        if acknowledgment.refusal.is_some() {
            Self::Refused
        } else {
            Self::Acknowledged {
                commit_seq: acknowledgment.commit_seq,
            }
        }
    }

    /// The one place that decides which phase changes are legal.
    pub(crate) fn advance(&self, next: Self) -> Result<Self> {
        match (self, &next) {
            (Self::Pending, Self::Pending) => invalid("pending is not a settled outcome"),
            (Self::Pending, _) => Ok(next),
            _ => invalid("a settled operation cannot change its outcome"),
        }
    }

    pub(crate) fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }
    pub(crate) fn is_acknowledged(&self) -> bool {
        matches!(self, Self::Acknowledged { .. })
    }
    /// Refused by the owner, or conflicting with what the owner holds.
    pub(crate) fn is_refused(&self) -> bool {
        matches!(self, Self::Refused | Self::Conflicted { .. })
    }
    pub(crate) fn commit_seq(&self) -> Option<i64> {
        match self {
            Self::Acknowledged { commit_seq } => *commit_seq,
            _ => None,
        }
    }
    fn confirmed_through(&self, applied: i64) -> bool {
        self.commit_seq()
            .is_some_and(|position| position <= applied)
    }
}

/// The immutable queue record, written once as `metadata.json`. Receipts,
/// revocation and acquisition standing live in their own files.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Record {
    /// Relative to its acquisition for an acquisition member until loaded.
    pub order: u64,
    pub identity: String,
    pub frame: Bytes,
    pub payload: Payload,
}

impl Record {
    pub(crate) fn material(&self) -> Option<&Material> {
        match &self.payload {
            Payload::Material(material) => Some(material.as_ref()),
            _ => None,
        }
    }
    pub(crate) fn body(&self) -> Option<&Bytes> {
        self.material().map(|material| &material.body)
    }
    pub(crate) fn capture(&self) -> Option<&Capture> {
        self.material()?.capture.as_ref()
    }
    pub(crate) fn captured(&self) -> Option<&Bytes> {
        self.material()?.captured.as_ref()
    }
    pub(crate) fn control(&self) -> Option<&Control> {
        self.material()?.control.as_ref()
    }
    pub(crate) fn exports(&self) -> &[exports::ExportBinding] {
        self.material()
            .map_or(&[], |material| material.exports.as_slice())
    }
    pub(crate) fn group(&self) -> Option<&Group> {
        match &self.payload {
            Payload::Events { group, .. } | Payload::ReplaySequence { group } => group.as_ref(),
            _ => None,
        }
    }
    /// A write the owner confirms with a queue position.
    pub(crate) fn is_write(&self) -> bool {
        matches!(
            self.payload,
            Payload::Events { .. } | Payload::ReplaySequence { .. }
        )
    }
    pub(crate) fn is_request(&self) -> bool {
        self.material().is_some_and(|material| material.request)
    }
    pub(crate) fn is_named(&self) -> bool {
        match &self.payload {
            Payload::Events { .. } => true,
            Payload::Material(material) => material.named,
            _ => false,
        }
    }
    pub(crate) fn logical_name(&self) -> Option<&str> {
        self.is_named().then(|| {
            if self.material().is_some() {
                self.identity
                    .strip_suffix(":observation")
                    .unwrap_or(&self.identity)
            } else {
                &self.identity
            }
        })
    }
}

/// An operation as loaded: its immutable record plus the standing that other
/// files decide. Never serialized.
#[derive(Clone, Debug)]
pub(crate) struct Prepared {
    pub record: Record,
    pub phase: Phase,
    /// Its acquisition was accepted with an unresolved basis conflict.
    pub unresolved: bool,
}
impl std::ops::Deref for Prepared {
    type Target = Record;
    fn deref(&self) -> &Record {
        &self.record
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Control {
    pub reading_version: String,
    pub parsed_upto: u64,
}

#[derive(Serialize, Deserialize)]
struct Acquisition {
    identity: String,
    intent: String,
    count: u64,
    order_base: Option<u64>,
    disposition: polylogue_store::primary::CaptureDisposition,
}

#[derive(Serialize, Deserialize)]
struct Checked<T> {
    value: T,
    digest: String,
}

/// The settled outcome and the acknowledgment bytes that proved it.
#[derive(Serialize, Deserialize)]
struct Receipt {
    payload: Bytes,
    phase: Phase,
}

/// Only pending operations live here. Ordered and capture-membership indexes
/// are rebuildable from operation directories, never source progress.
pub(crate) struct Spool {
    root: PathBuf,
    first: Cell<u64>,
    next: Cell<u64>,
    index_dirty: Cell<bool>,
    last_retired: Cell<Option<i64>>,
    retirement_dirty: Cell<bool>,
    _lock: File,
}

/// Encoded custody is verified and synced privately before publication takes
/// the shared queue lock. Relative member order grants no source authority.
pub(crate) struct StagedAcquisition {
    temporary: tempfile::TempDir,
    queue: Spool,
    identity: String,
    intent: String,
    count: u64,
}
impl StagedAcquisition {
    pub fn queue(&self) -> &Spool {
        &self.queue
    }
    pub fn finish(&self) -> Result<()> {
        if self.queue.next.get().checked_sub(1) != Some(self.count) {
            return invalid("acquisition member count differs from its complete plan");
        }
        sync(self.temporary.path())?;
        Ok(())
    }
}

/// A pinned immutable sidecar may be checked after releasing the queue lock.
pub(crate) struct PinnedCapture {
    header: Capture,
    control: Option<Control>,
    exports: Vec<exports::ExportBinding>,
    bytes: Option<(File, EncodedBlob)>,
}
impl PinnedCapture {
    pub fn decode(self) -> Result<Capture> {
        let Some((file, descriptor)) = self.bytes else {
            if !self.exports.is_empty() {
                return invalid("export custody has no captured body");
            }
            return Ok(self.header);
        };
        let prepared =
            hashed_blobs::Prepared::from_encoded(file, descriptor, None, ReadLimits::default())?;
        let capture = decode_capture(prepared.reader(None, ReadLimits::default())?)?;
        if capture_header(&capture) != self.header {
            return invalid("capture sidecar differs from its queue header");
        }
        if exports::capture_bindings(&capture)? != self.exports {
            return invalid("export sidecar differs from its immutable reading-key index");
        }
        if control(&capture) != self.control {
            return invalid("parser control differs from its immutable capture index");
        }
        Ok(capture)
    }
}

fn check_backup_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Acquire) {
        return invalid("producer backup acquisition cancelled");
    }
    Ok(())
}

fn pin_backup_tree(source: &Path, destination: &Path, cancelled: &AtomicBool) -> Result<()> {
    check_backup_cancelled(cancelled)?;
    let metadata = fs::symlink_metadata(source)?;
    if metadata.is_file() {
        fs::hard_link(source, destination)?;
    } else if metadata.is_dir() {
        fs::create_dir(destination)?;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            // Atomic publication leaves preparing directories invisible to a
            // reopened spool; preserve that same cut in its backup.
            if entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with('.'))
            {
                continue;
            }
            pin_backup_tree(
                &entry.path(),
                &destination.join(entry.file_name()),
                cancelled,
            )?;
        }
        sync(destination)?;
    } else {
        return invalid("producer backup contains a non-regular member");
    }
    Ok(())
}

impl Spool {
    /// Pin the complete accepted queue while its mutex is held. Payloads and
    /// receipts are immutable files, so links retain their original encoded
    /// bytes after ordinary retirement unlinks the live operations. Acquisitions
    /// published after this cut are absent; partially staged acquisitions are
    /// never included. The caller hashes/copies the pinned files outside the
    /// queue mutex, allowing acquisition to continue.
    pub(crate) fn pin_backup(&self, cancelled: &AtomicBool) -> Result<tempfile::TempDir> {
        check_backup_cancelled(cancelled)?;
        let pinned = tempfile::Builder::new()
            .prefix(".backup-cut-")
            .tempdir_in(self.root.parent().expect("producer spool parent"))?;
        let target = pinned
            .path()
            .join("producer-spool")
            .join(self.root.file_name().expect("producer spool identity"));
        fs::create_dir_all(target.join("operations"))?;
        for entry in fs::read_dir(self.root.join("operations"))? {
            let entry = entry?;
            if !entry.file_name().to_str().is_some_and(hex_name) {
                continue;
            }
            pin_backup_tree(
                &entry.path(),
                &target.join("operations").join(entry.file_name()),
                cancelled,
            )?;
        }
        sync(&target.join("operations"))?;
        sync(&target)?;
        sync(target.parent().expect("snapshot spool parent"))?;
        sync(pinned.path())?;
        Ok(pinned)
    }

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
            last_retired: Cell::new(None),
            retirement_dirty: Cell::new(true),
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

    fn operation(&self, identity: &str) -> Result<PathBuf> {
        let name = hash(identity.as_bytes());
        let direct = self.root.join("operations").join(&name);
        if direct.try_exists()? {
            return Ok(direct);
        }
        let Some(batch) = read_json::<String>(&self.index("locations", &name))? else {
            return Ok(direct);
        };
        if !hex_name(&batch) {
            return invalid("invalid acquisition location index");
        }
        Ok(self
            .root
            .join("operations")
            .join(batch)
            .join("operations")
            .join(name))
    }
    fn index(&self, kind: &str, key: &str) -> PathBuf {
        self.root.join("index").join(kind).join(key)
    }

    pub fn get(&self, identity: &str) -> Result<Option<Prepared>> {
        let Some(mut record) =
            read_checked::<Record>(&self.operation(identity)?.join("metadata.json"))?
        else {
            return Ok(None);
        };
        let mut unresolved = false;
        if record.identity != identity {
            return invalid("spool operation identity differs from its directory");
        }
        let directory = self.operation(identity)?;
        let operations = directory.parent().expect("operation parent");
        if operations != self.root.join("operations") {
            let acquisition: Acquisition = read_checked(
                &operations
                    .parent()
                    .expect("acquisition parent")
                    .join("acquisition.json"),
            )?
            .ok_or_else(|| {
                Error::Invalid("acquisition member has no accepted publication".into())
            })?;
            let base = acquisition
                .order_base
                .ok_or_else(|| Error::Invalid("acquisition has no accepted queue order".into()))?;
            if record.order == 0 || record.order > acquisition.count {
                return invalid("acquisition member has an invalid relative order");
            }
            unresolved = acquisition.disposition
                == polylogue_store::primary::CaptureDisposition::UnresolvedBasisConflict;
            record.order = base
                .checked_add(record.order)
                .ok_or_else(|| Error::Invalid("acquisition queue order exhausted".into()))?;
        }
        check_length(
            &directory.join("frame.zst"),
            record.frame.descriptor()?.stored_len,
        )?;
        if let Some(body) = record.body() {
            check_length(&directory.join("body.zst"), body.descriptor()?.stored_len)?;
        }
        if let Some(captured) = record.captured() {
            check_length(
                &directory.join("captured.zst"),
                captured.descriptor()?.stored_len,
            )?;
        }
        let phase = self
            .receipt(identity)?
            .map_or(Phase::Pending, |receipt| receipt.phase);
        Ok(Some(Prepared {
            record,
            phase,
            unresolved,
        }))
    }

    /// Decode the original immutable common encoding without collecting the frame.
    pub fn frame(&self, identity: &str) -> Result<BlobReader> {
        Ok(self
            .prepared_frame(identity)?
            .reader(None, ReadLimits::default())?)
    }

    pub fn prepared_frame(&self, identity: &str) -> Result<hashed_blobs::Prepared> {
        let prepared = self.required(identity)?;
        Ok(hashed_blobs::Prepared::from_encoded(
            regular(&self.operation(identity)?.join("frame.zst"))?,
            prepared.frame.descriptor()?,
            None,
            ReadLimits::default(),
        )?)
    }

    pub fn decode_frame(&self, identity: &str) -> Result<Ingress> {
        decode_json(self.frame(identity)?)
    }

    pub fn body(&self, identity: &str) -> Result<BlobReader> {
        self.prepared_body(identity)?
            .reader(None, ReadLimits::default())
            .map_err(Into::into)
    }

    /// The caller keeps the named spool inode immutable until the applied barrier.
    pub fn encoded_body(&self, identity: &str) -> Result<(PathBuf, EncodedBlob)> {
        let prepared = self.required(identity)?;
        let body = prepared
            .body()
            .ok_or_else(|| Error::Invalid("operation has no material body".into()))?;
        let descriptor = body.descriptor()?;
        let path = self.operation(identity)?.join("body.zst");
        check_length(&path, descriptor.stored_len)?;
        Ok((path, descriptor))
    }

    /// A pinned handle survives publication retries without a decoded copy.
    pub fn prepared_body(&self, identity: &str) -> Result<hashed_blobs::Prepared> {
        let (path, descriptor) = self.encoded_body(identity)?;
        Ok(hashed_blobs::Prepared::from_encoded(
            regular(&path)?,
            descriptor,
            None,
            ReadLimits::default(),
        )?)
    }

    fn receipt(&self, identity: &str) -> Result<Option<Receipt>> {
        let directory = self.operation(identity)?.join("ack");
        if directory.try_exists()? && !fs::symlink_metadata(&directory)?.file_type().is_dir() {
            return invalid("published acknowledgment is not a directory");
        }
        let receipt = read_checked::<Receipt>(&directory.join("receipt.json"))?;
        if let Some(receipt) = &receipt {
            if receipt.phase.is_pending() {
                return invalid("published acknowledgment has no settled outcome");
            }
            check_length(
                &directory.join("payload.zst"),
                receipt.payload.descriptor()?.stored_len,
            )?;
        } else if directory.try_exists()? {
            return invalid("published acknowledgment has no receipt");
        }
        Ok(receipt)
    }

    pub fn acknowledgment(&self, identity: &str) -> Result<Option<Acknowledgment>> {
        let Some(receipt) = self.receipt(identity)? else {
            return Ok(None);
        };
        if matches!(receipt.phase, Phase::Conflicted { .. }) {
            self.conflict_receipt(identity)?;
            return Ok(None);
        }
        let acknowledgment: Acknowledgment = decode_json(verified_encoded(
            &self.operation(identity)?.join("ack/payload.zst"),
            receipt.payload.descriptor()?,
        )?)?;
        let prepared = self.required(identity)?;
        validate_ack(&prepared, &acknowledgment)?;
        if receipt.phase != Phase::settled_by(&acknowledgment) {
            return invalid("acknowledgment summary differs from its immutable receipt");
        }
        Ok(Some(acknowledgment))
    }

    pub fn prepare(&self, frame: Ingress) -> Result<Prepared> {
        if let Some(prepared) = self.get(frame.frame())? {
            let desired = encoded_hash(&frame)?;
            if desired != (prepared.frame.blob.clone(), prepared.frame.length) {
                return invalid("prepared identity changed immutable frame contents");
            }
            return Ok(prepared);
        }
        if matches!(&frame, Ingress::Material { .. }) {
            return invalid("material must have its immutable body prepared before sending");
        }
        self.prepare_operation(frame, None::<std::io::Empty>, None, None, None)
    }

    pub fn prepare_material(&self, frame: Ingress, bytes: impl Read) -> Result<Prepared> {
        if !matches!(&frame, Ingress::Material { .. }) {
            return invalid("only a material frame has captured bytes");
        }
        self.prepare_operation(frame, Some(bytes), None, None, None)
    }

    pub fn prepare_encoded_material(
        &self,
        frame: Ingress,
        bytes: polylogue_store::primary::PreparedBody,
    ) -> Result<Prepared> {
        if !matches!(&frame, Ingress::Material { .. }) {
            return invalid("only a material frame can own encoded bytes");
        }
        self.prepare_operation(frame, None::<std::io::Empty>, None, None, Some(bytes))
    }

    pub fn prepare_encoded_capture(
        &self,
        frame: Ingress,
        bytes: polylogue_store::primary::PreparedBody,
        capture: &Capture,
    ) -> Result<Prepared> {
        if !matches!(&frame, Ingress::Material { .. }) {
            return invalid("only a material frame can own an encoded capture");
        }
        self.prepare_operation(
            frame,
            None::<std::io::Empty>,
            None,
            Some(capture),
            Some(bytes),
        )
    }

    pub fn stage_acquisition(
        &self,
        identity: &str,
        intent: &str,
        count: u64,
    ) -> Result<StagedAcquisition> {
        let temporary = tempfile::Builder::new()
            .prefix(&format!(".preparing-{}-", hash(identity.as_bytes())))
            .tempdir_in(self.root.join("operations"))?;
        fs::create_dir(temporary.path().join("operations"))?;
        fs::create_dir(temporary.path().join("retired"))?;
        let queue = Spool {
            root: temporary.path().into(),
            first: Cell::new(1),
            next: Cell::new(1),
            index_dirty: Cell::new(true),
            last_retired: Cell::new(None),
            retirement_dirty: Cell::new(true),
            _lock: self._lock.try_clone()?,
        };
        queue.rebuild_indexes()?;
        Ok(StagedAcquisition {
            temporary,
            queue,
            identity: identity.into(),
            intent: intent.into(),
            count,
        })
    }

    pub fn acquisition_disposition(
        &self,
        identity: &str,
    ) -> Result<Option<polylogue_store::primary::CaptureDisposition>> {
        let path = self
            .root
            .join("operations")
            .join(hash(identity.as_bytes()))
            .join("acquisition.json");
        let header = read_checked::<Acquisition>(&path)?;
        if header
            .as_ref()
            .is_some_and(|header| header.identity != identity)
        {
            return invalid("acquisition receipt has another identity");
        }
        Ok(header.map(|header| header.disposition))
    }

    /// Exact frozen retries return their existing disposition without a fresh
    /// source comparison or any mutation of accepted ordering.
    pub fn matching_acquisition(
        &self,
        staged: &StagedAcquisition,
    ) -> Result<Option<polylogue_store::primary::CaptureDisposition>> {
        let Some(disposition) = self.acquisition_disposition(&staged.identity)? else {
            return Ok(None);
        };
        let path = self
            .root
            .join("operations")
            .join(hash(staged.identity.as_bytes()))
            .join("acquisition.json");
        let acquisition =
            read_checked::<Acquisition>(&path)?.expect("read above under custody lock");
        if acquisition.intent != staged.intent || acquisition.count != staged.count {
            return Ok(None);
        }
        for member in fs::read_dir(staged.temporary.path().join("operations"))? {
            let desired = read_checked::<Record>(&member?.path().join("metadata.json"))?
                .ok_or_else(|| {
                    Error::Invalid("staged acquisition member has no metadata".into())
                })?;
            let Some(original) = self.get(&desired.identity)? else {
                return Ok(None);
            };
            if original.frame.identity() != desired.frame.identity()
                || original.body().map(Bytes::identity) != desired.body().map(Bytes::identity)
                || original.captured().map(Bytes::identity)
                    != desired.captured().map(Bytes::identity)
            {
                return Ok(None);
            }
        }
        Ok(Some(disposition))
    }

    /// Only small metadata and the atomic directory publication enter this step.
    pub fn publish_acquisition(
        &self,
        staged: StagedAcquisition,
        disposition: polylogue_store::primary::CaptureDisposition,
    ) -> Result<()> {
        self.ensure_indexes()?;
        let destination = self
            .root
            .join("operations")
            .join(hash(staged.identity.as_bytes()));
        if destination.try_exists()? {
            if self.matching_acquisition(&staged)? != Some(disposition) {
                return invalid("acquisition identity changed its frozen custody or disposition");
            }
            return Ok(());
        }
        if staged.queue.next.get().checked_sub(1) != Some(staged.count) {
            return invalid("acquisition member count differs from its complete plan");
        }
        let base = self
            .next
            .get()
            .checked_sub(1)
            .ok_or_else(|| Error::Invalid("invalid acquisition queue start".into()))?;
        let next = self
            .next
            .get()
            .checked_add(staged.count)
            .ok_or_else(|| Error::Invalid("producer queue order exhausted".into()))?;
        write_checked(
            &staged.temporary.path().join("acquisition.json"),
            &Acquisition {
                identity: staged.identity.clone(),
                intent: staged.intent.clone(),
                count: staged.count,
                order_base: Some(base),
                disposition,
            },
        )?;
        sync(staged.temporary.path())?;
        boundary("acquisition_synced");
        self.index_dirty.set(true);
        fs::rename(staged.temporary.path(), &destination)?;
        sync(&self.root.join("operations"))?;
        boundary("acquisition_published");
        let operations = destination.join("operations");
        let batch = hash(staged.identity.as_bytes());
        for entry in fs::read_dir(&operations)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name
                .to_str()
                .filter(|name| hex_name(name))
                .ok_or_else(|| Error::Invalid("invalid acquisition member".into()))?;
            write_index(&self.index("locations", name), &batch)?;
        }
        for entry in fs::read_dir(operations)? {
            let metadata = read_checked::<Record>(&entry?.path().join("metadata.json"))?
                .ok_or_else(|| {
                    Error::Invalid("published acquisition member has no metadata".into())
                })?;
            self.index_entry(&self.required(&metadata.identity)?)?;
        }
        self.next.set(next);
        self.index_dirty.set(false);
        Ok(())
    }

    #[cfg(test)]
    pub fn prepare_acquisition(
        &self,
        identity: &str,
        intent: &str,
        count: u64,
        stage: impl FnOnce(&Spool) -> Result<()>,
    ) -> Result<()> {
        let staged = self.stage_acquisition(identity, intent, count)?;
        stage(staged.queue())?;
        staged.finish()?;
        self.publish_acquisition(
            staged,
            polylogue_store::primary::CaptureDisposition::Prepared,
        )
    }

    pub fn pin_capture(&self, identity: &str) -> Result<PinnedCapture> {
        let Payload::Material(material) = self.required(identity)?.record.payload else {
            return invalid("operation has no captured facts");
        };
        let header = material
            .capture
            .ok_or_else(|| Error::Invalid("operation has no captured facts".into()))?;
        let bytes = material
            .captured
            .map(|captured| -> Result<_> {
                Ok((
                    regular(&self.operation(identity)?.join("captured.zst"))?,
                    captured.descriptor()?,
                ))
            })
            .transpose()?;
        Ok(PinnedCapture {
            header,
            control: material.control,
            exports: material.exports,
            bytes,
        })
    }

    pub fn capture(&self, identity: &str) -> Result<Capture> {
        self.pin_capture(identity)?.decode()
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
        let captures = sorted_members(captures)?;
        let group_hash = captures_hash(&captures)?;
        if let Some(existing) = self.group_for_sorted(&group_hash)?
            && existing.identity != frame.frame()
        {
            return invalid("capture group already has another immutable frame identity");
        }
        for identity in &captures {
            let raw = self.required(identity)?;
            if raw.material().is_none() {
                return invalid("capture group member is not a durable material");
            }
            if let Some(existing) = self.capture_group(identity)?
                && existing.group().map(|group| &group.captures_hash) != Some(&group_hash)
            {
                return invalid("capture already belongs to another product group");
            }
        }
        self.prepare_operation(
            frame,
            None::<std::io::Empty>,
            Some((&captures, group_hash)),
            None,
            None,
        )
    }

    pub fn group_for(&self, captures: &[String]) -> Result<Option<Prepared>> {
        self.ensure_indexes()?;
        self.group_for_sorted(&captures_hash(&sorted_members(captures)?)?)
    }

    fn group_for_sorted(&self, captures_hash: &str) -> Result<Option<Prepared>> {
        let Some(identity) = read_json::<String>(&self.index("groups", captures_hash))? else {
            return Ok(None);
        };
        let prepared = self.get(&identity)?.ok_or_else(|| {
            Error::Invalid("capture group index names a missing operation".into())
        })?;
        if prepared.group().map(|group| group.captures_hash.as_str()) != Some(captures_hash) {
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

    /// Prefix discovery walks derived names, never parser frame bodies. Exact
    /// names use `get` and do not need this index.
    pub fn visit_names(
        &self,
        prefix: &str,
        visit: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<()> {
        self.ensure_indexes()?;
        names::visit(&self.index("names", ""), prefix, &mut |identity| {
            let prepared = self.required(identity)?;
            let name = prepared.logical_name().ok_or_else(|| {
                Error::Invalid("named index points to an unnamed operation".into())
            })?;
            if name.starts_with(prefix) {
                visit(identity)?;
            }
            Ok(())
        })
    }

    pub fn acquisition_headers(
        &self,
        group: &str,
        after: Option<u64>,
        limit: usize,
    ) -> Result<(
        Option<polylogue_store::primary::CaptureDisposition>,
        Vec<Prepared>,
        Option<u64>,
    )> {
        if limit == 0 {
            return invalid("acquisition page limit is zero");
        }
        self.ensure_indexes()?;
        let directory = self.root.join("operations").join(hash(group.as_bytes()));
        let Some(acquisition) = read_checked::<Acquisition>(&directory.join("acquisition.json"))?
        else {
            return Ok((None, Vec::new(), None));
        };
        if acquisition.identity != group || acquisition.order_base.is_none() {
            return invalid("acquisition page has another or unpublished group identity");
        }
        let mut ordinal = match after {
            Some(after) => after
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("acquisition cursor overflow".into()))?,
            None => 0,
        };
        if ordinal > acquisition.count {
            return invalid("acquisition cursor exceeds its membership");
        }
        let root = self.index("acquisition-members", &hash(group.as_bytes()));
        let mut headers = Vec::new();
        let mut last = None;
        while ordinal < acquisition.count {
            if let Some(identity) = read_json::<String>(&root.join(format!("{ordinal:020}")))? {
                let prepared = self.required(&identity)?;
                if !matches!(prepared.capture(), Some(Capture::Input(input)) if input.parsed.is_empty()
                    && input.group.as_ref().is_some_and(|header| header.identity == group && header.ordinal == ordinal && header.count == acquisition.count))
                {
                    return invalid("acquisition index differs from its immutable member header");
                }
                if headers.len() == limit {
                    return Ok((Some(acquisition.disposition), headers, last));
                }
                headers.push(prepared);
                last = Some(ordinal);
            }
            ordinal += 1;
        }
        Ok((Some(acquisition.disposition), headers, None))
    }

    pub fn source_heads(
        &self,
        source: &polylogue_store::primary::SourceSelector,
        version: &str,
    ) -> Result<sources::Heads> {
        self.ensure_indexes()?;
        sources::get(self, source, version)
    }

    pub fn acknowledge(&self, acknowledgment: Acknowledgment) -> Result<()> {
        let prepared = self.required(&acknowledgment.frame)?;
        validate_ack(&prepared, &acknowledgment)?;
        if let Some(previous) = self.acknowledgment(&prepared.identity)? {
            if encoded_hash(&previous)? != encoded_hash(&acknowledgment)? {
                return invalid("acknowledgment changed for an immutable operation");
            }
            return Ok(());
        }
        let phase = prepared.phase.advance(Phase::settled_by(&acknowledgment))?;
        let directory = self.operation(&prepared.identity)?;
        let temporary = tempfile::Builder::new()
            .prefix(".ack-preparing-")
            .tempdir_in(&directory)?;
        let payload = write_encoded_json(&temporary.path().join("payload.zst"), &acknowledgment)?;
        write_checked(
            &temporary.path().join("receipt.json"),
            &Receipt { payload, phase },
        )?;
        sync(temporary.path())?;
        boundary("ack_synced");
        fs::rename(temporary.path(), directory.join("ack"))?;
        sync(&directory)?;
        boundary("ack_published");
        self.retirement_dirty.set(true);
        Ok(())
    }

    /// Conflict evidence retains the owner's prior receipt without acknowledging
    /// the offered body. It uses the operation's existing atomic receipt directory.
    pub fn record_conflict(
        &self,
        identity: &str,
        offered_hash: &str,
        receipt: &Acknowledgment,
    ) -> Result<()> {
        let prepared = self.required(identity)?;
        validate_conflict(&prepared, offered_hash, receipt)?;
        if let Some(old) = self.conflict_receipt(identity)? {
            if encoded_hash(&old)? != encoded_hash(receipt)? {
                return invalid("conflicting receipt evidence changed");
            }
            return Ok(());
        }
        let phase = prepared.phase.advance(Phase::Conflicted {
            offered: offered_hash.into(),
        })?;
        let directory = self.operation(identity)?;
        let temporary = tempfile::Builder::new()
            .prefix(".ack-preparing-")
            .tempdir_in(&directory)?;
        let payload = write_encoded_json(&temporary.path().join("payload.zst"), receipt)?;
        write_checked(
            &temporary.path().join("receipt.json"),
            &Receipt { payload, phase },
        )?;
        sync(temporary.path())?;
        boundary("ack_synced");
        fs::rename(temporary.path(), directory.join("ack"))?;
        sync(&directory)?;
        boundary("ack_published");
        self.retirement_dirty.set(true);
        Ok(())
    }

    pub fn conflict_receipt(&self, identity: &str) -> Result<Option<Acknowledgment>> {
        let Some(summary) = self.receipt(identity)? else {
            return Ok(None);
        };
        let Phase::Conflicted { offered } = summary.phase else {
            return Ok(None);
        };
        let prepared = self.required(identity)?;
        let receipt: Acknowledgment = decode_json(verified_encoded(
            &self.operation(identity)?.join("ack/payload.zst"),
            summary.payload.descriptor()?,
        )?)?;
        validate_conflict(&prepared, &offered, &receipt)?;
        Ok(Some(receipt))
    }

    pub fn retire(&self, applied: i64) -> Result<()> {
        if applied < 0 {
            return invalid("confirmed position is negative");
        }
        self.ensure_indexes()?;
        if self.last_retired.get() == Some(applied) && !self.retirement_dirty.get() {
            return Ok(());
        }
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
            if !prepared.phase.confirmed_through(applied) {
                continue;
            }
            if prepared.group().is_some() {
                self.visit_captures(&prepared.identity, |identity| {
                    self.remove(identity)?;
                    boundary("group_member_retired");
                    Ok(())
                })?;
            } else if prepared.material().is_some() {
                continue;
            }
            self.remove(&prepared.identity)?;
        }
        self.ensure_indexes()?;
        self.last_retired.set(Some(applied));
        self.retirement_dirty.set(false);
        Ok(())
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
                if prepared.material().is_some() {
                    materials += 1;
                } else if prepared.is_write() {
                    writes += 1;
                }
                for file in fs::read_dir(self.operation(&prepared.identity)?)? {
                    let file = file?;
                    if file.file_type()?.is_file() {
                        bytes = bytes.saturating_add(file.metadata()?.len());
                    } else if file.file_name() == "ack" && file.file_type()?.is_dir() {
                        for receipt_file in fs::read_dir(file.path())? {
                            let receipt_file = receipt_file?;
                            if receipt_file.file_type()?.is_file() {
                                bytes = bytes.saturating_add(receipt_file.metadata()?.len());
                            }
                        }
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
            .group()
            .cloned()
            .ok_or_else(|| Error::Invalid("operation is not a capture group".into()))?;
        let descriptor = group.encoding.descriptor()?;
        if descriptor.hash.to_string() != group.captures_hash {
            return invalid("capture group encoding differs from its immutable set");
        }
        let mut file =
            verified_encoded(&self.operation(identity)?.join("captures.zst"), descriptor)?;
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
        encoded_body: Option<polylogue_store::primary::PreparedBody>,
    ) -> Result<Prepared> {
        self.ensure_indexes()?;
        let identity = frame.frame().to_owned();
        if let Some(previous) = self.get(&identity)? {
            let desired = encoded_hash(&frame)?;
            if previous.frame.identity() != (desired.0.as_str(), desired.1) {
                return invalid("prepared identity changed immutable frame contents");
            }
            if previous.body().is_some() != (body.is_some() || encoded_body.is_some()) {
                return invalid("prepared identity changed its body contract");
            }
            if previous.group().map(|group| group.captures_hash.as_str())
                != captures.as_ref().map(|(_, digest)| digest.as_str())
            {
                return invalid("prepared identity changed its capture group");
            }
            if let Some(input) = body.as_mut() {
                let mut output = HashWriter::new(std::io::sink());
                std::io::copy(input, &mut output)?;
                let (blob, length) = output.finish();
                if previous.body().map(Bytes::identity) != Some((blob.as_str(), length)) {
                    return invalid("prepared identity changed immutable body contents");
                }
            }
            if let Some(encoded) = encoded_body {
                let descriptor = encoded.descriptor;
                let _validated = hashed_blobs::Prepared::from_encoded(
                    encoded.file,
                    descriptor.clone(),
                    None,
                    ReadLimits::default(),
                )?;
                let blob = descriptor.hash.to_string();
                if previous.body().map(Bytes::identity)
                    != Some((blob.as_str(), descriptor.decoded_len))
                {
                    return invalid("prepared identity changed immutable encoded body contents");
                }
            }
            let desired_capture = full_capture.map(encoded_hash).transpose()?;
            if previous.captured().map(Bytes::identity)
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
        let frame_bytes = write_encoded_json(&temporary.path().join("frame.zst"), &frame)?;
        let mut facts = material_facts(&frame)?;
        let captured = full_capture
            .map(|full| -> Result<Bytes> {
                let header = capture_header(full);
                if facts.as_ref().and_then(|facts| facts.capture.as_ref()) != Some(&header) {
                    return invalid("capture header differs from its material manifest");
                }
                let bytes = write_encoded_json(&temporary.path().join("captured.zst"), full)?;
                boundary("capture_synced");
                Ok(bytes)
            })
            .transpose()?;
        let body = if body.is_some() || encoded_body.is_some() {
            let descriptor = facts
                .as_ref()
                .ok_or_else(|| Error::Invalid("body has no material descriptor".into()))?;
            let path = temporary.path().join("body.zst");
            let bytes = if let Some(encoded) = encoded_body {
                write_prepared(&path, encoded)?
            } else {
                let preparation = Preparation::new(temporary.path())?;
                let encoded = preparation.plain_into(
                    create_payload(&path)?,
                    body.take().expect("decoded body"),
                    3,
                )?;
                let descriptor = encoded.descriptor();
                encoded.into_file().sync_all()?;
                Bytes::new(&descriptor)
            };
            if bytes.blob != descriptor.blob || bytes.length != descriptor.length {
                return invalid("captured bytes differ from their prepared material");
            }
            boundary("body_synced");
            Some(bytes)
        } else {
            None
        };
        let group = captures
            .map(|(captures, captures_hash)| -> Result<Group> {
                let path = temporary.path().join("captures.zst");
                let mut writer =
                    Preparation::new(temporary.path())?.plain_writer(create_payload(&path)?, 3)?;
                for capture in captures {
                    serde_json::to_writer(&mut writer, capture)?;
                    writer.write_all(b"\n")?;
                }
                let encoded = writer.finish()?;
                let descriptor = encoded.descriptor();
                encoded.into_file().sync_all()?;
                if descriptor.hash.to_string() != captures_hash {
                    return invalid("capture group serialization changed");
                }
                Ok(Group {
                    captures_hash,
                    captures_count: captures.len() as u64,
                    encoding: Encoding::new(&descriptor),
                })
            })
            .transpose()?;
        let order = self.next.get();
        let export_bindings = exports::bindings(&frame, full_capture)?;
        let payload = match (&frame, facts.take()) {
            (Ingress::Material { .. }, Some(facts)) => Payload::Material(Box::new(Material {
                id: facts.id,
                blob: facts.blob,
                length: facts.length,
                body: body.ok_or_else(|| {
                    Error::Invalid("material must have its immutable body prepared".into())
                })?,
                capture: facts.capture,
                captured,
                control: full_capture.and_then(control),
                exports: export_bindings,
                named: facts.named,
                request: facts.request,
            })),
            (Ingress::Events { .. }, _) => Payload::Events { group },
            (Ingress::ReplaySequence { .. }, _) => Payload::ReplaySequence { group },
            (Ingress::Signals { .. }, _) => Payload::Signals,
            (Ingress::RegisterKinds { .. }, _) => Payload::RegisterKinds,
            (Ingress::Chunk { .. }, _) => Payload::Chunk,
            (Ingress::LargeFrame { .. }, _) => Payload::LargeFrame,
            (Ingress::Material { .. }, None) => return invalid("material has no descriptor"),
        };
        let prepared = Record {
            order,
            identity: identity.clone(),
            frame: frame_bytes,
            payload,
        };
        write_checked(&temporary.path().join("metadata.json"), &prepared)?;
        sync(temporary.path())?;
        boundary("operation_synced");
        if prepared.group().is_some() {
            boundary("group_synced");
        }
        self.index_dirty.set(true);
        fs::rename(temporary.path(), self.operation(&identity)?)?;
        sync(&self.root.join("operations"))?;
        boundary("operation_published");
        if prepared.group().is_some() {
            boundary("group_published");
        }
        self.next.set(
            order
                .checked_add(1)
                .ok_or_else(|| Error::Invalid("producer queue order exhausted".into()))?,
        );
        let prepared = Prepared {
            record: prepared,
            phase: Phase::Pending,
            unresolved: false,
        };
        self.index_entry(&prepared)?;
        self.index_dirty.set(false);
        Ok(prepared)
    }

    fn remove(&self, identity: &str) -> Result<()> {
        let mut path = self.operation(identity)?;
        let acquisition = identity.rsplit_once(':').and_then(|(batch, ordinal)| {
            (batch.starts_with("acquisition:") && ordinal.parse::<u64>().is_ok()).then_some(batch)
        });
        let batch =
            acquisition.map(|batch| self.root.join("operations").join(hash(batch.as_bytes())));
        if let Some(batch) = &batch
            && let Some(metadata) = read_checked::<Acquisition>(&batch.join("acquisition.json"))?
        {
            if Some(metadata.identity.as_str()) != acquisition {
                return invalid("retiring acquisition has another identity");
            }
            // A crash may have retired this member before its acquisition
            // header. The confirmed group still owns the cleanup.
            path = batch.join("operations").join(hash(identity.as_bytes()));
        }
        self.retire_path(&path)?;
        if let Some(batch) = batch.filter(|batch| batch.exists())
            && fs::read_dir(batch.join("operations"))?.try_fold(
                true,
                |empty, entry| -> Result<bool> {
                    Ok(empty && !entry?.file_name().to_str().is_some_and(hex_name))
                },
            )?
        {
            self.retire_path(&batch)?;
        }
        Ok(())
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
        sync(directory.parent().expect("owned operation parent"))?;
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
        for name in [
            "order",
            "groups",
            "members",
            "locations",
            "names",
            "pending-source",
            "acquisition-members",
            "raw-generations",
            "raw-roots",
            "raw-children",
            "export-readings",
            "export-inputs",
        ] {
            fs::create_dir_all(index.join(name))?;
        }
        let (mut first, mut last) = (u64::MAX, 0);
        // Locations are derived from the one published acquisition directory.
        // Build them before resolving member receipts and capture dependencies.
        self.visit_operation_paths(|path, batch| {
            if let Some(batch) = batch {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| Error::Invalid("invalid acquisition member name".into()))?;
                let location = self.index("locations", name);
                if read_json::<String>(&location)?.is_some()
                    || self.root.join("operations").join(name).try_exists()?
                {
                    return invalid("operation occurs in more than one acquisition");
                }
                write_index(&location, batch)?;
            }
            Ok(())
        })?;
        self.visit_operation_paths(|path, _| {
            let name = path.file_name().and_then(|name| name.to_str())
                .ok_or_else(|| Error::Invalid("invalid spool operation name".into()))?;
            if path.join("frame.json").try_exists()? {
                return invalid(
                    "producer spool contains plaintext pending payloads; preserve them before an explicit transition",
                );
            }
            let metadata = read_checked::<Record>(&path.join("metadata.json"))?
                .ok_or_else(|| Error::Invalid("published operation has no metadata".into()))?;
            if hash(metadata.identity.as_bytes()) != name {
                return invalid("spool directory has another operation identity");
            }
            let prepared = self.required(&metadata.identity)?;
            first = first.min(prepared.order);
            last = last.max(prepared.order);
            self.index_entry(&prepared)?;
            Ok(())
        })?;
        self.first.set(if first == u64::MAX { 1 } else { first });
        self.next.set(
            last.checked_add(1)
                .ok_or_else(|| Error::Invalid("producer queue order exhausted".into()))?,
        );
        self.index_dirty.set(false);
        sync(&index)?;
        Ok(())
    }

    fn visit_operation_paths(
        &self,
        mut visit: impl FnMut(&Path, Option<&str>) -> Result<()>,
    ) -> Result<()> {
        for entry in fs::read_dir(self.root.join("operations"))? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name) = name.to_str().filter(|name| hex_name(name)) else {
                continue;
            };
            if !entry.file_type()?.is_dir() {
                return invalid("spool operation is not a directory");
            }
            if let Some(acquisition) =
                read_checked::<Acquisition>(&entry.path().join("acquisition.json"))?
            {
                if hash(acquisition.identity.as_bytes()) != name {
                    return invalid("acquisition directory has another identity");
                }
                for member in fs::read_dir(entry.path().join("operations"))? {
                    let member = member?;
                    let member_name = member.file_name();
                    if !member_name.to_str().is_some_and(hex_name) {
                        continue;
                    }
                    if !member.file_type()?.is_dir() {
                        return invalid("acquisition member is not a directory");
                    }
                    visit(&member.path(), Some(name))?;
                }
            } else {
                visit(&entry.path(), None)?;
            }
        }
        Ok(())
    }

    fn index_entry(&self, prepared: &Prepared) -> Result<()> {
        let path = self.operation(&prepared.identity)?;
        let parent = path.parent().expect("operation parent");
        if parent != self.root.join("operations") {
            let acquisition = read_checked::<Acquisition>(
                &parent
                    .parent()
                    .expect("acquisition parent")
                    .join("acquisition.json"),
            )?
            .ok_or_else(|| Error::Invalid("indexed member has no acquisition".into()))?;
            let base = acquisition
                .order_base
                .ok_or_else(|| Error::Invalid("indexed acquisition has no order".into()))?;
            let ordinal = prepared
                .order
                .checked_sub(base)
                .and_then(|relative| relative.checked_sub(1))
                .filter(|ordinal| *ordinal < acquisition.count)
                .ok_or_else(|| {
                    Error::Invalid("indexed member has another acquisition order".into())
                })?;
            let root = self.index(
                "acquisition-members",
                &hash(acquisition.identity.as_bytes()),
            );
            fs::create_dir_all(&root)?;
            write_index(&root.join(format!("{ordinal:020}")), &prepared.identity)?;
        }
        exports::insert(self, prepared)?;
        sources::insert(self, prepared)?;
        raw_ranges::insert(self, prepared)?;
        if let Some(name) = prepared.logical_name() {
            names::insert(&self.index("names", ""), name, &prepared.identity)?;
        }
        let order = self.index("order", &format!("{:020}", prepared.order));
        if read_json::<String>(&order)?.is_some_and(|identity| identity != prepared.identity) {
            return invalid("two pending operations have the same queue order");
        }
        write_index(&order, &prepared.identity)?;
        if let Some(group) = prepared.group() {
            let index = self.index("groups", &group.captures_hash);
            if read_json::<String>(&index)?.is_some_and(|identity| identity != prepared.identity) {
                return invalid("two frames claim the same capture group");
            }
            write_index(&index, &prepared.identity)?;
            self.visit_captures(&prepared.identity, |capture| {
                let index = self.index("members", &hash(capture.as_bytes()));
                if read_json::<String>(&index)?
                    .is_some_and(|identity| identity != prepared.identity)
                {
                    return invalid("capture belongs to two frames");
                }
                write_index(&index, &prepared.identity)?;
                boundary("group_indexed_member");
                Ok(())
            })?;
        }
        Ok(())
    }
}

fn hex_name(name: &str) -> bool {
    name.len() == 64 && name.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn control(capture: &Capture) -> Option<Control> {
    let Capture::Input(input) = capture else {
        return None;
    };
    input
        .parsed
        .iter()
        .rev()
        .find_map(|reading| reading.checkpoint.as_ref())
        .map(|checkpoint| Control {
            reading_version: checkpoint.reading_version.clone(),
            parsed_upto: checkpoint.parsed_upto,
        })
}

pub(crate) fn capture_header(capture: &Capture) -> Capture {
    match capture {
        Capture::Input(input) => {
            Capture::Input(Box::new(polylogue_store::primary::CapturedInput {
                generation: input.generation.clone(),
                raw_total: input.raw_total,
                raw_predecessor: input.raw_predecessor.clone(),
                input: input.input.clone(),
                rows: input.rows.clone(),
                index: input.index,
                previous: input.previous.clone(),
                hash: input.hash,
                length: input.length,
                group: input.group.clone(),
                arrival_rank: input.arrival_rank,
                parsed: Vec::new(),
                raw_range: input.raw_range,
                basis: input.basis.clone(),
            }))
        }
        Capture::Blob { hash, length } => Capture::Blob {
            hash: *hash,
            length: *length,
        },
    }
}

/// What a material frame says about itself before its bytes are stored.
struct MaterialFacts {
    id: String,
    blob: String,
    length: u64,
    capture: Option<Capture>,
    named: bool,
    request: bool,
}

fn material_facts(frame: &Ingress) -> Result<Option<MaterialFacts>> {
    let Ingress::Material {
        frame: name,
        material,
    } = frame
    else {
        return Ok(None);
    };
    let capture: Option<Capture> = material
        .manifest
        .details
        .get("polylogue_capture")
        .map(|capture| serde_json::from_value(capture.clone()))
        .transpose()?;
    if let Some(capture) = &capture
        && *capture != capture_header(capture)
    {
        return invalid(
            "raw material manifest must keep parser facts in the private capture sidecar",
        );
    }
    let parser = material.parser.as_str();
    Ok(Some(MaterialFacts {
        id: material.id.to_string(),
        blob: material.blob.clone(),
        length: material.length,
        capture,
        named: matches!(parser, "polylogue-products" | "polylogue-request-restore")
            && name.ends_with(":observation"),
        request: matches!(parser, "polylogue-request" | "polylogue-request-restore"),
    }))
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
fn validate_conflict(prepared: &Prepared, offered: &str, receipt: &Acknowledgment) -> Result<()> {
    if receipt.frame != prepared.identity
        || offered != prepared.frame.blob
        || receipt.number.is_none_or(|number| number <= 0)
        || receipt.content_hash() == Some(offered)
    {
        return invalid("conflict evidence does not bind this exact offered operation");
    }
    Ok(())
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
        if prepared.is_write() && position.is_none() {
            return invalid("a write acknowledgment has no confirmed position");
        }
        if let Some(expected) = prepared.material()
            && material != Some(expected.id.as_str())
        {
            return invalid("material acknowledgment has another material identity");
        }
    }
    Ok(())
}

fn sorted_members(captures: &[String]) -> Result<Vec<&str>> {
    let mut sorted: Vec<_> = captures.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return invalid("a capture group repeats a member");
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
fn decode_capture(reader: impl Read) -> Result<Capture> {
    decode_json(reader)
}

/// The checksum covers the stored bytes of the value, so a record written by
/// an earlier build verifies even where the current type no longer keeps one
/// of its fields.
fn read_checked<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    let Some(checked) = read_json::<Checked<Box<serde_json::value::RawValue>>>(path)? else {
        return Ok(None);
    };
    if hash_name(blake3::hash(checked.value.get().as_bytes())) != checked.digest {
        return invalid("spool metadata checksum differs from its contents");
    }
    Ok(Some(serde_json::from_str(checked.value.get())?))
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
    write_derived(path, &identity)
}

fn write_derived(path: &Path, value: &impl Serialize) -> Result<()> {
    // An interrupted derived index is rebuilt before it is read again; only
    // the immutable operation publication needs a durability barrier.
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    serde_json::to_writer(file, value)?;
    Ok(())
}

pub(crate) fn decode_json<T: serde::de::DeserializeOwned>(reader: impl Read) -> Result<T> {
    let mut decoder = serde_json::Deserializer::from_reader(reader);
    decoder.disable_recursion_limit();
    let value = T::deserialize(&mut decoder)?;
    decoder.end()?;
    Ok(value)
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
fn create_payload(path: &Path) -> Result<File> {
    Ok(OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)?)
}
fn write_encoded_json(path: &Path, value: &impl Serialize) -> Result<Bytes> {
    let mut writer = Preparation::new(path.parent().expect("payload parent"))?
        .plain_writer(create_payload(path)?, 3)?;
    serde_json::to_writer(&mut writer, value)?;
    let encoded = writer.finish()?;
    let descriptor = encoded.descriptor();
    encoded.into_file().sync_all()?;
    Ok(Bytes::new(&descriptor))
}
fn write_prepared(path: &Path, prepared: polylogue_store::primary::PreparedBody) -> Result<Bytes> {
    use std::os::unix::fs::MetadataExt;
    let descriptor = prepared.descriptor;
    let checked = hashed_blobs::Prepared::from_encoded(
        prepared.file,
        descriptor.clone(),
        None,
        ReadLimits::default(),
    )?;
    let mut input = checked.into_file();
    if let Some(source) = prepared.path
        && fs::hard_link(source, path).is_ok()
    {
        if let Ok(linked) = regular(path) {
            let a = input.metadata()?;
            let b = linked.metadata()?;
            if a.dev() == b.dev() && a.ino() == b.ino() {
                hashed_blobs::Prepared::from_encoded(
                    linked,
                    descriptor.clone(),
                    None,
                    ReadLimits::default(),
                )?
                .into_file()
                .sync_all()?;
                return Ok(Bytes::new(&descriptor));
            }
        }
        fs::remove_file(path)?;
    }
    input.rewind()?;
    let mut output = create_payload(path)?;
    std::io::copy(&mut input, &mut output)?;
    output.sync_all()?;
    hashed_blobs::Prepared::from_encoded(output, descriptor.clone(), None, ReadLimits::default())?;
    Ok(Bytes::new(&descriptor))
}
fn verified_encoded(path: &Path, descriptor: EncodedBlob) -> Result<BlobReader> {
    let prepared = hashed_blobs::Prepared::from_encoded(
        regular(path)?,
        descriptor,
        None,
        ReadLimits::default(),
    )?;
    Ok(prepared.reader(None, ReadLimits::default())?)
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

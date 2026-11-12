//! Pending producer operations. Immutable frame and body files become visible
//! together; small queue metadata never contains event or request payloads.
use crate::{Error, Result};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use hashed_blobs::{BlobReader, EncodedBlob, Preparation, ReadLimits};
use polylogue_store::primary::queue::{self, Lookup, Order, Queue};
use polylogue_store::primary::{Capture, ReplayState};
use serde::{Deserialize, Serialize};
use sinex_protocol::{Acknowledgment, FrameResult, Ingress};
use std::cell::RefCell;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
mod exports;
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

/// A material's body files: its one encoding (`body.zst`), or the encodings
/// of its content-defined chunks in order (`body-<n>.zst`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum Body {
    Whole(Bytes),
    Chunks(Vec<Bytes>),
}
impl Body {
    pub(crate) fn parts(&self) -> &[Bytes] {
        match self {
            Self::Whole(bytes) => std::slice::from_ref(bytes),
            Self::Chunks(chunks) => chunks,
        }
    }
    fn file_name(&self, index: usize) -> String {
        match self {
            Self::Whole(_) => "body.zst".into(),
            Self::Chunks(_) => format!("body-{index}.zst"),
        }
    }
    fn identity(&self) -> Vec<(&str, u64)> {
        self.parts().iter().map(Bytes::identity).collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Group {
    pub captures_hash: String,
}

/// A captured material and the facts the queue keeps about its capture.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Material {
    pub id: String,
    pub blob: String,
    pub length: u64,
    pub body: Body,
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

pub(crate) use polylogue_store::primary::queue::Phase;

fn acknowledgment_phase(acknowledgment: &Acknowledgment) -> Phase {
    if acknowledgment.refusal().is_some() {
        Phase::Refused
    } else {
        Phase::Acknowledged {
            commit_seq: acknowledgment.commit_seq(),
        }
    }
}

/// The immutable operation descriptor stored in the durable queue.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Record {
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
    pub(crate) fn body(&self) -> Option<&Body> {
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
    pub order: u64,
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
    disposition: polylogue_store::primary::CaptureDisposition,
}

#[derive(Serialize, Deserialize)]
struct Checked<T> {
    value: T,
    digest: String,
}

/// The finished outcome and the acknowledgment bytes that proved it.
#[derive(Serialize, Deserialize)]
struct StoredAcknowledgment {
    payload: Bytes,
    phase: Phase,
}

/// The file whose modification time records the last acknowledgement.
const ACKNOWLEDGED: &str = "last-acknowledged";

pub(crate) struct Spool {
    root: PathBuf,
    queue: Queue,
    waker: RefCell<Option<Arc<dyn Fn() + Send + Sync>>>,
    _lock: File,
}

/// Encoded capture is verified and synced privately before publication takes
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
        if self.queue.queue.len()? != self.count {
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
                return invalid("export capture has no captured body");
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
            if entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("queue.db") || name == "lock")
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

pub(crate) struct PinnedBackup {
    directory: tempfile::TempDir,
    database: queue::Snapshot,
    target: PathBuf,
}
impl PinnedBackup {
    pub fn finish(self, cancelled: &AtomicBool) -> Result<tempfile::TempDir> {
        self.database
            .write_to(&self.target.join("queue.db"), cancelled)?;
        sync(&self.target)?;
        Ok(self.directory)
    }
}

impl Spool {
    /// The read snapshot and immutable hard links hold the same queue cut.
    /// Database copying and payload hashing happen after releasing the mutex.
    pub(crate) fn pin_backup(&self, cancelled: &AtomicBool) -> Result<PinnedBackup> {
        check_backup_cancelled(cancelled)?;
        let database = self.queue.snapshot()?;
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
        Ok(PinnedBackup {
            directory: pinned,
            database,
            target,
        })
    }

    pub fn open(store: &Path, producer: &str) -> Result<Self> {
        let root = store.join("producer-spool").join(hash(producer.as_bytes()));
        fs::create_dir_all(root.join("operations"))?;
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
        if !root.join("queue.db").exists()
            && fs::read_dir(root.join("operations"))?.next().is_some()
        {
            return invalid(
                "producer spool has an earlier queue format; preserve it before an explicit transition",
            );
        }
        let queue = Queue::open(&root.join("queue.db"))?;
        let spool = Self {
            root,
            queue,
            waker: RefCell::new(None),
            _lock: lock,
        };
        spool.remove_unpublished()?;
        Ok(spool)
    }

    /// Operations, acquisitions and acknowledgments are prepared in dot-named
    /// directories and published by rename, so a dot-named directory left at
    /// open belongs to a preparation that crashed before publication. The
    /// spool lock means no other writer is preparing now. Some hold hard links
    /// to large bodies, so they are removed rather than left to accumulate.
    fn remove_unpublished(&self) -> Result<()> {
        let operations = self.root.join("operations");
        remove_dot_entries(&operations)?;
        for entry in fs::read_dir(&operations)? {
            let entry = entry?;
            if !entry.file_name().to_str().is_some_and(hex_name) || !entry.file_type()?.is_dir() {
                continue;
            }
            refuse_previous_operation(&entry.path())?;
            let members = entry.path().join("operations");
            if members.is_dir() {
                for member in fs::read_dir(&members)? {
                    refuse_previous_operation(&member?.path())?;
                }
            }
            let relative = format!(
                "operations/{}",
                entry.file_name().to_str().expect("hex name checked")
            );
            if !self.queue.references_directory(&relative)? {
                fs::remove_dir_all(entry.path())?;
                continue;
            }
            remove_dot_entries(&entry.path())?;
            if entry.path().join("acquisition.json").try_exists()? && members.is_dir() {
                remove_staging_database(&entry.path())?;
                remove_dot_entries(&members)?;
                for member in fs::read_dir(&members)? {
                    let member = member?;
                    if member.file_name().to_str().is_some_and(hex_name)
                        && member.file_type()?.is_dir()
                    {
                        let relative = format!(
                            "{relative}/operations/{}",
                            member.file_name().to_str().expect("hex name checked")
                        );
                        if self.queue.references_directory(&relative)? {
                            remove_dot_entries(&member.path())?;
                        } else {
                            fs::remove_dir_all(member.path())?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn operation(&self, identity: &str) -> Result<PathBuf> {
        Ok(match self.queue.get(identity)? {
            Some(entry) => self.root.join(entry.directory),
            None => self.root.join("operations").join(hash(identity.as_bytes())),
        })
    }

    fn prepared(&self, entry: queue::Entry) -> Result<Prepared> {
        let record: Record = serde_json::from_slice(&entry.metadata)?;
        if record.identity != entry.identity {
            return invalid("queue record has another identity");
        }
        Ok(Prepared {
            order: entry.order,
            record,
            phase: entry.phase,
            unresolved: entry.unresolved,
        })
    }

    pub fn get(&self, identity: &str) -> Result<Option<Prepared>> {
        self.queue
            .get(identity)?
            .map(|entry| self.prepared(entry))
            .transpose()
    }

    pub fn lookup(
        &self,
        lookup: Lookup<'_>,
        after: Option<u64>,
        through: Option<u64>,
        order: Order,
        limit: usize,
    ) -> Result<Vec<Prepared>> {
        self.queue
            .lookup(lookup, after, through, order, limit)?
            .into_iter()
            .map(|entry| self.prepared(entry))
            .collect()
    }

    pub fn ready(&self, in_flight: &[String], limit: usize) -> Result<Vec<Prepared>> {
        self.queue
            .ready(in_flight, limit)?
            .into_iter()
            .map(|entry| self.prepared(entry))
            .collect()
    }

    pub fn set_waker(&self, waker: Arc<dyn Fn() + Send + Sync>) {
        *self.waker.borrow_mut() = Some(waker);
    }

    fn wake(&self) {
        if let Some(waker) = self.waker.borrow().as_ref() {
            waker();
        }
    }

    /// Decode the original immutable common encoding without collecting the frame.
    pub fn frame(&self, identity: &str) -> Result<BlobReader> {
        Ok(self
            .prepared_frame(identity)?
            .reader(None, ReadLimits::default())?)
    }

    /// Open immutable files under the queue lock; codec verification belongs
    /// to the sender after that lock is released.
    pub fn pin_frame(&self, identity: &str) -> Result<(File, EncodedBlob)> {
        let descriptor = self.required(identity)?.frame.descriptor()?;
        Ok((
            regular(&self.operation(identity)?.join("frame.zst"))?,
            descriptor,
        ))
    }

    pub fn pin_body(&self, identity: &str) -> Result<Vec<(File, EncodedBlob)>> {
        self.encoded_body(identity)?
            .into_iter()
            .map(|(path, descriptor)| Ok((regular(&path)?, descriptor)))
            .collect()
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

    /// The material's bytes, read through its body files in order.
    pub fn body(&self, identity: &str) -> Result<Box<dyn Read + Send>> {
        let mut reader: Box<dyn Read + Send> = Box::new(std::io::empty());
        for part in self.prepared_body(identity)? {
            reader = Box::new(reader.chain(part.reader(None, ReadLimits::default())?));
        }
        Ok(reader)
    }

    /// The body files in order, each with its encoding. The caller keeps the
    /// named spool inodes immutable until the applied barrier.
    pub fn encoded_body(&self, identity: &str) -> Result<Vec<(PathBuf, EncodedBlob)>> {
        let prepared = self.required(identity)?;
        let body = prepared
            .body()
            .ok_or_else(|| Error::Invalid("operation has no material body".into()))?;
        let directory = self.operation(identity)?;
        body.parts()
            .iter()
            .enumerate()
            .map(|(index, part)| {
                let descriptor = part.descriptor()?;
                let path = directory.join(body.file_name(index));
                check_length(&path, descriptor.stored_len)?;
                Ok((path, descriptor))
            })
            .collect()
    }

    /// Pinned handles survive publication retries without a decoded copy.
    pub fn prepared_body(&self, identity: &str) -> Result<Vec<hashed_blobs::Prepared>> {
        self.encoded_body(identity)?
            .into_iter()
            .map(|(path, descriptor)| {
                Ok(hashed_blobs::Prepared::from_encoded(
                    regular(&path)?,
                    descriptor,
                    None,
                    ReadLimits::default(),
                )?)
            })
            .collect()
    }

    fn stored_acknowledgment(&self, identity: &str) -> Result<Option<StoredAcknowledgment>> {
        let directory = self.operation(identity)?.join("ack");
        if directory.try_exists()? && !fs::symlink_metadata(&directory)?.file_type().is_dir() {
            return invalid("published acknowledgment is not a directory");
        }
        let acknowledgment =
            read_checked::<StoredAcknowledgment>(&directory.join("acknowledgment.json"))?;
        if let Some(acknowledgment) = &acknowledgment {
            if acknowledgment.phase.is_pending() {
                return invalid("published acknowledgment has no finished outcome");
            }
            check_length(
                &directory.join("payload.zst"),
                acknowledgment.payload.descriptor()?.stored_len,
            )?;
        } else if directory.try_exists()? {
            return invalid("published acknowledgment has no acknowledgment");
        }
        Ok(acknowledgment)
    }

    pub fn acknowledgment(&self, identity: &str) -> Result<Option<Acknowledgment>> {
        let Some(summary) = self.stored_acknowledgment(identity)? else {
            return Ok(None);
        };
        if matches!(summary.phase, Phase::Conflicted { .. }) {
            self.conflict_acknowledgment(identity)?;
            return Ok(None);
        }
        let acknowledgment: Acknowledgment = decode_json(verified_encoded(
            &self.operation(identity)?.join("ack/payload.zst"),
            summary.payload.descriptor()?,
        )?)?;
        let prepared = self.required(identity)?;
        validate_ack(&prepared, &acknowledgment)?;
        if summary.phase != acknowledgment_phase(&acknowledgment) {
            return invalid("acknowledgment summary differs from its immutable acknowledgment");
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
        self.prepare_operation(frame, None::<std::io::Empty>, None, None, Some(vec![bytes]))
    }

    /// `bytes` are the body's encodings in order: one, or one per chunk the
    /// frame names.
    pub fn prepare_encoded_capture(
        &self,
        frame: Ingress,
        bytes: Vec<polylogue_store::primary::PreparedBody>,
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
        let queue = Spool {
            root: temporary.path().into(),
            queue: Queue::open(&temporary.path().join("queue.db"))?,
            waker: RefCell::new(None),
            _lock: self._lock.try_clone()?,
        };
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
        let header = self
            .queue
            .get(identity)?
            .map(|entry| serde_json::from_slice::<Acquisition>(&entry.metadata))
            .transpose()?;
        if header
            .as_ref()
            .is_some_and(|header| header.identity != identity)
        {
            return invalid("acquisition acknowledgment has another identity");
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
            read_checked::<Acquisition>(&path)?.expect("read above under capture lock");
        if acquisition.intent != staged.intent || acquisition.count != staged.count {
            return Ok(None);
        }
        let mut after = None;
        loop {
            let entries = staged.queue.pending_after(after, 128)?;
            if entries.is_empty() {
                break;
            }
            after = entries.last().map(|entry| entry.order);
            for desired in entries {
                let Some(original) = self.get(&desired.identity)? else {
                    return Ok(None);
                };
                if original.frame.identity() != desired.frame.identity()
                    || original.body().map(Body::identity) != desired.body().map(Body::identity)
                    || original.captured().map(Bytes::identity)
                        != desired.captured().map(Bytes::identity)
                {
                    return Ok(None);
                }
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
        if self.queue.get(&staged.identity)?.is_some() {
            if self.matching_acquisition(&staged)? != Some(disposition) {
                return invalid("acquisition identity changed its frozen capture or disposition");
            }
            return Ok(());
        }
        staged.finish()?;
        let header = Acquisition {
            identity: staged.identity.clone(),
            intent: staged.intent.clone(),
            count: staged.count,
            disposition,
        };
        write_checked(&staged.temporary.path().join("acquisition.json"), &header)?;
        // Close the private database at its original path before moving its
        // directory, so SQLite finishes its checkpoint at the same location.
        drop(staged.queue);
        sync(staged.temporary.path())?;
        boundary("acquisition_synced");
        let directory = format!("operations/{}", hash(staged.identity.as_bytes()));
        let destination = self.root.join(&directory);
        // No database publication means the immutable directory is still private.
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        fs::rename(staged.temporary.path(), &destination)?;
        sync(&self.root.join("operations"))?;
        let mut operation = empty_operation(
            staged.identity.clone(),
            directory,
            serde_json::to_vec(&header)?,
            queue::Kind::Acquisition,
        );
        operation.unresolved =
            disposition == polylogue_store::primary::CaptureDisposition::UnresolvedBasisConflict;
        let staged_queue = Queue::open(&destination.join("queue.db"))?;
        self.queue.publish_acquisition(&staged_queue, &operation)?;
        drop(staged_queue);
        boundary("acquisition_published");
        remove_staging_database(&destination)?;
        self.wake();
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
    /// Queue publication commits its complete membership with the frame.
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
        self.group_for_sorted(&captures_hash(&sorted_members(captures)?)?)
    }

    fn group_for_sorted(&self, captures_hash: &str) -> Result<Option<Prepared>> {
        self.queue
            .lookup(
                Lookup::GroupHash(captures_hash),
                None,
                None,
                Order::OldestFirst,
                1,
            )?
            .into_iter()
            .next()
            .map(|entry| self.prepared(entry))
            .transpose()
    }

    pub fn capture_group(&self, identity: &str) -> Result<Option<Prepared>> {
        self.queue
            .group_for_member(identity)?
            .map(|entry| self.prepared(entry))
            .transpose()
    }

    pub(crate) fn is_known_empty(&self) -> bool {
        self.queue.is_empty().unwrap_or(false)
    }

    pub fn pending_after(&self, after: Option<u64>, limit: usize) -> Result<Vec<Prepared>> {
        self.queue
            .after(after, limit)?
            .into_iter()
            .map(|entry| self.prepared(entry))
            .collect()
    }

    pub fn visit_names(
        &self,
        prefix: &str,
        visit: &mut dyn FnMut(&str) -> Result<()>,
    ) -> Result<()> {
        let selector = polylogue_store::primary::NamedProductSelector::Prefix(prefix.to_owned());
        let mut after = None;
        loop {
            let entries = self.queue.lookup(
                Lookup::Name(&selector),
                after,
                None,
                Order::OldestFirst,
                128,
            )?;
            if entries.is_empty() {
                return Ok(());
            }
            after = entries.last().map(|entry| entry.order);
            for entry in entries {
                visit(&entry.identity)?;
            }
        }
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
        let Some(disposition) = self.acquisition_disposition(group)? else {
            return Ok((None, Vec::new(), None));
        };
        let mut entries = self
            .queue
            .acquisition_after(group, after, limit.saturating_add(1))?;
        let next = if entries.len() > limit {
            entries.truncate(limit);
            self.queue
                .acquisition_ordinal(&entries.last().expect("nonempty page").identity)?
        } else {
            None
        };
        Ok((
            Some(disposition),
            entries
                .into_iter()
                .map(|entry| self.prepared(entry))
                .collect::<Result<_>>()?,
            next,
        ))
    }

    pub fn source_heads(
        &self,
        source: &polylogue_store::primary::SourceSelector,
        version: &str,
    ) -> Result<sources::Heads> {
        sources::get(self, source, version)
    }

    pub fn acknowledge(&self, acknowledgment: Acknowledgment) -> Result<()> {
        let prepared = self.required(&acknowledgment.frame)?;
        validate_ack(&prepared, &acknowledgment)?;
        if let Some(previous) = self.acknowledgment(&prepared.identity)? {
            if encoded_hash(&previous)? != encoded_hash(&acknowledgment)? {
                return invalid("acknowledgment changed for an immutable operation");
            }
            if prepared.phase.is_pending() {
                self.finish_acknowledgment(
                    &prepared.identity,
                    &acknowledgment_phase(&acknowledgment),
                    &acknowledgment,
                )?;
            }
            return Ok(());
        }
        if !prepared.phase.is_pending() {
            return invalid("a finished operation cannot change its outcome");
        }
        let phase = acknowledgment_phase(&acknowledgment);
        let directory = self.operation(&prepared.identity)?;
        let temporary = tempfile::Builder::new()
            .prefix(".ack-preparing-")
            .tempdir_in(&directory)?;
        let payload = write_encoded_json(&temporary.path().join("payload.zst"), &acknowledgment)?;
        write_checked(
            &temporary.path().join("acknowledgment.json"),
            &StoredAcknowledgment {
                payload,
                phase: phase.clone(),
            },
        )?;
        sync(temporary.path())?;
        boundary("ack_synced");
        fs::rename(temporary.path(), directory.join("ack"))?;
        sync(&directory)?;
        boundary("ack_files_published");
        self.finish_acknowledgment(&prepared.identity, &phase, &acknowledgment)?;
        boundary("ack_published");
        // The marker's modification time is when Sinex last acknowledged a
        // write; it outlives a restart and the removed operations.
        // A crash may leave it at an earlier acknowledgement, never a later one.
        fs::write(self.root.join(ACKNOWLEDGED), &acknowledgment.frame)?;
        Ok(())
    }

    /// When Sinex last acknowledged a write this spool sent, as recorded by
    /// the acknowledgement marker; `None` before the first acknowledgement.
    pub fn last_acknowledged(&self) -> Result<Option<std::time::SystemTime>> {
        match fs::metadata(self.root.join(ACKNOWLEDGED)) {
            Ok(metadata) => Ok(Some(metadata.modified()?)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// Conflict evidence retains the owner's prior acknowledgment without acknowledging
    /// the offered body. It uses the operation's existing atomic acknowledgment directory.
    pub fn record_conflict(
        &self,
        identity: &str,
        offered_hash: &str,
        acknowledgment: &Acknowledgment,
    ) -> Result<()> {
        let prepared = self.required(identity)?;
        validate_conflict(&prepared, offered_hash, acknowledgment)?;
        if let Some(old) = self.conflict_acknowledgment(identity)? {
            if encoded_hash(&old)? != encoded_hash(acknowledgment)? {
                return invalid("conflicting acknowledgment evidence changed");
            }
            if prepared.phase.is_pending() {
                self.finish_acknowledgment(
                    identity,
                    &Phase::Conflicted {
                        offered: offered_hash.into(),
                    },
                    acknowledgment,
                )?;
            }
            return Ok(());
        }
        if !prepared.phase.is_pending() {
            return invalid("a finished operation cannot change its outcome");
        }
        let phase = Phase::Conflicted {
            offered: offered_hash.into(),
        };
        let directory = self.operation(identity)?;
        let temporary = tempfile::Builder::new()
            .prefix(".ack-preparing-")
            .tempdir_in(&directory)?;
        let payload = write_encoded_json(&temporary.path().join("payload.zst"), acknowledgment)?;
        write_checked(
            &temporary.path().join("acknowledgment.json"),
            &StoredAcknowledgment {
                payload,
                phase: phase.clone(),
            },
        )?;
        sync(temporary.path())?;
        boundary("ack_synced");
        fs::rename(temporary.path(), directory.join("ack"))?;
        sync(&directory)?;
        boundary("ack_files_published");
        self.finish_acknowledgment(identity, &phase, acknowledgment)?;
        boundary("ack_published");
        Ok(())
    }

    fn finish_acknowledgment(
        &self,
        identity: &str,
        phase: &Phase,
        acknowledgment: &Acknowledgment,
    ) -> Result<()> {
        let admitted = acknowledgment
            .admitted()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        self.queue.acknowledge(
            identity,
            phase,
            &admitted,
            directory_bytes(&self.operation(identity)?.join("ack"))?,
        )?;
        self.wake();
        Ok(())
    }

    pub fn conflict_acknowledgment(&self, identity: &str) -> Result<Option<Acknowledgment>> {
        let Some(summary) = self.stored_acknowledgment(identity)? else {
            return Ok(None);
        };
        let Phase::Conflicted { offered } = summary.phase else {
            return Ok(None);
        };
        let prepared = self.required(identity)?;
        let acknowledgment: Acknowledgment = decode_json(verified_encoded(
            &self.operation(identity)?.join("ack/payload.zst"),
            summary.payload.descriptor()?,
        )?)?;
        validate_conflict(&prepared, &offered, &acknowledgment)?;
        Ok(Some(acknowledgment))
    }

    /// The caller supplies only a head verified after publication acknowledgments
    /// in this scope. Missing head evidence retains grouped source capture.
    ///
    /// `applied` answers whether `index.db` holds a fact for one Sinex event.
    /// An operation whose events were admitted is removed only when every one of
    /// them is in `index.db`: the applied position alone can pass positions the
    /// confirmed stream lost in a host crash, and removing then would drop the
    /// only local copy of writes the index never received.
    pub fn remove_confirmed(
        &self,
        state: &ReplayState,
        applied: &dyn Fn(&str) -> Result<bool>,
    ) -> Result<()> {
        if state.applied_position.is_some_and(|position| position < 0)
            || state.head.is_some_and(|position| position < 0)
        {
            return invalid("confirmed position is negative");
        }
        let caught_up = state.scope.as_ref().is_some_and(|scope| !scope.is_empty())
            && state
                .applied_position
                .zip(state.head)
                .is_some_and(|(applied, head)| applied >= head);
        let mut after = None;
        loop {
            let candidates = self.queue.confirmed_after(
                state.applied_position.unwrap_or(0),
                caught_up,
                after,
                128,
            )?;
            if candidates.is_empty() {
                break;
            }
            after = candidates.last().map(|candidate| candidate.entry.order);
            for candidate in candidates {
                if candidate.material {
                    continue;
                }
                if candidate
                    .admitted
                    .iter()
                    .map(|event| applied(event))
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .any(|present| !present)
                {
                    continue;
                }
                self.remove(&candidate.entry.identity)?;
            }
        }
        Ok(())
    }

    pub fn counts(&self) -> Result<(u64, u64, u64)> {
        Ok(self.queue.counts()?)
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
        encoded_body: Option<Vec<polylogue_store::primary::PreparedBody>>,
    ) -> Result<Prepared> {
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
                if previous.body().map(Body::identity) != Some(vec![(blob.as_str(), length)]) {
                    return invalid("prepared identity changed immutable body contents");
                }
            }
            if let Some(encoded) = encoded_body {
                let mut offered = Vec::with_capacity(encoded.len());
                for part in encoded {
                    let descriptor = part.descriptor;
                    let _validated = hashed_blobs::Prepared::from_encoded(
                        part.file,
                        descriptor.clone(),
                        None,
                        ReadLimits::default(),
                    )?;
                    offered.push((descriptor.hash.to_string(), descriptor.decoded_len));
                }
                if previous.body().map(Body::identity)
                    != Some(
                        offered
                            .iter()
                            .map(|(blob, length)| (blob.as_str(), *length))
                            .collect(),
                    )
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
            let body = match encoded_body {
                Some(mut encoded) if encoded.len() == 1 => Body::Whole(write_prepared(
                    &temporary.path().join("body.zst"),
                    encoded.pop().expect("one part"),
                )?),
                Some(encoded) => Body::Chunks(
                    encoded
                        .into_iter()
                        .enumerate()
                        .map(|(index, part)| {
                            write_prepared(
                                &temporary.path().join(format!("body-{index}.zst")),
                                part,
                            )
                        })
                        .collect::<Result<_>>()?,
                ),
                None => {
                    let path = temporary.path().join("body.zst");
                    let preparation = Preparation::new(temporary.path())?;
                    let encoded = preparation.plain_into(
                        create_payload(&path)?,
                        body.take().expect("decoded body"),
                        3,
                    )?;
                    let descriptor = encoded.descriptor();
                    encoded.into_file().sync_all()?;
                    Body::Whole(Bytes::new(&descriptor))
                }
            };
            // One encoding is the material's bytes; chunks are the ones the
            // frame names, in order.
            let matches = match &body {
                Body::Whole(bytes) => {
                    bytes.blob == descriptor.blob
                        && bytes.length == descriptor.length
                        && descriptor.chunks.is_empty()
                }
                Body::Chunks(chunks) => {
                    chunks.len() > 1
                        && chunks.len() == descriptor.chunks.len()
                        && chunks.iter().map(|bytes| bytes.length).sum::<u64>() == descriptor.length
                        && chunks.iter().zip(&descriptor.chunks).all(|(bytes, chunk)| {
                            bytes.blob == chunk.blob && bytes.length == chunk.length
                        })
                }
            };
            if !matches {
                return invalid("captured bytes differ from their prepared material");
            }
            boundary("body_synced");
            Some(body)
        } else {
            None
        };
        let group = captures.as_ref().map(|(_, captures_hash)| Group {
            captures_hash: captures_hash.clone(),
        });
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
            identity: identity.clone(),
            frame: frame_bytes,
            payload,
        };
        sync(temporary.path())?;
        boundary("operation_synced");
        if prepared.group().is_some() {
            boundary("group_synced");
        }
        let directory = format!("operations/{}", hash(identity.as_bytes()));
        let destination = self.root.join(&directory);
        if destination.exists() {
            fs::remove_dir_all(&destination)?;
        }
        fs::rename(temporary.path(), &destination)?;
        sync(&self.root.join("operations"))?;
        let mut operation = operation(&prepared, directory, &frame)?;
        operation.stored_bytes = directory_bytes(&destination)?;
        let members = captures
            .map(|(members, _)| members.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
            .unwrap_or_default();
        let entry = self.queue.prepare(
            &operation,
            prepared.group().map(|group| group.captures_hash.as_str()),
            &members,
        )?;
        boundary("operation_published");
        if prepared.group().is_some() {
            boundary("group_published");
            boundary("group_members_published");
        }
        self.wake();
        self.prepared(entry)
    }

    fn remove(&self, identity: &str) -> Result<()> {
        let paths = self.queue.remove(identity)?;
        boundary("operation_removed");
        let mut parents = std::collections::HashSet::new();
        for relative in paths {
            let path = self.root.join(relative);
            if path.exists() {
                fs::remove_dir_all(&path)?;
                parents.insert(path.parent().expect("operation parent").to_owned());
            }
            boundary("group_member_removed");
        }
        for parent in parents {
            if parent.exists() {
                sync(&parent)?;
            }
        }
        Ok(())
    }
}

/// Removes the `.preparing-*` and `.ack-preparing-*` directories in `directory`.
fn refuse_previous_operation(directory: &Path) -> Result<()> {
    if directory.join("frame.json").try_exists()? || directory.join("metadata.json").try_exists()? {
        return invalid(
            "producer spool contains an earlier pending payload format; preserve it before an explicit transition",
        );
    }
    Ok(())
}

fn remove_staging_database(directory: &Path) -> Result<()> {
    // Every accepted member is already in the main queue transaction. The
    // closed private database is temporary metadata, not another authority.
    for name in ["queue.db", "queue.db-wal", "queue.db-shm"] {
        match fs::remove_file(directory.join(name)) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    sync(directory)
}

fn remove_dot_entries(directory: &Path) -> Result<()> {
    let mut removed = false;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let unpublished = entry.file_name().to_str().is_some_and(|name| {
            name.starts_with(".preparing-") || name.starts_with(".ack-preparing-")
        });
        if unpublished && entry.file_type()?.is_dir() {
            fs::remove_dir_all(entry.path())?;
            removed = true;
        }
    }
    if removed {
        sync(directory)?;
    }
    Ok(())
}

fn empty_operation(
    identity: String,
    directory: String,
    metadata: Vec<u8>,
    kind: queue::Kind,
) -> queue::Operation {
    queue::Operation {
        identity,
        directory,
        metadata,
        kind,
        material: None,
        previous_material: None,
        logical_name: None,
        acquisition: None,
        ordinal: None,
        unresolved: false,
        source: None,
        raw: false,
        raw_generation: None,
        raw_predecessor: None,
        raw_root: false,
        reading_version: None,
        input: None,
        format: None,
        export_keys: Vec::new(),
        stored_bytes: 0,
        completion: None,
        captured: false,
        blob: None,
    }
}

fn operation(record: &Record, directory: String, frame: &Ingress) -> Result<queue::Operation> {
    let kind = match &record.payload {
        Payload::Material(material) => {
            if material.request {
                queue::Kind::Request
            } else {
                queue::Kind::Material
            }
        }
        Payload::Events { .. } => queue::Kind::Events,
        Payload::ReplaySequence { .. } => queue::Kind::Replay,
        Payload::Signals => queue::Kind::Signals,
        Payload::RegisterKinds => queue::Kind::Registration,
        Payload::Chunk => queue::Kind::Chunk,
        Payload::LargeFrame => queue::Kind::LargeFrame,
    };
    let mut operation = empty_operation(
        record.identity.clone(),
        directory,
        serde_json::to_vec(record)?,
        kind,
    );
    operation.material = record.material().map(|material| material.id.clone());
    if let Ingress::Material { material, .. } = frame {
        operation.completion = match material.parser.as_str() {
            "polylogue-request" if record.identity.starts_with("request:") => {
                Some(queue::Completion::Request)
            }
            "polylogue-products" | "polylogue-request-restore" if record.capture().is_none() => {
                Some(queue::Completion::Observation)
            }
            _ if record.capture().is_none()
                && material
                    .manifest
                    .details
                    .contains_key("polylogue_replacement") =>
            {
                Some(queue::Completion::Reading)
            }
            _ => None,
        };
        operation.previous_material = material.previous_part.as_ref().map(ToString::to_string);
    }
    operation.logical_name = record.logical_name().map(str::to_owned);
    operation.reading_version = record
        .control()
        .map(|control| control.reading_version.clone());
    operation.export_keys = record
        .exports()
        .iter()
        .map(|binding| binding.key.identity().to_string())
        .collect();
    if let Some(binding) = record.exports().first() {
        operation.input = Some(binding.key.input.id.clone());
        operation.format = Some(binding.key.input.format.clone());
    }
    operation.captured = record.capture().is_some();
    if let Some(Capture::Blob { hash, .. }) = record.capture() {
        operation.blob = Some(hash.to_string());
    }
    if let Some(Capture::Input(input)) = record.capture() {
        operation.input = Some(input.rows.input.clone());
        operation.format = Some(input.rows.format.clone());
        operation.source = input
            .basis
            .as_ref()
            .map(|basis| serde_json::to_vec(&basis.source))
            .transpose()?;
        operation.raw = input.raw_range.is_some();
        operation.raw_generation = if operation.raw {
            input
                .generation
                .as_ref()
                .map(serde_json::to_vec)
                .transpose()?
        } else {
            None
        };
        operation.raw_predecessor = input.raw_predecessor.clone();
        operation.raw_root = input.raw_predecessor.is_none()
            && input
                .raw_range
                .zip(input.generation.as_ref())
                .is_some_and(|(range, generation)| range.start == generation.window().start);
    }
    Ok(operation)
}

fn directory_bytes(directory: &Path) -> Result<u64> {
    fs::read_dir(directory)?.try_fold(0u64, |total, entry| {
        let entry = entry?;
        Ok(total.saturating_add(if entry.file_type()?.is_file() {
            entry.metadata()?.len()
        } else {
            0
        }))
    })
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
                route: input.route,
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
    chunks: Vec<sinex_protocol::MaterialChunk>,
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
        chunks: material.chunks.clone(),
        capture,
        named: matches!(parser, "polylogue-products" | "polylogue-request-restore")
            && name.ends_with(":observation"),
        request: matches!(parser, "polylogue-request" | "polylogue-request-restore"),
    }))
}

/// Checks an acknowledgment against the operation it answers: the identity,
/// the result its kind of frame can have, and the values the prepared frame
/// fixes. Serde has already refused every reply that is not one result.
fn validate_ack(prepared: &Prepared, acknowledgment: &Acknowledgment) -> Result<()> {
    if acknowledgment.frame != prepared.identity {
        return invalid("acknowledgment names another prepared operation");
    }
    if acknowledgment
        .commit_seq()
        .is_some_and(|position| position < 0)
    {
        return invalid("acknowledgment has a negative confirmed position");
    }
    match (&acknowledgment.result, &prepared.payload) {
        (FrameResult::Refused { .. }, _) => Ok(()),
        (FrameResult::KindsRegistered { .. }, Payload::RegisterKinds) => {
            exact_frame(prepared, acknowledgment)
        }
        (FrameResult::KindsRegistered { .. }, _) => {
            invalid("event acknowledgment claims kind registration")
        }
        // Sinex records a registration that declares nothing as an empty
        // admission.
        (
            FrameResult::Admitted {
                material: None,
                material_extents,
                admitted,
                counts,
                commit_seq: None,
            },
            Payload::RegisterKinds,
        ) if material_extents.is_empty()
            && admitted.is_empty()
            && *counts == sinex_model::AdmissionCounts::default() =>
        {
            exact_frame(prepared, acknowledgment)
        }
        (_, Payload::RegisterKinds) => {
            invalid("kind registration acknowledgment claims an event admission outcome")
        }
        (FrameResult::Replayed { commit_seq, .. }, Payload::ReplaySequence { .. }) => {
            if commit_seq.is_none() {
                exact_frame(prepared, acknowledgment)?;
                if !acknowledgment.admitted().is_empty() {
                    return invalid("newly admitted or replaced events need a confirmed position");
                }
            }
            Ok(())
        }
        (FrameResult::Replayed { .. }, _) | (_, Payload::ReplaySequence { .. }) => {
            invalid("acknowledgment answers another kind of operation")
        }
        (
            FrameResult::Admitted {
                material,
                material_extents,
                admitted,
                counts,
                commit_seq,
            },
            payload,
        ) => {
            if let Some(expected) = prepared.material()
                && material.as_ref().map(ToString::to_string).as_deref()
                    != Some(expected.id.as_str())
            {
                return invalid("material acknowledgment has another material identity");
            }
            // An absent position is an explicit owner outcome, not an empty
            // or partial acknowledgment.
            if commit_seq.is_none() && matches!(payload, Payload::Events { .. }) {
                exact_frame(prepared, acknowledgment)?;
                if !admitted.is_empty() || material.is_some() || !material_extents.is_empty() {
                    return invalid(
                        "unpositioned acknowledgment lacks exact durable frame evidence",
                    );
                }
                if counts.admitted != 0 || counts.replaced != 0 {
                    return invalid("newly admitted or replaced events need a confirmed position");
                }
            }
            Ok(())
        }
    }
}

/// The acknowledgment is the owner's durable record of this exact frame body.
fn exact_frame(prepared: &Prepared, acknowledgment: &Acknowledgment) -> Result<()> {
    if acknowledgment.number <= 0
        || acknowledgment.content_hash.as_deref() != Some(prepared.frame.blob.as_str())
    {
        return invalid("unpositioned acknowledgment lacks exact durable frame evidence");
    }
    Ok(())
}

fn validate_conflict(
    prepared: &Prepared,
    offered: &str,
    acknowledgment: &Acknowledgment,
) -> Result<()> {
    if acknowledgment.frame != prepared.identity
        || offered != prepared.frame.blob
        || acknowledgment.number <= 0
        || acknowledgment.content_hash.as_deref() == Some(offered)
    {
        return invalid("conflict evidence does not bind this exact offered operation");
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

//! Claude point-state adapter over an actual pinned SQLite predecessor. This
//! module interprets no provider records and performs no writes. Its counters
//! bound cumulative encoded cell work, not decoded allocation overhead or RSS.
use std::cell::{Cell, RefCell};
use std::io::{self, Write};

use polylogue_model::{Event, Link, MessageId, SessionId};
use polylogue_sources::claude_code::keyed::{
    BackgroundAddress, BodyPayload, Contribution, Header, LinkKey, PendingMessage, ReadView,
    RecordAddress, ResponseCell, ResponseKey, SessionCell,
};
use polylogue_sources::claude_code::{PARSER_VERSION, StepOutput};
use polylogue_sources::frame::Counts;
use polylogue_store::Error as StoreError;
use polylogue_store::parser_state::{Budget, CellChange, Head, Snapshot};
type Result<T> = std::result::Result<T, StoreError>;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub(crate) const HEADER: &str = "claude.header.v1";
pub(crate) const SESSION: &str = "claude.session.v1";
pub(crate) const RESPONSE: &str = "claude.response.v1";
const SIDE: &str = "claude.side.v1";
const LINK: &str = "claude.link.v1";
const BACKGROUND: &str = "claude.background.v1";
const OCCURRENCE: &str = "claude.occurrence.v1";
const RECORD: &str = "claude.record.v1";
pub(crate) const CONTRIBUTION: &str = "claude.contribution.v1";
const BODY: &str = "claude.body.v1";
pub(crate) const PENDING: &str = "claude.pending.v1";
pub(crate) const EFFECTS: &str = "claude.effects.v1";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Point attempts, including absent cells and a rejected attempt.
    pub cells: usize,
    /// Admitted namespace/key bytes and fetched value bytes over this attempt.
    pub encoded_bytes: usize,
}

/// Opaque keyset position for scheduling retained obligations. Its byte order
/// has no authority over source Message order or candidate selection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingCursor(Vec<u8>);

pub struct View<'a> {
    snapshot: &'a Snapshot,
    bodies: Option<&'a [polylogue_store::claude_bodies::Companion]>,
    budget: Budget,
    usage: Cell<Usage>,
    error: RefCell<Option<StoreError>>,
}
impl<'a> View<'a> {
    pub fn new(snapshot: &'a Snapshot, budget: Budget) -> Result<Self> {
        if snapshot.head.parser_version != PARSER_VERSION {
            return Err(StoreError::Invariant(
                "Claude state parser version differs from its snapshot".into(),
            ));
        }
        Ok(Self {
            snapshot,
            bodies: None,
            budget,
            usage: Cell::new(Usage::default()),
            error: RefCell::new(None),
        })
    }
    pub fn with_bodies(mut self, bodies: &'a [polylogue_store::claude_bodies::Companion]) -> Self {
        self.bodies = Some(bodies);
        self
    }
    /// Recover the original structured failure after the provider's existing
    /// String callback boundary. The owner must inspect this before publishing.
    pub fn take_error(&self) -> Option<StoreError> {
        self.error.borrow_mut().take()
    }
    fn callback<T>(&self, read: impl FnOnce() -> Result<T>) -> std::result::Result<T, String> {
        if let Some(error) = self.error.borrow().as_ref() {
            return Err(error.to_string());
        }
        read().map_err(|error| {
            let message = error.to_string();
            *self.error.borrow_mut() = Some(error);
            message
        })
    }
    pub fn pending(&self, message: &MessageId) -> Result<Option<PendingMessage>> {
        self.get(PENDING, &(message.session().as_str(), message.as_str()))
    }
    /// Read at most one obligation without collecting a session's responses.
    /// The returned cursor belongs to this pinned snapshot; callers restart at
    /// the beginning after a new predecessor and use source receipts for replay.
    pub fn next_pending(
        &self,
        after: Option<&PendingCursor>,
    ) -> Result<Option<(PendingCursor, PendingMessage)>> {
        let key = after.map_or(&[][..], |cursor| cursor.0.as_slice());
        let mut usage = self.usage.get();
        usage.cells = usage.cells.saturating_add(1);
        self.usage.set(usage);
        if usage.cells > self.budget.cells {
            return Err(admission("Claude state point attempts", usage.cells));
        }
        self.charge(PENDING.len().saturating_add(key.len()))?;
        let row = self
            .snapshot
            .neighbor(PENDING, key, true, self.remaining())
            .map_err(|error| match error {
                StoreError::ParserAdmission { required, .. } => StoreError::ParserAdmission {
                    resource: "Claude state read encoded bytes",
                    required: (self.usage.get().encoded_bytes as u64).saturating_add(required),
                },
                error => error,
            })?;
        let Some((key, bytes)) = row else {
            return Ok(None);
        };
        self.charge(key.len().saturating_add(bytes.len()))?;
        let pending: PendingMessage = serde_json::from_slice(&bytes).map_err(|error| {
            StoreError::Invariant(format!("invalid Claude pending state: {error}"))
        })?;
        let identity: (String, String) = serde_json::from_slice(&key).map_err(|error| {
            StoreError::Invariant(format!("invalid Claude pending key: {error}"))
        })?;
        if identity.0 != pending.message.session().as_str()
            || identity.1 != pending.message.as_str()
            || pending.session_native.is_empty()
        {
            return Err(StoreError::Invariant(
                "Claude pending key/Message identity mismatch".into(),
            ));
        }
        Ok(Some((PendingCursor(key), pending)))
    }
    pub fn effects(&self, step: i64) -> Result<Option<Effects>> {
        self.get(EFFECTS, &step)
    }
    pub fn usage(&self) -> Usage {
        self.usage.get()
    }
    fn remaining(&self) -> usize {
        self.budget
            .encoded_bytes
            .saturating_sub(self.usage.get().encoded_bytes)
    }
    fn charge(&self, bytes: usize) -> Result<()> {
        let mut usage = self.usage.get();
        let total = usage.encoded_bytes.saturating_add(bytes);
        if total > self.budget.encoded_bytes {
            return Err(admission("Claude state read encoded bytes", total));
        }
        usage.encoded_bytes = total;
        self.usage.set(usage);
        Ok(())
    }
    fn get<T: DeserializeOwned, K: Serialize + ?Sized>(
        &self,
        namespace: &str,
        key: &K,
    ) -> Result<Option<T>> {
        let mut usage = self.usage.get();
        usage.cells = usage.cells.saturating_add(1);
        self.usage.set(usage);
        if usage.cells > self.budget.cells {
            return Err(admission("Claude state point attempts", usage.cells));
        }
        self.charge(namespace.len())?;
        let key = encode(key, self.remaining(), self.usage.get().encoded_bytes)?;
        self.charge(key.len())?;
        let bytes = self
            .snapshot
            .cell(namespace, &key, self.remaining())
            .map_err(|e| match e {
                StoreError::ParserAdmission { required, .. } => StoreError::ParserAdmission {
                    resource: "Claude state read encoded bytes",
                    required: (self.usage.get().encoded_bytes as u64).saturating_add(required),
                },
                e => e,
            })?;
        match bytes {
            None => Ok(None),
            Some(bytes) => {
                self.charge(bytes.len())?;
                serde_json::from_slice(&bytes)
                    .map(Some)
                    .map_err(|e| StoreError::Invariant(format!("invalid {namespace} state: {e}")))
            }
        }
    }
}
impl View<'_> {
    fn read_header(&self) -> Result<Header> {
        let header = match self.get(HEADER, &())? {
            Some(header) => header,
            None if self.snapshot.head.token.step == 0 && self.snapshot.head.parsed_upto == 0 => {
                Header::default()
            }
            None => {
                return Err(StoreError::Invariant(
                    "committed Claude state has no header".into(),
                ));
            }
        };
        if header.parsed_upto != self.snapshot.head.parsed_upto {
            return Err(StoreError::Invariant(
                "Claude header differs from committed parse position".into(),
            ));
        }
        Ok(header)
    }
}
impl ReadView for View<'_> {
    type Error = String;
    fn background(&self, task: &str) -> std::result::Result<Option<BackgroundAddress>, String> {
        self.callback(|| self.get(BACKGROUND, task))
    }
    fn occurrence(&self, hash: &str) -> std::result::Result<u64, String> {
        self.callback(|| Ok(self.get(OCCURRENCE, hash)?.unwrap_or(0)))
    }
    fn header(&self) -> std::result::Result<Header, String> {
        self.callback(|| self.read_header())
    }
    fn session(&self, native: &str) -> std::result::Result<Option<SessionCell>, String> {
        self.callback(|| self.get(SESSION, native))
    }
    fn response(
        &self,
        session: &str,
        key: &ResponseKey,
    ) -> std::result::Result<Option<ResponseCell>, String> {
        self.callback(|| self.get(RESPONSE, &(session, key.role, &key.response, &key.request)))
    }
    fn record(
        &self,
        session: &str,
        uuid: &str,
    ) -> std::result::Result<Option<RecordAddress>, String> {
        self.callback(|| self.get(RECORD, &(session, uuid)))
    }
    fn contribution(
        &self,
        message: &MessageId,
        item: u64,
    ) -> std::result::Result<Option<Contribution>, String> {
        self.callback(|| {
            self.get(
                CONTRIBUTION,
                &(message.session().as_str(), message.as_str(), item),
            )
        })
    }
    fn side(&self, uuid: &str) -> std::result::Result<Option<String>, String> {
        self.callback(|| self.get(SIDE, uuid))
    }
    fn has_link(&self, key: &LinkKey) -> std::result::Result<bool, String> {
        self.callback(|| {
            Ok(self
                .get::<bool, _>(LINK, &(&key.parent, &key.child, key.kind, key.how))?
                .unwrap_or(false))
        })
    }
    fn body(&self, path: &str) -> std::result::Result<Option<BodyPayload>, String> {
        if let Some(bodies) = self.bodies {
            return Ok(bodies
                .iter()
                .find(|body| body.path == path)
                .map(|body| BodyPayload {
                    blob: body.blob,
                    length: body.length,
                }));
        }
        self.callback(|| self.get(BODY, path))
    }
}

/// Disposable output receipt, scoped by parser token generation and step. The
/// canonical owner acknowledges it only with the corresponding actual effects.
/// This is not a public Event ID or proof of canonical Message completion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Effects {
    pub events: Vec<(SessionId, Event)>,
    pub links: Vec<Link>,
    pub pending_messages: Vec<PendingMessage>,
    pub ignored: std::collections::BTreeMap<String, u64>,
    pub unrecognized: Vec<(String, u64)>,
    pub undecodable: Vec<u64>,
    #[serde(default)]
    pub text_repairs: Vec<polylogue_model::TextRepair>,
}
#[derive(Serialize)]
struct EffectsRef<'a> {
    events: &'a [(SessionId, Event)],
    links: &'a [Link],
    pending_messages: &'a [PendingMessage],
    ignored: &'a std::collections::BTreeMap<&'static str, u64>,
    unrecognized: &'a [(&'static str, u64)],
    undecodable: &'a [u64],
    text_repairs: Vec<polylogue_model::TextRepair>,
}

fn admission(resource: &'static str, required: usize) -> StoreError {
    StoreError::ParserAdmission {
        resource,
        required: required as u64,
    }
}

/// A logical encoded-length admission before each append/reservation. Allocation
/// can fail and allocator rounding is not bounded by this interface. `required`
/// on a refused serialization is a lower bound, since encoding stops immediately.
struct Limited {
    bytes: Vec<u8>,
    limit: usize,
    required: Option<usize>,
    allocation_failed: bool,
}
impl Write for Limited {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let required = self.bytes.len().saturating_add(bytes.len());
        if required > self.limit {
            self.required = Some(required);
            return Err(io::Error::other(
                "Claude state encoding allowance exhausted",
            ));
        }
        if self
            .bytes
            .try_reserve_exact(required - self.bytes.len())
            .is_err()
        {
            self.allocation_failed = true;
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "Claude state encoding allocation failed",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn encode<T: Serialize + ?Sized>(value: &T, limit: usize, already: usize) -> Result<Vec<u8>> {
    let mut writer = Limited {
        bytes: Vec::new(),
        limit,
        required: None,
        allocation_failed: false,
    };
    if let Err(e) = serde_json::to_writer(&mut writer, value) {
        if let Some(required) = writer.required {
            return Err(admission(
                "Claude state encoded bytes",
                already.saturating_add(required),
            ));
        }
        if writer.allocation_failed {
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "Claude state encoding allocation failed",
            )
            .into());
        }
        return Err(StoreError::Invariant(format!(
            "cannot encode Claude state: {e}"
        )));
    }
    Ok(writer.bytes)
}

/// Encode outside the writer transaction. The framer supplies the complete-record
/// position. Mutating this attempt's header does not publish it. The caller drops
/// its View/Snapshot and submits these cells with output to the existing CAS
/// writer; this does not certify canonical row finalization or acquired EOF.
pub fn encode_step(
    output: &mut StepOutput,
    head: &Head,
    parsed_upto: u64,
    counts: &Counts,
    budget: Budget,
) -> Result<Vec<CellChange>> {
    if head.parser_version != PARSER_VERSION {
        return Err(StoreError::Invariant(
            "Claude encoder parser version differs from head".into(),
        ));
    }
    let next = head
        .token
        .step
        .checked_add(1)
        .ok_or_else(|| StoreError::Invariant("Claude receipt step exhausted".into()))?;
    let delta = &output.delta;
    let cells = 2usize
        .saturating_add(delta.sessions.len())
        .saturating_add(delta.responses.len())
        .saturating_add(delta.addresses.len())
        .saturating_add(delta.occurrences.len())
        .saturating_add(delta.backgrounds.len())
        .saturating_add(delta.contributions.len())
        .saturating_add(delta.sides.len())
        .saturating_add(delta.links.len())
        .saturating_add(delta.pending_messages.len());
    if cells > budget.cells {
        return Err(admission("Claude state delta cells", cells));
    }
    let mut encoder = Encoder {
        changes: Vec::new(),
        bytes: 0,
        budget,
    };
    output.header.parsed_upto = parsed_upto;
    encoder.put(HEADER, &(), Some(&output.header))?;
    for cell in &delta.sessions {
        encoder.put(SESSION, &cell.session.native_id, Some(cell))?;
    }
    for (session, key, cell) in &delta.responses {
        encoder.put(
            RESPONSE,
            &(session, key.role, &key.response, &key.request),
            Some(cell),
        )?;
    }
    for (task, address) in &delta.backgrounds {
        encoder.put(BACKGROUND, task, Some(address))?;
    }
    for (hash, count) in &delta.occurrences {
        encoder.put(OCCURRENCE, hash, Some(count))?;
    }
    for (session, uuid, cell) in &delta.addresses {
        encoder.put(RECORD, &(session, uuid), Some(cell))?;
    }
    for (message, item, cell) in &delta.contributions {
        encoder.put(
            CONTRIBUTION,
            &(message.session().as_str(), message.as_str(), item),
            Some(cell),
        )?;
    }
    for (uuid, session) in &delta.sides {
        encoder.put(SIDE, uuid, Some(session))?;
    }
    for key in &delta.links {
        encoder.put(
            LINK,
            &(&key.parent, &key.child, key.kind, key.how),
            Some(&true),
        )?;
    }
    for pending in &delta.pending_messages {
        encoder.put(
            PENDING,
            &(pending.message.session().as_str(), pending.message.as_str()),
            Some(pending),
        )?;
    }
    encoder.put(
        EFFECTS,
        &next,
        Some(&EffectsRef {
            events: &output.events,
            links: &output.links,
            pending_messages: &delta.pending_messages,
            ignored: &counts.ignored,
            unrecognized: &counts.unrecognized,
            undecodable: &counts.undecodable,
            text_repairs: counts.text_repairs.entries(),
        }),
    )?;
    Ok(encoder.changes)
}
/// Companion bytes are already retained by the source owner. The owner pins their
/// availability manifest in Basis, then stages these cells with the same barrier.
/// None removes availability; a failed companion read must never call this as None.
pub fn encode_body(
    path: &str,
    payload: Option<&BodyPayload>,
    budget: Budget,
) -> Result<CellChange> {
    let mut encoder = Encoder {
        changes: Vec::new(),
        bytes: 0,
        budget,
    };
    encoder.put(BODY, path, payload)?;
    encoder
        .changes
        .pop()
        .ok_or_else(|| StoreError::Invariant("missing encoded Claude body cell".into()))
}
struct Encoder {
    changes: Vec<CellChange>,
    bytes: usize,
    budget: Budget,
}
impl Encoder {
    fn put<K: Serialize + ?Sized, V: Serialize + ?Sized>(
        &mut self,
        namespace: &str,
        key: &K,
        value: Option<&V>,
    ) -> Result<()> {
        if self.changes.len() >= self.budget.cells {
            return Err(admission(
                "Claude state delta cells",
                self.changes.len().saturating_add(1),
            ));
        }
        let base = self.bytes.saturating_add(namespace.len());
        if base > self.budget.encoded_bytes {
            return Err(admission("Claude state encoded bytes", base));
        }
        let key = encode(key, self.budget.encoded_bytes - base, base)?;
        let base = base.saturating_add(key.len());
        let value = value
            .map(|value| encode(value, self.budget.encoded_bytes - base, base))
            .transpose()?;
        self.bytes = base.saturating_add(value.as_ref().map_or(0, Vec::len));
        self.changes.try_reserve_exact(1).map_err(|_| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "Claude state delta allocation failed",
            )
        })?;
        self.changes.push(CellChange {
            namespace: namespace.into(),
            key,
            value,
        });
        Ok(())
    }
}

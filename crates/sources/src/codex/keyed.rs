//! Disposable Codex state, independently addressable by provider-owned keys.
//!
//! A ReadView is pinned to one committed input/chain/parser basis by its caller.
//! The overlay contains only touched keys; finish returns mutations, not a
//! serialized prefix or materialized message suffix. Payload/fanout scheduling
//! and atomic application with normalized rows belong to the caller.
use super::origins::{Origins, RecordRange};
use super::*;
use std::ops::Bound::{Excluded, Unbounded};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Key {
    Message(usize),
    Block(usize, usize),
    Call(String),
    /// Reverse annotation ownership, preserving duplicate native call IDs.
    CallBlock(String, usize, usize),
    Reasoning(String),
    Running(String),
    Outcome(String),
    Continuation(String),
    PendingStdin(String),
    Wrapper(usize),
    WrapperCall(String, usize),
    Child(String, usize),
    ChildCount(String),
    Occurrence(String),
    Model(usize),
    Mutable(usize, String),
    Prompt,
    BlockOrder(usize),
    Evidence(usize, usize, usize),
    EvidenceHead(usize),
    EvidenceItem(usize, String),
    Context(usize),
    CallWitness(String),
    StatusWitness(String),
    Result(String),
    NativeItem(String, String),
    OutputChunk(usize, usize, usize),
    PendingResult(usize, usize),
    ContinuationTarget(String),
    Presence(usize, String),
    PresenceBlock(usize, usize),
    Operation(String),
    OperationBlock(usize, usize),
    /// Exact native constructor record, scoped by the full operation ID.
    UnifiedOperation(String),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Cell {
    UnifiedOperation {
        raw_hash: [u8; 32],
        source: String,
    },
    Operation(polylogue_model::tool_item::Operation),
    /// A Message with no blocks; blocks have separate keys. Public IDs can change.
    Header {
        message: Message,
        blocks: usize,
    },
    Block(polylogue_model::Block),
    Call {
        at: (usize, usize),
        tool: String,
        stdin_session: Option<String>,
    },
    Position(usize, usize),
    Outcome(String, Option<String>, Option<i64>),
    Text(String),
    Number(usize),
    Present,
    Evidence(EvidenceUnit),
    EvidenceHead(EvidenceHead),
    Witness(super::presence::Witness),
    Presence(super::presence::Group),
    Association {
        call_name: Option<Vec<u8>>,
        session: Option<Vec<u8>>,
        call_name_origins: Origins,
        session_origins: Origins,
    },
}
/// Raw canonical JSON, tagged by provider membership, never normalized content.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceUnit {
    pub tag: String,
    pub canonical: Vec<u8>,
    pub whole_record: bool,
    #[serde(default)]
    pub origins: Origins,
}
// Provenance is separate from version membership. Moving an identical unit
// never advances its revision or salts its digest with scheduling coordinates.
impl PartialEq for EvidenceUnit {
    fn eq(&self, other: &Self) -> bool {
        self.tag == other.tag
            && self.canonical == other.canonical
            && self.whole_record == other.whole_record
    }
}
impl Eq for EvidenceUnit {}
impl EvidenceUnit {
    /// Validate the provider-owned address/role and one admitted raw JSON unit.
    /// This borrows the bytes; it neither recanonicalizes nor hashes a DTO.
    pub fn validate_at(&self, key: &Key) -> Result<(), String> {
        let (tag, whole_allowed, whole_required) = match key {
            Key::Evidence(_, 0, slot) => match slot % 3 {
                0 => ("item", true, false),
                1 => ("association:call_name", false, false),
                _ => ("association:process_session", false, false),
            },
            Key::Evidence(_, 1, 0) => ("status", true, false),
            Key::Evidence(_, 1, 1) => ("context:source", false, false),
            Key::Evidence(_, 1, 2) => ("context:model", false, false),
            Key::Evidence(_, 1, 3) => ("context:echo", true, true),
            Key::Evidence(_, 1, 4) => ("usage", false, false),
            _ => return Err("unsupported Codex evidence address".into()),
        };
        if self.tag != tag
            || (self.whole_record && !whole_allowed)
            || (!self.whole_record && whole_required)
        {
            return Err(
                "Codex evidence role/tag/whole-record flag differs from its address".into(),
            );
        }
        serde_json::from_slice::<&serde_json::value::RawValue>(&self.canonical)
            .map_err(|e| format!("invalid Codex raw evidence JSON: {e}"))?;
        Ok(())
    }
}
/// Provider-local version guard. The owner additionally pins input chain,
/// parser/schema and committed interpretation predecessor during digest work.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceHead {
    pub items: usize,
    pub units: u64,
    pub revision: u64,
    pub incomplete: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingVersion {
    pub message: usize,
    pub evidence_revision: u64,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeConflict {
    pub message: usize,
    pub fields: Vec<String>,
}
/// Explicit cursor for the owner's scheduled version preparation. Each call
/// supplies at most one cell key. No parser finish scans a Message's members.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCursor {
    pub phase: usize,
    pub slot: usize,
}
impl EvidenceHead {
    pub fn validate(&self) -> Result<(), String> {
        let slots = self
            .items
            .checked_mul(3)
            .and_then(|n| n.checked_add(5))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or("Codex evidence slot count overflow")?;
        if self.incomplete
            || self.items == 0
            || self.units < self.items as u64
            || self.units > slots
            || self.revision < self.units
        {
            return Err(
                "Codex evidence head is incomplete or has invalid item/unit/revision counts".into(),
            );
        }
        Ok(())
    }
    /// Every effective item owns its primary unit; association/context slots
    /// are optional and the selected-unit count is checked after enumeration.
    pub fn required_slot(&self, key: &Key) -> bool {
        matches!(key, Key::Evidence(_, 0, slot) if slot % 3 == 0 && slot / 3 < self.items)
    }

    pub fn next_slot(&self, message: usize, cursor: &mut EvidenceCursor) -> Option<Key> {
        if cursor.phase == 0 {
            if cursor.slot < self.items.checked_mul(3)? {
                let key = Key::Evidence(message, 0, cursor.slot);
                cursor.slot += 1;
                return Some(key);
            }
            cursor.phase = 1;
            cursor.slot = 0;
        }
        if cursor.phase == 1 && cursor.slot < 5 {
            let key = Key::Evidence(message, 1, cursor.slot);
            cursor.slot += 1;
            return Some(key);
        }
        None
    }
}
/// Point and ordered-neighbor reads within the same Key variant. For example,
/// next(Mutable(...)) never returns a Prompt payload from the next family.
/// Implementations must not scan/materialize all prior cells to answer these
/// operations. Errors abort the attempted step. Namespace/basis pinning is the
/// caller's responsibility; key order is the derived Rust Ord, not JSON order.
pub trait ReadView {
    fn read(&self, key: &Key) -> Result<Option<Cell>, String>;
    fn next(&self, after: &Key) -> Result<Option<(Key, Cell)>, String>;
    fn previous(&self, before: &Key) -> Result<Option<(Key, Cell)>, String>;
}
impl<V: ReadView + ?Sized> ReadView for &V {
    fn read(&self, key: &Key) -> Result<Option<Cell>, String> {
        (**self).read(key)
    }
    fn next(&self, key: &Key) -> Result<Option<(Key, Cell)>, String> {
        (**self).next(key)
    }
    fn previous(&self, key: &Key) -> Result<Option<(Key, Cell)>, String> {
        (**self).previous(key)
    }
}
#[derive(Clone, Default)]
pub struct MemoryReadView {
    pub cells: BTreeMap<Key, Cell>,
}
impl Key {
    pub fn same_family(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}
impl ReadView for MemoryReadView {
    fn read(&self, k: &Key) -> Result<Option<Cell>, String> {
        Ok(self.cells.get(k).cloned())
    }
    fn next(&self, k: &Key) -> Result<Option<(Key, Cell)>, String> {
        Ok(self
            .cells
            .range((Excluded(k), Unbounded))
            .next()
            .filter(|(next, _)| next.same_family(k))
            .map(|(k, v)| (k.clone(), v.clone())))
    }
    fn previous(&self, k: &Key) -> Result<Option<(Key, Cell)>, String> {
        Ok(self
            .cells
            .range(..k)
            .next_back()
            .filter(|(next, _)| next.same_family(k))
            .map(|(k, v)| (k.clone(), v.clone())))
    }
}
impl MemoryReadView {
    pub fn apply(&mut self, delta: &Delta) {
        for (k, v) in &delta.cells {
            match v {
                Some(v) => {
                    self.cells.insert(k.clone(), v.clone());
                }
                None => {
                    self.cells.remove(k);
                }
            }
        }
    }
}
/// Fixed logical continuation. Growing maps, message bodies and prompt text are
/// cells, never embedded here. The session's models map is not accumulated here.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Resume {
    pub parsed_upto: u64,
    session: Option<Session>,
    legacy: bool,
    base: usize,
    messages: usize,
    open_request: Option<usize>,
    open_anchored: bool,
    output_since_open: bool,
    prev_total: Option<Value>,
    model: Option<String>,
    pending_prompt: Option<usize>,
    mode: SessionMode,
    next_wrapper: usize,
    normalization_incomplete: bool,
    has_source_context: bool,
    has_model_context: bool,
    #[serde(default)]
    turn: Option<String>,
}
#[derive(Clone, Debug)]
pub struct Delta {
    pub resume: Resume,
    pub cells: BTreeMap<Key, Option<Cell>>,
    pub events: Vec<Event>,
    pub links: Vec<Link>,
    pub reached_back: bool,
    pub pending_versions: Vec<PendingVersion>,
    pub conflicts: Vec<NativeConflict>,
    pub pending_results: Vec<(usize, usize)>,
}
/// The single provider interpretation core, over a pinned view and touched-key overlay.
/// This stage does not bound a single record's wrapper fanout or public-ID
/// reanchor projection; callers must not advertise a fully bounded owner yet.
pub struct KeyedRecords<V> {
    view: V,
    changes: BTreeMap<Key, Option<Cell>>,
    failure: Option<String>,
    limits: Limits,
    retained_bytes: usize,
    sizes: BTreeMap<Key, usize>,
    current_raw: Vec<u8>,
    current_origins: Origins,
    effective_origins: Origins,
    // Intrinsic field provenance is separate from volatile-free version bytes.
    current_time_raw: Option<Vec<u8>>,
    pending_versions: BTreeMap<usize, u64>,
    effective_item: Option<Value>,
    effective_raw: Vec<u8>,
    replacement_block: Option<usize>,
    pub(crate) conflicts: Vec<NativeConflict>,
    pending_results: BTreeSet<(usize, usize)>,
    pub(crate) s: Resume,
    pub(crate) reached_back: bool,
    pub(crate) events: Vec<Event>,
    pub(crate) links: Vec<Link>,
}
/// Admission for touched-cell count and encoded cell/key size. This does not
/// account for allocator capacity, scalar metadata, events or links; their
/// physical lifetime remains charged by the integration owner.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub changed_cells: usize,
    pub encoded_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            changed_cells: 65_536,
            encoded_bytes: 64 * 1024 * 1024,
        }
    }
}
impl Resume {
    pub fn normalization_incomplete(&self) -> bool {
        self.normalization_incomplete
    }
    pub fn session(&self) -> Option<&Session> {
        self.session.as_ref()
    }
    pub fn message_count(&self) -> usize {
        self.messages
    }
    pub fn settled_before(&self) -> usize {
        self.base
    }
}
impl Cell {
    /// Whether this decoded cell has the shape owned by its provider key.
    pub fn valid_for(&self, key: &Key) -> bool {
        self.fits(key)
    }
    fn fits(&self, k: &Key) -> bool {
        if let (Key::UnifiedOperation(_), Self::UnifiedOperation { source, .. }) = (k, self) {
            return unified_source(source);
        }
        if matches!(
            (k, self),
            (Key::Operation(_), Self::Operation(_)) | (Key::OperationBlock(_, _), Self::Text(_))
        ) {
            return true;
        }
        if let (Key::Presence(_, key), Self::Witness(w)) = (k, self) {
            return super::presence::matches_message_key(key, w);
        }
        if let (Key::PresenceBlock(_, _), Self::Presence(group)) = (k, self) {
            return group.valid();
        }
        matches!((k,self),
        (Key::Message(_),Self::Header{message,..}) if message.blocks.is_empty())
            || matches!(
                (k, self),
                (Key::Block(_, _), Self::Block(_))
                    | (Key::Call(_), Self::Call { .. })
                    | (
                        Key::Reasoning(_)
                            | Key::Running(_)
                            | Key::PendingStdin(_)
                            | Key::Result(_)
                            | Key::EvidenceItem(_, _)
                            | Key::NativeItem(_, _)
                            | Key::ContinuationTarget(_),
                        Self::Position(_, _)
                    )
                    | (Key::Outcome(_), Self::Outcome(_, _, _))
                    | (
                        Key::Continuation(_)
                            | Key::CallBlock(_, _, _)
                            | Key::WrapperCall(_, _)
                            | Key::Model(_)
                            | Key::Mutable(_, _)
                            | Key::PendingResult(_, _),
                        Self::Present
                    )
                    | (
                        Key::Wrapper(_)
                            | Key::Child(_, _)
                            | Key::Prompt
                            | Key::OutputChunk(_, _, _),
                        Self::Text(_)
                    )
                    | (Key::ChildCount(_) | Key::Occurrence(_), Self::Number(_))
                    | (Key::BlockOrder(_), Self::Position(_, _))
                    | (
                        Key::Evidence(_, _, _) | Key::Context(_) | Key::StatusWitness(_),
                        Self::Evidence(_)
                    )
                    | (Key::EvidenceHead(_), Self::EvidenceHead(_))
                    | (Key::CallWitness(_), Self::Association { .. })
            )
    }
}
fn unified_source(source: &str) -> bool {
    matches!(
        source,
        "unified_exec_startup"
            | "unifiedExecStartup"
            | "unified_exec_interaction"
            | "unifiedExecInteraction"
    )
}
fn valid_command_duration(item: &Value) -> bool {
    if item
        .get("duration_ms")
        .is_some_and(|v| !v.is_null() && v.as_u64().is_none())
    {
        return false;
    }
    item.get("duration").is_none_or(|v| {
        v.is_null() || {
            v.get("secs")
                .and_then(Value::as_u64)
                .zip(v.get("nanos").and_then(Value::as_u64))
                .is_some_and(|(seconds, nanos)| {
                    nanos < 1_000_000_000
                        && seconds
                            .checked_mul(1000)
                            .and_then(|n| n.checked_add(nanos / 1_000_000))
                            .is_some()
                })
        }
    })
}
impl<V: ReadView> KeyedRecords<V> {
    pub fn new(view: V, resume: Resume) -> Self {
        Self::with_limits(view, resume, Limits::default())
    }
    pub fn with_limits(view: V, resume: Resume, limits: Limits) -> Self {
        Self {
            view,
            changes: BTreeMap::new(),
            failure: None,
            limits,
            retained_bytes: 0,
            sizes: BTreeMap::new(),
            current_raw: Vec::new(),
            current_origins: Origins::Unknown,
            effective_origins: Origins::Unknown,
            current_time_raw: None,
            pending_versions: BTreeMap::new(),
            effective_item: None,
            effective_raw: Vec::new(),
            replacement_block: None,
            conflicts: Vec::new(),
            pending_results: BTreeSet::new(),
            s: resume,
            reached_back: false,
            events: Vec::new(),
            links: Vec::new(),
        }
    }
    fn missing(&mut self, what: &str) {
        if self.failure.is_none() {
            self.failure = Some(format!("missing Codex {what} cell"));
        }
    }
    fn get(&mut self, k: Key) -> Option<Cell> {
        if self.failure.is_some() {
            return None;
        }
        if let Some(v) = self.changes.get(&k) {
            return v.clone();
        }
        match self.view.read(&k) {
            Ok(Some(v)) if !v.fits(&k) => {
                self.failure = Some(format!("invalid Codex cell for {k:?}"));
                None
            }
            Ok(v) => v,
            Err(e) => {
                self.failure = Some(e);
                None
            }
        }
    }
    fn change(&mut self, k: Key, v: Option<Cell>) {
        if self.failure.is_some() {
            return;
        }
        if let Some(v) = &v {
            if !v.fits(&k) {
                self.failure = Some(format!("invalid Codex cell for {k:?}"));
                return;
            }
        }
        // Count encoded payload without allocating another serialization buffer.
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0 = self.0.saturating_add(b.len());
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut bytes = Count(0);
        if let Err(e) = serde_json::to_writer(&mut bytes, &(&k, &v)) {
            self.failure = Some(e.to_string());
            return;
        }
        let size = bytes
            .0
            .saturating_add(std::mem::size_of::<(Key, Option<Cell>)>());
        let total = self
            .retained_bytes
            .saturating_sub(*self.sizes.get(&k).unwrap_or(&0))
            .saturating_add(size);
        if total > self.limits.encoded_bytes
            || (!self.changes.contains_key(&k) && self.changes.len() >= self.limits.changed_cells)
        {
            self.failure =
                Some("Codex state overlay admission exceeded; attempted step remains owed".into());
            return;
        }
        self.retained_bytes = total;
        self.sizes.insert(k.clone(), size);
        self.changes.insert(k, v);
    }
    fn put(&mut self, k: Key, v: Cell) {
        self.change(k, Some(v));
    }
    fn remove(&mut self, k: Key) -> Option<Cell> {
        let old = self.get(k.clone());
        self.change(k, None);
        old
    }
    fn neighbor(&mut self, mut key: Key, forward: bool) -> Option<(Key, Cell)> {
        if self.failure.is_some() {
            return None;
        }
        loop {
            let base = if forward {
                self.view.next(&key)
            } else {
                self.view.previous(&key)
            };
            let base = match base {
                Ok(v) => v,
                Err(e) => {
                    self.failure = Some(e);
                    return None;
                }
            };
            if let Some((k, v)) = &base {
                if (forward && k <= &key)
                    || (!forward && k >= &key)
                    || !key.same_family(k)
                    || !v.fits(k)
                {
                    self.failure = Some("invalid Codex ordered state read".into());
                    return None;
                }
            }
            let changed = if forward {
                self.changes.range((Excluded(&key), Unbounded)).next()
            } else {
                self.changes.range(..&key).next_back()
            }
            .filter(|(next, _)| next.same_family(&key))
            .map(|(k, v)| (k.clone(), v.clone()));
            let candidate = match (base, changed) {
                (None, None) => return None,
                (Some((k, v)), None) => return Some((k, v)),
                (None, Some(v)) => v,
                (Some((bk, bv)), Some((ck, cv))) => {
                    if (forward && bk < ck) || (!forward && bk > ck) {
                        return Some((bk, bv));
                    } else {
                        (ck, cv)
                    }
                }
            };
            match candidate {
                (k, Some(v)) => return Some((k, v)),
                (k, None) => key = k,
            }
        }
    }
    fn header(&mut self, i: usize) -> Option<(Message, usize)> {
        match self.get(Key::Message(i)) {
            Some(Cell::Header { message, blocks }) => Some((message, blocks)),
            _ => {
                if self.failure.is_none() && i < self.s.messages {
                    self.failure = Some(format!("missing Codex message header {i}"));
                }
                None
            }
        }
    }
    fn edit(&mut self, i: usize, f: impl FnOnce(&mut Message)) {
        if i < self.s.base {
            self.reached_back = true;
            return;
        }
        if let Some((mut m, n)) = self.header(i) {
            let was = m.written_by == WrittenBy::Model;
            f(&mut m);
            let now = m.written_by == WrittenBy::Model;
            if was != now {
                if now {
                    self.put(Key::Model(i), Cell::Present);
                } else {
                    self.remove(Key::Model(i));
                }
            }
            self.put(
                Key::Message(i),
                Cell::Header {
                    message: m,
                    blocks: n,
                },
            );
        }
    }
    fn push_message(&mut self, mut m: Message) {
        let i = self.s.messages;
        let header_time = (self.s.legacy && i == 0 && m.at.is_none())
            .then(|| {
                self.s
                    .session
                    .as_ref()
                    .and_then(|session| session.started_at)
            })
            .flatten();
        let applied_header_fallback = header_time.is_some();
        if applied_header_fallback {
            m.at = header_time;
        }
        self.s.messages += 1;
        self.put(
            Key::EvidenceHead(i),
            Cell::EvidenceHead(EvidenceHead::default()),
        );
        self.put(Key::BlockOrder(i), Cell::Position(0, 0));
        let blocks = std::mem::take(&mut m.blocks);
        let n = blocks.len();
        if m.written_by == WrittenBy::Model {
            self.put(Key::Model(i), Cell::Present);
        }
        self.put(
            Key::Message(i),
            Cell::Header {
                message: m,
                blocks: n,
            },
        );
        self.witness_time(i, false);
        if applied_header_fallback {
            match self.get(Key::Context(2)) {
                Some(Cell::Evidence(time)) => self.witness_from(
                    i,
                    "at",
                    super::presence::message(
                        polylogue_model::merge::MessageField::At,
                        &time.canonical,
                        header_time.and_then(|time| time.at()),
                    ),
                    time.origins,
                ),
                _ => self.missing("legacy header timestamp"),
            }
        }
        for (b, v) in blocks.into_iter().enumerate() {
            let next = match v.id.slot() {
                BlockSlot::Item(n) => (*n as usize + 1, 0),
                BlockSlot::Reasoning(n) => (0, *n as usize + 1),
                _ => (0, 0),
            };
            self.put(Key::BlockOrder(i), Cell::Position(next.0, next.1));
            if let BlockContent::ToolCall(c) = &v.content {
                if let Some(id) = &c.call_id {
                    self.put(Key::CallBlock(id.as_str().to_owned(), i, b), Cell::Present);
                }
            }
            self.put(Key::Block(i, b), Cell::Block(v));
            let raw = self.current_raw.clone();
            let origins = self.current_origins.at_path(&self.raw_item_path());
            self.witness_block_raw(i, b, &raw, &origins, super::presence::Owner::Primary);
        }
    }
    fn push_block(&mut self, i: usize, content: BlockContent) -> Option<usize> {
        if i < self.s.base {
            self.reached_back = true;
            return None;
        }
        let (m, n) = self.header(i)?;
        let id = self.next_block_id(i, &m.id, matches!(content, BlockContent::Thinking(_)))?;
        let b = polylogue_model::Block { id, content };
        if let BlockContent::ToolCall(c) = &b.content {
            if let Some(id) = &c.call_id {
                self.put(Key::CallBlock(id.as_str().to_owned(), i, n), Cell::Present);
            }
        }
        self.put(Key::Block(i, n), Cell::Block(b));
        self.put(
            Key::Message(i),
            Cell::Header {
                message: m,
                blocks: n + 1,
            },
        );
        Some(n)
    }
    fn edit_block(&mut self, i: usize, b: usize, f: impl FnOnce(&mut BlockContent)) {
        if i < self.s.base {
            self.reached_back = true;
            return;
        }
        if let Some(Cell::Block(mut v)) = self.get(Key::Block(i, b)) {
            f(&mut v.content);
            self.put(Key::Block(i, b), Cell::Block(v));
        } else if self.failure.is_none() {
            self.failure = Some(format!("missing Codex block {i}:{b}"));
        }
    }
    fn call(&mut self, c: &str) -> Option<CallInfo> {
        match self.get(Key::Call(c.into())) {
            Some(Cell::Call {
                at,
                tool,
                stdin_session,
            }) => Some(CallInfo {
                at,
                tool,
                stdin_session,
            }),
            _ => None,
        }
    }
    fn put_call(&mut self, c: String, v: CallInfo) {
        let old = self.call(&c).map(|v| v.at.0);
        if self.wrapper_is_open(&c) {
            self.pin(format!("wrapper:{c}"), old, Some(v.at.0));
        }
        if self.get(Key::PendingStdin(c.clone())).is_some() {
            self.pin(format!("stdin:{c}"), old, Some(v.at.0));
            self.put(Key::PendingStdin(c.clone()), Cell::Position(v.at.0, v.at.1));
        }
        self.put(
            Key::Call(c),
            Cell::Call {
                at: v.at,
                tool: v.tool,
                stdin_session: v.stdin_session,
            },
        );
    }
    fn position(&mut self, k: Key) -> Option<(usize, usize)> {
        match self.get(k) {
            Some(Cell::Position(i, b)) => Some((i, b)),
            _ => None,
        }
    }
    fn number(&mut self, k: Key) -> usize {
        match self.get(k) {
            Some(Cell::Number(n)) => n,
            _ => 0,
        }
    }
    fn pin(&mut self, label: String, old: Option<usize>, new: Option<usize>) {
        if let Some(i) = old {
            self.remove(Key::Mutable(i, label.clone()));
        }
        if let Some(i) = new {
            self.put(Key::Mutable(i, label), Cell::Present);
        }
    }
    fn wrappers(&mut self) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        let mut k = Key::Wrapper(0);
        if let Some(Cell::Text(c)) = self.get(k.clone()) {
            out.push((0, c));
        }
        while let Some((next, v)) = self.neighbor(k, true) {
            match (next, v) {
                (Key::Wrapper(n), Cell::Text(c)) => {
                    k = Key::Wrapper(n);
                    out.push((n, c));
                }
                _ => break,
            }
        }
        out
    }
    fn wrapper_is_open(&mut self, c: &str) -> bool {
        let k = Key::WrapperCall(c.into(), 0);
        matches!(self.get(k.clone()), Some(Cell::Present))
            || matches!(self.neighbor(k,true),Some((Key::WrapperCall(ref id,_),_)) if id==c)
    }
    fn wrapper_one(&mut self) -> Option<String> {
        let first = match self.get(Key::Wrapper(0)) {
            Some(Cell::Text(c)) => Some((Key::Wrapper(0), c)),
            _ => match self.neighbor(Key::Wrapper(0), true) {
                Some((Key::Wrapper(n), Cell::Text(c))) => Some((Key::Wrapper(n), c)),
                _ => None,
            },
        }?;
        if matches!(self.neighbor(first.0, true), Some((Key::Wrapper(_), _))) {
            None
        } else {
            Some(first.1)
        }
    }
    fn wrapper_push(&mut self, c: String, i: usize) {
        if self.get(Key::ChildCount(c.clone())).is_none() {
            self.put(Key::ChildCount(c.clone()), Cell::Number(0));
        }
        let n = self.s.next_wrapper;
        self.s.next_wrapper += 1;
        self.put(Key::Wrapper(n), Cell::Text(c.clone()));
        self.put(Key::WrapperCall(c.clone(), n), Cell::Present);
        self.pin(format!("wrapper:{c}"), None, Some(i));
    }
    fn wrapper_remove(&mut self, c: &str) {
        let mut k = Key::WrapperCall(c.into(), 0);
        loop {
            let hit = if matches!(self.get(k.clone()), Some(Cell::Present)) {
                Some(k.clone())
            } else {
                self.neighbor(k.clone(), true).map(|x| x.0)
            };
            let Some(Key::WrapperCall(id, n)) = hit else {
                break;
            };
            if id != c {
                break;
            }
            self.remove(Key::Wrapper(n));
            self.remove(Key::WrapperCall(id, n));
            if let Some(call) = self.call(c) {
                self.pin(format!("wrapper:{c}"), Some(call.at.0), None);
            } else {
                self.missing("wrapper call");
                return;
            }
            k = Key::WrapperCall(c.into(), n);
        }
    }
    fn child(&mut self, c: String, id: String) {
        let n = self.number(Key::ChildCount(c.clone()));
        self.put(Key::Child(c.clone(), n), Cell::Text(id));
        self.put(Key::ChildCount(c), Cell::Number(n + 1));
    }
    fn last_model(&mut self) -> Option<usize> {
        match self.neighbor(Key::Model(usize::MAX), false) {
            Some((Key::Model(i), _)) => Some(i),
            _ => None,
        }
    }
    fn floor(&mut self) -> usize {
        if self.s.messages == 0 {
            return 0;
        }
        let mut lo = self.s.messages - 1;
        if let Some(i) = self.s.open_request {
            lo = lo.min(i);
        }
        if let Some(i) = self.s.pending_prompt {
            lo = lo.min(i);
        }
        let last_model = self.last_model();
        if let Some(i) = last_model {
            lo = lo.min(i);
            // A matching prompt echo may reclassify this model-written
            // subagent prompt as Human. An unattached token_count must then
            // still find its predecessor, just as a whole parse does.
            if self.s.pending_prompt == Some(i) {
                if let Some((Key::Model(previous), _)) = self.neighbor(Key::Model(i), false) {
                    lo = lo.min(previous);
                }
            }
        }
        let first = match self.get(Key::Mutable(0, String::new())) {
            Some(v) => Some((Key::Mutable(0, String::new()), v)),
            None => self.neighbor(Key::Mutable(0, String::new()), true),
        };
        if let Some((Key::Mutable(i, _), _)) = first {
            lo = lo.min(i);
        }
        lo.max(self.s.base)
    }
    pub fn finish_delta(mut self) -> Result<Delta, String> {
        if let Some(e) = self.failure.take() {
            return Err(e);
        }
        let base = self.floor();
        if let Some(e) = self.failure.take() {
            return Err(e);
        }
        self.s.base = base;
        Ok(Delta {
            resume: self.s,
            cells: self.changes,
            events: self.events,
            links: self.links,
            reached_back: self.reached_back,
            conflicts: self.conflicts,
            pending_results: self.pending_results.into_iter().collect(),
            pending_versions: self
                .pending_versions
                .into_iter()
                .map(|(message, evidence_revision)| PendingVersion {
                    message,
                    evidence_revision,
                })
                .collect(),
        })
    }
}
impl<V: ReadView> RecordParser for KeyedRecords<V> {
    type Error = String;
    type Ignored = Ignored;
    type Output = Delta;
    fn record(&mut self, rec: &Record<'_>) -> Result<RecordOutcome<Ignored>, String> {
        if let Some(e) = &self.failure {
            return Err(e.clone());
        }
        super::require_scalar_json(&rec.value)?;
        // Capture exact provider bytes before decoded Value can normalize numbers.
        if !rec.table.is_empty() {
            return Err("Codex SQLite row origins are not implemented".into());
        }
        if rec.end.checked_sub(rec.start) != u64::try_from(rec.bytes.len()).ok() {
            return Err("Codex original-record range differs from framed bytes".into());
        }
        self.current_origins = Origins::Record(RecordRange::new(rec.start, rec.end)?);
        self.effective_origins = Origins::Unknown;
        self.effective_raw.clear();
        self.current_time_raw = raw::canonical_field(rec.bytes, &["timestamp"]).ok();
        self.current_raw = raw::canonical(rec.bytes)?;
        self.effective_item = None;
        self.replacement_block = None;
        let out = self.interpret(rec);
        if let Some(e) = &self.failure {
            Err(e.clone())
        } else {
            Ok(out)
        }
    }
    fn finish(self) -> Result<Delta, String> {
        self.finish_delta()
    }
}

impl<V: ReadView> KeyedRecords<V> {
    fn witness(&mut self, i: usize, key: &str, value: Result<super::presence::Witness, String>) {
        let origins = if key == "at" || key == "completed_at" {
            self.current_origins.at_path(&["timestamp"])
        } else {
            self.current_origins.clone()
        };
        self.witness_from(i, key, value, origins);
    }
    fn witness_from(
        &mut self,
        i: usize,
        key: &str,
        value: Result<super::presence::Witness, String>,
        origins: Origins,
    ) {
        match value {
            Ok(mut w) => {
                use super::selection::{FactSelection, Transform};
                use polylogue_model::merge::{MessageField, Slot};
                let transform = match &w.slot {
                    Slot::Message(MessageField::At | MessageField::CompletedAt) => {
                        Some(Transform::ProviderTimeV1)
                    }
                    Slot::Message(MessageField::Model) => Some(Transform::JsonStringV1),
                    Slot::Extra { .. } => Some(Transform::JsonValueV1),
                    _ => None,
                };
                w.selection = transform.and_then(|transform| {
                    origins
                        .selected()
                        .map(|raw| FactSelection::new(transform, vec![raw]))
                });
                w.origins = origins;
                self.put(Key::Presence(i, key.into()), Cell::Witness(w));
            }
            Err(e) => self.failure = Some(e),
        }
    }
    fn witness_usage(
        &mut self,
        i: usize,
        name: &str,
        field: polylogue_model::merge::UsageField,
        raw: &[u8],
        has_cached_input: bool,
    ) {
        use super::selection::{FactSelection, Transform};
        use polylogue_model::merge::UsageField;
        let origins = self
            .current_origins
            .at_path(&["payload", "info", "last_token_usage"]);
        let selection = (|| {
            if field == UsageField::Input && has_cached_input {
                Some(FactSelection::new(
                    Transform::UncachedInputV1,
                    vec![
                        origins.field("input_tokens").selected()?,
                        origins.field("cached_input_tokens").selected()?,
                    ],
                ))
            } else {
                Some(FactSelection::new(
                    Transform::JsonValueV1,
                    vec![origins.field(name).selected()?],
                ))
            }
        })();
        match super::presence::usage(field, raw) {
            Ok(mut witness) => {
                witness.origins = origins.field(name);
                witness.selection = selection;
                self.put(
                    Key::Presence(i, format!("usage:{name}")),
                    Cell::Witness(witness),
                );
            }
            Err(error) => self.failure = Some(error),
        }
    }
    fn witness_time(&mut self, i: usize, completed: bool) {
        if let Some(bytes) = self.current_time_raw.clone() {
            if let Ok(text) = serde_json::from_slice::<String>(&bytes) {
                if let Some(at) = parse_timestamp(&text) {
                    let (key, field) = if completed {
                        (
                            "completed_at",
                            polylogue_model::merge::MessageField::CompletedAt,
                        )
                    } else {
                        ("at", polylogue_model::merge::MessageField::At)
                    };
                    self.witness(i, key, super::presence::message(field, &bytes, Some(at)));
                }
            }
        }
    }
    fn witness_block_raw(
        &mut self,
        i: usize,
        b: usize,
        raw: &[u8],
        origins: &Origins,
        owner: super::presence::Owner,
    ) {
        use super::presence::{Kind, Owner};
        let Some(Cell::Block(block)) = self.get(Key::Block(i, b)) else {
            self.missing("presence block");
            return;
        };
        let kind = match block.content {
            BlockContent::Text { .. } => Kind::Text,
            BlockContent::Thinking(_) => Kind::Thinking,
            BlockContent::ToolCall(_) => Kind::Call,
            BlockContent::ToolResult(_) => Kind::Result,
            _ => Kind::Other,
        };
        let path = if self.s.legacy {
            vec![]
        } else if raw::field(raw, &["payload", "item"]).is_ok() {
            vec!["payload", "item"]
        } else {
            vec!["payload"]
        };
        match super::presence::block(raw, &path, block.id.slot(), kind, owner, origins) {
            Ok(next) => {
                let mut group = match self.get(Key::PresenceBlock(i, b)) {
                    Some(Cell::Presence(group)) => group,
                    None => super::presence::Group::default(),
                    _ => {
                        self.missing("presence group");
                        return;
                    }
                };
                let nonempty = next.nonempty;
                group.replace(owner, next);
                self.put(Key::PresenceBlock(i, b), Cell::Presence(group));
                if owner == Owner::Primary && !nonempty {
                    self.remove(Key::Presence(i, "extra:phase".into()));
                    return;
                }
            }
            Err(e) => self.failure = Some(e),
        }
        if owner == Owner::Primary {
            if let Ok(phase) =
                raw::field(raw, &path).and_then(|item| raw::canonical_field(item, &["phase"]))
            {
                self.witness_from(
                    i,
                    "extra:phase",
                    super::presence::extra("phase", &phase),
                    origins.field("phase"),
                );
            }
        }
    }
    fn witness_block(&mut self, i: usize, b: usize, owner: super::presence::Owner) {
        let raw = if self.effective_raw.is_empty() {
            self.current_raw.clone()
        } else {
            self.effective_raw.clone()
        };
        let origins = if self.effective_raw.is_empty() {
            self.current_origins.at_path(&self.raw_item_path())
        } else {
            self.effective_origins.clone()
        };
        self.witness_block_raw(i, b, &raw, &origins, owner);
    }
    fn raw_item_path(&self) -> Vec<&'static str> {
        if self.s.legacy {
            vec![]
        } else if raw::field(&self.current_raw, &["payload", "item"]).is_ok() {
            vec!["payload", "item"]
        } else {
            vec!["payload"]
        }
    }
    fn native_key(&self, id: &str) -> Key {
        let scope = if raw::field(&self.current_raw, &["payload", "item"]).is_ok() {
            raw::field(&self.current_raw, &["payload", "turn_id"])
                .ok()
                .and_then(|b| serde_json::from_slice::<String>(b).ok())
                .map(|turn| format!("completed:{turn}"))
                .unwrap_or_else(|| "legacy-completed".into())
        } else {
            "response_item".into()
        };
        Key::NativeItem(scope, id.into())
    }
    fn known_owner(&mut self, item: &Value) -> Option<usize> {
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())?;
        self.position(self.native_key(id))
            .map(|(message, _)| message)
    }
    fn preview_item(&mut self, item: &Value) -> Value {
        let Some(i) = self.known_owner(item) else {
            return item.clone();
        };
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            return item.clone();
        };
        let slot = match self.get(Key::EvidenceItem(i, id.into())) {
            Some(Cell::Position(slot, _)) => slot,
            _ => {
                self.missing("native item position");
                return item.clone();
            }
        };
        let previous = match self.get(Key::Evidence(i, 0, slot)) {
            Some(Cell::Evidence(unit)) => unit,
            _ => {
                self.missing("native item evidence");
                return item.clone();
            }
        };
        match raw::overlay(
            &previous.canonical,
            &self.current_raw,
            &self.raw_item_path(),
        ) {
            Ok((_, fields)) => fields.value,
            Err(error) => {
                self.failure = Some(error);
                item.clone()
            }
        }
    }
    fn effective(&self, original: &Value) -> Value {
        self.effective_item
            .clone()
            .unwrap_or_else(|| original.clone())
    }
    fn owned_block(&mut self, i: usize, content: BlockContent) -> Option<usize> {
        if let Some(block) = self.replacement_block.take() {
            let previous = match self.get(Key::Block(i, block)) {
                Some(Cell::Block(value)) => Some(value.content),
                _ => {
                    self.missing("owned block");
                    return None;
                }
            };
            if let Some(BlockContent::ToolCall(old)) = previous {
                if let Some(id) = old.call_id {
                    let new = match &content {
                        BlockContent::ToolCall(new) => new.call_id.as_ref().map(NativeId::as_str),
                        _ => None,
                    };
                    if new != Some(id.as_str()) {
                        self.remove(Key::CallBlock(id.as_str().to_owned(), i, block));
                        if self
                            .call(id.as_str())
                            .is_some_and(|call| call.at == (i, block))
                        {
                            self.wrapper_remove(id.as_str());
                            self.remove(Key::Call(id.as_str().to_owned()));
                            self.remove(Key::CallWitness(id.as_str().to_owned()));
                            if let Some(Cell::Position(message, _)) =
                                self.remove(Key::PendingStdin(id.as_str().to_owned()))
                            {
                                self.pin(format!("stdin:{id}"), Some(message), None);
                            }
                        }
                    }
                }
            }
            if let BlockContent::ToolCall(new) = &content {
                if let Some(id) = &new.call_id {
                    self.put(
                        Key::CallBlock(id.as_str().to_owned(), i, block),
                        Cell::Present,
                    );
                }
            }
            self.edit_block(i, block, |value| *value = content);
            self.witness_block(i, block, super::presence::Owner::Primary);
            Some(block)
        } else {
            let block = self.push_block(i, content)?;
            self.witness_block(i, block, super::presence::Owner::Primary);
            Some(block)
        }
    }
    fn pending_result(&mut self, i: usize, block: usize) {
        self.put(Key::PendingResult(i, block), Cell::Present);
        self.pending_results.insert((i, block));
    }
    fn require_result(&mut self, i: usize, block: usize) -> bool {
        match self.get(Key::Block(i, block)) {
            Some(Cell::Block(block)) if matches!(block.content, BlockContent::ToolResult(_)) => {
                true
            }
            _ => {
                self.missing("promised tool result");
                false
            }
        }
    }
    fn result_chunk(&mut self, i: usize, block: usize, slot: usize, output: &Option<ToolOutput>) {
        self.witness_block(i, block, super::presence::Owner::Continuation);
        if let Some(ToolOutput::Text(text)) = output {
            self.put(Key::OutputChunk(i, block, slot), Cell::Text(text.clone()));
            self.pending_result(i, block);
        } else {
            self.remove(Key::OutputChunk(i, block, slot));
            self.pending_result(i, block);
        }
    }
    fn current_item_slot(&mut self, i: usize, item: &Value) -> Option<usize> {
        if let Some(id) = item
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            if let Some(Cell::Position(slot, _)) = self.get(Key::EvidenceItem(i, id.into())) {
                return Some(slot);
            }
            self.missing("item slot");
            return None;
        }
        self.evidence_head(i)
            .items
            .checked_sub(1)
            .and_then(|n| n.checked_mul(3))
    }
    fn remove_evidence(&mut self, i: usize, phase: usize, slot: usize) {
        let key = Key::Evidence(i, phase, slot);
        if self.get(key.clone()).is_none() {
            return;
        }
        let mut head = self.evidence_head(i);
        let Some(units) = head.units.checked_sub(1) else {
            self.missing("evidence count");
            return;
        };
        let Some(revision) = head.revision.checked_add(1) else {
            self.failure = Some("Codex evidence revision overflow".into());
            return;
        };
        head.units = units;
        head.revision = revision;
        self.remove(key);
        self.put(Key::EvidenceHead(i), Cell::EvidenceHead(head));
        self.edit(i, |message| message.version_hash = None);
        self.pending_versions.insert(i, revision);
    }
    fn evidence_head(&mut self, i: usize) -> EvidenceHead {
        match self.get(Key::EvidenceHead(i)) {
            Some(Cell::EvidenceHead(head)) => head,
            None => {
                self.missing("evidence head");
                EvidenceHead::default()
            }
            _ => {
                self.missing("evidence head");
                EvidenceHead::default()
            }
        }
    }
    fn evidence_unit(&mut self, i: usize, phase: usize, slot: usize, unit: EvidenceUnit) {
        let key = Key::Evidence(i, phase, slot);
        let old = self.get(key.clone());
        if matches!(&old, Some(Cell::Evidence(previous)) if previous == &unit) {
            return;
        }
        let mut head = self.evidence_head(i);
        if old.is_none() {
            let Some(units) = head.units.checked_add(1) else {
                self.failure = Some("Codex evidence count overflow".into());
                return;
            };
            head.units = units;
        }
        let Some(revision) = head.revision.checked_add(1) else {
            self.failure = Some("Codex evidence revision overflow".into());
            return;
        };
        head.revision = revision;
        self.put(key, Cell::Evidence(unit));
        self.put(Key::EvidenceHead(i), Cell::EvidenceHead(head));
        self.edit(i, |m| m.version_hash = None);
        self.pending_versions.insert(i, revision);
    }
    /// Effective raw snapshots retain one slot and block position per native
    /// item. Exact re-emission adds no unit; later fields replace in place.
    fn own_evidence(&mut self, i: usize, item: &Value, primary: bool) -> bool {
        self.effective_item = Some(item.clone());
        self.replacement_block = None;
        let native = item
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        let existing = native.and_then(|id| match self.get(Key::EvidenceItem(i, id.to_owned())) {
            Some(Cell::Position(slot, block)) => Some((slot, block)),
            _ => None,
        });
        if primary
            && existing.is_none()
            && native.is_some_and(|id| self.position(self.native_key(id)).is_some())
        {
            self.missing("native item position");
            return false;
        }
        let mut unit = EvidenceUnit {
            tag: "item".into(),
            canonical: self.current_raw.clone(),
            whole_record: true,
            origins: self.current_origins.at_path(&self.raw_item_path()),
        };
        let slot = if let Some((slot, block)) = existing {
            if matches!(self.get(Key::Evidence(i,0,slot)), Some(Cell::Evidence(old)) if old == unit)
            {
                return false;
            }
            self.replacement_block = Some(block);
            let previous = match self.get(Key::Evidence(i, 0, slot)) {
                Some(Cell::Evidence(unit)) => unit,
                _ => {
                    self.missing("owned item evidence");
                    return false;
                }
            };
            let path = self.raw_item_path();
            match raw::overlay(&previous.canonical, &self.current_raw, &path) {
                Ok((canonical, fields)) => {
                    unit.origins =
                        previous
                            .origins
                            .overlay(&unit.origins, &fields.retained, &fields.provided);
                    unit.whole_record = canonical == self.current_raw;
                    unit.canonical = canonical;
                    let original = raw::field(&previous.canonical, &path)
                        .ok()
                        .and_then(|raw| serde_json::from_slice::<Value>(raw).ok());
                    let association_changed = original.as_ref().is_some_and(|old| {
                        ["role", "type", "call_id", "name"]
                            .iter()
                            .any(|key| old.get(*key) != fields.value.get(*key))
                    });
                    self.effective_item = Some(fields.value);
                    if association_changed {
                        self.s.normalization_incomplete = true;
                        let mut head = self.evidence_head(i);
                        head.incomplete = true;
                        self.put(Key::EvidenceHead(i), Cell::EvidenceHead(head));
                    }
                    if unit == previous {
                        return false;
                    }
                    if primary && !fields.conflicts.is_empty() {
                        self.conflicts.push(NativeConflict {
                            message: i,
                            fields: fields.conflicts,
                        });
                    }
                }
                Err(error) => {
                    self.failure = Some(error);
                    return false;
                }
            }
            slot
        } else {
            let mut head = self.evidence_head(i);
            let Some(next) = head.items.checked_add(1) else {
                self.failure = Some("Codex evidence item overflow".into());
                return false;
            };
            let Some(slot) = head.items.checked_mul(3) else {
                self.failure = Some("Codex evidence slot overflow".into());
                return false;
            };
            head.items = next;
            self.put(Key::EvidenceHead(i), Cell::EvidenceHead(head));
            if let Some(id) = native {
                let block = if primary {
                    self.header(i).map(|(_, blocks)| blocks).unwrap_or(0)
                } else {
                    0
                };
                self.put(
                    Key::EvidenceItem(i, id.to_owned()),
                    Cell::Position(slot, block),
                );
                if primary {
                    self.put(self.native_key(id), Cell::Position(i, block));
                }
            }
            slot
        };
        self.effective_raw = unit.canonical.clone();
        self.effective_origins = unit.origins.clone();
        self.evidence_unit(i, 0, slot, unit);
        true
    }
    fn fixed_evidence(
        &mut self,
        i: usize,
        slot: usize,
        tag: &str,
        canonical: Vec<u8>,
        whole_record: bool,
    ) {
        self.evidence_unit(
            i,
            1,
            slot,
            EvidenceUnit {
                tag: tag.into(),
                canonical,
                whole_record,
                origins: self.current_origins.clone(),
            },
        );
    }
    fn context_for(&mut self, i: usize, source: bool, model: bool) {
        for (wanted, key, slot) in [(source, 0, 1), (model, 1, 2)] {
            if wanted {
                if let Some(Cell::Evidence(unit)) = self.get(Key::Context(key)) {
                    if key == 1 {
                        if serde_json::from_slice::<String>(&unit.canonical).is_ok() {
                            self.witness_from(
                                i,
                                "model",
                                super::presence::message(
                                    polylogue_model::merge::MessageField::Model,
                                    &unit.canonical,
                                    None,
                                ),
                                unit.origins.clone(),
                            );
                        } else {
                            self.remove(Key::Presence(i, "model".into()));
                        }
                    }
                    self.evidence_unit(i, 1, slot, unit);
                } else if (key == 0 && self.s.has_source_context)
                    || (key == 1 && self.s.has_model_context)
                {
                    self.missing("required context evidence");
                }
            }
        }
    }
    fn raw_context(&mut self, raw: &[u8], path: &[&str], key: usize, tag: &str) {
        if let Ok(value) = raw::field(raw, path).and_then(|bytes| {
            polylogue_model::identity::canonical_record(bytes, &[]).map_err(|e| e.to_string())
        }) {
            if key == 0 {
                self.s.has_source_context = true;
            } else if key == 1 {
                self.s.has_model_context = true;
            }
            self.put(
                Key::Context(key),
                Cell::Evidence(EvidenceUnit {
                    tag: tag.into(),
                    canonical: value,
                    whole_record: false,
                    origins: self.current_origins.at_path(path),
                }),
            );
        }
    }
    fn association_evidence(&mut self, i: usize, call_id: &str, item: &Value) {
        let Some(slot) = self.current_item_slot(i, item) else {
            self.missing("output evidence slot");
            return;
        };
        if let Some(Cell::Association {
            call_name,
            session,
            call_name_origins,
            session_origins,
        }) = self.get(Key::CallWitness(call_id.into()))
        {
            for (offset, tag, value, origins) in [
                (1, "association:call_name", call_name, call_name_origins),
                (2, "association:process_session", session, session_origins),
            ] {
                if let Some(canonical) = value {
                    self.evidence_unit(
                        i,
                        0,
                        slot + offset,
                        EvidenceUnit {
                            tag: tag.into(),
                            canonical,
                            whole_record: false,
                            origins,
                        },
                    );
                } else {
                    self.remove_evidence(i, 0, slot + offset);
                }
            }
        } else if self.call(call_id).is_some() {
            self.missing("call association evidence");
        }
    }
    fn next_block_id(&mut self, i: usize, message: &MessageId, thinking: bool) -> Option<BlockId> {
        let (normal, reasoning) = match self.get(Key::BlockOrder(i)) {
            Some(Cell::Position(n, r)) => (n, r),
            None => {
                self.missing("block order");
                return None;
            }
            _ => {
                self.missing("block order");
                return None;
            }
        };
        let value = if thinking { reasoning } else { normal };
        let index = match u32::try_from(value) {
            Ok(n) => n,
            Err(_) => {
                self.failure = Some("Codex block index capacity".into());
                return None;
            }
        };
        let Some(next) = value.checked_add(1) else {
            self.failure = Some("Codex block order overflow".into());
            return None;
        };
        self.put(
            Key::BlockOrder(i),
            Cell::Position(
                if thinking { normal } else { next },
                if thinking { next } else { reasoning },
            ),
        );
        Some(if thinking {
            BlockId::reasoning(message.clone(), index)
        } else {
            BlockId::new(message.clone(), index)
        })
    }
    fn sid(&self) -> Option<SessionId> {
        self.s.session.as_ref().map(|s| s.id.clone())
    }

    fn message_id(&mut self, native: Option<&str>, _payload: &Value) -> Option<MessageId> {
        let sid = self.sid()?;
        match native.filter(|n| !n.is_empty()) {
            Some(n) => MessageId::new(sid, n).ok(),
            None => {
                // Fallback ID over the provider's own record, timestamps and ordinal left out.
                let hash = BlobHash::of(&self.current_raw).to_string();
                let short = &hash[..12];
                let n = self.number(Key::Occurrence(short.into()));
                self.put(Key::Occurrence(short.into()), Cell::Number(n + 1));
                let n = match u32::try_from(n) {
                    Ok(n) => n,
                    Err(_) => {
                        self.failure = Some("Codex fallback counter capacity".into());
                        return None;
                    }
                };
                Some(
                    MessageId::from_stored(format!("{}:~{short}-{n}", sid.as_str()), sid)
                        .expect("nonempty generated identity"),
                )
            }
        }
    }

    fn close_request(&mut self) {
        self.s.open_request = None;
        self.s.open_anchored = false;
        self.s.output_since_open = false;
    }

    fn start_session(
        &mut self,
        id: &str,
        at: Option<Timestamp>,
        meta: &Value,
    ) -> RecordOutcome<Ignored> {
        let Ok(sid) = SessionId::new(&Origin::new("codex").expect("valid product name"), id) else {
            return RecordOutcome::Unrecognized("bad thread id");
        };
        let context_raw = self.current_raw.clone();
        self.raw_context(
            &context_raw,
            if self.s.legacy {
                &["source"]
            } else {
                &["payload", "source"]
            },
            0,
            "context:source",
        );
        let spawned = meta
            .get("source")
            .and_then(|s| s.get("subagent"))
            .and_then(|s| s.get("thread_spawn"));
        self.s.mode = match meta.get("source") {
            Some(Value::String(src)) if src.text() == "cli" || src.text() == "vscode" => {
                SessionMode::Interactive
            }
            Some(Value::String(src)) if src.text() == "exec" => SessionMode::Exec,
            Some(Value::Object(o)) if o.contains_key(&"subagent".into()) => SessionMode::Subagent,
            _ => SessionMode::Unknown,
        };
        let mut s = match Session::new(
            sid.clone(),
            Origin::new("codex").expect("valid product name"),
            id.to_owned(),
            if spawned.is_some() {
                SessionKind::Subagent
            } else {
                SessionKind::Main
            },
        ) {
            Ok(s) => s,
            Err(_) => return RecordOutcome::Unrecognized("invalid session relation"),
        };
        s.cwd = meta.get("cwd").and_then(Value::as_str).map(str::to_owned);
        s.harness = Some("codex".into());
        s.harness_version = meta
            .get("cli_version")
            .and_then(Value::as_str)
            .map(str::to_owned);
        if let Some(git) = meta.get("git") {
            s.commit = git
                .get("commit_hash")
                .and_then(Value::as_str)
                .map(str::to_owned);
            s.branch = git.get("branch").and_then(Value::as_str).map(str::to_owned);
            s.repo = git
                .get("repository_url")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
        s.started_at = at.map(provider_time);
        let forked = meta.get("forked_from_id").and_then(Value::as_str);
        // This source core records the lineage link; cross-input replay-prefix
        // correspondence has not been supplied, so inherited counts are open.
        if forked.is_some() {
            self.s.normalization_incomplete = true;
        }
        if let Some(spawn) = spawned {
            let parent = spawn.get("parent_thread_id").and_then(Value::as_str);
            if let Some(parent) = parent.and_then(|p| {
                SessionId::new(&Origin::new("codex").expect("valid product name"), p).ok()
            }) {
                match Link::new(
                    LinkKind::Subagent,
                    if forked.is_some() {
                        LinkHow::Copied
                    } else {
                        LinkHow::Fresh
                    },
                    parent,
                    sid.clone(),
                    "session_meta.source.subagent.thread_spawn",
                ) {
                    Ok(mut link) => {
                        link.attrs = Some(spawn.clone());
                        self.links.push(link);
                    }
                    Err(error) => self.failure = Some(error.to_string()),
                }
            }
        } else if let Some(parent) = forked.and_then(|p| {
            SessionId::new(&Origin::new("codex").expect("valid product name"), p).ok()
        }) {
            match Link::new(
                LinkKind::Continuation,
                LinkHow::Fork,
                parent,
                sid.clone(),
                "session_meta.forked_from_id",
            ) {
                Ok(link) => self.links.push(link),
                Err(error) => self.failure = Some(error.to_string()),
            }
        }
        self.s.session = Some(s);
        if let Some(bi) = meta.get("base_instructions") {
            // What the model was told, as an event like `world_state` (spec fix, codex.md).
            self.events.push(Event {
                kind: "base_instructions".into(),
                at: at.map(provider_time),
                after_message: None,
                model_saw: Some(true),
                data: bi.clone(),
            });
        }
        RecordOutcome::Used
    }

    fn event(
        &mut self,
        kind: &str,
        at: Option<Timestamp>,
        model_saw: bool,
        data: &Value,
    ) -> RecordOutcome<Ignored> {
        let after = self
            .s
            .messages
            .checked_sub(1)
            .and_then(|i| self.header(i))
            .map(|(m, _)| m.id);
        self.events.push(Event {
            kind: kind.into(),
            at: at.map(provider_time),
            after_message: after,
            model_saw: Some(model_saw),
            data: data.clone(),
        });
        RecordOutcome::Used
    }

    /// The open request's model message, opening one if none is open.
    fn request_message(&mut self, first_item: &Value, at: Option<Timestamp>) -> Option<usize> {
        if let Some(i) = self.known_owner(first_item) {
            return Some(i);
        }
        let reopen = self.s.legacy && self.s.output_since_open;
        if let (Some(i), false) = (self.s.open_request, reopen) {
            self.witness_time(i, true);
            self.edit(i, |m| {
                m.completed_at = at.map(provider_time).or(m.completed_at)
            });
            return Some(i);
        }
        self.close_request();
        let native = first_item.get("id").and_then(Value::as_str);
        let id = self.message_id(native, first_item)?;
        let mut m = Message::new(
            id,
            item_id_kind(first_item),
            Role::Assistant,
            WrittenBy::Model,
        );
        m.model = self.s.model.clone();
        m.at = at.map(provider_time);
        self.push_message(m);
        let i = self.s.messages - 1;
        self.s.open_request = Some(i);
        self.s.open_anchored = first_item.get("type").and_then(Value::as_str) != Some("reasoning");
        Some(i)
    }

    /// A request's message takes its ID from its first item that is not
    /// `reasoning`, the kind rewrites drop, so the ID survives a lossy rewrite
    /// (spec fix, codex.md). Called when a non-reasoning item joins the request.
    fn anchor_request(&mut self, i: usize, item: &Value) {
        if self.s.open_anchored || self.s.open_request != Some(i) {
            return;
        }
        self.s.open_anchored = true;
        let native = item.get("id").and_then(Value::as_str);
        let Some(id) = self.message_id(native, item) else {
            return;
        };
        // Blocks are internally addressed by ordinal. Reanchoring changes their public
        // IDs on projection; no need to touch every preceding block here.
        self.edit(i, |m| {
            m.id = id.clone();
            m.id_kind = item_id_kind(item);
        });
    }

    fn response_item(&mut self, p: &Value, at: Option<Timestamp>) -> RecordOutcome<Ignored> {
        let preview = self.preview_item(p);
        let p = &preview;
        let ty = p.get("type").and_then(Value::as_str).unwrap_or("");
        match ty {
            "message" => self.message_item(p, at),
            t if is_model_item(t) => self.model_item(p, t, at),
            "function_call_output" | "custom_tool_call_output" | "tool_search_output" => {
                self.output_item(p, at)
            }
            "internal_chat_message_metadata_passthrough" => {
                RecordOutcome::Ignored(Ignored::InternalPassthrough)
            }
            "ghost_snapshot" | "agent_message" => self.event(ty, at, false, p),
            _ => RecordOutcome::Unrecognized("response_item type"),
        }
    }

    fn message_item(&mut self, p: &Value, at: Option<Timestamp>) -> RecordOutcome<Ignored> {
        let role = p.get("role").and_then(Value::as_str).unwrap_or("");
        let text = texts(p.get("content")).join("\n");
        if role == "assistant" {
            // The assistant message is one more item of the open request (or opens one).
            let Some(i) = self.request_message_for_assistant(p, at) else {
                return RecordOutcome::Unrecognized("no session");
            };
            if !self.own_evidence(i, p, true) {
                return RecordOutcome::Used;
            }
            if self.replacement_block.is_none() {
                self.context_for(i, false, true);
            }
            let p = self.effective(p);
            self.edit(i, |m| {
                if let Some(phase) = p.get("phase") {
                    m.attrs = Some(Value::Object(
                        [("phase".into(), phase.clone())].into_iter().collect(),
                    ));
                }
            });
            let content = self.message_content(&p);
            self.owned_block(i, content);
            return RecordOutcome::Used;
        }
        if let Some(i) = self.known_owner(p) {
            if !self.own_evidence(i, p, true) {
                return RecordOutcome::Used;
            }
            let effective = self.effective(p);
            let text = texts(effective.get("content")).join("\n");
            let changed_text = self
                .replacement_block
                .and_then(|b| self.get(Key::Block(i, b)))
                .is_some_and(|cell| match cell {
                    Cell::Block(block) => match &block.content {
                        BlockContent::Text { text: old } => old != &text,
                        _ => false,
                    },
                    _ => false,
                });
            if changed_text {
                self.remove_evidence(i, 1, 3);
            }
            if role == "user" {
                self.edit(i, |message| message.role = Role::User);
                self.classify_prompt(i, &text);
                if is_harness_wrapper(&text) {
                    self.remove_evidence(i, 1, 1);
                } else {
                    self.context_for(i, true, false);
                }
            } else {
                self.edit(i, |message| {
                    message.role = Role::System;
                    message.written_by = WrittenBy::Harness;
                });
            }
            let content = self.message_content(&effective);
            self.owned_block(i, content);
            return RecordOutcome::Used;
        }
        self.close_request();
        let native = p.get("id").and_then(Value::as_str);
        let Some(id) = self.message_id(native, p) else {
            return RecordOutcome::Unrecognized("message before session_meta");
        };
        let (r, wb) = match role {
            "user" => (Role::User, WrittenBy::Harness),
            _ => (Role::System, WrittenBy::Harness),
        };
        let mut m = Message::new(id, item_id_kind(p), r, wb);
        m.at = at.map(provider_time);
        let content = self.message_content(p);
        if let Err(e) = m.push(BlockSlot::Item(0), content) {
            self.failure = Some(e.to_string());
            return RecordOutcome::Used;
        }
        self.push_message(m);
        let index = self.s.messages - 1;
        self.own_evidence(index, p, true);
        if let Some(native) = p.get("id").and_then(Value::as_str) {
            self.put(
                Key::EvidenceItem(index, native.into()),
                Cell::Position(0, 0),
            );
            self.put(self.native_key(native), Cell::Position(index, 0));
        }
        self.context_for(index, role == "user" && !is_harness_wrapper(&text), false);
        if role == "user" {
            self.classify_prompt(index, &text);
        }
        RecordOutcome::Used
    }

    fn message_content(&mut self, item: &Value) -> BlockContent {
        let content = item.get("content");
        let textual = match content {
            Some(Value::String(_)) => true,
            Some(Value::Array(items)) => items.iter().all(|item| {
                item.is_string()
                    || (item.get("text").is_some_and(Value::is_string)
                        && item.get("type").is_none_or(|kind| {
                            matches!(kind.as_str(), Some("text" | "input_text" | "output_text"))
                        }))
            }),
            _ => false,
        };
        if textual {
            BlockContent::Text {
                text: texts(content).join("\n"),
            }
        } else {
            self.s.normalization_incomplete = true;
            BlockContent::Other {
                type_name: "codex_unmapped_message_content".into(),
                data: item.clone(),
            }
        }
    }
    fn classify_prompt(&mut self, index: usize, text: &str) {
        self.edit(index, |message| {
            message.written_by = WrittenBy::Harness;
            message.prompt_kind = None;
        });
        if self.s.legacy {
            if !is_harness_wrapper(text) {
                self.mark_human(index);
            }
            return;
        }
        if !is_harness_wrapper(text) {
            match self.s.mode {
                SessionMode::Interactive => self.mark_human(index),
                SessionMode::Subagent => {
                    self.edit(index, |message| message.written_by = WrittenBy::Model)
                }
                SessionMode::Exec => {}
                SessionMode::Unknown => {
                    self.edit(index, |message| message.written_by = WrittenBy::Unknown)
                }
            }
        }
        self.s.pending_prompt = Some(index);
        self.put(Key::Prompt, Cell::Text(text.into()));
    }
    fn request_message_for_assistant(&mut self, p: &Value, at: Option<Timestamp>) -> Option<usize> {
        let i = self.request_message(p, at)?;
        self.anchor_request(i, p);
        Some(i)
    }

    fn mark_human(&mut self, index: usize) {
        self.edit(index, |m| {
            m.written_by = WrittenBy::Human;
            m.prompt_kind = Some(PromptKind::Prose);
        });
    }

    fn model_item(&mut self, p: &Value, ty: &str, at: Option<Timestamp>) -> RecordOutcome<Ignored> {
        if ty == "reasoning" {
            let rid = p.get("id").and_then(Value::as_str).map(str::to_owned);
            let existing = rid
                .as_ref()
                .and_then(|r| self.position(Key::Reasoning(r.clone())));
            let i = if let Some((mi, _)) = existing {
                mi
            } else {
                let Some(i) = self.request_message(p, at) else {
                    return RecordOutcome::Unrecognized("no session");
                };
                i
            };
            if !self.own_evidence(i, p, true) {
                return RecordOutcome::Used;
            }
            let p = self.effective(p);
            let summary = texts(p.get("summary"));
            let encrypted = p
                .get("encrypted_content")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let thinking = Thinking {
                text: if summary.is_empty() {
                    None
                } else {
                    Some(summary.join("\n"))
                },
                signature: None,
                // No plaintext or ciphertext establishes no encryption fact.
                encrypted: if encrypted.is_some() || !summary.is_empty() {
                    Some(encrypted.is_some() && summary.is_empty())
                } else {
                    None
                },
                data: encrypted,
            };
            if existing.is_none() {
                self.context_for(i, false, true);
            }
            let Some(bi) = self.owned_block(i, BlockContent::Thinking(thinking)) else {
                return RecordOutcome::Used;
            };
            if let Some(r) = rid {
                self.put(Key::Reasoning(r), Cell::Position(i, bi));
            }
            return RecordOutcome::Used;
        }
        let Some(i) = self.request_message(p, at) else {
            return RecordOutcome::Unrecognized("no session");
        };
        self.anchor_request(i, p);
        if !self.own_evidence(i, p, true) {
            return RecordOutcome::Used;
        }
        let replacing = self.replacement_block.is_some();
        if !replacing {
            self.context_for(i, false, true);
        }
        let p = self.effective(p);
        let recorded_name = p
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                matches!(ty, "web_search_call" | "tool_search_call")
                    .then(|| ty.trim_end_matches("_call").to_owned())
            });
        let name = recorded_name.clone().unwrap_or_default();
        let call_id = p.get("call_id").and_then(Value::as_str).map(str::to_owned);
        let input = match p.get("arguments").or_else(|| p.get("input")) {
            Some(Value::String(s)) => {
                Some(super::scalar_json(s.text()).unwrap_or_else(|| Value::String(s.clone())))
            }
            Some(v) => Some(v.clone()),
            None => None,
        };
        let command = if is_shell_tool(&name) {
            input
                .as_ref()
                .and_then(|input| input.get("cmd").or_else(|| input.get("command")))
                .map(|c| match c {
                    Value::Array(parts) => parts
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(" "),
                    Value::String(s) => s.text().to_owned(),
                    other => other.to_string(),
                })
        } else {
            None
        };
        let stdin_session = (name == "write_stdin")
            .then(|| {
                input
                    .as_ref()
                    .and_then(|input| input.get("session_id"))
                    .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned))
            })
            .flatten();
        let operation_command = command.clone();
        let Some(bi) = self.owned_block(
            i,
            BlockContent::ToolCall(ToolCall {
                tool: recorded_name,
                call_id: call_id.as_ref().map(|id| NativeId::unicode(id)),
                input,
                command,
            }),
        ) else {
            return RecordOutcome::Used;
        };
        if let Some(c) = call_id {
            if is_shell_tool(&name) || name == "exec" {
                self.direct_operation(&c, &name, operation_command, i, bi);
            }
            let raw_name = raw::canonical_field(
                &self.effective_raw,
                if self.s.legacy {
                    &["name"]
                } else {
                    &["payload", "name"]
                },
            )
            .ok();
            let session = if raw_name.as_deref() == Some(b"\"write_stdin\"") {
                raw::session_argument(&self.effective_raw, self.s.legacy).ok()
            } else {
                None
            };
            self.put(
                Key::CallWitness(c.clone()),
                Cell::Association {
                    call_name: raw_name,
                    session,
                    call_name_origins: self.effective_origins.field("name"),
                    session_origins: self.effective_origins.field("arguments"),
                },
            );
            self.put_call(
                c.clone(),
                CallInfo {
                    at: (i, bi),
                    tool: name.clone(),
                    stdin_session,
                },
            );
            if name == "write_stdin" {
                self.put(Key::PendingStdin(c.clone()), Cell::Position(i, bi));
                self.pin(format!("stdin:{c}"), None, Some(i));
            }
            if name == "exec"
                && (!replacing
                    || (!self.wrapper_is_open(&c)
                        && self.position(Key::Result(c.clone())).is_none()))
            {
                self.wrapper_push(c, i);
            }
        }
        RecordOutcome::Used
    }

    fn output_item(&mut self, p: &Value, at: Option<Timestamp>) -> RecordOutcome<Ignored> {
        if let Some(call) = p.get("call_id").and_then(Value::as_str) {
            if let Some(key) = self.operation_key(call) {
                if let Some(Cell::Operation(mut operation)) = self.get(key.clone()) {
                    let independent_unified = self.unified_operation_proved(&operation);
                    let output = recorded_output(p.get("output"));
                    let text = match &output {
                        Some(ToolOutput::Text(s)) => s.clone(),
                        _ => String::new(),
                    };
                    let (outcome, exit) = output_outcome(
                        operation.tool.as_deref().unwrap_or(""),
                        p.get("output").unwrap_or(&Value::Null),
                        &text,
                    );
                    operation.result_present = !matches!(
                        outcome,
                        ToolOutcome::Unknown {
                            reason: UnknownReason::NoResult
                        }
                    );
                    operation.output = if operation.result_present {
                        output.and_then(|o| match o {
                            ToolOutput::Text(s) => Some(s),
                            _ => None,
                        })
                    } else {
                        None
                    };
                    operation.outcome = operation
                        .result_present
                        .then(|| outcome.as_str().to_owned());
                    operation.exit_code = exit;
                    if independent_unified || !operation.origins.iter().any(|origin| matches!(origin, polylogue_model::tool_item::ToolOrigin::Rollout { tag, .. } if tag == "status")) {
                        operation.duration_ms = native_duration(p);
                    }
                    if let Some(origin) = self.operation_origin() {
                        operation.origins.retain(|o|!matches!(o,polylogue_model::tool_item::ToolOrigin::Rollout{tag,..}if tag=="result"));
                        operation.origins.push(origin);
                    }
                    self.put(key, Cell::Operation(operation));
                }
            }
        }
        if let Some(i) = self.known_owner(p) {
            if !self.own_evidence(i, p, true) {
                return RecordOutcome::Used;
            }
            let effective = self.effective(p);
            let Some(call_id) = effective
                .get("call_id")
                .and_then(Value::as_str)
                .map(str::to_owned)
            else {
                return RecordOutcome::Unrecognized("output without call_id");
            };
            let block = self.replacement_block.take().unwrap_or(0);
            let old = match self.get(Key::Block(i, block)) {
                Some(Cell::Block(block)) => match block.content {
                    BlockContent::ToolResult(result) => Some(result),
                    _ => None,
                },
                _ => None,
            };
            let old_call = old.as_ref().and_then(|result| result.call_id.clone());
            if old_call.as_ref().map(NativeId::as_str) != Some(call_id.as_str()) {
                if let Some(old_call) = old_call {
                    if self.position(Key::Result(old_call.as_str().to_owned())) == Some((i, block))
                    {
                        self.remove(Key::Result(old_call.as_str().to_owned()));
                    }
                }
                self.remove_evidence(i, 1, 0);
            }
            let recorded = recorded_output(effective.get("output"));
            if effective.get("output").is_some() && recorded.is_none() {
                self.s.normalization_incomplete = true;
            }
            let text = match &recorded {
                Some(ToolOutput::Text(text)) => text.as_str(),
                _ => "",
            };
            let tool = self
                .call(&call_id)
                .map(|call| call.tool)
                .unwrap_or_default();
            let output = effective.get("output").cloned().unwrap_or(Value::Null);
            let (outcome, exit_code, duration) = if self.get(Key::Evidence(i, 1, 0)).is_some() {
                if let Some(result) = old.as_ref() {
                    (result.outcome.clone(), result.exit_code, result.duration_ms)
                } else {
                    let (outcome, exit) = output_outcome(&tool, &output, text);
                    (outcome, exit, native_duration(&effective))
                }
            } else {
                let (outcome, exit) = output_outcome(&tool, &output, text);
                (outcome, exit, native_duration(&effective))
            };
            self.edit_block(i, block, |value| {
                *value = BlockContent::ToolResult(ToolResult {
                    data: None,
                    call_id: Some(NativeId::unicode(&call_id)),
                    outcome: outcome.clone(),
                    exit_code,
                    duration_ms: duration,
                    output: recorded.clone(),
                })
            });
            self.witness_block(i, block, super::presence::Owner::Primary);
            self.put(Key::Result(call_id.clone()), Cell::Position(i, block));
            self.association_evidence(i, &call_id, &effective);
            if let Some(slot) = self.current_item_slot(i, &effective) {
                self.result_chunk(i, block, slot, &recorded);
            }
            if let Some((target, bi)) = self.position(Key::ContinuationTarget(call_id.clone())) {
                if !self.require_result(target, bi) {
                    return RecordOutcome::Used;
                }
                if self.own_evidence(target, &effective, false) {
                    self.association_evidence(target, &call_id, &effective);
                    if let Some(slot) = self.current_item_slot(target, &effective) {
                        self.result_chunk(target, bi, slot, &recorded);
                    }
                    if exit_code.is_some() {
                        self.edit_block(target, bi, |value| {
                            if let BlockContent::ToolResult(result) = value {
                                result.outcome = outcome.clone();
                                result.exit_code = exit_code;
                            }
                        });
                    }
                }
            }
            return RecordOutcome::Used;
        }
        let Some(call_id) = p.get("call_id").and_then(Value::as_str).map(str::to_owned) else {
            return RecordOutcome::Unrecognized("output without call_id");
        };
        let output = p.get("output").cloned().unwrap_or(Value::Null);
        let recorded = recorded_output(p.get("output"));
        if p.get("output").is_some() && recorded.is_none() {
            self.s.normalization_incomplete = true;
        }
        let text = match &recorded {
            Some(ToolOutput::Text(text)) => text.clone(),
            _ => String::new(),
        };
        let call = self.call(&call_id);
        let tool = call.as_ref().map(|c| c.tool.clone()).unwrap_or_default();
        if let Some(Cell::Position(i, _)) = self.remove(Key::PendingStdin(call_id.clone())) {
            if call.is_none() {
                self.missing("pending stdin call");
                return RecordOutcome::Used;
            }
            self.pin(format!("stdin:{call_id}"), Some(i), None);
        }
        let status_witness = match self.remove(Key::StatusWitness(call_id.clone())) {
            Some(Cell::Evidence(unit)) => Some(unit),
            _ => None,
        };
        let duration = status_witness
            .as_ref()
            .and_then(unit_item)
            .as_ref()
            .and_then(native_duration)
            .or_else(|| native_duration(p));
        let (outcome, exit_code) = match self.remove(Key::Outcome(call_id.clone())) {
            Some(Cell::Outcome(o, r, e)) => (outcome_of_words(&o, r.as_deref()), e),
            _ => output_outcome(&tool, &output, &text),
        };
        self.wrapper_remove(&call_id);
        // Running commands (spec/sources/codex.md, running commands): "Process running with
        // session ID N" is not a result; a `write_stdin` on that session continues the command,
        // and the one that reports the exit sets the command's outcome.
        let running_id = super::running_session(&text).map(str::to_owned);
        let (outcome, exit_code) =
            if running_id.is_some() && (is_shell_tool(&tool) || tool == "write_stdin") {
                (
                    ToolOutcome::Unknown {
                        reason: UnknownReason::NoResult,
                    },
                    None,
                )
            } else {
                (outcome, exit_code)
            };
        if tool == "write_stdin" {
            let session = self.call(&call_id).and_then(|c| c.stdin_session);
            if let Some(target) = session
                .as_ref()
                .and_then(|s| self.position(Key::Running(s.clone())))
            {
                if !self.require_result(target.0, target.1) {
                    return RecordOutcome::Used;
                }
                self.put(Key::Continuation(call_id.clone()), Cell::Present);
                self.put(
                    Key::ContinuationTarget(call_id.clone()),
                    Cell::Position(target.0, target.1),
                );
                self.own_evidence(target.0, p, false);
                self.association_evidence(target.0, &call_id, p);
                if let Some(slot) = self.current_item_slot(target.0, p) {
                    self.result_chunk(target.0, target.1, slot, &recorded);
                }
                if exit_code.is_some() {
                    self.edit_block(target.0, target.1, |b| {
                        if let BlockContent::ToolResult(r) = b {
                            r.outcome = outcome.clone();
                            r.exit_code = exit_code;
                        }
                    });
                    if let Some(s) = &session {
                        self.remove(Key::Running(s.clone()));
                        self.pin(format!("running:{s}"), Some(target.0), None);
                    }
                }
            }
        }
        let native = p.get("id").and_then(Value::as_str);
        let Some(id) = self.message_id(native, p) else {
            return RecordOutcome::Unrecognized("output before session_meta");
        };
        let mut m = Message::new(id, item_id_kind(p), Role::Tool, WrittenBy::Harness);
        m.at = at.map(provider_time);
        if let Err(e) = m.push(
            BlockSlot::Item(0),
            BlockContent::ToolResult(ToolResult {
                data: None,
                call_id: Some(NativeId::unicode(&call_id)),
                outcome,
                exit_code,
                duration_ms: duration,
                output: recorded,
            }),
        ) {
            self.failure = Some(e.to_string());
            return RecordOutcome::Used;
        }
        self.push_message(m);
        let result_i = self.s.messages - 1;
        self.put(Key::Result(call_id.clone()), Cell::Position(result_i, 0));
        self.own_evidence(result_i, p, true);
        if let Some(native) = p.get("id").and_then(Value::as_str) {
            self.put(
                Key::EvidenceItem(result_i, native.into()),
                Cell::Position(0, 0),
            );
            self.put(self.native_key(native), Cell::Position(result_i, 0));
        }
        self.association_evidence(result_i, &call_id, p);
        if let Some(slot) = self.current_item_slot(result_i, p) {
            let output = match self.get(Key::Block(result_i, 0)) {
                Some(Cell::Block(block)) => match block.content {
                    BlockContent::ToolResult(result) => result.output,
                    _ => None,
                },
                _ => None,
            };
            self.result_chunk(result_i, 0, slot, &output);
        }
        if let Some(unit) = status_witness {
            self.witness_block_raw(
                result_i,
                0,
                &unit.canonical,
                &unit.origins,
                super::presence::Owner::Status,
            );
            self.evidence_unit(result_i, 1, 0, unit);
        }
        if let Some(rid) = running_id.filter(|_| is_shell_tool(&tool)) {
            let i = self.s.messages - 1;
            let old = self.position(Key::Running(rid.clone())).map(|p| p.0);
            self.pin(format!("running:{rid}"), old, Some(i));
            self.put(Key::Running(rid), Cell::Position(i, 0));
        }
        if self.s.legacy {
            self.s.output_since_open = true;
        }
        RecordOutcome::Used
    }

    fn operation_origin(&self) -> Option<polylogue_model::tool_item::ToolOrigin> {
        let range = self
            .current_origins
            .advance(&mut self.current_origins.cursor())
            .ok()??;
        let value: Value = serde_json::from_slice(&self.current_raw).ok()?;
        let item = if self.s.legacy {
            &value
        } else {
            value.get("payload")?
        };
        let ty = item.get("type")?.as_str()?;
        let tag = if ty.ends_with("_output") {
            "result"
        } else if ty == "item_completed" {
            "status"
        } else {
            "call"
        };
        Some(polylogue_model::tool_item::ToolOrigin::Rollout {
            tag: tag.into(),
            start: range.start,
            end: range.end,
            raw_hash: *blake3::hash(&self.current_raw).as_bytes(),
        })
    }
    fn operation_key(&mut self, call: &str) -> Option<Key> {
        let call = self.call(call)?;
        let Cell::Text(id) = self.get(Key::OperationBlock(call.at.0, call.at.1))? else {
            return None;
        };
        let provisional = id.starts_with("operation:provisional-v1:");
        Some(Key::Operation(format!(
            "{}:{id}",
            if provisional { 0 } else { 1 }
        )))
    }
    fn completed_operation(
        &mut self,
        payload: &Value,
        item: &Value,
    ) -> Option<polylogue_model::tool_item::Operation> {
        let invocation = item.get("id")?.as_str()?;
        let turn = payload
            .get("turn_id")
            .and_then(Value::as_str)
            .or(self.s.turn.as_deref());
        let key = if let Some(turn) = turn {
            Key::Operation(format!(
                "1:{}",
                polylogue_model::tool_item::native_id(
                    &self.s.session.as_ref()?.id,
                    turn,
                    invocation
                )
                .ok()?
            ))
        } else {
            self.operation_key(invocation)?
        };
        match self.get(key) {
            Some(Cell::Operation(operation)) => Some(operation),
            _ => None,
        }
    }
    fn completed_direct_call(&mut self, payload: &Value, item: &Value) -> bool {
        let Some(operation) = self.completed_operation(payload, item) else {
            return false;
        };
        let Some(call) = self
            .call(&operation.invocation)
            .filter(|c| c.tool != "exec")
        else {
            return false;
        };
        operation.wrapper.is_none()
            && operation.message
                == self
                    .header(call.at.0)
                    .map(|(m, _)| m.id.as_str().to_owned())
            && operation.message.is_some()
    }
    fn unified_operation_proved(
        &mut self,
        operation: &polylogue_model::tool_item::Operation,
    ) -> bool {
        if !operation.native
            || operation.kind != "CommandExecution"
            || operation.correspondence != "established"
            || operation.agent_class != "unknown"
            || (operation.result_present
                && !operation.origins.iter().any(|origin| {
                    matches!(origin,
                polylogue_model::tool_item::ToolOrigin::Rollout { tag, .. } if tag == "result")
                }))
        {
            return false;
        }
        let Ok(id) = operation.id() else {
            return false;
        };
        let Some(Cell::UnifiedOperation { raw_hash, source }) = self.get(Key::UnifiedOperation(id))
        else {
            return false;
        };
        unified_source(&source)
            && operation.origins.iter().any(|origin| {
                matches!(origin,
            polylogue_model::tool_item::ToolOrigin::Rollout { tag, raw_hash: current, .. }
                if tag == "status" && current == &raw_hash)
            })
    }
    fn direct_operation(
        &mut self,
        call: &str,
        tool: &str,
        command: Option<String>,
        message: usize,
        block: usize,
    ) {
        let Some(session) = self.s.session.as_ref().map(|s| s.id.clone()) else {
            return;
        };
        let Some(origin) = self.operation_origin() else {
            return;
        };
        let Some(Cell::Block(source_block)) = self.get(Key::Block(message, block)) else {
            return;
        };
        let anchor = source_block.id.as_str().to_owned();
        // An exec wrapper has a proved request block, not a native
        // CommandExecution item ID. Keep that constructor distinction.
        let native = tool != "exec" && self.s.turn.is_some();
        let mut operation = polylogue_model::tool_item::Operation {
            kind: if tool == "exec" {
                "CodeExecution"
            } else {
                "CommandExecution"
            }
            .into(),
            tool: Some(tool.into()),
            native_session: self.s.session.as_ref().unwrap().native_id.clone(),
            direct_anchors: vec![],
            session,
            turn: self.s.turn.clone(),
            invocation: call.into(),
            native,
            provisional_anchor: (!native).then_some(anchor),
            agent_class: "agent".into(),
            correspondence: if self.s.turn.is_some() {
                "established"
            } else {
                "unresolved"
            }
            .into(),
            call_present: true,
            command: command.map(polylogue_model::tool_item::Command::Text),
            result_present: false,
            output: None,
            outcome: None,
            exit_code: None,
            duration_ms: None,
            message: self.header(message).map(|(m, _)| m.id.as_str().to_owned()),
            wrapper: None,
            wrapper_overlaps: vec![],
            origins: vec![origin],
        };
        if let Ok(id) = operation.id() {
            if let Some(Cell::Operation(mut previous)) = self.get(Key::Operation(format!("1:{id}")))
            {
                if previous.native && previous.agent_class != "agent" {
                    // A call supplies its own Message association, not a new
                    // classification or result for an earlier native command.
                    if self.unified_operation_proved(&previous) {
                        previous.tool = operation.tool;
                        previous.message = operation.message;
                        previous.origins.extend(operation.origins);
                    } else {
                        self.s.normalization_incomplete = true;
                    }
                    operation = previous;
                }
            }
            self.put(Key::OperationBlock(message, block), Cell::Text(id));
        }
        if let Ok(id) = operation.id() {
            self.put(
                Key::Operation(format!("{}:{id}", if operation.native { 1 } else { 0 })),
                Cell::Operation(operation),
            );
        }
    }

    fn command_operation(&mut self, payload: &Value, item: &Value, _at: Option<Timestamp>) -> bool {
        // Only the pinned core shape establishes native correspondence. Older
        // adapter records retain their existing Message interpretation below.
        if !item.get("command").is_some_and(Value::is_array) {
            self.s.normalization_incomplete = true;
            return false;
        }
        let Some(session) = self.s.session.as_ref().map(|s| s.id.clone()) else {
            return false;
        };
        if payload.get("thread_id").and_then(Value::as_str)
            != self.s.session.as_ref().map(|s| s.native_id.as_str())
        {
            self.s.normalization_incomplete = true;
            return false;
        }
        let turn = payload
            .get("turn_id")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let mut cursor = self.current_origins.cursor();
        let range = match self.current_origins.advance(&mut cursor) {
            Ok(Some(r)) => r,
            _ => {
                self.failure = Some("native operation has no retained origin".into());
                return false;
            }
        };
        let origin = polylogue_model::tool_item::ToolOrigin::Rollout {
            tag: "status".into(),
            start: range.start,
            end: range.end,
            raw_hash: *blake3::hash(&self.current_raw).as_bytes(),
        };
        let mut operation = match super::history::command(
            item,
            session,
            &self.s.session.as_ref().unwrap().native_id,
            turn,
            origin,
            false,
        ) {
            Ok(v) => v,
            Err(error) => {
                self.s.normalization_incomplete = true;
                let _ = error;
                return false;
            }
        };
        if operation.agent_class == "agent"
            && payload.get("started_at_ms").is_none_or(Value::is_null)
            && item.get("exit_code").is_none_or(Value::is_null)
            && item.get("process_id").is_none_or(Value::is_null)
        {
            operation.agent_class = "unknown".into();
            operation.result_present = false;
            operation.output = None;
            operation.outcome = None;
            operation.exit_code = None;
            operation.duration_ms = None;
        }
        let previous = self
            .operation_key(&operation.invocation)
            .and_then(|key| self.get(key));
        if let Some(Cell::Operation(previous)) = previous {
            if previous.kind == "CodeExecution" {
                self.failure=Some("CommandExecution ID aliases its code wrapper; constructor correspondence is conflicting".into());
                return false;
            }
            if !previous.native || previous.turn == operation.turn {
                if operation.native && !previous.native {
                    if let Ok(anchor) = previous.id() {
                        operation.direct_anchors.push(anchor);
                    }
                }
                if !operation.native {
                    operation.provisional_anchor = previous.provisional_anchor;
                }
                if operation.native && previous.native && operation.turn == previous.turn
                    && item.get("source").and_then(Value::as_str).is_some_and(unified_source)
                    && previous.origins.iter().any(|origin| matches!(origin,
                        polylogue_model::tool_item::ToolOrigin::Rollout { tag, .. } if tag == "result")) {
                    // Preserve only the separately recorded response output;
                    // unknown native attribution supplies no result of its own.
                    operation.result_present = previous.result_present;
                    operation.output = previous.output.clone();
                    operation.outcome = previous.outcome.clone();
                    operation.exit_code = previous.exit_code;
                    operation.duration_ms = previous.duration_ms;
                }
                operation.tool = previous.tool;
                operation.message = previous.message;
                operation.origins.splice(0..0,previous.origins.into_iter().filter(|origin|!matches!(origin,polylogue_model::tool_item::ToolOrigin::Rollout{tag,..}if tag=="status")));
            }
        }
        if !operation.native && operation.provisional_anchor.is_none() {
            let hash = blake3::hash(&self.current_raw).to_hex().to_string();
            let count = self.number(Key::Occurrence(format!("native-operation:{hash}")));
            self.put(
                Key::Occurrence(format!("native-operation:{hash}")),
                Cell::Number(count + 1),
            );
            operation.provisional_anchor = Some(format!("source-record:{hash}:{count}"));
        }
        operation.native_session = self.s.session.as_ref().unwrap().native_id.clone();
        let mut wrappers = Vec::new();
        if operation.message.is_none() {
            for (_, wrapper) in self.wrappers() {
                if let Some(call) = self.call(&wrapper) {
                    if let Some(Cell::Text(id)) =
                        self.get(Key::OperationBlock(call.at.0, call.at.1))
                    {
                        let key = Key::Operation(format!(
                            "{}:{id}",
                            if id.starts_with("operation:provisional-v1:") {
                                0
                            } else {
                                1
                            }
                        ));
                        if let Some(Cell::Operation(owner)) = self.get(key) {
                            if owner.turn == operation.turn && owner.turn.is_some() {
                                wrappers.push((wrapper, id, owner.origins));
                            }
                        }
                    }
                }
            }
        }
        // CommandExecution has no parent-call field in this pinned schema.
        // Temporal overlap supports suppression, never a Message/parent edge.
        operation.wrapper = None;
        operation.wrapper_overlaps = wrappers.iter().map(|(_, id, _)| id.clone()).collect();
        for (raw, id, origins) in wrappers {
            for origin in origins {
                if let polylogue_model::tool_item::ToolOrigin::Rollout {
                    tag,
                    start,
                    end,
                    raw_hash,
                } = origin
                {
                    if tag == "call" {
                        operation
                            .origins
                            .push(polylogue_model::tool_item::ToolOrigin::Rollout {
                                tag: format!("wrapper_overlap:{id}"),
                                start,
                                end,
                                raw_hash,
                            });
                    }
                }
            }
            self.child(raw, format!("~native:{}", operation.invocation));
        }
        match operation.id() {
            Ok(id) => {
                if operation.native
                    && operation.correspondence == "established"
                    && item
                        .get("source")
                        .and_then(Value::as_str)
                        .is_some_and(unified_source)
                    && valid_command_duration(item)
                {
                    self.put(
                        Key::UnifiedOperation(id.clone()),
                        Cell::UnifiedOperation {
                            raw_hash: *blake3::hash(&self.current_raw).as_bytes(),
                            source: item.get("source").and_then(Value::as_str).unwrap().into(),
                        },
                    );
                }
                self.put(
                    Key::Operation(format!("{}:{id}", if operation.native { 1 } else { 0 })),
                    Cell::Operation(operation),
                );
            }
            Err(error) => self.failure = Some(error.to_string()),
        };
        true
    }

    fn item_completed(&mut self, p: &Value, at: Option<Timestamp>) -> RecordOutcome<Ignored> {
        let original = p.get("item").cloned().unwrap_or(Value::Null);
        // Every replacement constructor for a claimed native scope removes
        // the old proof before validation, including conflicting item kinds.
        if let (Some(session), Some(turn), Some(id)) = (
            self.s.session.as_ref(),
            p.get("turn_id").and_then(Value::as_str),
            original.get("id").and_then(Value::as_str),
        ) {
            if let Ok(id) = polylogue_model::tool_item::native_id(&session.id, turn, id) {
                let key = Key::UnifiedOperation(id);
                let conflicting = self.get(key.clone()).is_some()
                    && original.get("type").and_then(Value::as_str) != Some("CommandExecution");
                self.remove(key);
                if conflicting {
                    self.s.normalization_incomplete = true;
                    return RecordOutcome::Unrecognized("native command constructor changed");
                }
            }
        }
        let item = self.preview_item(&original);
        let ty = item.get("type").and_then(Value::as_str).unwrap_or("");
        let native_admitted = ty == "CommandExecution" && self.command_operation(p, &item, at);
        if ty == "CommandExecution"
            && !native_admitted
            && item.get("command").is_some_and(Value::is_array)
        {
            return RecordOutcome::Unrecognized(
                "unsupported or conflicting native CommandExecution",
            );
        }
        if native_admitted {
            let direct = self.completed_direct_call(p, &item);
            if direct
                && self
                    .completed_operation(p, &item)
                    .is_some_and(|operation| {
                        // This completion must supply its own full scope and
                        // status record; an unscoped correction cannot borrow
                        // a previously retained native proof via current turn.
                        p.get("turn_id").and_then(Value::as_str) == operation.turn.as_deref()
                            && p.get("turn_id").and_then(Value::as_str).is_some()
                            && p.get("thread_id").and_then(Value::as_str) == Some(operation.native_session.as_str())
                            && item.get("id").and_then(Value::as_str) == Some(operation.invocation.as_str())
                            && operation.origins.iter().any(|origin| matches!(origin,
                                polylogue_model::tool_item::ToolOrigin::Rollout { tag, raw_hash, .. }
                                    if tag == "status" && raw_hash == blake3::hash(&self.current_raw).as_bytes()))
                            && self.unified_operation_proved(&operation)
                            && self.call(&operation.invocation).is_some_and(|call| {
                                matches!(self.get(Key::OperationBlock(call.at.0, call.at.1)),
                                    Some(Cell::Text(id)) if operation.id().ok().as_ref() == Some(&id))
                            })
                    })
            {
                // Supported unified lifecycle evidence has uncertain operation
                // attribution, not missing independently retained Message text.
                // Do not project its native status into the call/output body.
                return self.event("unclassified_command_item", at, false, p);
            }
            let owned = self
                .completed_operation(p, &item)
                .is_some_and(|operation| operation.wrapper.is_some());
            if !direct && !owned {
                return RecordOutcome::Used;
            }
        }
        match ty {
            "CommandExecution" | "FileChange" | "McpToolCall" => {
                if ty == "CommandExecution"
                    && item
                        .get("source")
                        .and_then(Value::as_str)
                        .is_some_and(|source| source != "agent")
                {
                    self.s.normalization_incomplete = true;
                    return self.event("non_agent_command_item", at, false, p);
                }
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| {
                        item.get("id")
                            .and_then(Value::as_str)
                            .filter(|id| self.call(id).is_some())
                            .map(str::to_owned)
                    });
                let is_plain_call = if native_admitted {
                    self.completed_direct_call(p, &item)
                } else {
                    call_id
                        .as_ref()
                        .and_then(|c| self.call(c))
                        .is_some_and(|c| c.tool != "exec")
                };
                if is_plain_call {
                    if let Some(c) = call_id {
                        let result = self.position(Key::Result(c.clone()));
                        let previous = result
                            .and_then(|(i, _)| self.get(Key::Evidence(i, 1, 0)))
                            .or_else(|| self.get(Key::StatusWitness(c.clone())));
                        let mut witness = EvidenceUnit {
                            tag: "status".into(),
                            canonical: self.current_raw.clone(),
                            whole_record: true,
                            origins: self.current_origins.at_path(&["payload", "item"]),
                        };
                        let mut effective = item.clone();
                        if let Some(Cell::Evidence(old)) = previous {
                            let id_path = ["payload", "item", "id"];
                            let turn_path = ["payload", "turn_id"];
                            let same_id = raw::field(&old.canonical, &id_path)
                                .ok()
                                .zip(raw::field(&self.current_raw, &id_path).ok())
                                .is_some_and(|(a, b)| a == b);
                            let same_turn = raw::field(&old.canonical, &turn_path).ok()
                                == raw::field(&self.current_raw, &turn_path).ok();
                            if same_id && same_turn {
                                match raw::overlay(
                                    &old.canonical,
                                    &self.current_raw,
                                    &["payload", "item"],
                                ) {
                                    Ok((canonical, fields)) => {
                                        witness.origins = old.origins.overlay(
                                            &witness.origins,
                                            &fields.retained,
                                            &fields.provided,
                                        );
                                        witness.whole_record = canonical == self.current_raw;
                                        witness.canonical = canonical;
                                        effective = fields.value;
                                        if !fields.conflicts.is_empty() {
                                            if let Some((message, _)) = result {
                                                self.conflicts.push(NativeConflict {
                                                    message,
                                                    fields: fields.conflicts,
                                                });
                                            } else {
                                                self.event(
                                                    "native_status_conflict",
                                                    at,
                                                    false,
                                                    &serde_json::json!({"fields":fields.conflicts})
                                                        .into(),
                                                );
                                            }
                                        }
                                    }
                                    Err(error) => {
                                        self.failure = Some(error);
                                        return RecordOutcome::Used;
                                    }
                                }
                            }
                        }
                        let (outcome, exit) = native_outcome(&effective);
                        let duration = native_duration(&effective);
                        if effective
                            .get("duration")
                            .is_some_and(|value| !value.is_null())
                            && duration.is_none()
                        {
                            self.s.normalization_incomplete = true;
                        }
                        if let Some((mi, bi)) = result {
                            if !self.require_result(mi, bi) {
                                return RecordOutcome::Used;
                            }
                            self.edit_block(mi, bi, |block| {
                                if let BlockContent::ToolResult(result) = block {
                                    result.outcome = outcome.clone();
                                    result.exit_code = exit;
                                    result.duration_ms = duration;
                                }
                            });
                            self.witness_block_raw(
                                mi,
                                bi,
                                &witness.canonical,
                                &witness.origins,
                                super::presence::Owner::Status,
                            );
                            self.evidence_unit(mi, 1, 0, witness);
                        } else {
                            let (o, r) = outcome_words(&outcome);
                            self.put(Key::StatusWitness(c.clone()), Cell::Evidence(witness));
                            self.put(Key::Outcome(c), Cell::Outcome(o, r, exit));
                        }
                    }
                    return RecordOutcome::Used;
                }
                // A turn-scoped native operation does not prove a Message owner.
                // Preserve ambiguous operations as session evidence until the target
                // has a session-owned tool projection. Never attach to latest wrapper.
                let named = item
                    .get("parent_call_id")
                    .or_else(|| item.get("call_id"))
                    .and_then(Value::as_str)
                    .filter(|c| self.wrapper_is_open(c))
                    .map(str::to_owned);
                let owner = if native_admitted {
                    None
                } else {
                    named.or_else(|| self.wrapper_one())
                };
                let prior_owner = self.known_owner(&item);
                let wrapper_message = if let Some(place) = owner.as_ref() {
                    match self.call(place) {
                        Some(call) => Some(call.at.0),
                        None => {
                            self.missing("wrapper call");
                            return RecordOutcome::Used;
                        }
                    }
                } else {
                    None
                };
                let mi = prior_owner.or(wrapper_message);
                let Some(mi) = mi else {
                    self.s.normalization_incomplete = true;
                    let child_id = item.get("id").and_then(Value::as_str).unwrap_or("");
                    // Counting suppression is not an asserted parent edge. This
                    // affected-wrapper fanout still needs owner scheduling.
                    for (_, w) in self.wrappers() {
                        self.child(w, format!("~unowned:{child_id}"));
                    }
                    return self.event("unowned_tool_item", at, false, p);
                };
                if !self.own_evidence(mi, &item, true) {
                    return RecordOutcome::Used;
                }
                let replacing = self.replacement_block.is_some();
                let item = self.effective(&item);
                let (outcome, exit) = native_outcome(&item);
                let duration = native_duration(&item);
                if item.get("duration").is_some_and(|value| !value.is_null()) && duration.is_none()
                {
                    self.s.normalization_incomplete = true;
                }
                let child_id = item
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let command = item
                    .get("command")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let tool = if ty == "CommandExecution" {
                    "exec_command"
                } else if ty == "FileChange" {
                    "apply_patch"
                } else {
                    "mcp"
                };
                let Some(call_block) = self.owned_block(
                    mi,
                    BlockContent::ToolCall(ToolCall {
                        tool: Some(tool.into()),
                        call_id: Some(NativeId::unicode(&child_id)),
                        input: Some(item.clone()),
                        command,
                    }),
                ) else {
                    return RecordOutcome::Used;
                };
                if let Some(session) = self.s.session.as_ref() {
                    if let Some(turn) = p
                        .get("turn_id")
                        .and_then(Value::as_str)
                        .or(self.s.turn.as_deref())
                    {
                        if let Ok(id) =
                            polylogue_model::tool_item::native_id(&session.id, turn, &child_id)
                        {
                            self.put(Key::OperationBlock(mi, call_block), Cell::Text(id));
                        }
                    }
                }
                let result = BlockContent::ToolResult(ToolResult {
                    data: None,
                    call_id: Some(NativeId::unicode(&child_id)),
                    outcome,
                    exit_code: exit,
                    duration_ms: duration,
                    output: recorded_output(
                        item.get("aggregated_output").or_else(|| item.get("output")),
                    ),
                });
                if replacing {
                    self.edit_block(mi, call_block + 1, |content| *content = result);
                } else {
                    self.push_block(mi, result);
                }
                self.witness_block(mi, call_block + 1, super::presence::Owner::Primary);
                if !replacing {
                    if let Some(wrapper) = owner {
                        self.child(wrapper, child_id);
                    }
                }
                RecordOutcome::Used
            }
            "ContextCompaction" => self.event("compaction", at, true, &item),
            "Plan" => self.event("plan", at, false, &item),
            "SubAgentActivity" | "CollabAgentToolCall" | "Extension" => {
                self.event("item_completed", at, false, p)
            }
            _ => {
                self.s.normalization_incomplete = true;
                RecordOutcome::Unrecognized("item_completed type not implemented")
            }
        }
    }

    fn token_count(&mut self, p: &Value, at: Option<Timestamp>) -> RecordOutcome<Ignored> {
        let info = p.get("info").filter(|i| !i.is_null());
        if let Some(rl) = p.get("rate_limits") {
            self.events.push(Event {
                kind: "rate_limit".into(),
                at: at.map(provider_time),
                after_message: None,
                model_saw: Some(false),
                data: rl.clone(),
            });
        }
        let Some(info) = info else {
            self.close_request();
            return RecordOutcome::Used;
        };
        let total = info.get("total_token_usage").cloned();
        if total.is_some() && total == self.s.prev_total {
            // A repeat of the previous running total adds no usage; kept for checking.
            return self.event("token_count_repeat", at, false, p);
        }
        self.s.prev_total = total;
        if let Some(last) = info.get("last_token_usage") {
            let g = |k: &str| last.get(k).and_then(Value::as_u64);
            let cached = g("cached_input_tokens");
            let usage = Usage {
                input: g("input_tokens").map(|i| i.saturating_sub(cached.unwrap_or(0))),
                output: g("output_tokens"),
                cache_read: cached,
                cache_write: g("cache_write_input_tokens"),
                reasoning: g("reasoning_output_tokens"),
                ..Usage::default()
            };
            // The last model message is always unsettled (see `carry`), so the search covers it.
            let target = self.s.open_request.or_else(|| self.last_model());
            match target {
                Some(i) => {
                    use polylogue_model::merge::UsageField as U;
                    for (name, field) in [
                        ("input_tokens", U::Input),
                        ("output_tokens", U::Output),
                        ("cached_input_tokens", U::CacheRead),
                        ("cache_write_input_tokens", U::CacheWrite),
                        ("reasoning_output_tokens", U::Reasoning),
                    ] {
                        let key = format!("usage:{name}");
                        if last.get(name).and_then(Value::as_u64).is_some() {
                            match raw::canonical_field(
                                &self.current_raw,
                                &["payload", "info", "last_token_usage", name],
                            ) {
                                Ok(raw) => {
                                    self.witness_usage(i, name, field, &raw, cached.is_some())
                                }
                                Err(e) => self.failure = Some(e),
                            }
                        } else {
                            self.remove(Key::Presence(i, key));
                        }
                    }
                    self.edit(i, |m| m.usage = Some(usage));
                    match raw::field(&self.current_raw, &["payload", "info"]).and_then(|bytes| {
                        polylogue_model::identity::canonical_record(bytes, &[])
                            .map_err(|e| e.to_string())
                    }) {
                        Ok(raw) => self.fixed_evidence(i, 4, "usage", raw, false),
                        Err(error) => self.failure = Some(error),
                    }
                }
                None => return self.event("token_count_unattached", at, false, p),
            }
        }
        self.close_request();
        RecordOutcome::Used
    }

    fn event_msg(&mut self, p: &Value, at: Option<Timestamp>) -> RecordOutcome<Ignored> {
        let ty = p.get("type").and_then(Value::as_str).unwrap_or("");
        match ty {
            "user_message" => {
                let text = p.get("message").and_then(Value::as_str).unwrap_or("");
                if let Some(index) = self.s.pending_prompt.take() {
                    match self.get(Key::Prompt) {
                        Some(Cell::Text(pending)) if pending == text => {
                            let changed = self
                                .header(index)
                                .is_some_and(|(m, _)| m.written_by != WrittenBy::Human);
                            self.mark_human(index);
                            if changed {
                                self.fixed_evidence(
                                    index,
                                    3,
                                    "context:echo",
                                    self.current_raw.clone(),
                                    true,
                                );
                            }
                            self.remove(Key::Prompt);
                        }
                        Some(Cell::Text(_)) => self.s.pending_prompt = Some(index),
                        _ => self.missing("pending prompt text"),
                    }
                }
                RecordOutcome::Used
            }
            "token_count" => self.token_count(p, at),
            "item_completed" => self.item_completed(p, at),
            "task_started" => {
                self.s.turn = p.get("turn_id").and_then(Value::as_str).map(str::to_owned);
                self.event("turn_start", at, false, p)
            }
            "task_complete" => self.event("turn_end", at, false, p),
            "turn_aborted" => self.event("turn_aborted", at, false, p),
            "context_compacted" => self.event("compaction", at, true, p),
            "thread_goal_updated" => self.event("goal", at, false, p),
            "agent_message" | "thread_settings_applied" => self.event(ty, at, false, p),
            _ => RecordOutcome::Unrecognized("event_msg type"),
        }
    }
    fn interpret(&mut self, rec: &Record<'_>) -> RecordOutcome<Ignored> {
        let v = &rec.value;
        if v.get("record_type").and_then(Value::as_str) == Some("state") {
            return RecordOutcome::Ignored(Ignored::LegacyStateMarker);
        }
        let at = v
            .get("timestamp")
            .and_then(Value::as_str)
            .and_then(parse_timestamp);
        if self.s.session.is_none() && v.get("type").is_none() && v.get("id").is_some() {
            // The 2025 bare header: {id, timestamp, instructions, git}.
            self.s.legacy = true;
            self.raw_context(rec.bytes, &["timestamp"], 2, "legacy:time");
            let id = v.get("id").and_then(Value::as_str).unwrap_or("").to_owned();
            return self.start_session(&id, at, v);
        }
        let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
        if self.s.legacy {
            // 2025 records are bare items without an envelope and without a time.
            return self.response_item(v, None);
        }
        let p = v.get("payload").cloned().unwrap_or(Value::Null);
        match ty {
            "session_meta" if self.s.session.is_none() => {
                let id = p.get("id").and_then(Value::as_str).unwrap_or("").to_owned();
                self.start_session(&id, at, &p)
            }
            "session_meta" => self.event("repeated_session_meta", at, false, &p),
            "turn_context" => {
                self.s.turn = p.get("turn_id").and_then(Value::as_str).map(str::to_owned);
                if let Some(model) = p.get("model") {
                    self.s.model = model.as_str().map(str::to_owned);
                    if !model.is_string() {
                        self.s.normalization_incomplete = true;
                    }
                    let context_raw = self.current_raw.clone();
                    self.raw_context(&context_raw, &["payload", "model"], 1, "context:model");
                }
                self.event("turn_context", at, false, &p)
            }
            "response_item" => self.response_item(&p, at),
            "event_msg" => self.event_msg(&p, at),
            "compacted" => self.event("compaction", at, true, &p),
            "world_state" => self.event("world_state", at, true, &p),
            "token_usage_record" | "inter_agent_communication_metadata" => {
                self.event(ty, at, false, &p)
            }
            _ => RecordOutcome::Unrecognized("record type"),
        }
    }
}

#[cfg(test)]
mod legacy_time_tests {
    use super::*;

    #[test]
    fn explicitly_timed_first_message_keeps_its_own_witness_in_legacy_context() {
        let memory = MemoryReadView::default();
        let origin = Origin::new("codex").unwrap();
        let session_id = SessionId::new(&origin, "synthetic-time").unwrap();
        let mut session = Session::new(
            session_id.clone(),
            origin,
            "synthetic-time",
            SessionKind::Main,
        )
        .unwrap();
        session.started_at = Some(provider_time(Timestamp(0)));
        let resume = Resume {
            session: Some(session),
            legacy: true,
            ..Default::default()
        };
        let mut parser = KeyedRecords::new(&memory, resume);
        let source_time = b"\"2026-09-02T10:00:01.000Z\"";
        parser.current_time_raw = Some(source_time.to_vec());
        parser.current_origins = Origins::Record(RecordRange::new(100, 200).unwrap());
        let mut message = Message::new(
            MessageId::new(session_id, "first").unwrap(),
            IdKind::Provider,
            Role::User,
            WrittenBy::Human,
        );
        message.at = Some(provider_time(
            parse_timestamp("2026-09-02T10:00:01.000Z").unwrap(),
        ));
        let expected = message.at;
        parser.push_message(message);
        let delta = parser.finish_delta().unwrap();
        let Some(Some(Cell::Header { message, .. })) = delta.cells.get(&Key::Message(0)) else {
            panic!("missing first Message");
        };
        assert_eq!(message.at, expected);
        let Some(Some(Cell::Witness(witness))) = delta.cells.get(&Key::Presence(0, "at".into()))
        else {
            panic!("missing first time witness");
        };
        assert_eq!(witness.canonical, source_time);
        assert_eq!(
            witness.origins,
            Origins::Located(super::super::selection::SelectedRaw {
                record: RecordRange::new(100, 200).unwrap(),
                path: vec![super::super::selection::PathPart::Key("timestamp".into())]
            })
        );
    }
}

//! Disposable Codex state, independently addressable by provider-owned keys.
//!
//! A ReadView is pinned to one committed input/chain/parser basis by its caller.
//! The overlay contains only touched keys; finish returns mutations, not a
//! serialized prefix or materialized message suffix. Payload/fanout scheduling
//! and atomic application with normalized rows belong to the caller.
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
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Cell {
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
    Association {
        call_name: Option<Vec<u8>>,
        session: Option<Vec<u8>>,
    },
}
/// Raw canonical JSON, tagged by provider membership, never normalized content.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceUnit {
    pub tag: String,
    pub canonical: Vec<u8>,
    pub whole_record: bool,
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
        for (b, v) in blocks.into_iter().enumerate() {
            let next = match v.id.slot() {
                BlockSlot::Item(n) => (*n as usize + 1, 0),
                BlockSlot::Reasoning(n) => (0, *n as usize + 1),
                _ => (0, 0),
            };
            self.put(Key::BlockOrder(i), Cell::Position(next.0, next.1));
            if let BlockContent::ToolCall(c) = &v.content {
                if let Some(id) = &c.call_id {
                    self.put(Key::CallBlock(id.clone(), i, b), Cell::Present);
                }
            }
            self.put(Key::Block(i, b), Cell::Block(v));
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
                self.put(Key::CallBlock(id.clone(), i, n), Cell::Present);
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
        // Capture exact provider bytes before decoded Value can normalize numbers.
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
                        BlockContent::ToolCall(new) => new.call_id.as_deref(),
                        _ => None,
                    };
                    if new != Some(id.as_str()) {
                        self.remove(Key::CallBlock(id.clone(), i, block));
                        if self.call(&id).is_some_and(|call| call.at == (i, block)) {
                            self.wrapper_remove(&id);
                            self.remove(Key::Call(id.clone()));
                            self.remove(Key::CallWitness(id.clone()));
                            if let Some(Cell::Position(message, _)) =
                                self.remove(Key::PendingStdin(id.clone()))
                            {
                                self.pin(format!("stdin:{id}"), Some(message), None);
                            }
                        }
                    }
                }
            }
            if let BlockContent::ToolCall(new) = &content {
                if let Some(id) = &new.call_id {
                    self.put(Key::CallBlock(id.clone(), i, block), Cell::Present);
                }
            }
            self.edit_block(i, block, |value| *value = content);
            Some(block)
        } else {
            self.push_block(i, content)
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
            },
        );
    }
    fn context_for(&mut self, i: usize, source: bool, model: bool) {
        for (wanted, key, slot) in [(source, 0, 1), (model, 1, 2)] {
            if wanted {
                if let Some(Cell::Evidence(unit)) = self.get(Key::Context(key)) {
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
                }),
            );
        }
    }
    fn association_evidence(&mut self, i: usize, call_id: &str, item: &Value) {
        let Some(slot) = self.current_item_slot(i, item) else {
            self.missing("output evidence slot");
            return;
        };
        if let Some(Cell::Association { call_name, session }) =
            self.get(Key::CallWitness(call_id.into()))
        {
            for (offset, tag, value) in [
                (1, "association:call_name", call_name),
                (2, "association:process_session", session),
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
            Some(Value::String(src)) if src == "cli" || src == "vscode" => SessionMode::Interactive,
            Some(Value::String(src)) if src == "exec" => SessionMode::Exec,
            Some(Value::Object(o)) if o.contains_key("subagent") => SessionMode::Subagent,
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
                    m.attrs = Some(serde_json::json!({ "phase": phase }));
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
            Some(Value::Array(items)) => items
                .iter()
                .all(|item| item.is_string() || item.get("text").is_some_and(Value::is_string)),
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
                encrypted: Some(encrypted.is_some() && summary.is_empty()),
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
                Some(serde_json::from_str::<Value>(s).unwrap_or_else(|_| Value::String(s.clone())))
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
                    Value::String(s) => s.clone(),
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
        let Some(bi) = self.owned_block(
            i,
            BlockContent::ToolCall(ToolCall {
                tool: recorded_name,
                call_id: call_id.clone(),
                input,
                command,
            }),
        ) else {
            return RecordOutcome::Used;
        };
        if let Some(c) = call_id {
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
            if old_call.as_deref() != Some(call_id.as_str()) {
                if let Some(old_call) = old_call {
                    if self.position(Key::Result(old_call.clone())) == Some((i, block)) {
                        self.remove(Key::Result(old_call));
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
            let (outcome, exit_code) = if self.get(Key::Evidence(i, 1, 0)).is_some() {
                old.as_ref()
                    .map(|result| (result.outcome.clone(), result.exit_code))
                    .unwrap_or_else(|| output_outcome(&tool, &output, text))
            } else {
                output_outcome(&tool, &output, text)
            };
            self.edit_block(i, block, |value| {
                *value = BlockContent::ToolResult(ToolResult {
                    call_id: Some(call_id.clone()),
                    outcome: outcome.clone(),
                    exit_code,
                    duration_ms: None,
                    output: recorded.clone(),
                })
            });
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
        let (outcome, exit_code) = match self.remove(Key::Outcome(call_id.clone())) {
            Some(Cell::Outcome(o, r, e)) => (outcome_of_words(&o, r.as_deref()), e),
            _ => output_outcome(&tool, &output, &text),
        };
        self.wrapper_remove(&call_id);
        // Running commands (spec/sources/codex.md, running commands): "Process running with
        // session ID N" is not a result; a `write_stdin` on that session continues the command,
        // and the one that reports the exit sets the command's outcome.
        let running_id = text
            .lines()
            .find_map(|l| l.strip_prefix("Process running with session ID "))
            .map(|r| r.trim().to_owned());
        let (outcome, exit_code) = if running_id.is_some() && is_shell_tool(&tool) {
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
                call_id: Some(call_id.clone()),
                outcome,
                exit_code,
                duration_ms: None,
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

    fn item_completed(&mut self, p: &Value, at: Option<Timestamp>) -> RecordOutcome<Ignored> {
        let original = p.get("item").cloned().unwrap_or(Value::Null);
        let item = self.preview_item(&original);
        let ty = item.get("type").and_then(Value::as_str).unwrap_or("");
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
                let (outcome, exit) = native_outcome(&item);
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
                let is_plain_call = call_id
                    .as_ref()
                    .and_then(|c| self.call(c))
                    .is_some_and(|c| c.tool != "exec");
                if is_plain_call {
                    if let Some(c) = call_id {
                        let (o, r) = outcome_words(&outcome);
                        let witness = EvidenceUnit {
                            tag: "status".into(),
                            canonical: self.current_raw.clone(),
                            whole_record: true,
                        };
                        if let Some((mi, bi)) = self.position(Key::Result(c.clone())) {
                            if !self.require_result(mi, bi) {
                                return RecordOutcome::Used;
                            }
                            self.edit_block(mi, bi, |block| {
                                if let BlockContent::ToolResult(result) = block {
                                    result.outcome = outcome.clone();
                                    result.exit_code = exit;
                                }
                            });
                            self.evidence_unit(mi, 1, 0, witness);
                        } else {
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
                let owner = named.or_else(|| self.wrapper_one());
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
                        call_id: Some(child_id.clone()),
                        input: Some(item.clone()),
                        command,
                    }),
                ) else {
                    return RecordOutcome::Used;
                };
                let result = BlockContent::ToolResult(ToolResult {
                    call_id: Some(child_id.clone()),
                    outcome,
                    exit_code: exit,
                    duration_ms: None,
                    output: recorded_output(
                        item.get("aggregated_output").or_else(|| item.get("output")),
                    ),
                });
                if replacing {
                    self.edit_block(mi, call_block + 1, |content| *content = result);
                } else {
                    self.push_block(mi, result);
                }
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
            "task_started" => self.event("turn_start", at, false, p),
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

impl MemoryReadView {
    /// Explicit materialization for inspection/testing. The state-backed
    /// delta route never calls this or scans all messages during finish.
    pub fn message(&self, i: usize) -> Result<Option<Message>, String> {
        let Some(Cell::Header { message, blocks }) = self.cells.get(&Key::Message(i)) else {
            return Ok(None);
        };
        let mut m = message.clone();
        for b in 0..*blocks {
            let Some(Cell::Block(block)) = self.cells.get(&Key::Block(i, b)) else {
                return Err(format!("missing Codex block {i}:{b}"));
            };
            let mut block = block.clone();
            block.id = match block.id.slot() {
                BlockSlot::Item(n) => BlockId::new(m.id.clone(), *n),
                BlockSlot::Reasoning(n) => BlockId::reasoning(m.id.clone(), *n),
                BlockSlot::Attachment(key) => {
                    BlockId::attachment(m.id.clone(), key).map_err(|e| e.to_string())?
                }
            };
            if let BlockContent::ToolResult(result) = &mut block.content {
                if self.cells.contains_key(&Key::PendingResult(i, b)) {
                    let mut chunks = String::new();
                    let mut present = false;
                    for (key, value) in self
                        .cells
                        .range(Key::OutputChunk(i, b, 0)..=Key::OutputChunk(i, b, usize::MAX))
                    {
                        match (key, value) {
                            (Key::OutputChunk(_, _, _), Cell::Text(text)) => {
                                chunks.push_str(text);
                                present = true;
                            }
                            _ => return Err("invalid output chunk".into()),
                        }
                    }
                    result.output = present.then_some(ToolOutput::Text(chunks));
                }
            }
            m.blocks.push(block);
        }
        Ok(Some(m))
    }
    /// Project only explicitly requested messages, applying derived annotations.
    pub fn projected_message(&self, i: usize) -> Result<Option<Message>, String> {
        let Some(mut m) = self.message(i)? else {
            return Ok(None);
        };
        let mut continued = Vec::new();
        let mut children = serde_json::Map::new();
        for b in &m.blocks {
            if let BlockContent::ToolCall(c) = &b.content {
                if let Some(id) = &c.call_id {
                    if self.cells.contains_key(&Key::Continuation(id.clone())) {
                        continued.push(Value::String(id.clone()));
                    }
                    if c.tool.as_deref() == Some("exec") {
                        if let Some(Cell::Number(n)) = self.cells.get(&Key::ChildCount(id.clone()))
                        {
                            if *n > 0 {
                                let mut ids = Vec::new();
                                for j in 0..*n {
                                    let Some(Cell::Text(v)) =
                                        self.cells.get(&Key::Child(id.clone(), j))
                                    else {
                                        return Err("missing wrapper child".into());
                                    };
                                    ids.push(v.clone());
                                }
                                children.insert(id.clone(), serde_json::json!(ids));
                            }
                        }
                    }
                }
            }
        }
        if !continued.is_empty() || !children.is_empty() {
            let attrs = m.attrs.get_or_insert_with(|| serde_json::json!({}));
            if let Some(o) = attrs.as_object_mut() {
                if !continued.is_empty() {
                    o.insert("continuation_calls".into(), Value::Array(continued));
                }
                if !children.is_empty() {
                    o.insert("code_children".into(), Value::Object(children));
                }
            }
        }
        Ok(Some(m))
    }
}

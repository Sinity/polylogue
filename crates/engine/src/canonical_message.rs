//! Final-head Codex and Claude Message materialization through one durable core. Worker computation
//! owns decoding/serialization/hash work; apply owns one bounded writer turn.
//! This certifies source projections only. Effects-v3 dirty updates, full merge,
//! canonical selection, operations, search and exports remain separate work.
//! Retained sources bind immutable identity or an observed chain/context before selectors.
//! Each field replays admitted source records/leaf/descriptor serialization on
//! the worker; this is not a universal CPU/RSS or append-locality bound.
mod claude;
use crate::{state::codex::Attempt, version_proof};
pub use claude::{ClaudeRequest, start as start_claude};
use polylogue_model::{
    merge::{FinishState, RouteKind, Slot},
    source_leaf::{self, Subject},
    *,
};
use polylogue_sources::codex::{
    keyed::{Cell, EvidenceCursor, EvidenceHead, Key, ReadView},
    origins::{self, Origins, RecordRange},
    presence::{Group, Witness},
    projection::{Cursor as SourceCursor, Mode, Step, Unit},
    selection::{self, Binding, FactSelection},
};
use polylogue_store::{
    Error as StoreError, ReadHandle, Writer,
    contributions::{
        BlockKind, BlockPresence, Build, Identity, MessageLayout, Preparation, PreparedProjection,
        ProjectionState, ProjectionStep, SmallField, SmallFieldEntry, Source, SourceProjection,
        SourceRange, SourceRangeEntry,
    },
    parser_state::{Budget, Snapshot, Token},
    values,
};
use serde::{Deserialize, Serialize};
use std::{
    io::{self, Write},
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Debug)]
pub struct Request {
    pub message: usize,
    pub revision: u64,
    pub logical_key: [u8; 32],
    pub source_binding: BlobHash,
}
#[derive(Clone, Copy)]
pub struct Budgets {
    pub read: Budget,
    pub metadata: usize,
    pub value_chunk: usize,
    pub hash_bytes: u64,
}
#[derive(Debug)]
pub enum Error {
    Store(StoreError),
    Source(String),
    Leaf(source_leaf::Error),
    Proof(version_proof::Error),
    Unsupported(&'static str),
    UnsupportedField {
        path: Vec<String>,
        reason: &'static str,
    },
    Cancelled,
    Selection(selection::ValueError<ResolveError>),
    Descriptor(selection::Error),
    ClaudeSelection(
        polylogue_sources::selection::ValueError<
            ResolveError,
            polylogue_sources::claude_code::selection::Error,
        >,
    ),
    ClaudeDescriptor(polylogue_sources::claude_code::selection::EncodeError),
    Evidence(polylogue_model::version_evidence::Error),
    Invariant(&'static str),
}
impl From<StoreError> for Error {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}
impl From<source_leaf::Error> for Error {
    fn from(e: source_leaf::Error) -> Self {
        Self::Leaf(e)
    }
}
impl From<version_proof::Error> for Error {
    fn from(e: version_proof::Error) -> Self {
        Self::Proof(e)
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(e) => write!(f, "{e}"),
            Self::Source(e) => f.write_str(e),
            Self::Leaf(e) => write!(f, "{e}"),
            Self::Proof(e) => write!(f, "{e}"),
            Self::Unsupported(e) | Self::Invariant(e) => f.write_str(e),
            Self::UnsupportedField { path, reason } => write!(f, "{reason}: {path:?}"),
            Self::Selection(e) => write!(f, "{e}"),
            Self::Descriptor(e) => write!(f, "{e}"),
            Self::ClaudeSelection(e) => write!(f, "{e}"),
            Self::ClaudeDescriptor(e) => write!(f, "{e}"),
            Self::Evidence(e) => write!(f, "{e}"),
            Self::Cancelled => f.write_str("Message materialization cancelled"),
        }
    }
}
impl std::error::Error for Error {}
#[derive(Debug)]
pub enum ResolveError {
    Store(StoreError),
    Cancelled,
    Unbound,
}
impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(e) => write!(f, "{e}"),
            Self::Cancelled => f.write_str("origin resolution cancelled"),
            Self::Unbound => f.write_str("origin range is not in candidate source SET"),
        }
    }
}
impl std::error::Error for ResolveError {}
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
struct EvidenceGuard {
    head: EvidenceBasis,
    // Retain the existing disposable evidence-guard encoding for immutable
    // inputs; the typed binding also admits a validated observed chain key.
    #[serde(rename = "source_hash")]
    source_binding: BlobHash,
}
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
enum EvidenceBasis {
    Codex(EvidenceHead),
    Claude(claude::Head),
}
impl EvidenceBasis {
    fn revision(&self) -> u64 {
        match self {
            Self::Codex(h) => h.revision,
            Self::Claude(h) => h.pending.revision,
        }
    }
    fn codex(&self) -> Result<&EvidenceHead> {
        match self {
            Self::Codex(h) => Ok(h),
            _ => Err(Error::Invariant("expected Codex evidence basis")),
        }
    }
}
/// A catalog-selected projection must still name this provider's exact current
/// predecessor, raw evidence head and retained source binding.
pub(crate) fn current_codex_projection(
    source: &SourceProjection,
    token: &Token,
    message: &MessageId,
    head: &EvidenceHead,
    source_binding: BlobHash,
) -> std::result::Result<bool, StoreError> {
    head.validate().map_err(StoreError::Invariant)?;
    let guard: EvidenceGuard = serde_json::from_slice(source.evidence_guard())
        .map_err(|e| StoreError::Invariant(format!("Codex selected source guard: {e}")))?;
    Ok(source.predecessor() == token
        && source.message() == message
        && source.evidence_revision() == head.revision
        && guard
            == EvidenceGuard {
                head: EvidenceBasis::Codex(head.clone()),
                source_binding,
            })
}
enum EvidenceProof {
    Codex(version_proof::Proof),
    Claude(claude::Proof),
}
impl EvidenceProof {
    fn validate(&self, handle: &Handle) -> Result<BlobHash> {
        match self {
            Self::Codex(p)
                if p.token() == handle.build.predecessor()
                    && p.head() == handle.head.codex()?
                    && p.message() == handle.request.message =>
            {
                Ok(p.digest())
            }
            Self::Claude(p) => p.validate(handle),
            _ => Err(Error::Invariant("source proof does not match preparation")),
        }
    }
}
type Result<T> = std::result::Result<T, Error>;
#[derive(Clone)]
struct IdentityOwned {
    session: SessionId,
    origin: Origin,
    native: String,
    native_encoding: Option<String>,
    kind: SessionKind,
    message: MessageId,
    role: Role,
    by: WrittenBy,
}
impl IdentityOwned {
    fn borrowed(&self, sequence: u64) -> Identity<'_> {
        Identity {
            session: &self.session,
            origin: &self.origin,
            native_session: &self.native,
            native_session_encoding: self.native_encoding.as_deref(),
            session_kind: self.kind,
            message: &self.message,
            sequence,
            role: self.role,
            written_by: self.by,
        }
    }
}
#[derive(Clone)]
pub struct Handle {
    identity: IdentityOwned,
    request: Request,
    build: Build,
    head: EvidenceBasis,
    guard: Vec<u8>,
    previous: Option<SourceProjection>,
}
impl Handle {
    pub fn build(&self) -> &Build {
        &self.build
    }
    pub fn request(&self) -> &Request {
        &self.request
    }
}
#[derive(Clone)]
pub struct State {
    handle: Handle,
    preparation: Preparation,
    projection: ProjectionState,
}
impl State {
    pub fn sequence(&self) -> u64 {
        self.projection.sequence
    }
    pub fn active_value(&self) -> Option<&values::Head> {
        self.preparation.active_value.as_ref()
    }
    pub fn active_origin_value(&self) -> Option<&values::Head> {
        self.preparation.active_origin_value.as_ref()
    }
    pub fn projection_ready(&self) -> bool {
        self.preparation.projection.is_some()
    }
}
impl Action {
    pub fn operation(&self) -> &'static str {
        match &self.inner {
            ActionKind::Prepare { .. } => "prepare",
            ActionKind::Existing { .. } => "existing",
            ActionKind::Turn { change, .. } => match change {
                Change::SmallFieldRun(_) => "small_field_run",
                Change::SourceRangeRun(_) => "source_range_run",
                Change::Advance(_) => "advance",
                Change::SourceRange { .. } => "source_range",
                Change::BeginOrigin { .. } => "origin_begin",
                Change::SealOrigin(_) => "origin_seal",
                Change::Layout { .. } => "layout",
                Change::BeginValue { .. } => "value_begin",
                Change::Append { .. } => "value_append",
                Change::Seal(_) => "value_seal",
                Change::Field { .. } => "field",
                Change::SmallField { .. } => "small_field",
                Change::BeginBlock { .. } => "block_begin",
                Change::Command(_) => "tool_command",
                Change::Install(_) => "block_install",
                Change::Finish { .. } => "finish",
                Change::Publish(_) => "source_publish",
            },
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
struct Cursor {
    source: Option<SourceCursor>,
    stage: Stage,
    verified: Option<Verified>,
}
#[derive(Clone, Serialize, Deserialize)]
enum Stage {
    Claude(claude::Cursor),
    Citations(Citations),
    Layout,
    Next,
    MessageField { witness: SourceCursor },
    AwaitPresence { index: usize },
    Text { index: usize },
    Fields { index: usize, field: usize },
    Install { index: usize },
    Proof,
}
#[derive(Clone, Serialize, Deserialize)]
enum Citations {
    Evidence {
        slots: EvidenceCursor,
        current: Option<Key>,
        origins: origins::Cursor,
    },
    Projection {
        source: SourceCursor,
    },
    Message {
        before: SourceCursor,
        after: SourceCursor,
        origins: origins::Cursor,
    },
    Block {
        source: SourceCursor,
        index: usize,
        field: usize,
        origins: origins::Cursor,
    },
}
#[derive(Clone, Serialize, Deserialize)]
struct Verified {
    value_hash: BlobHash,
    value_length: u64,
    descriptor_hash: BlobHash,
    descriptor_length: u64,
}
struct Guard {
    handle: Handle,
    sequence: u64,
    cursor: Option<Vec<u8>>,
    metadata: usize,
    active: Option<values::Head>,
    active_origin: Option<values::Head>,
}
pub struct Action {
    inner: ActionKind,
}
enum ActionKind {
    Prepare {
        identity: IdentityOwned,
        request: Request,
        token: Token,
        head: EvidenceBasis,
        guard: Vec<u8>,
        previous: Option<SourceProjection>,
        allowance: usize,
    },
    Existing {
        token: Token,
        request: Request,
        source: SourceProjection,
        allowance: usize,
    },
    Turn {
        guard: Guard,
        change: Change,
    },
}
enum Change {
    SmallFieldRun(Vec<SmallFieldEntry>),
    SourceRangeRun(Vec<SourceRangeEntry>),
    SourceRange {
        range: SourceRange,
        next: Vec<u8>,
    },
    BeginOrigin {
        length: u64,
    },
    SealOrigin(values::Validation),
    Advance(Vec<u8>),
    Layout {
        layout: MessageLayout,
        next: Vec<u8>,
    },
    BeginValue {
        codec: values::Codec,
        length: u64,
    },
    Append {
        id: values::ValueId,
        offset: u64,
        bytes: Vec<u8>,
    },
    Seal(values::Validation),
    Field {
        slot: Slot,
        time: Option<i64>,
        block: bool,
        supplied: bool,
        next: Vec<u8>,
    },
    SmallField {
        field: SmallField,
        slot: Slot,
        time: Option<i64>,
        block: bool,
        supplied: bool,
        next: Vec<u8>,
    },
    BeginBlock {
        group: u64,
        local: u64,
        kind: OwnedBlockKind,
        presence: BlockPresence,
        next: Vec<u8>,
    },
    Command(Vec<u8>),
    Install(Vec<u8>),
    Finish {
        proof: EvidenceProof,
        finish: FinishState,
        next: Vec<u8>,
    },
    Publish(PreparedProjection),
}
enum OwnedBlockKind {
    Text,
    Thinking {
        encrypted: Option<bool>,
    },
    ToolCall {
        operation: String,
    },
    ToolResult {
        outcome: ToolOutcome,
        exit_code: Option<i64>,
        duration_ms: Option<u64>,
    },
}
impl OwnedBlockKind {
    fn borrowed(&self) -> BlockKind<'_> {
        match self {
            Self::Text => BlockKind::Text,
            Self::Thinking { encrypted } => BlockKind::Thinking {
                encrypted: *encrypted,
            },
            Self::ToolCall { operation } => BlockKind::ToolCall {
                operation_id: operation,
                at: None,
                action: None,
            },
            Self::ToolResult {
                outcome,
                exit_code,
                duration_ms,
            } => BlockKind::ToolResult {
                outcome,
                exit_code: *exit_code,
                duration_ms: *duration_ms,
            },
        }
    }
}
pub enum Applied {
    Prepared(Handle),
    Progress,
    Source(SourceProjection),
    Existing(SourceProjection),
}
fn cancelled(flag: &AtomicBool) -> Result<()> {
    if flag.load(Ordering::Acquire) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
fn read(attempt: &Attempt<'_>, key: &Key) -> Result<Option<Cell>> {
    let result = attempt.read(key);
    if let Some(e) = attempt.take_failure() {
        return Err(e.into());
    }
    result.map_err(Error::Source)
}
fn header(attempt: &Attempt<'_>, message: usize) -> Result<Message> {
    match read(attempt, &Key::Message(message))? {
        Some(Cell::Header { message, .. }) => Ok(message),
        _ => Err(Error::Unsupported("missing Codex Message header")),
    }
}
fn evidence(attempt: &Attempt<'_>, message: usize) -> Result<EvidenceHead> {
    match read(attempt, &Key::EvidenceHead(message))? {
        Some(Cell::EvidenceHead(h)) => {
            if h.incomplete {
                return Err(Error::Unsupported(
                    "Codex source evidence membership is incomplete",
                ));
            }
            h.validate().map_err(Error::Source)?;
            Ok(h)
        }
        _ => Err(Error::Unsupported("missing Codex evidence head")),
    }
}
/// Called off actor with the actor's current reference (if any). Typed guard
/// decoding is here, never in apply. Only exact same-head replay skips work.
pub fn start(
    snapshot: &Snapshot,
    request: &Request,
    current: Option<SourceProjection>,
    budget: Budget,
    allowance: usize,
) -> Result<Action> {
    if snapshot.head.status != "finalizing" || !snapshot.head.finalization_pending {
        return Err(Error::Unsupported(
            "Message materializer currently requires a pinned finalizing source head",
        ));
    }
    let attempt = Attempt::new(snapshot, budget);
    let resumed = attempt.resume();
    if let Some(e) = attempt.take_failure() {
        return Err(e.into());
    }
    let resume = resumed.map_err(Error::Source)?;
    let session = resume
        .session()
        .ok_or(Error::Unsupported("Codex session metadata unavailable"))?;
    let message = header(&attempt, request.message)?;
    let head = evidence(&attempt, request.message)?;
    if message.id.session() != &session.id {
        return Err(Error::Invariant("Codex Message/Session relation mismatch"));
    }
    let typed_guard = EvidenceGuard {
        head: EvidenceBasis::Codex(head.clone()),
        source_binding: request.source_binding,
    };
    let guard = bounded_json(&typed_guard, allowance)?;
    if let Some(previous) = &current {
        let prior: EvidenceGuard = serde_json::from_slice(previous.evidence_guard())
            .map_err(|e| Error::Source(e.to_string()))?;
        if previous.message() != &message.id {
            return Err(Error::Invariant(
                "current source reference names another Message",
            ));
        }
        if previous.predecessor() == &snapshot.head.token && prior != typed_guard {
            return Err(Error::Invariant("same-token source evidence guard changed"));
        }
        if previous.predecessor() == &snapshot.head.token && prior == typed_guard {
            return Ok(Action {
                inner: ActionKind::Existing {
                    token: snapshot.head.token.clone(),
                    request: request.clone(),
                    source: previous.clone(),
                    allowance,
                },
            });
        }
    }
    Ok(Action {
        inner: ActionKind::Prepare {
            identity: IdentityOwned {
                session: session.id.clone(),
                origin: session.origin.clone(),
                native: session.native_id.clone(),
                native_encoding: session.native_id_encoding.clone(),
                kind: session.kind,
                message: message.id,
                role: message.role,
                by: message.written_by,
            },
            request: request.clone(),
            token: snapshot.head.token.clone(),
            head: EvidenceBasis::Codex(head),
            guard,
            previous: current,
            allowance,
        },
    })
}
/// Fixed metadata only; no cursor JSON decode or payload hashing on the actor.
pub fn inspect(writer: &mut Writer, handle: &Handle, allowance: usize) -> Result<State> {
    // Both reads validate this exact existing Build. Inspection cannot reserve
    // rows or allocate a successor; only the initial Prepare action does that.
    let projection = writer.message_projection(&handle.build, allowance)?;
    let preparation = writer.inspect_message_preparation(
        &handle.build,
        handle.identity.borrowed(handle.request.message as u64),
        Source {
            predecessor: handle.build.predecessor(),
            revision: handle.request.revision,
            evidence_revision: handle.head.revision(),
            logical_key: handle.request.logical_key,
            route: RouteKind::Native,
        },
        allowance,
    )?;
    if preparation.build != handle.build {
        return Err(Error::Invariant("Message preparation identity changed"));
    }
    Ok(State {
        handle: handle.clone(),
        preparation,
        projection,
    })
}
struct Limited {
    bytes: Vec<u8>,
    maximum: usize,
    refused: bool,
}
impl Write for Limited {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if b.len() > self.maximum.saturating_sub(self.bytes.len()) {
            self.refused = true;
            return Err(io::Error::other("Message metadata admission"));
        }
        self.bytes.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn bounded_json(value: &impl Serialize, maximum: usize) -> Result<Vec<u8>> {
    let mut writer = Limited {
        bytes: vec![],
        maximum,
        refused: false,
    };
    let result = serde_json::to_writer(&mut writer, value);
    if writer.refused {
        return Err(StoreError::ParserAdmission {
            resource: "Message materializer metadata",
            required: (maximum as u64).saturating_add(1),
        }
        .into());
    }
    result.map_err(|e| Error::Source(e.to_string()))?;
    Ok(writer.bytes)
}
fn action(state: &State, b: Budgets, change: Change) -> Action {
    Action {
        inner: ActionKind::Turn {
            guard: Guard {
                handle: state.handle.clone(),
                sequence: state.projection.sequence,
                cursor: state.projection.cursor.clone(),
                metadata: b.metadata,
                active: state.preparation.active_value.clone(),
                active_origin: state.preparation.active_origin_value.clone(),
            },
            change,
        },
    }
}
fn advance(state: &State, cursor: &Cursor, b: Budgets) -> Result<Action> {
    Ok(action(
        state,
        b,
        Change::Advance(bounded_json(cursor, b.metadata)?),
    ))
}
fn block(attempt: &Attempt<'_>, message: usize, index: usize) -> Result<(Block, Group)> {
    let value = match read(attempt, &Key::Block(message, index))? {
        Some(Cell::Block(b)) => b,
        _ => return Err(Error::Unsupported("missing Codex block")),
    };
    if !matches!(
        value.content,
        BlockContent::Text { .. }
            | BlockContent::Thinking(_)
            | BlockContent::ToolCall(_)
            | BlockContent::ToolResult(_)
    ) {
        return Err(Error::Unsupported(
            "Codex block kind has no canonical materializer yet",
        ));
    }
    let group = match read(attempt, &Key::PresenceBlock(message, index))? {
        Some(Cell::Presence(g)) => g,
        _ => return Err(Error::Unsupported("Codex block presence is unresolved")),
    };
    Ok((value, group))
}

fn message_witness(attempt: &Attempt<'_>, before: &SourceCursor) -> Result<Witness> {
    let mut source = before.clone();
    let step = source.advance(attempt);
    if let Some(e) = attempt.take_failure() {
        return Err(e.into());
    }
    match step.map_err(Error::Source)? {
        Step::Unit(Unit::MessageWitness(w)) => Ok(w),
        _ => Err(Error::Invariant(
            "field locator no longer selects its source witness",
        )),
    }
}
fn cited(range: RecordRange) -> SourceRange {
    SourceRange {
        start: range.start,
        end: range.end,
    }
}
fn one_origin(origins: &Origins, cursor: &mut origins::Cursor) -> Result<Option<RecordRange>> {
    if matches!(origins, Origins::Unknown) {
        return Err(Error::Unsupported("source record contributors are unknown"));
    }
    origins.advance(cursor).map_err(Error::Source)
}
fn citation_progress(
    state: &State,
    cursor: &mut Cursor,
    next: Stage,
    range: Option<RecordRange>,
    b: Budgets,
) -> Result<Action> {
    cursor.stage = next;
    let next = bounded_json(cursor, b.metadata)?;
    Ok(action(
        state,
        b,
        match range {
            Some(range) => Change::SourceRange {
                range: cited(range),
                next,
            },
            None => Change::Advance(next),
        },
    ))
}
fn citation_turn(
    attempt: &Attempt<'_>,
    state: &State,
    cursor: &mut Cursor,
    scan: Citations,
    b: Budgets,
) -> Result<Action> {
    let message = state.handle.request.message;
    match scan {
        Citations::Evidence {
            mut slots,
            current,
            mut origins,
        } => {
            let key = match current {
                Some(key) => key,
                None => match state.handle.head.codex()?.next_slot(message, &mut slots) {
                    Some(key) => key,
                    None => {
                        return citation_progress(
                            state,
                            cursor,
                            Stage::Citations(Citations::Projection {
                                source: SourceCursor::new(message, Mode::Message),
                            }),
                            None,
                            b,
                        );
                    }
                },
            };
            match read(attempt, &key)? {
                Some(Cell::Evidence(unit)) => {
                    unit.validate_at(&key).map_err(Error::Source)?;
                    let range = one_origin(&unit.origins, &mut origins)?;
                    let next = if range.is_some() {
                        Citations::Evidence {
                            slots,
                            current: Some(key),
                            origins,
                        }
                    } else {
                        Citations::Evidence {
                            slots,
                            current: None,
                            origins: origins::Cursor::default(),
                        }
                    };
                    citation_progress(state, cursor, Stage::Citations(next), range, b)
                }
                None if !state.handle.head.codex()?.required_slot(&key) => citation_progress(
                    state,
                    cursor,
                    Stage::Citations(Citations::Evidence {
                        slots,
                        current: None,
                        origins: origins::Cursor::default(),
                    }),
                    None,
                    b,
                ),
                _ => Err(Error::Unsupported(
                    "source evidence locator is missing or invalid",
                )),
            }
        }
        Citations::Projection { mut source } => {
            let before = source.clone();
            let step = source.advance(attempt);
            if let Some(e) = attempt.take_failure() {
                return Err(e.into());
            }
            let next = match step.map_err(Error::Source)? {
                Step::Done => Stage::Layout,
                Step::Absent => {
                    return Err(Error::Unsupported(
                        "source disappeared during citation scan",
                    ));
                }
                Step::Unit(Unit::MessageWitness(_)) => Stage::Citations(Citations::Message {
                    before,
                    after: source,
                    origins: origins::Cursor::default(),
                }),
                Step::Unit(Unit::BlockPresence {
                    index,
                    group: Some(_),
                }) => Stage::Citations(Citations::Block {
                    source,
                    index,
                    field: 0,
                    origins: origins::Cursor::default(),
                }),
                Step::Unit(Unit::BlockPresence { group: None, .. }) => {
                    return Err(Error::Unsupported("block citation witnesses are unknown"));
                }
                _ => Stage::Citations(Citations::Projection { source }),
            };
            citation_progress(state, cursor, next, None, b)
        }
        Citations::Message {
            before,
            after,
            mut origins,
        } => {
            let w = message_witness(attempt, &before)?;
            let range = one_origin(&w.origins, &mut origins)?;
            let next = if range.is_some() {
                Citations::Message {
                    before,
                    after,
                    origins,
                }
            } else {
                Citations::Projection { source: after }
            };
            citation_progress(state, cursor, Stage::Citations(next), range, b)
        }
        Citations::Block {
            source,
            index,
            field,
            mut origins,
        } => {
            let group = match read(attempt, &Key::PresenceBlock(message, index))? {
                Some(Cell::Presence(group)) => group,
                _ => return Err(Error::Unsupported("block citation witnesses disappeared")),
            };
            let Some(w) = group.fields.get(field) else {
                return citation_progress(
                    state,
                    cursor,
                    Stage::Citations(Citations::Projection { source }),
                    None,
                    b,
                );
            };
            let range = one_origin(&w.origins, &mut origins)?;
            let next = if range.is_some() {
                Citations::Block {
                    source,
                    index,
                    field,
                    origins,
                }
            } else {
                Citations::Block {
                    source,
                    index,
                    field: field
                        .checked_add(1)
                        .ok_or(Error::Invariant("citation field cursor overflow"))?,
                    origins: origins::Cursor::default(),
                }
            };
            citation_progress(state, cursor, Stage::Citations(next), range, b)
        }
    }
}
struct DigestSink<'a> {
    hasher: blake3::Hasher,
    length: u64,
    flag: &'a AtomicBool,
    maximum: Option<u64>,
    exceeded: bool,
}
impl<'a> DigestSink<'a> {
    fn new(flag: &'a AtomicBool) -> Self {
        Self {
            hasher: blake3::Hasher::new(),
            length: 0,
            flag,
            maximum: None,
            exceeded: false,
        }
    }
    fn digest(&self) -> BlobHash {
        BlobHash::from_bytes(*self.hasher.finalize().as_bytes())
    }
}
impl Write for DigestSink<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.flag.load(Ordering::Acquire) {
            return Err(io::Error::other("field verification cancelled"));
        }
        let count = u64::try_from(bytes.len())
            .map_err(|_| io::Error::other("field byte count overflow"))?;
        self.length = self
            .length
            .checked_add(count)
            .ok_or_else(|| io::Error::other("field byte count overflow"))?;
        if self.maximum.is_some_and(|maximum| self.length > maximum) {
            self.exceeded = true;
            return Err(io::Error::other("selected value exceeds normalized length"));
        }
        self.hasher.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[derive(Clone, Copy)]
enum FieldSelector<'a> {
    Codex(&'a FactSelection),
    Claude(&'a polylogue_sources::claude_code::selection::LeafSelection),
    ClaudeBody(&'a polylogue_store::claude_bodies::Companion),
    Author(&'a polylogue_sources::claude_code::selection::AuthorSelection),
}
impl FieldSelector<'_> {
    fn write_bound(&self, binding: Binding<'_>, sink: &mut impl Write) -> Result<()> {
        let cb = polylogue_sources::claude_code::selection::Binding {
            input: binding.input,
            chain: binding.chain,
        };
        match self {
            Self::Codex(s) => s.write_bound(binding, sink).map_err(Error::Descriptor),
            Self::Claude(s) => s.write_bound(cb, sink).map_err(Error::ClaudeDescriptor),
            Self::Author(s) => s.write_bound(cb, sink).map_err(Error::ClaudeDescriptor),
            Self::ClaudeBody(body) => serde_json::to_writer(sink, &serde_json::json!({"format":"claude-code-tool-result-body","codec_version":1,"binding":{"input":body.input,"chain":body.chain},"revision":body.revision,"path":body.path,"blob":body.blob,"start":0,"end":body.length,"transform":"utf8-tool-output-v1"})).map_err(|e| Error::Source(e.to_string())),
        }
    }
    fn write_value(
        &self,
        reader: &ReadHandle,
        resolver: impl FnMut(RecordRange) -> std::result::Result<Vec<u8>, ResolveError>,
        allowance: usize,
        sink: &mut impl Write,
    ) -> Result<()> {
        match self {
            Self::ClaudeBody(body) => {
                let bytes = reader.claude_body(body)?;
                let text = String::from_utf8(bytes)
                    .map_err(|_| Error::Unsupported("Claude body is not UTF-8"))?;
                serde_json::to_writer(sink, &ToolOutput::Text(text))
                    .map_err(|e| Error::Source(e.to_string()))
            }
            Self::Codex(s) => s
                .write_value(resolver, allowance, sink)
                .map_err(Error::Selection),
            Self::Claude(s) => s
                .write_value(resolver, allowance, sink)
                .map_err(Error::ClaudeSelection),
            Self::Author(s) => s
                .write_value(resolver, allowance, sink)
                .map_err(Error::ClaudeSelection),
        }
    }
}
struct FieldDescriptor<'a> {
    slot: &'a Slot,
    time: Option<i64>,
    supplied: bool,
    selection: FieldSelector<'a>,
}
impl<'a> FieldDescriptor<'a> {
    fn codex(w: &'a Witness) -> Result<Self> {
        Ok(Self {
            slot: &w.slot,
            time: w.source_time.map(|t| t.0),
            supplied: true,
            selection: FieldSelector::Codex(
                w.selection
                    .as_ref()
                    .ok_or(Error::Unsupported("source field selector is unavailable"))?,
            ),
        })
    }
}
fn verify_selected_field(
    snapshot: &Snapshot,
    reader: &ReadHandle,
    state: &State,
    selection: &FieldSelector<'_>,
    expected_length: u64,
    expected_hash: BlobHash,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<()> {
    if let FieldSelector::ClaudeBody(body) = selection {
        snapshot
            .charge_selector_record(usize::try_from(body.length).map_err(|_| StoreError::Range)?)?;
    }
    let mut value = DigestSink::new(flag);
    if snapshot.cumulative_reads_enforced() {
        value.maximum = Some(expected_length);
    }
    let evaluated = selection.write_value(
        reader,
        |range| {
            if flag.load(Ordering::Acquire) {
                return Err(ResolveError::Cancelled);
            }
            if !snapshot
                .has_message_source(&state.handle.build, cited(range))
                .map_err(ResolveError::Store)?
            {
                return Err(ResolveError::Unbound);
            }
            let length = range
                .end
                .checked_sub(range.start)
                .ok_or(ResolveError::Unbound)?;
            snapshot
                .charge_selector_record(usize::try_from(length).map_err(|_| ResolveError::Unbound)?)
                .map_err(ResolveError::Store)?;
            reader
                .read_source_range_checked(
                    &snapshot
                        .message_source_input_id(&state.handle.build, b.metadata)
                        .map_err(ResolveError::Store)?,
                    snapshot.head.token.chain,
                    state.handle.request.source_binding,
                    range.start,
                    length,
                    || {
                        if flag.load(Ordering::Acquire) {
                            Err(polylogue_store::Error::Cancelled)
                        } else {
                            Ok(())
                        }
                    },
                )
                .map_err(ResolveError::Store)
        },
        b.read.encoded_bytes,
        &mut value,
    );
    cancelled(flag)?;
    if value.exceeded {
        return Err(Error::Invariant(
            "source selector output differs from normalized field value",
        ));
    }
    evaluated?;
    if value.length != expected_length || value.digest() != expected_hash {
        return Err(Error::Invariant(
            "source selector output differs from normalized field value",
        ));
    }
    Ok(())
}

fn field_turn(
    snapshot: &Snapshot,
    reader: &ReadHandle,
    state: &State,
    cursor: &mut Cursor,
    witness: FieldDescriptor<'_>,
    source: ValueSource<'_>,
    block: bool,
    next: Stage,
    input_id: &str,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<Action> {
    let selection = witness.selection;
    if state.preparation.active_value.is_none()
        && state.preparation.active_origin_value.is_none()
        && cursor.verified.is_none()
    {
        let limit = b
            .value_chunk
            .min(values::MAX_CHUNK)
            .min(b.read.encoded_bytes)
            .min(b.metadata);
        if limit > 0 {
            let (length, bytes) = encode_window(&source, 0, limit, flag)?;
            if length == bytes.len() as u64 {
                let binding = Binding {
                    input: input_id,
                    chain: state.handle.build.predecessor().chain,
                };
                let descriptor = ValueSource::Descriptor { selection, binding };
                let (origin_length, origin) = encode_window(&descriptor, 0, limit, flag)?;
                if origin_length == origin.len() as u64
                    && (!snapshot.cumulative_reads_enforced()
                        || length.saturating_mul(2).saturating_add(origin_length) <= b.hash_bytes)
                {
                    cancelled(flag)?;
                    let field = SmallField::prepare(bytes, origin, limit)?;
                    let next_cursor = Cursor {
                        source: cursor.source.clone(),
                        stage: next.clone(),
                        verified: None,
                    };
                    let encoded = bounded_json(&next_cursor, b.metadata)?;
                    let step = ProjectionStep {
                        expected_sequence: state.projection.sequence,
                        next_cursor: &encoded,
                    };
                    match field.admit(witness.slot, block, &step, b.metadata) {
                        Ok(()) => {
                            verify_selected_field(
                                snapshot,
                                reader,
                                state,
                                &selection,
                                field.value_length(),
                                BlobHash::from_bytes(field.value_hash()),
                                b,
                                flag,
                            )?;
                            cancelled(flag)?;
                            return Ok(action(
                                state,
                                b,
                                Change::SmallField {
                                    field,
                                    slot: witness.slot.clone(),
                                    time: witness.time,
                                    block,
                                    supplied: witness.supplied,
                                    next: encoded,
                                },
                            ));
                        }
                        Err(StoreError::ParserAdmission { .. }) => {} // ordinary streaming remains available
                        Err(error) => return Err(error.into()),
                    }
                }
            }
        }
    }
    if let ValueTurn::Action(change) =
        value_turn(reader, state, source, values::Codec::Json, b, flag)?
    {
        return Ok(action(state, b, change));
    }
    let active = state
        .preparation
        .active_value
        .as_ref()
        .ok_or(Error::Invariant("field has no normalized producer"))?;
    let value_hash = BlobHash::from_bytes(
        active
            .hash
            .ok_or(Error::Invariant("ready normalized value has no digest"))?,
    );
    let binding = Binding {
        input: input_id,
        chain: state.handle.build.predecessor().chain,
    };
    let mut descriptor = DigestSink::new(flag);
    let described = selection.write_bound(binding, &mut descriptor);
    cancelled(flag)?;
    described?;
    if let Some(verified) = &cursor.verified {
        if verified.value_hash != value_hash
            || verified.value_length != active.length
            || verified.descriptor_hash != descriptor.digest()
            || verified.descriptor_length != descriptor.length
        {
            return Err(Error::Invariant(
                "durable field verification no longer matches its normalized value and selector",
            ));
        }
    } else {
        if state.preparation.active_origin_value.is_some() {
            return Err(Error::Unsupported(
                "unverified origin producer cannot attach to field",
            ));
        }
        verify_selected_field(
            snapshot,
            reader,
            state,
            &selection,
            active.length,
            value_hash,
            b,
            flag,
        )?;
        cursor.verified = Some(Verified {
            value_hash,
            value_length: active.length,
            descriptor_hash: descriptor.digest(),
            descriptor_length: descriptor.length,
        });
        return advance(state, cursor, b);
    }
    match owned_value_turn(
        reader,
        state,
        ValueSource::Descriptor { selection, binding },
        values::Codec::Json,
        true,
        b,
        flag,
    )? {
        ValueTurn::Action(change) => Ok(action(state, b, change)),
        ValueTurn::Ready => {
            let origin = state
                .preparation
                .active_origin_value
                .as_ref()
                .ok_or(Error::Invariant("verified origin producer disappeared"))?;
            if origin.length != descriptor.length
                || origin.hash != Some(*descriptor.digest().bytes())
            {
                return Err(Error::Invariant(
                    "ready descriptor differs from verified source selector",
                ));
            }
            cursor.stage = next;
            cursor.verified = None;
            Ok(action(
                state,
                b,
                Change::Field {
                    slot: witness.slot.clone(),
                    time: witness.time,
                    block,
                    supplied: witness.supplied,
                    next: bounded_json(cursor, b.metadata)?,
                },
            ))
        }
    }
}

/// Hash/validation currently completes the selected Message/value on the worker
/// with cooperative piece checks. This is not a bounded resumed hash turn.
mod runs;
pub fn compute(
    snapshot: &Snapshot,
    reader: &ReadHandle,
    state: &State,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<Action> {
    runs::compute(snapshot, reader, state, b, flag)
}
fn compute_one(
    snapshot: &Snapshot,
    reader: &ReadHandle,
    state: &State,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<Action> {
    cancelled(flag)?;
    if snapshot.head.token != *state.handle.build.predecessor() {
        return Err(Error::Invariant(
            "materializer snapshot predecessor changed",
        ));
    }
    if b.value_chunk == 0 || b.value_chunk > 65536 || b.hash_bytes == 0 {
        return Err(Error::Invariant("invalid materializer byte allowance"));
    }
    if matches!(state.handle.head, EvidenceBasis::Claude(_)) {
        return claude::compute(snapshot, reader, state, b, flag);
    }
    let attempt = Attempt::new(snapshot, b.read);
    let current_head = evidence(&attempt, state.handle.request.message)?;
    if &current_head != state.handle.head.codex()? {
        return Err(Error::Invariant("materializer evidence head changed"));
    }
    let input_id = snapshot.message_source_input_id(&state.handle.build, b.metadata)?;
    let expected_input =
        InputId::imported(&state.handle.request.source_binding, "codex-rollout", &[]);
    if input_id != expected_input.as_str() {
        return Err(Error::Invariant(
            "raw source hash does not identify this immutable Codex input",
        ));
    }
    if let Some(prepared) = &state.preparation.projection {
        return Ok(action(state, b, Change::Publish(prepared.clone())));
    }
    let mut cursor: Cursor = match &state.projection.cursor {
        Some(bytes) => serde_json::from_slice(bytes).map_err(|e| Error::Source(e.to_string()))?,
        None if state.projection.sequence == 0 => Cursor {
            source: Some(SourceCursor::new(
                state.handle.request.message,
                Mode::Message,
            )),
            stage: Stage::Citations(Citations::Evidence {
                slots: EvidenceCursor::default(),
                current: None,
                origins: origins::Cursor::default(),
            }),
            verified: None,
        },
        None => return Err(Error::Invariant("materializer lost durable cursor")),
    };
    if state.preparation.active_origin_value.is_some()
        && (cursor.verified.is_none()
            || !matches!(
                cursor.stage,
                Stage::MessageField { .. } | Stage::Fields { .. }
            ))
    {
        return Err(Error::Unsupported(
            "source origin producer has no verified owning field stage",
        ));
    }
    let message = state.handle.request.message;
    match cursor.stage.clone() {
        Stage::Citations(citations) => citation_turn(&attempt, state, &mut cursor, citations, b),
        Stage::Layout => {
            cursor.stage = Stage::Next;
            Ok(action(
                state,
                b,
                Change::Layout {
                    layout: MessageLayout::SeparateReasoning,
                    next: bounded_json(&cursor, b.metadata)?,
                },
            ))
        }
        Stage::Next | Stage::AwaitPresence { .. } => {
            let source = cursor
                .source
                .as_mut()
                .ok_or(Error::Invariant("Codex projection lost its cursor"))?;
            let before = source.clone();
            let step = source.advance(&attempt);
            if let Some(e) = attempt.take_failure() {
                return Err(e.into());
            }
            match step.map_err(Error::Source)? {
                Step::Continue => advance(state, &cursor, b),
                Step::Absent => Err(Error::Unsupported(
                    "Codex Message disappeared during projection",
                )),
                Step::Done => {
                    let mut driver = version_proof::Driver::start(
                        snapshot,
                        message,
                        state.handle.head.revision(),
                        b.read,
                    )?;
                    let proof = loop {
                        cancelled(flag)?;
                        match driver.advance(b.read, b.hash_bytes)? {
                            version_proof::Advance::Complete(p) => break p,
                            version_proof::Advance::Progress {
                                minimum_bytes: Some(required),
                                ..
                            } if required > b.hash_bytes => {
                                return Err(StoreError::ParserAdmission {
                                    resource: "Message version proof framing bytes",
                                    required,
                                }
                                .into());
                            }
                            _ => {}
                        }
                    };
                    if proof.token() != state.handle.build.predecessor()
                        || proof.head() != state.handle.head.codex()?
                        || proof.message() != message
                    {
                        return Err(Error::Invariant("Message version proof basis mismatch"));
                    }
                    let h = header(&attempt, message)?;
                    let finish = match h.in_progress {
                        Some(true) => FinishState::InProgress,
                        Some(false) => FinishState::Finished,
                        None => FinishState::Unknown,
                    };
                    cursor.stage = Stage::Proof;
                    Ok(action(
                        state,
                        b,
                        Change::Finish {
                            proof: EvidenceProof::Codex(proof),
                            finish,
                            next: bounded_json(&cursor, b.metadata)?,
                        },
                    ))
                }
                Step::Unit(Unit::Header(h)) => {
                    if h.id != state.handle.identity.message {
                        return Err(Error::Invariant("Message projection identity changed"));
                    }
                    advance(state, &cursor, b)
                }
                Step::Unit(Unit::MessageWitness(w)) => {
                    let _ = w;
                    cursor.verified = None;
                    cursor.stage = Stage::MessageField { witness: before };
                    advance(state, &cursor, b)
                }
                Step::Unit(Unit::Block { index, value }) => {
                    if !matches!(
                        value.content,
                        BlockContent::Text { .. }
                            | BlockContent::Thinking(_)
                            | BlockContent::ToolCall(_)
                            | BlockContent::ToolResult(_)
                    ) {
                        return Err(Error::Unsupported(
                            "Codex block kind has no canonical materializer yet",
                        ));
                    }
                    cursor.stage = Stage::AwaitPresence { index };
                    advance(state, &cursor, b)
                }
                Step::Unit(Unit::BlockPresence { index, group }) => {
                    if !matches!(cursor.stage,Stage::AwaitPresence{index:i} if i==index)
                        || group.is_none()
                    {
                        return Err(Error::Unsupported("Codex block suppliedness is unresolved"));
                    }
                    cursor.stage = Stage::Text { index };
                    advance(state, &cursor, b)
                }
                Step::Unit(
                    Unit::Continuation { .. }
                    | Unit::WrapperStart { .. }
                    | Unit::WrapperChild { .. }
                    | Unit::ResultStart { .. }
                    | Unit::ResultChunk { .. }
                    | Unit::ResultEnd { .. },
                ) => advance(state, &cursor, b),
            }
        }
        Stage::MessageField { witness } => {
            let w = message_witness(&attempt, &witness)?;
            let h = header(&attempt, message)?;
            field_turn(
                snapshot,
                reader,
                state,
                &mut cursor,
                FieldDescriptor::codex(&w)?,
                ValueSource::Leaf {
                    subject: Subject::Message(&h),
                    slot: &w.slot,
                },
                false,
                Stage::Next,
                &input_id,
                b,
                flag,
            )
        }
        Stage::Text { index } => {
            let (value, group) = block(&attempt, message, index)?;
            let (text, kind) = match &value.content {
                BlockContent::Text { text } => (Some(text.as_str()), OwnedBlockKind::Text),
                BlockContent::Thinking(t) => (
                    t.text.as_deref(),
                    OwnedBlockKind::Thinking {
                        encrypted: t.encrypted,
                    },
                ),
                BlockContent::ToolCall(call) => {
                    let operation = match read(&attempt, &Key::OperationBlock(message, index))? {
                        Some(Cell::Text(id)) => id,
                        _ => {
                            // Generic/legacy response calls have a declared Session-scoped
                            // invocation, not a modern CommandExecution constructor.
                            let call_id = call.call_id.as_ref().ok_or(Error::Unsupported(
                                "Codex call has no supplied invocation ID",
                            ))?;
                            polylogue_model::DeclaredOperationId::from_native(
                                value.id.message().session(),
                                call_id,
                            )
                            .map_err(|_| {
                                Error::Unsupported("Codex declared call scope unavailable")
                            })?
                            .as_str()
                            .to_owned()
                        }
                    };
                    (None, OwnedBlockKind::ToolCall { operation })
                }
                BlockContent::ToolResult(result) => (
                    None,
                    OwnedBlockKind::ToolResult {
                        outcome: result.outcome.clone(),
                        exit_code: result.exit_code,
                        duration_ms: result.duration_ms,
                    },
                ),
                _ => return Err(Error::Unsupported("unsupported staged block")),
            };
            if let Some(text) = text {
                if let ValueTurn::Action(change) = value_turn(
                    reader,
                    state,
                    ValueSource::Text(text),
                    values::Codec::Utf8,
                    b,
                    flag,
                )? {
                    return Ok(action(state, b, change));
                }
            } else if state.preparation.active_value.is_some() {
                return Err(Error::Invariant("textless Thinking has an active value"));
            }
            cursor.stage = Stage::Fields { index, field: 0 };
            Ok(action(
                state,
                b,
                Change::BeginBlock {
                    group: 0,
                    local: u64::try_from(index)
                        .map_err(|_| Error::Invariant("Codex block index overflow"))?,
                    kind,
                    presence: BlockPresence {
                        source_block: group.nonempty,
                        source_text: group.nonempty_text,
                    },
                    next: bounded_json(&cursor, b.metadata)?,
                },
            ))
        }
        Stage::Fields { index, field } => {
            let (value, group) = block(&attempt, message, index)?;
            let Some(w) = group.fields.get(field) else {
                cursor.stage = Stage::Install { index };
                return advance(state, &cursor, b);
            };
            let next = Stage::Fields {
                index,
                field: field
                    .checked_add(1)
                    .ok_or(Error::Invariant("block field cursor overflow"))?,
            };
            // Extras are opaque mapper-owned values, not semantic Block leaves.
            // The field turn still verifies this value against its exact raw selector.
            let extras = match &w.slot {
                Slot::Extra { key, .. } => Some(RawJson::Object(
                    [(
                        key.as_str().into(),
                        RawJson::parse(&w.canonical)
                            .map_err(|e| Error::Source(e.to_string()))?
                            .into_opaque(),
                    )]
                    .into_iter()
                    .collect(),
                )),
                _ => None,
            };
            let subject = match &extras {
                Some(values) => Subject::Extras {
                    block: Some(value.id.slot()),
                    values,
                },
                None => Subject::Block {
                    slot: value.id.slot(),
                    content: &value.content,
                },
            };
            field_turn(
                snapshot,
                reader,
                state,
                &mut cursor,
                FieldDescriptor::codex(w)?,
                ValueSource::Leaf {
                    subject,
                    slot: &w.slot,
                },
                true,
                next,
                &input_id,
                b,
                flag,
            )
        }
        Stage::Install { .. } => {
            cursor.stage = Stage::Next;
            Ok(action(
                state,
                b,
                Change::Install(bounded_json(&cursor, b.metadata)?),
            ))
        }
        Stage::Claude(_) => Err(Error::Invariant("Claude cursor on Codex evidence basis")),
        Stage::Proof => Err(Error::Invariant(
            "proof cursor lacks completed source projection",
        )),
    }
}
enum ValueSource<'a> {
    Descriptor {
        selection: FieldSelector<'a>,
        binding: Binding<'a>,
    },
    Text(&'a str),
    Opaque(RawJson),
    Leaf {
        subject: Subject<'a>,
        slot: &'a Slot,
    },
}
enum ValueTurn {
    Action(Change),
    Ready,
}

fn same_reference(a: Option<&SourceProjection>, b: Option<&SourceProjection>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.body() == b.body()
                && a.predecessor() == b.predecessor()
                && a.message() == b.message()
                && a.evidence_revision() == b.evidence_revision()
                && a.version_hash() == b.version_hash()
                && a.evidence_guard() == b.evidence_guard()
        }
        _ => false,
    }
}
/// Apply ONE owned turn. No provider callback, serde, value hashing, source or
/// blob IO occurs here. The sole writer rechecks source/cursor/value identity.
pub fn apply(writer: &mut Writer, action: Action) -> Result<Applied> {
    match action.inner {
        ActionKind::Prepare {
            identity,
            request,
            token,
            head,
            guard,
            previous,
            allowance,
        } => {
            let current =
                writer.current_source_projection(&token, &request.logical_key, allowance)?;
            if !same_reference(current.as_ref(), previous.as_ref()) {
                return Err(StoreError::MessagePredecessorChanged.into());
            }
            let source = Source {
                predecessor: &token,
                revision: request.revision,
                evidence_revision: head.revision(),
                logical_key: request.logical_key,
                route: RouteKind::Native,
            };
            if writer.retire_superseded_message_turn(&identity.message, &source, allowance)? {
                return Ok(Applied::Progress);
            }
            let preparation = writer.prepare_message(
                identity.borrowed(request.message as u64),
                source,
                allowance,
            )?;
            Ok(Applied::Prepared(Handle {
                identity,
                request,
                build: preparation.build,
                head,
                guard,
                previous,
            }))
        }
        ActionKind::Existing {
            token,
            request,
            source,
            allowance,
        } => {
            let current =
                writer.current_source_projection(&token, &request.logical_key, allowance)?;
            if !same_reference(current.as_ref(), Some(&source)) {
                return Err(StoreError::MessagePredecessorChanged.into());
            }
            Ok(Applied::Existing(source))
        }
        ActionKind::Turn { guard, change } => {
            let now = inspect(writer, &guard.handle, guard.metadata)?;
            if now.projection.sequence != guard.sequence
                || now.projection.cursor != guard.cursor
                || now.preparation.active_value != guard.active
                || now.preparation.active_origin_value != guard.active_origin
            {
                return Err(StoreError::MessagePredecessorChanged.into());
            }
            let build = &guard.handle.build;
            match change {
                Change::SmallFieldRun(entries) => {
                    writer.write_small_field_run(build, &entries, guard.metadata)?
                }
                Change::SourceRangeRun(entries) => {
                    writer.add_message_source_run(build, &entries, guard.metadata)?
                }
                Change::SourceRange { range, next } => writer.add_message_source(
                    build,
                    range,
                    &ProjectionStep {
                        expected_sequence: guard.sequence,
                        next_cursor: &next,
                    },
                    guard.metadata,
                )?,
                Change::BeginOrigin { length } => {
                    writer.begin_message_origin_value(build, length)?;
                }
                Change::SealOrigin(proof) => {
                    writer.seal_message_origin_value(build, &proof)?;
                }
                Change::Advance(next) => writer.advance_message_projection(
                    build,
                    &ProjectionStep {
                        expected_sequence: guard.sequence,
                        next_cursor: &next,
                    },
                    guard.metadata,
                )?,
                Change::Layout { layout, next } => writer.set_message_layout(
                    build,
                    layout,
                    &ProjectionStep {
                        expected_sequence: guard.sequence,
                        next_cursor: &next,
                    },
                    guard.metadata,
                )?,
                Change::BeginValue { codec, length } => {
                    writer.begin_message_value(build, codec, length)?;
                }
                Change::Append { id, offset, bytes } => {
                    if bytes.len() > 65536 {
                        return Err(Error::Invariant("materializer append exceeds one chunk"));
                    }
                    writer.append_value_chunk(&id, offset, &bytes, bytes.len())?;
                }
                Change::Seal(proof) => {
                    writer.seal_message_value(build, &proof)?;
                }
                Change::Field {
                    slot,
                    time,
                    block,
                    supplied,
                    next,
                } => {
                    if block {
                        writer.write_block_field(
                            build,
                            &slot,
                            time,
                            supplied,
                            &ProjectionStep {
                                expected_sequence: guard.sequence,
                                next_cursor: &next,
                            },
                            guard.metadata,
                        )?;
                    } else {
                        writer.write_message_field(
                            build,
                            &slot,
                            time,
                            supplied,
                            &ProjectionStep {
                                expected_sequence: guard.sequence,
                                next_cursor: &next,
                            },
                            guard.metadata,
                        )?;
                    }
                }
                Change::SmallField {
                    field,
                    slot,
                    time,
                    block,
                    supplied,
                    next,
                } => {
                    writer.write_small_field(
                        build,
                        &slot,
                        time,
                        supplied,
                        block,
                        &field,
                        &ProjectionStep {
                            expected_sequence: guard.sequence,
                            next_cursor: &next,
                        },
                        guard.metadata,
                    )?;
                }
                Change::BeginBlock {
                    group,
                    local,
                    kind,
                    presence,
                    next,
                } => {
                    writer.begin_block(
                        build,
                        group,
                        local,
                        kind.borrowed(),
                        presence,
                        &ProjectionStep {
                            expected_sequence: guard.sequence,
                            next_cursor: &next,
                        },
                        guard.metadata,
                    )?;
                }
                Change::Command(next) => {
                    writer.write_tool_command(
                        build,
                        &ProjectionStep {
                            expected_sequence: guard.sequence,
                            next_cursor: &next,
                        },
                        guard.metadata,
                    )?;
                }
                Change::Install(next) => {
                    writer.install_block(
                        build,
                        &ProjectionStep {
                            expected_sequence: guard.sequence,
                            next_cursor: &next,
                        },
                        guard.metadata,
                    )?;
                }
                Change::Finish {
                    proof,
                    finish,
                    next,
                } => {
                    let digest = proof.validate(&guard.handle)?;
                    writer.finish_message_projection(
                        build,
                        &ProjectionStep {
                            expected_sequence: guard.sequence,
                            next_cursor: &next,
                        },
                        digest,
                        finish,
                        guard.metadata,
                    )?;
                }
                Change::Publish(prepared) => {
                    let source = writer.publish_source_projection(
                        &prepared,
                        guard.handle.previous.as_ref(),
                        &guard.handle.guard,
                        guard.metadata,
                    )?;
                    return Ok(Applied::Source(source));
                }
            }
            Ok(Applied::Progress)
        }
    }
}

impl ValueSource<'_> {
    fn write(&self, sink: &mut impl Write) -> Result<()> {
        match self {
            Self::Descriptor { selection, binding } => selection.write_bound(*binding, sink),
            Self::Text(text) => sink
                .write_all(text.as_bytes())
                .map_err(|e| Error::Store(StoreError::Io(e))),
            Self::Opaque(value) => {
                serde_json::to_writer(sink, value).map_err(|e| Error::Source(e.to_string()))
            }
            Self::Leaf { subject, slot } => {
                let subject = match subject {
                    Subject::Message(m) => Subject::Message(m),
                    Subject::Block { slot, content } => Subject::Block { slot, content },
                    Subject::Extras { block, values } => Subject::Extras {
                        block: *block,
                        values,
                    },
                };
                match source_leaf::write(subject, slot, sink)? {
                    source_leaf::PresenceStatus::Written => Ok(()),
                    source_leaf::PresenceStatus::Missing => Err(Error::Unsupported(
                        "mapper suppliedness has no normalized known leaf",
                    )),
                    source_leaf::PresenceStatus::Unsupported => Err(Error::Unsupported(
                        "mapper slot has no normalized leaf projection",
                    )),
                }
            }
        }
    }
}
/// Replays the one admitted leaf encoding to count/window it without retaining
/// the whole output. This is deliberately NOT an append-local compute bound:
/// a large normalized leaf is serialized again for each chunk until a future
/// producer/spool supplies a seekable encoding. Captured output stays <=64KiB.
struct Window {
    offset: u64,
    maximum: usize,
    seen: u64,
    bytes: Vec<u8>,
    cancelled: bool,
}
struct Sink<'a> {
    window: Window,
    flag: &'a AtomicBool,
}
impl Write for Sink<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.flag.load(Ordering::Acquire) {
            self.window.cancelled = true;
            return Err(io::Error::other("Message encoding cancelled"));
        }
        let length = u64::try_from(bytes.len())
            .map_err(|_| io::Error::other("Message encoded length overflow"))?;
        let end = self
            .window
            .seen
            .checked_add(length)
            .ok_or_else(|| io::Error::other("Message encoded length overflow"))?;
        if end > self.window.offset && self.window.bytes.len() < self.window.maximum {
            let skip = usize::try_from(self.window.offset.saturating_sub(self.window.seen))
                .map_err(|_| io::Error::other("Message encoding offset overflow"))?;
            let take = (bytes.len() - skip).min(self.window.maximum - self.window.bytes.len());
            self.window
                .bytes
                .extend_from_slice(&bytes[skip..skip + take]);
        }
        self.window.seen = end;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn encode_window(
    source: &ValueSource<'_>,
    offset: u64,
    maximum: usize,
    flag: &AtomicBool,
) -> Result<(u64, Vec<u8>)> {
    let mut sink = Sink {
        window: Window {
            offset,
            maximum,
            seen: 0,
            bytes: vec![],
            cancelled: false,
        },
        flag,
    };
    let result = source.write(&mut sink);
    if sink.window.cancelled {
        return Err(Error::Cancelled);
    }
    result?;
    cancelled(flag)?;
    Ok((sink.window.seen, sink.window.bytes))
}
fn value_turn(
    reader: &ReadHandle,
    state: &State,
    source: ValueSource<'_>,
    codec: values::Codec,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<ValueTurn> {
    owned_value_turn(reader, state, source, codec, false, b, flag)
}
fn owned_value_turn(
    reader: &ReadHandle,
    state: &State,
    source: ValueSource<'_>,
    codec: values::Codec,
    origin: bool,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<ValueTurn> {
    let active = if origin {
        &state.preparation.active_origin_value
    } else {
        &state.preparation.active_value
    };
    let Some(active) = active else {
        let (length, _) = encode_window(&source, 0, 0, flag)?;
        return Ok(ValueTurn::Action(if origin {
            Change::BeginOrigin { length }
        } else {
            Change::BeginValue { codec, length }
        }));
    };
    if active.codec != codec {
        return Err(Error::Invariant(
            "active Message value codec differs from cursor",
        ));
    }
    match active.state {
        values::State::Ready => Ok(ValueTurn::Ready),
        values::State::Retiring => Err(Error::Invariant("active Message value is retiring")),
        values::State::Building => {
            if active.appended < active.length {
                let (length, bytes) = encode_window(&source, active.appended, b.value_chunk, flag)?;
                if length != active.length || bytes.is_empty() {
                    return Err(Error::Invariant(
                        "normalized value encoding differs from admitted length",
                    ));
                }
                return Ok(ValueTurn::Action(Change::Append {
                    id: active.id.clone(),
                    offset: active.appended,
                    bytes,
                }));
            }
            let mut proof = reader.validate_value(&active.id)?;
            while !proof.progress().complete {
                cancelled(flag)?;
                proof.advance(reader, b.value_chunk)?;
            }
            cancelled(flag)?;
            Ok(ValueTurn::Action(if origin {
                Change::SealOrigin(proof)
            } else {
                Change::Seal(proof)
            }))
        }
    }
}

pub(crate) use claude::current_projection as current_claude_projection;

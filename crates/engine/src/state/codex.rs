//! Codex typed cells over one pinned disposable SQLite parser-state snapshot.
//! Interpretation and encoding happen outside the output writer transaction.
//! Budgets measure cumulative encoded cell traffic, not allocator/RSS usage.
use std::cell::{Cell as Counter, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use polylogue_sources::codex::{
    self,
    keyed::{Cell, Delta, Key, ReadView, Resume},
};
use polylogue_store::Error as StoreError;
use polylogue_store::parser_state::{Budget, CellChange, Snapshot};
type Result<T> = std::result::Result<T, StoreError>;
use polylogue_model::{Event, Link, merge::Slot};
use polylogue_sources::codex::keyed::NativeConflict;
use serde::{Deserialize, Serialize};

pub const RESUME_NAMESPACE: &str = "codex.resume";
pub const RESUME_KEY: &[u8] = b"";
const PREFIX: &str = "codex.";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadUsage {
    /// Admitted lookup attempts, including absent keys and Resume.
    pub cells: usize,
    /// Namespace/lookup key bytes plus returned encoded key/value bytes.
    pub encoded_bytes: usize,
}
/// A new instance is required for each attempted provider step. The original
/// typed first failure stays here when the provider's String callback returns
/// an error; the owner must take it before classifying a source refusal.
pub struct Attempt<'a> {
    snapshot: &'a Snapshot,
    budget: Budget,
    usage: Counter<ReadUsage>,
    failure: RefCell<Option<StoreError>>,
    failed: Counter<bool>,
}
impl<'a> Attempt<'a> {
    pub fn new(snapshot: &'a Snapshot, budget: Budget) -> Self {
        Self {
            snapshot,
            budget,
            usage: Counter::new(ReadUsage::default()),
            failure: RefCell::new(None),
            failed: Counter::new(false),
        }
    }
    pub fn usage(&self) -> ReadUsage {
        self.usage.get()
    }
    pub fn take_failure(&self) -> Option<StoreError> {
        self.failure.borrow_mut().take()
    }
    fn checked<T>(&self, f: impl FnOnce() -> Result<T>) -> std::result::Result<T, String> {
        if let Some(error) = self.failure.borrow().as_ref() {
            return Err(error.to_string());
        }
        if self.failed.get() {
            return Err("Codex state attempt already failed".into());
        }
        match f() {
            Ok(v) => Ok(v),
            Err(error) => {
                let text = error.to_string();
                self.failed.set(true);
                *self.failure.borrow_mut() = Some(error);
                Err(text)
            }
        }
    }
    fn charge(&self, cells: usize, bytes: usize) -> Result<()> {
        let used = self.usage.get();
        let needed_cells = used.cells.saturating_add(cells);
        let needed_bytes = used.encoded_bytes.saturating_add(bytes);
        if needed_cells > self.budget.cells {
            return Err(admission("Codex read cells", needed_cells));
        }
        if needed_bytes > self.budget.encoded_bytes {
            return Err(admission("Codex read encoded bytes", needed_bytes));
        }
        self.usage.set(ReadUsage {
            cells: needed_cells,
            encoded_bytes: needed_bytes,
        });
        Ok(())
    }
    fn remaining(&self) -> usize {
        self.budget
            .encoded_bytes
            .saturating_sub(self.usage.get().encoded_bytes)
    }
    pub fn resume(&self) -> std::result::Result<Resume, String> {
        self.checked(|| {
            if self.snapshot.head.parser_version != codex::PARSER_VERSION {
                return Err(StoreError::Invariant(
                    "Codex state parser version differs from the pinned provider".into(),
                ));
            }
            self.charge(1, RESUME_NAMESPACE.len() + RESUME_KEY.len())?;
            let bytes = self
                .snapshot
                .cell(RESUME_NAMESPACE, RESUME_KEY, self.remaining())?;
            let resume = match bytes {
                Some(bytes) => {
                    self.charge(0, bytes.len())?;
                    serde_json::from_slice::<Resume>(&bytes).map_err(|e| invalid("Resume", e))?
                }
                None if self.snapshot.head.token.step == 0
                    && self.snapshot.head.parsed_upto == 0 =>
                {
                    Resume::default()
                }
                None => {
                    return Err(StoreError::Invariant(
                        "missing committed Codex Resume cell".into(),
                    ));
                }
            };
            if resume.parsed_upto != self.snapshot.head.parsed_upto {
                return Err(StoreError::Invariant(
                    "Codex Resume byte position differs from pinned complete-record checkpoint"
                        .into(),
                ));
            }
            Ok(resume)
        })
    }
    fn lookup_key(&self, key: &Key) -> Result<(String, Vec<u8>)> {
        let used = self.usage.get();
        if used.cells >= self.budget.cells {
            return Err(admission("Codex read cells", used.cells.saturating_add(1)));
        }
        let ns_len = PREFIX.len().saturating_add(key.family().len());
        if ns_len > self.remaining() {
            return Err(admission(
                "Codex read encoded bytes",
                used.encoded_bytes.saturating_add(ns_len),
            ));
        }
        let key_len = key
            .ordered_len_limited(self.remaining() - ns_len)
            .map_err(|needed| {
                admission(
                    "Codex read encoded bytes",
                    used.encoded_bytes
                        .saturating_add(ns_len)
                        .saturating_add(needed),
                )
            })?;
        self.charge(1, ns_len.saturating_add(key_len))?;
        let bytes = limited(key_len, 0, "Codex lookup key bytes", |w| {
            key.write_ordered(w)
        })?;
        let mut namespace = String::with_capacity(ns_len);
        namespace.push_str(PREFIX);
        namespace.push_str(key.family());
        Ok((namespace, bytes))
    }
    fn decode(&self, key: &Key, bytes: &[u8]) -> Result<Cell> {
        let cell = serde_json::from_slice::<Cell>(bytes).map_err(|e| invalid("cell", e))?;
        if !cell.valid_for(key) {
            return Err(StoreError::Invariant(format!(
                "Codex decoded cell shape differs from key family {}",
                key.family()
            )));
        }
        Ok(cell)
    }
    fn point(&self, key: &Key) -> Result<Option<Cell>> {
        let (ns, k) = self.lookup_key(key)?;
        match self.snapshot.cell(&ns, &k, self.remaining())? {
            None => Ok(None),
            Some(bytes) => {
                self.charge(0, bytes.len())?;
                self.decode(key, &bytes).map(Some)
            }
        }
    }
    fn neighbor(&self, key: &Key, after: bool) -> Result<Option<(Key, Cell)>> {
        let (ns, k) = self.lookup_key(key)?;
        let Some((encoded, value)) = self.snapshot.neighbor(&ns, &k, after, self.remaining())?
        else {
            return Ok(None);
        };
        self.charge(0, encoded.len().saturating_add(value.len()))?;
        let found = Key::from_ordered_bytes(&encoded).map_err(|e| invalid("ordered key", e))?;
        if !found.same_family(key) || (after && found <= *key) || (!after && found >= *key) {
            return Err(StoreError::Invariant(
                "Codex SQLite neighbor differs from requested family/order".into(),
            ));
        }
        let cell = self.decode(&found, &value)?;
        Ok(Some((found, cell)))
    }
}
impl ReadView for Attempt<'_> {
    fn read(&self, key: &Key) -> std::result::Result<Option<Cell>, String> {
        self.checked(|| self.point(key))
    }
    fn next(&self, key: &Key) -> std::result::Result<Option<(Key, Cell)>, String> {
        self.checked(|| self.neighbor(key, true))
    }
    fn previous(&self, key: &Key) -> std::result::Result<Option<(Key, Cell)>, String> {
        self.checked(|| self.neighbor(key, false))
    }
}
fn invalid(what: &str, error: impl std::fmt::Display) -> StoreError {
    StoreError::Invariant(format!("invalid Codex {what}: {error}"))
}
fn admission(resource: &'static str, required: usize) -> StoreError {
    StoreError::ParserAdmission {
        resource,
        required: u64::try_from(required).unwrap_or(u64::MAX),
    }
}

/// Bounded writer used for key and JSON encoding before output admission. No
/// write grows the Vec before checking its length. Unexpected allocator spare
/// capacity is an explicit failure, not a claimed physical memory bound.
struct Limited {
    bytes: Vec<u8>,
    limit: usize,
    base: usize,
    resource: &'static str,
    failure: Option<StoreError>,
}
impl Write for Limited {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.failure.is_some() {
            return Err(std::io::Error::other(
                "Codex encoding attempt already failed",
            ));
        }
        let required = self.bytes.len().saturating_add(bytes.len());
        let error = if required > self.limit {
            Some(admission(self.resource, self.base.saturating_add(required)))
        } else if let Err(e) = self.bytes.try_reserve_exact(required - self.bytes.len()) {
            Some(StoreError::Io(std::io::Error::other(e)))
        } else if self.bytes.capacity() > self.limit {
            Some(admission(
                "Codex encoded allocation capacity",
                self.base.saturating_add(self.bytes.capacity()),
            ))
        } else {
            None
        };
        if let Some(error) = error {
            self.failure = Some(error);
            return Err(std::io::Error::other("Codex encoding admission failed"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn limited<E: std::fmt::Display>(
    limit: usize,
    base: usize,
    resource: &'static str,
    f: impl FnOnce(&mut Limited) -> std::result::Result<(), E>,
) -> Result<Vec<u8>> {
    let mut writer = Limited {
        bytes: Vec::new(),
        limit,
        base,
        resource,
        failure: None,
    };
    let result = f(&mut writer);
    if let Some(error) = writer.failure {
        return Err(error);
    }
    result.map_err(|e| invalid("encoding", e))?;
    Ok(writer.bytes)
}
/// Adapter-only layout identity. Include this in the owning parser basis;
/// provider interpretation/version evidence is unchanged by this encoding.
pub const STATE_LAYOUT_VERSION: u32 = 3;
pub const EFFECTS_NAMESPACE: &str = "codex.effects.v3";
const EFFECT_KINDS: usize = 8;
const EFFECT_KEY_BYTES: usize = 17;
const EFFECT_HEADER_BYTES: usize = 256;

/// Stable ordering is event, link, pending version, conflict, pending result,
/// reachback, changed Message, deferred call impact. Source vector order and
/// sorted dirty addresses are retained without a second interpretation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub enum Effect {
    Event(Event),
    Link(Link),
    PendingVersion {
        message: usize,
        evidence_revision: u64,
    },
    Conflict(NativeConflict),
    PendingResult {
        message: usize,
        block: usize,
    },
    ReachedBack,
    MessageChanged {
        message: usize,
    },
    CallImpact {
        call_id: String,
    },
}
impl Effect {
    fn kind(&self) -> u8 {
        match self {
            Self::Event(_) => 1,
            Self::Link(_) => 2,
            Self::PendingVersion { .. } => 3,
            Self::Conflict(_) => 4,
            Self::PendingResult { .. } => 5,
            Self::ReachedBack => 6,
            Self::MessageChanged { .. } => 7,
            Self::CallImpact { .. } => 8,
        }
    }
}
#[derive(Serialize)]
enum EffectRef<'a> {
    Event(&'a Event),
    Link(&'a Link),
    PendingVersion {
        message: usize,
        evidence_revision: u64,
    },
    Conflict(&'a NativeConflict),
    PendingResult {
        message: usize,
        block: usize,
    },
    ReachedBack,
    MessageChanged {
        message: usize,
    },
    CallImpact {
        call_id: &'a str,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EffectsHeader {
    version: u32,
    step: u64,
    counts: [u64; EFFECT_KINDS],
}
impl EffectsHeader {
    fn decode(bytes: &[u8], step: u64) -> Result<Self> {
        let header: Self =
            serde_json::from_slice(bytes).map_err(|e| invalid("effects header", e))?;
        if header.version != STATE_LAYOUT_VERSION || header.step != step || header.counts[5] > 1 {
            return Err(invalid(
                "effects header",
                "layout, step, or reachback count differs",
            ));
        }
        header
            .counts
            .iter()
            .try_fold(0u64, |n, c| n.checked_add(*c))
            .ok_or(StoreError::Range)?;
        Ok(header)
    }
}
mod flat_session;
pub(crate) use flat_session::{FlatSessionEvents, FlatSessionStep};
/// Only scalar position is durable. Opening a reader revalidates its fixed
/// header against the pinned snapshot; no decoded aggregate receipt is carried.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EffectsCursor {
    step: u64,
    kind: u8,
    ordinal: u64,
}
impl EffectsCursor {
    /// Current immutable receipt address; None after all family boundaries.
    /// The owner may retain this key before advance for atomic consumption,
    /// but must validate advance successfully before mutating any receipt.
    pub fn key(&self) -> Option<[u8; EFFECT_KEY_BYTES]> {
        (self.kind <= EFFECT_KINDS as u8 && self.kind > 0)
            .then(|| effect_key(self.step, self.kind, self.ordinal))
    }
}
pub enum EffectStep {
    Item(Effect),
    Boundary,
    Done,
}
pub struct EffectsReader<'a, 's> {
    attempt: &'a Attempt<'s>,
    header: EffectsHeader,
}
fn effect_key(step: u64, kind: u8, ordinal: u64) -> [u8; EFFECT_KEY_BYTES] {
    let mut key = [0; EFFECT_KEY_BYTES];
    key[..8].copy_from_slice(&step.to_be_bytes());
    key[8] = kind;
    key[9..].copy_from_slice(&ordinal.to_be_bytes());
    key
}
impl<'s> Attempt<'s> {
    fn effect_cell(&self, step: u64, kind: u8, ordinal: u64) -> Result<Option<Vec<u8>>> {
        self.charge(1, EFFECTS_NAMESPACE.len() + EFFECT_KEY_BYTES)?;
        let allowance = if kind == 0 {
            self.remaining().min(EFFECT_HEADER_BYTES)
        } else {
            self.remaining()
        };
        let bytes = self.snapshot.cell(
            EFFECTS_NAMESPACE,
            &effect_key(step, kind, ordinal),
            allowance,
        )?;
        if let Some(bytes) = &bytes {
            self.charge(0, bytes.len())?;
        }
        Ok(bytes)
    }
    pub fn effects(&self, step: u64) -> std::result::Result<EffectsReader<'_, 's>, String> {
        self.checked(|| {
            if step == 0
                || step
                    > u64::try_from(self.snapshot.head.token.step).map_err(|_| StoreError::Range)?
            {
                return Err(invalid("effects step", "outside committed snapshot"));
            }
            let bytes = self.effect_cell(step, 0, 0)?.ok_or_else(|| {
                invalid(
                    "effects header",
                    "missing committed header (layout replay required)",
                )
            })?;
            let header = EffectsHeader::decode(&bytes, step)?;
            Ok(EffectsReader {
                attempt: self,
                header,
            })
        })
    }
}
impl EffectsReader<'_, '_> {
    pub fn header_key(&self) -> [u8; EFFECT_KEY_BYTES] {
        effect_key(self.header.step, 0, 0)
    }
    pub fn start(&self) -> EffectsCursor {
        EffectsCursor {
            step: self.header.step,
            kind: 1,
            ordinal: 0,
        }
    }
    /// At most one admitted point lookup. Each kind ends with an absence probe
    /// at its declared count, detecting a shortened header over contiguous
    /// immutable units. Boundary is a real cursor step, not a hidden scan.
    /// Failure leaves the cursor unchanged and fuses the whole Attempt.
    pub fn advance(&self, cursor: &mut EffectsCursor) -> std::result::Result<EffectStep, String> {
        self.attempt.checked(|| {
            if cursor.step != self.header.step
                || !(1..=EFFECT_KINDS as u8 + 1).contains(&cursor.kind)
            {
                return Err(invalid("effects cursor", "step or kind differs"));
            }
            if cursor.kind == EFFECT_KINDS as u8 + 1 {
                if cursor.ordinal != 0 {
                    return Err(invalid("effects cursor", "invalid completed ordinal"));
                }
                return Ok(EffectStep::Done);
            }
            let count = self.header.counts[usize::from(cursor.kind - 1)];
            if cursor.ordinal > count {
                return Err(invalid("effects cursor", "ordinal exceeds declared count"));
            }
            let bytes = self
                .attempt
                .effect_cell(cursor.step, cursor.kind, cursor.ordinal)?;
            if cursor.ordinal == count {
                if bytes.is_some() {
                    return Err(invalid(
                        "effects header",
                        "extra unit beyond declared count",
                    ));
                }
                cursor.kind += 1;
                cursor.ordinal = 0;
                return Ok(EffectStep::Boundary);
            }
            let bytes = bytes.ok_or_else(|| invalid("effect unit", "missing declared ordinal"))?;
            let unit: Effect =
                serde_json::from_slice(&bytes).map_err(|e| invalid("effect unit", e))?;
            if unit.kind() != cursor.kind {
                return Err(invalid("effect unit", "payload kind differs from key"));
            }
            cursor.ordinal += 1;
            Ok(EffectStep::Item(unit))
        })
    }
    /// Inspection convenience only: collects the same point cursor, subject to
    /// this Attempt's cumulative admission. It does not bound a whole result or
    /// certify canonical publication, hashing, or consumption of obligations.
    pub fn collect(&self) -> std::result::Result<Effects, String> {
        let mut cursor = self.start();
        let mut out = Effects::default();
        loop {
            match self.advance(&mut cursor)? {
                EffectStep::Item(Effect::Event(v)) => out.events.push(v),
                EffectStep::Item(Effect::Link(v)) => out.links.push(v),
                EffectStep::Item(Effect::PendingVersion {
                    message,
                    evidence_revision,
                }) => out.pending_versions.push((message, evidence_revision)),
                EffectStep::Item(Effect::Conflict(v)) => out.conflicts.push(v),
                EffectStep::Item(Effect::PendingResult { message, block }) => {
                    out.pending_results.push((message, block))
                }
                EffectStep::Item(Effect::ReachedBack) => out.reached_back = true,
                EffectStep::Item(Effect::MessageChanged { message }) => {
                    out.changed_messages.push(message)
                }
                EffectStep::Item(Effect::CallImpact { call_id }) => out.call_impacts.push(call_id),
                EffectStep::Boundary => {}
                EffectStep::Done => return Ok(out),
            }
        }
    }
}
/// Whole inspection DTO, never a persisted protocol or finalizer work queue.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Effects {
    pub events: Vec<Event>,
    pub links: Vec<Link>,
    pub reached_back: bool,
    pub pending_versions: Vec<(usize, u64)>,
    pub conflicts: Vec<NativeConflict>,
    pub pending_results: Vec<(usize, usize)>,
    pub changed_messages: Vec<usize>,
    pub call_impacts: Vec<String>,
}
/// Owns encoded buffers only. It excludes the source Delta, SQLite snapshot,
/// decoder allocations and canonical output rows, all separately owner-charged.
pub struct Encoded {
    pub changes: Vec<CellChange>,
    pub encoded_bytes: usize,
    pub allocated_capacity_bytes: usize,
}
fn add_bytes(total: &mut usize, n: usize, budget: Budget) -> Result<()> {
    let required = total.saturating_add(n);
    if required > budget.encoded_bytes {
        return Err(admission("Codex delta encoded bytes", required));
    }
    *total = required;
    Ok(())
}
/// The caller sets Delta.resume.parsed_upto from complete framing progress and
/// handles reached_back before application. next_step is pinned predecessor step
/// + 1, never a byte offset; the writer must reject a stale predecessor.
/// This encoding earns no canonical row completeness or finalization claim.
pub fn encode_delta(delta: &Delta, next_step: u64, budget: Budget) -> Result<Encoded> {
    if next_step == 0 {
        return Err(invalid("effects step", "committed steps start at one"));
    }
    let impacts = impacts(delta, budget)?;
    let counts = [
        delta.events.len(),
        delta.links.len(),
        delta.pending_versions.len(),
        delta.conflicts.len(),
        delta.pending_results.len(),
        usize::from(delta.reached_back),
        impacts.messages.len(),
        impacts.calls.len(),
    ];
    let effect_units = counts
        .iter()
        .try_fold(0usize, |n, c| n.checked_add(*c))
        .ok_or_else(|| admission("Codex delta cells", usize::MAX))?;
    let receipt_cells = effect_units
        .checked_add(1)
        .and_then(|n| n.checked_add(impacts.local_cells))
        .ok_or(StoreError::Range)?;
    let cells = delta
        .cells
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_add(receipt_cells))
        .ok_or(StoreError::Range)?;
    if cells > budget.cells {
        return Err(admission("Codex delta cells", cells));
    }
    // Reject the fixed encoded lower bound before growing the output Vec.
    let minimum = receipt_cells
        .checked_mul(EFFECTS_NAMESPACE.len() + EFFECT_KEY_BYTES)
        .and_then(|n| {
            impacts
                .local_cells
                .checked_mul(IMPACT_KEY_BYTES - EFFECT_KEY_BYTES)
                .and_then(|m| n.checked_add(m))
        })
        .and_then(|n| {
            delta
                .cells
                .len()
                .checked_mul(PREFIX.len())
                .and_then(|m| n.checked_add(m))
        })
        .and_then(|n| n.checked_add(RESUME_NAMESPACE.len()))
        .ok_or(StoreError::Range)?;
    if minimum > budget.encoded_bytes {
        return Err(admission("Codex delta encoded bytes", minimum));
    }
    let mut changes = Vec::new();
    changes
        .try_reserve_exact(cells)
        .map_err(|e| StoreError::Io(std::io::Error::other(e)))?;
    let mut bytes = 0;
    add_bytes(
        &mut bytes,
        RESUME_NAMESPACE.len() + RESUME_KEY.len(),
        budget,
    )?;
    let resume = limited(
        budget.encoded_bytes - bytes,
        bytes,
        "Codex delta encoded bytes",
        |w| serde_json::to_writer(w, &delta.resume),
    )?;
    add_bytes(&mut bytes, resume.len(), budget)?;
    changes.push(CellChange {
        namespace: RESUME_NAMESPACE.into(),
        key: RESUME_KEY.into(),
        value: Some(resume),
    });
    let header = EffectsHeader {
        version: STATE_LAYOUT_VERSION,
        step: next_step,
        counts: counts.map(|n| n as u64),
    };
    push_effect(&mut changes, &mut bytes, next_step, 0, 0, &header, budget)?;
    for (n, v) in delta.events.iter().enumerate() {
        push_effect(
            &mut changes,
            &mut bytes,
            next_step,
            1,
            n as u64,
            &EffectRef::Event(v),
            budget,
        )?;
    }
    for (n, v) in delta.links.iter().enumerate() {
        push_effect(
            &mut changes,
            &mut bytes,
            next_step,
            2,
            n as u64,
            &EffectRef::Link(v),
            budget,
        )?;
    }
    for (n, v) in delta.pending_versions.iter().enumerate() {
        push_effect(
            &mut changes,
            &mut bytes,
            next_step,
            3,
            n as u64,
            &EffectRef::PendingVersion {
                message: v.message,
                evidence_revision: v.evidence_revision,
            },
            budget,
        )?;
    }
    for (n, v) in delta.conflicts.iter().enumerate() {
        push_effect(
            &mut changes,
            &mut bytes,
            next_step,
            4,
            n as u64,
            &EffectRef::Conflict(v),
            budget,
        )?;
    }
    for (n, (message, block)) in delta.pending_results.iter().enumerate() {
        push_effect(
            &mut changes,
            &mut bytes,
            next_step,
            5,
            n as u64,
            &EffectRef::PendingResult {
                message: *message,
                block: *block,
            },
            budget,
        )?;
    }
    if delta.reached_back {
        push_effect(
            &mut changes,
            &mut bytes,
            next_step,
            6,
            0,
            &EffectRef::ReachedBack,
            budget,
        )?;
    }
    for (n, (message, impact)) in impacts.messages.iter().enumerate() {
        push_effect(
            &mut changes,
            &mut bytes,
            next_step,
            7,
            n as u64,
            &EffectRef::MessageChanged { message: *message },
            budget,
        )?;
        let header = ImpactHeader {
            version: STATE_LAYOUT_VERSION,
            step: next_step,
            message: *message,
            header_dirty: impact.header_dirty,
            counts: [impact.blocks.len() as u64, impact.witnesses.len() as u64],
        };
        push_impact(
            &mut changes,
            &mut bytes,
            next_step,
            *message,
            0,
            0,
            &header,
            budget,
        )?;
        for (ordinal, block) in impact.blocks.iter().enumerate() {
            push_impact(
                &mut changes,
                &mut bytes,
                next_step,
                *message,
                1,
                ordinal as u64,
                &ImpactRef::Block { block: *block },
                budget,
            )?;
        }
        for (ordinal, (key, slot)) in impact.witnesses.iter().enumerate() {
            push_impact(
                &mut changes,
                &mut bytes,
                next_step,
                *message,
                2,
                ordinal as u64,
                &ImpactRef::MessageWitness { key, slot },
                budget,
            )?;
        }
    }
    for (n, call) in impacts.calls.iter().enumerate() {
        push_effect(
            &mut changes,
            &mut bytes,
            next_step,
            8,
            n as u64,
            &EffectRef::CallImpact { call_id: call },
            budget,
        )?;
    }
    for (key, value) in &delta.cells {
        if let Some(value) = value {
            if !value.valid_for(key) {
                return Err(StoreError::Invariant(
                    "Codex delta cell shape differs from provider key".into(),
                ));
            }
        }
        let family = key.family();
        let ns_len = PREFIX.len() + family.len();
        add_bytes(&mut bytes, ns_len, budget)?;
        let key_len = key
            .ordered_len_limited(budget.encoded_bytes - bytes)
            .map_err(|needed| {
                admission("Codex delta encoded bytes", bytes.saturating_add(needed))
            })?;
        add_bytes(&mut bytes, key_len, budget)?;
        let encoded = limited(key_len, 0, "Codex delta key bytes", |w| {
            key.write_ordered(w)
        })?;
        let value = match value {
            Some(value) => {
                let encoded = limited(
                    budget.encoded_bytes - bytes,
                    bytes,
                    "Codex delta encoded bytes",
                    |w| serde_json::to_writer(w, value),
                )?;
                add_bytes(&mut bytes, encoded.len(), budget)?;
                Some(encoded)
            }
            None => None,
        };
        let mut namespace = String::with_capacity(ns_len);
        namespace.push_str(PREFIX);
        namespace.push_str(family);
        changes.push(CellChange {
            namespace,
            key: encoded,
            value,
        });
    }
    let allocated_capacity_bytes = changes.iter().fold(
        changes
            .capacity()
            .saturating_mul(std::mem::size_of::<CellChange>()),
        |n, c| {
            n.saturating_add(c.namespace.capacity())
                .saturating_add(c.key.capacity())
                .saturating_add(c.value.as_ref().map_or(0, Vec::capacity))
        },
    );
    Ok(Encoded {
        changes,
        encoded_bytes: bytes,
        allocated_capacity_bytes,
    })
}

fn push_effect(
    changes: &mut Vec<CellChange>,
    bytes: &mut usize,
    step: u64,
    kind: u8,
    ordinal: u64,
    unit: &impl Serialize,
    budget: Budget,
) -> Result<()> {
    add_bytes(bytes, EFFECTS_NAMESPACE.len() + EFFECT_KEY_BYTES, budget)?;
    let value = limited(
        budget.encoded_bytes - *bytes,
        *bytes,
        "Codex delta encoded bytes",
        |w| serde_json::to_writer(w, unit),
    )?;
    add_bytes(bytes, value.len(), budget)?;
    changes.push(CellChange {
        namespace: EFFECTS_NAMESPACE.into(),
        key: effect_key(step, kind, ordinal).to_vec(),
        value: Some(value),
    });
    Ok(())
}

// The same immutable step receipt, addressed by message rather than filtering
// its global stream. These are addresses/obligations, never copied Messages.
const IMPACT_KEY_BYTES: usize = 26;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub enum Impact {
    Block { block: usize },
    MessageWitness { key: String, slot: Slot },
}
#[derive(Serialize)]
enum ImpactRef<'a> {
    Block { block: usize },
    MessageWitness { key: &'a str, slot: &'a Slot },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImpactHeader {
    version: u32,
    step: u64,
    message: usize,
    header_dirty: bool,
    counts: [u64; 2],
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ImpactCursor {
    step: u64,
    message: usize,
    kind: u8,
    ordinal: u64,
}
impl ImpactCursor {
    pub fn key(&self) -> Option<[u8; IMPACT_KEY_BYTES]> {
        (1..=2)
            .contains(&self.kind)
            .then(|| impact_key(self.step, self.message, self.kind, self.ordinal))
    }
}
pub enum ImpactStep {
    Item(Impact),
    Boundary,
    Done,
}
pub struct ImpactsReader<'a, 's> {
    attempt: &'a Attempt<'s>,
    header: ImpactHeader,
}
fn impact_key(step: u64, message: usize, kind: u8, ordinal: u64) -> [u8; IMPACT_KEY_BYTES] {
    let mut key = [0; IMPACT_KEY_BYTES];
    key[..8].copy_from_slice(&step.to_be_bytes());
    key[8] = 9;
    key[9..17].copy_from_slice(&(message as u64).to_be_bytes());
    key[17] = kind;
    key[18..].copy_from_slice(&ordinal.to_be_bytes());
    key
}
impl<'s> Attempt<'s> {
    fn impact_cell(
        &self,
        step: u64,
        message: usize,
        kind: u8,
        ordinal: u64,
    ) -> Result<Option<Vec<u8>>> {
        self.charge(1, EFFECTS_NAMESPACE.len() + IMPACT_KEY_BYTES)?;
        let allowance = if kind == 0 {
            self.remaining().min(EFFECT_HEADER_BYTES)
        } else {
            self.remaining()
        };
        let bytes = self.snapshot.cell(
            EFFECTS_NAMESPACE,
            &impact_key(step, message, kind, ordinal),
            allowance,
        )?;
        if let Some(bytes) = &bytes {
            self.charge(0, bytes.len())?;
        }
        Ok(bytes)
    }
    /// One direct header lookup; unrelated Messages' impacts are never read.
    pub fn message_impacts(
        &self,
        step: u64,
        message: usize,
    ) -> std::result::Result<ImpactsReader<'_, 's>, String> {
        self.checked(|| {
            if step == 0
                || step
                    > u64::try_from(self.snapshot.head.token.step).map_err(|_| StoreError::Range)?
            {
                return Err(invalid("impact step", "outside committed snapshot"));
            }
            let bytes = self
                .impact_cell(step, message, 0, 0)?
                .ok_or_else(|| invalid("impact header", "missing declared Message impacts"))?;
            let header: ImpactHeader =
                serde_json::from_slice(&bytes).map_err(|e| invalid("impact header", e))?;
            if header.version != STATE_LAYOUT_VERSION
                || header.step != step
                || header.message != message
                || header.counts[0].checked_add(header.counts[1]).is_none()
            {
                return Err(invalid(
                    "impact header",
                    "layout, identity or count differs",
                ));
            }
            Ok(ImpactsReader {
                attempt: self,
                header,
            })
        })
    }
}
impl ImpactsReader<'_, '_> {
    pub fn header_dirty(&self) -> bool {
        self.header.header_dirty
    }
    pub fn header_key(&self) -> [u8; IMPACT_KEY_BYTES] {
        impact_key(self.header.step, self.header.message, 0, 0)
    }
    pub fn start(&self) -> ImpactCursor {
        ImpactCursor {
            step: self.header.step,
            message: self.header.message,
            kind: 1,
            ordinal: 0,
        }
    }
    /// One whole admitted unit or one count-boundary absence probe. This cursor
    /// is persisted only after the owner's corresponding canonical mutation.
    pub fn advance(&self, cursor: &mut ImpactCursor) -> std::result::Result<ImpactStep, String> {
        self.attempt.checked(|| {
            if cursor.step != self.header.step
                || cursor.message != self.header.message
                || !(1..=3).contains(&cursor.kind)
            {
                return Err(invalid("impact cursor", "identity or kind differs"));
            }
            if cursor.kind == 3 {
                return if cursor.ordinal == 0 {
                    Ok(ImpactStep::Done)
                } else {
                    Err(invalid("impact cursor", "invalid completed ordinal"))
                };
            }
            let count = self.header.counts[usize::from(cursor.kind - 1)];
            if cursor.ordinal > count {
                return Err(invalid("impact cursor", "ordinal exceeds declared count"));
            }
            let bytes = self.attempt.impact_cell(
                cursor.step,
                cursor.message,
                cursor.kind,
                cursor.ordinal,
            )?;
            if cursor.ordinal == count {
                if bytes.is_some() {
                    return Err(invalid("impact header", "extra unit beyond declared count"));
                }
                cursor.kind += 1;
                cursor.ordinal = 0;
                return Ok(ImpactStep::Boundary);
            }
            let bytes = bytes.ok_or_else(|| invalid("impact unit", "missing declared ordinal"))?;
            let unit: Impact =
                serde_json::from_slice(&bytes).map_err(|e| invalid("impact unit", e))?;
            match &unit {
                Impact::Block { .. } if cursor.kind == 1 => {}
                Impact::MessageWitness { key, slot }
                    if cursor.kind == 2
                        && codex::presence::message_slot(key).as_ref() == Some(slot) => {}
                _ => {
                    return Err(invalid(
                        "impact unit",
                        "kind or source-owned witness slot differs",
                    ));
                }
            }
            cursor.ordinal += 1;
            Ok(ImpactStep::Item(unit))
        })
    }
}
#[derive(Default)]
struct MessageImpacts {
    header_dirty: bool,
    blocks: BTreeSet<usize>,
    witnesses: BTreeMap<String, Slot>,
}
struct Impacts {
    messages: BTreeMap<usize, MessageImpacts>,
    calls: BTreeSet<String>,
    local_cells: usize,
    cells: usize,
    bytes: usize,
    budget: Budget,
}
impl Impacts {
    fn charge(&mut self, cells: usize, bytes: usize) -> Result<()> {
        let n = self.cells.checked_add(cells).ok_or(StoreError::Range)?;
        if n > self.budget.cells {
            return Err(admission("Codex impact cells", n));
        }
        let b = self.bytes.checked_add(bytes).ok_or(StoreError::Range)?;
        if b > self.budget.encoded_bytes {
            return Err(admission("Codex impact address bytes", b));
        }
        self.cells = n;
        self.bytes = b;
        Ok(())
    }
    fn message(&mut self, message: usize) -> Result<&mut MessageImpacts> {
        if !self.messages.contains_key(&message) {
            self.charge(
                2,
                2 * EFFECTS_NAMESPACE.len() + EFFECT_KEY_BYTES + IMPACT_KEY_BYTES,
            )?;
            self.local_cells += 1;
            self.messages.insert(message, MessageImpacts::default());
        }
        self.messages
            .get_mut(&message)
            .ok_or_else(|| invalid("impact message", "admitted message is missing"))
    }
    fn block(&mut self, message: usize, block: usize) -> Result<()> {
        if !self.message(message)?.blocks.contains(&block) {
            self.charge(1, EFFECTS_NAMESPACE.len() + IMPACT_KEY_BYTES)?;
            self.local_cells += 1;
            self.messages
                .get_mut(&message)
                .ok_or_else(|| invalid("impact message", "admitted message is missing"))?
                .blocks
                .insert(block);
        }
        Ok(())
    }
    fn witness(&mut self, message: usize, key: &str) -> Result<()> {
        if !self.message(message)?.witnesses.contains_key(key) {
            self.charge(
                1,
                (EFFECTS_NAMESPACE.len() + IMPACT_KEY_BYTES)
                    .saturating_add(key.len().saturating_mul(2)),
            )?;
            let slot = codex::presence::message_slot(key).ok_or_else(|| {
                invalid("impact witness", "unknown source-owned message witness key")
            })?;
            self.local_cells += 1;
            self.messages
                .get_mut(&message)
                .ok_or_else(|| invalid("impact message", "admitted message is missing"))?
                .witnesses
                .insert(key.into(), slot);
        }
        Ok(())
    }
    fn call(&mut self, call: &str) -> Result<()> {
        if !self.calls.contains(call) {
            self.charge(
                1,
                (EFFECTS_NAMESPACE.len() + EFFECT_KEY_BYTES).saturating_add(call.len()),
            )?;
            self.calls.insert(call.into());
        }
        Ok(())
    }
}
fn impacts(delta: &Delta, budget: Budget) -> Result<Impacts> {
    let base_units = [
        delta.events.len(),
        delta.links.len(),
        delta.pending_versions.len(),
        delta.conflicts.len(),
        delta.pending_results.len(),
        usize::from(delta.reached_back),
    ]
    .iter()
    .try_fold(0usize, |n, c| n.checked_add(*c))
    .ok_or(StoreError::Range)?;
    let base_cells = delta
        .cells
        .len()
        .checked_add(base_units)
        .and_then(|n| n.checked_add(2))
        .ok_or(StoreError::Range)?;
    let base_bytes = base_units
        .checked_add(1)
        .ok_or(StoreError::Range)?
        .checked_mul(EFFECTS_NAMESPACE.len() + EFFECT_KEY_BYTES)
        .and_then(|n| {
            delta
                .cells
                .len()
                .checked_mul(PREFIX.len())
                .and_then(|m| n.checked_add(m))
        })
        .and_then(|n| n.checked_add(RESUME_NAMESPACE.len()))
        .ok_or(StoreError::Range)?;
    let mut out = Impacts {
        messages: BTreeMap::new(),
        calls: BTreeSet::new(),
        local_cells: 0,
        cells: 0,
        bytes: 0,
        budget,
    };
    out.charge(base_cells, base_bytes)?;
    for v in &delta.pending_versions {
        out.message(v.message)?;
    }
    for v in &delta.conflicts {
        out.message(v.message)?;
    }
    for (message, block) in &delta.pending_results {
        out.block(*message, *block)?;
    }
    for (key, value) in &delta.cells {
        match key {
            Key::Message(message) => out.message(*message)?.header_dirty = true,
            Key::Block(message, block)
            | Key::PresenceBlock(message, block)
            | Key::PendingResult(message, block)
            | Key::OutputChunk(message, block, _)
            | Key::CallBlock(_, message, block) => out.block(*message, *block)?,
            Key::Presence(message, key) => out.witness(*message, key)?,
            Key::Continuation(call) | Key::ChildCount(call) | Key::Child(call, _) => {
                out.call(call)?
            }
            _ => {}
        }
        if matches!(key, Key::Block(_, _)) && value.is_none() {
            return Err(invalid(
                "impact block",
                "source block deletion requires an explicit projection contract",
            ));
        }
    }
    Ok(out)
}
fn push_impact(
    changes: &mut Vec<CellChange>,
    bytes: &mut usize,
    step: u64,
    message: usize,
    kind: u8,
    ordinal: u64,
    unit: &impl Serialize,
    budget: Budget,
) -> Result<()> {
    add_bytes(bytes, EFFECTS_NAMESPACE.len() + IMPACT_KEY_BYTES, budget)?;
    let value = limited(
        budget.encoded_bytes - *bytes,
        *bytes,
        "Codex delta encoded bytes",
        |w| serde_json::to_writer(w, unit),
    )?;
    add_bytes(bytes, value.len(), budget)?;
    changes.push(CellChange {
        namespace: EFFECTS_NAMESPACE.into(),
        key: impact_key(step, message, kind, ordinal).to_vec(),
        value: Some(value),
    });
    Ok(())
}

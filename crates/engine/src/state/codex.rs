//! Codex typed cells over one pinned disposable SQLite parser-state snapshot.
//! Interpretation and encoding happen outside the output writer transaction.
//! Budgets measure cumulative encoded cell traffic, not allocator/RSS usage.
use std::cell::{Cell as Counter, RefCell};
use std::io::Write;

use polylogue_sources::codex::{
    self,
    keyed::{Cell, Delta, Key, ReadView, Resume},
};
use polylogue_store::Error as StoreError;
use polylogue_store::parser_state::{Budget, CellChange, Snapshot};
type Result<T> = std::result::Result<T, StoreError>;
use polylogue_model::{Event, Link};
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
/// Step-scoped obligations, retained until their owning output transaction
/// commits. They are not proof of canonical publication or completed hashing.
pub const EFFECTS_NAMESPACE: &str = "codex.effects";
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Effects {
    pub events: Vec<Event>,
    pub links: Vec<Link>,
    pub reached_back: bool,
    /// (stable provider message index, evidence revision to revalidate).
    pub pending_versions: Vec<(usize, u64)>,
    pub conflicts: Vec<NativeConflict>,
    pub pending_results: Vec<(usize, usize)>,
}
impl Effects {
    /// The owner admits the encoded cell before decoding. Decoder allocations
    /// and whole-cell effects are not a hard RSS bound.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| invalid("effects", e))
    }
}
#[derive(Serialize)]
struct EffectsRef<'a> {
    events: &'a [Event],
    links: &'a [Link],
    reached_back: bool,
    pending_versions: VersionsRef<'a>,
    conflicts: &'a [NativeConflict],
    pending_results: &'a [(usize, usize)],
}
struct VersionsRef<'a>(&'a [codex::keyed::PendingVersion]);
impl Serialize for VersionsRef<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for v in self.0 {
            seq.serialize_element(&(v.message, v.evidence_revision))?;
        }
        seq.end()
    }
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
    let cells = delta
        .cells
        .len()
        .checked_add(2)
        .ok_or_else(|| admission("Codex delta cells", usize::MAX))?;
    if cells > budget.cells {
        return Err(admission("Codex delta cells", cells));
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
    add_bytes(&mut bytes, EFFECTS_NAMESPACE.len() + 8, budget)?;
    let effects = EffectsRef {
        events: &delta.events,
        links: &delta.links,
        reached_back: delta.reached_back,
        pending_versions: VersionsRef(&delta.pending_versions),
        conflicts: &delta.conflicts,
        pending_results: &delta.pending_results,
    };
    let value = limited(
        budget.encoded_bytes - bytes,
        bytes,
        "Codex delta encoded bytes",
        |w| serde_json::to_writer(w, &effects),
    )?;
    add_bytes(&mut bytes, value.len(), budget)?;
    changes.push(CellChange {
        namespace: EFFECTS_NAMESPACE.into(),
        key: next_step.to_be_bytes().to_vec(),
        value: Some(value),
    });
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

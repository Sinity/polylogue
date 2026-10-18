//! Atomic installation of one worker-verified small normalized field and its
//! source descriptor. Streaming/in-flight producers keep their existing API.
use super::{Build, ProjectionStep, blocks, writer};
use crate::{Error, Result, Writer, db, values};
use polylogue_model::merge::Slot;
use rusqlite::{Transaction, params};
use std::path::Path;

/// Immutable worker-hashed chunks, not source-selector authority. The engine
/// must verify the selected source against these exact normalized bytes first.
pub struct SmallField {
    value: values::small::Prepared,
    origin: values::small::Prepared,
}
impl SmallField {
    pub fn prepare(value: Vec<u8>, origin: Vec<u8>, chunk_allowance: usize) -> Result<Self> {
        Ok(Self {
            value: values::small::Prepared::json(value, chunk_allowance)?,
            origin: values::small::Prepared::json(origin, chunk_allowance)?,
        })
    }
    pub fn value_hash(&self) -> [u8; 32] {
        self.value.hash()
    }
    pub fn encoded_bytes(&self) -> usize {
        self.value.len().saturating_add(self.origin.len())
    }
    pub fn hash_work(&self) -> u64 {
        (self.value.len() as u64)
            .saturating_mul(2)
            .saturating_add(self.origin.len() as u64)
    }
    pub fn value_length(&self) -> u64 {
        self.value.len() as u64
    }
    /// The worker and writer use the same aggregate admission. Reject before
    /// allocation; a worker can then choose the unchanged streaming path.
    pub fn admit(
        &self,
        slot: &Slot,
        block: bool,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        let key = key(slot, block, allowance)?;
        let bytes = self
            .value
            .len()
            .checked_add(self.origin.len())
            .and_then(|n| n.checked_add(key.len()))
            .ok_or(Error::Range)?;
        blocks::admit(step, allowance.saturating_sub(bytes)).map_err(|error| match error {
            Error::ParserAdmission { required, .. } => Error::ParserAdmission {
                resource: "small field metadata bytes",
                required: required.saturating_add(bytes as u64),
            },
            other => other,
        })
    }
}
/// One verified small field and its exact intermediate cursor checkpoint.
pub struct SmallFieldEntry {
    pub slot: Slot,
    pub source_time: Option<i64>,
    pub supplied: bool,
    pub block: bool,
    pub field: SmallField,
    pub expected_sequence: u64,
    pub next_cursor: Vec<u8>,
}
impl SmallFieldEntry {
    pub fn retained_bytes(&self, allowance: usize) -> Result<usize> {
        let key = key(&self.slot, self.block, allowance)?;
        self.field
            .value
            .len()
            .checked_add(self.field.origin.len())
            .and_then(|n| n.checked_add(self.next_cursor.len()))
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .and_then(|n| n.checked_add(key.len()))
            .ok_or(Error::Range)
    }
    fn step(&self) -> ProjectionStep<'_> {
        ProjectionStep {
            expected_sequence: self.expected_sequence,
            next_cursor: &self.next_cursor,
        }
    }
}
fn write_in(
    tx: &Transaction<'_>,
    path: &Path,
    build: &Build,
    slot: &Slot,
    source_time: Option<i64>,
    supplied: bool,
    block: bool,
    field: &SmallField,
    step: &ProjectionStep<'_>,
    allowance: usize,
) -> Result<()> {
    field.admit(slot, block, step, allowance)?;
    let key = key(slot, block, allowance)?;
    writer::check(tx, path, build)?;
    let active: bool = db(path, || {
        tx.query_row("SELECT active_value_pk IS NOT NULL OR active_origin_value_pk IS NOT NULL FROM message_build WHERE pk=?1",[build.pk],|r|r.get(0))
    })?;
    if active {
        return Err(Error::MessagePreparationPending);
    }
    let value = field.value.install_in(tx, path)?;
    let origin = field.origin.install_in(tx, path)?;
    db(path, || {
        tx.execute(
            "UPDATE message_build SET active_value_pk=?2,active_origin_value_pk=?3 WHERE pk=?1",
            params![build.pk, value.id.pk(), origin.id.pk()],
        )
    })?;
    if block {
        blocks::write_field_in(tx, path, build, slot, &key, source_time, supplied)?;
    } else {
        writer::write_field_in(tx, path, build, &key, None, source_time, supplied)?;
    }
    blocks::commit_cursor(tx, path, build, step)
}
fn key(slot: &Slot, block: bool, allowance: usize) -> Result<Vec<u8>> {
    if block {
        return writer::block_field_key(slot, allowance);
    }
    if matches!(
        slot,
        Slot::Block { .. } | Slot::Extra { block: Some(_), .. }
    ) {
        return Err(Error::Invariant(
            "block-local fields require active block ownership".into(),
        ));
    }
    writer::field_key(slot, allowance)
}
impl Writer {
    /// Chunks, dedup/producer leases, shared field validation/root transfer and
    /// cursor CAS commit together. A failed late check leaves no visible IDs.
    pub fn write_small_field(
        &mut self,
        build: &Build,
        slot: &Slot,
        source_time: Option<i64>,
        supplied: bool,
        block: bool,
        field: &SmallField,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        write_in(
            &tx,
            &path,
            build,
            slot,
            source_time,
            supplied,
            block,
            field,
            step,
            allowance,
        )?;
        db(&path, || tx.commit())?;
        Ok(())
    }
    /// All intermediate fields and cursor CAS steps commit together. Each member
    /// uses the single-field live checks; any late failure rolls back the run.
    pub fn write_small_field_run(
        &mut self,
        build: &Build,
        entries: &[SmallFieldEntry],
        allowance: usize,
    ) -> Result<()> {
        let mut retained = 0usize;
        if entries.is_empty() {
            return Err(Error::Invariant("empty small field run".into()));
        }
        for entry in entries {
            retained = retained
                .checked_add(entry.retained_bytes(allowance)?)
                .ok_or(Error::Range)?;
            if retained > allowance {
                return Err(Error::ParserAdmission {
                    resource: "small field run bytes",
                    required: retained as u64,
                });
            }
            entry
                .field
                .admit(&entry.slot, entry.block, &entry.step(), allowance)?;
        }
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        for entry in entries {
            write_in(
                &tx,
                &path,
                build,
                &entry.slot,
                entry.source_time,
                entry.supplied,
                entry.block,
                &entry.field,
                &entry.step(),
                allowance,
            )?;
        }
        db(&path, || tx.commit())?;
        Ok(())
    }
}

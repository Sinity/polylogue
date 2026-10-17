//! Atomic installation of one worker-verified small normalized field and its
//! source descriptor. Streaming/in-flight producers keep their existing API.
use super::{Build, ProjectionStep, blocks, writer};
use crate::{Error, Result, Writer, db, values};
use polylogue_model::merge::Slot;
use rusqlite::params;

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
        field.admit(slot, block, step, allowance)?;
        let key = key(slot, block, allowance)?;
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check(&tx, &path, build)?;
        let active: bool = db(&path, || {
            tx.query_row("SELECT active_value_pk IS NOT NULL OR active_origin_value_pk IS NOT NULL FROM message_build WHERE pk=?1",[build.pk],|r|r.get(0))
        })?;
        if active {
            return Err(Error::MessagePreparationPending);
        }
        let value = field.value.install_in(&tx, &path)?;
        let origin = field.origin.install_in(&tx, &path)?;
        db(&path, || {
            tx.execute(
                "UPDATE message_build SET active_value_pk=?2,active_origin_value_pk=?3 WHERE pk=?1",
                params![build.pk, value.id.pk(), origin.id.pk()],
            )
        })?;
        if block {
            blocks::write_field_in(&tx, &path, build, slot, &key, source_time, supplied)?;
        } else {
            writer::write_field_in(&tx, &path, build, &key, None, source_time, supplied)?;
        }
        blocks::commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(())
    }
}

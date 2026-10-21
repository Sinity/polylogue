//! Select current unit references before opening any payload, then install one
//! complete view in a transaction. Historical manifests contain no row bodies.
use super::{ExcisionPublication, Unit};
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::io::Read;

mod batch;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestoreUnit {
    pub unit: Unit,
    pub batch: Option<String>,
    /// Older original events do not carry a byte count.
    pub bytes: Option<u64>,
    pub rows: u64,
    pub blobs: Vec<String>,
    pub excision: Option<ExcisionPublication>,
}

mod catalog;
pub use catalog::{Catalog, Selection};
#[cfg(test)]
pub(super) fn select(units: &[RestoreUnit]) -> Result<Selection> {
    let mut catalog = Catalog::new()?;
    for (index, unit) in units.iter().enumerate() {
        catalog.push(index as u64 + 1, unit)?;
    }
    catalog.finish()
}

pub(super) fn install<R: Read>(
    conn: &mut Connection,
    units: &Selection,
    identity: Option<&str>,
    mut fetch: impl FnMut(&RestoreUnit) -> Result<Option<R>>,
) -> Result<u64> {
    // Hash all descriptors, including tombstones. A repeat cannot reapply stale
    // row bodies, and the checkpoint commits with the corresponding rows.
    let mut fingerprint = blake3::Hasher::new();
    serde_json::to_writer(&mut fingerprint, &(identity, units))?;
    let fingerprint = fingerprint.finalize().to_hex().to_string();
    let tx = conn.transaction()?;
    if let Some(archive_id) = identity {
        if archive_id.is_empty() {
            return Err(invalid("restored archive has no identity"));
        }
        let current: Option<String> = tx
            .query_row("SELECT value FROM meta WHERE key='archive_id'", [], |r| {
                r.get(0)
            })
            .optional()?;
        if current.as_deref() != Some(archive_id) && populated(&tx)? {
            return Err(invalid("restore target is a nonempty different archive"));
        }
        // An originating archive need not have a restore checkpoint. Its local
        // excisions still take precedence over older confirmed row bodies.
        let mut local = tx.prepare("SELECT commit_seq,id FROM excision")?;
        for unit in local.query_map([], |row| {
            Ok(Unit {
                commit_seq: row.get(0)?,
                kind: "excision".into(),
                key: row.get(1)?,
            })
        })? {
            let unit = unit?;
            let confirmed = units.find(&unit)?;
            let publication = super::excision::publication(&tx, &unit)?;
            if confirmed.is_none_or(|confirmed| confirmed.excision != publication) {
                return Err(invalid("confirmed view does not include a local excision"));
            }
        }
    }
    let held: Option<String> = tx
        .query_row(
            "SELECT value FROM meta WHERE key='sinex_restored_view'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    let repeated = held.as_deref() == Some(&fingerprint);
    let excision_before = if repeated {
        None
    } else {
        Some((
            crate::excise::recovery::current(&tx)?,
            crate::excise::recovery::fingerprint(&tx)?,
        ))
    };
    if !repeated {
        crate::excise::recovery::remember_policies(&tx, &units.catalog.db)?;
        tx.execute_batch("PRAGMA defer_foreign_keys = ON")?;
        // All deletions precede inserts: shared revision bases are restored in
        // this transaction. A later read/refusal rolls all deletions back.
        units.each_unit(|unit| remove(&tx, unit))?;
    }
    let mut inserted = 0;
    units.each(|unit| {
        let payload = fetch(unit)?;
        inserted += batch::install(unit, payload, (!repeated).then_some(&tx), &units.catalog.db)?;
        Ok(())
    })?;
    if let Some(archive_id) = identity {
        if archive_id.is_empty() {
            return Err(invalid("restored archive has no identity"));
        }
        let seq = units.commit_seq();
        let current:i64=tx.query_row("SELECT COALESCE((SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'),0)",[],|r|r.get(0))?;
        for (key, value) in [
            ("archive_id", archive_id.to_owned()),
            ("write_counter", seq.max(current).to_string()),
        ] {
            tx.execute("INSERT INTO meta(key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[key,value.as_str()])?;
        }
    }
    tx.execute("INSERT INTO meta(key,value) VALUES ('sinex_restored_view',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [&fingerprint])?;
    if let Some((revision, before)) = excision_before
        && crate::excise::recovery::fingerprint(&tx)? == before
    {
        crate::excise::recovery::set(&tx, &revision)?;
    }
    if !repeated && !crate::excise::recovery::preserves_policies(&tx, &units.catalog.db)? {
        return Err(invalid("restored view changes an existing excision rule"));
    }
    tx.commit()?;
    Ok(inserted)
}

fn revision_key(unit: &Unit) -> Result<(&str, i64)> {
    let (input, number) = unit
        .key
        .rsplit_once('/')
        .ok_or_else(|| invalid("invalid revision unit key"))?;
    let number = number
        .parse::<i64>()
        .map_err(|_| invalid("invalid revision number"))?;
    if number < 1 {
        return Err(invalid("invalid revision number"));
    }
    Ok((input, number))
}

fn remove(conn: &Connection, unit: &Unit) -> Result<()> {
    let key = [&unit.key];
    match unit.kind.as_str() {
        "note" => {
            for table in ["note_target", "note_relation"] {
                conn.execute(&format!("DELETE FROM {table} WHERE note_id=?1"), key)?;
            }
            conn.execute("DELETE FROM note WHERE id=?1", key)?;
        }
        "delivery" => {
            conn.execute("DELETE FROM delivery_event WHERE delivery=?1", key)?;
            conn.execute("DELETE FROM delivery WHERE id=?1", key)?;
        }
        "excision" => {
            conn.execute(
                "DELETE FROM observation WHERE class='excised' AND excision=?1",
                key,
            )?;
            conn.execute("DELETE FROM excision WHERE id=?1", key)?;
        }
        "setting" => {
            conn.execute("DELETE FROM setting WHERE id=?1", key)?;
        }
        "revision" => {
            let (input, number) = revision_key(unit)?;
            conn.execute("DELETE FROM extent WHERE input=?1 AND offset IN (SELECT e.offset FROM extent e JOIN revision r ON r.input=e.input AND r.number=?2 AND r.chain=e.chain WHERE e.input=?1 AND e.offset<r.length AND e.offset>=COALESCE((SELECT MAX(p.length) FROM revision p WHERE p.input=r.input AND p.chain=r.chain AND p.number<r.number),0)) AND chain=(SELECT chain FROM revision WHERE input=?1 AND number=?2)",rusqlite::params![input,number])?;
            conn.execute(
                "DELETE FROM observation WHERE input=?1 AND revision=?2",
                rusqlite::params![input, number],
            )?;
            conn.execute(
                "DELETE FROM revision WHERE input=?1 AND number=?2",
                rusqlite::params![input, number],
            )?;
        }
        _ => return Err(invalid("unknown selected unit kind")),
    }
    Ok(())
}
fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}

/// Include retained tables outside the publication catalog: an archive with
/// only kept results, variants or client requests still belongs to its owner.
fn populated(conn: &Connection) -> Result<bool> {
    let tables:Vec<String>=conn.prepare("SELECT name FROM sqlite_schema WHERE type='table' AND name<>'meta' AND name NOT GLOB 'sqlite_*'")?
        .query_map([],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?;
    for table in tables {
        let quoted = table.replace('"', "\"\"");
        if conn.query_row(
            &format!("SELECT EXISTS(SELECT 1 FROM \"{quoted}\")"),
            [],
            |r| r.get::<_, bool>(0),
        )? {
            return Ok(true);
        }
    }
    let seq: i64 = conn.query_row(
        "SELECT COALESCE((SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'),0)",
        [],
        |r| r.get(0),
    )?;
    Ok(seq > 0)
}

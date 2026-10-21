//! Select current unit references before opening any payload, then install one
//! complete view in a transaction. Historical manifests contain no row bodies.
use super::{ExcisionPublication, Unit};
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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

/// A checked current view. Callers can inspect it, but only `select` can
/// construct one, so install cannot accept an unauthorised empty unit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct Selection(Vec<RestoreUnit>);
impl std::ops::Deref for Selection {
    type Target = [RestoreUnit];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Validate the confirmed catalog and fold later replacements before payload IO.
/// A replacement must name an earlier, exact unit identity in this archive.
pub fn select(units: &[RestoreUnit]) -> Result<Selection> {
    let mut selected = BTreeMap::new();
    let mut authorized_tombstones = std::collections::BTreeSet::new();
    let mut previous = None;
    for original in units {
        let unit = &original.unit;
        if unit.commit_seq < 1
            || unit.key.is_empty()
            || !["revision", "note", "excision", "setting", "delivery"]
                .contains(&unit.kind.as_str())
            || previous.is_some_and(|p: &Unit| p >= unit)
        {
            return Err(invalid(
                "confirmed unit identities are not in strict archive order",
            ));
        }
        previous = Some(unit);
        if unit.kind == "excision" && original.excision.is_none() {
            return Err(invalid("legacy excision has no replacement metadata"));
        }
        if let Some(hash) = &original.batch {
            super::excision::batch_hash(hash)?;
        } else if original.rows != 0 {
            return Err(invalid("a tombstone names rows"));
        }
        selected.insert(unit.clone(), original.clone());
        if let Some(publication) = &original.excision {
            if unit.kind != "excision" {
                return Err(invalid("replacement metadata belongs to no excision"));
            }
            publication.validate()?;
            for replacement in &publication.replacements {
                if replacement.unit >= *unit || !selected.contains_key(&replacement.unit) {
                    return Err(invalid("replacement names no earlier exact unit identity"));
                }
                if replacement.batch.is_none() {
                    authorized_tombstones.insert(replacement.unit.clone());
                }
                let inherited = selected
                    .get(&replacement.unit)
                    .and_then(|u| u.excision.clone());
                selected.insert(
                    replacement.unit.clone(),
                    RestoreUnit {
                        unit: replacement.unit.clone(),
                        batch: replacement.batch.clone(),
                        bytes: Some(replacement.bytes),
                        rows: replacement.rows,
                        blobs: replacement.blobs.clone(),
                        excision: inherited,
                    },
                );
            }
        }
    }
    for current in selected.values() {
        if (current.batch.is_none() && !authorized_tombstones.contains(&current.unit))
            || (current.batch.is_some() && current.rows == 0)
        {
            return Err(invalid("an empty unit has no confirmed excision tombstone"));
        }
    }
    Ok(Selection(selected.into_values().collect()))
}

pub(super) fn install(
    conn: &mut Connection,
    units: &Selection,
    identity: Option<&str>,
    mut fetch: impl FnMut(&RestoreUnit) -> Result<Option<Vec<u8>>>,
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
            let confirmed = units
                .binary_search_by(|candidate| candidate.unit.cmp(&unit))
                .ok()
                .map(|index| &units[index]);
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
    if !repeated {
        tx.execute_batch("PRAGMA defer_foreign_keys = ON")?;
        // All deletions precede inserts: shared revision bases are restored in
        // this transaction. A later read/refusal rolls all deletions back.
        for unit in units.iter() {
            remove(&tx, &unit.unit)?;
        }
    }
    let mut inserted = 0;
    for unit in units.iter() {
        let payload = fetch(unit)?;
        check_batch(unit, payload.as_deref())?;
        if !repeated && let Some(payload) = payload {
            inserted += super::apply(&tx, &payload)?;
        }
    }
    if let Some(archive_id) = identity {
        if archive_id.is_empty() {
            return Err(invalid("restored archive has no identity"));
        }
        let seq = units.iter().map(|u| u.unit.commit_seq).max().unwrap_or(0);
        let current:i64=tx.query_row("SELECT COALESCE((SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'),0)",[],|r|r.get(0))?;
        for (key, value) in [
            ("archive_id", archive_id.to_owned()),
            ("write_counter", seq.max(current).to_string()),
        ] {
            tx.execute("INSERT INTO meta(key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[key,value.as_str()])?;
        }
    }
    tx.execute("INSERT INTO meta(key,value) VALUES ('sinex_restored_view',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [&fingerprint])?;
    tx.commit()?;
    Ok(inserted)
}

fn check_batch(unit: &RestoreUnit, payload: Option<&[u8]>) -> Result<()> {
    match (&unit.batch, payload) {
        (None, None) if unit.rows == 0 => Ok(()),
        (Some(name), Some(bytes)) => {
            let hash = super::excision::batch_hash(name)?;
            if blake3::hash(bytes).as_bytes() != &hash.0
                || unit.bytes.is_some_and(|n| n != bytes.len() as u64)
            {
                return Err(invalid(
                    "selected unit payload failed its hash/length check",
                ));
            }
            let rows = bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty());
            if rows.clone().count() as u64 != unit.rows {
                return Err(invalid("selected unit row count differs"));
            }
            let mut inputs = std::collections::BTreeSet::new();
            let mut machines = std::collections::BTreeSet::new();
            if unit.unit.kind == "excision" {
                for line in rows.clone() {
                    let value: serde_json::Value = serde_json::from_slice(line)?;
                    let row = &value["row"];
                    if value["table"] == "observation" {
                        if let Some(input) = row["input"].as_str() {
                            inputs.insert(input.to_owned());
                        }
                        if let Some(machine) = row["machine"].as_str() {
                            machines.insert(machine.to_owned());
                        }
                    }
                    if value["table"] == "excision" {
                        let refused: serde_json::Value = serde_json::from_str(
                            row["refused_json"]
                                .as_str()
                                .ok_or_else(|| invalid("excision has no metadata"))?,
                        )?;
                        if let Some(replacements) =
                            refused["publication"]["replacements"].as_array()
                        {
                            for r in replacements {
                                if r["unit"]["kind"] == "revision" {
                                    let key = r["unit"]["key"].as_str().ok_or_else(|| {
                                        invalid("revision replacement has no key")
                                    })?;
                                    let (input, _) = key.rsplit_once('/').ok_or_else(|| {
                                        invalid("invalid revision replacement key")
                                    })?;
                                    inputs.insert(input.to_owned());
                                }
                            }
                        }
                    }
                }
            }
            let mut owning = 0;
            for line in rows {
                let value: serde_json::Value = serde_json::from_slice(line)?;
                let table = value["table"]
                    .as_str()
                    .ok_or_else(|| invalid("row names no table"))?;
                let row = &value["row"];
                let is_owner = match unit.unit.kind.as_str() {
                    "note" => match table {
                        "note" => {
                            owning += 1;
                            row["id"] == unit.unit.key && row["commit_seq"] == unit.unit.commit_seq
                        }
                        "note_target" | "note_relation" => row["note_id"] == unit.unit.key,
                        _ => false,
                    },
                    "delivery" => match table {
                        "delivery" => {
                            owning += 1;
                            row["id"] == unit.unit.key && row["commit_seq"] == unit.unit.commit_seq
                        }
                        "delivery_event" => row["delivery"] == unit.unit.key,
                        _ => false,
                    },
                    "setting" => {
                        owning += 1;
                        table == "setting"
                            && row["id"] == unit.unit.key
                            && row["commit_seq"] == unit.unit.commit_seq
                    }
                    "excision" => match table {
                        "excision" => {
                            owning += 1;
                            let refused: serde_json::Value = serde_json::from_str(
                                row["refused_json"]
                                    .as_str()
                                    .ok_or_else(|| invalid("excision has no metadata"))?,
                            )?;
                            let publication: ExcisionPublication =
                                serde_json::from_value(refused["publication"].clone())?;
                            if unit.excision.as_ref() != Some(&publication) {
                                return Err(invalid(
                                    "excision batch disagrees with its confirmed mapping",
                                ));
                            }
                            row["id"] == unit.unit.key && row["commit_seq"] == unit.unit.commit_seq
                        }
                        "observation" => {
                            row["class"] == "excised" && row["excision"] == unit.unit.key
                        }
                        "input" => row["id"].as_str().is_some_and(|id| inputs.contains(id)),
                        "machine" => row["id"].as_str().is_some_and(|id| machines.contains(id)),
                        _ => false,
                    },
                    "revision" => {
                        let (input, number) = revision_key(&unit.unit)?;
                        match table {
                            "revision" => {
                                owning += 1;
                                row["input"] == input
                                    && row["number"] == number
                                    && row["commit_seq"] == unit.unit.commit_seq
                            }
                            "input" => row["id"] == input,
                            "extent" => row["input"] == input,
                            "observation" => row["input"] == input && row["revision"] == number,
                            "blob" | "machine" => true,
                            _ => false,
                        }
                    }
                    _ => false,
                };
                if !is_owner {
                    return Err(invalid("selected batch contains a row outside its unit"));
                }
            }
            if owning != 1 {
                return Err(invalid("selected batch has no unique owning row"));
            }
            Ok(())
        }
        _ => Err(invalid(
            "selected payload missing, or a tombstone supplied bytes",
        )),
    }
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

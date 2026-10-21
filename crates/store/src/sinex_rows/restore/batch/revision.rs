//! Disk-backed reference checks for one selected revision's row batch.
use super::{RestoreUnit, Rows, invalid, revision_key};
use crate::Result;
use rusqlite::{Connection, params};
use serde_json::Value;

pub(super) struct Revision<'a> {
    input: &'a str,
    chain: i64,
    length: i64,
    scratch: &'a Connection,
}
impl<'a> Revision<'a> {
    pub(super) fn prepare(
        unit: &'a RestoreUnit,
        rows: &mut Rows,
        scratch: &'a Connection,
    ) -> Result<Self> {
        let (input, number) = revision_key(&unit.unit)?;
        scratch.execute("DELETE FROM blob_range", [])?;
        let mut owner = None;
        rows.each(|value| {
            let row = &value["row"];
            match value["table"].as_str() {
                Some("revision")
                    if row["input"] == input
                        && row["number"] == number
                        && row["commit_seq"] == unit.unit.commit_seq =>
                {
                    let chain = row["chain"]
                        .as_i64()
                        .filter(|n| *n > 0)
                        .ok_or_else(|| invalid("revision has no valid chain"))?;
                    let length = row["length"]
                        .as_i64()
                        .filter(|n| *n >= 0)
                        .ok_or_else(|| invalid("revision has no valid length"))?;
                    if owner.replace((chain, length)).is_some() {
                        return Err(invalid("selected batch has no unique owning row"));
                    }
                }
                Some("extent") if row["input"] == input => {
                    let hash = blob_hash(&row["blob"])?;
                    allow(scratch, "blob", &hash)?;
                    let offset = row["blob_offset"].as_i64().filter(|n| *n >= 0);
                    let length = row["length"].as_i64().filter(|n| *n > 0);
                    let needed = offset
                        .zip(length)
                        .and_then(|(offset, length)| offset.checked_add(length))
                        .ok_or_else(|| invalid("extent has an invalid source blob slice"))?;
                    scratch.execute(
                        "INSERT INTO blob_range VALUES (?1,?2)
                        ON CONFLICT(hash) DO UPDATE SET needed=MAX(needed,excluded.needed)",
                        params![hash, needed],
                    )?;
                }
                Some("observation") if row["input"] == input && row["revision"] == number => {
                    if let Some(machine) = row["machine"].as_str() {
                        allow(scratch, "machine", machine)?;
                    }
                }
                _ => {}
            }
            Ok(())
        })?;
        for name in &unit.blobs {
            let hash = super::super::super::excision::batch_hash(name)?;
            allow(scratch, "declared", &crate::sinex_rows::hex(&hash.0))?;
        }
        let (chain, length) =
            owner.ok_or_else(|| invalid("selected batch has no unique owning row"))?;
        Ok(Self {
            input,
            chain,
            length,
            scratch,
        })
    }

    pub(super) fn extent(&self, row: &Value) -> bool {
        row["input"] == self.input
            && row["chain"] == self.chain
            && row["offset"]
                .as_i64()
                .is_some_and(|n| n >= 0 && n < self.length)
    }

    pub(super) fn contains(&self, table: &str, row: &Value) -> Result<bool> {
        let key = match table {
            "machine" => match row["id"].as_str() {
                Some(id) => id.to_owned(),
                None => return Ok(false),
            },
            "blob" => {
                if row["place"] != "file"
                    || row["encoding"] != "plain"
                    || row["state"] != "present"
                    || row.get("base") != Some(&Value::Null)
                    || row.get("data") != Some(&Value::Null)
                    || !row["size"].as_i64().is_some_and(|n| n >= 0)
                    || row["stored_size"] != row["size"]
                {
                    return Err(invalid("restored source blob is not a plain present file"));
                }
                let hash = blob_hash(&row["hash"])?;
                let fits: bool = self.scratch.query_row(
                    "SELECT EXISTS(SELECT 1 FROM blob_range WHERE hash=?1 AND needed<=?2)",
                    params![hash, row["size"].as_i64().expect("checked size")],
                    |r| r.get(0),
                )?;
                if !fits {
                    return Err(invalid("extent slice exceeds its source blob"));
                }
                allow(self.scratch, "supplied", &hash)?;
                hash
            }
            _ => return Ok(false),
        };
        Ok(self.scratch.query_row(
            "SELECT EXISTS(SELECT 1 FROM allowed WHERE kind=?1 AND id=?2)",
            params![table, key],
            |r| r.get(0),
        )?)
    }

    pub(super) fn finish(&self) -> Result<()> {
        let mismatch:bool=self.scratch.query_row("SELECT EXISTS(
            SELECT id FROM allowed WHERE kind='blob' EXCEPT SELECT id FROM allowed WHERE kind='supplied')
            OR EXISTS(SELECT id FROM allowed WHERE kind='supplied' EXCEPT SELECT id FROM allowed WHERE kind='blob')
            OR EXISTS(SELECT id FROM allowed WHERE kind='blob' EXCEPT SELECT id FROM allowed WHERE kind='declared')
            OR EXISTS(SELECT id FROM allowed WHERE kind='declared' EXCEPT SELECT id FROM allowed WHERE kind='blob')",[],|r|r.get(0))?;
        if mismatch {
            return Err(invalid(
                "revision source references disagree with its selected descriptor or blob rows",
            ));
        }
        Ok(())
    }
}

fn allow(scratch: &Connection, kind: &str, id: &str) -> Result<()> {
    scratch.execute("INSERT OR IGNORE INTO allowed VALUES (?1,?2)", [kind, id])?;
    Ok(())
}
fn blob_hash(value: &Value) -> Result<String> {
    let hex = value["hex"]
        .as_str()
        .ok_or_else(|| invalid("source reference has no blob hash"))?;
    let bytes = crate::sinex_rows::unhex(hex)?;
    if bytes.len() != 32 {
        return Err(invalid("source reference has an invalid blob hash length"));
    }
    Ok(crate::sinex_rows::hex(&bytes))
}

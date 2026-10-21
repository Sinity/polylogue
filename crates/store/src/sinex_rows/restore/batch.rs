//! Verify one selected batch on disk, then validate and apply one row at a time.
use super::{ExcisionPublication, RestoreUnit, invalid, revision_key};
use crate::Result;
use rusqlite::Connection;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, Write};

pub(super) fn install<R: Read>(
    unit: &RestoreUnit,
    payload: Option<R>,
    target: Option<&Connection>,
    scratch: &Connection,
) -> Result<u64> {
    let mut rows = match (&unit.batch, payload) {
        (None, None) if unit.rows == 0 => return Ok(0),
        (Some(name), Some(reader)) => Rows::verified(unit, name, reader)?,
        _ => {
            return Err(invalid(
                "selected payload missing, or a tombstone supplied bytes",
            ));
        }
    };
    validate(unit, &mut rows, scratch)?;
    let mut inserted = 0;
    if let Some(target) = target {
        rows.each(|value| {
            inserted += super::super::apply_row(target, value)?;
            Ok(())
        })?;
    }
    Ok(inserted)
}

struct Rows(File);
impl Rows {
    fn verified(unit: &RestoreUnit, name: &str, mut reader: impl Read) -> Result<Self> {
        let hash = super::super::excision::batch_hash(name)?;
        let mut file = tempfile::tempfile()?;
        let mut actual = blake3::Hasher::new();
        let mut length = 0;
        let mut buffer = [0; 64 * 1024];
        loop {
            let n = match reader.read(&mut buffer) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => result?,
            };
            if n == 0 {
                break;
            }
            actual.update(&buffer[..n]);
            file.write_all(&buffer[..n])?;
            length += n as u64;
        }
        if actual.finalize().as_bytes() != &hash.0 || unit.bytes.is_some_and(|n| n != length) {
            return Err(invalid(
                "selected unit payload failed its hash/length check",
            ));
        }
        Ok(Self(file))
    }

    fn each(&mut self, mut visit: impl FnMut(serde_json::Value) -> Result<()>) -> Result<()> {
        self.0.rewind()?;
        let mut reader = BufReader::new(&mut self.0);
        let mut line = Vec::new();
        loop {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 {
                break;
            }
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if !line.is_empty() {
                visit(serde_json::from_slice(&line)?)?;
            }
        }
        Ok(())
    }
}

fn validate(unit: &RestoreUnit, rows: &mut Rows, scratch: &Connection) -> Result<()> {
    scratch.execute("DELETE FROM allowed", [])?;
    if unit.unit.kind == "excision" {
        rows.each(|value| {
            let row = &value["row"];
            if value["table"] == "observation" {
                if let Some(input) = row["input"].as_str() {
                    scratch
                        .execute("INSERT OR IGNORE INTO allowed VALUES ('input',?1)", [input])?;
                }
                if let Some(machine) = row["machine"].as_str() {
                    scratch.execute(
                        "INSERT OR IGNORE INTO allowed VALUES ('machine',?1)",
                        [machine],
                    )?;
                }
            }
            if value["table"] == "excision" {
                let refused: serde_json::Value = serde_json::from_str(
                    row["refused_json"]
                        .as_str()
                        .ok_or_else(|| invalid("excision has no metadata"))?,
                )?;
                if let Some(replacements) = refused["publication"]["replacements"].as_array() {
                    for r in replacements {
                        if r["unit"]["kind"] == "revision" {
                            let key = r["unit"]["key"]
                                .as_str()
                                .ok_or_else(|| invalid("revision replacement has no key"))?;
                            let (input, _) = key
                                .rsplit_once('/')
                                .ok_or_else(|| invalid("invalid revision replacement key"))?;
                            scratch.execute(
                                "INSERT OR IGNORE INTO allowed VALUES ('input',?1)",
                                [input],
                            )?;
                        }
                    }
                }
            }
            Ok(())
        })?;
    }
    let mut owning = 0;
    let mut count = 0;
    rows.each(|value| {
        count += 1;
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
                "observation" => row["class"] == "excised" && row["excision"] == unit.unit.key,
                "input" | "machine" => match row["id"].as_str() {
                    Some(id) => scratch.query_row(
                        "SELECT EXISTS(SELECT 1 FROM allowed WHERE kind=?1 AND id=?2)",
                        [table, id],
                        |r| r.get(0),
                    )?,
                    None => false,
                },
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
        Ok(())
    })?;
    if count != unit.rows {
        return Err(invalid("selected unit row count differs"));
    }
    if owning != 1 {
        return Err(invalid("selected batch has no unique owning row"));
    }
    Ok(())
}

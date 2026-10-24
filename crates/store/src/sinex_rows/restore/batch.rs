//! Verify one selected batch on disk, then validate and apply one row at a time.
use super::{ExcisionPublication, RestoreUnit, invalid, revision_key};
use crate::Result;
use rusqlite::Connection;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek};

mod extracted;
mod revision;

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
            let table = value["table"].as_str().unwrap_or("");
            let key = match table {
                "observation" if unit.unit.kind != "observation" => {
                    value["row"]["id"].as_str().map(str::to_owned)
                }
                "delivery_event" if unit.unit.kind != "delivery_event" => {
                    Some(serde_json::to_string(&(
                        value["row"]["delivery"].clone(),
                        value["row"]["what"].clone(),
                        value["row"]["at"].clone(),
                    ))?)
                }
                _ => None,
            };
            if let Some(key) = key {
                let superseded: bool = scratch.query_row(
                    "SELECT EXISTS(SELECT 1 FROM selected WHERE kind=?1 AND key=?2)",
                    [table, &key],
                    |r| r.get(0),
                )?;
                if superseded {
                    return Ok(());
                }
            }
            let machine = (table == "machine").then(|| (
                value["row"]["id"].as_str().unwrap().to_owned(),
                value["row"]["host"].as_str().unwrap().to_owned(),
            ));
            let added = super::super::apply_row(target, value)?;
            if let Some((id, host)) = machine {
                // The original three-column payload is unchanged. Its enclosing
                // verified unit proves when this alias was already present.
                target.execute(
                    "UPDATE machine SET first_commit_seq=CASE WHEN ?4 THEN ?3 ELSE MIN(first_commit_seq,?3) END WHERE id=?1 AND host=?2",
                    rusqlite::params![id,host,unit.unit.commit_seq,added>0],
                )?;
            }
            inserted += added;
            Ok(())
        })?;
    }
    Ok(inserted)
}

struct Rows(File);
impl Rows {
    fn verified(unit: &RestoreUnit, name: &str, reader: impl Read) -> Result<Self> {
        let hash = super::super::excision::batch_hash(name)?;
        let mut file = tempfile::tempfile()?;
        crate::blobs::copy_verified(
            reader,
            &mut file,
            polylogue_model::BlobHash(hash.0),
            unit.bytes,
        )?;
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
    if unit.unit.kind == "extracted_blob" {
        return extracted::validate(unit, rows, scratch);
    }
    scratch.execute("DELETE FROM allowed", [])?;
    let revision = if unit.unit.kind == "revision" {
        Some(revision::Revision::prepare(unit, rows, scratch)?)
    } else {
        None
    };
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
                if let Some(sessions) = refused.get("sessions") {
                    Vec::<polylogue_model::SessionId>::deserialize(sessions)?;
                } else if row["scope"]
                    .as_str()
                    .is_some_and(|scope| scope.starts_with("input:"))
                {
                    return Err(invalid("input excision has no retained session selection"));
                }
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
    if unit.unit.kind == "observation" {
        rows.each(|value| {
            let row = &value["row"];
            if value["table"] == "observation" && row["id"] == unit.unit.key {
                if !row["revision"].is_null() {
                    let number = row["revision"]
                        .as_i64()
                        .filter(|n| *n > 0)
                        .ok_or_else(|| invalid("observation has an invalid revision"))?;
                    let input = row["input"]
                        .as_str()
                        .ok_or_else(|| invalid("observation input missing"))?;
                    let key = format!("{input}/{number}");
                    // Historical observations survive excision. Its selected
                    // revision tombstone retains the referenced identity.
                    let retained: bool = scratch.query_row(
                        "SELECT EXISTS(SELECT 1 FROM selected WHERE kind='revision' AND key=?1)",
                        [key],
                        |r| r.get(0),
                    )?;
                    if !retained {
                        return Err(invalid("observation names no selected revision"));
                    }
                }
                for kind in ["input", "machine"] {
                    let id = row[kind]
                        .as_str()
                        .ok_or_else(|| invalid("observation dependency missing"))?;
                    scratch.execute("INSERT OR IGNORE INTO allowed VALUES (?1,?2)", [kind, id])?;
                }
            }
            Ok(())
        })?;
    }
    let mut observation_dependencies = [false; 2];
    let mut owning = 0;
    let mut count = 0;
    rows.each(|value| {
        count += 1;
        let table = value["table"]
            .as_str()
            .ok_or_else(|| invalid("row names no table"))?;
        let row = &value["row"];
        if table == "machine"
            && (row["id"].as_str().is_none()
                || row["host"].as_str().is_none()
                || row["first_seen"].as_i64().is_none()
                || row.as_object().is_none_or(|r| r.len() != 3))
        {
            return Err(invalid("machine dependency has an unsupported shape"));
        }
        let is_owner = match unit.unit.kind.as_str() {
            "delivery_event" => {
                owning += 1;
                table == "delivery_event"
                    && row["commit_seq"] == unit.unit.commit_seq
                    && serde_json::to_string(&(
                        row["delivery"].clone(),
                        row["what"].clone(),
                        row["at"].clone(),
                    ))? == unit.unit.key
            }
            "observation" => match table {
                "observation" => {
                    owning += 1;
                    row["id"] == unit.unit.key && row["commit_seq"] == unit.unit.commit_seq
                }
                "input" | "machine" => match row["id"].as_str() {
                    Some(id) => {
                        let allowed: bool = scratch.query_row(
                            "SELECT EXISTS(SELECT 1 FROM allowed WHERE kind=?1 AND id=?2)",
                            [table, id],
                            |r| r.get(0),
                        )?;
                        if allowed {
                            observation_dependencies[usize::from(table == "machine")] = true;
                        }
                        allowed
                    }
                    None => false,
                },
                _ => false,
            },
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
                "delivery_event" => {
                    row["delivery"] == unit.unit.key
                        && row
                            .get("commit_seq")
                            .is_none_or(|value| value.as_i64() == Some(0))
                }
                _ => false,
            },
            "setting" => {
                owning += 1;
                table == "setting"
                    && row["id"] == unit.unit.key
                    && row["commit_seq"] == unit.unit.commit_seq
            }
            "excision" => match table {
                "excision_document" => {
                    let id = row["document_id"]
                        .as_str()
                        .ok_or_else(|| invalid("document exclusion has no identity"))?;
                    polylogue_model::DocumentId::parse(id).map_err(|e| invalid(&e.to_string()))?;
                    row["excision_id"] == unit.unit.key
                }
                "vector_deletion_policy" => {
                    crate::excise::vector_policy::digest(
                        row["version"]
                            .as_i64()
                            .ok_or_else(|| invalid("policy version missing"))?,
                        row["hashes_json"]
                            .as_str()
                            .ok_or_else(|| invalid("policy hashes missing"))?,
                    )?;
                    row["excision_id"] == unit.unit.key
                }
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
                    row["class"] == "excised"
                        && row["excision"] == unit.unit.key
                        && row
                            .get("commit_seq")
                            .is_none_or(|value| value.as_i64() == Some(0))
                }
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
                    "extent" => revision.as_ref().expect("revision prepared").extent(row),
                    "observation" => {
                        row["input"] == input
                            && row["revision"] == number
                            && row
                                .get("commit_seq")
                                .is_none_or(|value| value.as_i64() == Some(0))
                    }
                    "blob" | "machine" => revision
                        .as_ref()
                        .expect("revision prepared")
                        .contains(table, row)?,
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
    if unit.unit.kind == "observation" && observation_dependencies.iter().any(|present| !present) {
        return Err(invalid("observation batch omits its input or machine"));
    }
    if let Some(revision) = revision {
        revision.finish()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;

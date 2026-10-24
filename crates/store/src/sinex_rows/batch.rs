//! File-backed JSONL batches of one archive unit's rows.
use super::{Unit, hex};
use rusqlite::OptionalExtension;
use rusqlite::{
    Connection, params_from_iter,
    types::{Value as Sql, ValueRef},
};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::{BufWriter, Seek, Write};

/// A unit's rows as JSON lines, and the input blobs they cite.
pub struct Batch {
    pub file: std::fs::File,
    pub bytes: u64,
    pub rows: u64,
    pub blobs: Vec<[u8; 32]>,
}

#[cfg(test)]
impl Batch {
    pub(super) fn contents(&self) -> Vec<u8> {
        use std::os::unix::fs::FileExt;
        let mut bytes = vec![0; usize::try_from(self.bytes).unwrap()];
        self.file.read_exact_at(&mut bytes, 0).unwrap();
        bytes
    }
}

// Keep bytes independent of serde_json's optional insertion-order feature.
#[derive(Serialize)]
struct BatchRow<'a> {
    row: BTreeMap<&'a str, Value>,
    table: &'a str,
}

/// Reads a unit's rows. Each query names its table, so a row is restored
/// into the table it came from.
pub fn read(archive: &Connection, unit: &Unit) -> crate::Result<Batch> {
    let mut out = Batch {
        file: tempfile::tempfile()?,
        bytes: 0,
        rows: 0,
        blobs: Vec::new(),
    };
    let key = [Sql::Text(unit.key.clone())];
    match unit.kind.as_str() {
        "revision" => {
            let (input, number) = unit.key.rsplit_once('/').unwrap_or((&unit.key, "0"));
            let number: i64 = number.parse().unwrap_or(0);
            let args = [Sql::Text(input.to_owned()), Sql::Integer(number)];
            let exists: bool = archive.query_row(
                "SELECT EXISTS(SELECT 1 FROM revision WHERE input = ?1 AND number = ?2)",
                params_from_iter(args.iter()),
                |r| r.get(0),
            )?;
            if !exists {
                return Ok(out);
            }
            let (chain, length): (i64, i64) = archive.query_row(
                "SELECT chain,length FROM revision WHERE input=?1 AND number=?2",
                params_from_iter(args.iter()),
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            crate::retention::require_range(archive, "", input, chain, 0..length as u64)?;

            add(
                archive,
                &mut out,
                "input",
                "SELECT * FROM input WHERE id = ?1",
                &args[..1],
            )?;
            add(
                archive,
                &mut out,
                "revision",
                "SELECT * FROM revision WHERE input = ?1 AND number = ?2",
                &args,
            )?;
            // The extents this revision added to its chain: the ones from the
            // end of the chain's previous revision to its own end.
            let extents = "SELECT e.* FROM extent e JOIN revision r
                             ON r.input = e.input AND r.number = ?2 AND e.chain = r.chain
                           WHERE e.input = ?1 AND e.offset < r.length
                             AND e.offset >= COALESCE((SELECT MAX(p.length) FROM revision p
                                 WHERE p.input = r.input AND p.chain = r.chain AND p.number < r.number), 0)";
            // In Sinex every blob is a plain file in its content store, so
            // the restored row says so: no inline bytes, no delta base.
            let blobs_of_extents = format!(
                "SELECT b.hash, b.size, b.size AS stored_size, 'file' AS place,
                        'plain' AS encoding, NULL AS base, 'present' AS state, NULL AS data
                 FROM blob b WHERE b.hash IN (SELECT blob FROM ({extents}))"
            );
            add(archive, &mut out, "blob", &blobs_of_extents, &args)?;
            add(archive, &mut out, "extent", extents, &args)?;
            add(
                archive,
                &mut out,
                "machine",
                "SELECT m.* FROM machine m WHERE m.id IN
                   (SELECT machine FROM observation WHERE input = ?1 AND revision = ?2 AND commit_seq=0)",
                &args,
            )?;
            add(
                archive,
                &mut out,
                "observation",
                "SELECT * FROM observation WHERE input = ?1 AND revision = ?2 AND commit_seq=0",
                &args,
            )?;
        }
        "extracted_blob" => {
            use crate::retention::extracted::{Key, MATCH};
            let source = Key::parse(&unit.key)?;
            let args = source.args();
            let state:Option<(String,String)>=archive.query_row(&format!("SELECT x.state,b.state FROM extracted_blob x JOIN blob b ON b.hash=x.blob WHERE {}",MATCH.split(" AND ").map(|c|format!("x.{c}")).collect::<Vec<_>>().join(" AND ")),params_from_iter(args.iter()),|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
            let Some((association, blob)) = state else {
                return Ok(out);
            };
            if association != "present" || blob != "present" {
                return Err(crate::StoreError::InputUnfinished {
                    input: source.input.clone(),
                    reason: "decoded source association is unfinished".into(),
                });
            }
            crate::retention::require_range(
                archive,
                "",
                &source.input,
                source.chain,
                source.start as u64..source.end as u64,
            )?;
            add(
                archive,
                &mut out,
                "extracted_blob",
                &format!("SELECT * FROM extracted_blob WHERE {MATCH}"),
                &args,
            )?;
            add(
                archive,
                &mut out,
                "blob",
                "SELECT hash,size,size AS stored_size,'file' AS place,'plain' AS encoding,NULL AS base,'present' AS state,NULL AS data FROM blob WHERE hash=?1",
                &args[6..],
            )?;
        }
        "observation" => {
            let exists: bool = archive.query_row(
                "SELECT EXISTS(SELECT 1 FROM observation WHERE id=?1)",
                [&unit.key],
                |r| r.get(0),
            )?;
            if !exists {
                return Ok(out);
            }
            let revision:Option<(String,i64,i64)>=archive.query_row("SELECT r.input,r.chain,r.length FROM observation o JOIN revision r ON r.input=o.input AND r.number=o.revision WHERE o.id=?1",[&unit.key],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?))).optional()?;
            if let Some((input, chain, length)) = revision {
                crate::retention::require_range(archive, "", &input, chain, 0..length as u64)?;
            }
            add(
                archive,
                &mut out,
                "input",
                "SELECT * FROM input WHERE id IN (SELECT input FROM observation WHERE id=?1)",
                &key,
            )?;
            add(
                archive,
                &mut out,
                "machine",
                "SELECT * FROM machine WHERE id IN (SELECT machine FROM observation WHERE id=?1)",
                &key,
            )?;
            add(
                archive,
                &mut out,
                "observation",
                "SELECT * FROM observation WHERE id=?1",
                &key,
            )?;
        }
        "delivery_event" => {
            let (delivery, what, at) = super::delivery_event_key(unit)?;
            add(
                archive,
                &mut out,
                "delivery_event",
                "SELECT * FROM delivery_event WHERE delivery=?1 AND what=?2 AND at=?3",
                &[Sql::Text(delivery), Sql::Text(what), Sql::Integer(at)],
            )?;
        }
        "note" => {
            add(
                archive,
                &mut out,
                "note",
                "SELECT * FROM note WHERE id = ?1",
                &key,
            )?;
            add(
                archive,
                &mut out,
                "note_target",
                "SELECT * FROM note_target WHERE note_id = ?1",
                &key,
            )?;
            add(
                archive,
                &mut out,
                "note_relation",
                "SELECT * FROM note_relation WHERE note_id = ?1",
                &key,
            )?;
        }
        "excision" => {
            // Deleted revision units still name an input. Keep that descriptor
            // with the excision rather than reopening a retired original batch.
            add(archive,&mut out,"input",
                "SELECT * FROM input WHERE id IN (
                    SELECT input FROM observation WHERE class='excised' AND excision=?1 AND commit_seq=0
                    UNION SELECT i.id FROM input i,excision x,json_each(x.refused_json,'$.publication.replacements') j
                    WHERE x.id=?1 AND json_extract(j.value,'$.unit.kind')='revision'
                      AND substr(json_extract(j.value,'$.unit.key'),1,length(i.id)+1)=i.id||'/'
                      AND instr(substr(json_extract(j.value,'$.unit.key'),length(i.id)+2),'/')=0)",&key)?;
            add(
                archive,
                &mut out,
                "machine",
                "SELECT * FROM machine WHERE id IN (SELECT machine FROM observation WHERE class='excised' AND excision=?1 AND commit_seq=0)",
                &key,
            )?;
            add(
                archive,
                &mut out,
                "excision",
                "SELECT * FROM excision WHERE id = ?1",
                &key,
            )?;
            let has_documents:bool=archive.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='excision_document')",[],|r|r.get(0))?;
            if has_documents {
                add(
                    archive,
                    &mut out,
                    "excision_document",
                    "SELECT * FROM excision_document WHERE excision_id=?1 ORDER BY document_id",
                    &key,
                )?;
            }
            let has_policy:bool=archive.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='vector_deletion_policy')",[],|r|r.get(0))?;
            if has_policy {
                add(
                    archive,
                    &mut out,
                    "vector_deletion_policy",
                    "SELECT * FROM vector_deletion_policy WHERE excision_id=?1",
                    &key,
                )?;
            }
            add(
                archive,
                &mut out,
                "observation",
                "SELECT * FROM observation WHERE class = 'excised' AND excision = ?1 AND commit_seq=0 ORDER BY rowid",
                &key,
            )?;
        }
        "setting" => add(
            archive,
            &mut out,
            "setting",
            "SELECT * FROM setting WHERE id = ?1",
            &key,
        )?,
        "delivery" => {
            add(
                archive,
                &mut out,
                "delivery",
                "SELECT * FROM delivery WHERE id = ?1",
                &key,
            )?;
            add(
                archive,
                &mut out,
                "delivery_event",
                "SELECT * FROM delivery_event WHERE delivery = ?1 AND commit_seq=0",
                &key,
            )?;
        }
        _ => {
            return Err(crate::StoreError::Invalid(
                "unknown publication unit kind".into(),
            ));
        }
    }
    out.bytes = out.file.stream_position()?;
    out.file.rewind()?;
    Ok(out)
}

fn add(
    archive: &Connection,
    out: &mut Batch,
    table: &str,
    sql: &str,
    args: &[Sql],
) -> crate::Result<()> {
    let mut stmt = archive.prepare(sql)?;
    let names: Vec<String> = stmt
        .column_names()
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    let mut rows = stmt.query(params_from_iter(args.iter()))?;
    let mut writer = BufWriter::new(&mut out.file);
    while let Some(r) = rows.next()? {
        let mut row = BTreeMap::new();
        for (i, name) in names.iter().enumerate() {
            let value = match r.get_ref(i)? {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(n) => json!(n),
                ValueRef::Real(x) => json!(x),
                ValueRef::Text(t) => Value::String(String::from_utf8_lossy(t).into_owned()),
                ValueRef::Blob(b) => {
                    if table == "blob"
                        && name == "hash"
                        && let Ok(h) = <[u8; 32]>::try_from(b)
                    {
                        out.blobs.push(h);
                    }
                    json!({ "hex": hex(b) })
                }
            };
            row.insert(name.as_str(), value);
        }
        let line = BatchRow { row, table };
        serde_json::to_writer(&mut writer, &line).map_err(|error| match error.io_error_kind() {
            Some(kind) => crate::StoreError::Io(std::io::Error::new(kind, error)),
            None => crate::StoreError::Json(error),
        })?;
        writer.write_all(b"\n")?;
        out.rows += 1;
    }
    writer.flush()?;
    Ok(())
}

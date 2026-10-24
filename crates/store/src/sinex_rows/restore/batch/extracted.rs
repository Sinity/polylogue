//! A decoded source association declares exactly one verified decoded blob.
use super::{RestoreUnit, Rows, invalid};
use crate::{Result, retention::extracted::Key};
use rusqlite::{Connection, params};

pub(super) fn validate(unit: &RestoreUnit, rows: &mut Rows, scratch: &Connection) -> Result<()> {
    let key = Key::parse(&unit.unit.key)?;
    let name = format!("blake3:{}", key.blob);
    if unit.blobs != [name] || unit.rows != 2 {
        return Err(invalid("decoded association must declare exactly its blob"));
    }
    let mut association = false;
    let mut blob = false;
    let mut count = 0;
    rows.each(|value| {
        count += 1;
        let row = &value["row"];
        match value["table"].as_str() {
            Some("extracted_blob") if !association => {
                if row["input"] != key.input
                    || row["chain"] != key.chain
                    || row["source_start"] != key.start
                    || row["source_end"] != key.end
                    || row["source_hash"]["hex"] != key.source.to_string()
                    || row["parser_version"] != key.parser
                    || row["blob"]["hex"] != key.blob.to_string()
                    || row["state"] != "present"
                    || row["commit_seq"] != unit.unit.commit_seq
                {
                    return Err(invalid(
                        "decoded association differs from its selected source unit",
                    ));
                }
                association = true;
            }
            Some("blob") if !blob => {
                let size = row["size"]
                    .as_i64()
                    .filter(|n| *n >= 0)
                    .ok_or_else(|| invalid("decoded blob has no valid size"))?;
                if row["hash"]["hex"] != key.blob.to_string()
                    || row["place"] != "file"
                    || row["encoding"] != "plain"
                    || row["state"] != "present"
                    || !row["base"].is_null()
                    || !row["data"].is_null()
                    || row["stored_size"] != size
                {
                    return Err(invalid("decoded source blob is not a plain present file"));
                }
                let verified: bool = scratch.query_row(
                    "SELECT EXISTS(SELECT 1 FROM verified_source WHERE hash=?1 AND bytes=?2)",
                    params![key.blob.to_string(), size],
                    |r| r.get(0),
                )?;
                if !verified {
                    return Err(invalid("decoded catalog size differs from verified bytes"));
                }
                blob = true;
            }
            _ => return Err(invalid("row is outside its decoded source unit")),
        }
        Ok(())
    })?;
    if !association || !blob || count != 2 {
        return Err(invalid("decoded association omits its owner or blob"));
    }
    Ok(())
}

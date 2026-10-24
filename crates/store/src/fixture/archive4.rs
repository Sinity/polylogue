//! Historical DDL fixtures, separate from witnesses produced by old binaries.
//! The four SQL files are verbatim from f3fd29a19347faa0f0c2448f526cca8e46e0720e,
//! under `crates/store/src/schema` (base) and `schema/migrations` (002–004).
//! That revision is the final Archive4 before 40ea89ac1c introduced 005.

use crate::{Archive, Result, archive::cache, archive::open_write};
use rusqlite::{Connection, params};
use std::path::Path;

/// Creates only the historical canonical database. Opening it through
/// `Archive::open_or_create` later performs the normal upgrade and index setup.
pub fn create(root: &Path) -> Result<Archive> {
    std::fs::create_dir_all(root)?;
    let archive = Archive::at(root);
    let db = open_write(&archive.archive_db(), cache::SETUP)?;
    for (i, sql) in [
        include_str!("archive4/archive.sql"),
        include_str!("archive4/002-excision-revision.sql"),
        include_str!("archive4/003-vector-excision.sql"),
        include_str!("archive4/004-local-blob-excision.sql"),
    ]
    .iter()
    .enumerate()
    {
        db.execute_batch(sql)?;
        db.pragma_update(None, "user_version", (i + 1) as u32)?;
    }
    db.execute(
        "INSERT INTO meta(key,value) VALUES ('archive_id',?1)",
        [uuid::Uuid::now_v7().to_string()],
    )?;
    Ok(archive)
}

/// Seeds authored data using the old columns and returns its historical closure.
/// The body uses the same dictionary-free zstd representation as old writers.
pub fn delivery(db: &Connection) -> Result<Vec<u8>> {
    let body = b"Synthetic legacy body";
    let packed = crate::encoding::pack_bytes(body)?;
    db.execute("INSERT INTO delivery(id,recipient_kind,sender,event,at,content_hash,bytes,body,refs_json,left_out_json,commit_seq)
        VALUES ('legacy:[opaque]','operator','cli','advisory',1,?1,?2,?3,'[]','[]',1)",
        params![blake3::hash(body).as_bytes(),body.len() as i64,packed])?;
    db.execute("INSERT INTO delivery_event(delivery,what,at,detail) VALUES ('legacy:[opaque]','failed',2,'legacy private evidence')", [])?;
    db.execute(
        "INSERT INTO meta(key,value) VALUES ('write_counter','2')",
        [],
    )?;
    let mut bytes = Vec::new();
    for table in ["delivery", "delivery_event"] {
        let mut query = db.prepare(&format!("SELECT * FROM {table}"))?;
        let columns = query
            .column_names()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let mut rows = query.query([])?;
        while let Some(row) = rows.next()? {
            let mut object = serde_json::Map::new();
            for (i, name) in columns.iter().enumerate() {
                use rusqlite::types::ValueRef;
                let value = match row.get_ref(i)? {
                    ValueRef::Null => serde_json::Value::Null,
                    ValueRef::Integer(n) => n.into(),
                    ValueRef::Real(n) => n.into(),
                    ValueRef::Text(_) => row.get::<_, String>(i)?.into(),
                    ValueRef::Blob(b) => serde_json::json!({"hex": crate::sinex_rows::hex(b)}),
                };
                object.insert(name.clone(), value);
            }
            serde_json::to_writer(
                &mut bytes,
                &serde_json::json!({"table": table, "row": object}),
            )?;
            bytes.push(b'\n');
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests;

//! The one Index1 transition: split immutable payloads without replaying inputs.
use super::*;
use crate::encoding::{self, RowPacker};
use polylogue_model::Block;
use rusqlite::params;

const BLOCKS: &str = include_str!("blocks.sql");
const BEFORE: &str = include_str!("blocks_before_shared.sql");
const VALUES: &str = include_str!("block_values.sql");
const LINEAGE: &str = include_str!("lineage.sql");

/// Builds the two exact predecessor shapes: before and after lineage shipped.
/// This is a schema comparison, never an old-format reader used after conversion.
fn predecessor(primary: bool, lineage: bool) -> Result<Connection> {
    let expected = Connection::open_in_memory()?;
    crate::sql_functions::register(&expected)?;
    let mut index = INDEX.replacen(BLOCKS, BEFORE, 1).replacen(VALUES, "", 1);
    if !lineage {
        index = index.replacen(LINEAGE, "", 1).replacen(
            "  lineage           INTEGER NOT NULL DEFAULT 0,\n",
            "",
            1,
        );
    }
    expected.execute_batch(&index)?;
    expected.execute_batch(crate::coverage::SCHEMA)?;
    expected.execute_batch(crate::variants::SCHEMA)?;
    expected.execute_batch(crate::parser_context::SCHEMA)?;
    expected.execute_batch(crate::parser_context::VERSION_INDEX)?;
    expected.execute_batch(OPEN_INDEXES)?;
    if primary {
        add_primary(&expected)?;
    }
    Ok(expected)
}

fn matches(conn: &Connection, expected: &Connection) -> Result<bool> {
    let found = shape(conn)?;
    let wanted = shape(expected)?;
    let building = building(conn)?;
    let optional: HashSet<_> = OPEN_INDEXES
        .split(';')
        .filter_map(|statement| {
            let line = statement
                .lines()
                .filter(|line| !line.trim_start().starts_with("--"))
                .collect::<Vec<_>>()
                .join(" ");
            line.split_whitespace().nth(5).map(str::to_owned)
        })
        .collect();
    Ok(found.iter().all(|item| wanted.contains(item))
        && wanted.iter().all(|item| {
            found.contains(item)
                || (item.0 == "index"
                    && (optional.contains(&item.1)
                        || building
                            && crate::writer::deferred_index_names().any(|name| name == item.1)))
        }))
}

pub(super) fn convert(conn: &Connection, primary: bool) -> Result<()> {
    let lineage: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE name='session_lineage' AND type='table')",
        [],
        |row| row.get(0),
    )?;
    if !matches(conn, &predecessor(primary, lineage)?)? {
        return Err(StoreError::Invalid(
            "Index1 does not match the shared-block conversion's declared predecessor; the index is unchanged".into(),
        ));
    }
    conn.execute_batch(VALUES)?;
    let mut packer = RowPacker::new()?;
    let mut after = 0i64;
    loop {
        // One packed reading bounds memory; no read statement stays active over a write.
        let row = conn.prepare_cached(
            "SELECT rowid,version_hash,body FROM message_version WHERE rowid>?1 ORDER BY rowid LIMIT 1",
        )?.query_row([after], |row| Ok((row.get::<_,i64>(0)?,row.get::<_,Vec<u8>>(1)?,row.get::<_,Vec<u8>>(2)?))).optional()?;
        let Some((pk, hash, body)) = row else { break };
        let mut header: serde_json::Value = encoding::unpack_json(&body)?;
        let blocks = header
            .get_mut("blocks")
            .and_then(serde_json::Value::as_array_mut)
            .ok_or_else(|| {
                StoreError::Invalid("a stored message version has no block array".into())
            })?;
        for (idx, block) in std::mem::take(blocks).into_iter().enumerate() {
            let block: Block = serde_json::from_value(block)?;
            let (body, data) = packer.pack_retained_block_payload(&block)?;
            let value = crate::versions::store_block_payload(conn, &body, data.as_deref())?;
            conn.prepare_cached(
                "INSERT INTO message_version_block(version_hash,idx,value_pk) VALUES (?1,?2,?3)",
            )?
            .execute(params![hash, idx as i64, value])?;
        }
        let fields = packer.pack_json(&header)?;
        conn.prepare_cached("UPDATE message_version SET body=?2 WHERE rowid=?1")?
            .execute(params![pk, fields])?;
        after = pk;
    }
    conn.execute_batch(
        "ALTER TABLE block ADD COLUMN value_pk INTEGER REFERENCES block_value(pk) DEFERRABLE INITIALLY DEFERRED;",
    )?;
    after = 0;
    loop {
        let row = conn
            .prepare_cached(
                "SELECT pk,body,data,signature FROM block WHERE pk>?1 ORDER BY pk LIMIT 1",
            )?
            .query_row([after], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Option<Vec<u8>>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            })
            .optional()?;
        let Some((pk, body, data, signature)) = row else {
            break;
        };
        let block = encoding::unpack_block(&body)?;
        let retained_signature = match &block {
            Block::Thinking { signature, .. } => signature.as_deref(),
            _ => None,
        };
        if signature.as_deref() != retained_signature {
            return Err(StoreError::Invalid(
                "a block's duplicate signature differs from its typed payload".into(),
            ));
        }
        let body = packer.pack_json(&block)?;
        let data = data
            .map(|bytes| packer.pack_json(&encoding::unpack_json::<serde_json::Value>(&bytes)?))
            .transpose()?;
        let value = crate::versions::store_block_payload(conn, &body, data.as_deref())?;
        conn.prepare_cached("UPDATE block SET value_pk=?2 WHERE pk=?1")?
            .execute(params![pk, value])?;
        after = pk;
    }
    conn.execute_batch(
        "ALTER TABLE block ALTER COLUMN value_pk SET NOT NULL;
         ALTER TABLE block DROP COLUMN body;
         ALTER TABLE block DROP COLUMN data;
         ALTER TABLE block DROP COLUMN signature;
         CREATE INDEX block_value_ref ON block(value_pk);",
    )?;
    if !lineage {
        add_lineage(conn)?;
    }
    crate::fault::check(
        "index.shared_blocks.converted",
        conn.path().unwrap_or("memory"),
    )?;
    Ok(())
}

fn add_lineage(conn: &Connection) -> Result<()> {
    // derive_due has no inbound references. Its existing key and flags survive;
    // rebuilding only this small queue places the new column in the current format.
    let expected = predecessor(false, true)?;
    let ddl: String = expected.query_row(
        "SELECT sql FROM sqlite_schema WHERE type='table' AND name='derive_due'",
        [],
        |row| row.get(0),
    )?;
    let triggers = conn
        .prepare(
            "SELECT name,sql FROM sqlite_schema WHERE type='trigger' AND instr(sql,'derive_due')>0",
        )?
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for (name, _) in &triggers {
        conn.execute_batch(&format!("DROP TRIGGER \"{}\"", name.replace('"', "\"\"")))?;
    }
    conn.execute_batch("ALTER TABLE derive_due RENAME TO derive_due_before_shared")?;
    conn.execute_batch(&ddl)?;
    conn.execute_batch(
        "INSERT INTO derive_due(session_pk,rows_changed,repo_lookup,rewritten,marked)
         SELECT session_pk,rows_changed,repo_lookup,rewritten,marked FROM derive_due_before_shared;
         DROP TABLE derive_due_before_shared;",
    )?;
    for (_, sql) in triggers {
        conn.execute_batch(&sql)?;
    }
    conn.execute_batch(LINEAGE)?;
    conn.execute_batch(
        "INSERT INTO derive_due(session_pk,rows_changed,lineage) SELECT pk,0,1 FROM session WHERE true
         ON CONFLICT(session_pk) DO UPDATE SET lineage=1,marked=marked+1;",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;

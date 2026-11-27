use super::*;
use crate::ConnExt as _;
use crate::OptionExt as _;
use polylogue_model::JsonExt;

/// The row-dump records whose comparison hashes commit with this step.
#[derive(
    Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct RowComparison {
    rows: Vec<(String, String, Option<Vec<u8>>)>,
    schema: Option<Value>,
    pub from: u64,
    pub length: u64,
}
impl RowComparison {
    pub fn records(bytes: &[u8]) -> crate::Result<Self> {
        let mut out = Self::default();
        for line in bytes.split(|b| *b == b'\n').filter(|line| !line.is_empty()) {
            let value: Value =
                serde_json::from_slice(line).map_err(crate::StoreError::stored_data)?;
            if let Some(schema) = value.get("schema") {
                out.schema =
                    Some(json!({"schema":schema,"table_metadata":value.get("table_metadata")}));
                continue;
            }
            let table = value.str("t").or_stored("row dump record has no table")?;
            let key = value
                .get("k")
                .or_stored("row dump record has no key")?
                .to_string();
            let hash = if value.get("gone") == Some(&Value::Bool(true)) {
                None
            } else {
                let row = value.get("r").filter(|r| r.is_object()).ok_or_else(|| {
                    crate::StoreError::stored_data("row dump record has no values")
                })?;
                Some(row_hash(row).as_bytes().to_vec())
            };
            out.rows.push((table.into(), key, hash));
        }
        Ok(out)
    }
    pub fn read(mut reader: impl std::io::BufRead) -> crate::Result<Self> {
        let mut out = Self::default();
        let mut line = Vec::new();
        while reader.read_until(b'\n', &mut line)? > 0 {
            let part = Self::records(&line)?;
            out.rows.extend(part.rows);
            if part.schema.is_some() {
                out.schema = part.schema;
            }
            line.clear();
        }
        Ok(out)
    }
    pub(crate) fn write(&self, conn: &Connection, input_pk: i64) -> crate::Result<()> {
        let length = i64::try_from(self.length).map_err(|_| {
            crate::StoreError::internal("database comparison exceeds archive address space")
        })?;
        let previous: Option<i64> = conn.scalar_optional(
            "SELECT length FROM tool_database WHERE input_pk=?1",
            [input_pk],
        )?;
        let previous = previous
            .map(u64::try_from)
            .transpose()
            .map_err(|_| crate::StoreError::stored_data("negative database comparison length"))?;
        // Domain parsing may finish after newer row-dump bytes. Its older
        // values must not move the comparison state backwards.
        if previous.is_some_and(|previous| previous >= self.length) {
            return Ok(());
        }
        // Later queued groups can outlive an earlier failed transaction.
        // Only a contiguous prefix can become the recovery position.
        if self.from > previous.unwrap_or(0) {
            return Err(crate::StoreError::internal(format!(
                "database comparison starts at {} after acknowledged {}",
                self.from,
                previous.unwrap_or(0)
            )));
        }
        for (table, key, hash) in &self.rows {
            if let Some(hash) = hash {
                conn.prepare_cached("INSERT INTO tool_row(input_pk,table_name,key,hash) VALUES(?1,?2,?3,?4) ON CONFLICT(input_pk,table_name,key) DO UPDATE SET hash=excluded.hash")?.execute(params![input_pk,table,key,hash])?;
            } else {
                conn.prepare_cached(
                    "DELETE FROM tool_row WHERE input_pk=?1 AND table_name=?2 AND key=?3",
                )?
                .execute(params![input_pk, table, key])?;
            }
        }
        if let Some((table, _, _)) = self.rows.first() {
            crate::fault::check("tool_database.comparison_commit", table)?;
        }
        let schema = self
            .schema
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        conn.execute("INSERT INTO tool_database(input_pk,length,schema) VALUES(?1,?2,COALESCE(?3,'{}')) ON CONFLICT(input_pk) DO UPDATE SET length=excluded.length,schema=COALESCE(?3,tool_database.schema)", params![input_pk,length,schema])?;
        Ok(())
    }
}

/// The comparison state recorded for one input, copied out of the index
/// in one read so no read stays open while the rows are compared and sent:
/// during a build the index file has a rollback journal, where an open read
/// blocks every commit of the writer that the sends wait for.
pub(super) struct Catalog {
    conn: Connection,
    pub schema: Option<Value>,
}
impl Catalog {
    pub fn open(archive: &Archive, input: &str) -> crate::Result<Self> {
        let conn = archive.open_index_connection(
            &archive.index_db(),
            crate::archive::index_build::Reads::Pipeline,
        )?;
        conn.pragma_update(None, "temp_store", "file")?;
        conn.pragma_update(Some("temp"), "cache_size", -1024)?;
        conn.execute_batch(
            "CREATE TEMP TABLE seen(table_name TEXT,key TEXT,PRIMARY KEY(table_name,key)) WITHOUT ROWID;
             CREATE TEMP TABLE known(table_name TEXT,key TEXT,hash BLOB,PRIMARY KEY(table_name,key)) WITHOUT ROWID;
             BEGIN;",
        )?;
        let schema = recorded_schema(&conn, input)?;
        conn.execute(
            "INSERT INTO temp.known(table_name,key,hash) SELECT r.table_name,r.key,r.hash FROM tool_row r JOIN input_ref i ON i.pk=r.input_pk WHERE i.id=?1",
            [input],
        )?;
        conn.execute_batch("COMMIT")?;
        Ok(Self { conn, schema })
    }
    pub fn changed(&self, table: &str, key: &Value, row: &Value) -> crate::Result<bool> {
        let key = key.to_string();
        self.conn
            .prepare_cached("INSERT INTO temp.seen VALUES(?1,?2)")?
            .execute(params![table, key])?;
        let old: Option<Vec<u8>> = self
            .conn
            .prepare_cached("SELECT hash FROM temp.known WHERE table_name=?1 AND key=?2")?
            .query_row(params![table, key], |r| r.get::<_, Option<Vec<u8>>>(0))
            .optional()?
            .flatten();
        Ok(old.as_deref() != Some(row_hash(row).as_bytes()))
    }
    pub fn gone(
        &self,
        table: &str,
        visit: &mut dyn FnMut(Value) -> Result<(), DatabaseError>,
    ) -> Result<(), DatabaseError> {
        let mut stmt = self.conn.prepare("SELECT key FROM temp.known k WHERE table_name=?1 AND NOT EXISTS(SELECT 1 FROM temp.seen s WHERE s.table_name=k.table_name AND s.key=k.key) ORDER BY key").map_err(crate::StoreError::from)?;
        let mut rows = stmt.query([table]).map_err(crate::StoreError::from)?;
        while let Some(row) = rows.next().map_err(crate::StoreError::from)? {
            let key: String = row.get(0).map_err(crate::StoreError::from)?;
            visit(serde_json::from_str(&key).map_err(std::io::Error::from)?)?;
        }
        Ok(())
    }
    pub fn dropped(
        &self,
        tables: &[&str],
        visit: &mut dyn FnMut(&str, Value) -> Result<(), DatabaseError>,
    ) -> Result<(), DatabaseError> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT table_name FROM temp.known ORDER BY table_name")
            .map_err(crate::StoreError::from)?;
        let names = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(crate::StoreError::from)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(crate::StoreError::from)?;
        for table in names.iter().filter(|t| !tables.contains(&t.as_str())) {
            self.gone(table, &mut |key| visit(table, key))?;
        }
        Ok(())
    }
}

/// The header the archive's comparison state holds for `input`, if any.
pub(super) fn recorded_schema(index: &Connection, input: &str) -> crate::Result<Option<Value>> {
    let schema: Option<String> = index.scalar_optional(
        "SELECT t.schema FROM tool_database t JOIN input_ref i ON i.pk=t.input_pk WHERE i.id=?1",
        [input],
    )?;
    Ok(schema.map(|s| serde_json::from_str(&s)).transpose()?)
}

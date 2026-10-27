use super::*;

/// The row-dump records whose comparison hashes commit with this step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Comparison {
    rows: Vec<(String, String, Option<Vec<u8>>)>,
    schema: Option<Value>,
    pub from: u64,
    pub length: u64,
}
impl Comparison {
    pub fn records(bytes: &[u8]) -> crate::Result<Self> {
        let mut out = Self::default();
        for line in bytes.split(|b| *b == b'\n').filter(|line| !line.is_empty()) {
            let value: Value = serde_json::from_slice(line)
                .map_err(|e| crate::StoreError::Invalid(e.to_string()))?;
            if let Some(schema) = value.get("schema") {
                out.schema = Some(schema.clone());
                continue;
            }
            let table = value
                .get("t")
                .and_then(Value::as_str)
                .ok_or_else(|| crate::StoreError::Invalid("row dump record has no table".into()))?;
            let key = value
                .get("k")
                .ok_or_else(|| crate::StoreError::Invalid("row dump record has no key".into()))?
                .to_string();
            let hash = if value.get("gone") == Some(&Value::Bool(true)) {
                None
            } else {
                let row = value.get("r").filter(|r| r.is_object()).ok_or_else(|| {
                    crate::StoreError::Invalid("row dump record has no values".into())
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
            crate::StoreError::Invalid("database comparison exceeds archive address space".into())
        })?;
        let previous: Option<i64> = conn
            .query_row(
                "SELECT length FROM tool_database WHERE input_pk=?1",
                [input_pk],
                |r| r.get(0),
            )
            .optional()?;
        let previous = previous.map(u64::try_from).transpose().map_err(|_| {
            crate::StoreError::Invalid("negative database comparison length".into())
        })?;
        // Domain parsing may finish after newer row-dump bytes. Its older
        // values must not move the comparison state backwards.
        if previous.is_some_and(|previous| previous >= self.length) {
            return Ok(());
        }
        // Later queued groups can outlive an earlier failed transaction.
        // Only a contiguous prefix can become the recovery position.
        if self.from > previous.unwrap_or(0) {
            return Err(crate::StoreError::Invalid(format!(
                "database comparison starts at {} after acknowledged {}",
                self.from,
                previous.unwrap_or(0)
            )));
        }
        for (table, key, hash) in &self.rows {
            if let Some(hash) = hash {
                conn.prepare_cached("INSERT INTO tool_row(input_pk,tbl,key,hash) VALUES(?1,?2,?3,?4) ON CONFLICT(input_pk,tbl,key) DO UPDATE SET hash=excluded.hash")?.execute(params![input_pk,table,key,hash])?;
            } else {
                conn.prepare_cached(
                    "DELETE FROM tool_row WHERE input_pk=?1 AND tbl=?2 AND key=?3",
                )?
                .execute(params![input_pk, table, key])?;
            }
        }
        if let Some((table, _, _)) = self.rows.first() {
            crate::fault::check("tool_database.comparison_commit", table)?;
        }
        let schema = self.schema.as_ref().map(serde_json::to_string).transpose()?;
        conn.execute("INSERT INTO tool_database(input_pk,length,schema) VALUES(?1,?2,COALESCE(?3,'{}')) ON CONFLICT(input_pk) DO UPDATE SET length=excluded.length,schema=COALESCE(?3,tool_database.schema)", params![input_pk,length,schema])?;
        Ok(())
    }
}

pub(super) struct Catalog {
    conn: Connection,
    pk: Option<i64>,
    pub schema: Option<Value>,
}
impl Catalog {
    pub fn open(archive: &Archive, input: &str) -> crate::Result<Self> {
        let conn = crate::archive::open_read(&archive.index_db())?;
        conn.pragma_update(None, "temp_store", "file")?;
        conn.pragma_update(Some("temp"), "cache_size", -1024)?;
        conn.execute_batch(
            "CREATE TEMP TABLE seen(tbl TEXT,key TEXT,PRIMARY KEY(tbl,key)) WITHOUT ROWID; BEGIN;",
        )?;
        let pk = conn
            .query_row("SELECT pk FROM input_ref WHERE id=?1", [input], |r| {
                r.get(0)
            })
            .optional()?;
        let schema: Option<String> = conn
            .query_row(
                "SELECT schema FROM tool_database WHERE input_pk=?1",
                [pk],
                |r| r.get(0),
            )
            .optional()?;
        let schema = schema.map(|s| serde_json::from_str(&s)).transpose()?;
        Ok(Self { conn, pk, schema })
    }
    pub fn changed(
        &self,
        table: &str,
        key: &Value,
        row: &Value,
    ) -> crate::Result<bool> {
        let key = key.to_string();
        self.conn
            .prepare_cached("INSERT INTO temp.seen VALUES(?1,?2)")?
            .execute(params![table, key])?;
        let old: Option<Vec<u8>> = self
            .conn
            .prepare_cached("SELECT hash FROM tool_row WHERE input_pk=?1 AND tbl=?2 AND key=?3")?
            .query_row(params![self.pk, table, key], |r| r.get(0))
            .optional()?;
        Ok(old.as_deref() != Some(row_hash(row).as_bytes()))
    }
    pub fn gone(
        &self,
        table: &str,
        visit: &mut dyn FnMut(Value) -> Result<(), DatabaseError>,
    ) -> Result<(), DatabaseError> {
        let mut stmt = self.conn.prepare("SELECT key FROM tool_row WHERE input_pk=?1 AND tbl=?2 AND NOT EXISTS(SELECT 1 FROM temp.seen s WHERE s.tbl=tool_row.tbl AND s.key=tool_row.key) ORDER BY key").map_err(crate::StoreError::from)?;
        let mut rows = stmt
            .query(params![self.pk, table])
            .map_err(crate::StoreError::from)?;
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
            .prepare("SELECT DISTINCT tbl FROM tool_row WHERE input_pk=?1 ORDER BY tbl")
            .map_err(crate::StoreError::from)?;
        let names = stmt
            .query_map([self.pk], |r| r.get::<_, String>(0))
            .map_err(crate::StoreError::from)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(crate::StoreError::from)?;
        for table in names.iter().filter(|t| !tables.contains(&t.as_str())) {
            self.gone(table, &mut |key| visit(table, key))?;
        }
        Ok(())
    }
}

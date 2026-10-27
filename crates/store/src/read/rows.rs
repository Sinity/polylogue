//! Read-only row iteration inside one read transaction, one decoded row at a time.
use super::{Reader, SqlValue};
use crate::{Result, StoreError};

impl Reader<'_> {
    /// Two aggregate counts from one statement, read by column position.
    pub fn query_counts(&self, sql: &str, params: &[SqlValue]) -> Result<(u64, u64)> {
        crate::fault::check("read.query", sql)?;
        let conn = self.conn()?;
        let mut stmt = conn.prepare_cached(sql)?;
        for (i, p) in params.iter().take(stmt.parameter_count()).enumerate() {
            stmt.raw_bind_parameter(i + 1, p)?;
        }
        Ok(stmt
            .raw_query()
            .next()?
            .map(|row| Ok::<_, rusqlite::Error>((row.get(0)?, row.get(1)?)))
            .transpose()?
            .unwrap_or_default())
    }

    /// Collect a read-only query. Parameters past the statement's highest
    /// number are ignored, so one list serves several compiled statements.
    pub fn query(&self, sql: &str, params: &[SqlValue]) -> Result<Vec<serde_json::Value>> {
        let mut out = Vec::new();
        self.query_each(sql, params, |row| {
            out.push(row);
            Ok::<_, StoreError>(())
        })?;
        Ok(out)
    }

    /// Visit rows without holding the complete answer. The callback may
    /// stop on an output or validation error; no later row is then visited.
    /// A single decoded row and SQLite's ordering work remain separate costs.
    pub fn query_each<E>(
        &self,
        sql: &str,
        params: &[SqlValue],
        mut visit: impl FnMut(serde_json::Value) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E>
    where
        E: From<StoreError>,
    {
        crate::fault::check("read.query", sql).map_err(E::from)?;
        let conn = self.conn().map_err(E::from)?;
        let mut stmt = conn.prepare_cached(sql).map_err(StoreError::from)?;
        let names: Vec<String> = stmt
            .column_names()
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        for (i, p) in params.iter().take(stmt.parameter_count()).enumerate() {
            stmt.raw_bind_parameter(i + 1, p)
                .map_err(StoreError::from)?;
        }
        let mut rows = stmt.raw_query();
        while let Some(r) = rows.next().map_err(StoreError::from)? {
            let mut obj = serde_json::Map::new();
            for (i, name) in names.iter().enumerate() {
                use rusqlite::types::ValueRef;
                let v = match r.get_ref(i).map_err(StoreError::from)? {
                    ValueRef::Null => serde_json::Value::Null,
                    ValueRef::Integer(n) => n.into(),
                    ValueRef::Real(f) => serde_json::json!(f),
                    ValueRef::Text(t) => String::from_utf8_lossy(t).into_owned().into(),
                    ValueRef::Blob(b) => b
                        .iter()
                        .map(|x| format!("{x:02x}"))
                        .collect::<String>()
                        .into(),
                };
                obj.insert(name.clone(), v);
            }
            visit(serde_json::Value::Object(obj))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

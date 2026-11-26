use crate::Result;
use rusqlite::{Connection, OptionalExtension, Params, types::FromSql};

pub trait ConnExt {
    fn scalar<T: FromSql>(&self, sql: &str, params: impl Params) -> Result<T>;
    fn scalar_optional<T: FromSql>(&self, sql: &str, params: impl Params) -> Result<Option<T>>;
    fn exists(&self, sql: &str, params: impl Params) -> Result<bool>;
}
impl ConnExt for Connection {
    fn scalar<T: FromSql>(&self, sql: &str, params: impl Params) -> Result<T> {
        Ok(self.query_row(sql, params, |row| row.get(0))?)
    }
    fn scalar_optional<T: FromSql>(&self, sql: &str, params: impl Params) -> Result<Option<T>> {
        Ok(self.query_row(sql, params, |row| row.get(0)).optional()?)
    }
    fn exists(&self, sql: &str, params: impl Params) -> Result<bool> {
        self.scalar(sql, params)
    }
}

/// A failure without its own error, named by what was wrong: text kept as
/// a refusal (`or_invalid`, the writer's remaining form), a broken invariant
/// (`or_internal`) or stored data this build cannot read (`or_stored`).
pub trait ResultExt<T> {
    fn or_invalid(self, what: impl ToString) -> Result<T>;
    fn or_internal(self, what: impl ToString) -> Result<T>;
    fn or_stored(self, what: impl ToString) -> Result<T>;
}
impl<T, E> ResultExt<T> for std::result::Result<T, E> {
    fn or_invalid(self, what: impl ToString) -> Result<T> {
        self.map_err(|_| crate::StoreError::invalid(what))
    }
    fn or_internal(self, what: impl ToString) -> Result<T> {
        self.map_err(|_| crate::StoreError::internal(what))
    }
    fn or_stored(self, what: impl ToString) -> Result<T> {
        self.map_err(|_| crate::StoreError::stored_data(what))
    }
}
pub trait OptionExt<T> {
    fn or_invalid(self, what: impl ToString) -> Result<T>;
    fn or_internal(self, what: impl ToString) -> Result<T>;
    fn or_stored(self, what: impl ToString) -> Result<T>;
}
impl<T> OptionExt<T> for Option<T> {
    fn or_invalid(self, what: impl ToString) -> Result<T> {
        self.ok_or_else(|| crate::StoreError::invalid(what))
    }
    fn or_internal(self, what: impl ToString) -> Result<T> {
        self.ok_or_else(|| crate::StoreError::internal(what))
    }
    fn or_stored(self, what: impl ToString) -> Result<T> {
        self.ok_or_else(|| crate::StoreError::stored_data(what))
    }
}
pub fn to_i64<T: TryInto<i64>>(value: T, what: impl ToString) -> Result<i64> {
    value.try_into().or_internal(what)
}
pub fn to_u64<T: TryInto<u64>>(value: T, what: impl ToString) -> Result<u64> {
    value.try_into().or_internal(what)
}
pub fn meta_get(conn: &Connection, key: &str) -> Result<Option<String>> {
    conn.scalar_optional("SELECT value FROM meta WHERE key=?1", [key])
}
pub fn meta_set(conn: &Connection, key: &str, value: impl rusqlite::ToSql) -> Result<()> {
    conn.execute("INSERT INTO meta(key,value) VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", rusqlite::params![key,value])?;
    Ok(())
}
/// The write counter of the last committed transaction (`meta`).
pub(crate) fn write_counter(conn: &Connection) -> Result<i64> {
    Ok(conn
        .scalar_optional(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'write_counter'",
            [],
        )?
        .unwrap_or(0))
}
#[cfg(test)]
mod tests;

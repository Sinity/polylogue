//! A private derived SQLite file; readers never open the source archive.
use crate::{Result, StoreError};
use polylogue_model::ChangePosition;
use query_language::catalog::OutputSchema;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    time::Duration,
};

const VERSION: u32 = 2;
const SCHEMA: &str = "
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE session(id TEXT PRIMARY KEY,native_id TEXT NOT NULL,prefix TEXT NOT NULL,title TEXT NOT NULL,title_key TEXT NOT NULL,origin TEXT NOT NULL,repo TEXT,last_activity INTEGER,first_prompt TEXT) WITHOUT ROWID;
CREATE INDEX session_native ON session(native_id,id);
CREATE INDEX session_recent ON session(last_activity DESC,id);
CREATE TABLE contribution(session TEXT NOT NULL,field TEXT NOT NULL,value TEXT NOT NULL,n INTEGER NOT NULL CHECK(n>0),PRIMARY KEY(session,field,value)) WITHOUT ROWID;
CREATE TABLE value_count(field TEXT NOT NULL,value TEXT NOT NULL,n INTEGER NOT NULL CHECK(n>=0),PRIMARY KEY(field,value)) WITHOUT ROWID;
CREATE TABLE item(kind TEXT NOT NULL,id TEXT NOT NULL,name TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(kind,id)) WITHOUT ROWID;
CREATE INDEX item_name ON item(kind,name,id);
";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub archive: String,
    pub position: ChangePosition,
    pub written_at: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub native_id: String,
    pub prefix: String,
    pub title: String,
    pub origin: String,
    pub repo: Option<String>,
    pub last_activity: Option<i64>,
    pub first_prompt: Option<String>,
    /// Each field/value is counted once or by its declared native grain.
    pub values: BTreeMap<String, BTreeMap<String, u64>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValueCount {
    pub value: String,
    pub count: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub kind: Option<query_language::catalog::FieldKind>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Saved {
    pub family: String,
    pub revision: String,
    pub name: String,
    pub text: String,
    pub parameters: Vec<Parameter>,
    pub help: String,
    pub schema: Option<OutputSchema>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Kept {
    pub id: String,
    pub name: Option<String>,
    pub unit: Option<String>,
    pub row_count: Option<u64>,
    pub rows: Option<OutputSchema>,
    pub matches: Option<OutputSchema>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Setting {
    pub key: String,
    pub value: serde_json::Value,
    pub choices: Vec<String>,
}

pub struct CatalogWriter {
    conn: Connection,
}
pub struct CatalogReader {
    conn: Connection,
    pub metadata: Metadata,
}

fn configure(conn: &Connection) -> Result<()> {
    conn.busy_timeout(Duration::from_millis(40))?;
    conn.execute_batch("PRAGMA mmap_size=268435456; PRAGMA temp_store=MEMORY;")?;
    Ok(())
}
fn count(value: u64) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| StoreError::Invalid("completion count exceeds SQLite's integer range".into()))
}
fn text_key(value: &str) -> String {
    polylogue_model::text::key(value)
}

impl CatalogWriter {
    /// Build in a private sibling, then close, sync and rename the coherent
    /// database before opening its canonical journal path for incremental writes.
    pub fn build(
        path: &Path,
        metadata: &Metadata,
        populate: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<Self> {
        let parent = path
            .parent()
            .ok_or_else(|| StoreError::Invalid("completion catalog has no parent".into()))?;
        fs::create_dir_all(parent)?;
        let temporary = parent.join(format!(".catalog-{}", uuid::Uuid::now_v7()));
        let guard = Temporary(temporary.clone());
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        let conn = Connection::open_with_flags(
            &temporary,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        configure(&conn)?;
        conn.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;")?;
        conn.execute_batch(SCHEMA)?;
        let mut writer = Self { conn };
        writer.update(metadata, populate)?;
        writer
            .conn
            .close()
            .map_err(|(_, error)| StoreError::from(error))?;
        fs::File::open(&temporary)?.sync_all()?;
        fs::rename(&temporary, path)?;
        fs::File::open(parent)?.sync_all()?;
        drop(guard);
        Self::open(path)
    }
    fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        configure(&conn)?;
        conn.execute_batch("PRAGMA synchronous=FULL;")?;
        Ok(Self { conn })
    }
    /// The position is committed with every changed section, never beforehand.
    pub fn update(
        &mut self,
        metadata: &Metadata,
        apply: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        let result = (|| {
            apply(self)?;
            for (key, value) in [
                ("version", VERSION.to_string()),
                ("archive", metadata.archive.clone()),
                ("position", serde_json::to_string(&metadata.position)?),
                ("written_at", metadata.written_at.to_string()),
            ] {
                self.conn.execute("INSERT INTO metadata VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", params![key,value])?;
            }
            self.conn.execute_batch("COMMIT")?;
            Ok(())
        })();
        if result.is_err() {
            let _ = self.conn.execute_batch("ROLLBACK");
        }
        result
    }
    pub fn put_catalog(&mut self, catalog: &query_language::catalog::Catalog) -> Result<()> {
        self.conn.execute("INSERT INTO metadata VALUES('catalog',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [serde_json::to_string(catalog)?])?;
        Ok(())
    }
    pub fn put_session(&mut self, session: &Session) -> Result<()> {
        self.conn.execute("INSERT INTO session VALUES(?1,?2,'',?3,?4,?5,?6,?7,?8) ON CONFLICT(id) DO UPDATE SET native_id=excluded.native_id,title=excluded.title,title_key=excluded.title_key,origin=excluded.origin,repo=excluded.repo,last_activity=excluded.last_activity,first_prompt=excluded.first_prompt",
            params![session.id,session.native_id,session.title,text_key(&session.title),session.origin,session.repo,session.last_activity,session.first_prompt])?;
        self.put_values(&format!("session:{}", session.id), &session.values)?;
        self.refresh_neighbors(&session.id)
    }
    pub fn put_values(
        &mut self,
        owner: &str,
        values: &BTreeMap<String, BTreeMap<String, u64>>,
    ) -> Result<()> {
        self.remove_values(owner)?;
        for (field, values) in values {
            for (value, n) in values {
                if *n == 0 {
                    continue;
                }
                let n = count(*n)?;
                self.conn.execute(
                    "INSERT INTO contribution VALUES(?1,?2,?3,?4)",
                    params![owner, field, value, n],
                )?;
                self.conn.execute("INSERT INTO value_count VALUES(?1,?2,?3) ON CONFLICT(field,value) DO UPDATE SET n=n+excluded.n", params![field,value,n])?;
            }
        }
        Ok(())
    }
    pub fn remove_session(&mut self, id: &str) -> Result<()> {
        let neighbors = self.neighbors(id)?;
        self.remove_values(&format!("session:{id}"))?;
        self.conn.execute("DELETE FROM session WHERE id=?1", [id])?;
        for neighbor in neighbors {
            self.refresh_prefix(&neighbor)?;
        }
        Ok(())
    }
    fn remove_values(&self, id: &str) -> Result<()> {
        let old = self
            .conn
            .prepare("SELECT field,value,n FROM contribution WHERE session=?1")?
            .query_map([id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for (field, value, n) in old {
            self.conn.execute(
                "UPDATE value_count SET n=n-?3 WHERE field=?1 AND value=?2",
                params![field, value, n],
            )?;
            self.conn.execute(
                "DELETE FROM value_count WHERE field=?1 AND value=?2 AND n=0",
                params![field, value],
            )?;
        }
        self.conn
            .execute("DELETE FROM contribution WHERE session=?1", [id])?;
        Ok(())
    }
    fn neighbors(&self, id: &str) -> Result<Vec<String>> {
        let mut values = vec![];
        for sql in [
            "SELECT id FROM session WHERE id<?1 ORDER BY id DESC LIMIT 1",
            "SELECT id FROM session WHERE id>?1 ORDER BY id LIMIT 1",
        ] {
            if let Some(value) = self
                .conn
                .query_row(sql, [id], |row| row.get(0))
                .optional()?
            {
                values.push(value);
            }
        }
        Ok(values)
    }
    fn refresh_neighbors(&self, id: &str) -> Result<()> {
        self.refresh_prefix(id)?;
        for neighbor in self.neighbors(id)? {
            self.refresh_prefix(&neighbor)?;
        }
        Ok(())
    }
    fn refresh_prefix(&self, id: &str) -> Result<()> {
        let longest = self
            .neighbors(id)?
            .iter()
            .map(|neighbor| common_prefix(id, neighbor))
            .max()
            .unwrap_or(0);
        let minimum = id.find(':').map_or(0, |at| at + 1);
        let at = longest.max(minimum);
        let end = id
            .char_indices()
            .find(|(index, _)| *index >= at)
            .map_or(id.len(), |(index, c)| index + c.len_utf8());
        self.conn.execute(
            "UPDATE session SET prefix=?2 WHERE id=?1",
            params![id, &id[..end]],
        )?;
        Ok(())
    }
    pub fn put_saved(&mut self, item: &Saved) -> Result<()> {
        self.put_item("saved", &item.family, &item.name, item)
    }
    pub fn put_kept(&mut self, item: &Kept) -> Result<()> {
        self.put_item(
            "kept",
            &item.id,
            item.name.as_deref().unwrap_or(&item.id),
            item,
        )
    }
    pub fn put_setting(&mut self, item: &Setting) -> Result<()> {
        self.put_item("setting", &item.key, &item.key, item)
    }
    fn put_item(&self, kind: &str, id: &str, name: &str, item: &impl Serialize) -> Result<()> {
        self.conn.execute("INSERT INTO item VALUES(?1,?2,?3,?4) ON CONFLICT(kind,id) DO UPDATE SET name=excluded.name,payload=excluded.payload", params![kind,id,name,serde_json::to_string(item)?])?;
        Ok(())
    }
    pub fn remove_saved(&mut self, family:&str)->Result<()> {
        self.conn.execute("DELETE FROM item WHERE kind='saved' AND id=?1",[family])?;
        Ok(())
    }
}
impl CatalogReader {
    pub fn open(path: &Path, archive: &str) -> Result<Option<Self>> {
        if !path.is_file() {
            return Ok(None);
        }
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        configure(&conn)?;
        conn.execute_batch("PRAGMA query_only=ON; BEGIN")?;
        let get = |key: &str| {
            conn.query_row("SELECT value FROM metadata WHERE key=?1", [key], |row| {
                row.get::<_, String>(0)
            })
        };
        if get("version")? != VERSION.to_string() || get("archive")? != archive {
            return Ok(None);
        }
        let metadata = Metadata {
            archive: archive.into(),
            position: serde_json::from_str(&get("position")?)?,
            written_at: get("written_at")?
                .parse()
                .map_err(|_| StoreError::Invalid("invalid completion time".into()))?,
        };
        Ok(Some(Self { conn, metadata }))
    }
    pub fn catalog(&self) -> Result<query_language::catalog::Catalog> {
        let value: String = self.conn.query_row(
            "SELECT value FROM metadata WHERE key='catalog'",
            [],
            |row| row.get(0),
        )?;
        Ok(serde_json::from_str(&value)?)
    }
    pub fn values(&self, field: &str, prefix: &str) -> Result<Vec<ValueCount>> {
        let upper = prefix_end(prefix);
        let mut statement = self.conn.prepare("SELECT value,n FROM value_count WHERE field=?1 AND value>=?2 AND (?3 IS NULL OR value<?3) AND n>0 ORDER BY n DESC,value")?;
        Ok(statement
            .query_map(params![field, prefix, upper], |row| {
                Ok(ValueCount {
                    value: row.get(0)?,
                    count: row.get::<_, i64>(1)? as u64,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn sessions(&self, prefix: &str) -> Result<Vec<Session>> {
        let upper = prefix_end(prefix);
        let sql = "SELECT id,native_id,prefix,title,origin,repo,last_activity FROM session WHERE (id>=?1 AND (?2 IS NULL OR id<?2)) OR (native_id>=?1 AND (?2 IS NULL OR native_id<?2)) ORDER BY last_activity IS NULL,last_activity DESC,id";
        let mut statement = self.conn.prepare(sql)?;
        let mut sessions = statement
            .query_map(params![prefix, upper], session)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if sessions.is_empty() && !prefix.is_empty() {
            let pattern = fuzzy_pattern(&text_key(prefix));
            sessions=self.conn.prepare("SELECT id,native_id,prefix,title,origin,repo,last_activity FROM session WHERE title_key LIKE ?1 ESCAPE '\\' ORDER BY last_activity IS NULL,last_activity DESC,id")?.query_map([pattern],session)?.collect::<rusqlite::Result<Vec<_>>>()?;
        }
        Ok(sessions)
    }
    pub fn saved(&self, name: &str) -> Result<Vec<Saved>> {
        let mut statement = self
            .conn
            .prepare("SELECT payload FROM item WHERE kind='saved' AND name=?1 ORDER BY id")?;
        statement
            .query_map([name], |row| row.get::<_, String>(0))?
            .map(|row| Ok(serde_json::from_str(&row?)?))
            .collect()
    }
    pub fn kept(&self, id: &str) -> Result<Option<Kept>> {
        let payload = self
            .conn
            .query_row(
                "SELECT payload FROM item WHERE kind='kept' AND id=?1",
                [id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        payload
            .map(|payload| Ok(serde_json::from_str(&payload)?))
            .transpose()
    }
    pub fn session_details(&self, id: &str) -> Result<Option<Session>> {
        let mut value=self.conn.query_row("SELECT id,native_id,prefix,title,origin,repo,last_activity FROM session WHERE id=?1",[id],session).optional()?;
        if let Some(value) = value.as_mut() {
            value.first_prompt = self.conn.query_row(
                "SELECT first_prompt FROM session WHERE id=?1",
                [id],
                |row| row.get(0),
            )?;
        }
        Ok(value)
    }
    pub fn settings(&self, prefix: &str) -> Result<Vec<Setting>> {
        self.items("setting", prefix)
    }
    pub fn saved_names(&self, prefix: &str) -> Result<Vec<Saved>> {
        self.items("saved", prefix)
    }
    pub fn kept_names(&self, prefix: &str) -> Result<Vec<Kept>> {
        let mut statement=self.conn.prepare("SELECT payload FROM item WHERE kind='kept' AND ((name>=?1 AND (?2 IS NULL OR name<?2)) OR (id>=?1 AND (?2 IS NULL OR id<?2))) ORDER BY name,id")?;
        statement
            .query_map(params![prefix, prefix_end(prefix)], |row| {
                row.get::<_, String>(0)
            })?
            .map(|row| Ok(serde_json::from_str(&row?)?))
            .collect()
    }
    fn items<T: serde::de::DeserializeOwned>(&self, kind: &str, prefix: &str) -> Result<Vec<T>> {
        let mut statement=self.conn.prepare("SELECT payload FROM item WHERE kind=?1 AND name>=?2 AND (?3 IS NULL OR name<?3) ORDER BY name,id")?;
        let rows = statement.query_map(params![kind, prefix, prefix_end(prefix)], |row| {
            row.get::<_, String>(0)
        })?;
        rows.map(|row| Ok(serde_json::from_str(&row?)?)).collect()
    }
}
fn session(row: &rusqlite::Row<'_>) -> rusqlite::Result<Session> {
    Ok(Session {
        id: row.get(0)?,
        native_id: row.get(1)?,
        prefix: row.get(2)?,
        title: row.get(3)?,
        origin: row.get(4)?,
        repo: row.get(5)?,
        last_activity: row.get(6)?,
        first_prompt: None,
        values: BTreeMap::new(),
    })
}
fn common_prefix(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(a, b)| a == b)
        .map(|(c, _)| c.len_utf8())
        .sum()
}
fn prefix_end(prefix: &str) -> Option<String> {
    let mut value = prefix.to_owned();
    while let Some(c) = value.pop() {
        let mut next = u32::from(c) + 1;
        if (0xd800..=0xdfff).contains(&next) {
            next = 0xe000;
        }
        if let Some(c) = char::from_u32(next) {
            value.push(c);
            return Some(value);
        }
    }
    None
}
fn fuzzy_pattern(prefix: &str) -> String {
    let mut pattern = String::from("%");
    for c in prefix.chars() {
        if matches!(c, '%' | '_' | '\\') {
            pattern.push('\\');
        }
        pattern.push(c);
        pattern.push('%');
    }
    pattern
}
struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests;

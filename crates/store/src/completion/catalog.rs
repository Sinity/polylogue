//! A private derived SQLite file; readers never open the source archive.
use crate::ConnExt as _;
use crate::{OptionExt as _, ResultExt as _};
use crate::{Result, StoreError};
use polylogue_model::ChangePosition;
use query_language::catalog::OutputSchema;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    time::Duration,
};

/// The file behind a path is still the one this process opened or created
/// when its device and inode match.
fn file_identity(file: &fs::Metadata) -> (u64, u64) {
    (file.dev(), file.ino())
}

const VERSION: u32 = 7;
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
    /// Each field/value is counted once or by its declared grain.
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
    pub relation_scope: Option<OutputSchema>,
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
    pub value: polylogue_model::settings::Current,
    pub choices: Vec<String>,
}

/// Derived dependencies needed to continue publication after reopening the cache.
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct Dependencies {
    pub names: BTreeMap<String, usize>,
    pub sources: std::collections::BTreeSet<String>,
}
/// Publication state reloaded on resume: each family's dependencies and each
/// kept result's stored shape.
#[derive(Default)]
pub struct PublisherVocabulary {
    pub families: BTreeMap<String, Dependencies>,
    pub kept: BTreeMap<String, polylogue_query::Kept>,
}

pub struct CatalogWriter {
    conn: Connection,
    pub(super) path: PathBuf,
    identity: (u64, u64),
}
pub struct CatalogReader {
    conn: Connection,
    pub metadata: Metadata,
    path: PathBuf,
    identity: (u64, u64),
}

fn configure(conn: &Connection) -> Result<()> {
    conn.busy_timeout(Duration::from_millis(40))?;
    conn.execute_batch("PRAGMA mmap_size=268435456; PRAGMA temp_store=MEMORY;")?;
    Ok(())
}
// SQLite's persistent WAL option preserves the journals for read-only clients
// after the writer closes; it does not change transaction or checkpoint rules.
#[allow(unsafe_code)]
fn persist_wal(conn: &Connection) -> Result<()> {
    let mut enabled: std::ffi::c_int = 1;
    // SAFETY: the connection and integer remain live for the synchronous call;
    // this opcode reads one integer and configures the connection's main file.
    let code = unsafe {
        rusqlite::ffi::sqlite3_file_control(
            conn.handle(),
            c"main".as_ptr(),
            rusqlite::ffi::SQLITE_FCNTL_PERSIST_WAL,
            (&mut enabled as *mut std::ffi::c_int).cast(),
        )
    };
    if code != rusqlite::ffi::SQLITE_OK {
        return Err(rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(code), None).into());
    }
    Ok(())
}
fn count(value: u64) -> Result<i64> {
    crate::to_i64(value, "completion count exceeds SQLite's integer range")
}
fn text_key(value: &str) -> String {
    polylogue_model::text::key(value)
}

fn read_metadata(conn: &Connection, archive: &str) -> Result<Option<Metadata>> {
    let get = |key: &str| conn.scalar::<String>("SELECT value FROM metadata WHERE key=?1", [key]);
    if get("version")? != VERSION.to_string() || get("archive")? != archive {
        return Ok(None);
    }
    Ok(Some(Metadata {
        archive: archive.into(),
        position: serde_json::from_str(&get("position")?)?,
        written_at: get("written_at")?
            .parse()
            .or_stored("invalid completion time")?,
    }))
}

impl CatalogWriter {
    /// Seed a synthetic catalog for cross-crate read tests. Production builds
    /// allocate their database file through CatalogPointer.
    #[cfg(feature = "fixture")]
    pub fn fixture(
        path: &Path,
        metadata: &Metadata,
        populate: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<Self> {
        Self::build(path, metadata, populate)
    }
    /// Build in a private sibling, then close, sync and rename the coherent
    /// database before opening its fresh journal namespace for incremental writes.
    pub(super) fn build(
        path: &Path,
        metadata: &Metadata,
        populate: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<Self> {
        let parent = path
            .parent()
            .or_internal("completion catalog has no parent")?;
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
        let file = fs::metadata(&temporary)?;
        let mut writer = Self {
            conn,
            path: temporary.clone(),
            identity: file_identity(&file),
        };
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
        let before = fs::metadata(path)?;
        // Cache directories may be aliases; NOFOLLOW protects the database
        // itself after resolving its directory, not the configured directory.
        let database = fs::canonicalize(
            path.parent()
                .or_internal("completion catalog has no parent")?,
        )?
        .join(
            path.file_name()
                .or_internal("completion catalog has no file name")?,
        );
        let conn = Connection::open_with_flags(
            database,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        configure(&conn)?;
        // New readers must keep the previous committed vocabulary while a
        // new catalog commits. Rollback journaling excludes them at COMMIT.
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;")?;
        persist_wal(&conn)?;
        // Initialize the WAL and shared memory as the writer, before activation.
        // Offline readers need neither directory writes nor journal creation.
        conn.query_row("SELECT count(*) FROM metadata", [], |_| Ok(()))?;
        let file = fs::metadata(path)?;
        if file_identity(&before) != file_identity(&file) {
            return Err(StoreError::ScopeChanged);
        }
        Ok(Self {
            conn,
            path: path.to_owned(),
            identity: file_identity(&file),
        })
    }
    pub(super) fn resume(
        path: &Path,
        current: &Metadata,
        catalog: &query_language::catalog::Catalog,
        machine: &str,
    ) -> Result<Option<(Self, Metadata, PublisherVocabulary)>> {
        // Validate the derived format before opening a writer. Release this
        // readonly_shm mapping first, so it cannot make our writer read-only.
        let identity = {
            let cache = match CatalogReader::open(path, &current.archive) {
                Ok(Some(cache)) => cache,
                Ok(None) => {
                    crate::log!(
                        warning,
                        "completion catalog: missing or incompatible header; rebuilding"
                    );
                    return Ok(None);
                }
                Err(StoreError::Sqlite(error))
                    if error.sqlite_error_code() == Some(rusqlite::ErrorCode::CannotOpen) =>
                {
                    // Offline reads cannot create a missing WAL mapping. A
                    // replacement gets its own complete journal namespace.
                    crate::log!(
                        warning,
                        "completion catalog: cannot read cached database or journals ({error}); rebuilding"
                    );
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            cache.check()?;
            cache.identity
        };
        if fs::metadata(path)?.permissions().readonly() {
            crate::log!(
                warning,
                "completion catalog: database has no write permission; rebuilding"
            );
            return Ok(None);
        }
        let writer = Self::open(path)?;
        if writer.identity != identity {
            return Ok(None);
        }
        writer.conn.execute_batch("BEGIN IMMEDIATE")?;
        let loaded: Result<Option<(Metadata, PublisherVocabulary)>> = (|| {
            let Some(metadata) = read_metadata(&writer.conn, &current.archive)? else {
                return Ok(None);
            };
            let position = &metadata.position;
            let cached_catalog: String = writer
                .conn
                .scalar("SELECT value FROM metadata WHERE key='catalog'", [])?;
            let incompatible = if position.index_instance != current.position.index_instance {
                Some("index generation changed")
            } else if position.index < 0 || position.archive < 0 {
                Some("negative stored position")
            } else if position.index > current.position.index
                || position.archive > current.position.archive
            {
                Some("stored position is ahead of source")
            } else if serde_json::from_str::<serde_json::Value>(&cached_catalog)?
                != serde_json::to_value(catalog)?
            {
                Some("language catalog changed")
            } else if writer.conn.scalar::<String>(
                "SELECT value FROM metadata WHERE key='publisher_machine'",
                [],
            )? != machine
            {
                Some("publishing machine changed")
            } else {
                None
            };
            if let Some(reason) = incompatible {
                crate::log!(warning, "completion catalog: {reason}; rebuilding");
                return Ok(None);
            }
            let mut vocabulary = PublisherVocabulary::default();
            let mut statement = writer.conn.prepare("SELECT kind,id,payload FROM item WHERE kind IN ('dependencies','kept_binding') ORDER BY kind,id")?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                let kind: String = row.get(0)?;
                let id: String = row.get(1)?;
                let payload: String = row.get(2)?;
                if kind == "dependencies" {
                    vocabulary
                        .families
                        .insert(id, serde_json::from_str(&payload)?);
                } else {
                    let kept: polylogue_query::Kept = serde_json::from_str(&payload)?;
                    kept.validate_output_metadata()
                        .map_err(|error| StoreError::internal(error.to_string()))?;
                    vocabulary.kept.insert(id, kept);
                }
            }
            Ok(Some((metadata, vocabulary)))
        })();
        // Reopening tests write capability and pins one snapshot without
        // changing the cache's rows, position or last-publication time.
        // A load failure explains why the cache is unusable, so it is reported
        // ahead of a rollback failure.
        let rolled_back = writer.conn.execute_batch("ROLLBACK");
        let loaded = loaded?;
        rolled_back?;
        let Some((metadata, vocabulary)) = loaded else {
            return Ok(None);
        };
        if !writer.is_current_file() {
            return Ok(None);
        }
        Ok(Some((writer, metadata, vocabulary)))
    }

    /// Publication dependencies share the transaction and position of their visible rows.
    pub fn put_dependencies(&mut self, id: &str, dependencies: &Dependencies) -> Result<()> {
        if dependencies.names.is_empty() && dependencies.sources.is_empty() {
            self.conn
                .execute("DELETE FROM item WHERE kind='dependencies' AND id=?1", [id])?;
            Ok(())
        } else {
            self.put_item("dependencies", id, "", dependencies)
        }
    }
    pub fn put_kept_binding(
        &mut self,
        id: &str,
        binding: Option<&polylogue_query::Kept>,
    ) -> Result<()> {
        if let Some(binding) = binding {
            self.put_item("kept_binding", id, "", binding)
        } else {
            self.conn
                .execute("DELETE FROM item WHERE kind='kept_binding' AND id=?1", [id])?;
            Ok(())
        }
    }
    pub fn put_publisher_machine(&mut self, machine: &str) -> Result<()> {
        self.conn.execute("INSERT INTO metadata VALUES('publisher_machine',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [machine])?;
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn connection(&self) -> &Connection {
        &self.conn
    }
    /// A deleted derived file is rebuilt on the next publication.
    pub fn is_current_file(&self) -> bool {
        fs::metadata(&self.path).is_ok_and(|file| file_identity(&file) == self.identity)
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
            crate::fault::check("completion.before_commit", &self.path.to_string_lossy())?;
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
    /// A session's row, and the values of each field `session.values`
    /// names: those fields' values become exactly the given ones. Values of
    /// other fields (the counted models and tools) stay as they are.
    pub fn put_session(&mut self, session: &Session) -> Result<()> {
        let existed: bool = self.conn.exists(
            "SELECT EXISTS(SELECT 1 FROM session WHERE id=?1)",
            [&session.id],
        )?;
        self.conn.execute("INSERT INTO session VALUES(?1,?2,'',?3,?4,?5,?6,?7,?8) ON CONFLICT(id) DO UPDATE SET native_id=excluded.native_id,title=excluded.title,title_key=excluded.title_key,origin=excluded.origin,repo=excluded.repo,last_activity=excluded.last_activity,first_prompt=excluded.first_prompt",
            params![session.id,session.native_id,session.title,text_key(&session.title),session.origin,session.repo,session.last_activity,session.first_prompt])?;
        let contributor = format!("session:{}", session.id);
        for (field, values) in &session.values {
            let stored = self
                .conn
                .prepare_cached("SELECT value FROM contribution WHERE session=?1 AND field=?2")?
                .query_map([&contributor, field], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            for value in stored {
                if !values.contains_key(&value) {
                    self.set_value(&contributor, field, &value, 0)?;
                }
            }
            for (value, n) in values {
                self.set_value(&contributor, field, value, *n)?;
            }
        }
        // An ID's shortest unique prefix changes only when IDs come or go.
        if !existed {
            self.refresh_neighbors(&session.id)?;
        }
        Ok(())
    }
    /// One change read from the index ([`crate::Reader::visit_completion_sessions`]).
    pub fn apply(&mut self, change: super::SessionChange) -> Result<()> {
        use super::SessionChange;
        match change {
            SessionChange::Upsert(session) => self.put_session(&session),
            SessionChange::Recreated(session) => {
                self.remove_session(&session.id)?;
                self.put_session(&session)
            }
            SessionChange::Removed(id) => self.remove_session(&id),
            SessionChange::Value {
                session,
                field,
                value,
                count,
            } => self.set_value(&format!("session:{session}"), &field, &value, count),
        }
    }
    /// One contributor's count of one value; 0 removes it.
    pub fn set_value(&mut self, contributor: &str, field: &str, value: &str, n: u64) -> Result<()> {
        let old: i64 = self
            .conn
            .prepare_cached(
                "SELECT n FROM contribution WHERE session=?1 AND field=?2 AND value=?3",
            )?
            .query_row(params![contributor, field, value], |row| row.get(0))
            .optional()?
            .unwrap_or(0);
        let new = count(n)?;
        if old == new {
            return Ok(());
        }
        if new == 0 {
            self.conn
                .prepare_cached(
                    "DELETE FROM contribution WHERE session=?1 AND field=?2 AND value=?3",
                )?
                .execute(params![contributor, field, value])?;
        } else {
            self.conn
                .prepare_cached("INSERT INTO contribution VALUES(?1,?2,?3,?4) ON CONFLICT(session,field,value) DO UPDATE SET n=excluded.n")?
                .execute(params![contributor, field, value, new])?;
        }
        // SQLite checks the proposed row of an upsert, so a decrease updates.
        if new > old {
            self.conn
                .prepare_cached("INSERT INTO value_count VALUES(?1,?2,?3) ON CONFLICT(field,value) DO UPDATE SET n=n+excluded.n")?
                .execute(params![field, value, new - old])?;
        } else {
            self.conn
                .prepare_cached("UPDATE value_count SET n=n-?3 WHERE field=?1 AND value=?2")?
                .execute(params![field, value, old - new])?;
            self.conn
                .prepare_cached("DELETE FROM value_count WHERE field=?1 AND value=?2 AND n=0")?
                .execute(params![field, value])?;
        }
        Ok(())
    }
    pub fn put_values(
        &mut self,
        contributor: &str,
        values: &BTreeMap<String, BTreeMap<String, u64>>,
    ) -> Result<()> {
        self.remove_values(contributor)?;
        for (field, values) in values {
            for (value, n) in values {
                self.set_value(contributor, field, value, *n)?;
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
            .query_map([id], |row| <(String, String, i64)>::try_from(row))?
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
            if let Some(value) = self.conn.scalar_optional(sql, [id])? {
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
        if item.key != item.value.key {
            return Err(StoreError::internal(
                "setting cache key differs from its typed value",
            ));
        }
        self.put_item("setting", &item.key, &item.key, item)
    }
    /// Existing cache keys are invalidation candidates, never the source of truth.
    pub fn setting_keys(&self) -> Result<Vec<String>> {
        Ok(self
            .conn
            .prepare("SELECT id FROM item WHERE kind='setting' ORDER BY id")?
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn remove_setting(&mut self, key: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM item WHERE kind='setting' AND id=?1", [key])?;
        Ok(())
    }
    /// Replace only authored vocabulary after the authored records were rewritten.
    /// Session contributions remain independent of this category.
    pub fn clear_authored(&mut self) -> Result<()> {
        for family in self.definition_families()? {
            self.remove_values(&format!("annotation:{family}"))?;
        }
        self.conn.execute(
            "DELETE FROM item WHERE kind IN ('saved','kept','setting','dependencies','kept_binding')",
            [],
        )?;
        Ok(())
    }

    /// Includes tag-only contributors as well as saved definitions.
    pub fn definition_families(&self) -> Result<Vec<String>> {
        Ok(self.conn.prepare("SELECT id FROM item WHERE kind='saved' UNION SELECT substr(session,12) FROM contribution WHERE session GLOB 'annotation:*' ORDER BY 1")?
            .query_map([],|row|row.get(0))?.collect::<rusqlite::Result<_>>()?)
    }
    fn put_item(&self, kind: &str, id: &str, name: &str, item: &impl Serialize) -> Result<()> {
        self.conn.execute("INSERT INTO item VALUES(?1,?2,?3,?4) ON CONFLICT(kind,id) DO UPDATE SET name=excluded.name,payload=excluded.payload", params![kind,id,name,serde_json::to_string(item)?])?;
        Ok(())
    }
    pub fn remove_saved(&mut self, family: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM item WHERE kind='saved' AND id=?1", [family])?;
        Ok(())
    }
}
impl CatalogReader {
    pub fn open(path: &Path, archive: &str) -> Result<Option<Self>> {
        if !path.is_file() {
            return Ok(None);
        }
        let before = fs::metadata(path)?;
        // Read-only database access alone may still create or rebuild SHM.
        // SQLite's VFS option keeps offline clients read-only there as well.
        let mut uri = String::from("file:");
        for byte in path.as_os_str().as_encoded_bytes() {
            if byte.is_ascii_alphanumeric() || b"/._-".contains(byte) {
                uri.push(char::from(*byte));
            } else {
                use std::fmt::Write;
                write!(&mut uri, "%{byte:02X}").expect("writing a String cannot fail");
            }
        }
        uri.push_str("?readonly_shm=1");
        let conn = Connection::open_with_flags(
            uri,
            OpenFlags::SQLITE_OPEN_READ_ONLY
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_URI,
        )?;
        configure(&conn)?;
        conn.execute_batch("PRAGMA query_only=ON; BEGIN")?;
        let Some(metadata) = read_metadata(&conn, archive)? else {
            return Ok(None);
        };
        let reader = Self {
            conn,
            metadata,
            path: path.to_owned(),
            identity: file_identity(&before),
        };
        reader.check()?;
        Ok(Some(reader))
    }
    /// A revoked or replaced pointer cannot authorize an already-open snapshot.
    pub fn check(&self) -> Result<()> {
        let current = fs::metadata(&self.path).map_err(|_| StoreError::ScopeChanged)?;
        if file_identity(&current) != self.identity {
            return Err(StoreError::ScopeChanged);
        }
        Ok(())
    }
    pub fn catalog(&self) -> Result<query_language::catalog::Catalog> {
        let value: String = self
            .conn
            .scalar("SELECT value FROM metadata WHERE key='catalog'", [])?;
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
        let sql = format!(
            "SELECT {SESSION_COLUMNS} FROM session WHERE (id>=?1 AND (?2 IS NULL OR id<?2)) OR (native_id>=?1 AND (?2 IS NULL OR native_id<?2)) ORDER BY last_activity IS NULL,last_activity DESC,id"
        );
        let mut statement = self.conn.prepare(&sql)?;
        let mut sessions = statement
            .query_map(params![prefix, upper], session)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        // A prefix of only combining marks has an empty key, which would match every title.
        let key = text_key(prefix);
        if sessions.is_empty() && !key.is_empty() {
            // The prefix's characters appear in order in the title; compared here
            // because a LIKE pattern has a length limit and a prefix has none.
            let mut statement = self.conn.prepare(&format!(
                "SELECT {SESSION_COLUMNS},title_key FROM session ORDER BY last_activity IS NULL,last_activity DESC,id"
            ))?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                if in_order(&key, &row.get::<_, String>(7)?) {
                    sessions.push(session(row)?);
                }
            }
        }
        Ok(sessions)
    }
    pub fn saved(&self, name: &str) -> Result<Vec<Saved>> {
        let mut statement = self
            .conn
            .prepare("SELECT payload FROM item WHERE kind='saved' AND name=?1 ORDER BY id")?;
        statement
            .query_map([name], |row| row.get::<_, String>(0))?
            .map(|row| Ok(serde_json::from_str::<Saved>(&row?)?))
            .collect()
    }
    pub fn kept(&self, id: &str) -> Result<Option<Kept>> {
        let payload = self.conn.scalar_optional::<String>(
            "SELECT payload FROM item WHERE kind='kept' AND id=?1",
            [id],
        )?;
        payload
            .map(|payload| Ok(serde_json::from_str(&payload)?))
            .transpose()
    }
    pub fn session_details(&self, id: &str) -> Result<Option<Session>> {
        Ok(self
            .conn
            .query_row(
                &format!("SELECT {SESSION_COLUMNS},first_prompt FROM session WHERE id=?1"),
                [id],
                |row| {
                    let mut value = session(row)?;
                    value.first_prompt = row.get(7)?;
                    Ok(value)
                },
            )
            .optional()?)
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
/// The columns `session` reads, by position.
const SESSION_COLUMNS: &str = "id,native_id,prefix,title,origin,repo,last_activity";
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
/// The smallest string above every string that starts with `prefix`; `None`
/// means no upper bound (the empty prefix, or a prefix of only the greatest
/// character).
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
fn in_order(needle: &str, haystack: &str) -> bool {
    let mut rest = haystack.chars();
    needle.chars().all(|c| rest.any(|h| h == c))
}
struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests;

//! Private SQLite ownership and typed foundation storage. No product ingestion is registered.
//!
//! A read handle cannot acquire mutation capability:
//! ```compile_fail
//! fn mutate(reader: &mut polylogue_store::ReadHandle) {
//!     reader.put_file_blob(b"unauthorized").unwrap();
//! }
//! ```
//! Connections and SQL remain private:
//! ```compile_fail
//! fn escape(writer: &polylogue_store::Writer) { let _ = &writer.archive; }
//! ```
use polylogue_model::BlobHash;
use rusqlite::{Connection, OpenFlags};
use std::{
    fs::{self, DirBuilder, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::{
        ffi::OsStrExt,
        fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
    sync::{Arc, OnceLock, atomic::AtomicBool},
};
mod capture;
pub use capture::{BlobCapture, PreparedBlob};
pub mod contributions;
mod native;
mod retained;
pub use retained::{ImmutableImport, ImportObservation, InputFormat, RetainedInput};
pub mod parser_state;
pub mod recovery;
pub mod values;
mod wal;

pub const ARCHIVE_VERSION: u32 = 2;
pub const ARCHIVE_SPEC_COMMIT: &str = "8c4e2210692282420d7a5ca91dc005b409a668f1";
pub const EMBEDDINGS_VERSION: u32 = 0;
pub const INDEX_VERSION: u32 = 19;
/// Exact owning forward-schema revision, independently qualified from inbox index8.
pub const INDEX_SQL_BLAKE3: &str =
    "194b3aa184952a2e67e911d6260b53df50669ebe922a5d2d7404332e46e42b0a";
pub const INDEX_SPEC_COMMIT: &str = "aca97f216bc3c56a81f4b2523b81744ce42c6454";
const ARCHIVE_SQL: &str = include_str!("../schema/archive.sql");
const INDEX_SQL: &str = include_str!("../schema/index.sql");
const EMBEDDINGS_SQL: &str = include_str!("../schema/embeddings.sql");
type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("controlled copy recovery refused: {0}")]
    RecoveryRefused(String),
    #[error("recovery failed; owned staging retained at {path}: {error}")]
    RecoveryStaging {
        path: PathBuf,
        #[source]
        error: Box<Error>,
    },
    #[error(
        "immutable interpretation context needs {required} framed bytes; maximum declaration is {maximum}"
    )]
    ImportContextTooLarge { required: usize, maximum: usize },
    #[error("import request {id} was reused with different content")]
    ImportRequestConflict { id: String },
    #[error(
        "normalized value chunk of {required} bytes must be split into chunks of at most {maximum} bytes"
    )]
    ValueChunkSplitRequired { required: usize, maximum: usize },
    #[error("message preparation predecessor changed")]
    MessagePredecessorChanged,
    #[error("message has an unfinished preparation; resume or retire it first")]
    MessagePreparationPending,
    #[error("normalized value predecessor changed")]
    ValuePredecessorChanged,
    #[error("normalized value is not integrity sealed")]
    ValueNotReady,
    #[error("retaining another input requires the excision-aware collector")]
    ExcisionCheckRequired,
    #[error("parser predecessor changed; discard the attempted step")]
    ParserPredecessorChanged,
    #[error("provider state has unfinished prepared work; resume it before admitting a new parse")]
    ParserWorkPending,
    #[error("{resource} requires a larger admission ({required})")]
    ParserAdmission {
        resource: &'static str,
        required: u64,
    },
    #[error("parser state invariant: {0}")]
    Invariant(String),
    #[error("filesystem: {0}")]
    Io(#[from] io::Error),
    #[error("SQLite at {file}: {source}")]
    Sqlite {
        file: PathBuf,
        #[source]
        source: rusqlite::Error,
    },
    #[error(
        "checksum failure at {file}, page {page} (offset {offset}, page size {page_size}): {source}"
    )]
    CorruptPage {
        file: PathBuf,
        page: u64,
        offset: u64,
        page_size: u32,
        #[source]
        source: rusqlite::Error,
    },
    #[error(
        "controlled recovery required at {file}: {reason} (frame {frame:?}); retained bytes were not changed"
    )]
    RecoveryRequired {
        file: PathBuf,
        reason: &'static str,
        frame: Option<u64>,
    },
    #[error("store startup cancelled")]
    Cancelled,
    #[error("archive writer already held at {archive} by {holder}")]
    WriterHeld { archive: PathBuf, holder: String },
    #[error(
        "{file} schema version {found} is newer than supported {supported}; no database was changed"
    )]
    NewerSchema {
        file: PathBuf,
        found: u32,
        supported: u32,
    },
    #[error("index at {file} requires rebuild: found version {found}, expected {expected}")]
    IndexRebuildRequired {
        file: PathBuf,
        found: u32,
        expected: u32,
    },
    #[error("unsupported or inconsistent schema at {0}")]
    Schema(PathBuf),
    #[error("SQLite/checksum initialization: {0}")]
    Initialization(String),
    #[error("blob: {0}")]
    Blob(#[from] hashed_blobs::Error),
    #[error("archive is missing while retained store content exists at {0}")]
    MissingArchive(PathBuf),
    #[error("blob is not recorded in this archive")]
    UnknownBlob,
    #[error("blob range is outside retained bytes")]
    Range,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Status {
    pub archive_version: u32,
    pub index_version: u32,
    pub embeddings_version: u32,
    pub write_counter: u64,
    pub recovery: Option<polylogue_model::RecoveryQualification>,
}

/// Acquired only by the daemon's startup path. Its lock lives as long as its connections.
/// File::try_lock uses the opened lock inode, so aliases of an archive compete.
pub struct Writer {
    root: PathBuf,
    archive: Connection,
    _index: Connection,
    _embeddings: Connection,
    _lock: Arc<File>,
}

/// Read-only SQLite handles and typed queries; never a raw connection or SQL surface.
pub struct ReadHandle {
    root: PathBuf,
    archive: Connection,
    index: Connection,
    embeddings: Connection,
    // Fields drop in declaration order: connections close before authority releases.
    _lock: Arc<File>,
}

fn initialize_sqlite() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let version = rusqlite::version_number();
        if version < 3_053_000 {
            return Err(format!("SQLite {version} is older than 3.53.0"));
        }
        if version != 3_053_002 {
            return Err(format!("cksumvfs expects SQLite 3.53.2, found {version}"));
        }
        let rc = native::register();
        if rc != rusqlite::ffi::SQLITE_OK {
            return Err(format!("register cksumvfs returned {rc}"));
        }
        Ok(())
    })
    .clone()
    .map_err(Error::Initialization)
}
fn db<T>(file: &Path, operation: impl FnOnce() -> rusqlite::Result<T>) -> Result<T> {
    native::take_checksum_fault();
    let result = operation();
    let fault = native::take_checksum_fault();
    result.map_err(|source| {
        if matches!(&source, rusqlite::Error::SqliteFailure(e, _) if e.extended_code == rusqlite::ffi::SQLITE_IOERR_DATA) {
            if let Some(fault) = fault {
                return Error::CorruptPage { file: fault.file, page: fault.page, offset: fault.offset, page_size: fault.page_size, source };
            }
        }
        Error::Sqlite { file: file.into(), source }
    })
}
fn open(file: &Path, readonly: bool, durable: bool) -> Result<Connection> {
    let flags = if readonly {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    } else {
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE
    } | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    db(file, || {
        Connection::open_with_flags_and_vfs(file, flags, if durable { "cksmvfs" } else { "unix" })
    })
}
fn inspect_immutable(file: &Path, durable: bool) -> Result<Connection> {
    // Without a WAL, immutable mode prevents SQLite creating shared-memory/journal siblings.
    // Never use this mode when retained WAL bytes may contain newer schema or content.
    let mut uri = String::from("file:");
    for byte in file.as_os_str().as_bytes() {
        if byte.is_ascii_alphanumeric() || b"/._-~".contains(byte) {
            uri.push(*byte as char);
        } else {
            uri.push_str(&format!("%{byte:02X}"));
        }
    }
    uri.push_str("?immutable=1");
    db(file, || {
        Connection::open_with_flags_and_vfs(
            uri,
            OpenFlags::SQLITE_OPEN_READ_ONLY
                | OpenFlags::SQLITE_OPEN_URI
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            if durable { "cksmvfs" } else { "unix" },
        )
    })
}
fn counter(conn: &Connection, file: &Path) -> Result<u64> {
    let text: String = db(file, || {
        conn.query_row(
            "SELECT value FROM meta WHERE key='write_counter'",
            [],
            |r| r.get(0),
        )
    })?;
    text.parse::<u64>()
        .ok()
        .filter(|n| *n <= i64::MAX as u64)
        .ok_or_else(|| Error::Schema(file.into()))
}
fn version(conn: &Connection, file: &Path) -> Result<u32> {
    db(file, || {
        conn.query_row("PRAGMA user_version", [], |row| row.get(0))
    })
}
fn sibling(file: &Path, suffix: &str) -> PathBuf {
    let mut name = file.as_os_str().to_os_string();
    name.push(suffix);
    name.into()
}
fn owned_metadata(path: &Path) -> Result<Option<fs::Metadata>> {
    match path.symlink_metadata() {
        Ok(meta) if meta.file_type().is_symlink() => Err(Error::Schema(path.into())),
        Ok(meta) => Ok(Some(meta)),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(Error::Io(err)),
    }
}
fn validate_version(
    root: &Path,
    name: &str,
    expected: u32,
    durable: bool,
    legacy: bool,
) -> Result<()> {
    let file = root.join(name);
    // Inspect retained siblings even when the main file is missing or empty.
    // A WAL may hold the only committed schema/content and must never be overwritten as fresh.
    let main = owned_metadata(&file)?;
    let wal = sibling(&file, "-wal");
    let shm = sibling(&file, "-shm");
    let wal_meta = owned_metadata(&wal)?;
    let shm_meta = owned_metadata(&shm)?;
    let has_wal = wal_meta.as_ref().is_some_and(|m| m.len() != 0);
    if main.as_ref().is_none_or(|m| m.len() == 0) {
        if has_wal || shm_meta.as_ref().is_some_and(|m| m.len() != 0) {
            return Err(wal::required(
                &file,
                "missing or empty main with retained WAL/shared-memory bytes",
                None,
            ));
        }
        return Ok(());
    }
    // No journal mode, schema, counters, chmod, or blob mutations precede this check.
    if has_wal && shm_meta.is_none() {
        return Err(wal::required(
            &wal,
            "orphan WAL requires controlled recovery",
            None,
        ));
    }
    let conn = if has_wal {
        open(&file, true, durable)?
    } else {
        inspect_immutable(&file, durable)?
    };
    let found = version(&conn, &file)?;
    if durable && found > expected {
        return Err(Error::NewerSchema {
            file,
            found,
            supported: expected,
        });
    }
    if !durable && found != expected {
        return Err(Error::IndexRebuildRequired {
            file,
            found,
            expected,
        });
    }
    if durable && found != expected && !(legacy && name == "archive.db" && found == 1) {
        return Err(Error::Schema(file));
    }
    validate_schema(
        &conn,
        &file,
        match name {
            "archive.db" => ARCHIVE_SQL,
            "embeddings.db" => EMBEDDINGS_SQL,
            _ => INDEX_SQL,
        },
    )?;
    if name != "embeddings.db" {
        counter(&conn, &file)?;
    }
    if name == "archive.db" {
        let qualification = recovery::qualification(&conn, &file)?;
        if found == 1 && qualification.is_some() {
            return Err(Error::Schema(file));
        }
    }
    if name == "index.db" {
        let stored: String = db(&file, || {
            conn.query_row(
                "SELECT value FROM meta WHERE key='INDEX_VERSION'",
                [],
                |r| r.get(0),
            )
        })?;
        let found = stored
            .parse::<u32>()
            .map_err(|_| Error::Schema(file.clone()))?;
        if found != expected {
            return Err(Error::IndexRebuildRequired {
                file,
                found,
                expected,
            });
        }
    }
    Ok(())
}
fn preflight(root: &Path) -> Result<()> {
    preflight_inner(root, false)
}
fn preflight_inner(root: &Path, legacy: bool) -> Result<()> {
    validate_version(root, "archive.db", ARCHIVE_VERSION, true, legacy)?;
    validate_version(root, "embeddings.db", EMBEDDINGS_VERSION, true, false)?;
    validate_version(root, "index.db", INDEX_VERSION, false, false)?;
    if root.try_exists()? && owned_metadata(&root.join("archive.db"))?.is_none() {
        for entry in fs::read_dir(root)? {
            let path = entry?.path();
            if path.file_name().is_some_and(|n| n == "writer.lock") {
                continue;
            }
            return Err(Error::MissingArchive(root.into()));
        }
    }
    Ok(())
}
fn private_dir(path: &Path) -> Result<()> {
    if path
        .symlink_metadata()
        .is_ok_and(|m| m.file_type().is_symlink())
    {
        return Err(Error::Schema(path.into()));
    }
    DirBuilder::new().recursive(true).mode(0o700).create(path)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}
fn initialize_db(
    root: &Path,
    name: &str,
    sql: &str,
    expected: u32,
    durable: bool,
) -> Result<Connection> {
    let path = root.join(name);
    let fresh = !path.exists() || path.metadata()?.len() == 0;
    // Pre-create private files so SQLite's default creation mode never exposes content.
    if fresh {
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&path)?;
    }
    let conn = open(&path, false, durable)?;
    if fresh && durable {
        let rc = native::reserve(&conn);
        if rc != rusqlite::ffi::SQLITE_OK {
            return Err(Error::Initialization(format!(
                "reserve checksum bytes: {rc}"
            )));
        }
    }
    db(&path, || {
        conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA secure_delete=ON;")
    })?;
    if fresh {
        db(&path, || conn.execute_batch("BEGIN IMMEDIATE"))?;
        db(&path, || conn.execute_batch(sql))?;
        if name != "embeddings.db" {
            db(&path, || {
                conn.execute(
                    "INSERT INTO meta(key,value) VALUES ('write_counter','0')",
                    [],
                )
            })?;
        }
        if name == "index.db" {
            db(&path, || {
                conn.execute("INSERT INTO meta(key,value) VALUES ('generation',lower(hex(randomblob(16)))),('parser_generation','0')", [])
            })?;
            db(&path, || {
                conn.execute(
                    "INSERT INTO meta(key,value) VALUES ('INDEX_VERSION', ?1)",
                    [INDEX_VERSION.to_string()],
                )
            })?;
        }
        db(&path, || conn.pragma_update(None, "user_version", expected))?;
        db(&path, || conn.execute_batch("COMMIT"))?;
    } else {
        // Version zero is not a license to adopt an unrelated database as an archive.
        validate_schema(&conn, &path, sql)?;
    }
    db(&path, || conn.pragma_update(None, "journal_mode", "WAL"))?;
    db(&path, || {
        conn.pragma_update(None, "synchronous", if durable { "FULL" } else { "NORMAL" })
    })?;
    private_files(root)?;
    Ok(conn)
}
fn schema(conn: &Connection) -> rusqlite::Result<Vec<(String, String, String)>> {
    let mut stmt = conn.prepare("SELECT type,name,sql FROM sqlite_schema WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type,name")?;
    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect()
}
fn validate_schema(conn: &Connection, path: &Path, sql: &str) -> Result<()> {
    let expected = db(path, || Connection::open_in_memory())?;
    db(path, || expected.execute_batch(sql))?;
    if db(path, || schema(conn))? != db(path, || schema(&expected))? {
        return Err(Error::Schema(path.into()));
    }
    if path.file_name().is_some_and(|s| s != "index.db") {
        let mut bytes = [0; 21];
        File::open(path)?.read_exact(&mut bytes)?;
        if bytes[20] != 8 {
            return Err(Error::Schema(path.into()));
        }
    }
    Ok(())
}
fn private_files(root: &Path) -> Result<()> {
    for name in [
        "archive.db",
        "index.db",
        "embeddings.db",
        "archive.db-wal",
        "archive.db-shm",
        "index.db-wal",
        "index.db-shm",
        "embeddings.db-wal",
        "embeddings.db-shm",
        "writer.lock",
    ] {
        let path = root.join(name);
        if let Ok(metadata) = path.symlink_metadata() {
            if metadata.file_type().is_symlink() {
                return Err(Error::Schema(path));
            }
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }
    }
    Ok(())
}
// A retained archive must already have its coordination inode. Missing-lock
// inspection may explain a refusal using raw headers, but never opens SQLite.
fn startup_lock(root: &Path, create: bool, writable: bool) -> Result<File> {
    let path = root.join("writer.lock");
    let lock = OpenOptions::new()
        .read(true)
        .write(writable)
        .create_new(create)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&path);
    let lock = match lock {
        Ok(lock) => lock,
        Err(e) if create && e.kind() == io::ErrorKind::AlreadyExists => {
            return startup_lock(root, false, writable);
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            for (name, supported) in [
                ("archive.db", ARCHIVE_VERSION),
                ("embeddings.db", EMBEDDINGS_VERSION),
            ] {
                let file = root.join(name);
                let Some(meta) = owned_metadata(&file)? else {
                    continue;
                };
                if meta.len() >= 100 {
                    let mut header = [0; 100];
                    File::open(&file)?.read_exact(&mut header)?;
                    if &header[..16] == b"SQLite format 3\0" {
                        let found = u32::from_be_bytes(header[60..64].try_into().unwrap());
                        if found > supported {
                            return Err(Error::NewerSchema {
                                file,
                                found,
                                supported,
                            });
                        }
                    }
                }
            }
            let mut sidecar = false;
            for entry in fs::read_dir(root)? {
                let name = entry?.file_name();
                sidecar |= name == "archive.db-wal" || name == "archive.db-shm";
            }
            if owned_metadata(&root.join("archive.db"))?.is_none() && !sidecar {
                return Err(Error::MissingArchive(root.into()));
            }
            return Err(wal::required(
                &path,
                "missing archive coordination lock",
                None,
            ));
        }
        Err(e) => return Err(Error::Io(e)),
    };
    if !lock.metadata()?.is_file() {
        return Err(Error::Schema(path));
    }
    match lock.try_lock() {
        Ok(()) => Ok(lock),
        Err(std::fs::TryLockError::WouldBlock) => Err(Error::WriterHeld {
            archive: root.into(),
            holder: fs::read_to_string(&path)
                .unwrap_or_else(|_| "unknown holder".into())
                .trim()
                .into(),
        }),
        Err(std::fs::TryLockError::Error(e)) => Err(Error::Io(e)),
    }
}
impl Writer {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_cancellable(path, &AtomicBool::new(false))
    }
    pub fn open_cancellable(path: &Path, cancelled: &AtomicBool) -> Result<Self> {
        wal::check_cancel(cancelled)?;
        initialize_sqlite()?;
        if !path.try_exists()? {
            private_dir(path)?;
        }
        let root = fs::canonicalize(path)?;
        let empty = fs::read_dir(&root)?.next().is_none();
        if empty {
            native::no_cow(&root)?;
        }
        let mut lock = startup_lock(&root, empty, true)?;
        wal::validate(&root, cancelled)?;
        let index_rebuild = recovery::begin_index_rebuild(&root, cancelled)?;
        recovery::migrate(&root, cancelled)?;
        preflight(&root)?;
        wal::check_cancel(cancelled)?;
        private_dir(&root)?;
        lock.set_len(0)?;
        writeln!(lock, "pid={}", std::process::id())?;
        lock.sync_all()?;
        let archive = initialize_db(&root, "archive.db", ARCHIVE_SQL, ARCHIVE_VERSION, true)?;
        let index = initialize_db(&root, "index.db", INDEX_SQL, INDEX_VERSION, false)?;
        let embeddings = initialize_db(
            &root,
            "embeddings.db",
            EMBEDDINGS_SQL,
            EMBEDDINGS_VERSION,
            true,
        )?;
        if let Some(proof) = index_rebuild {
            recovery::finish_index_rebuild(&root, proof, cancelled)?;
        }
        hashed_blobs::Store::open(&root, vec![], 0o600)?;
        private_dir(&root.join("blobs"))?;
        private_dir(&root.join("blobs/tmp"))?;
        private_files(&root)?;
        File::open(&root)?.sync_all()?;
        Ok(Self {
            root,
            archive,
            _index: index,
            _embeddings: embeddings,
            _lock: Arc::new(lock),
        })
    }
    pub fn read_handle(&self) -> Result<ReadHandle> {
        ReadHandle::owned(self.root.clone(), Arc::clone(&self._lock))
    }
    pub fn status(&self) -> Result<Status> {
        self.read_handle()?.status()
    }
    /// Foundation byte storage, not provider ingestion or a revision protocol.
    pub fn put_file_blob(&mut self, bytes: &[u8]) -> Result<BlobHash> {
        let proof = self
            .blob_capture()?
            .capture(bytes, &AtomicBool::new(false))?;
        self.retain_captured_blob(proof)
    }
}
impl ReadHandle {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_cancellable(path, &AtomicBool::new(false))
    }
    pub fn open_cancellable(path: &Path, cancelled: &AtomicBool) -> Result<Self> {
        wal::check_cancel(cancelled)?;
        initialize_sqlite()?;
        let root = fs::canonicalize(path)?;
        let lock = startup_lock(&root, false, false)?;
        wal::validate(&root, cancelled)?;
        if recovery::raw_versions(&root)? == Some(1) {
            return Err(wal::required(
                &root.join("archive.db"),
                "format migration requires the daemon owner",
                None,
            ));
        }
        preflight(&root)?;
        wal::check_cancel(cancelled)?;
        Self::owned(root, Arc::new(lock))
    }
    fn owned(root: PathBuf, lock: Arc<File>) -> Result<Self> {
        Ok(Self {
            archive: open(&root.join("archive.db"), true, true)?,
            index: open(&root.join("index.db"), true, false)?,
            embeddings: open(&root.join("embeddings.db"), true, true)?,
            root,
            _lock: lock,
        })
    }
    pub fn status(&self) -> Result<Status> {
        let file = self.root.join("archive.db");
        let snapshot = db(&file, || self.archive.unchecked_transaction())?;
        let write_counter = counter(&snapshot, &file)?;
        Ok(Status {
            archive_version: version(&snapshot, &file)?,
            index_version: version(&self.index, &self.root.join("index.db"))?,
            embeddings_version: version(&self.embeddings, &self.root.join("embeddings.db"))?,
            write_counter,
            recovery: recovery::qualification(&snapshot, &file)?,
        })
    }
    pub fn read_blob_range(&self, hash: BlobHash, offset: u64, length: u64) -> Result<Vec<u8>> {
        let path = self.root.join("archive.db");
        let size: Option<i64> = db(&path, || {
            self.archive
                .query_row(
                    "SELECT size FROM blob WHERE hash=?1 AND state='present'",
                    [hash.bytes().as_slice()],
                    |r| r.get(0),
                )
                .optional()
        })?;
        let size = u64::try_from(size.ok_or(Error::UnknownBlob)?)
            .map_err(|_| Error::Schema(path.clone()))?;
        if offset.checked_add(length).is_none_or(|end| end > size) {
            return Err(Error::Range);
        }
        let store = hashed_blobs::Reader::new(&self.root, vec![]);
        Ok(store.read_range(&hashed_blobs::Hash(*hash.bytes()), offset, length)?)
    }
}
use rusqlite::OptionalExtension;
#[cfg(test)]
mod tests;

pub mod prose_search;

//! Archives written by Polylogue builds before the version 1 baseline, read
//! as a declared older input format by `polylogue import --from-archive`.
//!
//! The format is the table declaration in `older_archive/archive.sql` at
//! `user_version` 1 to 4, with stored bytes either inline in `blob.data` as a
//! zstd frame or in a file `blobs/<first two hex digits>/<hash hex>.zst` of
//! zstd frames followed by a seek table. Such an archive is only ever read:
//! it is opened read-only and immutable when it has no write-ahead log, and
//! otherwise the database and its log are copied into the importing
//! archive's staging folder and the copy is read. Nothing in the older
//! archive's folder is written, renamed or deleted.

use crate::input_rows::{Extent, ObservationRow, Revision};
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub mod authored;
#[cfg(any(test, feature = "fixture"))]
pub mod fixture;

/// The older format's `archive.db` declaration, as its last build created it
/// (version 4). Earlier versions lack the tables and columns
/// [`declared_tables`] removes.
pub const ARCHIVE_SQL: &str = include_str!("older_archive/archive.sql");
/// The older format's `embeddings.db` declaration.
pub const EMBEDDINGS_SQL: &str = include_str!("older_archive/embeddings.sql");
/// The newest `user_version` an archive in the older format has.
pub const NEWEST_VERSION: u32 = 4;

/// Observation classes the older format wrote that this archive keeps as they are.
const OBSERVATION_CLASSES: [&str; 6] = [
    "first",
    "appended",
    "rewritten",
    "deleted",
    "back",
    "imported",
];

/// Tables whose rows are carried into this archive row for row, with their
/// own counts in the import report. Every other table's rows are also kept
/// verbatim in the rows input ([`OlderArchive::write_rows`]).
const BYTE_TABLES: [&str; 6] = [
    "machine",
    "blob",
    "input",
    "revision",
    "extent",
    "observation",
];

/// Tables of operational queues that have no counterpart in this archive.
/// Their rows stay readable in the rows input; the report lists them as refused.
pub const QUEUE_TABLES: [&str; 5] = [
    "chat_action",
    "chat_action_attachment",
    "chat_action_event",
    "backfill_job",
    "backfill_job_event",
];

/// Tables that record content removed on purpose. This archive has no
/// excision; their rows stay in the rows input and are reported as refused.
pub const EXCISION_TABLES: [&str; 2] = ["excision", "excised"];

/// Tables this archive recomputes for itself: the write counter and the row
/// counts kept by triggers.
pub const DERIVED_TABLES: [&str; 2] = ["meta", "row_count"];

/// One machine and a host name it was seen with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Machine {
    pub id: String,
    pub host: String,
    pub first_seen: i64,
}

/// One input with every revision, extent and observation the older archive
/// holds for it, in the form this archive stores them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Input {
    pub id: String,
    pub format: String,
    pub revisions: Vec<Revision>,
    pub extents: Vec<Extent>,
    /// In the order the older archive inserted them: the last is the input's
    /// latest observation.
    pub observations: Vec<ObservationRow>,
    /// Why the input cannot be carried, when the older archive records
    /// something about it this archive has no counterpart for.
    pub refusal: Option<String>,
    /// Observations whose location text may not be the exact path bytes:
    /// the older format wrote paths as display text.
    pub inexact_locations: u64,
}

impl Input {
    /// The blobs its extents cite, each once.
    pub fn blobs(&self) -> BTreeSet<BlobHash> {
        self.extents.iter().map(|extent| extent.blob).collect()
    }
}

/// One stored blob as the older archive describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blob {
    pub hash: BlobHash,
    pub size: u64,
    pub place: String,
    pub encoding: String,
    pub state: String,
}

/// A note, tag, title, saved query or other authored row, with what it is
/// about and how it relates to other things.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    pub id: String,
    pub first_note: String,
    pub kind: String,
    pub key: Option<String>,
    pub value: Option<String>,
    pub author: String,
    pub machine: String,
    pub created_at: i64,
    pub valid_until: Option<i64>,
    pub targets: Vec<String>,
    /// `(kind, other)` pairs from `note_relation`.
    pub relations: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Setting {
    pub id: String,
    pub machine: String,
    pub key: String,
    pub value: String,
    pub author: String,
    pub set_at: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeptResult {
    pub id: String,
    pub name: Option<String>,
    pub query: String,
    pub ran_at: i64,
    pub read_at: String,
    pub rows_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delivery {
    pub id: String,
    pub recipient_kind: String,
    pub recipient_id: Option<String>,
    pub sender: String,
    pub event: String,
    pub cwd: Option<String>,
    pub at: i64,
    pub content_hash: Vec<u8>,
    pub bytes: i64,
    /// The delivered text, zstd, as stored.
    pub body: Vec<u8>,
    pub refs_json: String,
    pub left_out_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryEvent {
    pub delivery: String,
    pub what: String,
    pub at: i64,
    pub detail: Option<String>,
}

/// One client write request. Its body is the whole request as JSON, zstd;
/// archives before version 3 kept none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub id: String,
    pub created_at: i64,
    pub body: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StoredVector {
    pub text_hash: Vec<u8>,
    pub model: String,
    pub chunk: i64,
    pub dims: i64,
    /// `dims` little-endian f32 values, as stored.
    pub vector: Vec<u8>,
    pub embedded_at: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorFailure {
    pub text_hash: Vec<u8>,
    pub model: String,
    pub error: String,
    pub failed_at: i64,
}

/// An archive in the older format, open for reading.
pub struct OlderArchive {
    root: PathBuf,
    archive: Connection,
    embeddings: Option<Connection>,
    version: u32,
    machines: Vec<Machine>,
    /// Copies of a database whose write-ahead log was not empty, removed when
    /// the reader is dropped.
    _copies: Option<tempfile::TempDir>,
    /// A shared hold on the older archive's lock file while it is read, so
    /// a daemon of the older build cannot start on it meanwhile.
    _lock: Option<std::fs::File>,
}

impl std::fmt::Debug for OlderArchive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OlderArchive")
            .field("root", &self.root)
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

fn refused(message: impl Into<String>) -> StoreError {
    StoreError::Invalid(message.into())
}

/// A path as an SQLite URI file name.
pub(crate) fn uri(path: &Path) -> String {
    let mut out = String::from("file:");
    for c in path.to_string_lossy().chars() {
        match c {
            '%' => out.push_str("%25"),
            '?' => out.push_str("%3f"),
            '#' => out.push_str("%23"),
            c => out.push(c),
        }
    }
    out
}

/// Opens one database of the older archive without writing anything next to
/// it. A database with a write-ahead log holding frames is copied with its
/// log into `copies` and the copy is opened, so its committed state is read.
pub(crate) fn open_source(
    path: &Path,
    copies: &mut Option<tempfile::TempDir>,
    scratch: &Path,
) -> Result<Connection> {
    let log = PathBuf::from(format!("{}-wal", path.display()));
    let logged = std::fs::metadata(&log).is_ok_and(|meta| meta.len() > 0);
    let conn = if logged {
        if copies.is_none() {
            std::fs::create_dir_all(scratch)?;
            *copies = Some(tempfile::tempdir_in(scratch)?);
        }
        let folder = copies.as_ref().expect("copy folder").path();
        let name = path
            .file_name()
            .ok_or_else(|| refused("a database path has no file name"))?;
        let copy = folder.join(name);
        std::fs::copy(path, &copy)?;
        std::fs::copy(&log, PathBuf::from(format!("{}-wal", copy.display())))?;
        Connection::open_with_flags(
            &copy,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?
    } else {
        Connection::open_with_flags(
            format!("{}?immutable=1", uri(path)),
            OpenFlags::SQLITE_OPEN_READ_ONLY
                | OpenFlags::SQLITE_OPEN_URI
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?
    };
    conn.pragma_update(None, "query_only", true)?;
    Ok(conn)
}

/// Each table's columns, in order, as `sql` declares them.
pub(crate) fn tables_of(conn: &Connection) -> Result<BTreeMap<String, Vec<String>>> {
    let names: Vec<String> = conn
        .prepare(
            // A virtual table's module may not be loaded here; its backing
            // tables are ordinary tables and are listed.
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
             AND sql NOT LIKE 'CREATE VIRTUAL TABLE%'",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut tables = BTreeMap::new();
    for name in names {
        let columns: Vec<String> = conn
            .prepare(&format!("PRAGMA table_info(\"{name}\")"))?
            .query_map([], |r| r.get(1))?
            .collect::<rusqlite::Result<_>>()?;
        tables.insert(name, columns);
    }
    Ok(tables)
}

/// The tables and columns an older archive at `version` has: the version 4
/// declaration without what versions 2, 3 and 4 added.
pub fn declared_tables(version: u32) -> Result<BTreeMap<String, Vec<String>>> {
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(ARCHIVE_SQL)?;
    let mut tables = tables_of(&conn)?;
    if version < 4 {
        tables.remove("row_count");
    }
    if version < 3
        && let Some(columns) = tables.get_mut("client_request")
    {
        columns.retain(|column| column != "request" && column != "commit_seq");
    }
    if version < 2 {
        tables.remove("backfill_job");
        tables.remove("backfill_job_event");
    }
    Ok(tables)
}

fn declared_embeddings() -> Result<BTreeMap<String, Vec<String>>> {
    let conn = Connection::open_in_memory()?;
    conn.execute_batch(EMBEDDINGS_SQL)?;
    tables_of(&conn)
}

/// Names the first difference between two table sets.
fn difference(
    found: &BTreeMap<String, Vec<String>>,
    declared: &BTreeMap<String, Vec<String>>,
) -> Option<String> {
    for (name, columns) in found {
        match declared.get(name) {
            None => return Some(format!("table {name} is not in the older format")),
            Some(expected) if expected != columns => {
                return Some(format!(
                    "table {name} has columns {} where the older format has {}",
                    columns.join(","),
                    expected.join(",")
                ));
            }
            Some(_) => {}
        }
    }
    declared
        .keys()
        .find(|name| !found.contains_key(*name))
        .map(|name| format!("table {name} of the older format is missing"))
}

fn hash_of(bytes: Vec<u8>) -> rusqlite::Result<BlobHash> {
    let length = bytes.len();
    let array: [u8; 32] = bytes.try_into().map_err(|_| {
        rusqlite::Error::FromSqlConversionFailure(
            length,
            rusqlite::types::Type::Blob,
            "a blob hash has 32 bytes".into(),
        )
    })?;
    Ok(BlobHash(array))
}

impl OlderArchive {
    /// Opens the older archive in `root` (the folder holding `archive.db`,
    /// `blobs/` and `embeddings.db`). `scratch` is where a database with a
    /// write-ahead log is copied: the importing archive's staging folder.
    pub fn open(root: &Path, scratch: &Path) -> Result<Self> {
        let archive_db = root.join("archive.db");
        if !archive_db.is_file() {
            return Err(refused(format!(
                "{} has no archive.db: name the folder of an archive in the older format",
                root.display()
            )));
        }
        let lock = match std::fs::File::open(root.join("lock")) {
            Ok(file) => match file.try_lock_shared() {
                Ok(()) => Some(file),
                Err(std::fs::TryLockError::WouldBlock) => {
                    return Err(refused(format!(
                        "a running Polylogue holds {}; stop it before importing",
                        root.display()
                    )));
                }
                Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let mut copies = None;
        let archive = open_source(&archive_db, &mut copies, scratch)?;
        let version = crate::schema::version(&archive)?;
        if version == 0 || version > NEWEST_VERSION {
            return Err(refused(format!(
                "{} is at version {version}, not an archive in the older format (versions 1 to {NEWEST_VERSION})",
                archive_db.display()
            )));
        }
        if let Some(problem) = difference(&tables_of(&archive)?, &declared_tables(version)?) {
            return Err(refused(format!(
                "{} is not an archive in the older format: {problem}",
                archive_db.display()
            )));
        }
        let embeddings_db = root.join("embeddings.db");
        let embeddings = if embeddings_db.is_file() {
            let conn = open_source(&embeddings_db, &mut copies, scratch)?;
            if let Some(problem) = difference(&tables_of(&conn)?, &declared_embeddings()?) {
                return Err(refused(format!(
                    "{} is not an embeddings database in the older format: {problem}",
                    embeddings_db.display()
                )));
            }
            Some(conn)
        } else {
            None
        };
        // One read transaction per database, so every read sees one state.
        archive.execute_batch("BEGIN")?;
        if let Some(conn) = &embeddings {
            conn.execute_batch("BEGIN")?;
        }
        let machines = archive
            .prepare("SELECT id, host, first_seen FROM machine ORDER BY id, first_seen, host")?
            .query_map([], |r| {
                Ok(Machine {
                    id: r.get(0)?,
                    host: r.get(1)?,
                    first_seen: r.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(Self {
            root: root.to_path_buf(),
            archive,
            embeddings,
            version,
            machines,
            _copies: copies,
            _lock: lock,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn machines(&self) -> &[Machine] {
        &self.machines
    }

    /// Rows per table of `archive.db`, and of `embeddings.db` under
    /// `embedded` and `embed_failure`.
    pub fn row_counts(&self) -> Result<BTreeMap<String, u64>> {
        let mut counts = BTreeMap::new();
        for name in tables_of(&self.archive)?.into_keys() {
            let n: i64 =
                self.archive
                    .query_row(&format!("SELECT COUNT(*) FROM \"{name}\""), [], |r| {
                        r.get(0)
                    })?;
            counts.insert(name, n as u64);
        }
        for name in ["embedded", "embed_failure"] {
            let n: i64 = match &self.embeddings {
                Some(conn) => {
                    conn.query_row(&format!("SELECT COUNT(*) FROM {name}"), [], |r| r.get(0))?
                }
                None => 0,
            };
            counts.insert(name.to_owned(), n as u64);
        }
        Ok(counts)
    }

    /// Input IDs in order, after `after`, at most `count` of them.
    pub fn input_ids(&self, after: Option<&str>, count: usize) -> Result<Vec<String>> {
        Ok(self
            .archive
            .prepare_cached("SELECT id FROM input WHERE id > ?1 ORDER BY id LIMIT ?2")?
            .query_map(params![after.unwrap_or(""), count as i64], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Groups of two or more inputs with an observation of the same file:
    /// the same machine, device and inode. The older format recorded a file
    /// reached through two paths (a linked directory) as one input per path.
    /// Each group's IDs are in order; a device and inode can also be reused
    /// by a later file, so callers compare the inputs' bytes before treating
    /// them as one.
    pub fn same_file_groups(&self) -> Result<Vec<Vec<String>>> {
        let mut statement = self.archive.prepare(
            "SELECT group_concat(input, char(31)) FROM (
               SELECT DISTINCT machine, device, inode, input FROM observation
               WHERE device IS NOT NULL AND inode IS NOT NULL
               ORDER BY machine, device, inode, input)
             GROUP BY machine, device, inode HAVING COUNT(*) > 1",
        )?;
        let rows = statement.query_map([], |r| r.get::<_, String>(0))?;
        let mut groups = Vec::new();
        for row in rows {
            let mut ids: Vec<String> = row?.split('\u{1f}').map(str::to_owned).collect();
            ids.sort();
            groups.push(ids);
        }
        Ok(groups)
    }

    /// Whether the older archive holds an input with this ID.
    pub fn has_input(&self, id: &str) -> Result<bool> {
        Ok(self
            .archive
            .prepare_cached("SELECT 1 FROM input WHERE id = ?1")?
            .exists([id])?)
    }

    fn host(&self, machine: &str) -> String {
        self.machines
            .iter()
            .filter(|m| m.id == machine)
            .min_by_key(|m| m.first_seen)
            .map_or_else(String::new, |m| m.host.clone())
    }

    /// One input with all its rows.
    pub fn input(&self, id: &str) -> Result<Input> {
        let format: String =
            self.archive
                .query_row("SELECT format FROM input WHERE id = ?1", [id], |r| r.get(0))?;
        let revisions = self
            .archive
            .prepare_cached(
                "SELECT number, produced, chain, length FROM revision WHERE input = ?1 ORDER BY number",
            )?
            .query_map([id], |r| {
                Ok(Revision {
                    number: r.get(0)?,
                    produced: r.get(1)?,
                    chain: r.get(2)?,
                    length: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let extents = self
            .archive
            .prepare_cached(
                "SELECT chain, offset, length, blob, blob_offset FROM extent WHERE input = ?1
                 ORDER BY chain, offset",
            )?
            .query_map([id], |r| {
                Ok(Extent {
                    chain: r.get(0)?,
                    offset: r.get(1)?,
                    length: r.get(2)?,
                    blob: hash_of(r.get(3)?)?,
                    blob_offset: r.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut refusal = None;
        let mut inexact_locations = 0;
        let mut observations = Vec::new();
        let mut statement = self.archive.prepare_cached(
            "SELECT o.id, o.machine, o.path, o.class, o.revision, o.excision, o.length, o.device,
                    o.inode, o.size, o.mtime_ns, o.ctime_ns, o.details_json, o.polylogue_version,
                    o.seen_at, z.zip_input
             FROM observation o LEFT JOIN observation_zip z ON z.observation_id = o.id
             WHERE o.input = ?1 ORDER BY o.rowid",
        )?;
        let mut rows = statement.query([id])?;
        while let Some(r) = rows.next()? {
            let machine: String = r.get(1)?;
            let path: String = r.get(2)?;
            let class: String = r.get(3)?;
            let excision: Option<String> = r.get(5)?;
            let container: Option<String> = r.get(15)?;
            if refusal.is_none() {
                if class == "excised" || excision.is_some() {
                    refusal = Some(
                        "its stored bytes were changed by an excision, which this archive has no counterpart for"
                            .to_owned(),
                    );
                } else if !OBSERVATION_CLASSES.contains(&class.as_str()) {
                    refusal = Some(format!(
                        "observation class {class} is not one this archive keeps"
                    ));
                } else if let Some(container) = container {
                    refusal = Some(format!(
                        "it was read as a member of export {container}, a membership the older format recorded without the member's name"
                    ));
                }
            }
            // The older format wrote paths as display text, which replaces
            // bytes that are not UTF-8 with U+FFFD; such a path no longer
            // names its location exactly.
            if refusal.is_none() && !self.machines.iter().any(|m| m.id == machine) {
                refusal = Some(format!("its machine {machine} is not in the machine table"));
            }
            let exact = !path.contains('\u{FFFD}');
            if !exact {
                inexact_locations += 1;
            }
            observations.push(ObservationRow {
                id: r.get(0)?,
                host: self.host(&machine),
                machine,
                path_bytes: exact.then(|| path.as_bytes().to_vec()),
                path,
                class,
                revision: r.get(4)?,
                chain: None,
                length: r.get(6)?,
                device: r.get(7)?,
                inode: r.get(8)?,
                size: r.get(9)?,
                mtime_ns: r.get(10)?,
                ctime_ns: r.get(11)?,
                details_json: r.get(12)?,
                polylogue_version: r.get(13)?,
                seen_at: r.get(14)?,
                container: None,
            });
        }
        Ok(Input {
            id: id.to_owned(),
            format,
            revisions,
            extents,
            observations,
            refusal,
            inexact_locations,
        })
    }

    /// Every blob hash, in order.
    pub fn blob_hashes(&self) -> Result<Vec<BlobHash>> {
        Ok(self
            .archive
            .prepare("SELECT hash FROM blob ORDER BY hash")?
            .query_map([], |r| hash_of(r.get(0)?))?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn blob(&self, hash: &BlobHash) -> Result<Option<Blob>> {
        Ok(self
            .archive
            .prepare_cached("SELECT size, place, encoding, state FROM blob WHERE hash = ?1")?
            .query_row([&hash.0[..]], |r| {
                Ok(Blob {
                    hash: *hash,
                    size: r.get::<_, i64>(0)? as u64,
                    place: r.get(1)?,
                    encoding: r.get(2)?,
                    state: r.get(3)?,
                })
            })
            .optional()?)
    }

    /// Where a file blob's bytes are.
    pub fn blob_path(&self, hash: &BlobHash) -> PathBuf {
        let hex = hash.to_string();
        self.root
            .join("blobs")
            .join(&hex[..2])
            .join(format!("{hex}.zst"))
    }

    /// Writes a blob's decoded bytes to `out`. The outer error is a failure
    /// to read or write; the inner one is why the stored bytes cannot be
    /// carried (a missing file, bytes that do not decode, or bytes whose
    /// length or BLAKE3 differ from the row).
    pub fn copy_blob(
        &self,
        blob: &Blob,
        out: &mut dyn Write,
    ) -> Result<std::result::Result<(), String>> {
        if blob.encoding != "plain" {
            return Ok(Err(format!("encoding {} is not plain zstd", blob.encoding)));
        }
        let source: Box<dyn Read> = match blob.place.as_str() {
            "inline" => {
                let data: Option<Vec<u8>> = self.archive.query_row(
                    "SELECT data FROM blob WHERE hash = ?1",
                    [&blob.hash.0[..]],
                    |r| r.get(0),
                )?;
                match data {
                    Some(data) => Box::new(std::io::Cursor::new(data)),
                    None => return Ok(Err("an inline blob has no data".into())),
                }
            }
            "file" => match std::fs::File::open(self.blob_path(&blob.hash)) {
                Ok(file) => Box::new(file),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(Err(format!(
                        "its file {} is missing",
                        self.blob_path(&blob.hash).display()
                    )));
                }
                Err(error) => return Err(error.into()),
            },
            place => return Ok(Err(format!("place {place} is neither inline nor file"))),
        };
        let mut decoder = match zstd::stream::read::Decoder::new(source) {
            Ok(decoder) => decoder,
            Err(error) => return Ok(Err(format!("its bytes do not decode: {error}"))),
        };
        let mut hasher = blake3::Hasher::new();
        let mut length = 0u64;
        let mut buffer = vec![0u8; 64 << 10];
        loop {
            let n = match decoder.read(&mut buffer) {
                Ok(n) => n,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Ok(Err(format!("its bytes do not decode: {error}"))),
            };
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
            out.write_all(&buffer[..n])?;
            length += n as u64;
        }
        if length != blob.size {
            return Ok(Err(format!(
                "it decodes to {length} bytes where its row says {}",
                blob.size
            )));
        }
        if hasher.finalize().as_bytes() != &blob.hash.0 {
            return Ok(Err("its decoded bytes do not hash to its name".into()));
        }
        Ok(Ok(()))
    }

    /// Notes with their targets and relations, in the order they were written.
    pub fn notes(&self) -> Result<Vec<Note>> {
        let mut notes: Vec<Note> = self
            .archive
            .prepare(
                "SELECT id, first_note, kind, key, value, author, machine, created_at, valid_until
                 FROM note ORDER BY commit_seq, rowid",
            )?
            .query_map([], |r| {
                Ok(Note {
                    id: r.get(0)?,
                    first_note: r.get(1)?,
                    kind: r.get(2)?,
                    key: r.get(3)?,
                    value: r.get(4)?,
                    author: r.get(5)?,
                    machine: r.get(6)?,
                    created_at: r.get(7)?,
                    valid_until: r.get(8)?,
                    targets: Vec::new(),
                    relations: Vec::new(),
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        let mut targets = self
            .archive
            .prepare_cached("SELECT ref FROM note_target WHERE note_id = ?1 ORDER BY ref")?;
        let mut relations = self.archive.prepare_cached(
            "SELECT kind, other FROM note_relation WHERE note_id = ?1 ORDER BY kind, other",
        )?;
        for note in &mut notes {
            note.targets = targets
                .query_map([&note.id], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?;
            note.relations = relations
                .query_map([&note.id], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<rusqlite::Result<_>>()?;
        }
        Ok(notes)
    }

    /// Targets and relations whose note does not exist: rows the older
    /// archive holds that no note carries.
    pub fn loose_note_rows(&self) -> Result<Vec<(String, String)>> {
        Ok(self
            .archive
            .prepare(
                "SELECT 'note_target', note_id || ' ' || ref FROM note_target
                   WHERE note_id NOT IN (SELECT id FROM note)
                 UNION ALL
                 SELECT 'note_relation', note_id || ' ' || kind || ' ' || other FROM note_relation
                   WHERE note_id NOT IN (SELECT id FROM note)
                 ORDER BY 1, 2",
            )?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Settings in the order they took effect: by `set_at`, then ID.
    pub fn settings(&self) -> Result<Vec<Setting>> {
        Ok(self
            .archive
            .prepare(
                "SELECT id, machine, key, value, author, set_at FROM setting ORDER BY set_at, id",
            )?
            .query_map([], |r| {
                Ok(Setting {
                    id: r.get(0)?,
                    machine: r.get(1)?,
                    key: r.get(2)?,
                    value: r.get(3)?,
                    author: r.get(4)?,
                    set_at: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn kept_results(&self) -> Result<Vec<KeptResult>> {
        Ok(self
            .archive
            .prepare(
                "SELECT id, name, query, ran_at, read_at, rows_json FROM kept_result ORDER BY id",
            )?
            .query_map([], |r| {
                Ok(KeptResult {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    query: r.get(2)?,
                    ran_at: r.get(3)?,
                    read_at: r.get(4)?,
                    rows_json: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn deliveries(&self) -> Result<Vec<Delivery>> {
        Ok(self
            .archive
            .prepare(
                "SELECT id, recipient_kind, recipient_id, sender, event, cwd, at, content_hash,
                        bytes, body, refs_json, left_out_json
                 FROM delivery ORDER BY commit_seq, id",
            )?
            .query_map([], |r| {
                Ok(Delivery {
                    id: r.get(0)?,
                    recipient_kind: r.get(1)?,
                    recipient_id: r.get(2)?,
                    sender: r.get(3)?,
                    event: r.get(4)?,
                    cwd: r.get(5)?,
                    at: r.get(6)?,
                    content_hash: r.get(7)?,
                    bytes: r.get(8)?,
                    body: r.get(9)?,
                    refs_json: r.get(10)?,
                    left_out_json: r.get(11)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn delivery_events(&self) -> Result<Vec<DeliveryEvent>> {
        Ok(self
            .archive
            .prepare(
                "SELECT delivery, what, at, detail FROM delivery_event ORDER BY delivery, at, what",
            )?
            .query_map([], |r| {
                Ok(DeliveryEvent {
                    delivery: r.get(0)?,
                    what: r.get(1)?,
                    at: r.get(2)?,
                    detail: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Client requests in commit order. Before version 3 none kept its body.
    pub fn requests(&self) -> Result<Vec<Request>> {
        let sql = if self.version >= 3 {
            "SELECT id, created_at, request FROM client_request ORDER BY commit_seq, id"
        } else {
            "SELECT id, created_at, NULL FROM client_request ORDER BY created_at, id"
        };
        Ok(self
            .archive
            .prepare(sql)?
            .query_map([], |r| {
                Ok(Request {
                    id: r.get(0)?,
                    created_at: r.get(1)?,
                    body: r.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Rows of the tables with no counterpart here, per table, for the report.
    pub fn uncarried(&self, tables: &[&str]) -> Result<Vec<(String, u64)>> {
        let found = tables_of(&self.archive)?;
        let mut out = Vec::new();
        for table in tables {
            if !found.contains_key(*table) {
                continue;
            }
            let n: i64 =
                self.archive
                    .query_row(&format!("SELECT COUNT(*) FROM \"{table}\""), [], |r| {
                        r.get(0)
                    })?;
            out.push(((*table).to_owned(), n as u64));
        }
        Ok(out)
    }

    /// Every stored vector, in `embedded` order.
    pub fn vectors(&self, mut visit: impl FnMut(StoredVector) -> Result<()>) -> Result<()> {
        let Some(conn) = &self.embeddings else {
            return Ok(());
        };
        let mut statement = conn.prepare(
            "SELECT text_hash, model, chunk, dims, vector, embedded_at FROM embedded ORDER BY id",
        )?;
        let mut rows = statement.query([])?;
        while let Some(r) = rows.next()? {
            visit(StoredVector {
                text_hash: r.get(0)?,
                model: r.get(1)?,
                chunk: r.get(2)?,
                dims: r.get(3)?,
                vector: r.get(4)?,
                embedded_at: r.get(5)?,
            })?;
        }
        Ok(())
    }

    pub fn vector_failures(&self) -> Result<Vec<VectorFailure>> {
        let Some(conn) = &self.embeddings else {
            return Ok(Vec::new());
        };
        Ok(conn
            .prepare(
                "SELECT text_hash, model, error, failed_at FROM embed_failure ORDER BY model, text_hash",
            )?
            .query_map([], |r| {
                Ok(VectorFailure {
                    text_hash: r.get(0)?,
                    model: r.get(1)?,
                    error: r.get(2)?,
                    failed_at: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Writes each refused input as JSON lines, each carrying the input's
    /// refusal `reason`: its `input`, `revision`, `extent`, `observation` and
    /// `observation_zip` rows, then the `blob` row of every blob it cites
    /// with the blob's stored bytes as they are (the row's `data` for an
    /// inline blob; `file_hex` for a file, or `file_missing` when there is
    /// none), so a refused input
    /// whose source is gone keeps every row and byte. Values are written as
    /// in [`Self::write_rows`].
    pub fn write_refused_inputs(
        &self,
        refused: &[(String, String)],
        out: &mut dyn Write,
    ) -> Result<u64> {
        let tables = tables_of(&self.archive)?;
        let mut written = 0;
        let mut line = |table: &str,
                        columns: &[String],
                        row: &rusqlite::Row<'_>,
                        reason: &str,
                        extra: Option<(&str, serde_json::Value)>,
                        out: &mut dyn Write|
         -> Result<()> {
            let mut values = serde_json::Map::new();
            for (index, name) in columns.iter().enumerate() {
                values.insert(name.clone(), json_value(row.get_ref(index)?));
            }
            let mut record = serde_json::json!({"table": table, "reason": reason, "row": values});
            if let Some((key, value)) = extra {
                record[key] = value;
            }
            serde_json::to_writer(&mut *out, &record)?;
            out.write_all(b"\n")?;
            written += 1;
            Ok(())
        };
        for (id, reason) in refused {
            for (table, filter) in [
                ("input", "id = ?1"),
                ("revision", "input = ?1"),
                ("extent", "input = ?1"),
                ("observation", "input = ?1"),
                (
                    "observation_zip",
                    "observation_id IN (SELECT id FROM observation WHERE input = ?1)",
                ),
            ] {
                let Some(columns) = tables.get(table) else {
                    continue;
                };
                let order = if table == "observation" {
                    "rowid".to_owned()
                } else {
                    (1..=columns.len())
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                };
                let mut statement = self.archive.prepare(&format!(
                    "SELECT * FROM \"{table}\" WHERE {filter} ORDER BY {order}"
                ))?;
                let mut rows = statement.query([id])?;
                while let Some(row) = rows.next()? {
                    line(table, columns, row, reason, None, out)?;
                }
            }
            let columns = &tables["blob"];
            let mut statement = self.archive.prepare(
                "SELECT * FROM blob WHERE hash IN (SELECT blob FROM extent WHERE input = ?1)
                 ORDER BY hash",
            )?;
            let mut rows = statement.query([id])?;
            while let Some(row) = rows.next()? {
                let place: String = row.get("place")?;
                let extra = if place != "file" {
                    None
                } else {
                    let hash = hash_of(row.get("hash")?)?;
                    match std::fs::read(self.blob_path(&hash)) {
                        Ok(bytes) => Some(("file_hex", serde_json::Value::String(hex(&bytes)))),
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                            Some(("file_missing", serde_json::Value::Bool(true)))
                        }
                        Err(error) => return Err(error.into()),
                    }
                };
                line("blob", columns, row, reason, extra, out)?;
            }
        }
        Ok(written)
    }

    /// Writes every row of every table other than the stored bytes'
    /// ([`BYTE_TABLES`]) as JSON lines, `{"table":…,"row":{column:value}}`,
    /// table by table in name order and each table's rows in the order of
    /// all its columns, so the same archive always gives the same bytes.
    /// Text that is not UTF-8 and blob values are written as hex
    /// (`{"sqlite_text_hex":…}`, `{"sqlite_blob_hex":…}`) and real numbers
    /// by their bits when JSON cannot hold them, so every value is kept
    /// exactly. These are the authored, operational and derived rows the
    /// archive held, kept whole beside their mapped forms.
    pub fn write_rows(&self, out: &mut dyn Write) -> Result<u64> {
        let mut written = 0;
        for (table, columns) in tables_of(&self.archive)? {
            if BYTE_TABLES.contains(&table.as_str()) {
                continue;
            }
            let order = (1..=columns.len())
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let mut statement = self
                .archive
                .prepare(&format!("SELECT * FROM \"{table}\" ORDER BY {order}"))?;
            let mut rows = statement.query([])?;
            while let Some(row) = rows.next()? {
                let mut values = serde_json::Map::new();
                for (index, name) in columns.iter().enumerate() {
                    values.insert(name.clone(), json_value(row.get_ref(index)?));
                }
                serde_json::to_writer(
                    &mut *out,
                    &serde_json::json!({"table": table, "row": values}),
                )?;
                out.write_all(b"\n")?;
                written += 1;
            }
        }
        Ok(written)
    }
}

/// The deliveries an archive holds, and their events as
/// `(delivery, outcome, time)`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StoredDeliveries {
    pub ids: BTreeSet<String>,
    pub events: BTreeSet<(String, String, i64)>,
}

impl crate::read::Reader<'_> {
    /// The deliveries this archive holds, and their events as
    /// `(delivery, outcome, time)`: what an older archive's import actually
    /// stored, whichever request carried it.
    pub fn stored_deliveries(&self) -> Result<StoredDeliveries> {
        let conn = self.conn()?;
        let ids = conn
            .prepare("SELECT id FROM source_delivery")?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let events = conn
            .prepare("SELECT delivery, what, at FROM source_delivery_event")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(StoredDeliveries { ids, events })
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn json_value(value: ValueRef<'_>) -> serde_json::Value {
    match value {
        ValueRef::Null => serde_json::Value::Null,
        ValueRef::Integer(n) => n.into(),
        ValueRef::Real(n) => serde_json::Number::from_f64(n)
            .map(serde_json::Value::Number)
            .unwrap_or_else(
                || serde_json::json!({"sqlite_real_bits": format!("{:016x}", n.to_bits())}),
            ),
        ValueRef::Text(bytes) => match std::str::from_utf8(bytes) {
            Ok(text) => text.into(),
            Err(_) => serde_json::json!({"sqlite_text_hex": hex(bytes)}),
        },
        ValueRef::Blob(bytes) => serde_json::json!({"sqlite_blob_hex": hex(bytes)}),
    }
}

#[cfg(test)]
mod tests;

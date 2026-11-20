//! Read-only access to Python Polylogue's source catalog. Its stored sessions
//! become inputs with their source facts and append lineage. The original
//! database and its nonempty WAL are kept as bytes. Other catalog rows are
//! left in those originals.
use crate::staging::PreparedBytes;
use crate::{Archive, FileFacts, Result, StoreError};
use rusqlite::Connection;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// Open the committed standalone observations without waiting for its index.
/// A location match is only a hint. The writer verifies the bytes and receipt.
pub fn import_locations(archive: &Archive) -> Result<impl Fn(&str, &[u8]) -> Result<bool>> {
    let conn = crate::archive::open_read(&archive.archive_db())?;
    Ok(move |machine: &str, path: &[u8]| {
        let mut statement = conn.prepare_cached(
            "SELECT EXISTS(SELECT 1 FROM observation WHERE machine=?1 AND path_bytes=?2 AND class='imported')",
        )?;
        Ok(statement.query_row(rusqlite::params![machine, path], |row| row.get(0))?)
    })
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Input {
    pub raw_id: String,
    pub origin: String,
    pub source_path: String,
    pub capture_mode: Option<String>,
    pub blob_hash: [u8; 32],
    pub blob_size: u64,
    pub revision_kind: String,
    pub previous: Option<String>,
    pub append_start: Option<u64>,
    pub append_end: Option<u64>,
    /// Python Polylogue's name for this revision of its source.
    pub source_revision: Option<String>,
    /// The revision this append continues, by that name; read when
    /// Python Polylogue recorded no previous row ID.
    pub previous_revision: Option<String>,
}

/// The exact catalog files and the session declarations read from them.
pub struct Catalog {
    pub database: PreparedBytes,
    pub wal: Option<PreparedBytes>,
    pub inputs: Vec<Input>,
}

impl Catalog {
    /// Reads session declarations and stages the exact catalog of the same
    /// stable source. SQLite opens a private pair when the WAL has frames.
    pub fn capture(root: &Path, archive: &Archive) -> Result<Option<Self>> {
        if !root.join("blob").is_dir() {
            return Ok(None);
        }
        let Some(mut database) = CatalogFile::open(root.join("source.db"))? else {
            return Ok(None);
        };
        let wal_path = root.join("source.db-wal");
        let mut wal = CatalogFile::open(wal_path.clone())?;
        let mut copies = None;
        let conn =
            crate::older_archive::open_source(&database.path, &mut copies, &archive.staging_dir())?;
        conn.execute_batch("BEGIN")?;
        let columns = crate::older_archive::tables_of(&conn)?
            .remove("raw_sessions")
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
        if !["raw_id", "origin", "source_path", "blob_hash", "blob_size"]
            .iter()
            .all(|name| columns.contains(*name))
        {
            return Err(StoreError::Invalid(
                "Python Polylogue source.db has no declared raw_sessions catalog".into(),
            ));
        }
        let inputs = Self::inputs(&conn, &columns)?;
        let database_bytes = database.prepare(archive)?;
        let wal_bytes = wal
            .as_mut()
            .filter(|file| file.facts.size > 0)
            .map(|file| file.prepare(archive))
            .transpose()?;
        database.verify()?;
        match wal {
            Some(file) => file.verify()?,
            None if CatalogFile::open(wal_path)?.is_some() => return Err(changed()),
            None => {}
        }
        Ok(Some(Self {
            database: database_bytes,
            wal: wal_bytes,
            inputs,
        }))
    }

    fn inputs(conn: &Connection, columns: &BTreeSet<String>) -> Result<Vec<Input>> {
        let optional = |name: &str, default: &str| {
            if columns.contains(name) {
                name.to_owned()
            } else {
                default.to_owned()
            }
        };
        let sql = format!(
            "SELECT raw_id,origin,source_path,blob_hash,blob_size,{},{},{},{},{},{},{} FROM raw_sessions ORDER BY raw_id",
            optional("capture_mode", "NULL"),
            optional("revision_kind", "'unknown'"),
            optional("predecessor_raw_id", "NULL"),
            optional("append_start_offset", "NULL"),
            optional("append_end_offset", "NULL"),
            optional("source_revision", "NULL"),
            optional("predecessor_source_revision", "NULL")
        );
        let mut rows = conn.prepare(&sql)?;
        let inputs = rows
            .query_map([], |row| {
                let bytes: Vec<u8> = row.get(3)?;
                let blob_hash = bytes.try_into().map_err(|bytes: Vec<u8>| {
                    rusqlite::Error::FromSqlConversionFailure(
                        bytes.len(),
                        rusqlite::types::Type::Blob,
                        "SHA256 blob hash must have 32 bytes".into(),
                    )
                })?;
                let size: i64 = row.get(4)?;
                let start: Option<i64> = row.get(8)?;
                let end: Option<i64> = row.get(9)?;
                if size < 0 || start.is_some_and(|n| n < 0) || end.is_some_and(|n| n < 0) {
                    return Err(rusqlite::Error::InvalidQuery);
                }
                Ok(Input {
                    raw_id: row.get(0)?,
                    origin: row.get(1)?,
                    source_path: row.get(2)?,
                    blob_hash,
                    blob_size: size as u64,
                    capture_mode: row.get(5)?,
                    revision_kind: row.get(6)?,
                    previous: row.get(7)?,
                    append_start: start.map(|n| n as u64),
                    append_end: end.map(|n| n as u64),
                    source_revision: row.get(10)?,
                    previous_revision: row.get(11)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(inputs)
    }
}

struct CatalogFile {
    path: PathBuf,
    file: File,
    facts: FileFacts,
}

impl CatalogFile {
    fn open(path: PathBuf) -> Result<Option<Self>> {
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        if !metadata.is_file() {
            return Err(StoreError::invalid(
                "Python Polylogue catalog files must be regular files",
            ));
        }
        let file = File::open(&path)?;
        let facts = FileFacts::of(&metadata);
        if FileFacts::of(&file.metadata()?) != facts {
            return Err(changed());
        }
        Ok(Some(Self { path, file, facts }))
    }

    fn prepare(&mut self, archive: &Archive) -> Result<PreparedBytes> {
        let bytes = archive.prepare_bytes(&mut self.file)?;
        if bytes.len() != self.facts.size {
            return Err(changed());
        }
        Ok(bytes)
    }

    fn verify(&self) -> Result<()> {
        if FileFacts::of(&self.file.metadata()?) != self.facts
            || std::fs::metadata(&self.path)
                .map(|meta| FileFacts::of(&meta))
                .ok()
                != Some(self.facts)
        {
            return Err(changed());
        }
        Ok(())
    }
}

fn changed() -> StoreError {
    StoreError::invalid("Python Polylogue catalog changed during capture")
}

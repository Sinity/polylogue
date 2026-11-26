//! The full-text files beside an index (docs/search.md: Index maintenance).
//! An index's entries are split among [`FILES`] SQLite files by text key, so
//! a build fills them on separate threads. They live in one directory under
//! the archive root, named by the index's meta `search_files`: renaming the
//! index file (building, installing, setting aside) never moves them, and the
//! name commits with the index. A directory leaves after its index file;
//! one left by a crash between the two is removed at the next open
//! ([`sweep`]).
//!
//! Every connection opened on an index attaches its files as `search_0` …
//! ([`attach`]); a build's writer does not, since its files are written by
//! the fill threads until the build ends.

use crate::ConnExt as _;
use crate::{Result, StoreError};
use rusqlite::Connection;
use std::path::{Path, PathBuf};

/// How many files hold an index's entries. Results do not depend on it.
pub(crate) const FILES: usize = 4;

/// The schema of one file.
pub(crate) const SCHEMA: &str = include_str!("../schema/search.sql");

/// The meta key naming an index's directory of full-text files.
const META: &str = "search_files";

/// The value of [`META`] for an index held in memory: its files are
/// in-memory databases attached to its one connection.
const IN_MEMORY: &str = ":memory:";

/// The file holding text `id`'s entries.
pub(crate) fn file_of(id: i64) -> usize {
    id.rem_euclid(FILES as i64) as usize
}

/// The schema name file `n` is attached under.
pub(crate) fn schema(n: usize) -> String {
    format!("search_{n}")
}

/// The archive root of an index file: index files live in the root or in
/// its rebuild's directory.
fn root_of(index: &Path) -> Option<&Path> {
    let parent = index.parent()?;
    if parent.file_name() == Some(std::ffi::OsStr::new(crate::index::file::DIR)) {
        parent.parent()
    } else {
        Some(parent)
    }
}

/// The path of the main database `conn` has open, or `None` in memory.
fn main_path(conn: &Connection) -> Option<PathBuf> {
    conn.path()
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
}

/// The directory named in the index meta of `conn`, when its main database
/// is an index.
fn named(conn: &Connection) -> Result<Option<String>> {
    if !conn.exists(
        "SELECT EXISTS(SELECT 1 FROM main.sqlite_schema WHERE type='table' AND name='meta')",
        [],
    )? {
        return Ok(None);
    }
    conn.scalar_optional("SELECT value FROM main.meta WHERE key=?1", [META])
}

/// The paths of the files of the index `conn` has open, or `None` when it
/// has none or holds them in memory.
pub(crate) fn paths(conn: &Connection) -> Result<Option<Vec<PathBuf>>> {
    let Some(name) = named(conn)? else {
        return Ok(None);
    };
    if name == IN_MEMORY {
        return Ok(None);
    }
    let index = main_path(conn).ok_or_else(|| {
        StoreError::Invalid("an index in memory names full-text files on disk".into())
    })?;
    Ok(Some(paths_in(&index, &name)?))
}

fn paths_in(index: &Path, name: &str) -> Result<Vec<PathBuf>> {
    let root = root_of(index)
        .ok_or_else(|| StoreError::Invalid("an index file has no directory".into()))?;
    let dir = root.join(name);
    Ok((0..FILES).map(|n| dir.join(format!("{n}.db"))).collect())
}

/// Names and creates the full-text files of a new index, in the transaction
/// that creates its schema. A build's files keep a rollback journal until
/// the build ends, when they switch to WAL with the index
/// ([`crate::archive::index_build::finish`]); any other index's files are in
/// WAL from the start.
pub(crate) fn create(conn: &Connection, building: bool) -> Result<()> {
    let Some(index) = main_path(conn) else {
        crate::meta_set(conn, META, IN_MEMORY)?;
        return Ok(());
    };
    let name = format!("search-{}", uuid::Uuid::now_v7().simple());
    let files = paths_in(&index, &name)?;
    let dir = files[0].parent().expect("a file in the directory");
    std::fs::create_dir(dir)?;
    for path in &files {
        let file = Connection::open(path)?;
        if !building {
            file.pragma_update(None, "journal_mode", "wal")?;
        }
        file.execute_batch(SCHEMA)?;
        drop(file);
        std::fs::File::open(path)?.sync_all()?;
    }
    std::fs::File::open(dir)?.sync_all()?;
    if let Some(root) = dir.parent() {
        std::fs::File::open(root)?.sync_all()?;
    }
    crate::meta_set(conn, META, &name)?;
    Ok(())
}

/// Whether `conn` has its index's files attached.
pub(crate) fn attached(conn: &Connection) -> Result<bool> {
    conn.exists(
        "SELECT EXISTS(SELECT 1 FROM pragma_database_list WHERE name=?1)",
        [schema(0)],
    )
}

/// Attaches the full-text files of the index `conn` has open, and the
/// functions that rank over them. Nothing happens for a database that is
/// not an index or whose files are already attached. A file that is
/// missing is an error: an index's files leave only after it.
pub(crate) fn attach(conn: &Connection) -> Result<()> {
    let Some(name) = named(conn)? else {
        return Ok(());
    };
    if attached(conn)? {
        return Ok(());
    }
    if name == IN_MEMORY {
        for n in 0..FILES {
            conn.execute("ATTACH DATABASE ':memory:' AS ?1", [schema(n)])?;
            conn.execute_batch(&in_schema(SCHEMA, &schema(n)))?;
        }
    } else {
        let index = main_path(conn)
            .ok_or_else(|| StoreError::Invalid("an index in memory names files on disk".into()))?;
        for (n, path) in paths_in(&index, &name)?.into_iter().enumerate() {
            if !path.is_file() {
                return Err(StoreError::Invalid(format!(
                    "the index's full-text file {} is missing",
                    path.display()
                )));
            }
            conn.execute(
                "ATTACH DATABASE ?1 AS ?2",
                rusqlite::params![path.to_string_lossy(), schema(n)],
            )?;
        }
    }
    super::rank::register(conn)
}

/// The schema text with its statements aimed at `schema`.
fn in_schema(sql: &str, schema: &str) -> String {
    sql.replace(
        "CREATE VIRTUAL TABLE ",
        &format!("CREATE VIRTUAL TABLE {schema}."),
    )
    .replace("CREATE TABLE ", &format!("CREATE TABLE {schema}."))
    .replace("INSERT INTO ", &format!("INSERT INTO {schema}."))
}

/// The full-text files the index file at `index` names, read without
/// attaching anything; `None` for a file that is not an index on disk.
pub(crate) fn named_on_disk(index: &Path) -> Option<Vec<PathBuf>> {
    let name = name_on_disk(index)?;
    paths_in(index, &name).ok()
}

/// Removes the directory holding `files`, once their index file is gone.
pub(crate) fn remove(files: &[PathBuf]) -> Result<()> {
    let Some(dir) = files.first().and_then(|file| file.parent()) else {
        return Ok(());
    };
    match std::fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// The directory an index file names, read without attaching anything.
fn name_on_disk(index: &Path) -> Option<String> {
    if !index.is_file() {
        return None;
    }
    let conn = Connection::open_with_flags(
        index,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;
    conn.busy_timeout(crate::archive::BUSY_TIMEOUT).ok()?;
    named(&conn).ok().flatten().filter(|name| name != IN_MEMORY)
}

/// Removes the directories of full-text files under `root` that no index
/// file names: left by a crash between removing an index and its files, or
/// by a build that failed before its schema committed. The caller holds the
/// archive lock. Index files are `index*.db` in the root (`index.db`) and the build directory's files; no other store
/// is opened. Nothing is removed when an index file cannot be read.
pub(crate) fn sweep(root: &Path) -> Result<()> {
    let mut named = std::collections::HashSet::new();
    for dir in [root.to_path_buf(), root.join(crate::index::file::DIR)] {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        for entry in entries {
            let path = entry?.path();
            let index_file = dir != root
                || path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("index"));
            if index_file && path.extension().is_some_and(|ext| ext == "db") && path.is_file() {
                match Connection::open_with_flags(
                    &path,
                    rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
                )
                .map_err(StoreError::from)
                .and_then(|conn| {
                    conn.busy_timeout(crate::archive::BUSY_TIMEOUT)?;
                    self::named(&conn)
                }) {
                    Ok(Some(name)) => {
                        named.insert(name);
                    }
                    Ok(None) => {}
                    Err(_) => return Ok(()),
                }
            }
        }
    }
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("search-") && entry.file_type()?.is_dir() && !named.contains(&name) {
            match std::fs::remove_dir_all(entry.path()) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(())
}

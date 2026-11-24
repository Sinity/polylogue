//! Python Polylogue's vector stores, the one store of the earlier Python
//! implementation `polylogue import --origin python-vectors PATH` reads,
//! since their vectors were paid for ([`vectors`]). A store is only read:
//! opened read-only and immutable, or, when its write-ahead log holds
//! frames, copied with the log into the importing archive's folder and the
//! copy read.

use crate::Result;
use rusqlite::{Connection, OpenFlags};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub mod vectors;

#[cfg(any(test, feature = "fixture"))]
pub mod fixture;

/// A path as an SQLite URI file name.
fn uri(path: &Path) -> String {
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

/// Opens one database of a store without writing anything next to it. A
/// database with a write-ahead log holding frames is copied with its log
/// into `copies` and the copy is opened, so its committed state is read.
fn open_source(
    path: &Path,
    copies: &mut Option<tempfile::TempDir>,
    scratch: &Path,
) -> Result<Connection> {
    let log = PathBuf::from(format!("{}-wal", path.display()));
    let logged = match std::fs::symlink_metadata(&log) {
        Ok(meta) if meta.is_symlink() => std::fs::metadata(&log)?.len() > 0,
        Ok(meta) => meta.len() > 0,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(error.into()),
    };
    let conn = if logged {
        if copies.is_none() {
            std::fs::create_dir_all(scratch)?;
            *copies = Some(tempfile::tempdir_in(scratch)?);
        }
        let folder = copies.as_ref().expect("copy folder").path();
        let name = path
            .file_name()
            .ok_or_else(|| polylogue_model::Reason::PathHasNoFileName {
                path: path.display().to_string(),
            })?;
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
    crate::read::limit::ReadLimit::observe_current(&conn)?;
    Ok(conn)
}

/// Each table's columns, in order, as `sql` declares them.
fn tables_of(conn: &Connection) -> Result<BTreeMap<String, Vec<String>>> {
    let names: Vec<String> = conn
        .prepare(
            // A virtual table's module may not be loaded here; its backing
            // tables are plain tables and are listed.
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

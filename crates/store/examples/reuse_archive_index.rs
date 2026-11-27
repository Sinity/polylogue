//! Offline operation on two sealed archive copies of the current format. The
//! later copy keeps its own archive and embeddings. The earlier copy must
//! contain an index completed and verified against that exact source cut.
//! Convert both cuts independently when needed and use a fresh later copy
//! without an index. This operation checks the stored prefix and installs a
//! copy of the index at its old position. The engine's one-time
//! `catch_up_archive_suffix` example reads the remaining stored chains.
//! For a release that changes reading contracts, instead open the copy
//! with `daemon --no-sources --until-idle`. Verify the finished candidate.
//!
//! cargo run -p polylogue-store --example reuse_archive_index -- <earlier> <later>

use polylogue_store::archive::{open_read, writer::Position};
use polylogue_store::index::file::{ArchiveId, IndexFiles, Installed, REUSED_FROM, Rebuilding};
use polylogue_store::{Archive, meta_get, meta_set, schema};
use rusqlite::Connection;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

#[path = "reuse_archive_index/continuation.rs"]
mod continuation;

type Error = Box<dyn std::error::Error>;

fn main() -> Result<(), Error> {
    let mut args = std::env::args_os().skip(1).peekable();
    if args.next_if(|arg| arg == "continue").is_some() {
        return continuation::run(args);
    }
    let (Some(earlier), Some(later), None) = (args.next(), args.next(), args.next()) else {
        return Err("usage: reuse_archive_index <earlier sealed copy> <later sealed copy>".into());
    };
    let (from, to) = reuse(Path::new(&earlier), Path::new(&later))?;
    println!("Installed the index at {from}; the archive is at {to}.");
    Ok(())
}

fn reuse(earlier: &Path, later: &Path) -> Result<(u64, u64), Error> {
    let (_earlier_lock, _later_lock) = (lock(earlier)?, lock(later)?);
    let source = Archive::at(earlier);
    let target = Archive::at(later);
    let (old, from) = open_archive(&source)?;
    let (new, to) = open_archive(&target)?;
    if to <= from {
        return Err(format!("later archive position {to} must exceed {from}").into());
    }
    let old_id = archive_id(&old)?;
    let new_id = archive_id(&new)?;
    checked_index(earlier, old_id, from)?;
    for name in [
        "index.db",
        "index.db-wal",
        "index.db-shm",
        "index.db-journal",
        polylogue_store::index::file::DIR,
    ] {
        match later.join(name).symlink_metadata() {
            Ok(_) => {
                return Err(format!(
                    "later archive already has {name}; use a copy without an index"
                )
                .into());
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    attach(&old, &target.archive_db())?;
    prove_record_prefix(&old, from)?;
    prove_segment_prefix(&old, earlier, later)?;
    prove_vectors(&source, &target)?;
    install_index(earlier, later, &new_id, from)?;
    Ok((from, to))
}

fn checked_index(earlier: &Path, old_id: ArchiveId, from: u64) -> Result<(), Error> {
    let index = IndexFiles::at(earlier, &old_id, Position(from))?;
    if index.rebuilding != Rebuilding::No
        || index.installed
            != (Installed::Current {
                archive: old_id,
                position: Position(from),
            })
    {
        return Err(format!("earlier archive needs an installed index at position {from}").into());
    }
    Ok(())
}

// The daemon uses this same file as its exclusive writer lock. Offline copies
// must carry it so both archives have a writer lock before any checks start.
fn lock(root: &Path) -> Result<File, Error> {
    let file = File::open(root.join("lock"))?;
    file.try_lock()
        .map_err(|error| format!("cannot lock {}: {error}", root.display()))?;
    Ok(file)
}

fn open_archive(archive: &Archive) -> Result<(Connection, u64), Error> {
    let conn = open_read(&archive.archive_db())?;
    if schema::version(&conn)? != schema::ARCHIVE_VERSION {
        return Err(format!(
            "{} needs the current archive format",
            archive.archive_db().display()
        )
        .into());
    }
    schema::ensure_archive(&conn)?;
    let position = meta_get(&conn, "write_counter")?
        .unwrap_or_else(|| "0".into())
        .parse()?;
    Ok((conn, position))
}

fn archive_id(conn: &Connection) -> Result<ArchiveId, Error> {
    Ok(ArchiveId(
        meta_get(conn, "archive_id")?.ok_or("archive has no identity")?,
    ))
}

fn attach(conn: &Connection, path: &Path) -> Result<(), Error> {
    conn.execute(
        "ATTACH DATABASE ?1 AS later",
        [path.to_str().ok_or("database path is not UTF-8")?],
    )?;
    Ok(())
}

fn prove_record_prefix(conn: &Connection, from: u64) -> Result<(), Error> {
    // Immutable capture and authored records must survive the cut. Rows with
    // a commit position cannot first appear on the wrong side of the cut.
    // The later archive supplies mutable authored state itself, including
    // completed requests, query runs, revision state, and signals.
    for (table, suffix) in [
        ("machine", ""),
        ("input", ""),
        ("blob", ""),
        ("saved_rows", ""),
        ("delivery", "new.commit_seq <= ?1"),
        ("delivery_event", "new.commit_seq <= ?1"),
        ("chat_action", "new.commit_seq <= ?1"),
        (
            "chat_action_attachment",
            "new.action IN (SELECT id FROM later.chat_action WHERE commit_seq <= ?1)",
        ),
        ("chat_action_event", "new.commit_seq <= ?1"),
        ("import_job", "new.commit_seq <= ?1"),
        ("import_job_event", "new.commit_seq <= ?1"),
        ("annotation_import", "new.commit_seq <= ?1"),
        ("signal_event", "new.commit_seq <= ?1"),
        (
            "observation_container",
            "new.observation_id IN (SELECT id FROM later.observation WHERE commit_seq <= ?1)",
        ),
        (
            "alternate_part",
            "new.alternate IN (SELECT id FROM later.alternate WHERE commit_seq <= ?1)",
        ),
        ("revision", "new.commit_seq <= ?1"),
        ("chain", "new.commit_seq <= ?1"),
        ("extent", "new.commit_seq <= ?1"),
        ("observation", "new.commit_seq <= ?1"),
        ("alternate", "new.commit_seq <= ?1"),
        ("annotation", "new.commit_seq <= ?1"),
        (
            "annotation_link",
            "new.annotation IN (SELECT id FROM later.annotation WHERE commit_seq <= ?1)",
        ),
    ] {
        compare_rows(conn, table, suffix, from)?;
    }
    Ok(())
}

fn segment_path(name: &str) -> Result<&Path, Error> {
    let path = Path::new(name);
    let mut parts = path.components();
    if !matches!((parts.next(), parts.next(), parts.next()),
        (Some(Component::Normal(dir)), Some(Component::Normal(_)), None) if dir == "segments")
    {
        return Err(format!("invalid segment name {name}").into());
    }
    Ok(path)
}

fn prove_segment_prefix(conn: &Connection, earlier: &Path, later: &Path) -> Result<(), Error> {
    let mut statement = conn.prepare(
        "SELECT name, (SELECT committed_end FROM main.segment WHERE name = s.name), committed_end FROM later.segment s",
    )?;
    let segments = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<u64>>(1)?,
            row.get::<_, u64>(2)?,
        ))
    })?;
    for segment in segments {
        let (name, old_end, new_end) = segment?;
        let path = segment_path(&name)?;
        let mut target = File::open(later.join(path))?;
        if target.metadata()?.len() != new_end {
            return Err(
                format!("later segment {name} does not end at its committed position").into(),
            );
        }
        if let Some(old_end) = old_end {
            let mut source = File::open(earlier.join(path))?;
            if source.metadata()?.len() != old_end || old_end > new_end {
                return Err(format!("segment {name} has incompatible committed lengths").into());
            }
            compare_segment(&name, &mut source, &mut target, old_end)?;
        }
    }
    let missing: bool = conn.query_row(
        "SELECT EXISTS(SELECT name FROM main.segment EXCEPT SELECT name FROM later.segment)",
        [],
        |row| row.get(0),
    )?;
    if missing {
        return Err("later archive is missing an earlier segment".into());
    }
    Ok(())
}

fn compare_rows(conn: &Connection, table: &str, suffix: &str, from: u64) -> Result<(), Error> {
    // Look up rows by their primary keys on the other side. A set operation
    // over complete rows would use a temporary copy of the largest tables.
    let columns = conn
        .prepare("SELECT name, pk FROM pragma_table_info(?1, 'main') ORDER BY cid")?
        .query_map([table], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (column, key) in columns {
        let column = format!("\"{}\"", column.replace('"', "\"\""));
        if key > 0 {
            keys.push(format!("old.{column} = new.{column}"));
        } else {
            values.push(format!("old.{column} IS new.{column}"));
        }
    }
    if keys.is_empty() {
        return Err(format!("{table} has no primary key").into());
    }
    let key = keys.join(" AND ");
    let same = keys
        .into_iter()
        .chain(values)
        .collect::<Vec<_>>()
        .join(" AND ");
    let missing: bool = conn.query_row(
        &format!(
            "SELECT EXISTS(SELECT 1 FROM main.{table} old
                    WHERE NOT EXISTS(SELECT 1 FROM later.{table} new WHERE {same}))"
        ),
        [],
        |row| row.get(0),
    )?;
    let extra = !suffix.is_empty()
        && conn.query_row(
            &format!(
                "SELECT EXISTS(SELECT 1 FROM later.{table} new
                    WHERE {suffix} AND NOT EXISTS(SELECT 1 FROM main.{table} old WHERE {key}))"
            ),
            [from],
            |row| row.get::<_, bool>(0),
        )?;
    if missing || extra {
        return Err(format!("archives differ in {table} at the earlier cut").into());
    }
    Ok(())
}

fn compare_segment(
    name: &str,
    earlier: &mut File,
    later: &mut File,
    length: u64,
) -> Result<(), Error> {
    let mut before = vec![0; 1 << 20];
    let mut after = vec![0; before.len()];
    let mut offset = 0;
    while offset < length {
        let n = (length - offset).min(before.len() as u64) as usize;
        earlier.read_exact(&mut before[..n])?;
        later.read_exact(&mut after[..n])?;
        if before[..n] != after[..n] {
            return Err(format!("segment {name} differs at or after byte {offset}").into());
        }
        offset += n as u64;
    }
    Ok(())
}

fn prove_vectors(earlier: &Archive, later: &Archive) -> Result<(), Error> {
    if !earlier.embeddings_db().try_exists()? {
        return Ok(());
    }
    let conn = open_read(&earlier.embeddings_db())?;
    if !later.embeddings_db().try_exists()? {
        return Err("later archive is missing embeddings.db".into());
    }
    attach(&conn, &later.embeddings_db())?;
    for table in ["embedded", "query_vector"] {
        compare_rows(&conn, table, "", 0)?;
    }
    Ok(())
}

fn install_index(
    earlier: &Path,
    later: &Path,
    identity: &ArchiveId,
    from: u64,
) -> Result<(), Error> {
    let previous = open_read(&earlier.join("index.db"))?;
    let pending: bool =
        previous.query_row("SELECT EXISTS(SELECT 1 FROM search_pending)", [], |row| {
            row.get(0)
        })?;
    if pending {
        return Err("earlier index still has text to index".into());
    }
    let name = meta_get(&previous, "search_files")?.ok_or("index names no search files")?;
    if !name
        .strip_prefix("search-")
        .is_some_and(|id| id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()))
    {
        return Err("index names an invalid search directory".into());
    }
    drop(previous);
    // A fresh directory name lets a retry proceed if an interrupted install
    // published only its search files. The daemon sweeps unreferenced files.
    let search_name = format!("search-{}", uuid::Uuid::now_v7().simple());
    let search = later.join(&search_name);
    let scratch = Archive::at(later).staging_dir();
    fs::create_dir_all(&scratch)?;
    let staging = tempfile::tempdir_in(&scratch)?;
    let stage_search = staging.path().join(&search_name);
    fs::create_dir(&stage_search)?;
    for n in 0..4 {
        let file = format!("{n}.db");
        let to = stage_search.join(&file);
        copy_database(&earlier.join(&name).join(file), &to)?;
        sync_database(&Connection::open(&to)?, &to)?;
    }
    File::open(&stage_search)?.sync_all()?;

    let stage_index = staging.path().join("index.db");
    copy_database(&earlier.join("index.db"), &stage_index)?;
    let mut index = Connection::open(&stage_index)?;
    index.pragma_update(None, "synchronous", "FULL")?;
    let tx = index.transaction()?;
    meta_set(&tx, "archive_id", &identity.0)?;
    meta_set(&tx, REUSED_FROM, from.to_string())?;
    meta_set(&tx, "search_files", &search_name)?;
    // The daemon establishes a fresh cursor scope on its next open.
    tx.execute(
        "DELETE FROM meta WHERE key IN ('boot_id', 'index_instance', 'cursor_key')",
        [],
    )?;
    tx.commit()?;
    sync_database(&index, &stage_index)?;
    drop(index);
    let staged = IndexFiles::at(staging.path(), identity, Position(from))?;
    if staged.installed
        != (Installed::Current {
            archive: identity.clone(),
            position: Position(from),
        })
    {
        return Err("staged index did not open at the expected position".into());
    }
    // The index becomes visible only after its search files are durable.
    let root = File::open(later)?;
    fs::rename(stage_search, &search)?;
    root.sync_all()?;
    fs::hard_link(stage_index, later.join("index.db"))?;
    root.sync_all()?;
    Ok(())
}

fn copy_database(from: &Path, to: &Path) -> Result<(), Error> {
    copy_file(from, to)?;
    let sidecar = |file: &Path, suffix: &str| -> PathBuf {
        let mut name = OsString::from(file);
        name.push(suffix);
        name.into()
    };
    let wal = sidecar(from, "-wal");
    if wal.try_exists()? {
        copy_file(&wal, &sidecar(to, "-wal"))?;
    }
    if sidecar(from, "-journal").try_exists()? {
        return Err(format!("{} has a rollback journal", from.display()).into());
    }
    Ok(())
}

fn copy_file(from: &Path, to: &Path) -> Result<(), Error> {
    // Btrfs clones only when the files agree on whether writes use copy-on-write.
    // A new file normally inherits the directory's flag, which can differ from
    // a native archive's database or WAL.
    if let (Some(source), Some(parent)) = (no_cow(from), to.parent().and_then(no_cow))
        && source != parent
    {
        File::create_new(to)?;
        let _ = std::process::Command::new("chattr")
            .arg(if source { "+C" } else { "-C" })
            .arg("--")
            .arg(to)
            .output();
    }
    // SQLite writes only into the staged copy.
    if !std::process::Command::new("cp")
        .arg("--reflink=auto")
        .arg("--")
        .arg(from)
        .arg(to)
        .status()?
        .success()
    {
        return Err(format!("could not copy {}", from.display()).into());
    }
    Ok(())
}

fn no_cow(path: &Path) -> Option<bool> {
    let result = std::process::Command::new("lsattr")
        .args(["-d", "--"])
        .arg(path)
        .output()
        .ok()?;
    if !result.status.success() {
        return None;
    }
    result
        .stdout
        .split(|byte| byte.is_ascii_whitespace())
        .next()
        .map(|flags| flags.contains(&b'C'))
}

fn sync_database(conn: &Connection, path: &Path) -> Result<(), Error> {
    let busy: i64 = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))?;
    if busy != 0 {
        return Err(format!("could not seal {}", path.display()).into());
    }
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
#[path = "reuse_archive_index/tests.rs"]
mod tests;

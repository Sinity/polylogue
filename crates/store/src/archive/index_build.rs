//! Building `index.db` in a private file. Every standalone build, a first
//! build or one replacing an index from another schema, writes
//! `index-build/index.db` with no journal and no sync. Only the build
//! pipeline opens that file: the writer, its resolver and parser-context
//! reads, the engine's planning reads and tool-database catalog reads. When
//! the build ends the file is synced, closed, switched to WAL and renamed
//! over `index.db`; serving readers then open the new file.
//!
//! Crash states, each handled by [`prepare`] at the next open:
//! - `built.db` present: the build was finished, synced and switched to WAL
//!   but not yet installed, and it is installed now;
//! - `unsynced` present: the build file may hold unsynced pages, so it is
//!   discarded and the build starts again from the stored bytes;
//! - the build file without `unsynced`: a clean stop synced it, or it was
//!   never written unsynced, and the build resumes in it.
//!
//! An index that is replaced is moved into the build directory as
//! `previous.db` when the build starts. While its schema identity is this
//! build's, serving readers keep reading it until the rename; otherwise reads
//! answer [`StoreError::IndexBuilding`].

use crate::{Result, StoreError, archive::Archive, schema};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// The directory beside `index.db` that holds a build in progress.
pub const DIR: &str = "index-build";

/// What a refused read and `status` report while an index is being built.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IndexBuild {
    /// Whether reads are answered from the index being replaced meanwhile.
    pub serving_previous: bool,
    /// Stored input chains, and how many of them the build has read at
    /// their current reading version or found without a parser.
    pub chains: u64,
    pub chains_read: u64,
}

impl std::fmt::Display for IndexBuild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the index is being built: {} of {} input chains read",
            self.chains_read, self.chains
        )
    }
}

/// The files of a build, under the archive root.
pub(crate) struct Paths {
    pub dir: PathBuf,
    pub build: PathBuf,
    /// The finished build, synced and in WAL, until it is renamed into place.
    pub built: PathBuf,
    pub unsynced: PathBuf,
    pub previous: PathBuf,
    pub index: PathBuf,
}

impl Paths {
    pub(crate) fn of(root: &Path) -> Self {
        let dir = root.join(DIR);
        Self {
            build: dir.join("index.db"),
            built: dir.join("built.db"),
            unsynced: dir.join("unsynced"),
            previous: dir.join("previous.db"),
            index: root.join("index.db"),
            dir,
        }
    }
}

type Release = dyn Fn() + Send + Sync;

/// The build state of one archive in this process, shared by every
/// [`Archive`] value for the same root.
#[derive(Default)]
pub(crate) struct State {
    inner: Mutex<Inner>,
    /// Pipeline read pools release their idle connections to the build file
    /// before it is renamed.
    releases: Mutex<Vec<Weak<Release>>>,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State").finish_non_exhaustive()
    }
}

#[derive(Default, Clone)]
struct Inner {
    building: bool,
    serving_previous: bool,
    chains: u64,
    chains_read: u64,
}

pub(crate) fn state(root: &Path) -> Arc<State> {
    static ARCHIVES: OnceLock<Mutex<std::collections::HashMap<PathBuf, Weak<State>>>> =
        OnceLock::new();
    let key = crate::read::scope::root_key(root);
    let mut archives = ARCHIVES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    archives.retain(|_, state| state.strong_count() > 0);
    if let Some(state) = archives.get(&key).and_then(Weak::upgrade) {
        return state;
    }
    let state = Arc::new(State::default());
    archives.insert(key, Arc::downgrade(&state));
    state
}

impl State {
    fn inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn building(&self) -> bool {
        self.inner().building
    }

    pub(crate) fn progress(&self) -> Option<IndexBuild> {
        let inner = self.inner();
        inner.building.then(|| IndexBuild {
            serving_previous: inner.serving_previous,
            chains: inner.chains,
            chains_read: inner.chains_read,
        })
    }

    pub(crate) fn record_progress(&self, chains: u64, chains_read: u64) {
        let mut inner = self.inner();
        inner.chains = chains;
        inner.chains_read = chains_read;
    }

    fn set(&self, building: bool, serving_previous: bool) {
        let mut inner = self.inner();
        inner.building = building;
        inner.serving_previous = building && serving_previous;
    }

    /// `release` runs before the build file is renamed; dropping the
    /// returned handle unregisters it.
    pub(crate) fn on_release(&self, release: Arc<Release>) {
        let mut releases = self
            .releases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        releases.retain(|release| release.strong_count() > 0);
        releases.push(Arc::downgrade(&release));
    }

    fn release(&self) {
        let releases: Vec<_> = self
            .releases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        for release in releases {
            release();
        }
    }
}

impl Archive {
    /// The index file serving readers open, or the build's progress when
    /// no index can be served until it ends.
    pub(crate) fn served_index(&self) -> Result<PathBuf> {
        let paths = Paths::of(&self.root);
        match self.index_build.progress() {
            None => Ok(paths.index),
            Some(build) if build.serving_previous => Ok(paths.previous),
            Some(build) => Err(StoreError::IndexBuilding(Box::new(build))),
        }
    }

    /// The build in progress, if `index.db` is being built.
    pub fn index_build(&self) -> Option<IndexBuild> {
        self.index_build.progress()
    }
}

/// Opens the standalone index at archive open. The caller holds the archive
/// lock, and nothing else has an index file open.
pub(crate) fn prepare(archive: &Archive) -> Result<()> {
    let paths = Paths::of(&archive.root);
    if paths.built.exists() {
        install(archive)?;
    } else if paths.unsynced.exists() {
        discard(&paths)?;
    }
    let mut replaced = None;
    if paths.index.exists() {
        let index =
            crate::archive::open_runtime_writer(&paths.index, crate::archive::cache::SETUP)?;
        match archive.ensure_index(&index, false) {
            Ok(()) => {
                // A built index is current; anything in the build directory
                // is left from before it was installed.
                remove_build_files(&paths)?;
                archive.index_build.set(false, false);
                crate::archive::generation::open(archive, &index)?;
                crate::fault::check("index.after_creation", "")?;
                return Ok(());
            }
            Err(error) => {
                let Some(reason) = crate::archive::index_rebuild::reason(&error) else {
                    return Err(error);
                };
                let previous = schema::index_identity(&index)?;
                drop(index);
                set_aside(&paths)?;
                crate::fault::check("index.after_set_aside", &paths.index.to_string_lossy())?;
                replaced = Some((reason, previous));
            }
        }
    }
    if !paths.dir.exists() {
        fs::create_dir(&paths.dir)?;
        crate::archive::no_cow(&paths.dir);
        sync_directory(&archive.root)?;
    }
    let mut build = None;
    if paths.build.exists() {
        let conn = open_setup(&paths.build)?;
        match archive.ensure_index(&conn, true) {
            Ok(()) => build = Some(conn),
            // A build file from another binary is discarded like any other
            // derived file this build cannot open.
            Err(error) if crate::archive::index_rebuild::reason(&error).is_some() => {}
            Err(error) => return Err(error),
        }
    }
    let build = if let Some(build) = build {
        build
    } else {
        remove(&paths.build)?;
        let build = open_setup(&paths.build)?;
        archive.ensure_index(&build, true)?;
        if let Some((reason, previous)) = replaced {
            crate::archive::index_rebuild::record(
                &build,
                reason,
                previous,
                schema::build_index_identity(false)?,
            )?;
        }
        sync_directory(&paths.dir)?;
        build
    };
    let serving_previous = paths.previous.exists() && servable(&paths.previous)?;
    archive.index_build.set(true, serving_previous);
    crate::archive::generation::open(archive, &build)?;
    crate::fault::check("index.after_creation", "")?;
    Ok(())
}

/// Whether `previous` can be read by this build: its schema identity is
/// this build's (indexes may differ).
fn servable(previous: &Path) -> Result<bool> {
    let conn = crate::archive::open_read(previous)?;
    Ok(schema::index_identity(&conn)? == schema::build_index_identity(false)?)
}

/// A connection to the build file for schema setup: a rollback journal on
/// disk with every commit synced, so setup needs no `unsynced` marker.
fn open_setup(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_URI,
    )?;
    conn.pragma_update(None, "journal_mode", "delete")?;
    conn.pragma_update(None, "synchronous", "full")?;
    conn.busy_timeout(crate::archive::BUSY_TIMEOUT)?;
    Ok(conn)
}

/// The writer's connection to the build file. `unsynced` is created and
/// synced before this connection may write anything: its commits do not
/// sync and its rollback journal is in memory, so a lost process or boot can
/// leave the file torn. A failed transaction still rolls back in process.
pub(crate) fn open_writer(archive: &Archive) -> Result<Connection> {
    let paths = Paths::of(&archive.root);
    mark_unsynced(&paths)?;
    let conn = Connection::open_with_flags(
        &paths.build,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_URI,
    )?;
    conn.pragma_update(Some("main"), "journal_mode", "memory")?;
    conn.pragma_update(Some("main"), "synchronous", "off")?;
    conn.pragma_update(None, "temp_store", "memory")?;
    conn.busy_timeout(crate::archive::BUSY_TIMEOUT)?;
    crate::archive::writer_settings(&conn, crate::archive::cache::INDEX_WRITER)?;
    crate::fault::check("index.build.after_marker", &archive.root.to_string_lossy())?;
    Ok(conn)
}

fn mark_unsynced(paths: &Paths) -> Result<()> {
    File::create(&paths.unsynced)?.sync_all()?;
    sync_directory(&paths.dir)
}

/// A clean stop during a build: what the writer committed reaches the disk,
/// then `unsynced` goes, so the next start resumes in the build file. The
/// writer has no transaction open.
pub(crate) fn keep_progress(archive: &Archive) -> Result<()> {
    let paths = Paths::of(&archive.root);
    File::open(&paths.build)?.sync_all()?;
    crate::fault::check(
        "index.build.after_stop_sync",
        &archive.root.to_string_lossy(),
    )?;
    remove(&paths.unsynced)?;
    sync_directory(&paths.dir)
}

/// Ends a build whose rows, search and indexes have committed: syncs the
/// build file, switches it to WAL while nothing has it open, names it
/// `built.db`, and renames that over `index.db`. The writer has closed its connection; pipeline pools
/// release theirs here. An idle connection holds no lock in a rollback
/// journal mode, and a pool drops one opened through the build path at its
/// next checkout.
pub(crate) fn finish(archive: &Archive) -> Result<()> {
    let paths = Paths::of(&archive.root);
    archive.index_build.release();
    File::open(&paths.build)?.sync_all()?;
    {
        let conn = open_setup(&paths.build)?;
        let mode: String = conn.query_row("PRAGMA main.journal_mode = wal", [], |r| r.get(0))?;
        if mode != "wal" {
            return Err(StoreError::Invalid(format!(
                "the built index stayed in journal mode {mode}"
            )));
        }
    }
    // Closing the last connection checkpointed and removed the log; a log
    // left with frames in it would hold part of the index.
    for suffix in ["-wal", "-shm"] {
        let file = with_suffix(&paths.build, suffix);
        match fs::metadata(&file) {
            Ok(metadata) if suffix == "-wal" && metadata.len() > 0 => {
                return Err(StoreError::Invalid(format!(
                    "the built index left {} bytes in its write-ahead log",
                    metadata.len()
                )));
            }
            Ok(_) => remove(&file)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    File::open(&paths.build)?.sync_all()?;
    crate::fault::check(
        "index.build.after_finish_sync",
        &archive.root.to_string_lossy(),
    )?;
    fs::rename(&paths.build, &paths.built)?;
    sync_directory(&paths.dir)?;
    remove(&paths.unsynced)?;
    sync_directory(&paths.dir)?;
    crate::fault::check("index.build.before_rename", &archive.root.to_string_lossy())?;
    install(archive)
}

/// Renames `built.db` over `index.db`, then removes the index it replaced. A serving reader of `previous.db` keeps its open file until it
/// ends; new readers open `index.db`.
fn install(archive: &Archive) -> Result<()> {
    let paths = Paths::of(&archive.root);
    // `index.db` was moved aside when the build started, so a log beside it
    // is left from that file and must not be read with the new one.
    for suffix in ["-wal", "-shm"] {
        remove(&with_suffix(&paths.index, suffix))?;
    }
    fs::rename(&paths.built, &paths.index)?;
    sync_directory(&archive.root)?;
    archive.index_build.set(false, false);
    remove_build_files(&paths)
}

/// Moves `index.db` and its write-ahead log into the build directory as
/// `previous.db`. Its shared-memory index is rebuilt from the log by the
/// first reader.
fn set_aside(paths: &Paths) -> Result<()> {
    if !paths.dir.exists() {
        fs::create_dir(&paths.dir)?;
        crate::archive::no_cow(&paths.dir);
    }
    remove(&with_suffix(&paths.index, "-shm"))?;
    for suffix in ["-shm", "-wal", ""] {
        remove(&with_suffix(&paths.previous, suffix))?;
    }
    for suffix in ["-wal", ""] {
        match fs::rename(
            with_suffix(&paths.index, suffix),
            with_suffix(&paths.previous, suffix),
        ) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    sync_directory(&paths.dir)?;
    sync_directory(paths.index.parent().unwrap_or(Path::new(".")))
}

/// Discards a build a crash may have torn. The previous index stays.
fn discard(paths: &Paths) -> Result<()> {
    for suffix in ["-journal", "-wal", "-shm", ""] {
        remove(&with_suffix(&paths.build, suffix))?;
    }
    remove(&paths.unsynced)?;
    sync_directory(&paths.dir)
}

fn remove_build_files(paths: &Paths) -> Result<()> {
    if !paths.dir.exists() {
        return Ok(());
    }
    for path in [&paths.build, &paths.built, &paths.previous] {
        for suffix in ["-journal", "-wal", "-shm", ""] {
            remove(&with_suffix(path, suffix))?;
        }
    }
    remove(&paths.unsynced)?;
    match fs::remove_dir(&paths.dir) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn remove(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn sync_directory(dir: &Path) -> Result<()> {
    File::open(dir)?.sync_all()?;
    Ok(())
}

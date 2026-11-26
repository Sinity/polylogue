//! Building `index.db` in a private file. Every standalone build, a first
//! build or one replacing an index from another schema, writes
//! `index-build/index.db` with no transaction syncs: the engine prepares
//! each chain's reading into a run file under `index-build/runs/`, and the
//! writer loads the runs, composes every session and creates the indexes
//! and search entries ([`crate::writer::build`]). Only the build pipeline
//! opens the build file: the writer, the engine's planning reads and
//! tool-database catalog reads. When the build ends the file is synced,
//! closed, switched to WAL and renamed over `index.db`; serving readers then
//! open the new file. Pipeline readers use `built.db` after its first
//! rename. Path metadata and both renames share the build-state lock, so
//! each selected path still names its file. A new archive's index is built
//! the same way: its first pass stores the files its sources find and the
//! build reads them; with nothing to read, the empty build installs at the
//! end of that pass.
//!
//! Crash states, each handled by [`prepare`] at the next open:
//! - `built.db` present: the build was finished, synced and switched to WAL
//!   but not yet installed, and it is installed now;
//! - otherwise a build file or runs: a build that did not finish, discarded
//!   with its runs; the build starts again from the stored bytes, and the
//!   reason the index is replaced is read again from `previous.db`.
//!
//! An index that is replaced is moved into the build directory as
//! `previous.db` when the build starts. While its schema identity is this
//! build's, serving readers keep reading it until the rename; otherwise reads
//! answer [`StoreError::IndexBuilding`].

use crate::ConnExt as _;
use crate::{Result, StoreError, archive::Archive, schema};
use polylogue_model::LockExt;
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
    /// Stored chains of formats with a reading, counted before the build
    /// plans them, and how many of them it has read or found unreadable.
    pub chains: u64,
    pub chains_read: u64,
    /// Newest readable chains' bytes and their committed parser positions.
    pub bytes: u64,
    pub bytes_read: u64,
}

impl std::fmt::Display for IndexBuild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the index is being built: {} of {} bytes parsed; {} of {} input chains read",
            self.bytes_read, self.bytes, self.chains_read, self.chains
        )
    }
}

/// The files of a build, under the archive root.
pub(crate) struct Paths {
    pub dir: PathBuf,
    pub build: PathBuf,
    /// The finished build, synced and in WAL, until it is renamed into place.
    pub built: PathBuf,
    /// The run files a build prepares its chains' readings into.
    pub runs: PathBuf,
    pub previous: PathBuf,
    pub index: PathBuf,
}

impl Paths {
    pub(crate) fn of(root: &Path) -> Self {
        let dir = root.join(DIR);
        Self {
            build: dir.join("index.db"),
            built: dir.join("built.db"),
            runs: dir.join("runs"),
            previous: dir.join("previous.db"),
            index: root.join("index.db"),
            dir,
        }
    }
}

/// The index file a pool reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reads {
    /// The file the build pipeline writes: the private build file while a
    /// build runs, otherwise `index.db`.
    Pipeline,
    /// The file client-facing reads are served from; while a build runs and
    /// no previous index can be served, reads answer
    /// [`StoreError::IndexBuilding`].
    Serving,
}

#[derive(Clone, Copy)]
enum BuildFile {
    Writing,
    Ready,
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
    file: Option<BuildFile>,
    serving_previous: bool,
    chains: u64,
    chains_read: u64,
    bytes: u64,
    bytes_read: u64,
    /// The build's steps, counts and samples.
    measured: super::build_progress::Record,
}

pub(crate) fn state(root: &Path) -> Arc<State> {
    static ARCHIVES: OnceLock<Mutex<std::collections::HashMap<PathBuf, Weak<State>>>> =
        OnceLock::new();
    let key = crate::read::scope::root_key(root);
    let mut archives = ARCHIVES.get_or_init(Default::default).locked();
    archives.retain(|_, state| state.strong_count() > 0);
    if let Some(state) = archives.get(&key).and_then(Weak::upgrade) {
        return state;
    }
    // A handle on an archive no other handle in this process has open, such
    // as one taken without the archive lock, finds a build in progress on
    // disk; the archive's open sets the state from what it prepares.
    let state = Arc::new(State::default());
    let paths = Paths::of(root);
    let file = if paths.built.exists() {
        Some(BuildFile::Ready)
    } else if paths.build.exists() {
        Some(BuildFile::Writing)
    } else {
        None
    };
    state.set(file, false);
    archives.insert(key, Arc::downgrade(&state));
    state
}

impl Inner {
    fn progress(&self) -> Option<IndexBuild> {
        self.file.map(|_| IndexBuild {
            serving_previous: self.serving_previous,
            chains: self.chains,
            chains_read: self.chains_read,
            bytes: self.bytes,
            bytes_read: self.bytes_read,
        })
    }

    fn pipeline_index<'a>(&self, paths: &'a Paths) -> &'a Path {
        match self.file {
            Some(BuildFile::Writing) => &paths.build,
            Some(BuildFile::Ready) => &paths.built,
            None => &paths.index,
        }
    }

    fn served_index<'a>(&self, paths: &'a Paths) -> Result<&'a Path> {
        match self.progress() {
            None => Ok(&paths.index),
            Some(build) if build.serving_previous => Ok(&paths.previous),
            Some(build) => Err(StoreError::IndexBuilding(Box::new(build))),
        }
    }
}

impl State {
    fn inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.locked()
    }

    pub(crate) fn building(&self) -> bool {
        self.inner().file.is_some()
    }

    pub(crate) fn progress(&self) -> Option<IndexBuild> {
        self.inner().progress()
    }

    /// A chain read was a chain due, so the total is never below the count:
    /// the two arrive from counters read at different moments while the
    /// planner is still adding to one of them.
    pub(crate) fn record_progress(&self, chains: u64, chains_read: u64) {
        let mut inner = self.inner();
        inner.chains = chains.max(chains_read);
        inner.chains_read = chains_read;
    }

    fn set(&self, file: Option<BuildFile>, serving_previous: bool) {
        let mut inner = self.inner();
        if inner.file.is_some() != file.is_some() {
            inner.measured = Default::default();
        }
        inner.file = file;
        inner.serving_previous = file.is_some() && serving_previous;
    }

    /// `release` runs before the build file is renamed; dropping the
    /// returned handle unregisters it.
    pub(crate) fn on_release(&self, release: Arc<Release>) {
        let mut releases = self.releases.locked();
        releases.retain(|release| release.strong_count() > 0);
        releases.push(Arc::downgrade(&release));
    }

    fn release(&self) {
        let releases: Vec<_> = self
            .releases
            .locked()
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        for release in releases {
            release();
        }
    }
}

impl Archive {
    /// The index file the build pipeline (the writer, its resolver and
    /// parser-context reads, the engine's planning reads, tool-database
    /// catalog reads) opens: `index-build/index.db` while a build
    /// writes, then `index-build/built.db` until it is installed as `index.db`
    /// ([`crate::archive::index_build`]). Client-facing readers open
    /// [`Archive::served_index`] instead.
    pub fn index_db(&self) -> PathBuf {
        self.index_build
            .inner()
            .pipeline_index(&Paths::of(&self.root))
            .to_path_buf()
    }

    /// Read the pipeline index, permitting native journal recovery only for
    /// the private file this archive is currently building. query_only still
    /// rejects application writes and leaves SQLite free to recover its journal.
    pub(crate) fn open_index_read(&self) -> Result<Connection> {
        let conn = self.open_index_connection(&self.index_db(), Reads::Pipeline)?;
        conn.pragma_update(None, "query_only", true)?;
        Ok(conn)
    }

    /// An internal read connection with TEMP writes available to hydration
    /// and tool-database comparisons. Its facade owns application-write refusal.
    pub(crate) fn open_index_connection(&self, path: &Path, reads: Reads) -> Result<Connection> {
        let recover = {
            let paths = Paths::of(&self.root);
            let inner = self.index_build.inner();
            reads == Reads::Pipeline && inner.file.is_some() && path == inner.pipeline_index(&paths)
        };
        // No build-state lock is held while SQLite waits for the writer.
        super::open_read_connection(path, recover)
    }

    /// Resolve and inspect a read path while installation cannot rename it.
    pub(crate) fn with_index_path<T>(
        &self,
        reads: Reads,
        read: impl FnOnce(&Path) -> Result<T>,
    ) -> Result<T> {
        let paths = Paths::of(&self.root);
        let inner = self.index_build.inner();
        let path = match reads {
            Reads::Pipeline => inner.pipeline_index(&paths),
            Reads::Serving => inner.served_index(&paths)?,
        };
        read(path)
    }

    /// The index file serving readers open, or the build's progress when
    /// no index can be served until it ends.
    pub(crate) fn served_index(&self) -> Result<PathBuf> {
        self.with_index_path(Reads::Serving, |path| Ok(path.to_path_buf()))
    }

    /// Records how many chains a build has found due a reading and how
    /// many of them it has read.
    pub fn record_build_progress(&self, chains: u64, chains_read: u64) {
        self.index_build.record_progress(chains, chains_read);
    }

    /// Record byte progress from newest chains' stored parser positions.
    pub fn record_build_bytes(&self, bytes: u64, bytes_read: u64) {
        let mut inner = self.index_build.inner();
        inner.bytes = bytes;
        inner.bytes_read = bytes_read.min(bytes);
    }

    /// Advance byte progress only after a range's write is acknowledged.
    pub fn advance_build_bytes(&self, bytes: u64) {
        let mut inner = self.index_build.inner();
        if inner.file.is_some() {
            inner.bytes_read = inner.bytes_read.saturating_add(bytes).min(inner.bytes);
        }
    }

    /// The build in progress, if `index.db` is being built.
    pub fn index_build(&self) -> Option<IndexBuild> {
        self.index_build.progress()
    }

    /// Records the steps the build in progress runs after the one it is
    /// in, in order, for `status` to list those still to come.
    pub fn record_build_plan(&self, steps: &[&str]) {
        let mut inner = self.index_build.inner();
        if inner.file.is_some() {
            inner.measured.plan(steps);
        }
    }

    /// Records that the build in progress started `step` at `since`.
    pub fn record_build_step(&self, step: &str, since: polylogue_model::Timestamp) {
        let mut inner = self.index_build.inner();
        if inner.file.is_some() {
            inner.measured.step(step, since);
        }
    }

    /// Records how much of its work `step` of the build has done, and
    /// samples it with the writer's busy time.
    pub fn record_build_count(&self, step: &str, count: super::build_progress::StepCount) {
        let mut inner = self.index_build.inner();
        if inner.file.is_some() {
            inner.measured.count(
                std::time::Instant::now(),
                step,
                count,
                crate::timing::writer_seconds,
            );
        }
    }

    /// Records each parse worker's CPU seconds over the last minute.
    pub fn record_build_workers(&self, cpu_seconds_per_minute: Vec<f64>) {
        let mut inner = self.index_build.inner();
        if inner.file.is_some() {
            inner
                .measured
                .workers(std::time::Instant::now(), cpu_seconds_per_minute);
        }
    }

    /// The steps, rate and estimates of the build in progress.
    pub fn build_progress(&self) -> Option<super::build_progress::BuildProgress> {
        let inner = self.index_build.inner();
        inner.file?;
        Some(
            inner
                .measured
                .report(std::time::Instant::now(), crate::timing::writer_seconds()),
        )
    }
}

/// Opens the standalone index at archive open. The caller holds the archive
/// lock, and nothing else has an index file open.
pub(crate) fn prepare(archive: &Archive) -> Result<()> {
    let paths = Paths::of(&archive.root);
    if paths.built.exists() {
        install(archive)?;
    } else if paths.dir.exists() {
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
                archive.index_build.set(None, false);
                crate::fault::check("index.after_creation", "")?;
                return crate::archive::generation::open(archive, &index);
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
    } else if paths.previous.exists() {
        // A build that did not finish set this index aside; the reason it
        // was replaced still holds for the build that starts again.
        replaced = crate::archive::index_rebuild::set_aside_reason(&crate::archive::open_read(
            &paths.previous,
        )?)?;
    }
    if create_build_dir(&paths)? {
        sync_directory(&archive.root)?;
    }
    let build = open_setup(&paths.build)?;
    archive.ensure_index(&build, true)?;
    if let Some((reason, previous)) = &replaced {
        crate::archive::index_rebuild::record(
            &build,
            reason.clone(),
            previous.clone(),
            schema::build_index_identity()?,
        )?;
    }
    sync_directory(&paths.dir)?;
    let serving_previous = paths.previous.exists() && servable(&paths.previous)?;
    archive
        .index_build
        .set(Some(BuildFile::Writing), serving_previous);
    archive
        .index_build
        .inner()
        .measured
        .reason(replaced.map(|(reason, _)| reason));
    crate::fault::check("index.after_creation", "")?;
    crate::archive::generation::open(archive, &build)
}

/// Whether `previous` can be read by this build: its schema identity is
/// this build's (indexes may differ).
fn servable(previous: &Path) -> Result<bool> {
    let conn = crate::archive::open_read(previous)?;
    Ok(schema::index_identity(&conn)? == schema::build_index_identity()?)
}

/// A connection to the build file for schema setup: a rollback journal on
/// disk with every commit synced.
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

/// The writer's connection to the build file. Its commits do not sync: a
/// lost process or boot discards the file at the next open, and the build
/// starts again from the stored bytes. A failed transaction still rolls
/// back in process.
pub(crate) fn open_writer(archive: &Archive) -> Result<Connection> {
    let paths = Paths::of(&archive.root);
    let conn = Connection::open_with_flags(
        &paths.build,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_URI,
    )?;
    conn.pragma_update(Some("main"), "journal_mode", "persist")?;
    conn.pragma_update(Some("main"), "synchronous", "off")?;
    conn.pragma_update(None, "temp_store", "file")?;
    conn.busy_timeout(crate::archive::BUSY_TIMEOUT)?;
    crate::archive::writer_settings(&conn, archive.memory_budget().build_cache_mib())?;
    crate::archive::enforce_references(&conn)?;
    Ok(conn)
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
    let search = {
        let conn = open_setup(&paths.build)?;
        crate::search::files::paths(&conn)?.unwrap_or_default()
    };
    for file in search.iter().chain([&paths.build]) {
        to_wal(file)?;
    }
    if let Some(dir) = search.first().and_then(|file| file.parent()) {
        sync_directory(dir)?;
    }
    crate::fault::check(
        "index.build.after_finish_sync",
        &archive.root.to_string_lossy(),
    )?;
    {
        let mut inner = archive.index_build.inner();
        fs::rename(&paths.build, &paths.built)?;
        inner.file = Some(BuildFile::Ready);
        sync_directory(&paths.dir)?;
    }
    crate::fault::check("index.build.before_rename", &archive.root.to_string_lossy())?;
    install(archive)
}

/// Syncs a finished build's database file and switches it to WAL while
/// nothing has it open.
fn to_wal(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    {
        let conn = open_setup(path)?;
        let mode: String = conn.scalar("PRAGMA main.journal_mode = wal", [])?;
        if mode != "wal" {
            return Err(StoreError::Invalid(format!(
                "the built index stayed in journal mode {mode}"
            )));
        }
    }
    // Closing the last connection checkpointed and removed the log; a log
    // left with frames in it would hold part of the index.
    for suffix in ["-wal", "-shm"] {
        let file = with_suffix(path, suffix);
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
    File::open(path)?.sync_all()?;
    Ok(())
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
    {
        let mut inner = archive.index_build.inner();
        fs::rename(&paths.built, &paths.index)?;
        inner.file = None;
        inner.serving_previous = false;
        sync_directory(&archive.root)?;
    }
    remove_build_files(&paths)
}

/// Moves `index.db` and its write-ahead log into the build directory as
/// `previous.db`. Its shared-memory index is rebuilt from the log by the
/// first reader.
fn set_aside(paths: &Paths) -> Result<()> {
    create_build_dir(paths)?;
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

/// Creates the build directory when absent; whether it was created.
fn create_build_dir(paths: &Paths) -> Result<bool> {
    if paths.dir.exists() {
        return Ok(false);
    }
    fs::create_dir(&paths.dir)?;
    crate::archive::no_cow(&paths.dir);
    Ok(true)
}

/// Removes a SQLite database file with its journal, log and shared-memory
/// siblings, then the full-text files it names.
fn remove_database(path: &Path) -> Result<()> {
    let search = crate::search::files::named_on_disk(path);
    for suffix in ["-journal", "-wal", "-shm", ""] {
        remove(&with_suffix(path, suffix))?;
    }
    if let Some(search) = search {
        crate::search::files::remove(&search)?;
    }
    Ok(())
}

/// Discards a build that did not finish, with its runs. The previous index
/// stays.
fn discard(paths: &Paths) -> Result<()> {
    remove_database(&paths.build)?;
    remove_runs(paths)?;
    sync_directory(&paths.dir)
}

/// Discards a build that failed in this process with `error` and creates
/// its file again, empty, as [`prepare`] does after a crash, keeping the
/// record of why the index is rebuilt; `status` shows the failure until the
/// build ends. The writer has closed its connection; pipeline pools
/// release their idle ones, and a pool drops any other at its next use, the
/// file being a new one.
pub(crate) fn restart(archive: &Archive, error: &str) -> Result<()> {
    let paths = Paths::of(&archive.root);
    archive.index_build.release();
    archive
        .index_build
        .inner()
        .measured
        .failed(error.to_owned());
    let rebuild = crate::meta_get(
        &open_setup(&paths.build)?,
        crate::archive::index_rebuild::META_KEY,
    )?;
    discard(&paths)?;
    let build = open_setup(&paths.build)?;
    archive.ensure_index(&build, true)?;
    if let Some(record) = rebuild {
        crate::meta_set(&build, crate::archive::index_rebuild::META_KEY, record)?;
    }
    sync_directory(&paths.dir)?;
    crate::archive::generation::open(archive, &build)
}

/// Removes one run file and the text map beside it.
pub(crate) fn remove_run(path: &Path) -> Result<()> {
    remove(path)?;
    remove(&path.with_extension("texts"))
}

/// Removes the run files of a build.
pub(crate) fn remove_runs(paths: &Paths) -> Result<()> {
    match fs::remove_dir_all(&paths.runs) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn remove_build_files(paths: &Paths) -> Result<()> {
    if !paths.dir.exists() {
        return Ok(());
    }
    for path in [&paths.build, &paths.built, &paths.previous] {
        remove_database(path)?;
    }
    remove_runs(paths)?;
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

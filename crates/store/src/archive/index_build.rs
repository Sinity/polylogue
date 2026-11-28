//! Rebuilding `index.db` in a private file, and which index file each reader
//! opens. A bulk rebuild writes `index-rebuild/build.db` with no
//! transaction syncs, through its own connection ([`crate::index::rebuild`]).
//! Besides the rebuild, only the rebuild pipeline's reads open the build
//! file: the engine's planning reads and tool-database catalog reads. When
//! the rebuild ends the file is synced, closed, switched to WAL and renamed
//! `ready.db` ([`ready`]), and the index writer, stopped for it, renames it
//! over `index.db` ([`install`]); serving readers then open the new file. Path selection and both renames share the state lock, so
//! each selected path still names its file. A serving daemon with no
//! usable index installs an empty ordinary schema at open. Ordinary
//! capture and replay fill it as they commit.
//!
//! The files and their states are [`crate::index::file`]'s. This module
//! keeps, per archive and process, the files' state as the archive's holder
//! last stepped it (a process that does not hold the archive derives it from
//! disk on each use) and the rebuild's progress for `status`.
//!
//! While a rebuild runs, serving readers read `index.db` when this build can
//! read it ([`Installed::Current`]); otherwise reads answer
//! [`StoreError::IndexBuilding`] with the rebuild's reason and progress.

use crate::ConnExt as _;
use crate::index::file::{
    self, IndexFiles, IndexRebuild, Installed, Paths, RebuildReason, Rebuilding, Transition,
};
use crate::{Result, StoreError, archive::Archive};
use polylogue_model::LockExt;
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// What a refused read and `status` report while an index is being rebuilt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IndexBuild {
    /// Why the index is rebuilt, or why no index can be read.
    pub reason: RebuildReason,
    /// Stored chains of formats with a reading, counted before the rebuild
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
            "the index is being built ({}): {} of {} bytes parsed; {} of {} input chains read",
            self.reason, self.bytes_read, self.bytes, self.chains_read, self.chains
        )
    }
}

/// The index files and what the process knows of them beyond the files:
/// what `status` shows of the index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IndexStatus {
    #[serde(flatten)]
    pub files: IndexFiles,
    /// The last attempt to install a finished rebuild failed; the writer has
    /// no index connection and tries again at the next commit event, and the
    /// next start installs it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install_failure: Option<polylogue_model::Failure>,
    /// The stages the rebuild in progress has ended, with their records.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stages: Vec<crate::index::rebuild::StageRecord>,
}

/// The index file a pool reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reads {
    /// The file the rebuild pipeline writes: the private build file while a
    /// rebuild runs, otherwise `index.db`.
    Pipeline,
    /// The file client-facing reads are served from; while no index can be
    /// read, reads answer [`StoreError::IndexBuilding`].
    Serving,
}

type Release = dyn Fn() + Send + Sync;

/// The index state of one archive in this process, shared by every
/// [`Archive`] value for the same root.
pub(crate) struct State {
    root: PathBuf,
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

#[derive(Default)]
struct Inner {
    /// The files as the archive's holder in this process last stepped them;
    /// `None` in a process that does not hold the archive, which derives
    /// them from disk on each use.
    files: Option<IndexFiles>,
    install_failure: Option<polylogue_model::Failure>,
    chains: u64,
    chains_read: u64,
    bytes: u64,
    bytes_read: u64,
    /// The rebuild's steps, counts and samples.
    measured: super::build_progress::Record,
    /// The records of the rebuild's stages that have ended.
    stages: Vec<crate::index::rebuild::StageRecord>,
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
    let state = Arc::new(State {
        root: root.to_path_buf(),
        inner: Mutex::default(),
        releases: Mutex::default(),
    });
    archives.insert(key, Arc::downgrade(&state));
    state
}

/// A held lock on the state, with the files it describes.
struct Held<'a> {
    inner: std::sync::MutexGuard<'a, Inner>,
    files: IndexFiles,
}

impl Held<'_> {
    fn progress(&self) -> Option<IndexBuild> {
        let Rebuilding::Writing { reason } = &self.files.rebuilding else {
            return None;
        };
        Some(self.build(reason.clone()))
    }

    fn build(&self, reason: RebuildReason) -> IndexBuild {
        IndexBuild {
            reason,
            chains: self.inner.chains,
            chains_read: self.inner.chains_read,
            bytes: self.inner.bytes,
            bytes_read: self.inner.bytes_read,
        }
    }

    fn pipeline_index(&self, paths: &Paths) -> PathBuf {
        match self.files.rebuilding {
            Rebuilding::Writing { .. } => paths.build.clone(),
            Rebuilding::Ready => paths.ready.clone(),
            Rebuilding::No => paths.index.clone(),
        }
    }

    fn served_index(&self, paths: &Paths) -> Result<PathBuf> {
        if self.files.serves() {
            return Ok(paths.index.clone());
        }
        let reason = self.files.reason().unwrap_or(RebuildReason::NoIndex);
        Err(StoreError::IndexBuilding(Box::new(self.build(reason))))
    }
}

impl State {
    fn held(&self) -> Result<Held<'_>> {
        let inner = self.inner.locked();
        let files = match &inner.files {
            Some(files) => files.clone(),
            None => IndexFiles::read(&self.root)?,
        };
        Ok(Held { inner, files })
    }

    /// Whether a rebuild is being written.
    pub(crate) fn building(&self) -> bool {
        self.held()
            .is_ok_and(|held| matches!(held.files.rebuilding, Rebuilding::Writing { .. }))
    }

    /// Sets the files as the archive's open found or made them.
    /// The counts and measurements start again with the open.
    fn opened(&self, files: IndexFiles) {
        *self.inner.locked() = Inner {
            files: Some(files),
            ..Inner::default()
        };
    }

    /// Moves the files to their next state. A rebuild that starts, or starts
    /// again, starts its counts and measurements again.
    pub(crate) fn step(&self, to: Transition) -> Result<()> {
        let mut inner = self.inner.locked();
        let starting = matches!(to, Transition::Start(_));
        let installing = matches!(to, Transition::Install);
        let files = inner.files.as_mut().ok_or_else(|| {
            StoreError::internal("only the archive's holder changes its index files")
        })?;
        files.step(&self.root, to)?;
        if starting {
            let files = inner.files.take();
            *inner = Inner {
                files,
                ..Inner::default()
            };
        }
        if installing {
            inner.install_failure = None;
        }
        Ok(())
    }

    /// The last attempt to install the finished rebuild failed with `failure`.
    pub(crate) fn install_failed(&self, failure: polylogue_model::Failure) {
        self.inner.locked().install_failure = Some(failure);
    }

    /// A chain read was a chain due, so the total is never below the count:
    /// the two arrive from counters read at different moments while the
    /// planner is still adding to one of them.
    pub(crate) fn record_progress(&self, chains: u64, chains_read: u64) {
        let mut inner = self.inner.locked();
        inner.chains = chains.max(chains_read);
        inner.chains_read = chains_read;
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

    /// Runs `record` on the measurements while a rebuild is being written.
    fn measure(&self, record: impl FnOnce(&mut super::build_progress::Record)) {
        if let Ok(mut held) = self.held()
            && matches!(held.files.rebuilding, Rebuilding::Writing { .. })
        {
            record(&mut held.inner.measured);
        }
    }
}

impl Archive {
    /// The index file the rebuild pipeline (the writer, its resolver and
    /// parser-context reads, the engine's planning reads, tool-database
    /// catalog reads) opens: `index-rebuild/build.db` while a rebuild
    /// writes, then `index-rebuild/ready.db` until it is installed as
    /// `index.db` ([`crate::index::file`]). Client-facing readers open
    /// [`Archive::served_index`] instead.
    pub fn index_db(&self) -> PathBuf {
        let paths = Paths::of(&self.root);
        match self.index_build.held() {
            Ok(held) => held.pipeline_index(&paths),
            Err(_) => paths.index,
        }
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
        let recover = reads == Reads::Pipeline && path == Paths::of(&self.root).build;
        // No state lock is held while SQLite waits for the writer.
        super::open_read_connection(path, recover)
    }

    /// Resolve and inspect a read path while an install cannot rename it.
    pub(crate) fn with_index_path<T>(
        &self,
        reads: Reads,
        read: impl FnOnce(&Path) -> Result<T>,
    ) -> Result<T> {
        let paths = Paths::of(&self.root);
        let held = self.index_build.held()?;
        let path = match reads {
            Reads::Pipeline => held.pipeline_index(&paths),
            Reads::Serving => held.served_index(&paths)?,
        };
        read(&path)
    }

    /// The index file serving readers open, or the rebuild's reason and
    /// progress when no index can be read until it is installed.
    pub(crate) fn served_index(&self) -> Result<PathBuf> {
        self.with_index_path(Reads::Serving, |path| Ok(path.to_path_buf()))
    }

    /// Which index files exist and what they mean.
    pub fn index_files(&self) -> Result<IndexFiles> {
        Ok(self.index_build.held()?.files)
    }

    /// The index files, with a failed install's error.
    pub fn index_status(&self) -> Result<IndexStatus> {
        let held = self.index_build.held()?;
        Ok(IndexStatus {
            install_failure: held.inner.install_failure.clone(),
            stages: if matches!(held.files.rebuilding, Rebuilding::No) {
                Vec::new()
            } else {
                held.inner.stages.clone()
            },
            files: held.files,
        })
    }

    /// Records how many chains a rebuild has found due a reading and how
    /// many of them it has read.
    pub fn record_build_progress(&self, chains: u64, chains_read: u64) {
        self.index_build.record_progress(chains, chains_read);
    }

    /// Record byte progress from newest chains' stored parser positions.
    pub fn record_build_bytes(&self, bytes: u64, bytes_read: u64) {
        let mut inner = self.index_build.inner.locked();
        inner.bytes = bytes;
        inner.bytes_read = bytes_read.min(bytes);
    }

    /// Advance byte progress only after a range's write is acknowledged.
    pub fn advance_build_bytes(&self, bytes: u64) {
        if let Ok(mut held) = self.index_build.held()
            && held.progress().is_some()
        {
            held.inner.bytes_read = held
                .inner
                .bytes_read
                .saturating_add(bytes)
                .min(held.inner.bytes);
        }
    }

    /// The rebuild being written, if any, with its reason and counts.
    pub fn index_build(&self) -> Option<IndexBuild> {
        self.index_build.held().ok()?.progress()
    }

    /// Records the steps the rebuild in progress runs after the one it is
    /// in, in order, for `status` to list those still to come.
    pub fn record_build_plan(&self, steps: &[&str]) {
        self.index_build.measure(|measured| measured.plan(steps));
    }

    /// Records that the rebuild in progress started `step` at `since`.
    pub fn record_build_step(&self, step: &str, since: polylogue_model::Timestamp) {
        self.index_build
            .measure(|measured| measured.step(step, since));
    }

    /// Records how much of its work `step` of the rebuild has done, and
    /// samples it with the writer's busy time.
    pub fn record_build_count(&self, step: &str, count: super::build_progress::StepCount) {
        self.index_build.measure(|measured| {
            measured.count(
                std::time::Instant::now(),
                step,
                count,
                crate::timing::writer_seconds,
            )
        });
    }

    /// Keeps the record of a rebuild stage that ended, for `status`.
    pub(crate) fn record_stage(&self, record: crate::index::rebuild::StageRecord) {
        if let Ok(mut held) = self.index_build.held()
            && !matches!(held.files.rebuilding, Rebuilding::No)
        {
            held.inner.stages.push(record);
        }
    }

    /// Records each parse worker's CPU seconds over the last minute.
    pub fn record_build_workers(&self, cpu_seconds_per_minute: Vec<f64>) {
        self.index_build.measure(|measured| {
            measured.workers(std::time::Instant::now(), cpu_seconds_per_minute)
        });
    }

    /// The rebuild in progress with its steps, rate and estimates, read
    /// under one lock so a rebuild that starts or ends meanwhile never gives
    /// one without the other.
    pub fn build_progress(&self) -> Option<(IndexBuild, super::build_progress::BuildProgress)> {
        let held = self.index_build.held().ok()?;
        let build = held.progress()?;
        Some((
            build,
            held.inner
                .measured
                .report(std::time::Instant::now(), crate::timing::writer_seconds()),
        ))
    }
}

/// Opens the standalone index at archive open. The caller holds the archive
/// lock, and nothing else in this process has an index file open. A finished
/// rebuild is installed only if it belongs to this record. With no usable
/// index, a serving daemon installs the ordinary schema for catch-up.
pub(crate) fn prepare(archive: &Archive, install_for_catch_up: bool) -> Result<()> {
    let root = &archive.root;
    let paths = Paths::of(root);
    let (id, position) =
        file::archive_identity(&crate::archive::open_read(&archive.archive_db())?)?;
    let mut files = IndexFiles::at(root, &id, position)?;
    match files.rebuilding {
        Rebuilding::Ready if file::installed(&paths.ready, Some((&id, position)))?.is_current() => {
            files.step(root, Transition::Install)?;
        }
        Rebuilding::Ready => files.step(root, Transition::Discard)?,
        // A directory with no reason is left from a crash before a
        // rebuild's reason was written, or after its install.
        Rebuilding::No if paths.dir.exists() => files.step(root, Transition::Discard)?,
        Rebuilding::No | Rebuilding::Writing { .. } => {}
    }
    if install_for_catch_up && !files.serves() {
        let record = IndexRebuild::new(
            files.reason().unwrap_or(RebuildReason::NoIndex),
            crate::archive::open_read(&paths.index)
                .and_then(|index| crate::schema::index_identity(&index))
                .ok(),
        )?;
        files.step(root, Transition::Start(record.clone()))?;
        {
            let index = open_setup(&paths.build)?;
            archive.ensure_index(&index, false)?;
            file::record_start(&index, &record, &id)?;
        }
        to_wal(&paths.build)?;
        files.step(root, Transition::Ready)?;
        files.step(root, Transition::Install)?;
    }
    let record = match (&files.installed, &files.rebuilding) {
        (_, Rebuilding::Writing { .. }) => file::rebuild_record(root)?
            .ok_or_else(|| StoreError::internal("a rebuild is being written with no reason"))?,
        (Installed::Current { .. }, _) => {
            let index =
                crate::archive::open_runtime_writer(&paths.index, crate::archive::cache::SETUP)?;
            match archive.ensure_index(&index, false) {
                Ok(()) => {
                    archive.index_build.opened(files);
                    crate::fault::check("index.after_creation", "")?;
                    return crate::archive::generation::open(archive, &index);
                }
                Err(error) => {
                    let Some(reason) = RebuildReason::of(&error) else {
                        return Err(error);
                    };
                    IndexRebuild::new(reason, crate::schema::index_identity(&index).ok())?
                }
            }
        }
        (Installed::None, _) => IndexRebuild::new(RebuildReason::NoIndex, None)?,
        (Installed::Unusable { why }, _) => {
            let previous = crate::archive::open_read(&paths.index)
                .and_then(|index| crate::schema::index_identity(&index))
                .ok();
            IndexRebuild::new(why.clone(), previous)?
        }
    };
    files.step(root, Transition::Start(record.clone()))?;
    crate::fault::check("index.after_reason", &root.to_string_lossy())?;
    let build = create_build_file(archive, &record, &id)?;
    archive.index_build.opened(files);
    crate::fault::check("index.after_creation", "")?;
    crate::archive::generation::open(archive, &build)
}

/// Creates the empty build file of a rebuild that has started, with the
/// rebuild's record and the archive it is built from in its meta.
fn create_build_file(
    archive: &Archive,
    record: &IndexRebuild,
    id: &file::ArchiveId,
) -> Result<Connection> {
    let paths = Paths::of(&archive.root);
    let build = open_setup(&paths.build)?;
    archive.ensure_index(&build, true)?;
    file::record_start(&build, record, id)?;
    file::sync_directory(&paths.dir)?;
    Ok(build)
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

/// The rebuild's connection to the build file. Its commits do not sync: a
/// lost process or boot discards the file at the next open, and the rebuild
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

/// Ends a rebuild whose rows, search and indexes have committed: syncs the
/// build file and its full-text files, switches them to WAL while nothing
/// has them open, and renames the build file `ready.db`. The rebuild has
/// closed its connection; pipeline pools release theirs here. An idle
/// connection holds no lock in a rollback journal mode, and a pool drops one
/// opened through the build path at its next checkout.
pub(crate) fn ready(archive: &Archive) -> Result<()> {
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
        file::sync_directory(dir)?;
    }
    crate::fault::check(
        "index.build.after_finish_sync",
        &archive.root.to_string_lossy(),
    )?;
    archive.index_build.step(Transition::Ready)
}

/// Renames a ready rebuild over `index.db`. The index writer has closed its
/// connection to the index it replaces; readers of that index keep their
/// open files.
pub(crate) fn install(archive: &Archive) -> Result<()> {
    archive.index_build.release();
    crate::fault::check("index.build.before_rename", &archive.root.to_string_lossy())?;
    archive.index_build.step(Transition::Install)
}

/// Syncs a finished rebuild's database file and switches it to WAL while
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
        let log = file::with_suffix(path, suffix);
        match std::fs::metadata(&log) {
            Ok(metadata) if suffix == "-wal" && metadata.len() > 0 => {
                return Err(StoreError::Invalid(format!(
                    "the built index left {} bytes in its write-ahead log",
                    metadata.len()
                )));
            }
            Ok(_) => file::remove(&log)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    File::open(path)?.sync_all()?;
    Ok(())
}

/// Discards a rebuild that failed in this process with `error` and starts
/// it again for the same reason, its file created again, empty, as
/// [`prepare`] does after a crash; `status` shows the failure until the
/// rebuild ends. The rebuild has closed its connection; pipeline pools
/// release their idle ones, and a pool drops any other at its next use, the
/// file being a new one.
pub(crate) fn restart(archive: &Archive, error: &str) -> Result<()> {
    archive.index_build.release();
    let record = file::rebuild_record(&archive.root)?
        .ok_or_else(|| StoreError::internal("a rebuild is being written with no reason"))?;
    archive
        .index_build
        .step(Transition::Start(record.clone()))?;
    archive
        .index_build
        .inner
        .locked()
        .measured
        .failed(error.to_owned());
    let (id, _) = file::archive_identity(&crate::archive::open_read(&archive.archive_db())?)?;
    let build = create_build_file(archive, &record, &id)?;
    crate::archive::generation::open(archive, &build)
}

/// Removes one sorted file and the text map beside it.
pub(crate) fn remove_run(path: &Path) -> Result<()> {
    file::remove(path)?;
    file::remove(&path.with_extension("texts"))
}

/// Removes the sorted files of a rebuild.
pub(crate) fn remove_runs(paths: &Paths) -> Result<()> {
    file::remove_tree(&paths.sorted)
}

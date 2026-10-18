//! Discovery and capture of Codex rollouts, prompt history and the
//! thread-history database under the configured home. Catalog rows add rollout
//! paths; they never hide rollouts found on disk.
use crate::{Error, ImportObservation, InputFormat, ReadHandle, StoredSource, Writer};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};
use std::{io::Read, os::unix::fs::OpenOptionsExt};
const ENTRIES: usize = 512;
const FILE_BYTES: u64 = 16 * 1024 * 1024;
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub home: PathBuf,
    pub machine: String,
    pub host: String,
    #[serde(default = "rollouts")]
    pub rollouts: PathBuf,
    #[serde(default = "history")]
    pub history: PathBuf,
    #[serde(default = "catalog")]
    pub catalog: PathBuf,
    #[serde(default)]
    pub paginated_history: Option<PathBuf>,
    #[serde(default = "history_interval")]
    pub paginated_interval_seconds: u64,
}
fn history_interval() -> u64 {
    300
}
pub fn paginated_path(config: &Config) -> crate::Result<Option<PathBuf>> {
    if config.paginated_interval_seconds == 0 || config.paginated_interval_seconds > 86400 {
        return Err(refused("paginated_interval_seconds requires 1..86400"));
    }
    config
        .paginated_history
        .as_deref()
        .map(|p| resolve(&fs::canonicalize(&config.home)?, p))
        .transpose()
}
fn rollouts() -> PathBuf {
    "sessions".into()
}
fn history() -> PathBuf {
    "history.jsonl".into()
}
fn catalog() -> PathBuf {
    "state_5.sqlite".into()
}
#[derive(Clone, Debug, Serialize)]
pub struct Availability {
    pub path: Option<PathBuf>,
    pub path_key: String,
    pub count_unit: &'static str,
    pub state: &'static str,
    /// Set only after a complete traversal; `None` while unknown.
    pub entries: Option<usize>,
    pub reason: Option<String>,
    pub reason_truncated: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct Acquisition {
    pub recipe: &'static str,
    pub scope: &'static str,
    pub catalog: Availability,
    pub rollouts: Availability,
    pub history: Availability,
    /// Up to 32 stored input IDs for display; `retained_count` has the total.
    pub retained: Vec<String>,
    pub retained_count: usize,
    pub retained_omitted: usize,
    pub retained_count_unit: &'static str,
    /// Rollout paths found but left for later, beyond the first 512.
    pub deferred_rollout_paths: usize,
    #[serde(skip)]
    pub admitted: Vec<String>,
    pub refused_count: usize,
    pub refused_omitted: usize,
    pub prompt_input: Option<String>,
    pub refused: Vec<Availability>,
    pub conversation_coverage: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paginated_history: Option<Availability>,
    /// The stored thread-history input, if captured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paginated_input: Option<String>,
}
fn refused(s: impl ToString) -> Error {
    Error::ObservedSourceRefused(s.to_string())
}
fn check(cancel: &AtomicBool) -> crate::Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}
fn resolve(home: &Path, path: &Path) -> crate::Result<PathBuf> {
    if path.components().any(|p| matches!(p, Component::ParentDir)) {
        return Err(refused("Codex paths may not traverse parents"));
    }
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        home.join(path)
    };
    if !resolved.starts_with(home) {
        return Err(refused("Codex source path is outside configured home"));
    }
    // Refuse a symlink in any component below the home.
    let mut current = home.to_path_buf();
    for part in resolved.strip_prefix(home).map_err(refused)?.components() {
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(m) if m.file_type().is_symlink() => {
                return Err(refused("Codex source symlink is unsupported"));
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(resolved)
}
fn availability_path(path: PathBuf) -> (Option<PathBuf>, String) {
    let key = blake3::hash(path.as_os_str().as_encoded_bytes())
        .to_hex()
        .to_string();
    let shown = serde_json::to_vec(&path)
        .is_ok_and(|p| p.len() <= 1024)
        .then_some(path);
    (shown, key)
}
fn available(path: PathBuf, entries: usize, unit: &'static str) -> Availability {
    let (path, path_key) = availability_path(path);
    Availability {
        path,
        path_key,
        count_unit: unit,
        state: if entries == 0 {
            "accessible_empty"
        } else {
            "accessible"
        },
        entries: Some(entries),
        reason: None,
        reason_truncated: false,
    }
}
fn unavailable(path: PathBuf, e: impl ToString) -> Availability {
    let (path, path_key) = availability_path(path);
    let reason = e.to_string();
    let reason_truncated = reason.chars().count() > 256;
    Availability {
        path,
        path_key,
        count_unit: "unknown",
        state: "unavailable",
        entries: None,
        reason: Some(reason.chars().take(256).collect()),
        reason_truncated,
    }
}
fn enumerate(
    path: &Path,
    depth: usize,
    visited: &mut usize,
    found: &mut BTreeSet<PathBuf>,
    cancel: &AtomicBool,
) -> crate::Result<()> {
    check(cancel)?;
    if depth > 8 {
        return Err(refused("Codex rollout directory depth exceeds 8"));
    }
    for entry in fs::read_dir(path)? {
        check(cancel)?;
        let entry = entry?;
        *visited += 1;
        if *visited > ENTRIES {
            return Err(refused("Codex rollout enumeration exceeds 512 entries"));
        }
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err(refused("Codex rollout enumeration encountered a symlink"));
        }
        if kind.is_dir() {
            enumerate(&entry.path(), depth + 1, visited, found, cancel)?;
        } else if kind.is_file()
            && entry
                .file_name()
                .to_str()
                .is_some_and(|s| s.starts_with("rollout-") && s.ends_with(".jsonl"))
        {
            found.insert(entry.path());
        }
    }
    Ok(())
}
/// One discovery pass over the rollout tree and the thread catalog. Needs no
/// writer.
pub struct Discovery {
    pub catalog: Availability,
    pub rollouts: Availability,
    pub history_path: PathBuf,
    pub candidates: Vec<Candidate>,
    pub deferred_rollout_paths: usize,
}
#[derive(Clone, Debug)]
pub struct Candidate {
    path: PathBuf,
    format: InputFormat,
}
impl Candidate {
    pub fn observation_id(&self, request: &str, seen_at: i64) -> crate::Result<String> {
        let path_text = self
            .path
            .to_str()
            .ok_or_else(|| refused("Codex path must be UTF-8"))?;
        crate::imports::observation_id(
            &[
                "polylogue.configured-codex-observation.v1",
                request,
                self.format.name(),
                path_text,
            ],
            seen_at,
        )
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn format(&self) -> InputFormat {
        self.format
    }
    pub fn paginated(path: PathBuf) -> Self {
        Self {
            path,
            format: InputFormat::CodexHistorySqlite,
        }
    }
    pub fn prompt(path: PathBuf) -> Self {
        Self {
            path,
            format: InputFormat::CodexPromptHistory,
        }
    }
    fn byte_limit(&self) -> u64 {
        if self.format == InputFormat::CodexPromptHistory {
            1024 * 1024
        } else {
            FILE_BYTES
        }
    }
    /// The file's current length; refuses a non-regular file or one over the
    /// format's size limit.
    pub fn declared_length(&self) -> crate::Result<u64> {
        Ok(self.regular_metadata()?.len())
    }
    fn regular_metadata(&self) -> crate::Result<fs::Metadata> {
        let m = fs::symlink_metadata(&self.path)?;
        if !m.is_file() || m.len() > self.byte_limit() {
            return Err(refused(
                "Codex source must be a regular file within its size limit",
            ));
        }
        Ok(m)
    }
    pub fn fingerprint(
        &self,
        cancel: &AtomicBool,
        remaining: u64,
    ) -> crate::Result<(crate::BlobHash, u64)> {
        self.with_file(cancel, |file, limit| {
            let length = file.metadata()?.len();
            let limit = limit.min(remaining);
            if length > limit {
                return Err(refused("Codex file exceeds the remaining byte budget"));
            }
            let mut file = file.take(length + 1);
            let mut digest = blake3::Hasher::new();
            let mut total = 0u64;
            let mut chunk = [0u8; 65536];
            loop {
                check(cancel)?;
                let count = file.read(&mut chunk)?;
                if count == 0 {
                    break;
                }
                total += count as u64;
                if total > limit {
                    return Err(refused("Codex file grew past its size limit"));
                }
                digest.update(&chunk[..count]);
            }
            Ok((
                crate::BlobHash::from_bytes(*digest.finalize().as_bytes()),
                total,
            ))
        })
    }
    fn with_file<T>(
        &self,
        cancel: &AtomicBool,
        work: impl FnOnce(&mut fs::File, u64) -> crate::Result<T>,
    ) -> crate::Result<T> {
        use std::os::unix::fs::MetadataExt;
        check(cancel)?;
        let metadata = self.regular_metadata()?;
        let limit = self.byte_limit();
        let mut file = fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
            .open(&self.path)?;
        let before = file.metadata()?;
        if !before.is_file()
            || before.len() != metadata.len()
            || before.ino() != metadata.ino()
            || before.dev() != metadata.dev()
        {
            return Err(refused("Codex source changed before capture"));
        }
        let result = work(&mut file, limit)?;
        let after = file.metadata()?;
        if after.len() != before.len()
            || after.mtime() != before.mtime()
            || after.mtime_nsec() != before.mtime_nsec()
            || after.ctime() != before.ctime()
            || after.ctime_nsec() != before.ctime_nsec()
        {
            return Err(refused("Codex source changed during capture"));
        }
        check(cancel)?;
        Ok(result)
    }
}
impl crate::BlobCapture {
    /// Capture a candidate file, refusing it if it changes while being read.
    pub fn capture_codex_candidate(
        &self,
        candidate: &Candidate,
        expected_length: u64,
        cancel: &AtomicBool,
    ) -> crate::Result<crate::PreparedBlob> {
        candidate.with_file(cancel, |file, limit| {
            let length = file.metadata()?.len();
            if length != expected_length {
                return Err(refused("Codex file length changed since it was measured"));
            }
            let proof = self.capture(file.take(length + 1), cancel)?;
            if proof.length() != length || proof.length() > limit {
                return Err(refused("Codex file grew past its size limit"));
            }
            Ok(proof)
        })
    }
}
fn read_catalog(
    home: &Path,
    catalog_path: PathBuf,
    paths: &mut BTreeSet<PathBuf>,
    cancel: &AtomicBool,
) -> crate::Result<Availability> {
    let state = (|| -> crate::Result<Availability> {
        let metadata = fs::symlink_metadata(&catalog_path)?;
        if !metadata.is_file() || metadata.len() > FILE_BYTES {
            return Err(refused(
                "Codex catalog requires a regular file of at most 16 MiB",
            ));
        }
        let connection = Connection::open_with_flags(
            &catalog_path,
            OpenFlags::SQLITE_OPEN_READ_ONLY
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )
        .map_err(|e| refused(e))?;
        connection
            .busy_timeout(std::time::Duration::from_millis(100))
            .map_err(refused)?;
        connection
            .execute_batch("BEGIN DEFERRED")
            .map_err(refused)?;
        let mut statement=connection.prepare("SELECT CASE WHEN length(CAST(rollout_path AS BLOB))<=65536 THEN rollout_path END FROM threads ORDER BY id LIMIT 513").map_err(refused)?;
        let mut rows = statement.query([]).map_err(refused)?;
        let mut count = 0;
        while let Some(row) = rows.next().map_err(refused)? {
            check(cancel)?;
            count += 1;
            if count > ENTRIES {
                return Err(refused("Codex thread catalog exceeds 512 rows"));
            }
            let path: String = row.get(0).map_err(refused)?;
            if path.len() > 65536 {
                return Err(refused("Codex catalog path exceeds 64 KiB"));
            }
            paths.insert(resolve(&home, Path::new(&path))?);
        }
        Ok(available(catalog_path.clone(), count, "thread_rows"))
    })();
    let state = match state {
        Ok(v) => v,
        Err(Error::Cancelled) => return Err(Error::Cancelled),
        Err(e) => unavailable(catalog_path, e),
    };
    Ok(state)
}
struct Paths {
    home: PathBuf,
    rollouts: PathBuf,
    history: PathBuf,
    catalog: PathBuf,
}
impl Paths {
    fn resolve(config: &Config) -> crate::Result<Self> {
        if !config.home.is_absolute() || config.machine.is_empty() || config.host.is_empty() {
            return Err(refused(
                "absolute Codex home and nonempty machine/host required",
            ));
        }
        let home = fs::canonicalize(&config.home)?;
        if !home.is_dir() {
            return Err(refused("Codex home must be a directory"));
        }
        Ok(Self {
            rollouts: resolve(&home, &config.rollouts)?,
            history: resolve(&home, &config.history)?,
            catalog: resolve(&home, &config.catalog)?,
            home,
        })
    }
}
pub fn discover(config: &Config, cancel: &AtomicBool) -> crate::Result<Discovery> {
    check(cancel)?;
    let Paths {
        home,
        rollouts: rollout_path,
        history: history_path,
        catalog: catalog_path,
    } = Paths::resolve(config)?;
    let mut paths = BTreeSet::new();
    let mut visited = 0;
    let rollout_state = match enumerate(&rollout_path, 0, &mut visited, &mut paths, cancel) {
        Ok(()) => available(rollout_path, paths.len(), "rollout_paths"),
        Err(Error::Cancelled) => return Err(Error::Cancelled),
        Err(e) => unavailable(rollout_path, e),
    };
    let catalog_state = read_catalog(&home, catalog_path, &mut paths, cancel)?;
    let deferred_rollout_paths = paths.len().saturating_sub(ENTRIES);
    let candidates = paths
        .into_iter()
        .take(ENTRIES)
        .map(|path| Candidate {
            path,
            format: InputFormat::CodexRollout,
        })
        .collect();
    Ok(Discovery {
        catalog: catalog_state,
        rollouts: rollout_state,
        history_path,
        candidates,
        deferred_rollout_paths,
    })
}
/// Walks the rollout tree a few entries per turn, holding at most nine open
/// directories. A finished cycle starts again from the top.
#[derive(Default)]
pub struct RolloutCursor {
    root: Option<PathBuf>,
    stack: Vec<(usize, fs::ReadDir)>,
    identity: Option<(u64, u64)>,
    cycle: u64,
    cycle_paths: usize,
    cycle_failed: bool,
}
pub struct DiscoveryTurn {
    pub discovery: Discovery,
    pub cycle: u64,
    pub cycle_complete: bool,
    pub visited_entries: usize,
}
impl RolloutCursor {
    pub fn next_turn(
        &mut self,
        config: &Config,
        entry_budget: usize,
        cancel: &AtomicBool,
    ) -> crate::Result<DiscoveryTurn> {
        check(cancel)?;
        if entry_budget == 0 || entry_budget > ENTRIES {
            return Err(refused("Codex enumeration turn requires 1..512 entries"));
        }
        let Paths {
            home,
            rollouts: rollout_path,
            history: history_path,
            catalog: catalog_path,
        } = Paths::resolve(config)?;
        use std::os::unix::fs::MetadataExt;
        let identity = fs::symlink_metadata(&rollout_path)
            .ok()
            .filter(|m| m.is_dir())
            .map(|m| (m.dev(), m.ino()));
        if self.root.as_ref() != Some(&rollout_path) || self.identity != identity {
            self.stack.clear();
            self.root = Some(rollout_path.clone());
            self.identity = identity;
        }
        if self.stack.is_empty() {
            self.cycle = self.cycle.saturating_add(1);
            self.cycle_paths = 0;
            self.cycle_failed = false;
            match fs::read_dir(&rollout_path) {
                Ok(entries) => self.stack.push((0, entries)),
                Err(e) => {
                    let mut paths = BTreeSet::new();
                    let catalog = read_catalog(&home, catalog_path, &mut paths, cancel)?;
                    let candidates = paths
                        .into_iter()
                        .map(|path| Candidate {
                            path,
                            format: InputFormat::CodexRollout,
                        })
                        .collect();
                    return Ok(DiscoveryTurn {
                        discovery: Discovery {
                            catalog,
                            rollouts: unavailable(rollout_path, e),
                            history_path,
                            candidates,
                            deferred_rollout_paths: 0,
                        },
                        cycle: self.cycle,
                        cycle_complete: false,
                        visited_entries: 0,
                    });
                }
            }
        }
        let mut paths = BTreeSet::new();
        let mut visited = 0;
        let mut failure = None;
        while visited < entry_budget && !self.stack.is_empty() {
            check(cancel)?;
            let (depth, next) = {
                let (depth, entries) = self.stack.last_mut().expect("cursor directory");
                (*depth, entries.next())
            };
            let entry = match next {
                None => {
                    self.stack.pop();
                    continue;
                }
                Some(Err(e)) => {
                    self.stack.pop();
                    self.cycle_failed = true;
                    failure = Some(e.to_string());
                    break;
                }
                Some(Ok(entry)) => entry,
            };
            visited += 1;
            let kind = match entry.file_type() {
                Ok(kind) => kind,
                Err(e) => {
                    self.cycle_failed = true;
                    failure = Some(e.to_string());
                    break;
                }
            };
            if kind.is_symlink() {
                self.cycle_failed = true;
                failure = Some("Codex rollout enumeration encountered a symlink".into());
                break;
            }
            if kind.is_dir() {
                if depth >= 8 {
                    self.cycle_failed = true;
                    failure = Some("Codex rollout directory depth exceeds 8".into());
                    break;
                }
                match fs::read_dir(entry.path()) {
                    Ok(entries) => self.stack.push((depth + 1, entries)),
                    Err(e) => {
                        self.cycle_failed = true;
                        failure = Some(e.to_string());
                        break;
                    }
                }
            } else if kind.is_file()
                && entry
                    .file_name()
                    .to_str()
                    .is_some_and(|s| s.starts_with("rollout-") && s.ends_with(".jsonl"))
            {
                self.cycle_paths = self.cycle_paths.saturating_add(1);
                paths.insert(entry.path());
            }
        }
        let complete = self.stack.is_empty();
        let rollouts = if let Some(error) = failure {
            unavailable(rollout_path, error)
        } else if self.cycle_failed {
            unavailable(
                rollout_path,
                "some entries in this enumeration cycle could not be read",
            )
        } else if complete {
            available(rollout_path, self.cycle_paths, "rollout_paths")
        } else {
            let mut state = available(rollout_path, 0, "rollout_paths");
            state.state = "partial";
            state.entries = None;
            state.reason = Some("enumeration cycle unfinished; later turns continue".into());
            state
        };
        let mut catalog_paths = BTreeSet::new();
        let catalog = read_catalog(&home, catalog_path, &mut catalog_paths, cancel)?;
        // Rollouts from this turn plus every catalog path: at most 1024.
        paths.extend(catalog_paths);
        let candidates = paths
            .into_iter()
            .map(|path| Candidate {
                path,
                format: InputFormat::CodexRollout,
            })
            .collect();
        Ok(DiscoveryTurn {
            discovery: Discovery {
                catalog,
                rollouts,
                history_path,
                candidates,
                deferred_rollout_paths: 0,
            },
            cycle: self.cycle,
            cycle_complete: complete,
            visited_entries: visited,
        })
    }
}
impl Writer {
    pub fn known_configured_snapshot(
        &mut self,
        candidate: &Candidate,
        hash: crate::BlobHash,
        machine: &str,
        host: &str,
        path: &str,
    ) -> crate::Result<Option<crate::StoredInput>> {
        use rusqlite::OptionalExtension;
        self.check_import_acquisition()?;
        let file = self.root.join("archive.db");
        let id: Option<String> = crate::db(&file, || {
            self.archive.query_row("SELECT i.id FROM input i JOIN observation o ON o.input=i.id JOIN machine m ON m.id=o.machine WHERE i.format=?1 AND i.original_hash=?2 AND i.context_json='[]' AND o.class='imported' AND o.machine=?3 AND o.path=?4 AND m.host=?5 AND i.id=?6 LIMIT 1",rusqlite::params![candidate.format.name(),hash.bytes().as_slice(),machine,path,host,polylogue_model::InputId::imported(&hash,candidate.format.name(),&[]).as_str()],|row|row.get(0)).optional()
        })?;
        match id {
            Some(id) => Ok(self.immutable_import(&id)?.map(|i| i.input)),
            None => Ok(None),
        }
    }
    pub fn acquire_codex_home(
        &mut self,
        config: &Config,
        request: &str,
        seen_at: i64,
        cancel: &AtomicBool,
    ) -> crate::Result<Acquisition> {
        self.check_import_acquisition()?;
        paginated_path(config)?;
        let plan = discover(config, cancel)?;
        let deferred_rollout_paths = plan.deferred_rollout_paths;
        let catalog_state = plan.catalog;
        let rollout_state = plan.rollouts;
        let history_path = plan.history_path;
        let paths = plan.candidates.into_iter().map(|c| c.path);
        let mut stored = Vec::new();
        let mut failures = Vec::new();
        let mut total = 0;
        for path in paths {
            check(cancel)?;
            match self.acquire_configured_file(
                &path,
                InputFormat::CodexRollout,
                config,
                request,
                seen_at,
                &mut total,
                cancel,
            ) {
                Ok(source) => {
                    stored.push(source.input.id);
                }
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                Err(e) => failures.push(unavailable(path, e)),
            }
        }
        let mut prompt_input = None;
        let history_state = match self.acquire_configured_file(
            &history_path,
            InputFormat::CodexPromptHistory,
            config,
            request,
            seen_at,
            &mut total,
            cancel,
        ) {
            Ok(source) => {
                let empty = source.input.length == 0;
                prompt_input = Some(source.input.id.clone());
                stored.push(source.input.id);
                let mut state = available(history_path, 1, "retained_inputs");
                if empty {
                    state.state = "accessible_empty";
                }
                state
            }
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e) => {
                let state = unavailable(history_path, e);
                failures.push(state.clone());
                state
            }
        };
        // The thread-history database is captured later, in turns, once the
        // owner is serving requests.
        let paginated_input = None;
        let paginated_history = if let Some(path) = paginated_path(config)? {
            Some(match fs::symlink_metadata(&path) {
                Ok(meta) if meta.is_file() => {
                    let mut state = available(path, 0, "retained_snapshots");
                    state.state = "pending";
                    state.entries = None;
                    state.reason = Some("captured later in owner turns".into());
                    state
                }
                Ok(_) => {
                    let state = unavailable(
                        path,
                        "thread history must be a regular non-symlink database",
                    );
                    failures.push(state.clone());
                    state
                }
                Err(error) => {
                    let state = unavailable(path, error);
                    failures.push(state.clone());
                    state
                }
            })
        } else {
            None
        };
        let retained_count = stored.len();
        let refused_count = failures.len();
        let admitted = stored.clone();
        stored.truncate(32);
        failures.truncate(4);
        let mut report = Acquisition {
            paginated_history,
            paginated_input,
            recipe: "configured-codex-home-v1",
            scope: "finite_startup_snapshots",
            catalog: catalog_state,
            rollouts: rollout_state,
            history: history_state,
            retained_omitted: retained_count - stored.len(),
            retained_count,
            retained_count_unit: "successful_source_attempts",
            deferred_rollout_paths,
            admitted,
            retained: stored,
            prompt_input,
            refused_omitted: refused_count - failures.len(),
            refused_count,
            refused: failures,
            conversation_coverage: None,
        };
        while serde_json::to_vec(&report).is_ok_and(|bytes| bytes.len() > 16384) {
            if report.refused.pop().is_some() {
                report.refused_omitted += 1;
            } else if report.retained.pop().is_some() {
                report.retained_omitted += 1;
            } else {
                break;
            }
        }
        Ok(report)
    }
    fn acquire_configured_file(
        &mut self,
        path: &Path,
        format: InputFormat,
        config: &Config,
        request: &str,
        seen_at: i64,
        total: &mut u64,
        cancel: &AtomicBool,
    ) -> crate::Result<StoredSource> {
        let candidate = Candidate {
            path: path.to_path_buf(),
            format,
        };
        let size = fs::symlink_metadata(path)?.len();
        *total = total.checked_add(size).ok_or(Error::Range)?;
        if *total > 32 * 1024 * 1024 {
            return Err(refused("Codex startup capture exceeds 32 MiB"));
        }
        let proof = self
            .blob_capture()?
            .capture_codex_candidate(&candidate, size, cancel)?;
        let hash = self.store_captured_blob(proof)?;
        let path_text = path
            .to_str()
            .ok_or_else(|| refused("Codex path must be UTF-8"))?;
        let id = candidate.observation_id(request, seen_at)?;
        let input = self.store_immutable_input(
            hash,
            format,
            &[],
            &ImportObservation {
                id: &id,
                machine: &config.machine,
                host: &config.host,
                path: path_text,
                seen_at,
            },
            131072,
        )?;
        self.stored_source(&input.id)?
            .ok_or_else(|| refused("stored Codex source is missing"))
    }
}
#[derive(Debug, Serialize)]
pub struct Provenance {
    pub machine: String,
    pub host: String,
    pub path: String,
    pub seen_at: i64,
}
impl ReadHandle {
    pub fn codex_prompt_provenance(&self, input: &str) -> crate::Result<Provenance> {
        let file = self.root.join("archive.db");
        crate::db(&file, || {
            self.archive.query_row("SELECT CASE WHEN length(CAST(o.machine AS BLOB))<=4096 THEN o.machine END,CASE WHEN length(CAST(m.host AS BLOB))<=4096 THEN m.host END,CASE WHEN length(CAST(o.path AS BLOB))<=65536 THEN o.path END,o.seen_at FROM observation o JOIN machine m ON m.id=o.machine WHERE o.input=?1 AND o.class='imported' ORDER BY o.rowid LIMIT 1",[input],|row|Ok(Provenance {machine:row.get(0)?,host:row.get(1)?,path:row.get(2)?,seen_at:row.get(3)?}))
        })
    }

    /// Every stored prompt-history input, at most 513.
    pub fn codex_prompt_sources(&self) -> crate::Result<Vec<StoredSource>> {
        let file = self.root.join("archive.db");
        let mut query = crate::db(&file, || {
            self.archive.prepare(
                "SELECT id FROM input WHERE format='codex-prompt-history' ORDER BY id LIMIT 513",
            )
        })?;
        let ids = crate::db(&file, || query.query_map([], |row| row.get::<_, String>(0)))?;
        let mut result = Vec::new();
        for id in ids {
            let id = crate::db(&file, || id)?;
            result.push(
                self.source(&id)?
                    .ok_or_else(|| refused("stored prompt source is missing"))?,
            );
        }
        Ok(result)
    }
}

/// File metadata of the thread-history database and its WAL, used to skip a
/// capture when nothing changed since the last poll. Not a content check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryStamp(Vec<Option<(u64, u64, u64, i64, i64, i64, i64)>>);
pub fn history_stamp(config: &Config) -> crate::Result<HistoryStamp> {
    use std::os::unix::fs::MetadataExt;
    let path =
        paginated_path(config)?.ok_or_else(|| refused("paginated history is not configured"))?;
    let mut wal = path.as_os_str().to_os_string();
    wal.push("-wal");
    let wal = PathBuf::from(wal);
    let mut values = Vec::with_capacity(2);
    for path in [path, wal] {
        match fs::symlink_metadata(path) {
            Ok(m) if m.is_file() => values.push(Some((
                m.dev(),
                m.ino(),
                m.len(),
                m.mtime(),
                m.mtime_nsec(),
                m.ctime(),
                m.ctime_nsec(),
            ))),
            Ok(_) => {
                return Err(refused(
                    "history observation requires regular non-symlink files",
                ));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => values.push(None),
            Err(e) => return Err(e.into()),
        }
    }
    if values[0].is_none() {
        return Err(refused("configured paginated history is unavailable"));
    }
    Ok(HistoryStamp(values))
}

#[cfg(test)]
mod tests;

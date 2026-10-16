//! Bounded configured-home discovery and acquisition. SQLite access and mutation stay
//! in store. Catalog membership never gates independent rollout enumeration.
use crate::{Error, ImportObservation, InputFormat, ReadHandle, RetainedSource, Writer};
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
    /// Exact admitted total only after complete traversal; unavailable is unknown.
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
    /// A bounded presentation sample, independently of completed acquisition.
    pub retained: Vec<String>,
    pub retained_count: usize,
    pub retained_omitted: usize,
    pub retained_count_unit: &'static str,
    /// Known union-prefix paths deferred by the finite startup admission.
    pub deferred_rollout_paths: usize,
    #[serde(skip)]
    pub admitted: Vec<String>,
    pub refused_count: usize,
    pub refused_omitted: usize,
    pub prompt_input: Option<String>,
    pub refused: Vec<Availability>,
    pub conversation_coverage: Option<bool>,
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
    // Refuse symlinks in every source component below the configured home.
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
/// One bounded discovery pass. SQLite catalog reads stay in store; callers may
/// run this outside the single writer. Membership never gates independent paths.
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
        let mut digest = blake3::Hasher::new();
        for value in [
            "polylogue.configured-codex-observation.v1",
            request,
            self.format.name(),
            path_text,
        ] {
            digest.update(&(value.len() as u64).to_be_bytes());
            digest.update(value.as_bytes());
        }
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&digest.finalize().as_bytes()[..16]);
        let millis = u64::try_from(seen_at).map_err(|_| Error::Range)? / 1000;
        if millis > 0xffffffffffff {
            return Err(Error::Range);
        }
        bytes[..6].copy_from_slice(&millis.to_be_bytes()[2..]);
        bytes[6] = (bytes[6] & 0x0f) | 0x70;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        let hex = bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
        Ok(format!(
            "{}-{}-{}-{}-{}",
            &hex[..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..]
        ))
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn format(&self) -> InputFormat {
        self.format
    }
    pub fn prompt(path: PathBuf) -> Self {
        Self {
            path,
            format: InputFormat::CodexPromptHistory,
        }
    }
    pub fn declared_length(&self) -> crate::Result<u64> {
        let m = fs::symlink_metadata(&self.path)?;
        let limit = if self.format == InputFormat::CodexPromptHistory {
            1024 * 1024
        } else {
            FILE_BYTES
        };
        if !m.is_file() || m.len() > limit {
            return Err(refused(
                "Codex source requires a regular file within its finite byte admission",
            ));
        }
        Ok(m.len())
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
                return Err(refused("Codex scan exceeds remaining byte admission"));
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
                    return Err(refused("Codex source changed beyond admission"));
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
        let metadata = fs::symlink_metadata(&self.path)?;
        let limit = if self.format == InputFormat::CodexPromptHistory {
            1024 * 1024
        } else {
            FILE_BYTES
        };
        if !metadata.is_file() || metadata.len() > limit {
            return Err(refused(
                "Codex source requires a regular file within its finite byte admission",
            ));
        }
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
    /// The same bounded stable-descriptor capture for startup and continuation.
    pub fn capture_codex_candidate(
        &self,
        candidate: &Candidate,
        expected_length: u64,
        cancel: &AtomicBool,
    ) -> crate::Result<crate::PreparedBlob> {
        candidate.with_file(cancel, |file, limit| {
            let length = file.metadata()?.len();
            if length != expected_length {
                return Err(refused("Codex source length changed after byte admission"));
            }
            let proof = self.capture(file.take(length + 1), cancel)?;
            if proof.length() != length || proof.length() > limit {
                return Err(refused("Codex source changed beyond admission"));
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
                return Err(refused("Codex catalog path exceeds admission"));
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
pub fn discover(config: &Config, cancel: &AtomicBool) -> crate::Result<Discovery> {
    check(cancel)?;
    if !config.home.is_absolute() || config.machine.is_empty() || config.host.is_empty() {
        return Err(refused(
            "absolute Codex home and nonempty machine/host required",
        ));
    }
    let home = fs::canonicalize(&config.home)?;
    if !home.is_dir() {
        return Err(refused("Codex home must be a directory"));
    }
    let rollout_path = resolve(&home, &config.rollouts)?;
    let history_path = resolve(&home, &config.history)?;
    let catalog_path = resolve(&home, &config.catalog)?;
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
/// A single configured rollout-tree traversal retains at most nine directory
/// iterators. A bounded turn advances them; exhaustion never restarts a prefix.
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
        if !config.home.is_absolute() || config.machine.is_empty() || config.host.is_empty() {
            return Err(refused(
                "absolute Codex home and nonempty machine/host required",
            ));
        }
        let home = fs::canonicalize(&config.home)?;
        if !home.is_dir() {
            return Err(refused("Codex home must be a directory"));
        }
        let rollout_path = resolve(&home, &config.rollouts)?;
        let history_path = resolve(&home, &config.history)?;
        let catalog_path = resolve(&home, &config.catalog)?;
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
                "current enumeration cycle encountered an unavailable/refused entry",
            )
        } else if complete {
            available(rollout_path, self.cycle_paths, "rollout_paths")
        } else {
            let mut state = available(rollout_path, 0, "rollout_paths");
            state.state = "partial";
            state.entries = None;
            state.reason =
                Some("bounded enumeration cycle unfinished; later turns continue".into());
            state
        };
        let mut catalog_paths = BTreeSet::new();
        let catalog = read_catalog(&home, catalog_path, &mut catalog_paths, cancel)?;
        // Both independently admitted prefixes remain owed. The candidate batch
        // is bounded by their union (1024 paths), then drains in 512-request turns.
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
    ) -> crate::Result<Option<crate::RetainedInput>> {
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
        let plan = discover(config, cancel)?;
        let deferred_rollout_paths = plan.deferred_rollout_paths;
        let catalog_state = plan.catalog;
        let rollout_state = plan.rollouts;
        let history_path = plan.history_path;
        let paths = plan.candidates.into_iter().map(|c| c.path);
        let mut retained = Vec::new();
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
                    retained.push(source.input.id);
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
                retained.push(source.input.id);
                let mut state = available(history_path, 1, "retained_inputs");
                if empty {
                    state.state = "accessible_empty";
                }
                state
            }
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e) => unavailable(history_path, e),
        };
        let retained_count = retained.len();
        let refused_count = failures.len();
        let admitted = retained.clone();
        retained.truncate(32);
        failures.truncate(4);
        let mut report = Acquisition {
            recipe: "configured-codex-home-v1",
            scope: "finite_startup_snapshots",
            catalog: catalog_state,
            rollouts: rollout_state,
            history: history_state,
            retained_omitted: retained_count - retained.len(),
            retained_count,
            retained_count_unit: "successful_source_attempts",
            deferred_rollout_paths,
            admitted,
            retained,
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
    ) -> crate::Result<RetainedSource> {
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
        let hash = self.retain_captured_blob(proof)?;
        let path_text = path
            .to_str()
            .ok_or_else(|| refused("Codex path must be UTF-8"))?;
        let id = candidate.observation_id(request, seen_at)?;
        let input = self.retain_immutable_input(
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
        self.retained_source(&input.id)?
            .ok_or_else(|| refused("retained Codex source missing"))
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

    /// Durable finite prompt snapshots. Bytes are verified through the ordinary
    /// retained source reader; no provider filesystem or SQLite leaves store.
    pub fn codex_prompt_sources(&self) -> crate::Result<Vec<RetainedSource>> {
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
                    .ok_or_else(|| refused("retained prompt source missing"))?,
            );
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;

//! Bounded startup acquisition. SQLite catalog access and archive mutation stay
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
use std::{
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
};
const ENTRIES: usize = 512;
const FILE_BYTES: u64 = 16 * 1024 * 1024;
#[derive(Debug, Deserialize)]
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
    pub path: PathBuf,
    pub state: &'static str,
    /// Exact admitted total only after complete traversal; unavailable is unknown.
    pub entries: Option<usize>,
    pub reason: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Acquisition {
    pub recipe: &'static str,
    pub scope: &'static str,
    pub catalog: Availability,
    pub rollouts: Availability,
    pub history: Availability,
    pub retained: Vec<String>,
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
fn available(path: PathBuf, entries: usize) -> Availability {
    Availability {
        path,
        state: if entries == 0 {
            "accessible_empty"
        } else {
            "accessible"
        },
        entries: Some(entries),
        reason: None,
    }
}
fn unavailable(path: PathBuf, e: impl ToString) -> Availability {
    Availability {
        path,
        state: "unavailable",
        entries: None,
        reason: Some(e.to_string()),
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
impl Writer {
    pub fn acquire_codex_home(
        &mut self,
        config: &Config,
        request: &str,
        seen_at: i64,
        cancel: &AtomicBool,
    ) -> crate::Result<Acquisition> {
        self.check_import_acquisition()?;
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
            Ok(()) => available(rollout_path, paths.len()),
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e) => unavailable(rollout_path, e),
        };
        // A pinned read transaction owns this finite catalog snapshot. It is
        // separate evidence from independently enumerated rollout snapshots.
        let catalog_state = (|| -> crate::Result<Availability> {
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
            Ok(available(catalog_path.clone(), count))
        })();
        let catalog_state = match catalog_state {
            Ok(v) => v,
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e) => unavailable(catalog_path, e),
        };
        if paths.len() > ENTRIES {
            return Err(refused(
                "Codex combined rollout acquisition exceeds 512 files",
            ));
        }
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
                Ok(source) => retained.push(source.input.id),
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
                let entries = usize::from(source.input.length > 0);
                prompt_input = Some(source.input.id.clone());
                retained.push(source.input.id);
                available(history_path, entries)
            }
            Err(Error::Cancelled) => return Err(Error::Cancelled),
            Err(e) => unavailable(history_path, e),
        };
        let report = Acquisition {
            recipe: "configured-codex-home-v1",
            scope: "finite_startup_snapshots",
            catalog: catalog_state,
            rollouts: rollout_state,
            history: history_state,
            retained,
            prompt_input,
            refused: failures,
            conversation_coverage: None,
        };
        if serde_json::to_vec(&report).map_err(refused)?.len() > 16384 {
            return Err(refused("Codex acquisition report exceeds 16 KiB"));
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
        let metadata = fs::symlink_metadata(path)?;
        if !metadata.is_file() || metadata.len() > FILE_BYTES {
            return Err(refused(
                "Codex source requires a regular file of at most 16 MiB",
            ));
        }
        *total = total.checked_add(metadata.len()).ok_or(Error::Range)?;
        if *total > 32 * 1024 * 1024 {
            return Err(refused("Codex startup capture exceeds 32 MiB"));
        }
        if format == InputFormat::CodexPromptHistory && metadata.len() > 1024 * 1024 {
            return Err(refused("Codex prompt history exceeds 1 MiB"));
        }
        let mut file = fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
            .open(path)?;
        let before = file.metadata()?;
        if !before.is_file()
            || before.len() != metadata.len()
            || before.ino() != metadata.ino()
            || before.dev() != metadata.dev()
        {
            return Err(refused("Codex source changed before capture"));
        }
        let proof = self
            .blob_capture()?
            .capture((&mut file).take(before.len() + 1), cancel)?;
        let after = file.metadata()?;
        if proof.length() != before.len()
            || after.len() != before.len()
            || after.mtime() != before.mtime()
            || after.mtime_nsec() != before.mtime_nsec()
            || after.ctime() != before.ctime()
            || after.ctime_nsec() != before.ctime_nsec()
        {
            return Err(refused("Codex source changed during capture"));
        }
        let hash = self.retain_captured_blob(proof)?;
        let path_text = path
            .to_str()
            .ok_or_else(|| refused("Codex path must be UTF-8"))?;
        // Keep the existing UUIDv7 observation/request contract. The startup
        // request supplies randomness; domain-framed file metadata distinguishes
        // the owned actions within that one acquisition time.
        let mut digest = blake3::Hasher::new();
        for value in [
            "polylogue.configured-codex-observation.v1",
            request,
            format.name(),
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
        let id = format!(
            "{}-{}-{}-{}-{}",
            &hex[..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..]
        );
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
            if result.len() == 512 {
                return Err(refused("retained prompt snapshots exceed 512"));
            }
            result.push(
                self.source(&id)?
                    .ok_or_else(|| refused("retained prompt source missing"))?,
            );
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(home: &Path) -> Config {
        Config {
            home: home.into(),
            machine: "synthetic".into(),
            host: "synthetic".into(),
            rollouts: rollouts(),
            history: history(),
            catalog: catalog(),
        }
    }
    #[test]
    fn unavailable_catalog_does_not_gate_rollout_and_cancel_does_not_mutate() {
        let root = tempfile::tempdir().unwrap();
        let home = root.path().join("home");
        fs::create_dir(&home).unwrap();
        fs::create_dir(home.join("sessions")).unwrap();
        fs::write(home.join("sessions/rollout-one.jsonl"), b"{}\n").unwrap();
        fs::write(home.join("state_5.sqlite"), b"broken catalog").unwrap();
        fs::write(home.join("history.jsonl"), b"").unwrap();
        let mut writer = Writer::open(&root.path().join("archive")).unwrap();
        let before = writer.status().unwrap().write_counter;
        assert!(matches!(
            writer.acquire_codex_home(&config(&home), "cancelled", 1, &AtomicBool::new(true)),
            Err(Error::Cancelled)
        ));
        assert_eq!(writer.status().unwrap().write_counter, before);
        let report = writer
            .acquire_codex_home(&config(&home), "first", 1, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(report.catalog.state, "unavailable");
        assert_eq!(report.catalog.entries, None);
        assert_eq!(report.rollouts.entries, Some(1));
        assert_eq!(report.retained.len(), 2);
        assert_eq!(report.history.state, "accessible_empty");
        assert_eq!(
            writer
                .read_handle()
                .unwrap()
                .codex_prompt_sources()
                .unwrap()
                .len(),
            1
        );
    }
    #[test]
    fn relative_resolution_refuses_parent_escape_and_symlinks() {
        let root = tempfile::tempdir().unwrap();
        let home = root.path();
        fs::create_dir(home.join("sessions")).unwrap();
        assert_eq!(
            resolve(home, Path::new("sessions/rollout-one.jsonl")).unwrap(),
            home.join("sessions/rollout-one.jsonl")
        );
        assert!(resolve(home, Path::new("../escape")).is_err());
        std::os::unix::fs::symlink(home.join("sessions"), home.join("linked")).unwrap();
        assert!(resolve(home, Path::new("linked/rollout-one.jsonl")).is_err());
    }
}

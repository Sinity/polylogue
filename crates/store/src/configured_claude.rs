//! Discovery of Claude transcripts under the configured projects directory.
//! Every open goes through the held root directory descriptor, so symlinks and
//! renames below the root cannot redirect it.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    os::unix::fs::MetadataExt,
    path::{Component, Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub root: PathBuf,
    pub machine: String,
    pub host: String,
    #[serde(default = "default_quantum")]
    pub acquisition_quantum_bytes: u64,
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        if !self.root.is_absolute()
            || self.machine.is_empty()
            || self.host.is_empty()
            || !(65536..=8 * 1024 * 1024).contains(&self.acquisition_quantum_bytes)
        {
            return Err(refused(
                "absolute Claude projects root, machine/host and 65536..8388608 acquisition quantum required",
            ));
        }
        Ok(())
    }
}
fn default_quantum() -> u64 {
    8 * 1024 * 1024
}
fn refused(text: impl Into<String>) -> Error {
    Error::ObservedSourceRefused(text.into())
}
#[derive(Debug)]
struct Root {
    path: PathBuf,
    file: File,
    identity: (u64, u64),
}
impl Root {
    fn open(path: &Path) -> Result<Self> {
        if !path.is_absolute() || !fs::symlink_metadata(path)?.is_dir() {
            return Err(refused(
                "Claude projects root requires an absolute non-symlink directory",
            ));
        }
        let path = fs::canonicalize(path)?;
        // Refuse a root inside any archive, so an archive never imports its own
        // or another archive's files.
        for parent in path.ancestors() {
            if parent.join("source.db").try_exists()? || parent.join("archive.db").try_exists()? {
                return Err(refused("Claude projects root is inside an archive"));
            }
        }
        let file = File::open(&path)?;
        let m = file.metadata()?;
        Ok(Self {
            path,
            file,
            identity: (m.dev(), m.ino()),
        })
    }
    fn check(&self) -> Result<()> {
        let m = fs::symlink_metadata(&self.path)?;
        if !m.is_dir() || (m.dev(), m.ino()) != self.identity {
            return Err(refused("Claude projects root changed during acquisition"));
        }
        Ok(())
    }
    fn open_relative(&self, relative: &Path, directory: bool) -> Result<File> {
        use rustix::fs::{Mode, OFlags, ResolveFlags, openat2};
        self.check()?;
        let flags = OFlags::RDONLY
            | OFlags::NOFOLLOW
            | OFlags::CLOEXEC
            | OFlags::NONBLOCK
            | if directory {
                OFlags::DIRECTORY
            } else {
                OFlags::empty()
            };
        let fd = openat2(
            &self.file,
            relative,
            flags,
            Mode::empty(),
            ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS,
        )
        .map_err(std::io::Error::from)?;
        Ok(File::from(fd))
    }
}
/// One transcript file and the context stored with it. Opens go through the
/// discovery root when there is one. The parser reads subagent membership from
/// the records; the folder only names the owner.
#[derive(Clone, Debug)]
pub struct Location {
    path: PathBuf,
    root: Option<Arc<Root>>,
    relative: PathBuf,
    context: Vec<String>,
}
impl Location {
    pub fn explicit_main(path: PathBuf) -> Result<Self> {
        let stem = path
            .file_stem()
            .and_then(|v| v.to_str())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| refused("UTF8 Claude main stem required"))?;
        Ok(Self {
            context: vec!["claude-main".into(), stem.into()],
            path,
            root: None,
            relative: PathBuf::new(),
        })
    }
    pub fn observation_id(&self, request: &str, seen_at: i64) -> Result<String> {
        let mut h = blake3::Hasher::new();
        for value in [
            "polylogue.claude-project-observation.v1",
            request,
            self.path
                .to_str()
                .ok_or_else(|| refused("UTF8 location required"))?,
        ] {
            h.update(&(value.len() as u64).to_be_bytes());
            h.update(value.as_bytes());
        }
        let millis = u64::try_from(seen_at).map_err(|_| Error::Range)? / 1000;
        if millis >= 1u64 << 48 {
            return Err(Error::Range);
        }
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&h.finalize().as_bytes()[..16]);
        bytes[..6].copy_from_slice(&millis.to_be_bytes()[2..]);
        bytes[6] = (bytes[6] & 15) | 0x70;
        bytes[8] = (bytes[8] & 63) | 0x80;
        let hex = bytes.iter().map(|v| format!("{v:02x}")).collect::<String>();
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
    pub fn context(&self) -> &[String] {
        &self.context
    }
    pub(crate) fn check(&self) -> Result<()> {
        if let Some(root) = &self.root {
            root.check()?;
        }
        Ok(())
    }
    pub(crate) fn open(&self) -> Result<File> {
        if let Some(root) = &self.root {
            root.open_relative(&self.relative, false)
        } else {
            use std::os::unix::fs::OpenOptionsExt;
            Ok(fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(&self.path)?)
        }
    }
    pub(crate) fn open_body(&self, path: &Path) -> Result<Option<File>> {
        let Some(root) = &self.root else {
            return crate::claude_bodies::open_body(&self.path, path);
        };
        let parts = components(&self.relative).ok_or_else(|| refused("invalid Claude location"))?;
        let owner = if self.context[0] == "claude-main" {
            &self.context[1]
        } else {
            &self.context[2]
        };
        let directory = root.path.join(&parts[0]).join(owner).join("tool-results");
        let name = path
            .file_name()
            .and_then(|v| v.to_str())
            .ok_or_else(|| refused("UTF8 body name required"))?;
        if path.parent() != Some(directory.as_path())
            || path.extension().and_then(|v| v.to_str()) != Some("txt")
            || !visible(name)
            || name.starts_with("hook-")
        {
            return Err(refused(
                "Claude body is outside the owning session/tool-results/*.txt layout",
            ));
        }
        let relative = path
            .strip_prefix(&root.path)
            .map_err(|_| refused("Claude body outside configured root"))?;
        match root.open_relative(relative, false) {
            Ok(file) => Ok(Some(file)),
            Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }
}
fn visible(s: &str) -> bool {
    !s.is_empty() && !s.starts_with('.')
}
fn transcript(s: &str) -> Option<&str> {
    s.strip_suffix(".jsonl")
        .or_else(|| s.strip_suffix(".ndjson"))
        .filter(|v| visible(v))
}
fn components(path: &Path) -> Option<Vec<String>> {
    path.components()
        .map(|c| match c {
            Component::Normal(v) => v.to_str().map(str::to_owned),
            _ => None,
        })
        .collect()
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    Transcript(Vec<String>),
    Deferred,
    Excluded,
}
fn kind(parts: &[String]) -> Kind {
    let s = parts.iter().map(String::as_str).collect::<Vec<_>>();
    if s.first().is_none_or(|p| !p.starts_with('-')) || s.iter().any(|p| !visible(p)) {
        return Kind::Excluded;
    }
    match s.as_slice() {
        [_, leaf] if transcript(leaf).is_some() => {
            Kind::Transcript(vec!["claude-main".into(), transcript(leaf).unwrap().into()])
        }
        [_, owner, "subagents", leaf] | [_, owner, "subagents", "workflows", _, leaf]
            if transcript(leaf).is_some_and(|v| v.starts_with("agent-") && v.len() > 6) =>
        {
            Kind::Transcript(vec![
                "claude-subagent".into(),
                transcript(leaf).unwrap()[6..].into(),
                (*owner).into(),
            ])
        }
        [_, "sessions-index.json"]
        | [_, "memory", ..]
        | [_, _, "workflows", _]
        | [_, _, "subagents", _]
        | [_, _, "subagents", "workflows", _, _]
        | [_, _, "tool-results", _] => Kind::Deferred,
        _ => Kind::Excluded,
    }
}
fn descend(parts: &[String]) -> bool {
    let s = parts.iter().map(String::as_str).collect::<Vec<_>>();
    if s.first().is_none_or(|p| !p.starts_with('-')) || s.iter().any(|p| !visible(p)) {
        return false;
    }
    matches!(
        s.as_slice(),
        [_] | [_, _]
            | [_, _, "subagents"]
            | [_, _, "subagents", "workflows"]
            | [_, _, "subagents", "workflows", _]
    )
}
struct Directory {
    relative: PathBuf,
    entries: fs::ReadDir,
    _file: File,
}
#[derive(Default)]
pub struct Cursor {
    root: Option<Arc<Root>>,
    stack: Vec<Directory>,
    cycle: u64,
    seen: u64,
    deferred: u64,
    excluded: u64,
    failed: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct Scan {
    pub state: &'static str,
    pub cycle: u64,
    pub cycle_complete: bool,
    pub visited_entries: usize,
    pub observed_transcripts: u64,
    pub deferred_entries: u64,
    pub excluded_entries: u64,
    pub reason: Option<String>,
    pub conversation_coverage: Option<bool>,
}
pub struct Turn {
    pub scan: Scan,
    pub locations: Vec<Location>,
}
impl Cursor {
    fn directory(root: &Root, relative: PathBuf) -> Result<Directory> {
        use std::os::fd::AsRawFd;
        let file = if relative.as_os_str().is_empty() {
            root.file.try_clone()?
        } else {
            root.open_relative(&relative, true)?
        };
        let entries = fs::read_dir(format!("/proc/self/fd/{}", file.as_raw_fd()))?;
        Ok(Directory {
            relative,
            entries,
            _file: file,
        })
    }
    pub fn next_turn(
        &mut self,
        config: &Config,
        budget: usize,
        cancel: &AtomicBool,
    ) -> Result<Turn> {
        config.validate()?;
        if budget == 0 || budget > 512 {
            return Err(refused("Claude discovery budget must be 1..512 entries"));
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Cancelled);
        }
        let current = Root::open(&config.root)?;
        if self
            .root
            .as_ref()
            .is_none_or(|r| r.path != current.path || r.identity != current.identity)
        {
            self.stack.clear();
            self.root = Some(Arc::new(current));
        }
        let root = self.root.as_ref().unwrap().clone();
        if self.stack.is_empty() {
            self.cycle = self.cycle.saturating_add(1);
            self.seen = 0;
            self.deferred = 0;
            self.excluded = 0;
            self.failed = false;
            self.stack.push(Self::directory(&root, PathBuf::new())?);
        }
        let mut visited = 0;
        let mut locations = Vec::new();
        let mut reason = None;
        while visited < budget && !self.stack.is_empty() {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            let dir = self.stack.last_mut().unwrap();
            let Some(entry) = dir.entries.next() else {
                self.stack.pop();
                continue;
            };
            visited += 1;
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    self.failed = true;
                    reason = Some(e.to_string());
                    self.stack.pop();
                    continue;
                }
            };
            let relative = dir.relative.join(entry.file_name());
            let Some(parts) = components(&relative) else {
                self.failed = true;
                reason = Some("non-UTF8 Claude entry refused".into());
                continue;
            };
            let file_type = match entry.file_type() {
                Ok(v) => v,
                Err(e) => {
                    self.failed = true;
                    reason = Some(e.to_string());
                    continue;
                }
            };
            if file_type.is_symlink() {
                self.failed = true;
                reason = Some("Claude source symlink refused".into());
                continue;
            }
            if file_type.is_dir() {
                if descend(&parts) {
                    match Self::directory(&root, relative) {
                        Ok(d) => self.stack.push(d),
                        Err(e) => {
                            self.failed = true;
                            reason = Some(e.to_string());
                        }
                    }
                } else {
                    self.excluded = self.excluded.saturating_add(1);
                }
            } else if file_type.is_file() {
                match kind(&parts) {
                    Kind::Transcript(context) => {
                        self.seen = self.seen.saturating_add(1);
                        locations.push(Location {
                            path: root.path.join(&relative),
                            root: Some(root.clone()),
                            relative,
                            context,
                        });
                    }
                    Kind::Deferred => self.deferred = self.deferred.saturating_add(1),
                    Kind::Excluded => self.excluded = self.excluded.saturating_add(1),
                }
            } else {
                self.failed = true;
                reason = Some("nonregular Claude entry refused".into());
            }
        }
        let complete = self.stack.is_empty();
        Ok(Turn {
            locations,
            scan: Scan {
                state: if self.failed {
                    "unavailable"
                } else if !complete {
                    "partial"
                } else if self.seen == 0 {
                    "accessible_empty"
                } else {
                    "accessible"
                },
                cycle: self.cycle,
                cycle_complete: complete,
                visited_entries: visited,
                observed_transcripts: self.seen,
                deferred_entries: self.deferred,
                excluded_entries: self.excluded,
                reason: reason.or_else(|| {
                    self.failed
                        .then(|| "some entries in this discovery cycle could not be read".into())
                }),
                conversation_coverage: None,
            },
        })
    }
}

#[cfg(test)]
#[path = "configured_claude/tests.rs"]
mod tests;

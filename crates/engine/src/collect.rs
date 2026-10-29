//! Finding inputs at the origins' fixed locations.
//! Exactly the layout each tool writes: each origin's location patterns are
//! walked under its root (the tool's own directory variable, or a default
//! under home), and a recognizer runs before a new
//! path's origin is decided. An unreadable root is a state, never an empty
//! backlog.

use crate::Config;
use polylogue_sources::{Arrival, ORIGINS, OriginSpec};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub struct Found {
    pub path: PathBuf,
    pub origin: &'static OriginSpec,
}

/// Reading a watched file or directory is not new input. In particular, the
/// collector's own directory opens must not wake another collection pass.
pub(crate) fn requires_scan(event: &notify::Event) -> bool {
    use notify::EventKind;
    use notify::event::{AccessKind, AccessMode};
    !matches!(
        event.kind,
        EventKind::Access(
            AccessKind::Read | AccessKind::Open(_) | AccessKind::Close(AccessMode::Read)
        )
    )
}

/// Collection is independent of parser availability.
fn collected() -> impl Iterator<Item = &'static OriginSpec> {
    ORIGINS.iter().filter(|o| o.arrival != Arrival::Import)
}

pub use polylogue_model::capture::Classification as Class;
use polylogue_model::capture::{Accessibility, Entry, RootCoverage};

/// Aggregated during the discovery walk, without reopening source trees.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Coverage {
    pub root: PathBuf,
    pub class: Class,
    pub pattern: String,
    pub reason: Option<String>,
    pub files: u64,
    pub bytes: u64,
    pub newest_mtime_ms: Option<u64>,
}

pub struct Scan {
    pub found: Vec<Found>,
    pub unreadable: Vec<PathBuf>,
    pub coverage: Vec<Coverage>,
    pub roots: Vec<RootCoverage>,
}

pub fn scan(config: &Config) -> (Vec<Found>, Vec<PathBuf>) {
    let scan = scan_with_coverage(config);
    (scan.found, scan.unreadable)
}

/// Changed files and the contents of newly created or moved directories.
/// A directory is walked only where an origin's location can match it.
pub fn scan_paths(
    config: &Config,
    paths: &std::collections::BTreeSet<PathBuf>,
) -> (Vec<Found>, Vec<PathBuf>) {
    let mut found = Vec::new();
    let mut unreadable = Vec::new();
    for origin in collected() {
        let root = config.root(origin.root);
        for pattern in origin.locations {
            let segs: Vec<_> = pattern.split('/').filter(|s| !s.is_empty()).collect();
            let mut matched = Vec::new();
            walk_paths(&root, &segs, paths, &mut matched, &mut unreadable);
            found.extend(
                matched
                    .into_iter()
                    .filter(|path| {
                        path.strip_prefix(&root)
                            .is_ok_and(|relative| matches_location(relative, pattern))
                    })
                    .map(|path| Found { path, origin }),
            );
        }
    }
    for tool in polylogue_sources::locations::TOOLS {
        let base = config.root(tool.root);
        let mut roots = vec![base.clone()];
        if tool.name == "hermes" {
            roots.extend(
                entries(&base.join("profiles"), &mut unreadable)
                    .into_iter()
                    .filter(|p| p.is_dir()),
            );
        }
        for root in roots {
            for pattern in tool.locations {
                let mut matched = Vec::new();
                let segs: Vec<_> = pattern.split('/').collect();
                walk_paths(&root, &segs, paths, &mut matched, &mut unreadable);
                for path in matched {
                    let relative = path.strip_prefix(&root).expect("location under root");
                    if !matches_location(relative, pattern) {
                        continue;
                    }
                    if git_metadata(relative, Some(tool))
                        || tool
                            .ignored
                            .iter()
                            .any(|(pattern, _)| matches_path(relative, pattern))
                    {
                        continue;
                    }
                    let origin = context_origin(tool, &path);
                    if !found.iter().any(|f| f.path == path) {
                        found.push(Found { path, origin });
                    }
                }
            }
        }
    }
    let claude_root = config.root(polylogue_sources::claude_code::ROOT);
    found.retain(|file| !user_project(&claude_root, &file.path));
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found.dedup_by(|a, b| a.path == b.path);
    (found, unreadable)
}

fn walk_paths(
    dir: &Path,
    segs: &[&str],
    paths: &std::collections::BTreeSet<PathBuf>,
    out: &mut Vec<PathBuf>,
    unreadable: &mut Vec<PathBuf>,
) {
    if paths.iter().any(|path| dir.starts_with(path)) {
        walk(dir, segs, out, unreadable);
        return;
    }
    if !paths.iter().any(|path| path.starts_with(dir)) {
        return;
    }
    let Some((seg, rest)) = segs.split_first() else {
        return;
    };
    if *seg == "**" {
        walk_paths(dir, rest, paths, out, unreadable);
        // Descend only along changed paths, without listing sibling directories.
        let children: std::collections::BTreeSet<_> = paths
            .iter()
            .filter_map(|path| {
                path.strip_prefix(dir)
                    .ok()?
                    .components()
                    .next()
                    .map(|part| dir.join(part))
            })
            .collect();
        for child in children {
            if child.is_dir() {
                walk_paths(&child, segs, paths, out, unreadable);
            }
        }
    } else if seg.contains('*') {
        let children: std::collections::BTreeSet<_> = paths
            .iter()
            .filter_map(|path| {
                path.strip_prefix(dir)
                    .ok()?
                    .components()
                    .next()
                    .map(|part| dir.join(part))
            })
            .collect();
        for child in children {
            if child
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|name| wildcard(seg, name))
            {
                if rest.is_empty() && child.is_file() {
                    out.push(child);
                } else if child.is_dir() {
                    walk_paths(&child, rest, paths, out, unreadable);
                }
            }
        }
    } else {
        let child = dir.join(seg);
        if rest.is_empty() {
            if paths.iter().any(|path| child.starts_with(path)) && child.is_file() {
                out.push(child);
            }
        } else {
            walk_paths(&child, rest, paths, out, unreadable);
        }
    }
}

/// A database sidecar wakes capture of its database, never the file pass.
pub(crate) fn database_path(config: &Config, path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    let name = name
        .strip_suffix("-wal")
        .or_else(|| name.strip_suffix("-shm"))
        .or_else(|| name.strip_suffix("-journal"))
        .unwrap_or(name);
    let db = path.with_file_name(name);
    collected()
        .filter(|o| o.capture == polylogue_sources::Capture::RowDump)
        .find_map(|origin| {
            let relative = db.strip_prefix(config.root(origin.root)).ok()?;
            origin
                .locations
                .iter()
                .any(|pattern| ends_in(relative, pattern))
                .then(|| db.clone())
        })
}

/// Every file in Polylogue's own spools (hook events, browser captures,
/// collectors' output): small directories, read again while a long pass
/// runs so what arrives there is not left behind it.
pub fn scan_spools(config: &Config) -> Vec<Found> {
    let mut found = Vec::new();
    let mut unreadable = Vec::new();
    for origin in collected().filter(|o| o.arrival == Arrival::Spool) {
        for pattern in origin.locations {
            let segs: Vec<_> = pattern.split('/').collect();
            let mut paths = Vec::new();
            walk(
                &config.root(origin.root),
                &segs,
                &mut paths,
                &mut unreadable,
            );
            found.extend(paths.into_iter().map(|path| Found { path, origin }));
        }
    }
    found
}

pub fn scan_with_coverage(config: &Config) -> Scan {
    use std::collections::{BTreeMap, BTreeSet};
    let mut areas = BTreeMap::new();
    for tool in polylogue_sources::locations::TOOLS {
        let root = config.root(tool.root);
        areas.insert(root.clone(), Some(tool));
        if tool.name == "hermes" {
            for profile in entries(&root.join("profiles"), &mut Vec::new()) {
                if fs::symlink_metadata(&profile).is_ok_and(|m| m.is_dir()) {
                    areas.insert(profile, Some(tool));
                }
            }
        }
    }
    for origin in collected() {
        for pattern in origin.locations {
            let base = config.root(origin.root);
            let root = if origin.root == polylogue_sources::Root::HOME
                || origin.root == polylogue_sources::Root::STATE
            {
                let first = pattern.split('/').next().unwrap_or("");
                if first.contains('*') {
                    continue;
                }
                base.join(first)
            } else {
                base
            };
            if !areas.keys().any(|area| root.starts_with(area)) {
                areas.entry(root).or_insert(None);
            }
        }
    }
    let roots: BTreeSet<_> = areas.keys().cloned().collect();
    let mut scan = Scan {
        found: Vec::new(),
        unreadable: Vec::new(),
        coverage: Vec::new(),
        roots: Vec::new(),
    };
    let mut rows = BTreeMap::new();
    for (root, tool) in areas {
        let mut files = Vec::new();
        let before = scan.unreadable.len();
        let mut status = Accessibility::Accessible;
        all_files(
            &root,
            &root,
            &roots,
            &mut files,
            &mut scan.unreadable,
            &mut status,
        );
        for path in files {
            let relative = path.strip_prefix(&root).unwrap_or(&path);
            let ignored = (tool.is_some_and(|tool| tool.name == "claude-code")
                && user_project(&root, &path))
            .then_some(("**", "user project under tool home"))
            .or_else(|| git_metadata(relative, tool).then_some((".git/**", "git metadata")))
            .or_else(|| {
                tool.and_then(|tool| {
                    tool.ignored
                        .iter()
                        .find(|(p, _)| matches_path(relative, p))
                        .map(|(pattern, reason)| (*pattern, *reason))
                })
            });
            let matched = if ignored.is_none() {
                collected()
                    .find_map(|origin| {
                        let relative = path.strip_prefix(config.root(origin.root)).ok()?;
                        origin
                            .locations
                            .iter()
                            .find(|p| matches_location(relative, p))
                            .map(|pattern| (origin, *pattern))
                    })
                    .or_else(|| {
                        tool.and_then(|tool| {
                            tool.locations
                                .iter()
                                .find(|p| matches_location(relative, p))
                                .map(|pattern| (context_origin(tool, &path), *pattern))
                        })
                    })
            } else {
                None
            };
            let (class, pattern, reason) = if let Some((pattern, reason)) = ignored {
                (Class::Ignored, pattern.to_owned(), Some(reason.to_owned()))
            } else if let Some((origin, pattern)) = matched {
                scan.found.push(Found {
                    path: path.clone(),
                    origin,
                });
                (Class::Collected, pattern.to_owned(), None)
            } else {
                (
                    Class::Unrecognized,
                    format!(
                        "{}/*.{}",
                        relative.parent().unwrap_or(Path::new("")).display(),
                        relative.extension().and_then(|e| e.to_str()).unwrap_or("")
                    ),
                    None,
                )
            };
            match fs::metadata(&path) {
                Ok(meta) => {
                    let row = rows
                        .entry((root.clone(), class.clone(), pattern.clone()))
                        .or_insert(Coverage {
                            root: root.clone(),
                            class,
                            pattern,
                            reason,
                            files: 0,
                            bytes: 0,
                            newest_mtime_ms: None,
                        });
                    row.files += 1;
                    row.bytes += meta.len();
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                        .and_then(|duration| duration.as_millis().try_into().ok());
                    row.newest_mtime_ms = row.newest_mtime_ms.max(mtime);
                }
                Err(_) => scan.unreadable.push(path),
            }
        }
        if status == Accessibility::Accessible && scan.unreadable.len() > before {
            status = Accessibility::Incomplete;
        }
        scan.roots.push(RootCoverage {
            path_bytes: root.as_os_str().as_encoded_bytes().to_vec(),
            status,
            entries: Vec::new(),
            unavailable_paths: scan.unreadable[before..]
                .iter()
                .map(|p| p.as_os_str().as_encoded_bytes().to_vec())
                .collect(),
        });
    }
    scan.found.sort_by(|a, b| a.path.cmp(&b.path));
    scan.found.dedup_by(|a, b| a.path == b.path);
    scan.unreadable.sort();
    scan.unreadable.dedup();
    scan.coverage = rows.into_values().collect();
    for row in &scan.coverage {
        if let Some(root) = scan
            .roots
            .iter_mut()
            .find(|root| root.path_bytes == row.root.as_os_str().as_encoded_bytes())
        {
            root.entries.push(Entry {
                class: row.class,
                pattern: row.pattern.clone(),
                reason: row.reason.clone(),
                files: row.files,
                bytes: row.bytes,
                newest_mtime_ms: row.newest_mtime_ms,
            });
        }
    }
    scan
}

fn user_project(root: &Path, path: &Path) -> bool {
    path.ancestors()
        .skip(1)
        .take_while(|ancestor| ancestor.starts_with(root) && *ancestor != root)
        .any(|ancestor| std::fs::symlink_metadata(ancestor.join(".git")).is_ok())
}

fn git_metadata(relative: &Path, tool: Option<&polylogue_sources::locations::Tool>) -> bool {
    relative.components().any(|p| p.as_os_str() == ".git")
        && !tool.is_some_and(|tool| {
            tool.name == "gemini-cli"
                && tool
                    .locations
                    .iter()
                    .any(|pattern| pattern.contains(".git/") && matches_path(relative, pattern))
        })
}

fn context_origin(tool: &polylogue_sources::locations::Tool, path: &Path) -> &'static OriginSpec {
    let database = path
        .extension()
        .is_some_and(|e| e == "db" || e == "sqlite" || e == "vscdb");
    let name = if tool.name == "hermes" && polylogue_sources::hermes::open(path).is_some() {
        "hermes"
    } else if tool.name == "codex" && path.extension().is_some_and(|e| e == "zst") {
        "codex-compressed"
    } else if database {
        "tool-database"
    } else {
        "tool-context"
    };
    polylogue_sources::origin::by_name(name).expect("collection origin")
}

fn all_files(
    root: &Path,
    path: &Path,
    roots: &std::collections::BTreeSet<PathBuf>,
    files: &mut Vec<PathBuf>,
    unreadable: &mut Vec<PathBuf>,
    status: &mut Accessibility,
) {
    if path != root && roots.contains(path) {
        return;
    }
    let metadata = if path == root {
        fs::metadata(path)
    } else {
        fs::symlink_metadata(path)
    };
    if path == root {
        *status = match &metadata {
            Ok(_) => Accessibility::Accessible,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Accessibility::Absent,
            Err(_) => Accessibility::Unavailable,
        };
    }
    match metadata {
        Ok(meta) if meta.is_dir() => {
            for child in entries(path, unreadable) {
                all_files(root, &child, roots, files, unreadable, status);
            }
        }
        Ok(meta) if meta.is_file() => files.push(path.to_path_buf()),
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => unreadable.push(path.to_path_buf()),
    }
}

/// A location matches the complete relative path, never a suffix guessed
/// from another tool's directory.
pub fn matches_path(path: &Path, pattern: &str) -> bool {
    match_path(path, pattern, false)
}
fn matches_location(path: &Path, pattern: &str) -> bool {
    match_path(path, pattern, true)
}
fn match_path(path: &Path, pattern: &str, explicit_hidden: bool) -> bool {
    fn matches(names: &[&str], segs: &[&str], explicit_hidden: bool) -> bool {
        match segs.split_first() {
            None => names.is_empty(),
            Some((&"**", rest)) => (0..=names.len())
                .take_while(|count| {
                    !explicit_hidden
                        || !names[..*count]
                            .iter()
                            .enumerate()
                            .any(|(index, name)| index + 1 < names.len() && name.starts_with('.'))
                })
                .any(|count| matches(&names[count..], rest, explicit_hidden)),
            Some((seg, rest)) => names.split_first().is_some_and(|(name, tail)| {
                (!explicit_hidden
                    || names.len() == 1
                    || !name.starts_with('.')
                    || seg.starts_with('.'))
                    && wildcard(seg, name)
                    && matches(tail, rest, explicit_hidden)
            }),
        }
    }
    let names: Vec<_> = path
        .components()
        .filter_map(|part| part.as_os_str().to_str())
        .collect();
    let segs: Vec<_> = pattern.split('/').filter(|part| !part.is_empty()).collect();
    matches(&names, &segs, explicit_hidden)
}

/// Whether `found.origin` claims the file, from its path and first bytes.
pub fn recognize(found: &Found) -> bool {
    let mut head = [0u8; 64];
    let n = fs::File::open(&found.path)
        .and_then(|mut f| f.read(&mut head))
        .unwrap_or(0);
    (found.origin.recognize)(&found.path, &head[..n])
}

/// An existing input directory is watched recursively. Until it exists,
/// watch its nearest existing ancestor for the next directory creation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchRoot {
    pub path: PathBuf,
    pub recursive: bool,
}

pub fn roots(config: &Config) -> Vec<WatchRoot> {
    let mut watches = std::collections::BTreeMap::new();
    for origin in collected() {
        for pattern in origin.locations {
            let fixed: Vec<&str> = pattern
                .split('/')
                .take_while(|s| !s.contains('*'))
                .collect();
            let target = config.root(origin.root).join(fixed.join("/"));
            let mut path = target.as_path();
            while !path.is_dir() {
                let Some(parent) = path.parent() else {
                    break;
                };
                path = parent;
            }
            if path.is_dir() {
                let recursive = path == target;
                let entry = watches.entry(path.to_path_buf()).or_insert(false);
                *entry |= recursive;
            }
        }
    }
    for tool in polylogue_sources::locations::TOOLS {
        let target = config.root(tool.root);
        let mut path = target.as_path();
        while !path.is_dir() {
            let Some(parent) = path.parent() else {
                break;
            };
            path = parent;
        }
        if path.is_dir() {
            *watches.entry(path.to_path_buf()).or_insert(false) |= path == target;
        }
    }
    watches
        .into_iter()
        .map(|(path, recursive)| WatchRoot { path, recursive })
        .collect()
}

fn walk(dir: &Path, segs: &[&str], out: &mut Vec<PathBuf>, unreadable: &mut Vec<PathBuf>) {
    let Some((seg, rest)) = segs.split_first() else {
        return;
    };
    if *seg == "**" {
        walk(dir, rest, out, unreadable);
        for sub in entries(dir, unreadable) {
            if fs::symlink_metadata(&sub).is_ok_and(|m| m.is_dir()) {
                walk(&sub, segs, out, unreadable);
            }
        }
        return;
    }
    let last = rest.is_empty();
    if !seg.contains('*') {
        let p = dir.join(seg);
        if last {
            if p.is_file() {
                out.push(p);
            }
        } else if fs::symlink_metadata(&p).is_ok_and(|m| m.is_dir()) {
            walk(&p, rest, out, unreadable);
        }
        return;
    }
    for p in entries(dir, unreadable) {
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !wildcard(seg, name) {
            continue;
        }
        if last {
            if p.is_file() {
                out.push(p);
            }
        } else if fs::symlink_metadata(&p).is_ok_and(|m| m.is_dir()) {
            walk(&p, rest, out, unreadable);
        }
    }
}

fn entries(dir: &Path, unreadable: &mut Vec<PathBuf>) -> Vec<PathBuf> {
    match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|entry| match entry {
                Ok(entry) => Some(entry.path()),
                Err(_) => {
                    unreadable.push(dir.to_path_buf());
                    None
                }
            })
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(_) => {
            unreadable.push(dir.to_path_buf());
            Vec::new()
        }
    }
}

/// `*` matches any run of characters within one path segment.
/// Whether `path` ends in the layout `pattern` describes: its last
/// components match the pattern's segments, a `**` matching any number.
pub fn ends_in(path: &Path, pattern: &str) -> bool {
    let segs: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
    let names: Vec<&str> = path
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    ends_with(&names, &segs)
}

fn ends_with(names: &[&str], segs: &[&str]) -> bool {
    let Some((last, rest)) = segs.split_last() else {
        return true;
    };
    if *last == "**" {
        return (0..=names.len()).any(|k| ends_with(&names[..names.len() - k], rest));
    }
    match names.split_last() {
        Some((name, before)) => {
            let matches = if last.contains('*') {
                wildcard(last, name)
            } else {
                last == name
            };
            matches && ends_with(before, rest)
        }
        None => false,
    }
}

fn wildcard(pattern: &str, name: &str) -> bool {
    if !pattern.contains('*') {
        return pattern == name;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !name.starts_with(first) || name.len() < first.len() + last.len() || !name.ends_with(last) {
        return false;
    }
    let mut rest = &name[first.len()..name.len() - last.len()];
    for mid in &parts[1..parts.len() - 1] {
        match rest.find(mid) {
            Some(i) => rest = &rest[i + mid.len()..],
            None => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests;

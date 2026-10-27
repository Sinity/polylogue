//! Finding inputs at the origins' fixed locations.
//! Exactly the layout each tool writes: each origin's location patterns are
//! walked under its root (the tool's own directory variable, or a default
//! under home), and a recognizer runs before a new
//! path's origin is decided. An unreadable root is a state, never an empty
//! backlog.

use crate::Config;
use polylogue_sources::{Arrival, ORIGINS, OriginSpec, Support};
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

/// Origins the loop collects from fixed locations.
fn collected() -> impl Iterator<Item = &'static OriginSpec> {
    ORIGINS
        .iter()
        .filter(|o| o.support == Support::Parsed && o.arrival != Arrival::Import)
}

/// Every file matching a live or spool location, and the directories that
/// exist but could not be read. Recognition is separate ([`recognize`]),
/// so known paths are not reopened every pass.
pub fn scan(config: &Config) -> (Vec<Found>, Vec<PathBuf>) {
    scan_origins(config, collected())
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
            found.extend(matched.into_iter().map(|path| Found { path, origin }));
        }
    }
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
    scan_origins(config, collected().filter(|o| o.arrival == Arrival::Spool)).0
}

fn scan_origins(
    config: &Config,
    origins: impl Iterator<Item = &'static OriginSpec>,
) -> (Vec<Found>, Vec<PathBuf>) {
    let mut found = Vec::new();
    let mut unreadable = Vec::new();
    for origin in origins {
        let root = config.root(origin.root);
        for pattern in origin.locations {
            let segs: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
            let mut paths = Vec::new();
            walk(&root, &segs, &mut paths, &mut unreadable);
            found.extend(paths.into_iter().map(|path| Found { path, origin }));
        }
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found.dedup_by(|a, b| a.path == b.path);
    (found, unreadable)
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
            if sub.is_dir() {
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
        } else if p.is_dir() {
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
        } else if p.is_dir() {
            walk(&p, rest, out, unreadable);
        }
    }
}

fn entries(dir: &Path, unreadable: &mut Vec<PathBuf>) -> Vec<PathBuf> {
    match fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
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

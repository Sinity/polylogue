//! Finding inputs at the origins' fixed locations (architecture 5.1).
//! Exactly the layout each tool writes: each origin's location patterns are
//! walked under the home directory, and a recognizer runs before a new
//! path's origin is decided. An unreadable root is a state, never an empty
//! backlog.

use polylogue_sources::{ORIGINS, OriginSpec, Route, Support};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub struct Found {
    pub path: PathBuf,
    pub origin: &'static OriginSpec,
}

/// Origins the loop collects from fixed locations.
fn collected() -> impl Iterator<Item = &'static OriginSpec> {
    ORIGINS
        .iter()
        .filter(|o| o.support == Support::Parsed && o.route != Route::Import)
}

/// Every file matching a live or spool location, and the directories that
/// exist but could not be read. Recognition is separate ([`recognize`]),
/// so known paths are not reopened every pass.
pub fn scan(home: &Path) -> (Vec<Found>, Vec<PathBuf>) {
    let mut found = Vec::new();
    let mut unreadable = Vec::new();
    for origin in collected() {
        for pattern in origin.locations {
            let segs: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
            let mut paths = Vec::new();
            walk(home, &segs, &mut paths, &mut unreadable);
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

/// The directories to watch: each pattern's longest prefix without a
/// wildcard, where it exists.
pub fn roots(home: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = collected()
        .flat_map(|o| o.locations.iter())
        .map(|pattern| {
            let fixed: Vec<&str> = pattern
                .split('/')
                .take_while(|s| !s.contains('*'))
                .collect();
            home.join(fixed.join("/"))
        })
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    out.dedup();
    out
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
mod tests {
    use super::*;

    #[test]
    fn wildcards_match_within_a_segment() {
        assert!(wildcard("*.jsonl", "a.jsonl"));
        assert!(wildcard("agent-*.jsonl", "agent-x1.jsonl"));
        assert!(!wildcard("agent-*.jsonl", "x.jsonl"));
        assert!(wildcard("*", "anything"));
        assert!(wildcard("a*b*c", "aXbYc"));
        assert!(!wildcard("a*b*c", "aXcYb"));
    }

    #[test]
    fn scans_exactly_the_claude_code_layout() {
        let home = tempfile::tempdir().unwrap();
        let p = home.path().join(".claude/projects/-proj");
        fs::create_dir_all(p.join("s1/subagents")).unwrap();
        fs::write(p.join("s1.jsonl"), "{}\n").unwrap();
        fs::write(p.join("s1/subagents/agent-a.jsonl"), "{}\n").unwrap();
        fs::write(p.join("notes.txt"), "x").unwrap();
        fs::write(p.join("s1/subagents/other.jsonl"), "{}\n").unwrap();
        let (found, unreadable) = scan(home.path());
        let names: Vec<_> = found
            .iter()
            .map(|f| f.path.strip_prefix(&p).unwrap().to_owned())
            .collect();
        assert_eq!(
            names,
            [
                PathBuf::from("s1/subagents/agent-a.jsonl"),
                PathBuf::from("s1.jsonl")
            ]
        );
        assert!(unreadable.is_empty());
        assert!(found.iter().all(recognize));
        assert_eq!(roots(home.path()), [home.path().join(".claude/projects")]);
    }
}

//! Writing repository lookups (spec sources/README.md, the repo-lookup
//! record). For each working directory the index holds that exists on this
//! host, the nearest directory with `.git` above it is read with `gix`, no
//! subprocess: its `origin` remote, credentials removed, and the root commit
//! reached by first parents from `HEAD`. The result is a small JSON input in
//! the lookup spool, written again only when the remote or root commit
//! changes, so the loop stores and parses it like any input and a rebuild
//! joins every session to the same repository.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// What a repository directory says about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lookup {
    pub remote: Option<String>,
    pub root_commit: Option<String>,
}

/// Lookups read in earlier passes, so a pass rereads a repository's
/// history only when its `HEAD` moved.
#[derive(Default)]
pub struct Cache {
    /// Working directory to the repository directory above it (`None`: no
    /// repository).
    repos: HashMap<PathBuf, Option<PathBuf>>,
    /// Repository directory to the `HEAD` id and root commit last read.
    roots: HashMap<PathBuf, (String, Option<String>)>,
}

/// The nearest directory at or above `dir` holding `.git` (a directory, or
/// the file a worktree has).
pub fn repository_of(dir: &Path) -> Option<PathBuf> {
    dir.ancestors()
        .find(|d| d.join(".git").exists())
        .map(Path::to_path_buf)
}

/// A remote URL without the user and password a `scheme://` URL may carry.
/// An scp-like `git@host:path` names no secret and is kept.
pub fn without_credentials(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_owned();
    };
    let host_end = rest.find('/').unwrap_or(rest.len());
    match rest[..host_end].rfind('@') {
        Some(at) => format!("{scheme}://{}", &rest[at + 1..]),
        None => url.to_owned(),
    }
}

/// Reads one repository directory.
pub fn read(repo_dir: &Path, cache: &mut Cache) -> Lookup {
    let Ok(repo) = gix::open(repo_dir) else {
        return Lookup {
            remote: None,
            root_commit: None,
        };
    };
    let remote = repo
        .find_remote("origin")
        .ok()
        .and_then(|r| {
            r.url(gix::remote::Direction::Fetch)
                .map(|u| u.to_bstring().to_string())
        })
        .map(|u| without_credentials(&u));
    let head = repo.head_id().ok().map(|id| id.detach());
    let root_commit = head.and_then(|head| {
        let key = head.to_string();
        if let Some((seen, root)) = cache.roots.get(repo_dir)
            && *seen == key
        {
            return root.clone();
        }
        let root = first_parent_root(&repo, head);
        cache
            .roots
            .insert(repo_dir.to_path_buf(), (key, root.clone()));
        root
    });
    Lookup {
        remote,
        root_commit,
    }
}

/// The commit with no parents reached from `head` by first parents. `None`
/// when history is missing (a shallow clone).
fn first_parent_root(repo: &gix::Repository, head: gix::ObjectId) -> Option<String> {
    let mut id = head;
    loop {
        let commit = repo.find_commit(id).ok()?;
        match commit.parent_ids().next() {
            Some(parent) => id = parent.detach(),
            None => return Some(id.to_string()),
        }
    }
}

/// Writes the lookups for `cwds` into `dir`, one file per repository named
/// by the hash of its path. Returns how many files changed.
pub fn refresh(
    cwds: impl IntoIterator<Item = PathBuf>,
    dir: &Path,
    cache: &mut Cache,
    read_at: &str,
) -> io::Result<usize> {
    let mut repos = BTreeSet::new();
    for cwd in cwds {
        let repo = cache
            .repos
            .entry(cwd.clone())
            .or_insert_with(|| repository_of(&cwd))
            .clone();
        repos.extend(repo);
    }
    let mut changed = 0;
    for repo in repos {
        let l = read(&repo, cache);
        let path = repo.display().to_string();
        let file = dir.join(format!(
            "{}.json",
            &blake3::hash(path.as_bytes()).to_hex()[..16]
        ));
        if let Some(old) = fs::read(&file)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            && old["remote"].as_str() == l.remote.as_deref()
            && old["root_commit"].as_str() == l.root_commit.as_deref()
        {
            continue;
        }
        let doc = serde_json::json!({
            "path": path,
            "remote": l.remote,
            "root_commit": l.root_commit,
            "read_at": read_at,
        });
        fs::create_dir_all(dir)?;
        let bytes = serde_json::to_vec(&doc)?;
        atomicwrites::AtomicFile::new(&file, atomicwrites::AllowOverwrite)
            .write(|output| output.write_all(&bytes))
            .map_err(io::Error::from)?;
        changed += 1;
    }
    Ok(changed)
}

/// Every working directory a session row names: its last and the earlier
/// ones `cwd_json` lists.
pub fn working_directories(rows: &[serde_json::Value]) -> BTreeSet<PathBuf> {
    let mut out = BTreeSet::new();
    for row in rows {
        if let Some(c) = row["cwd"].as_str() {
            out.insert(PathBuf::from(c));
        }
        if let Some(list) = row["cwd_json"]
            .as_str()
            .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
        {
            out.extend(list.into_iter().map(PathBuf::from));
        }
    }
    out.retain(|p| p.is_absolute() && p.is_dir());
    out
}

#[cfg(test)]
mod tests;

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
use std::io;
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
        let tmp = file.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_vec(&doc).expect("json"))?;
        fs::rename(&tmp, &file)?;
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
mod tests {
    use super::*;

    #[test]
    fn credentials_never_reach_a_lookup() {
        assert_eq!(
            without_credentials("https://user:tok@github.com/o/r.git"),
            "https://github.com/o/r.git"
        );
        assert_eq!(
            without_credentials("https://github.com/o/r.git"),
            "https://github.com/o/r.git"
        );
        assert_eq!(
            without_credentials("git@github.com:o/r.git"),
            "git@github.com:o/r.git"
        );
        // An @ in the path is not userinfo.
        assert_eq!(
            without_credentials("https://host/a@b/r.git"),
            "https://host/a@b/r.git"
        );
    }

    #[test]
    fn the_repository_is_the_nearest_git_directory_above() {
        let t = tempfile::tempdir().unwrap();
        let repo = t.path().join("repo");
        fs::create_dir_all(repo.join(".git")).unwrap();
        fs::create_dir_all(repo.join("src/deep")).unwrap();
        assert_eq!(repository_of(&repo.join("src/deep")), Some(repo.clone()));
        // Beside the repository, its directory is not the answer (the temp
        // directory may itself sit inside another repository).
        assert!(repository_of(t.path()).is_none_or(|r| !r.starts_with(t.path())));
    }

    #[test]
    fn a_lookup_is_written_once_and_again_only_when_it_changes() {
        let t = tempfile::tempdir().unwrap();
        let repo = t.path().join("repo");
        // Not a readable repository: remote and root stay unknown.
        fs::create_dir_all(repo.join(".git")).unwrap();
        fs::create_dir_all(repo.join("src")).unwrap();
        let out = t.path().join("lookups");
        let mut cache = Cache::default();
        let cwds = || vec![repo.join("src"), repo.clone()];
        assert_eq!(refresh(cwds(), &out, &mut cache, "t1").unwrap(), 1);
        assert_eq!(refresh(cwds(), &out, &mut cache, "t2").unwrap(), 0);
        let files: Vec<_> = fs::read_dir(&out).unwrap().collect();
        assert_eq!(files.len(), 1);
        let doc: serde_json::Value =
            serde_json::from_slice(&fs::read(files[0].as_ref().unwrap().path()).unwrap()).unwrap();
        assert_eq!(doc["path"], repo.display().to_string());
        assert!(doc["remote"].is_null() && doc["root_commit"].is_null());
        assert_eq!(doc["read_at"], "t1");
    }

    #[test]
    fn the_checkout_this_crate_is_built_from_reads_as_a_repository() {
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        let Some(repo) = repository_of(here) else {
            return; // a build outside a checkout (the Nix package)
        };
        let l = read(&repo, &mut Cache::default());
        let root = l.root_commit.expect("a full clone has a root commit");
        assert_eq!(root.len(), 40);
        if let Some(remote) = l.remote {
            assert!(!remote.contains('@') || !remote.contains("://"), "{remote}");
        }
    }
}

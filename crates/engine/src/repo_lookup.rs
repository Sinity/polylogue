//! Writing repository lookups. For each working directory the index holds that exists on this
//! host, the nearest directory with `.git` above it supplies its `origin`
//! remote and its root commits, read by common's `git-identity` so Sinex and
//! the session hook name the same repository: the remote without credentials,
//! the roots as every root reachable from `HEAD`, sorted, shallow-clone
//! boundaries left out. The result is a small JSON input in the lookup spool,
//! written again only when the remote or the roots change, so the loop stores
//! and parses it like any input and a rebuild joins every session to the same
//! repository.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime};

/// How long one repository may take to answer. A lookup is background
/// enrichment, so the bound is generous; a repository that takes longer reads
/// as unknown and is asked again on the next pass.
const ANSWER_WITHIN: Duration = Duration::from_secs(60);

/// What a repository directory says about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lookup {
    pub remote: Option<String>,
    /// Sorted; empty when the history reaches no root this checkout holds.
    pub root_commits: Vec<String>,
}

/// Repository results are reused while HEAD and the configuration files are
/// unchanged. Missing history is retried because a fetch can fill it.
#[derive(Default)]
pub struct Cache {
    lookups: HashMap<PathBuf, Cached>,
}

struct Cached {
    head: Option<String>,
    files: Vec<(PathBuf, Option<(SystemTime, u64)>)>,
    lookup: Lookup,
}

pub use git_identity::{repository_of, without_credentials};

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let mut command = Command::new("git");
    command.arg("-C").arg(repo).args(args);
    for name in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ] {
        command.env_remove(name);
    }
    command
        .env("GIT_NO_REPLACE_OBJECTS", "1")
        .env("GIT_NO_LAZY_FETCH", "1");
    git_output(&mut command)
}

fn git_output(command: &mut Command) -> Option<String> {
    let output = command.output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok())
        .flatten()
}

/// Reads one repository directory. Git's ownership refusal makes both fields
/// unavailable, just like an unreadable repository.
pub fn read(repo: &Path, cache: &mut Cache) -> Lookup {
    read_with(repo, cache, |args| git(repo, args))
}

fn read_with(
    repo: &Path,
    cache: &mut Cache,
    mut git: impl FnMut(&[&str]) -> Option<String>,
) -> Lookup {
    let unavailable = Lookup {
        remote: None,
        root_commits: Vec::new(),
    };
    let Some(config) = git(&["config", "--null", "--show-origin", "--name-only", "--list"]) else {
        cache.lookups.remove(repo);
        return unavailable;
    };
    let head = git(&["rev-parse", "--verify", "HEAD"]).map(|text| text.trim().to_owned());
    // Git supplies each included configuration file, including conditional
    // includes, so changes outside the checkout invalidate the result too.
    let mut paths: BTreeSet<PathBuf> = config
        .split('\0')
        .step_by(2)
        .filter_map(|origin| origin.strip_prefix("file:").map(PathBuf::from))
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                repo.join(path)
            }
        })
        .collect();
    // Git names the shallow file's path in every repository it will read,
    // whether or not the file exists, so no answer means Git refused the
    // repository (unsafe ownership, unreadable `.git`): both fields stay
    // unknown, as they do when the configuration cannot be listed.
    let Some(shallow) = git(&[
        "rev-parse",
        "--path-format=absolute",
        "--git-path",
        "shallow",
    ])
    .map(|text| PathBuf::from(text.trim())) else {
        cache.lookups.remove(repo);
        return unavailable;
    };
    paths.insert(shallow);
    let files = paths
        .into_iter()
        .map(|path| {
            let facts = fs::metadata(&path)
                .ok()
                .and_then(|meta| Some((meta.modified().ok()?, meta.len())));
            (path, facts)
        })
        .collect::<Vec<_>>();
    if let Some(old) = cache.lookups.get(repo)
        && old.head == head
        && old.files == files
        && !old.lookup.root_commits.is_empty()
    {
        return old.lookup.clone();
    }
    let deadline = Instant::now() + ANSWER_WITHIN;
    let remote = git_identity::remote(repo, deadline).ok().flatten();
    let root_commits = git_identity::root_commits(repo, deadline).unwrap_or_default();
    let lookup = Lookup {
        remote,
        root_commits,
    };
    cache.lookups.insert(
        repo.to_path_buf(),
        Cached {
            head,
            files,
            lookup: lookup.clone(),
        },
    );
    lookup
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
        let repo = repository_of(&cwd);
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
            && old["root_commits"] == serde_json::json!(l.root_commits)
        {
            continue;
        }
        let doc = serde_json::json!({
            "path": path,
            "remote": l.remote,
            "root_commits": l.root_commits,
            "read_at": read_at,
        });
        fs::create_dir_all(dir)?;
        let bytes = serde_json::to_vec(&doc)?;
        polylogue_model::atomic_write::write_atomic(&file, &bytes, None)?;
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

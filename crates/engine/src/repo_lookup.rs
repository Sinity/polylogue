//! Writing repository lookups. For each working directory the index holds that exists on this
//! host, the nearest directory with `.git` above it supplies its `origin`
//! remote and its root commits, read by common's `git-identity` so Sinex and
//! the session hook name the same repository: the remote without credentials,
//! the roots as every root reachable from `HEAD`, sorted, shallow-clone
//! boundaries left out. The result is a small JSON input in the lookup spool,
//! written again only when the remote or the roots change, so the loop stores
//! and parses it like any input and a rebuild joins every session to the same
//! repository. A read that fails, runs past its deadline or names no
//! remote and no root writes nothing, so the last good lookup stays.

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

/// Reads one repository directory: `None` when Git cannot answer (an
/// unreadable repository, its ownership refusal, a read past the deadline)
/// or the repository names neither a remote nor a root.
pub fn read(repo: &Path, cache: &mut Cache) -> Option<Lookup> {
    read_with(repo, cache, |args| git(repo, args), ANSWER_WITHIN)
}

fn read_with(
    repo: &Path,
    cache: &mut Cache,
    mut git: impl FnMut(&[&str]) -> Option<String>,
    answer_within: Duration,
) -> Option<Lookup> {
    let Some(config) = git(&["config", "--null", "--show-origin", "--name-only", "--list"]) else {
        cache.lookups.remove(repo);
        return None;
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
        return None;
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
        return Some(old.lookup.clone());
    }
    let deadline = Instant::now() + answer_within;
    let (Ok(remote), Ok(root_commits)) = (
        git_identity::remote(repo, deadline),
        git_identity::root_commits(repo, deadline),
    ) else {
        cache.lookups.remove(repo);
        return None;
    };
    if remote.is_none() && root_commits.is_empty() {
        cache.lookups.remove(repo);
        return None;
    }
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
    Some(lookup)
}

/// Writes the lookups for `cwds` into `dir`, one file per repository named
/// by the hash of its path. Returns how many files changed. A repository
/// whose file cannot be written does not stop the others; the first such
/// failure is returned, naming its repository, after all were tried.
pub fn refresh(
    cwds: impl IntoIterator<Item = PathBuf>,
    dir: &Path,
    cache: &mut Cache,
    read_at: &str,
) -> io::Result<usize> {
    let repos: BTreeSet<PathBuf> = cwds
        .into_iter()
        .filter_map(|cwd| repository_of(&cwd))
        .collect();
    let mut changed = 0;
    let mut failure = None;
    for repo in repos {
        let lookup = read(&repo, cache);
        match write_lookup(&repo, dir, lookup, read_at) {
            Ok(true) => changed += 1,
            Ok(false) => {}
            Err(error) => {
                failure.get_or_insert_with(|| {
                    io::Error::new(error.kind(), format!("{}: {error}", repo.display()))
                });
            }
        }
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(changed),
    }
}

/// Writes one repository's lookup file unless it already holds the same
/// remote and roots, or the read gave no lookup; whether it wrote.
fn write_lookup(
    repo: &Path,
    dir: &Path,
    lookup: Option<Lookup>,
    read_at: &str,
) -> io::Result<bool> {
    let Some(lookup) = lookup else {
        return Ok(false);
    };
    let path = repo.display().to_string();
    let file = dir.join(format!(
        "{}.json",
        &blake3::hash(path.as_bytes()).to_hex()[..16]
    ));
    if let Some(old) = fs::read(&file)
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        && old["remote"].as_str() == lookup.remote.as_deref()
        && old["root_commits"] == serde_json::json!(lookup.root_commits)
    {
        return Ok(false);
    }
    let doc = serde_json::json!({
        "path": path,
        "remote": lookup.remote,
        "root_commits": lookup.root_commits,
        "read_at": read_at,
    });
    fs::create_dir_all(dir)?;
    let bytes = serde_json::to_vec(&doc)?;
    polylogue_model::atomic_write::write_atomic(&file, &bytes, None)?;
    Ok(true)
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

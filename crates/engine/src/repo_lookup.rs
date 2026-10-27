//! Writing repository lookups. For each working directory the index holds that exists on this
//! host, the nearest directory with `.git` above it supplies its `origin`
//! remote, credentials removed, and Git supplies the root commit
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

/// Known roots are read again only when `HEAD` moves. Missing history is
/// tried again because a fetch can fill it without moving `HEAD`.
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

/// The Git directory and the shared directory (different in linked worktrees).
fn git_directories(repo_dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let dot_git = repo_dir.join(".git");
    let git_dir = if dot_git.is_dir() {
        dot_git
    } else {
        let text = fs::read_to_string(dot_git).ok()?;
        repo_dir.join(text.trim().strip_prefix("gitdir: ")?)
    }
    .canonicalize()
    .ok()?;
    let common = match fs::read_to_string(git_dir.join("commondir")) {
        Ok(path) => git_dir.join(path.trim()).canonicalize().ok()?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => git_dir.clone(),
        Err(_) => return None,
    };
    Some((git_dir, common))
}

/// Resolves loose and packed references without starting a process each pass.
fn head_id(git_dir: &Path, common: &Path) -> Option<String> {
    let mut text = fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let mut seen = BTreeSet::new();
    loop {
        let value = text.trim();
        let Some(name) = value.strip_prefix("ref: ") else {
            return (matches!(value.len(), 40 | 64)
                && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
            .then(|| value.to_owned());
        };
        if !name.starts_with("refs/")
            || !Path::new(name)
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_)))
        {
            return None;
        }
        if !seen.insert(name.to_owned()) {
            return None;
        }
        text = fs::read_to_string(git_dir.join(name))
            .or_else(|_| fs::read_to_string(common.join(name)))
            .ok()
            .or_else(|| {
                fs::read_to_string(common.join("packed-refs"))
                    .ok()?
                    .lines()
                    .filter_map(|line| line.split_once(' '))
                    .find(|(_, reference)| *reference == name)
                    .map(|(id, _)| id.to_owned())
            })?;
    }
}

/// Read configuration files directly, keeping Git's quoting and include rules.
fn remote_url(git_dir: &Path, common: &Path) -> Option<String> {
    use gix_config::{
        File, Source,
        file::{Metadata, includes, init},
        source::Kind,
    };
    let head = fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let branch = head.trim().strip_prefix("ref: ");
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut options = includes::Options::follow_without_conditional(home.as_deref());
    options.conditional = includes::conditional::Context {
        git_dir: Some(git_dir),
        branch_name: branch.and_then(|name| name.try_into().ok()),
    };
    let mut paths: Vec<Metadata> = [Kind::GitInstallation, Kind::System, Kind::Global]
        .iter()
        .flat_map(|kind| kind.sources())
        .filter_map(|source| {
            source
                .storage_location(&mut |name| std::env::var_os(name))
                .filter(|path| path.is_file())
                .map(|path| Metadata::from(*source).at(path))
        })
        .collect();
    let local = File::from_path_no_includes(common.join("config"), Source::Local).ok()?;
    let worktree_config = local.boolean("extensions.worktreeConfig").ok().flatten() == Some(true);
    paths.push(Metadata::from(Source::Local).at(common.join("config")));
    if worktree_config && git_dir.join("config.worktree").is_file() {
        paths.push(Metadata::from(Source::Worktree).at(git_dir.join("config.worktree")));
    }
    let mut config = File::from_paths_metadata(
        paths,
        init::Options {
            includes: options,
            ..Default::default()
        },
    )
    .ok()??;
    if let Ok(environment) = File::from_environment_overrides() {
        config.append(environment).ok()?;
    }
    let url = config.strings("remote.origin.url")?.into_iter().next()?;
    let url = String::from_utf8_lossy(&url);
    let mut rewrite: Option<(usize, String)> = None;
    for section in config.sections_by_name("url").into_iter().flatten() {
        let header = section.header();
        let Some(base) = header.subsection_name() else {
            continue;
        };
        for prefix in section.values("insteadOf") {
            let prefix = String::from_utf8_lossy(&prefix);
            if let Some(rest) = url.strip_prefix(prefix.as_ref())
                && rewrite
                    .as_ref()
                    .is_none_or(|(length, _)| prefix.len() > *length)
            {
                rewrite = Some((
                    prefix.len(),
                    format!("{}{rest}", String::from_utf8_lossy(base.as_ref())),
                ));
            }
        }
    }
    Some(without_credentials(
        rewrite.as_ref().map_or(url.as_ref(), |(_, url)| url),
    ))
}

/// Reads one repository directory.
pub fn read(repo_dir: &Path, cache: &mut Cache) -> Lookup {
    let Some((git_dir, common)) = git_directories(repo_dir) else {
        return Lookup {
            remote: None,
            root_commit: None,
        };
    };
    let remote = remote_url(&git_dir, &common);
    let root_commit = head_id(&git_dir, &common).and_then(|head| {
        if let Some((seen, root)) = cache.roots.get(repo_dir)
            && *seen == head
            && root.is_some()
        {
            return root.clone();
        }
        let output = std::process::Command::new("git")
            .arg("--git-dir")
            .arg(&git_dir)
            .args(["rev-list", "--first-parent", "--max-parents=0", &head])
            .env_remove("GIT_COMMON_DIR")
            .env_remove("GIT_OBJECT_DIRECTORY")
            .env("GIT_NO_REPLACE_OBJECTS", "1")
            .env("GIT_NO_LAZY_FETCH", "1")
            .output()
            .ok();
        let root = output
            .filter(|out| out.status.success())
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .and_then(|text| text.lines().next().map(str::to_owned))
            .filter(|root| {
                !fs::read_to_string(common.join("shallow"))
                    .is_ok_and(|text| text.lines().any(|id| id == root))
            });
        cache
            .roots
            .insert(repo_dir.to_path_buf(), (head, root.clone()));
        root
    });
    Lookup {
        remote,
        root_commit,
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
        let mut output = tempfile::NamedTempFile::new_in(dir)?;
        output.write_all(&bytes)?;
        output.as_file().sync_all()?;
        output.persist(&file).map_err(|e| e.error)?;
        fs::File::open(dir)?.sync_all()?;
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

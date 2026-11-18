//! The caller's current Git worktrees and their recorded repository identity.
use super::Failure;
use polylogue_model::query_objects::HereScope;
use polylogue_store::Reader;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn text(path: &Path) -> Result<String, Failure> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| Failure::bad_value("here needs UTF-8 repository paths"))
}

fn read(path: &Path) -> Result<String, Failure> {
    std::fs::read_to_string(path)
        .map_err(|error| Failure::io(format!("{}: {error}", path.display())))
}

fn absolute(base: &Path, value: &str) -> Result<PathBuf, Failure> {
    let path = base.join(value.trim());
    std::fs::canonicalize(&path)
        .map_err(|error| Failure::io(format!("{}: {error}", path.display())))
}

fn paths(cwd: &Path) -> Result<Vec<String>, Failure> {
    if !cwd.is_absolute() {
        return Err(Failure::bad_request(
            "here needs an absolute working directory",
        ));
    }
    let mut found = None;
    for directory in cwd.ancestors() {
        let marker = directory.join(".git");
        match std::fs::symlink_metadata(&marker) {
            Ok(metadata) if metadata.is_dir() => {
                found = Some((directory, marker));
                break;
            }
            Ok(_) => {
                let value = read(&marker)?;
                let location = value.trim().strip_prefix("gitdir:").ok_or_else(|| {
                    Failure::bad_value(format!(
                        "{} is not a Git directory pointer",
                        marker.display()
                    ))
                })?;
                found = Some((directory, absolute(directory, location)?));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(Failure::io(format!("{}: {error}", marker.display()))),
        }
    }
    let Some((directory, gitdir)) = found else {
        return Ok(vec![text(cwd)?]);
    };
    let common_file = gitdir.join("commondir");
    let common = match std::fs::read_to_string(&common_file) {
        Ok(value) => absolute(&gitdir, &value)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => gitdir,
        Err(error) => return Err(Failure::io(format!("{}: {error}", common_file.display()))),
    };
    let mut paths = BTreeSet::from([text(directory)?]);
    if common.file_name().is_some_and(|name| name == ".git")
        && let Some(main) = common.parent()
    {
        paths.insert(text(main)?);
    }
    match std::fs::read_dir(common.join("worktrees")) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry.map_err(Failure::io)?;
                if !entry.file_type().map_err(Failure::io)?.is_dir() {
                    continue;
                }
                let pointer = entry.path().join("gitdir");
                let value = match std::fs::read_to_string(&pointer) {
                    Ok(value) => value,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(error) => {
                        return Err(Failure::io(format!("{}: {error}", pointer.display())));
                    }
                };
                let marker = PathBuf::from(value.trim());
                let marker = if marker.is_absolute() {
                    marker
                } else {
                    entry.path().join(marker)
                };
                if let Some(worktree) = marker.parent() {
                    paths.insert(text(worktree)?);
                }
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(Failure::io(error)),
    }
    Ok(paths.into_iter().collect())
}

pub(crate) fn scope(cwd: &Path, reader: &Reader<'_>) -> Result<HereScope, Failure> {
    let paths = paths(cwd)?;
    let repo_roots = reader.repository_roots_at_paths(&paths)?;
    Ok(HereScope { paths, repo_roots })
}

#[cfg(test)]
mod tests;

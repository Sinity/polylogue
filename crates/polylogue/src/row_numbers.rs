//! `@N`: row N of the last list printed in this terminal. Lists printed
//! for a person number their rows; the row IDs are kept per terminal in the
//! archive's runtime directory, and `@N` in a later command is replaced by
//! the ID before the command runs. Output for programs (`--json`) is never remembered.

use std::fs;
use std::path::{Path, PathBuf};

/// The terminal this process talks to (`pts-3`), from the first standard
/// stream that is one.
fn terminal() -> Option<String> {
    for fd in [2, 0, 1] {
        let Ok(target) = fs::read_link(format!("/proc/self/fd/{fd}")) else {
            continue;
        };
        let t = target.to_string_lossy();
        if let Some(rest) = t.strip_prefix("/dev/")
            && (rest.starts_with("pts/") || rest.starts_with("tty"))
        {
            return Some(rest.replace('/', "-"));
        }
    }
    None
}

fn rows_file(paths: &crate::config::Paths, root: &Path) -> Option<PathBuf> {
    let tty = terminal()?;
    Some(paths.runtime_dir(root).join(format!("rows-{tty}")))
}

/// The references and search hits shown in one terminal row.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Row {
    pub reference: String,
    #[serde(default)]
    pub hits: Vec<String>,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ShownRows {
    pub question: String,
    pub shown_at: i64,
    pub rows: Vec<Row>,
}
/// Row selections are rewritten only in the selection position.
pub(crate) fn is_selection(word: &str) -> bool {
    word == "@all"
        || word.strip_prefix('@').is_some_and(|tail| {
            !tail.is_empty()
                && tail.starts_with(|c: char| c.is_ascii_digit())
                && tail
                    .chars()
                    .all(|c| c.is_ascii_digit() || ",.-".contains(c))
        })
}
pub(crate) fn select(word: &str, shown: &ShownRows) -> Result<Vec<String>, String> {
    if word == "@all" {
        // An empty selection would reach the daemon as no selection at all.
        if shown.rows.is_empty() {
            return Err("@all: the last list has 0 rows".into());
        }
        return Ok(shown.rows.iter().map(|row| row.reference.clone()).collect());
    }
    let row = |number: &str| -> Result<&Row, String> {
        if number == "0" {
            return Err("@0: rows start at 1".into());
        }
        number
            .parse::<usize>()
            .ok()
            .and_then(|n| n.checked_sub(1))
            .and_then(|n| shown.rows.get(n))
            .ok_or_else(|| format!("@{number}: the last list has {} rows", shown.rows.len()))
    };
    let mut references = Vec::new();
    for part in word
        .strip_prefix('@')
        .ok_or("row number starts with @")?
        .split(',')
    {
        if part.is_empty() {
            return Err(format!("{word}: a row number is missing between commas"));
        }
        if part.contains('.') && part.contains('-') {
            return Err(format!("@{part}: a row range has no hit; use @1-2 or @2.1"));
        }
        if let Some((number, hit)) = part.split_once('.') {
            let selected = row(number)?;
            let reference = hit
                .parse::<usize>()
                .ok()
                .and_then(|n| n.checked_sub(1))
                .and_then(|n| selected.hits.get(n))
                .ok_or_else(|| format!("row @{number} has no hit {hit}; use @{number}"))?;
            references.push(reference.clone());
        } else if let Some((first, last)) = part.split_once('-') {
            let first: usize = first.parse().map_err(|_| "invalid row range")?;
            let last: usize = last.parse().map_err(|_| "invalid row range")?;
            if first == 0 || last < first {
                return Err(format!("@{first}-{last} is not a row range"));
            }
            if last > shown.rows.len() {
                return Err(format!(
                    "@{first}-{last}: the last list has {} rows",
                    shown.rows.len()
                ));
            }
            references.extend(
                shown.rows[first - 1..last]
                    .iter()
                    .map(|row| row.reference.clone()),
            );
        } else {
            references.push(row(part)?.reference.clone());
        }
    }
    Ok(references)
}
pub(crate) fn remember_listing(paths: &crate::config::Paths, root: &Path, shown: &ShownRows) {
    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return;
    }
    let Some(path) = rows_file(paths, root) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    replace_listing(&path, shown, |path, bytes| {
        polylogue_model::atomic_write::write_atomic(path, bytes, None)
    });
}
/// Remembering rows is a convenience, so a failure is not reported; but a
/// list that cannot be written removes the previous one, so a row number
/// never names a row of an older list.
fn replace_listing(
    path: &Path,
    shown: &ShownRows,
    write: impl FnOnce(&Path, &[u8]) -> std::io::Result<()>,
) {
    let written = serde_json::to_vec(shown)
        .map_err(std::io::Error::other)
        .and_then(|bytes| write(path, &bytes));
    if written.is_err() {
        let _ = fs::remove_file(path);
    }
}
pub(crate) fn last_listing(paths: &crate::config::Paths, root: &Path) -> Result<ShownRows, String> {
    let path = rows_file(paths, root)
        .ok_or("row numbers work on a terminal; use the references printed above")?;
    let file = fs::File::open(path).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => "no list has been printed in this terminal yet".to_owned(),
        _ => format!("the last list cannot be read: {error}"),
    })?;
    serde_json::from_reader(file).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests;
